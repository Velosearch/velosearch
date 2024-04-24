// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use datafusion::{
    optimizer::{OptimizerRule, OptimizerConfig, optimizer::ApplyOrder},
    logical_expr::{LogicalPlan, logical_plan::{Boolean, Predicate}, BooleanQuery, Operator},
    prelude::Expr, error::DataFusionError, common::Result,
};


/// Optimizer pass that rewrites predicates of the form
///
/// ```text
/// (A = B AND <expr1>) OR (A = B AND <expr2>) OR ... (A = B AND <exprN>)
/// ```
///
/// Into
/// ```text
/// (A = B) AND (<expr1> OR <expr2> OR ... <exprN> )
/// ```
///
/// Predicates connected by `OR` typically not able to be broken down
/// and distributed as well as those connected by `AND`.
///
/// The idea is to rewrite predicates into `good_predicate1 AND
/// good_predicate2 AND ...` where `good_predicate` means the
/// predicate has special support in the execution engine.
///
/// Equality join predicates (e.g. `col1 = col2`), or single column
/// expressions (e.g. `col = 5`) are examples of predicates with
/// special support.
///
/// # TPCH Q19
///
/// This optimization is admittedly somewhat of a niche usecase. It's
/// main use is that it appears in TPCH Q19 and is required to avoid a
/// CROSS JOIN.
///
/// Specifically, Q19 has a WHERE clause that looks like
///
/// ```sql
/// where
///   p_partkey = l_partkey
///   and l_shipmode in (‘AIR’, ‘AIR REG’)
///   and l_shipinstruct = ‘DELIVER IN PERSON’
///   and (
///     (
///       and p_brand = ‘[BRAND1]’
///       and p_container in ( ‘SM CASE’, ‘SM BOX’, ‘SM PACK’, ‘SM PKG’)
///       and l_quantity >= [QUANTITY1] and l_quantity <= [QUANTITY1] + 10
///       and p_size between 1 and 5
///     )
///     or
///     (
///       and p_brand = ‘[BRAND2]’
///       and p_container in (‘MED BAG’, ‘MED BOX’, ‘MED PKG’, ‘MED PACK’)
///       and l_quantity >= [QUANTITY2] and l_quantity <= [QUANTITY2] + 10
///       and p_size between 1 and 10
///     )
///     or
///     (
///       and p_brand = ‘[BRAND3]’
///       and p_container in ( ‘LG CASE’, ‘LG BOX’, ‘LG PACK’, ‘LG PKG’)
///       and l_quantity >= [QUANTITY3] and l_quantity <= [QUANTITY3] + 10
///       and p_size between 1 and 15
///     )
/// )
/// ```
///
/// Naively planning this query will result in a CROSS join with that
/// single large OR filter. However, rewriting it using the rewrite in
/// this pass results in a proper join predicate, `p_partkey = l_partkey`:
///
/// ```sql
/// where
///   p_partkey = l_partkey
///   and l_shipmode in (‘AIR’, ‘AIR REG’)
///   and l_shipinstruct = ‘DELIVER IN PERSON’
///   and (
///     (
///       and p_brand = ‘[BRAND1]’
///       and p_container in ( ‘SM CASE’, ‘SM BOX’, ‘SM PACK’, ‘SM PKG’)
///       and l_quantity >= [QUANTITY1] and l_quantity <= [QUANTITY1] + 10
///       and p_size between 1 and 5
///     )
///     or
///     (
///       and p_brand = ‘[BRAND2]’
///       and p_container in (‘MED BAG’, ‘MED BOX’, ‘MED PKG’, ‘MED PACK’)
///       and l_quantity >= [QUANTITY2] and l_quantity <= [QUANTITY2] + 10
///       and p_size between 1 and 10
///     )
///     or
///     (
///       and p_brand = ‘[BRAND3]’
///       and p_container in ( ‘LG CASE’, ‘LG BOX’, ‘LG PACK’, ‘LG PKG’)
///       and l_quantity >= [QUANTITY3] and l_quantity <= [QUANTITY3] + 10
///       and p_size between 1 and 15
///     )
/// )
/// ```
///
#[derive(Default)]
pub struct RewriteBooleanPredicate;

impl RewriteBooleanPredicate {
    pub fn new() -> Self {
        Self::default()
    }
}

impl OptimizerRule for RewriteBooleanPredicate {
    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Option<LogicalPlan>> {
        match plan {
            LogicalPlan::Boolean(boolean) => {
                let predicate = predicate(&boolean.binary_expr)?;
                let rewritten_predicate = rewrite_predicate(predicate.0);
                let rewritten_expr = normalize_predicate(rewritten_predicate.clone().0);
                Ok(Some(LogicalPlan::Boolean(Boolean::try_new_with_predicate(
                    rewritten_expr,
                    Some(rewritten_predicate.0),
                    rewritten_predicate.3,
                    rewritten_predicate.2,
                    boolean.is_score,
                    boolean.input.clone(),
                )?)))
            },
            _ => Ok(None),
        }
    }

    fn name(&self) -> &str {
        "rewrite_boolean_predicate"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }
}



pub fn predicate(expr: &Expr) -> Result<(Predicate, f64, usize, usize)> {
    match expr {
    Expr::BooleanQuery(BooleanQuery {left, op, right }) => match op {
            Operator::BitwiseAnd => {
                let left = predicate(left)?;
                let right = predicate(right)?;
                let args = vec![left, right];
                Ok((Predicate::And { args }, 0., 0, 0))
            }
            Operator::BitwiseOr => {
                let left = predicate(left)?;
                let right = predicate(right)?;
                let args = vec![left, right];
                Ok((Predicate::Or { args }, 0., 0, 0))
            }
            _ => Err(DataFusionError::Internal(format!("Don't support op: {:}", op))),
        },
        _ => {
            Ok((Predicate::Other {
                expr: Box::new(expr.clone()),
            }, 0., 0, 0))
        }
    }
}

fn normalize_predicate(predicate: Predicate) -> Expr {
    match predicate {
        Predicate::And { args } => {
            assert!(args.len() >= 2);
            args.into_iter()
                .map(|v| normalize_predicate(v.0))
                .reduce(Expr::boolean_and)
                .expect("had more than one arg")
        }
        Predicate::Or { args } => {
            assert!(args.len() >= 2);
            args.into_iter()
                .map(|v| normalize_predicate(v.0))
                .reduce(Expr::boolean_or)
                .expect("had more than one arg")
        }
        Predicate::Other { expr } => *expr,
    }
}

pub fn rewrite_predicate(predicate: Predicate) -> (Predicate, f64, usize, usize) {
    let mut node_num = 0;
    let mut leaf_num = 0;
    match predicate {
        Predicate::And { args } => {
            let flatten_args = flatten_and_predicates(args.clone());
            let mut rewritten_args = vec![];
            for arg in flatten_args {
                let rewritten = rewrite_predicate(arg.0);
                node_num += rewritten.2;
                leaf_num += rewritten.3;
                rewritten_args.push(rewritten);
            }
            (Predicate::And {
                args: rewritten_args,
            }, 0., node_num + 1, leaf_num)
        }
        Predicate::Or { args } => {
            let flatten_args = flatten_or_predicates(args);
            let mut rewritten_args = vec![];
            for arg in flatten_args {
                let rewriten = rewrite_predicate(arg.0.clone());
                node_num += rewriten.2;
                leaf_num += rewriten.3;
                rewritten_args.push(rewriten);
            }
            // delete_duplicate_predicates(&rewritten_args)
            (Predicate::Or {
                args: rewritten_args,
            }, 0., node_num + 1, leaf_num)
        }
        Predicate::Other { expr } => (Predicate::Other {
            expr: Box::new(*expr),
        }, 0., 1, 1),
    }
}

fn flatten_and_predicates(
    and_predicates: impl IntoIterator<Item = (Predicate, f64, usize, usize)>,
) -> Vec<(Predicate, f64, usize, usize)> {
    let mut flattened_predicates = vec![];
    for predicate in and_predicates {
        match predicate.0 {
            Predicate::And { args } => {
                flattened_predicates
                    .extend_from_slice(flatten_and_predicates(args).as_slice());
            }
            _ => {
                flattened_predicates.push(predicate);
            }
        }
    }
    flattened_predicates
}

fn flatten_or_predicates(
    or_predicates: impl IntoIterator<Item = (Predicate, f64, usize, usize)>,
) -> Vec<(Predicate, f64, usize, usize)> {
    let mut flattened_predicates = vec![];
    for predicate in or_predicates {
        match predicate.0 {
            Predicate::Or { args } => {
                flattened_predicates
                    .extend_from_slice(flatten_or_predicates(args).as_slice());
            }
            _ => {
                flattened_predicates.push(predicate);
            }
        }
    }
    flattened_predicates
}

// fn delete_duplicate_predicates(or_predicates: &[(Predicate, f64, usize)]) -> Predicate {
//     let mut shortest_exprs: Vec<(Predicate, f64, usize)> = vec![];
//     let mut shortest_exprs_len = 0;
//     // choose the shortest AND predicate
//     for or_predicate in or_predicates.iter() {
//         match or_predicate.0 {
//             Predicate::And { args } => {
//                 let args_num = args.len();
//                 if shortest_exprs.is_empty() || args_num < shortest_exprs_len {
//                     shortest_exprs = args.clone();
//                     shortest_exprs_len = args_num;
//                 }
//             }
//             _ => {
//                 // if there is no AND predicate, it must be the shortest expression.
//                 shortest_exprs = vec![or_predicate.clone()];
//                 break;
//             }
//         }
//     }

//     // dedup shortest_exprs
//     shortest_exprs.dedup();

//     // Check each element in shortest_exprs to see if it's in all the OR arguments.
//     let mut exist_exprs: Vec<Predicate> = vec![];
//     for expr in shortest_exprs.iter() {
//         let found = or_predicates.iter().all(|or_predicate| match or_predicate.0 {
//             Predicate::And { args } => args.contains(expr),
//             _ => or_predicate == expr,
//         });
//         if found {
//             exist_exprs.push(expr.0.clone());
//         }
//     }
//     if exist_exprs.is_empty() {
//         return Predicate::Or {
//             args: or_predicates.to_vec(),
//         };
//     }

//     // Rebuild the OR predicate.
//     // (A AND B) OR A will be optimized to A.
//     let mut new_or_predicates = vec![];
//     for or_predicate in or_predicates.iter() {
//         match or_predicate.0 {
//             Predicate::And { args } => {
//                 let mut new_args = args.clone();
//                 new_args.retain(|expr| !exist_exprs.contains(expr));
//                 if !new_args.is_empty() {
//                     if new_args.len() == 1 {
//                         new_or_predicates.push(new_args[0].clone());
//                     } else {
//                         new_or_predicates.push(Predicate::And { args: new_args });
//                     }
//                 } else {
//                     new_or_predicates.clear();
//                     break;
//                 }
//             }
//             _ => {
//                 if exist_exprs.contains(or_predicate) {
//                     new_or_predicates.clear();
//                     break;
//                 }
//             }
//         }
//     }
//     if !new_or_predicates.is_empty() {
//         if new_or_predicates.len() == 1 {
//             exist_exprs.push(new_or_predicates[0].clone());
//         } else {
//             exist_exprs.push(Predicate::Or {
//                 args: flatten_or_predicates(new_or_predicates),
//             });
//         }
//     }

//     if exist_exprs.len() == 1 {
//         exist_exprs[0].clone()
//     } else {
//         Predicate::And {
//             args: flatten_and_predicates(exist_exprs),
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use datafusion::prelude::{col, boolean_and, boolean_or, Expr};

    use super::{predicate, rewrite_predicate};

    fn build_expr() -> Expr{
        let boolean_or = boolean_or(col("3"), col("2"));
        let boolean_expr = boolean_and(col("1"), boolean_or);
        boolean_and(col("4"), boolean_expr)
    }
    #[test]
    fn test_predicate() {
        let boolean_expr = build_expr();
        predicate(&boolean_expr).unwrap();
    }

    #[test]
    fn test_rewrite() {
        let boolean_expr = build_expr();
        let predicate = predicate(&boolean_expr).unwrap();
        let res = rewrite_predicate(predicate.0);
        assert_eq!(res.2, 6); // node num should be 6.
        assert_eq!(res.3, 4); // leaf num should be 4.
    }
}
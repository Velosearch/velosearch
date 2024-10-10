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

use std::vec;

use datafusion_expr::{LogicalPlan, Expr, logical_plan::Boolean, expr::BooleanQuery, Operator};
use datafusion_common::{Result, DataFusionError};
use crate::{OptimizerRule, OptimizerConfig, optimizer::ApplyOrder};

#[derive(Default)]
pub struct RewritePredicateCNF;

impl RewritePredicateCNF {
    pub fn new() -> Self {
        Self::default()
    }
}

impl OptimizerRule for RewritePredicateCNF {
    fn try_optimize(
            &self,
            plan: &LogicalPlan,
            _config: &dyn OptimizerConfig,
    ) -> Result<Option<datafusion_expr::LogicalPlan>> {
        match plan {
            LogicalPlan::Boolean(boolean) => {
                let predicate = predicate(&boolean.binary_expr)?;
                let rewritten_predicate = rewrite_predicate(predicate);
                let rewritten_expr = normalize_predicate(rewritten_predicate);
                Ok(Some(LogicalPlan::Boolean(Boolean::try_new(
                    rewritten_expr, 
                    0,
                    0,
                    boolean.is_score,
                    boolean.input.clone(),
                    boolean.projected_terms.clone(),
                )?)))
            }
            _ => Ok(None)
        }
    }

    fn name(&self) -> &str {
       "rewrite_predicate_cnf" 
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }
}

#[derive(Clone, PartialEq, Debug)]
enum Predicate {
    And { args: Vec<Predicate> },
    Or { args: Vec<Predicate> },
    Expr { expr: Box<Expr> },
}

fn predicate(expr: &Expr) -> Result<Predicate> {
    match expr {
        Expr::BooleanQuery(BooleanQuery { left, op, right }) => match op{
            Operator::BitwiseAnd => {
                let args = vec![predicate(left)?, predicate(right)?];
                Ok(Predicate::And { args })
            }
            Operator::BitwiseOr => {
                let left = predicate(left)?;
                let right = predicate(right)?;
                let predicates = match (left, right) {
                    (Predicate::And { args:l }, Predicate::And { args: r }) => {
                        l.into_iter()
                        .flat_map(|x| {
                            std::iter::repeat(x).zip(&r)
                        })
                        .map(|(l, r)| Predicate::Or { args: vec![l.clone(), r.clone()]})
                        .collect()
                    }
                    (Predicate::And { args: l }, Predicate::Or { args: r }) => {
                        l.into_iter()
                        .map(|p| {
                            Predicate::Or { args: [vec![p], r.clone()].concat() }
                        })
                        .collect()
                    }
                    (Predicate::Or { args: l }, Predicate::And { args: r }) => {
                        r.into_iter()
                        .map(|p| {
                            Predicate::Or { args: [vec![p], l.clone()].concat() }
                        })
                        .collect()
                    }
                    (Predicate::Or { args: l }, Predicate::Or { args: r }) => {
                        vec![Predicate::Or { args: [l, r].concat()}]
                    }
                    (Predicate::And { args: l }, r) => {
                        l.into_iter()
                        .map(|p| {
                            Predicate::Or { args: vec![p.clone(), r.clone()] } 
                        })
                        .collect()
                    }
                    (l, r) =>  {
                        vec![Predicate::Or { args: vec![l, r] }]
                    }
                };
                if predicates.len() == 1 {
                    Ok(predicates[0].to_owned())
                } else {
                    Ok(Predicate::And { args: predicates })
                }
            }
            _ => Err(DataFusionError::Plan(format!("Only supprt operator: BitwiseAnd, BitwiseOr")))
        }
        Expr::Not(boolean) => {

            match boolean.as_ref() {
                Expr::Not(expr) => predicate(expr),
                Expr::BooleanQuery(BooleanQuery { left, op, right }) => match op{
                    Operator::BitwiseAnd => {
                        predicate(&Expr::BooleanQuery(
                            BooleanQuery::new(
                                Box::new(Expr::Not(left.clone())),
                                Operator::BitwiseOr,
                                Box::new(Expr::Not(right.clone())),
                            )
                        ))
                    }
                    Operator::BitwiseOr => {
                        predicate(&Expr::BooleanQuery(
                            BooleanQuery::new(
                                Box::new(Expr::Not(left.clone())),
                                Operator::BitwiseAnd,
                                Box::new(Expr::Not(right.clone())),
                            )
                        ))
                    }
                    _ => Err(DataFusionError::Plan(format!("Only Support BitwiseAnd or BitwiseOr."))),
                }
                _ => Ok(Predicate::Expr {
                    expr: Box::new(expr.clone()),
                })
            }
        }
        _ => Ok(Predicate::Expr {
            expr: Box::new(expr.clone()),
        })
    }
}

fn normalize_predicate(predicate: Predicate) -> Expr {
    match predicate {
        Predicate::And { args } => {
            assert!(args.len() >= 2);
            args.into_iter()
                .map(normalize_predicate)
                .reduce(Expr::boolean_and)
                .expect("had more tahtn one arg")
        }
        Predicate::Or { args } => {
            assert!(args.len() >= 2);
            args.into_iter()
                .map(normalize_predicate)
                .reduce(Expr::boolean_or)
                .expect("had more than one arg")
        }
        Predicate::Expr { expr } => *expr,
    }
}

fn rewrite_predicate(predicate: Predicate) -> Predicate {
    match predicate {
        Predicate::And { args } => {
            let mut rewritten_args = Vec::with_capacity(args.len());
            for arg in args.iter() {
                rewritten_args.push(rewrite_predicate(arg.clone()));
            }
            rewritten_args = flatten_and_predicates(rewritten_args);
            Predicate::And {
                args: rewritten_args,
            }
        }
        Predicate::Or { args } => {
            let mut rewritten_args = vec![];
            for arg in args.iter() {
                rewritten_args.push(rewrite_predicate(arg.clone()));
            }
            rewritten_args = flatten_or_predicates(rewritten_args);
            delete_duplicate_predicates(&rewritten_args)
        }
        Predicate::Expr { expr } => Predicate::Expr { expr: Box::new(*expr) }
    }
}

fn flatten_and_predicates(
    and_predicate: impl IntoIterator<Item = Predicate>,
) -> Vec<Predicate> {
    let mut flatten_predicates = vec![];
    for predicate in and_predicate {
        match predicate {
            Predicate::And { args } => {
                flatten_predicates
                    .extend_from_slice(flatten_and_predicates(args).as_slice());
            }
            _ => {
                flatten_predicates.push(predicate)
            }
        }
    }
    flatten_predicates
}

fn flatten_or_predicates(
    or_predicates: impl IntoIterator<Item = Predicate>,
) -> Vec<Predicate> {
    let mut flattened_predicates = vec![];
    for predicate in or_predicates {
        match predicate {
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

fn delete_duplicate_predicates(or_predicates: &[Predicate]) -> Predicate {
    let mut shortest_exprs: Vec<Predicate> = vec![];
    let mut shortest_exprs_len = 0;
    // choose the shortest AND predicate
    for or_predicate in or_predicates.iter() {
        match or_predicate {
            Predicate::And { args } => {
                let args_num = args.len();
                if shortest_exprs.is_empty() || args_num < shortest_exprs_len {
                    shortest_exprs = (*args).clone();
                    shortest_exprs_len = args_num;
                }
            }
            _ => {
                // if there is no AND predicate, it must be the shortest expression.
                shortest_exprs = vec![or_predicate.clone()];
                break;
            }
        }
    }

    // dedup shortest_exprs
    shortest_exprs.dedup();

    // Check each element in shortest_exprs to see if it's in all the OR arguments.
    let mut exist_exprs: Vec<Predicate> = vec![];
    for expr in shortest_exprs.iter() {
        let found = or_predicates.iter().all(|or_predicate| match or_predicate {
            Predicate::And { args } => args.contains(expr),
            _ => or_predicate == expr,
        });
        if found {
            exist_exprs.push((*expr).clone());
        }
    }
    if exist_exprs.is_empty() {
        return Predicate::Or {
            args: or_predicates.to_vec(),
        };
    }

    // Rebuild the OR predicate.
    // (A AND B) OR A will be optimized to A.
    let mut new_or_predicates = vec![];
    for or_predicate in or_predicates.iter() {
        match or_predicate {
            Predicate::And { args } => {
                let mut new_args = (*args).clone();
                new_args.retain(|expr| !exist_exprs.contains(expr));
                if !new_args.is_empty() {
                    if new_args.len() == 1 {
                        new_or_predicates.push(new_args[0].clone());
                    } else {
                        new_or_predicates.push(Predicate::And { args: new_args });
                    }
                } else {
                    new_or_predicates.clear();
                    break;
                }
            }
            _ => {
                if exist_exprs.contains(or_predicate) {
                    new_or_predicates.clear();
                    break;
                }
            }
        }
    }
    if !new_or_predicates.is_empty() {
        if new_or_predicates.len() == 1 {
            exist_exprs.push(new_or_predicates[0].clone());
        } else {
            exist_exprs.push(Predicate::Or {
                args: flatten_or_predicates(new_or_predicates),
            });
        }
    }

    if exist_exprs.len() == 1 {
        exist_exprs[0].clone()
    } else {
        Predicate::And {
            args: flatten_and_predicates(exist_exprs),
        }
    }
}

#[cfg(test)]
mod test {
    use datafusion_common::{Result, ScalarValue};
    use datafusion_expr::{col, lit, boolean_and, boolean_or, Expr};

    use crate::rewrite_predicate_cnf::{predicate, Predicate, rewrite_predicate, normalize_predicate};

    #[test]
    fn test_rewrite_predicate() -> Result<()> {
        let equi_expr = col("t1.a").eq(col("t2.b"));
        let gt_expr = col("t1.c").gt(lit(ScalarValue::Int8(Some(1))));
        let lt_expr = col("t1.d").lt(lit(ScalarValue::Int8(Some(2))));
        let expr = boolean_or(
            boolean_and(equi_expr.clone(), gt_expr.clone()),
            boolean_and(equi_expr.clone(), lt_expr.clone()),
        );
        let predicate = predicate(&expr)?;
        assert_eq!(
            predicate,
            Predicate::And {
                args: vec![
                    Predicate::Or {
                        args: vec![
                            Predicate::Expr {
                                expr: Box::new(equi_expr.clone())
                            },
                            Predicate::Expr {
                                expr: Box::new(equi_expr.clone())
                            },
                        ]
                    },
                    Predicate::Or {
                        args: vec![
                            Predicate::Expr {
                                expr: Box::new(equi_expr.clone())
                            },
                            Predicate::Expr {
                                expr: Box::new(lt_expr.clone())
                            },
                        ]
                    },
                    Predicate::Or {
                        args: vec![
                            Predicate::Expr {
                                expr: Box::new(gt_expr.clone())
                            },
                            Predicate::Expr {
                                expr: Box::new(equi_expr.clone())
                            }
                        ]
                    },
                    Predicate::Or {
                        args: vec![
                            Predicate::Expr {
                                expr: Box::new(gt_expr.clone())
                            },
                            Predicate::Expr {
                                expr: Box::new(lt_expr.clone())
                            }
                        ]
                    }
                ]
            }
        );
        println!("success");
        let rewritten_predicate = rewrite_predicate(predicate);
        assert_eq!(
            format!("{:?}", rewritten_predicate),
            "And { args: [Expr { expr: t1.a = t2.b }, Or { args: [Expr { expr: t1.a = t2.b }, \
            Expr { expr: t1.d < Int8(2) }] }, Or { args: [Expr { expr: t1.c > Int8(1) }, Expr \
            { expr: t1.a = t2.b }] }, Or { args: [Expr { expr: t1.c > Int8(1) }, Expr { expr: t1.d < Int8(2) }] }] }"
        );
        let rewritten_expr = normalize_predicate(rewritten_predicate);
        assert_eq!(
            format!("{:}", rewritten_expr),
            "t1.a = t2.b & (t1.a = t2.b | t1.d < Int8(2)) & (t1.c > Int8(1) \
             | t1.a = t2.b) & (t1.c > Int8(1) | t1.d < Int8(2))"
        );
        Ok(())
    }

    #[test]
    fn rewrite_not_predicate() -> Result<()> {
        let a = col("1");
        let b = col("2");
        let d = col("4");
        let expr = boolean_and(
            boolean_and(a.clone(), b.clone()),
            Expr::Not(Box::new(
                boolean_and(
                    d.clone(),
                    boolean_or(d.clone(), a.clone())
            ))
        ));
        let predicate = predicate(&expr)?;
        assert_eq!(
            predicate,
            Predicate::And { 
                args: vec![
                    Predicate::And {
                        args: vec![
                            Predicate::Expr {
                                expr: Box::new(a.clone())
                            },
                            Predicate::Expr {
                                expr: Box::new(b.clone())
                            },
                        ]
                    },
                    Predicate::Or {
                        args: vec![
                            Predicate::Expr {
                                expr: Box::new(Expr::Not(Box::new(d.clone())))
                            },
                            Predicate::And {
                                args: vec![
                                    Predicate::Expr{
                                        expr: Box::new(Expr::Not(Box::new(d.clone()))),
                                    },
                                    Predicate::Expr{
                                        expr: Box::new(Expr::Not(Box::new(a.clone())))
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        );
        let rewritten_predicate = rewrite_predicate(predicate);
        assert_eq!(
            rewritten_predicate,
            Predicate::And {
                args: vec![
                    Predicate::Expr {
                        expr: Box::new(a.clone())
                    },
                    Predicate::Expr {
                        expr: Box::new(b.clone())
                    },
                    Predicate::Expr {
                        expr: Box::new(Expr::Not(Box::new(d.clone())))
                    }
                ]
            }
        );
        Ok(())
    }

    #[test]
    fn test_or_fold() -> Result<()> {
        let a = col("a");
        let b = col("b");
        let c = col("c");
        let d = col("d");
        let expr = boolean_or(
            boolean_or(a.clone(), b.clone()),
            boolean_and(c.clone(), d.clone())
        );
        let predicate = predicate(&expr)?;
        assert_eq!(
            predicate,
            Predicate::Or { args: vec![
                Predicate::Or { args: vec![
                    Predicate::Expr { expr: Box::new(a.clone()) },
                    Predicate::Expr { expr: Box::new(b.clone()) }
                ] },
                Predicate::And {
                    args: vec![
                        Predicate::Expr { expr: Box::new(c.clone())},
                        Predicate::Expr { expr: Box::new(d.clone())}
                    ]
                }
            ] }
        );

        let rewritten_predicate = rewrite_predicate(predicate);
        assert_eq!(
            rewritten_predicate,
            Predicate::And { args: vec![] }
        );

        Ok(())
    }
}
use std::{sync::Arc, collections::HashMap};

use datafusion::{
    prelude::{col, Expr, boolean_or, boolean_and, create_udaf, Column}, 
    logical_expr::{LogicalPlan, LogicalPlanBuilder, Projection, Volatility, Aggregate}, 
    execution::context::{SessionState, TaskContext}, 
    error::DataFusionError, 
    physical_plan::{ExecutionPlan, collect}, 
    arrow::{record_batch::RecordBatch, util::pretty, datatypes::DataType}, common::{DFSchema, DFSchemaRef, DFField}
};
use tokio::time::Instant;
use tracing::{debug, info};

use crate::{Result, utils::FastErr, physical_expr::CountValid};




/// A query that matches documents matching boolean combinations of other queries.
/// The bool query maps to `Lucene BooleanQuery` except `filter`. It's built using one or more
/// boolean clauses, each clause with a typed occurrence. The occurrence types are:
/// | Occur  | Description |
/// | :----: | :---------: |
/// | must | The clause must appear in matching documents |
/// | should | The clause should appear in the matching document |
/// | must_not | The clause must not appear in the matching documents |
/// 
/// 
pub struct BooleanPredicateBuilder {
   predicate: Expr,
}

impl BooleanPredicateBuilder {
    pub fn must(terms: &[&str]) -> Result<Self> {
        let mut terms = terms.into_iter();
        if terms.len() <= 1 {
            return Err(DataFusionError::Internal("The param of terms should at least two items".to_string()).into())
        }
        let mut predicate = boolean_and(col(*terms.next().unwrap()), col(*terms.next().unwrap()));
        for &expr in terms {
            predicate = boolean_and(predicate, col(expr));
        }
        Ok(BooleanPredicateBuilder {
            predicate
        })
    }

    pub fn with_must(self, right: BooleanPredicateBuilder) -> Result<Self> {
        Ok(BooleanPredicateBuilder { 
            predicate: boolean_and(self.predicate, right.predicate)
        })
    }

    pub fn should(terms: &[&str]) -> Result<Self> {
        let mut terms = terms.into_iter();
        if terms.len() <= 1 {
            return Err(DataFusionError::Internal("The param of terms should at least two items".to_string()).into())
        }
        let mut predicate = boolean_or(col(*terms.next().unwrap()), col(*terms.next().unwrap()));
        for &expr in terms {
            predicate = boolean_or(predicate, col(expr));
        }
        Ok(BooleanPredicateBuilder { 
            predicate: predicate
        })
    }

    pub fn with_should(self, right: BooleanPredicateBuilder) -> Result<Self> {
        Ok(BooleanPredicateBuilder {
            predicate: boolean_or(self.predicate, right.predicate)
        })
    }

    pub fn build(self) -> Expr {
        self.predicate
    }

    // pub fn with_must_not(self) -> result<BooleanPredicateBuilder> {
    //     unimplemented!()
    //     // Ok(BooleanPredicateBuilder {
    //     // })
    // }
}

/// BooleanQuery represents a full-text search query.
/// 
#[derive(Debug, Clone)]
pub struct BooleanQuery {
    plan: LogicalPlan,
    session_state: SessionState
}

impl BooleanQuery {
    /// Create a new BooleanQuery
    pub fn new(plan: LogicalPlan, session_state: SessionState) -> Self {
        Self {
            plan,
            session_state
        }
    }

    /// Create BooleanQuery based on a bitwise binary operation expression
    pub fn boolean_predicate(self, predicate: Expr, is_score: bool) -> Result<Self> {
        debug!("Build boolean_predicate");
        let mut project_exprs = binary_expr_columns(&predicate);
        project_exprs.push(col("__id__"));
        // let input_schema = DFSchema {
        //     fields: project_exprs.iter().map(|e| if let Expr::Column(c) = e {
        //         DFField::new(c.relation.as_ref().map(|v| v.as_str()), &c.name, DataType::Boolean, false)
        //     } else {
        //         unreachable!()
        //     }).collect(),
        //     metadata: HashMap::new(),
        // };
        match predicate {
            Expr::BooleanQuery(expr) => {
                // let project_plan = boolean_project(self.plan, project_exprs, input_schema).unwrap();
                // let project_plan = LogicalPlanBuilder::from(project_plan).build()?;
                Ok(Self {
                    plan: LogicalPlanBuilder::from(self.plan).boolean(Expr::BooleanQuery(expr), is_score, vec![])?.build()?,
                    session_state: self.session_state 
                })
            },
            _ => Err(FastErr::UnimplementErr("Predicate expression must be the BinaryExpr".to_string()))
        }   
    } 

    /// Count query
    pub fn count_agg(self) -> Result<Self> {
        // Add custom Aggregation Function
        let count_valid = create_udaf(
            "count_valid",
            DataType::Boolean,
            Arc::new(DataType::Int64),
            Volatility::Immutable,
            Arc::new(|_| Ok(Box::new(CountValid::new()))),
            Arc::new(vec![DataType::Int64]),
        );
        let aggregate = Aggregate::try_new_with_schema(
                Arc::new(self.plan),
                vec![],
                vec![count_valid.call(vec![Expr::Column(Column::new(Some(format!("__table__")), "mask"))])],
                Arc::new(DFSchema::new_with_metadata(vec![DFField::new(Some("__table__"), "mask", DataType::Boolean, false)], HashMap::new())?),
            )?;
        Ok(Self {
            plan: LogicalPlan::Aggregate(aggregate),
            session_state: self.session_state,
        })
    }

    /// Create a physical plan
    pub async fn create_physical_plan(self) -> Result<Arc<dyn ExecutionPlan>> {
        self.session_state
            .create_physical_plan(&self.plan).await
            .map_err(|e| FastErr::DataFusionErr(e))
    }

    /// Return a BooleanQuery with the explanation of its plan so far
    /// 
    /// if `analyze` is specified, runs the plan and reports metrics
    /// 
    pub fn explain(self, verbose: bool, analyze: bool) -> Result<BooleanQuery> {
        debug!("explain verbose: {:}, analyze: {:}", verbose, analyze);
        let plan = LogicalPlanBuilder::from(self.plan)
            .explain(verbose, analyze)?
            .build()?;
        Ok(BooleanQuery::new(plan, self.session_state))
    }

    /// Print results
    /// 
    pub async fn show(self) -> Result<()> {
        let results = self.collect().await?;
        debug!("Show() collect result from self.collect()");
        Ok(pretty::print_batches(&results)?)
    }

    /// Convert the logical plan represented by this BooleanQuery into a physical plan and
    /// execute it, collect all resulting batches into memory
    /// Executes this DataFrame and collects all results into a vecotr of RecordBatch.
    pub async fn collect(self) -> Result<Vec<RecordBatch>> {
        let task_ctx = Arc::new(self.task_ctx());
        debug!("Create physical plan");
        let plan = self.create_physical_plan().await?;
        debug!("Finish physical plan");
        let timer = Instant::now();
        let res = collect(plan, task_ctx).await.map_err(|e| FastErr::DataFusionErr(e));
        info!("Result Collect took {} us", timer.elapsed().as_micros());
        res
    }

    fn task_ctx(&self) -> TaskContext {
        TaskContext::from(&self.session_state)
    }

}

pub fn binary_expr_columns(be: &Expr) -> Vec<Expr> {
    debug!("Binary expr columns: {:?}", be);
    match be {
        Expr::BooleanQuery(b) => {
            let mut left_columns = binary_expr_columns(&b.left);
            left_columns.extend(binary_expr_columns(&b.right));
            left_columns
        },
        Expr::Column(c) => {
            vec![Expr::Column(c.clone())]
        },
        Expr::Literal(_) => { Vec::new() },
        _ => unreachable!()
    }
}

pub fn boolean_project(
        plan: LogicalPlan,
        expr: impl IntoIterator<Item = impl Into<Expr>>,
        input_dfschema: DFSchema,
    ) -> datafusion::common::Result<LogicalPlan> {
        let projected_expr: Vec<Expr> = expr.into_iter().map(|e| e.into()).collect();

        Ok(LogicalPlan::Projection(Projection::try_new_with_schema(
            projected_expr,
            Arc::new(plan.clone()),
            DFSchemaRef::new(input_dfschema),
        )?))
    }


#[cfg(test)]
pub mod tests {
    // use std::sync::Arc;
    // use std::time::Instant;

    // use datafusion::arrow::array::{UInt16Array, BooleanArray};
    // use datafusion::arrow::datatypes::{Schema, Field, DataType};
    // use datafusion::common::TermMeta;
    // use datafusion::from_slice::FromSlice;
    // use datafusion::prelude::col;
    // use adaptive_hybrid_trie::TermIdx;
    // use tracing::{Level};
    // use crate::batch::{BatchRange, PostingBatch};
    // use crate::{utils::Result, BooleanContext, datasources::posting_table::PostingTable};

    // use super::{BooleanPredicateBuilder, binary_expr_columns};

    // #[test]
    // fn boolean_must_builder() -> Result<()> {
    //     let predicate = BooleanPredicateBuilder::must(&["a", "b", "c"])?;
    //     assert_eq!(format!("{}", predicate.build()).as_str(), "((a & b) & c) = Int8(1)");
    //     Ok(())
    // }

    // #[test]
    // fn boolean_should() -> Result<()> {
    //     let predicate = BooleanPredicateBuilder::should(&["a", "b", "c"])?;
    //     assert_eq!(format!("{}", predicate.build()).as_str(), "((a | b) | c) = Int8(1)");
    //     Ok(())
    // }

    // #[test]
    // fn binary_expr_children_test() -> Result<()> {
    //     let predicate = BooleanPredicateBuilder::should(&["a", "b", "c"])?;
    //     assert_eq!(binary_expr_columns(&predicate.build()), vec![col("a"), col("b"), col("c")]);
    //     Ok(())
    // }

    // pub fn make_posting_schema(fields: Vec<&str>) -> Schema {
    //     Schema::new(
    //         fields.into_iter()
    //         .map(|f| if f == "__id__" {
    //                 Field::new(f, DataType::UInt32, false)
    //             } else {
    //                 Field::new(f, DataType::Boolean, false)
    //             })
    //         .collect()
    //     )
    // }

    

    // // #[tokio::test]
    // // async fn simple_query() -> Result<()> {
    // //     tracing_subscriber::fmt().with_max_level(Level::DEBUG).init();
    // //     let schema = Arc::new(make_posting_schema(vec!["__id__", "a", "b", "c", "d"]));
    // //     let range = Arc::new(BatchRange::new(0, 20));

    // //     let batch = PostingBatch::try_new(
    // //         schema.clone(),
    // //         vec![
    // //             Arc::new(UInt16Array::from_iter_values((0..20).into_iter())),
    // //             Arc::new(UInt16Array::from_slice([0, 2, 6, 8, 15])),
    // //             Arc::new(UInt16Array::from_slice([0, 4, 6, 13, 17])),
    // //             Arc::new(UInt16Array::from_slice([3, 7, 11, 17, 19])),
    // //             Arc::new(UInt16Array::from_slice([6, 7, 9, 14, 18]))
    // //         ],
    // //         range.clone()
    // //     ).expect("Can't try new a PostingBatch");

    // //     let mut term_idx = TermIdx::new();
    // //     term_idx.insert("a".to_string(), TermMeta {
    // //         distribution: Arc::new(BooleanArray::from_slice(&[true])),
    // //         nums: 5,
    // //         index: Arc::new(UInt16Array::from(vec![Some(1)])),
    // //         selectivity: 0.,
    // //     });
    // //     term_idx.insert("b".to_string(), TermMeta {
    // //         distribution: Arc::new(BooleanArray::from_slice(&[true])),
    // //         nums: 5,
    // //         index: Arc::new(UInt16Array::from(vec![Some(2)])),
    // //         selectivity: 0.,
    // //     });
    // //     term_idx.insert("c".to_string(), TermMeta {
    // //         distribution: Arc::new(BooleanArray::from_slice(&[true])),
    // //         nums: 5,
    // //         index: Arc::new(UInt16Array::from(vec![Some(3)])),
    // //         selectivity: 0.,
    // //     });
    // //     term_idx.insert("d".to_string(), TermMeta {
    // //         distribution: Arc::new(BooleanArray::from_slice(&[true])),
    // //         nums: 5,
    // //         index: Arc::new(UInt16Array::from(vec![Some(4)])),
    // //         selectivity: 0.
    // //     });
    // //     term_idx.insert("__id__".to_string(), TermMeta {
    // //         distribution: Arc::new(BooleanArray::from_slice(&[false])),
    // //         nums: 0,
    // //         index: Arc::new(UInt16Array::from(vec![Some(0)])),
    // //         selectivity: 0.,
    // //     });

    // //     let session_ctx = BooleanContext::new();
    // //     session_ctx.register_index("t", Arc::new(PostingTable::new(
    // //         schema.clone(),
    // //         vec![Arc::new(term_idx)],
    // //         vec![Arc::new(vec![batch])],
    // //         &BatchRange::new(0, 20)
    // //     ))).unwrap();

    // //     let index = session_ctx.index("t").await.unwrap();
    // //     let t = Instant::now();
    // //     index.boolean_predicate(BooleanPredicateBuilder::must(&["a", "b"]).unwrap().build()).unwrap()
    // //         .show().await.unwrap();
    // //     println!("{}", t.elapsed().as_nanos());
    // //     panic!("");
    // // }
}
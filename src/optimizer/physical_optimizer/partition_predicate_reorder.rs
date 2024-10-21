//! PartitionPredicateReorder optimizer that reorder the predicate in 
//! every partition

use std::sync::Arc;
use datafusion::{
    common::Result,
    physical_optimizer::PhysicalOptimizerRule, physical_plan::{ExecutionPlan, boolean::BooleanExec}, config::ConfigOptions, physical_expr::BooleanQueryExpr,

};

/// PartitionPredicateReorder optimizer that reorder the predicate in
/// every partition
#[derive(Default)]
pub struct PartitionPredicateReorder {}

impl PartitionPredicateReorder {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self::default()
    }
}

impl PhysicalOptimizerRule for PartitionPredicateReorder {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(boolean) = plan.as_any().downcast_ref::<BooleanExec>() {
            match &boolean.terms_stats {
                Some(_stats) => {
                    let predicate = boolean.predicate.clone();
                    let predicate = predicate.as_any().downcast_ref::<BooleanQueryExpr>().expect("Predicate should be BooleanQueryExpr");
                    if let Some(ref _cnf) = predicate.cnf_predicates {
                        // let reorder_predicate: HashMap<usize, Arc<dyn PhysicalExpr>> = stats
                        // .iter()
                        // .map(|_p| {
                        //     // TODO
                        //     let mut cnf: Vec<Dnf> = cnf
                        //         .iter()
                        //         .map(|dnf| dnf)
                        //         .cloned()
                        //         .collect();
                        //     cnf.sort_by(|l, r| l.selectivity().partial_cmp(&r.selectivity()).unwrap());
                        //     cnf
                        // })
                        // .map(|c| Arc::new(BooleanQueryExpr::new_with_cnf(predicate.predicate_tree.clone(), c)) as Arc<dyn PhysicalExpr>)
                        // .enumerate()
                        // .collect();
                        Ok(Arc::new(BooleanExec::try_new(
                            boolean.predicate.clone(),
                            boolean.input.clone(),
                            boolean.terms_stats.clone(),
                            boolean.is_score,
                            boolean.projected_terms.clone(),
                        )?))
                    } else {
                        Ok(plan)
                    }
                }
                None => Ok(plan)
            }
        } else {
            Ok(plan)
        }
    }

    fn name(&self) -> &str {
        "PartitionPredicateReorder"
    }

    fn schema_check(&self) -> bool {
        false
    }
}

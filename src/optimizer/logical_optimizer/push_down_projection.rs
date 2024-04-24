use std::collections::HashSet;

use datafusion::{optimizer::{OptimizerRule, OptimizerConfig}, logical_expr::LogicalPlan, prelude::Column};
use datafusion::common::Result;

/// Optimizer that remove unused projections from plans
#[derive(Default)]
pub struct PushDownProjection {}

impl OptimizerRule for PushDownProjection {
    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Option<LogicalPlan>> {
        // set all columns referred by the plan (and thus considered required by the root)
        let required_columns = plan
            .schema()
            .fields()
            .iter()
            .map(|f| f.qualified_column())
            .collect::<HashSet<Column>>();
        Ok(Some(optimize_plan(
            self,
            plan,
            &required_columns,
            false,
            config,
        )?))
    }

    fn name(&self) -> &str {
        "push_down_projection"
    }
}

impl PushDownProjection {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

/// Recursively transverses the logical plan removing expressions and that are not needed
fn optimize_plan(
    _optimizer: &PushDownProjection,
    _plan: &LogicalPlan,
    _required_columns: &HashSet<Column>,
    _hash_join: bool,
    _config: &dyn OptimizerConfig,
) -> Result<LogicalPlan> {
    unimplemented!()
}
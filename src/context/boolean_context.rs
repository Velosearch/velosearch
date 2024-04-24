
use std::{sync::Arc, collections::{HashSet, BTreeSet}};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use datafusion::{
    execution::{context::{SessionState, QueryPlanner}, runtime_env::RuntimeEnv}, 
    prelude::{SessionConfig, Expr, Column, create_udaf}, sql::TableReference, logical_expr::{LogicalPlanBuilder, LogicalPlan, TableSource, Volatility}, 
    datasource::{provider_as_source, TableProvider}, error::DataFusionError, 
    optimizer::OptimizerRule, 
    physical_optimizer::PhysicalOptimizerRule, scalar::ScalarValue, physical_plan::{PhysicalPlanner, ExecutionPlan}, arrow::datatypes::{Schema, DataType}
};
use parking_lot::RwLock;
use tokio::time::Instant;
use tracing::debug;

use crate::{query::boolean_query::BooleanQuery, utils::FastErr, BooleanPhysicalPlanner, MinOperationRange, RewriteBooleanPredicate, PrimitivesCombination, physical_expr::CountValid};
use crate::utils::Result;

#[derive(Clone)]
pub struct BooleanContext {
    /// UUID for this context
    session_id: String, 
    /// Session start time
    session_start_time: DateTime<Utc>,
    ///  Shared session state for the session
    state: Arc<RwLock<SessionState>>,
}

impl Default for BooleanContext {
    fn default() -> Self {
        Self::new()
    }
}

impl BooleanContext {
    pub fn new() -> Self {
        let config = SessionConfig::new()
            .set("datafusion.optimizer.max_passes", ScalarValue::UInt8(Some(0x1)));
        Self::with_config(config)
    }

    /// Creates a new session context using the provided session configuration.
    pub fn with_config(config: SessionConfig) -> Self {
        let runtime = Arc::new(RuntimeEnv::default());
        Self::with_config_rt(config, runtime)
    }

     /// Creates a new session context using the provided configuration and [`RuntimeEnv`].
     pub fn with_config_rt(config: SessionConfig, runtime: Arc<RuntimeEnv>) -> Self {
        let state = SessionState::with_config_rt(config, runtime);
        // Add custom optimizer rules
        let state = state.with_optimizer_rules(optimizer_rules());
        // Add custom physical optimizer rules
        let state = state.with_physical_optimizer_rules(physical_optimizer_rulse());
        // Add custom Physical Planner
        let state = state.with_query_planner(Arc::new(BooleanPlanner {}));
        Self::with_state(state)
    }

    /// Creates a new session context using the provided session state.
    pub fn with_state(state: SessionState) -> Self {
        Self {
            session_id: state.session_id().to_string(),
            session_start_time: Utc::now(),
            state: Arc::new(RwLock::new(state)),
        }
    }

    /// Retrieves a [`Index`] representing a table previously
    /// registered by calling the [`register_index`] funciton
    /// 
    /// Return an error if no tables has been registered with the 
    /// provided reference.
    /// 
    /// [`register_index`]: BooleanContext::register_index
    pub async fn index<'a>(
        &self,
        index_ref: impl Into<TableReference<'a>>,
    ) -> Result<BooleanQuery> {
        let index_ref = index_ref.into();
        let index = index_ref.table().to_owned();
        let provider = self.index_provider(index_ref).await?;
        let plan = LogicalPlanBuilder::scan(
            &index,
            provider_as_source(Arc::clone(&provider)),
            None,
        )?.build()?;
        Ok(BooleanQuery::new(plan, self.state()))
    }

    /// Return BooleanQuery with index and predicate
    pub async fn boolean<'a>(
        &self,
        index_ref: impl Into<TableReference<'a>>,
        predicate: Expr,
        is_score: bool,
    ) -> Result<BooleanQuery> {
        let index_ref = index_ref.into();
        let index = index_ref.table().to_owned();
        let provider = self.index_provider(index_ref).await?;
        let schema = &provider.schema();
        let project_exprs = [binary_expr_columns(&predicate), vec![Column::from_name("__id__")]].concat();
        let project = project_exprs
            .iter()
            .map(|e| schema.index_of(&e.name).unwrap_or(usize::MAX))
            .collect();
        if let Expr::BooleanQuery(expr) = predicate {
            let plan = LogicalPlanBuilder::scan(
                &index,
                provider_as_source(Arc::clone(&provider)),
                Some(project),
            )?.boolean(Expr::BooleanQuery(expr), is_score)?.build()?;
            Ok(BooleanQuery::new(
                plan,
                self.state(),
            ))
        } else {
            Err(FastErr::UnimplementErr(format!("")))
        }
    }

    /// Without provider overhide
    pub async fn boolean_with_provider<'a>(
        &self,
        table_source: Arc<dyn TableSource>,
        schema: &Schema,
        predicate: Expr,
        is_score: bool,
    ) -> Result<BooleanQuery> {
        debug!("start boolean builder");
        let project_exprs = binary_expr_columns(&predicate);
        let mut project: Vec<usize> = project_exprs
            .iter()
            .map(|e| schema.index_of(&e.name).unwrap_or(usize::MAX))
            .collect::<BTreeSet<usize>>()
            .into_iter()
            .collect();
        project.push(schema.index_of("__id__").unwrap());
        debug!("end project");
        if let Expr::BooleanQuery(expr) = predicate {
            let plan = LogicalPlanBuilder::scan(
                "__table__",
                table_source,
                Some(project),
            )?.boolean(Expr::BooleanQuery(expr), is_score)?.build()?;
            debug!("end build boolean query");
            Ok(BooleanQuery::new(
                plan,
                self.state(),
            ))
        } else {
            Err(FastErr::UnimplementErr(format!("")))
        }
    }

    /// Return a [`IndexProvider`] for the specified table.
    pub async fn index_provider<'a>(
        &self,
        index_ref: impl Into<TableReference<'a>>,
    ) -> Result<Arc<dyn TableProvider>> {
        let index_ref = index_ref.into();
        let index = index_ref.table().to_owned();
        let schema = self.state.read().schema_for_ref(index_ref)?;
        match schema.table(&index).await {
            Some(ref provider) => Ok(Arc::clone(provider)),
            _ => Err(FastErr::DataFusionErr(DataFusionError::Plan(format!("No index named '{index}'")))),
        }
    }

    /// Snapshots the [`SessionState`] of this [`BooleanContext`] setting the
    /// `query_execution_start_time` to the current time
    pub fn state(&self) -> SessionState {
        let mut state = self.state.read().clone();
        state.execution_props.start_execution();
        state
    }

    /// Register a [`TableProvider`] as a table that can be
    /// referenced from SQL statements executed against this context.
    /// 
    /// Return the [`TableProvider`] previously registered for this 
    /// reference, if any
    pub fn register_index<'a>(
        &'a self,
        index_ref: impl Into<TableReference<'a>>,
        provider: Arc<dyn TableProvider>,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        let index_ref = index_ref.into();
        let index = index_ref.table().to_owned();
        self.state
            .read()
            .schema_for_ref(index_ref)?
            .register_table(index, provider)
            .map_err(|e| e.into())
    }

    /// Return the session_id
    pub fn session_id(&self) -> String {
        self.session_id.clone()
    }

    /// Return the session_start_time
    pub fn session_start_time(&self) -> DateTime<Utc> {
        self.session_start_time.clone()
    }
}

fn optimizer_rules() -> Vec<Arc<dyn OptimizerRule + Sync + Send>> {
    vec![
        // Arc::new(SimplifyExpressions::new()),
        Arc::new(RewriteBooleanPredicate::new()),
        // Arc::new(SimplifyExpressions::new()),
        // Arc::new(EliminateFilter::new()),
        // Arc::new(PushDownFilter::new()),
        // Arc::new(SimplifyExpressions::new()),
        // Arc::new(PushDownProjection::new())
    ]
}

fn physical_optimizer_rulse() -> Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> {
    vec![
        // Arc::new(Repartition::new()),
        Arc::new(MinOperationRange::new()),
        // Arc::new(PrimitivesCombination::new()),
        // Arc::new(PartitionPredicateReorder::new()),
        // Arc::new(IntersectionSelection::new()),
        // Arc::new(EnforceDistribution::new()),
        // Arc::new(CoalesceBatches::new()),
        // Arc::new(PipelineChecker::new())
    ]
}

struct BooleanPlanner {}

#[async_trait]
impl QueryPlanner for BooleanPlanner {
    /// Given a `LogicalPlan`, create an `ExecutionPlan` suitable for execution
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let timer = Instant::now();
        let planner = BooleanPhysicalPlanner::default();
        let plan = planner
            .create_physical_plan(logical_plan, session_state)
            .await;
        debug!("Physical Optimizer took {} us", timer.elapsed().as_micros());
        plan
    }
}

pub fn binary_expr_columns(be: &Expr) -> Vec<Column> {
    match be {
        Expr::BooleanQuery(b) => {
            let mut left_columns = binary_expr_columns(&b.left);
            left_columns.extend(binary_expr_columns(&b.right));
            left_columns
        },
        Expr::Column(c) => {
            vec![c.clone()]
        },
        Expr::Literal(_) => { Vec::new() },
        _ => unreachable!()
    }
}
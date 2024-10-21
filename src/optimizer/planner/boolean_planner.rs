use std::{sync::Arc, collections::HashMap};

use async_trait::async_trait;
use datafusion::{
    physical_plan::{
        PhysicalPlanner, ExecutionPlan, PhysicalExpr, 
        expressions::{Column, Literal, binary, self}, 
        explain::ExplainExec, projection::ProjectionExec, boolean::BooleanExec, displayable, analyze::AnalyzeExec, AggregateExpr, aggregates::{self, PhysicalGroupBy, AggregateExec, AggregateMode}, udaf}, 
        execution::context::SessionState, error::{Result, DataFusionError}, 
        logical_expr::{
            LogicalPlan, expr::{BooleanQuery, AggregateFunction}, BinaryExpr, PlanType, ToStringifiedPlan, Projection, TableScan, StringifiedPlan, Operator, logical_plan::Predicate, Aggregate 
        },
        common::DFSchema, arrow::datatypes::{Schema, SchemaRef}, 
        prelude::Expr, physical_expr::execution_props::ExecutionProps, datasource::source_as_provider, 
        physical_optimizer::PhysicalOptimizerRule
    };
use futures::{future::BoxFuture, FutureExt};
use tracing::{debug, trace};

use crate::{datasources::{mmap_table::MmapExec, posting_table::PostingExec}, physical_expr::{boolean_eval::{PhysicalPredicate, Primitives, SubPredicate}, BooleanEvalExpr}};

/// Boolean physical query planner that converts a
/// `LogicalPlan` to an `ExecutionPlan` suitable for execution.
#[derive(Default)]
pub struct BooleanPhysicalPlanner { }

#[async_trait]
impl PhysicalPlanner for BooleanPhysicalPlanner {
    /// Create a physical plan from a logical plan
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match self.handle_explain(logical_plan, session_state).await? {
            Some(plan) => Ok(plan),
            None => {
                let plan = self
                    .create_boolean_plan(logical_plan, session_state)
                    .await?;
                self.optimize_internal(plan, session_state, |_, _| {})
            }
        }
    }

    fn create_physical_expr(
        &self,
        expr: &Expr,
        input_dfschema: &DFSchema,
        input_schema: &Schema,
        session_state: &SessionState,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        create_physical_expr(
            expr,
            input_dfschema,
            input_schema,
            session_state.execution_props(),
        )
    }
}

impl BooleanPhysicalPlanner {
    /// Create a physical plan from a logical plan
    fn create_boolean_plan<'a>(
        &'a self,
        logical_plan: &'a LogicalPlan,
        session_state: &'a SessionState,
    ) -> BoxFuture<'a, Result<Arc<dyn ExecutionPlan>>> {
        async move {
            let exec_plan: Result<Arc<dyn ExecutionPlan>> = match logical_plan {
                LogicalPlan::TableScan(TableScan {
                    source,
                    projection,
                    filters,
                    fetch,
                    ..
                }) => {
                    debug!("projection: {:?}", projection);
                    let source = source_as_provider(source)?;
                    debug!("source as provider");
                    // Remove all qualifiers from the scan as the provider
                    // doesn't know (nor should care) how the relation was
                    // referred to in the query
                    // let filters = unnormalize_cols(filters.iter().cloned());
                    // let unaliased: Vec<Expr> = filters.into_iter().map(unalias).collect();
                    source.scan(session_state, projection.as_ref(), &filters, *fetch).await
                }
                LogicalPlan::Projection(Projection { input, expr, ..}) => {
                    let input_exec = self.create_boolean_plan(input, session_state).await?;
                    let input_dfschema = input.as_ref().schema();
                    let input_schema: Arc<Schema> = input_exec.schema();

                    let physical_exprs = expr
                        .iter()
                        .map(|e| {
                            let physical_name = if let Expr::Column(col) = e {
                                    match input_schema.index_of(&col.name) {
                                        Ok(idx) => {
                                            // index physical field using logical fields index
                                            Ok(input_exec.schema().field(idx).name().to_string())
                                        }
                                        // logical column is not a derived column, safe to pass along to
                                        // physical_name
                                        Err(_) => physical_name(e),
                                    }
                                } else {
                                    physical_name(e)
                                };

                                tuple_err((
                                    self.create_physical_expr(
                                        e,
                                        input_dfschema,
                                        &input_schema,
                                        session_state,
                                    ),
                                    physical_name,
                                ))
                            }).collect::<Result<Vec<_>>>()?;
                    
                    Ok(Arc::new(ProjectionExec::try_new(
                        physical_exprs,
                        input_exec,
                    )?))
                }
                LogicalPlan::Boolean(boolean) => {
                    debug!("Create boolean plan");
                    let physical_input = self.create_boolean_plan(&boolean.input, session_state).await?;
                    if let Some(posting)= physical_input.as_any().downcast_ref::<PostingExec>() {
                        let posting = posting.to_owned();
                        debug!("Create boolean predicate");
                        let runtime_expr: Arc<dyn PhysicalExpr> = if let Some(ref predicate) = boolean.predicate {
                            let inputs: Vec<&str> = boolean.projected_terms.iter().map(String::as_str).collect();
                            let (term2idx, term2sel): (HashMap<&str, usize>, HashMap<&str, f64>) = inputs
                                .into_iter()
                                .enumerate()
                                .filter(|(_, s)| *s != "__id__")
                                .map(|(i, s)| {
                                    ((s, i), (s, 0.))
                                })
                                .unzip();
                            let builder = PhysicalPredicateBuilder::new(predicate, term2idx, term2sel);
                            let physical_predicate = builder.build()?;
                            Arc::new(BooleanEvalExpr::new(physical_predicate))
                        } else {
                            unreachable!()
                        };
                        
                        debug!("Optimize predicate on every partition");
                        // Should Optimize predicate on every partition.
                        // let num_partition = physical_input.output_partitioning().partition_count();
                        let partition_predicate = runtime_expr;
                        debug!("Finish creating boolean physical plan. Is_score: {}", boolean.is_score);
                        Ok(Arc::new(BooleanExec::try_new(partition_predicate, Arc::new(posting), None, boolean.is_score, Arc::new(boolean.projected_terms.clone()))?))
                    } else {
                        debug!("Create boolean predicate");
                        let mmep_table = physical_input.as_any().downcast_ref::<MmapExec>().unwrap().to_owned();
                        let runtime_expr: Arc<dyn PhysicalExpr> = if let Some(ref predicate) = boolean.predicate {
                            let schema = boolean.input.schema();
                            let inputs: Vec<&str> = schema.fields().iter().filter(|f| !f.name().starts_with("__NULL__")).map(|f| f.name().as_str()).collect();
                            let (term2idx,
                                term2sel): (HashMap<&str, usize>, HashMap<&str, f64>) = inputs
                                .into_iter()
                                .enumerate()
                                .filter(|(_, s)| *s != "__id__")
                                .map(|(i, s)| {
                                        ((s, i), (s, 0.))
                                })
                                .unzip();
                            let builder = PhysicalPredicateBuilder::new(predicate, term2idx, term2sel);
                            let physical_predicate = builder.build()?;
                            Arc::new(BooleanEvalExpr::new(physical_predicate))
                            } else {
                                unreachable!()
                            };
                            
                            debug!("Optimize predicate on every partition");
                            // Should Optimize predicate on every partition.
                            let partition_predicate = runtime_expr;
                            debug!("Finish creating boolean physical plan. Is_score: {}", boolean.is_score);
                            Ok(Arc::new(BooleanExec::try_new(partition_predicate, Arc::new(mmep_table), None, boolean.is_score, Arc::new(boolean.projected_terms.clone()))?))
                    }
                    
                }
                LogicalPlan::Analyze(a) => {
                    let input = self.create_boolean_plan(&a.input, session_state).await?;
                    let schema = SchemaRef::new((*a.schema).clone().into());
                    Ok(Arc::new(AnalyzeExec::new(a.verbose, input, schema)))
                }
                LogicalPlan::Aggregate(Aggregate {
                    input,
                    aggr_expr,
                    ..
                }) => {
                    // Initially need to perform the aggregate and merge the partitions
                    let input_exec = self.create_boolean_plan(input, session_state).await?;
                    let physical_input_schema = input_exec.schema();
                    let logical_input_schema = input.as_ref().schema();

                    let groups = PhysicalGroupBy::new_single(vec![]);
                    let aggregates = aggr_expr
                        .iter()
                        .map(|e| {
                            create_aggregate_expr(
                                e, 
                                logical_input_schema,
                                &physical_input_schema,
                                session_state.execution_props()
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let  initial_aggr = Arc::new(AggregateExec::try_new(
                        AggregateMode::Partial,
                        groups.clone(),
                        aggregates.clone(),
                        input_exec,
                        physical_input_schema.clone(),
                    )?);

                    Ok(Arc::new(AggregateExec::try_new(
                        AggregateMode::Final,
                        groups.clone(),
                        aggregates,
                        initial_aggr,
                        physical_input_schema.clone(),
                    )?))
                }
                _ => unreachable!("Don't support LogicalPlan {:?} in BooleanPlanner", logical_plan),
            };
            exec_plan
        }.boxed()
    }

    /// Handles capturing the various plans for EXPLAIN queries
    /// 
    /// Returns
    /// Some(plan) if optimized, and None if logical_plan was not an
    /// explain (and thus needs to be optimized as normal)
    async fn handle_explain(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let LogicalPlan::Explain(e) = logical_plan {
            let mut stringified_plans = vec![];

            let config = &session_state.config_options().explain;

            if !config.physical_plan_only {
                stringified_plans = e.stringified_plans.clone();
                if e.logical_optimization_succeeded {
                    stringified_plans.push(e.plan.to_stringified(PlanType::FinalLogicalPlan));
                }
            }

            if !config.logical_plan_only && e.logical_optimization_succeeded {
                match self
                    .create_boolean_plan(e.plan.as_ref(), session_state)
                    .await {
                        Ok(input) => {
                            stringified_plans.push(
                                displayable(input.as_ref())
                                .to_stringified(PlanType::InitialPhysicalPlan),
                            );

                            match self.optimize_internal(
                                input,
                                session_state,
                                |plan, optimizer| {
                                    let optimizer_name = optimizer.name().to_string();
                                    let plan_type = PlanType::OptimizedPhysicalPlan { optimizer_name };
                                    stringified_plans
                                        .push(displayable(plan).to_stringified(plan_type));
                                },
                            ) {
                                Ok(input) => stringified_plans.push(
                                    displayable(input.as_ref())
                                        .to_stringified(PlanType::FinalPhysicalPlan),
                                ),
                                Err(DataFusionError::Context(optimizer_name, e)) => {
                                    let plan_type = PlanType::OptimizedPhysicalPlan { optimizer_name };
                                    stringified_plans
                                        .push(StringifiedPlan::new(plan_type, e.to_string()))
                                }
                                Err(e) => return Err(e),
                            }
                        }
                        Err(e) => stringified_plans
                            .push(StringifiedPlan::new(PlanType::InitialPhysicalPlan, e.to_string())),
                    }
            }

            Ok(Some(Arc::new(ExplainExec::new(
                SchemaRef::new(e.schema.as_ref().to_owned().into()),
                stringified_plans,
                e.verbose,
            ))))
        } else {
            Ok(None)
        }
    }

    /// Optimize a physical plan by applying each physical optimizer,
    /// calling observer(plan, optimizer after each one
    fn optimize_internal<F>(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        session_state: &SessionState,
        mut observer: F,
    ) -> Result<Arc<dyn ExecutionPlan>>
    where
        F: FnMut(&dyn ExecutionPlan, &dyn PhysicalOptimizerRule)
    {
        let optimizers = session_state.physical_optimizers();
        debug!(
            "Input physical plan:\n{}\n",
            displayable(plan.as_ref()).indent()
        );
        trace!("Detailed input physical plan:\n{:?}", plan);

        let mut new_plan = plan;
        for optimizer in optimizers {
            let before_schema = new_plan.schema();
            new_plan = optimizer
                .optimize(new_plan, session_state.config_options())
                .map_err(|e| {
                    DataFusionError::Context(optimizer.name().to_string(), Box::new(e))
                })?;
            if optimizer.schema_check() && new_plan.schema() != before_schema {
                let e = DataFusionError::Internal(format!(
                    "PhysicalOptimizer rule '{}' failed, due to generate a different schema, original schema: {:?}, new schema: {:?}",
                    optimizer.name(),
                    before_schema,
                    new_plan.schema(),
                ));
                return Err(DataFusionError::Context(
                    optimizer.name().to_string(),
                    Box::new(e),
                ));
            }
            trace!(
                "Optimized physical plan by {}:\n{}\n",
                optimizer.name(),
                displayable(new_plan.as_ref()).indent()
            );
            observer(new_plan.as_ref(), optimizer.as_ref())
        }
        debug!(
            "Optimized physical plan:\n{}\n",
            displayable(new_plan.as_ref()).indent()
        );
        trace!("Detailed optimized physical plan:\n{:?}", new_plan);
        Ok(new_plan)
    }
}

fn create_physical_expr(
    e: &Expr,
    input_dfschema: &DFSchema,
    input_schema: &Schema,
    execution_props: &ExecutionProps,
) -> Result<Arc<dyn PhysicalExpr>> {
    // if input_schema.fields.len() != input_dfschema.fields().len() {
    //     return Err(DataFusionError::Internal(format!(
    //         "create_physical_expr expected same number of fields, got \
    //         Arrow schema with {} and DataFusion schema with {}",
    //         input_schema.fields.len(),
    //         input_dfschema.fields().len(),
    //     )));
    // }
    match e {
        Expr::Alias(expr, ..) => Ok(create_physical_expr(
            expr,
            input_dfschema,
            input_schema,
            execution_props,
        )?),
        Expr::Column(c) => {
            if &c.name == "mask" {
                Ok(Arc::new(Column::new(&c.name, 0)))
            } else {
                let idx = input_dfschema.index_of_column_by_name(None, &c.name)?;
                Ok(Arc::new(Column::new(&c.name, idx)))
            }
        }
        Expr::Literal(value) => Ok(Arc::new(Literal::new(value.clone()))),
        Expr::BinaryExpr(BinaryExpr { left, op, right}) => {
            let lhs = create_physical_expr(
                left,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            let rhs = create_physical_expr(
                right,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            binary(lhs, *op, rhs, input_schema)
        }
        Expr::BooleanQuery(BooleanQuery { left, op, right }) => {
            let lhs = create_physical_expr(
                left,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            let rhs = create_physical_expr(
                right,
                input_dfschema,
                input_schema,
                execution_props,
            )?;
            let op = match op {
                Operator::BitwiseAnd => Operator::And,
                Operator::BitwiseOr => Operator::Or,
                _ => unreachable!(),
            };
            binary(lhs, op, rhs, input_schema)
        }
        Expr::Not(expr) => expressions::not(create_physical_expr(
            expr,
            input_dfschema,
            input_schema,
            execution_props,
        )?),
        _ => unreachable!("Don't support expr {} in BooleanPlanner", e),
    }
}

fn physical_name(e: &Expr) -> Result<String> {
    create_physical_name(e, true)
}

fn create_physical_name(e: &Expr, is_first_expr: bool) -> Result<String> {
    match e {
        Expr::Column(c) => {
            if is_first_expr {
                Ok(c.name.clone())
            } else {
                Ok(c.flat_name())
            }
        }
        Expr::Alias(_, name) => Ok(name.clone()),
        Expr::Literal(value) => Ok(format!("{value:?}")),
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            let left = create_physical_name(left, false)?;
            let right = create_physical_name(right, false)?;
            Ok(format!("{left} {op} {right}"))
        }
        Expr::BooleanQuery(BooleanQuery { left, op, right }) => {
            let left = create_physical_name(left, false)?;
            let right = create_physical_name(right, false)?;
            Ok(format!("{left} {op} {right}"))
        }
        Expr::Not(expr) => {
            let expr = create_physical_name(expr, false)?;
            Ok(format!("NOT {expr}"))
        }
        Expr::AggregateUDF { fun, args, filter } => {
            if filter.is_some() {
                return Err(DataFusionError::Execution(
                    "aggregate expression with filter is not supported".to_string(),
                ));
            }
            let mut names = Vec::with_capacity(args.len());
            for e in args {
                names.push(create_physical_name(e, false)?);
            }
            Ok(format!("{}({})", fun.name, names.join(",")))
        }
        e => Err(DataFusionError::Internal(
            format!("Create physical name does not support {}", e)
        )),
    }
}

fn tuple_err<T, R>(value: (Result<T>, Result<R>)) -> Result<(T, R)> {
    match value {
        (Ok(e), Ok(e1)) => Ok((e, e1)),
        (Err(e), Ok(_)) => Err(e),
        (Ok(_), Err(e1)) => Err(e1),
        (Err(e), Err(_)) => Err(e),
    }
}

struct PhysicalPredicateBuilder<'a> {
    root: &'a Predicate,
    term2idx: HashMap<&'a str, usize>,
    term2sel: HashMap<&'a str, f64>,
}

impl<'a> PhysicalPredicateBuilder<'a> {
    fn new(root: &'a Predicate, term2idx: HashMap<&'a str, usize>, term2sel: HashMap<&'a str, f64>) -> Self {
        Self {
            root,
            term2idx,
            term2sel,
        }
    }

    fn build(self) -> Result<Option<PhysicalPredicate>> {
        Ok(self.convert_physical(self.root)?.map(|v| v.sub_predicate))
    }

    fn convert_physical(&self, predicate: &Predicate) -> Result<Option<SubPredicate>> {
        match predicate {
            Predicate::And { args } => {
                let mut nodes = vec![];
                let mut sel = 1.0;
                let mut node_num = 0;
                let mut leaf_num = 0;
                for arg in args {
                    if let Some(sub_predicate) = self.convert_physical(&arg.0)? {
                        node_num += sub_predicate.node_num();
                        leaf_num += sub_predicate.leaf_num();
                        sel *= sub_predicate.sel();
                        nodes.push(sub_predicate);
                    } else {
                        return Ok(None);
                    }
                }
                let rank = (sel - 1.) / leaf_num as f64;
                nodes.sort_by(|l, r| l.rank().partial_cmp(&r.rank()).unwrap());
                let mut cs: f64 = 1.;
                for node in &mut nodes {
                    node.cs = 1. - (1. - cs).powi(8);
                    cs *= node.sel();
                }
                let physical_predicate = PhysicalPredicate::And { args: nodes };
                Ok(Some(SubPredicate::new(
                    physical_predicate,
                    node_num,
                    leaf_num,
                    sel,
                    rank,
                    1.,
                )))
            }
            Predicate::Or { args } => {
                let mut nodes = vec![];
                let mut sel = 0.;
                let mut node_num = 0;
                let mut leaf_num = 0;
                for arg in args {
                    if let Some(sub_predicate) = self.convert_physical(&arg.0)? {
                        node_num += sub_predicate.node_num();
                        leaf_num += sub_predicate.leaf_num();
                        sel += sub_predicate.sel() * (1. - sel);
                        nodes.push(sub_predicate);
                    }
                }
                let physical_predicate = PhysicalPredicate::Or { args: nodes };
                let rank = (sel - 1.) / leaf_num as f64;
                Ok(Some(SubPredicate::new(
                    physical_predicate,
                    node_num + 1,
                    leaf_num,
                    sel,
                    rank,
                    1.,
                )))
            }
            Predicate::Other { expr } => {
                let expr_name = expr.display_name()?;
                let sel = self.term2sel.get(expr_name.as_str()).cloned().unwrap_or(0.);
                if let Some(index) = self.term2idx.get(expr_name.as_str()).cloned() {
                    let predicate = PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new(expr_name.as_str(), index)) };
                    Ok(Some(
                        SubPredicate::new(predicate, 1, 1, sel, sel - 1., 1.)
                    ))
                } else {
                    Ok(None)
                }
            }
        }
    }
}

/// Create an aggregate expression from a logical expression or an alias
pub fn create_aggregate_expr(
    e: &Expr,
    logical_input_schema: &DFSchema,
    physical_input_schema: &Schema,
    execution_props: &ExecutionProps,
) -> Result<Arc<dyn AggregateExpr>> {
    // unpack (nested) aliased logical expressions, e.g. "sum(col) as total"
    let (name, e) = match e {
        Expr::Alias(sub_expr, alias) => (alias.clone(), sub_expr.as_ref()),
        _ => (physical_name(e)?, e),
    };

    create_aggregate_expr_with_name(
        e,
        name,
        logical_input_schema,
        physical_input_schema,
        execution_props,
    )
}

/// Create an aggregate expression with a name from a logical expression
pub fn create_aggregate_expr_with_name(
    e: &Expr,
    name: impl Into<String>,
    logical_input_schema: &DFSchema,
    physical_input_schema: &Schema,
    execution_props: &ExecutionProps,
) -> Result<Arc<dyn AggregateExpr>> {
    match e {
        Expr::AggregateFunction(AggregateFunction {
            fun,
            distinct,
            args,
            ..
        }) => {
            let args = args
                .iter()
                .map(|e| {
                    create_physical_expr(
                        e,
                        logical_input_schema,
                        physical_input_schema,
                        execution_props,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            aggregates::create_aggregate_expr(
                fun,
                *distinct,
                &args,
                physical_input_schema,
                name,
            )
        }
        Expr::AggregateUDF { fun, args, .. } => {
            let args = args
                .iter()
                .map(|e| {
                    create_physical_expr(
                        e,
                        logical_input_schema,
                        physical_input_schema,
                        execution_props,
                    )
                })
                .collect::<Result<Vec<_>>>()?;

            udaf::create_aggregate_expr(fun, &args, physical_input_schema, name)
        }
        other => Err(DataFusionError::Internal(format!(
            "Invalid aggregate expression '{other:?}'"
        ))),
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use datafusion::prelude::{boolean_or, boolean_and, col};

    use crate::optimizer::logical_optimizer::{predicate, rewrite_predicate};

    use super::PhysicalPredicateBuilder;

    #[test]
    fn test_physical_predicate_builder() {
        let boolean_or = boolean_or(col("3"), col("2"));
        let boolean_expr = boolean_and(col("1"), boolean_or);
        let boolean_expr = boolean_and(col("4"), boolean_expr);
        let predicate = predicate(&boolean_expr).unwrap();
        let predicate = rewrite_predicate(predicate.0).0;
        println!("{:?}", predicate);
        
        let term1 = format!("1");
        let term2 = format!("2");
        let term3 = format!("3");
        let term4 = format!("4");
        let term2idx = HashMap::from([(term1.as_str(), 0), (term2.as_str(), 1), (term3.as_str(), 2), (term4.as_str(), 3)]);
        let term2sel = HashMap::from([(term1.as_str(), 0.1), (term2.as_str(), 0.2), (term3.as_str(), 0.3), (term4.as_str(), 0.4)]);
        let builder = PhysicalPredicateBuilder::new(&predicate, term2idx, term2sel);

        let physical_predicate = builder.build().unwrap();
        println!("{:?}", physical_predicate);
    }
}
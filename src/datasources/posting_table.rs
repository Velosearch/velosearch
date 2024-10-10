use std::{any::Any, ops::Range, sync::{atomic::{AtomicUsize, Ordering}, Arc, RwLock}, task::Poll, borrow::BorrowMut};

use async_trait::async_trait;
use datafusion::{
    arrow::{datatypes::{SchemaRef, Schema, Field, DataType}, record_batch::RecordBatch, array::UInt64Array}, 
    datasource::TableProvider, 
    logical_expr::TableType, execution::context::SessionState, prelude::Expr, error::{Result, DataFusionError}, 
    physical_plan::{ExecutionPlan, Partitioning, DisplayFormatType, project_schema, RecordBatchStream, metrics::{ExecutionPlanMetricsSet, MetricsSet}}, common::TermMeta};
use futures::Stream;
use adaptive_hybrid_trie::TermIdx;
use roaring::RoaringBitmap;
use serde::{Serialize, ser::SerializeStruct};
use tracing::{debug, info};

use crate::{batch::{BatchRange, PostingBatch, PostingBatchBuilder}, physical_expr::BooleanEvalExpr};

use super::ExecutorWithMetadata;

/// The update/delete operations add entries in the `UpdateQueue`.
/// Update operations add entry in queue and then add new docs.
/// Delete operations directly add entry in this queue.
struct UpdateQueue {
    builder: RwLock<Option<PostingBatchBuilder>>,
}

impl UpdateQueue {
    fn new() -> Self {
        Self {
            builder: RwLock::new(Some(PostingBatchBuilder::new())),
        }
    }

    fn add_doc(&self, doc: Vec<String>, doc_id: usize) -> Result<()> {
        self.add_docs(vec![doc], vec![doc_id]);
        Ok(())
    }

    fn add_docs(&self, docs: Vec<Vec<String>>, ids: Vec<usize>) {
        let mut guard = self.builder
            .write()
            .unwrap();
        let builder = match guard.as_mut() {
            Some(v) => v,
            None => guard.insert(PostingBatchBuilder::new()),
        };
        docs.into_iter().zip(ids.into_iter())
        .for_each(|(doc, doc_id)| {
            builder
            .add_docs(doc, doc_id)
            .unwrap()
        });
    }

    fn clear(&self) {
        if let Some(v) =  self.builder.write().unwrap().as_mut()  {
            v.clear();
        }
    }

    /// Flush the updateQueue into a PostingBatch.
    /// The deleteQueue should flush in the PostingTable
    fn flush(&self) -> Result<Option<PostingBatch>> {
        let builder = self.builder.write().unwrap().take();
        // Clear the updateQueue
        self.clear();
        let posting_batch = builder.map(PostingBatchBuilder::build)
            .transpose()
            .unwrap();
        Ok(posting_batch)
    }
}

/// The implementation of vectorized table provider of Cocoa.
pub struct PostingTable {
    schema: SchemaRef,
    term_idx: Arc<TermIdx<TermMeta>>,
    postings: RwLock<Vec<Arc<PostingBatch>>>,
    pub partitions_num: AtomicUsize,
    update_queue: Arc<UpdateQueue>,
}

impl PostingTable {
    pub fn new(
        schema: SchemaRef,
        term_idx: Arc<TermIdx<TermMeta>>,
        batches: Vec<Arc<PostingBatch>>,
        _range: &BatchRange,
        partitions_num: usize,
    ) -> Self {
        // construct field map to index the position of the fields in schema
        Self {
            schema,
            term_idx,
            postings: RwLock::new(batches),
            partitions_num: AtomicUsize::new(partitions_num),
            update_queue: Arc::new(UpdateQueue::new()),
        }
    }

    /// Add document into update_queue
    pub fn add_document(&self, doc: Vec<String>, doc_id: usize) {
        self.update_queue.add_doc(doc, doc_id).unwrap();
    }

    /// Commit the modifications in update queue.
    /// 1. flush the update queue.
    /// 2. merge the segments according to merge policy.
    pub fn commit(&self) {
        self.flush();
    }

    /// Force to flush the in-memory updateQueue into compact PostingBatch
    /// 
    /// Flush Policy: 
    /// A segment is not searchable after a flush has completed and
    /// is only searchable after a commit.
    pub fn flush(&self) {
        let flush_batch = self.update_queue.flush().unwrap();
        if let Some(batch) = flush_batch {
            self.postings
            .write()
            .as_mut()
            .unwrap()
            .push(Arc::new(batch));
        }
    }

    /// Force to merge segments indicated by the `segments` index vector.
    pub fn merge(&self, segments: Vec<usize>) {
        todo!()
    }

    #[inline]
    pub fn stat_of(&self, term_name: &str, _partition: usize) -> Option<TermMeta> {
        #[cfg(feature="trie_idx")]
        let meta = self.term_idx.get(term_name);
        #[cfg(feature="hash_idx")]
        let meta = self.term_idx.get(term_name).cloned();
        meta
    }

    pub fn stats_of(&self, term_names: &[&str], partition: usize) -> Vec<Option<TermMeta>> {
        term_names
            .into_iter()
            .map(|v| self.stat_of(v, partition))
            .collect()
    }

    pub fn memory_consumption(&self) -> usize {
        let mut postings: usize = 0;
        let mut offsets: usize = 0;
        let mut all: usize = 0;
        self.postings.read().unwrap().iter()
            .for_each(|v| {
                let size = v.memory_consumption();
                all += size.0;
                postings += size.1;
                offsets += size.2;
            });
        info!("posting size: {:}", postings);
        info!("offsets size: {:}", offsets);
        all
    }
}

impl Serialize for PostingTable {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
        where
            S: serde::Serializer {
        let mut state = serializer.serialize_struct("PostingTable", 4)?;
        state.serialize_field("schema", &self.schema)?;
        
        state.end()
    }
}

#[async_trait]
impl TableProvider for PostingTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        // @TODO how to search field index efficiently
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
       TableType::Base 
    }

    async fn scan(
        &self,
        _state: &SessionState,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        debug!("PostingTable scan");
        let posting_guard = self.postings.read().unwrap();
        let postings: Vec<Arc<PostingBatch>> = match projection {
            Some(v) => v.into_iter()
                .map(|i| posting_guard[*i].clone())
                .collect(),
            None => posting_guard.iter()
                .map(|i| i.clone())
                .collect(),
        };
        Ok(Arc::new(PostingExec::try_new(
            postings, 
            self.term_idx.clone(),
            self.schema(), 
            projection.cloned(),
            None,
            vec![],
            self.partitions_num.load(Ordering::Relaxed),
        )?))
    }
}

#[derive(Clone)]
pub struct PostingExec {
    pub partitions: Vec<Arc<PostingBatch>>,
    pub schema: SchemaRef,
    pub term_idx: Arc<TermIdx<TermMeta>>,
    pub projected_schema: SchemaRef,
    pub projection: Option<Vec<usize>>,
    pub partition_min_range: Option<Arc<RoaringBitmap>>,
    pub is_score: bool,
    pub projected_term_meta: Vec<Option<TermMeta>>,
    pub predicate: Option<BooleanEvalExpr>,
    pub partitions_num: usize,
    metric: ExecutionPlanMetricsSet,
    pub distri: Vec<Option<Arc<RoaringBitmap>>>,
    pub idx: Vec<Option<u32>>,
}

impl std::fmt::Debug for PostingExec {
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
       write!(f, "partitions: [...]")?;
       write!(f, "schema: {:?}", self.projected_schema)?;
       write!(f, "projection: {:?}", self.projection)?;
       write!(f, "is_score: {:}", self.is_score)
   } 
}

impl ExecutionPlan for PostingExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.projected_schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        // this is a leaf node and has no children
        vec![]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        Partitioning::UnknownPartitioning(self.partitions_num)
    }

    fn output_ordering(&self) -> Option<&[datafusion::physical_expr::PhysicalSortExpr]> {
        None
    }

    fn with_new_children(
            self: Arc<Self>,
            _: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::Internal(format!(
            "Children cannot be replaced in {self:?}"
        )))
    }

    fn execute(
            &self,
            partition: usize,
            context: Arc<datafusion::execution::context::TaskContext>,
        ) -> Result<datafusion::physical_plan::SendableRecordBatchStream> {
        // let default_step_len = *STEP_LEN.lock();
        // const default_step_len: usize = 1024;

        let task_len = self.partition_min_range.as_ref().unwrap().len() as usize;
        // let batch_len = if self.partitions_num * default_step_len > task_len {
            
        // } else {
        //     task_len / self.partitions_num
        // };
        let batch_len = task_len / self.partitions_num;
        debug!("Start PostingExec::execute for partition {} of context session_id {} and task_id {:?}", partition, context.session_id(), context.task_id());
        
        Ok(Box::pin(PostingStream::try_new(
            self.partitions[0].clone(),
            self.projected_schema.clone(),
            self.partition_min_range.as_ref().unwrap().clone(),
            self.distri.clone(),
            self.idx.clone(),
            self.predicate.clone(),
            self.is_score,
            (batch_len * partition)..(batch_len * partition + batch_len),
        )?))
    }

    fn fmt_as(
        &self, 
        t: datafusion::physical_plan::DisplayFormatType, 
        f: &mut std::fmt::Formatter
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f,
                    "PostingExec: partition_size={:?}, is_score: {:}, predicate: {:?}",
                    self.partitions.len(),
                    self.is_score,
                    self.predicate.as_ref().map(|v| v.predicate.as_ref().map(|v| unsafe { &(*v.get()) })),
                )
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metric.clone_inner())
    }

    fn statistics(&self) -> datafusion::physical_plan::Statistics {
        todo!()
    }
}

impl PostingExec {
    /// Create a new execution plan for reading in-memory record batches
    /// The provided `schema` shuold not have the projection applied.
    pub fn try_new(
        partitions: Vec<Arc<PostingBatch>>,
        term_idx: Arc<TermIdx<TermMeta>>,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
        partition_min_range: Option<Arc<RoaringBitmap>>,
        projected_term_meta: Vec<Option<TermMeta>>,
        partitions_num: usize,
    ) -> Result<Self> {
        let projected_schema = project_schema(&schema, projection.as_ref())?;
        // let (distris, indices ) = projected_term_meta.iter()
        //     .map(| v| match v {
        //         Some(v) => (Some(v.valid_bitmap[0].clone()), v.index[0].clone() ),
        //         None => (None, None),
        //     })
        //     .unzip();
        Ok(Self {
            partitions: partitions,
            term_idx,
            schema,
            projected_schema,
            projection,
            partition_min_range,
            is_score: false,
            projected_term_meta,
            predicate: None,
            partitions_num,
            metric: ExecutionPlanMetricsSet::new(),
            distri: vec![],
            idx: vec![],
        })
    }
}

impl ExecutorWithMetadata for PostingExec {
    /// Get TermMeta From &[&str]
    fn term_metas_of(&self, terms: &[&str]) -> Vec<Option<TermMeta>> {
        let term_idx = self.term_idx.clone();
        terms
            .into_iter()
            .map(|&t| {
                #[cfg(feature="trie_idx")]
                let meta = term_idx.get(t);
                #[cfg(feature="hash_idx")]
                let meta = term_idx.get(t).cloned();
                meta
            })
            .collect()
    }

    /// Get TermMeta From &str
    fn term_meta_of(&self, term: &str) -> Option<TermMeta> {
        #[cfg(not(feature="hash_idx"))]
        let meta = self.term_idx.get(term);
        #[cfg(feature="hash_idx")]
        let meta = self.term_idx.get(term).cloned();
        meta
    }
    
    fn set_term_meta(&mut self, term_meta: Vec<Option<TermMeta>>) {
        self.projected_term_meta = term_meta;
    }
}

pub struct PostingStream {
    /// Vector of recorcd batches
    posting_lists:  Arc<PostingBatch>,
    /// Schema representing the data
    schema: SchemaRef,
    /// is_score
    is_score: bool,
    /// min_range
    min_range: Vec<u32>,
    /// distris
    distris: Vec<Option<Arc<RoaringBitmap>>>,
    /// indecis
    indices: Vec<Option<u32>>,
    /// index the bucket
    index: usize,
    /// 
    predicate: Option<BooleanEvalExpr>,
    /// task range
    task_range: Range<usize>,
    /// empty schema
    _empty_batch: RecordBatch,
    /// step length
    step_length: usize,
}

impl PostingStream {
    /// Create an iterator for a vector of record batches
    pub fn try_new(
        data: Arc<PostingBatch>,
        schema: SchemaRef,
        min_range: Arc<RoaringBitmap>,
        distris: Vec<Option<Arc<RoaringBitmap>>>,
        indices: Vec<Option<u32>>,
        predicate: Option<BooleanEvalExpr>,
        is_score: bool,
        task_range: Range<usize>,
    ) -> Result<Self> {
        debug!("Try new a PostingStream, min range len: {:}", min_range.len());
        let min_range = min_range.as_ref()
            .iter()
            .map(|v| v)
            .collect();
        Ok(Self {
            posting_lists: data,
            schema,
            min_range,
            distris,
            is_score,
            indices,
            predicate,
            index: task_range.start,
            task_range,
            _empty_batch: RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new("mask", DataType::UInt64, false)]))),
            step_length: 1024,
        })
    }
}

impl Stream for PostingStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        let end = self.task_range.end.min(self.min_range.len());
        if self.index >= end {
            return Poll::Ready(None);
        }
        debug!("index: {:}, task_range: {:?}", self.index, self.task_range);
        let batch = if self.is_score {
            let mut cnt = 0;
            while self.index < end {
                let res = if self.index + self.step_length >= end {
                    self.posting_lists.roaring_predicate_with_score(&self.distris, &self.schema, &self.indices,  &self.min_range[self.index..end], &self.predicate.as_ref().unwrap())
                } else {
                    self.posting_lists.roaring_predicate_with_score(&self.distris, &self.schema, &self.indices,  &self.min_range[self.index..(self.index + self.step_length)], &self.predicate.as_ref().unwrap())
                };
                self.index += self.step_length;
                cnt += res.unwrap();
            }
            cnt as usize
        } else {
            let mut cnt = 0;
            while self.index < end {
                let res = if self.index + self.step_length >= end {
                    self.posting_lists.roaring_predicate(&self.distris, &self.indices, &self.min_range[self.index..end], &self.predicate.as_ref().unwrap())
                } else {
                    self.posting_lists.roaring_predicate(&self.distris, &self.indices, &self.min_range[self.index..(self.index+self.step_length)], &self.predicate.as_ref().unwrap())
                };
                self.index += self.step_length;
                cnt += res.unwrap();
            }
            cnt
        };
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("mask", DataType::UInt64, false)])),
            vec![
                Arc::new(UInt64Array::from(vec![batch as u64])),
            ]
        )?;
        // return Poll::Ready(Some(Ok(self.empty_batch.clone())))
        return Poll::Ready(Some(Ok(batch)));
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

impl RecordBatchStream for PostingStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

pub fn make_posting_schema(fields: Vec<&str>) -> Schema {
    Schema::new(
        [fields.into_iter()
        .map(|f| Field::new(f, DataType::Boolean, false))
        .collect(), vec![Field::new("__id__", DataType::UInt32, false)]].concat()
    )
}


#[cfg(test)]
mod tests {}
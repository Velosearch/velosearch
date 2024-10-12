use std::{any::Any, ops::Range, sync::{atomic::{AtomicUsize, Ordering}, Arc}, task::Poll};
use itertools::Itertools;
use async_trait::async_trait;
use datafusion::{
    arrow::{datatypes::{SchemaRef, Schema, Field, DataType}, record_batch::RecordBatch, array::UInt64Array}, 
    datasource::TableProvider, 
    logical_expr::TableType, execution::context::SessionState, prelude::Expr, error::{Result, DataFusionError}, 
    physical_plan::{ExecutionPlan, Partitioning, DisplayFormatType, RecordBatchStream, metrics::{ExecutionPlanMetricsSet, MetricsSet}, PhysicalExpr}, common::TermMeta};
use futures::Stream;
use roaring::RoaringBitmap;
use tokio::sync::RwLock;
use tracing::{debug, info};
use crate::{batch::{PostingBatch, PostingBatchBuilder, merge_segments}, physical_expr::BooleanEvalExpr};

const DEFAULT_LEVEL_LOG_SIZE: f64 = 0.75;
const DEFAULT_MIN_LAYER_SIZE: u32 = 10_000;
const DEFAULT_MIN_NUM_SEGMENTS_IN_MERGE: usize = 8;
const DEFAULT_MAX_DOCS_BEFORE_MERGE: usize = 10_000_000;
// The default value of 1 means that deletes are not taken in account when
// identifying merge candidates. This is not a very sensible default: it was
// set like that for backward compatibility and might change in the near future.
const DEFAULT_DEL_DOCS_RATIO_BEFORE_MERGE: f32 = 1.0f32;

/// `LogMergePolicy` tries to merge segments that have a similar number of dodcuments
#[derive(Debug, Clone)]
pub struct LogMergePolicy {
    min_num_segments: usize,
    /// Set the maximum number docs in a segment for it to be considered for
    /// merging. A segment can still reach more than max_docs, by merging many
    /// smaller ones
    max_docs_before_merge: usize,
    /// Set the minimum segment size under which all segment belong
    /// to the same level.
    min_layer_size: u32,
    /// Segments are grouped in levels according to their sizes.
    /// These levels are defined as intervals of exponentially growing sizes.
    /// level_log_size define the factor by which one should multiply the limit
    /// to reach a level, in order to get the limit to reach the following
    /// level.
    level_log_size: f64,
    /// Set the ratio of deleted documents in a segment to tolerate.
    /// If it is exceeded by any segment at a log level, a merge
    /// will be triggered for that level.
    /// If there is a single segment at a level, we effectively end up expunging
    /// deleted documents from it.
    del_docs_ratio_before_merge: f32,
}

impl LogMergePolicy {
    fn clip_min_size(&self, size: u32) -> u32 {
        std::cmp::max(self.min_layer_size, size)
    }

    /// Given the list of segment metas, returns the list of merge candidates.
    ///
    /// This call happens on the segment updater thread, and will block
    /// other segment updates, so all implementations should happen rapidly.
    pub fn compute_merge_candidates(&self, segments: &[SegmentMeta]) -> Vec<Vec<usize>> {
        let size_sorted_segments = segments
            .iter()
            .filter(|seg| seg.num_docs() <= (self.max_docs_before_merge as u32))
            .sorted_by_key(|seg| std::cmp::Reverse(seg.max_doc()))
            .collect::<Vec<_>>();

        if size_sorted_segments.is_empty() {
            return vec![];
        }

        let mut current_max_log_size = f64::MAX;
        let mut levels = vec![];
        for (_, merge_group) in &size_sorted_segments.into_iter().group_by(|segment| {
            let segment_log_size = f64::from(self.clip_min_size(segment.num_docs())).log2();
            if segment_log_size < (current_max_log_size - self.level_log_size) {
                // update current_max_log_size to create a new group
                current_max_log_size = segment_log_size;
            }
            // return current_max_log_size to be grouped to the current group
            current_max_log_size
        }) {
            levels.push(merge_group.collect::<Vec<_>>())
        }

        levels
            .into_iter()
            .filter(|level| {level.len() >= self.min_num_segments})
            .map(|segments| segments.into_iter().map(SegmentMeta::id).collect())
            .collect()
    }
}

impl Default for LogMergePolicy {
    fn default() -> Self {
        LogMergePolicy {
            min_num_segments: DEFAULT_MIN_NUM_SEGMENTS_IN_MERGE,
            max_docs_before_merge: DEFAULT_MAX_DOCS_BEFORE_MERGE,
            min_layer_size: DEFAULT_MIN_LAYER_SIZE,
            level_log_size: DEFAULT_LEVEL_LOG_SIZE,
            del_docs_ratio_before_merge: DEFAULT_DEL_DOCS_RATIO_BEFORE_MERGE,
        }
    }
}

pub struct SegmentMeta {
    id: usize,
    num_docs: u32,
    max_doc: usize,
}

impl SegmentMeta {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    pub fn max_doc(&self) -> usize {
        self.max_doc
    }
}

/// The update/delete operations add entries in the `UpdateQueue`.
/// Update operations add entry in queue and then add new docs.
/// Delete operations directly add entry in this queue.
struct UpdateQueue {
    builder: RwLock<PostingBatchBuilder>,
}

impl UpdateQueue {
    fn new(base: u32) -> Self {
        Self {
            builder: RwLock::new(PostingBatchBuilder::new(base)),
        }
    }

    async fn add_doc(&self, doc: Vec<String>, doc_id: usize) -> Result<()> {
        self.add_docs(vec![doc], doc_id).await;
        Ok(())
    }

    async fn add_docs(&self, docs: Vec<Vec<String>>, doc_id: usize) {
        let mut doc_id = doc_id;
        let mut guard = self.builder
            .write()
            .await;
        docs.into_iter()
        .for_each(|doc| {
            guard
            .add_docs(doc, doc_id)
            .unwrap();
            doc_id += 1;
        });
    }

    async fn clear(&self) {
        self.builder.write().await.clear();
    }

    /// Flush the updateQueue into a PostingBatch.
    /// The deleteQueue should flush in the PostingTable
    async fn flush(&self) -> Result<Option<PostingBatch>> {
        let posting_batch = self.builder.write().await.build().ok();
        // Clear the updateQueue
        self.clear().await;
        Ok(posting_batch)
    }
}

/// The implementation of vectorized table provider of Cocoa.
pub struct PostingTable {
    schema: SchemaRef,
    postings: RwLock<Vec<Arc<PostingBatch>>>,
    pub partitions_num: AtomicUsize,
    update_queue: Arc<UpdateQueue>,
    doc_id: AtomicUsize,
}

impl PostingTable {
    pub fn new(
        schema: SchemaRef,
        batches: Vec<Arc<PostingBatch>>,
        partitions_num: usize,
    ) -> Self {
        // construct field map to index the position of the fields in schema
        let doc_id = batches.last().map_or(0, |v| v.range().end());
        Self {
            schema,
            postings: RwLock::new(batches),
            partitions_num: AtomicUsize::new(partitions_num),
            update_queue: Arc::new(UpdateQueue::new(doc_id)),
            doc_id: AtomicUsize::new(doc_id as usize),
        }
    }

    /// Pick merges that are now required.
    /// Return the indexes of picked segments.
    pub async fn find_merges(&self) -> Vec<usize> {
        todo!()
    }

    pub async fn schedule_merge(&self) {
        // just for test
        let segment = merge_segments(self.postings.write().await.clone()).unwrap();
        let mut guard = self.postings.write().await;
        guard.clear();
        guard.push(segment);
    }

    /// Add document into update_queue
    pub async fn add_document(&self, doc: Vec<String>) {
        self.update_queue.add_doc(doc, self.doc_id.load(Ordering::Relaxed)).await.unwrap();
        self.doc_id.fetch_add(1, Ordering::AcqRel);
    }

    /// Add batched documents into update_queue
    pub async fn add_documents(&self, docs: Vec<Vec<String>>) {
        self.update_queue.add_docs(docs, self.doc_id.fetch_add(512, Ordering::Relaxed)).await;
    }

    /// Commit the modifications in update queue.
    /// 1. flush the update queue.
    /// 2. merge the segments according to merge policy.
    pub async fn commit(&self) {
        self.flush().await;
    }

    /// Force to flush the in-memory updateQueue into compact PostingBatch
    /// 
    /// Flush Policy: 
    /// A segment is not searchable after a flush has completed and
    /// is only searchable after a commit.
    pub async fn flush(&self) {
        let flush_batch = self.update_queue.flush().await.unwrap();
        if let Some(batch) = flush_batch {
            self.postings
            .write()
            .await
            .push(Arc::new(batch));
        }
    }

    /// Force to merge segments indicated by the `segments` index vector.
    pub fn merge(&self, _segments: Vec<usize>) {
        todo!()
    }

    #[inline]
    pub async fn stat_of(&self, term_name: &str, partition: usize) -> Option<TermMeta> {
        #[cfg(feature="trie_idx")]
        let meta = self.term_idx.get(term_name);
        #[cfg(feature="hash_idx")]
        let meta = self.postings.read().await[partition].term_idx.get(term_name).cloned();
        meta
    }

    pub async fn stats_of(&self, term_names: &[&str], partition: usize) -> Vec<Option<TermMeta>> {
        let mut metas = vec![];
        let guard = self.postings.read().await;
        for term in term_names {
            metas.push(guard[partition].term_idx.get(term).cloned())
        }
        return metas;
    }

    pub fn memory_consumption(&self) -> usize {
        // todo: replace with async collect method
        let postings: usize = 0;
        let offsets: usize = 0;
        let all: usize = 0;
        // self.postings.read().unwrap().iter()
        //     .for_each(|v| {
        //         let size = v.memory_consumption();
        //         all += size.0;
        //         postings += size.1;
        //         offsets += size.2;
        //     });
        info!("posting size: {:}", postings);
        info!("offsets size: {:}", offsets);
        all
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
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        debug!("PostingTable scan");
        let posting_guard = self.postings.read().await;
        // posting checkpoint.
        let postings: Vec<Arc<PostingBatch>> = posting_guard.iter()
            .map(|i| i.clone())
            .collect();
        let segment_num = postings.len();
        Ok(Arc::new(PostingExec::try_new(
            postings, 
            segment_num,
            vec![],
        )?))
    }
}

#[derive(Clone)]
pub struct PostingExec {
    pub partitions: Vec<Arc<PostingBatch>>,
    pub projected_schema: SchemaRef,
    pub is_score: bool,
    pub predicate: Option<BooleanEvalExpr>,
    pub partitions_num: usize,
    metric: ExecutionPlanMetricsSet,
    pub distri: Vec<Option<Arc<RoaringBitmap>>>,
    pub idx: Vec<Option<u32>>,
    pub projected_terms: Arc<Vec<String>>,
}

impl std::fmt::Debug for PostingExec {
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
       write!(f, "partitions: [...]")?;
       write!(f, "schema: {:?}", self.projected_terms)?;
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
        Arc::new(Schema::empty())
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        // this is a leaf node and has no children
        vec![]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> datafusion::physical_plan::Partitioning {
        Partitioning::UnknownPartitioning(self.partitions.len())
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
        let postings = self.partitions[partition].clone();
        // Pruning to get the min range
        let term_stats: Vec<Option<&TermMeta>> = postings.term_metas_of(&self.projected_terms);
        let invalid = Arc::new(RoaringBitmap::new());
        let prune_bitmap: Vec<Arc<RoaringBitmap>> = term_stats.iter()
            .map(|t| {
                match t {
                    Some(t) => t.valid_bitmap().clone(),
                    None => invalid.clone(),
                }
            })
            .collect();
        let skip_bitmap = match self.predicate.as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanEvalExpr>() {
            Some(p) => {
                p.eval_bitmap(&prune_bitmap).unwrap()
            }
            None => unreachable!("Must have valid predicate.")
        };

        let (distri, index): (Vec<Option<Arc<RoaringBitmap>>>, Vec<Option<u32>>) = 
            term_stats.into_iter()
            .map(|t| {
                match t {
                    Some(t) => (Some(t.valid_bitmap().clone()), Some(t.index)),
                    None => (None, None),
                }
            })
            // .unzip::<Vec<Option<RoaringBitmap>>, Vec<Option<u32>>>();
            .unzip();
        
        let task_len = skip_bitmap.len() as usize;
        // let batch_len = task_len / self.partitions_num;
        debug!("Start PostingExec::execute for partition {} of context session_id {} and task_id {:?}", partition, context.session_id(), context.task_id());
        
        Ok(Box::pin(PostingStream::try_new(
            postings,
            self.projected_schema.clone(),
            skip_bitmap,
            distri,
            index,
            self.predicate.clone(),
            self.is_score,
            // (batch_len * partition)..(batch_len * partition + batch_len),
            0..task_len
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
        partitions_num: usize,
        projected_terms: Vec<String>,
    ) -> Result<Self> {
        // let projected_schema = project_schema(&schema, projection.as_ref())?;
        // let (distris, indices ) = projected_term_meta.iter()
        //     .map(| v| match v {
        //         Some(v) => (Some(v.valid_bitmap[0].clone()), v.index[0].clone() ),
        //         None => (None, None),
        //     })
        //     .unzip();
        let fields: Vec<Field> = projected_terms.iter()
            .map(|f| Field::new(f.clone(), DataType::Date64, false))
            .collect();
        let projected_schema = Arc::new(Schema::new(fields));
        Ok(Self {
            partitions: partitions,
            projected_schema,
            is_score: false,
            predicate: None,
            partitions_num,
            metric: ExecutionPlanMetricsSet::new(),
            distri: vec![],
            idx: vec![],
            projected_terms: Arc::new(projected_terms),
        })
    }
}

pub struct PostingStream {
    /// Vector of recorcd batches
    posting_lists:  Arc<PostingBatch>,
    /// Schema representing the data
    projected_schema: SchemaRef,
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
        projected_schema: SchemaRef,
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
            projected_schema,
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
                    self.posting_lists.roaring_predicate_with_score(&self.distris, &self.projected_schema, &self.indices,  &self.min_range[self.index..end], &self.predicate.as_ref().unwrap())
                } else {
                    self.posting_lists.roaring_predicate_with_score(&self.distris, &self.projected_schema, &self.indices,  &self.min_range[self.index..(self.index + self.step_length)], &self.predicate.as_ref().unwrap())
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
        self.projected_schema.clone()
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
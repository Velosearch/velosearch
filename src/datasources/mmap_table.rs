use std::{any::Any, arch::x86_64::{_mm512_popcnt_epi64, _mm512_reduce_add_epi64}, fs::{self, File, Permissions}, io::Write, ops::Range, path::Path, slice::from_raw_parts, sync::Arc, task::Poll};

use async_trait::async_trait;
use datafusion::{
    arrow::{array::UInt64Array, datatypes::{DataType, Field, Schema, SchemaRef}, record_batch::RecordBatch}, common::TermMeta, datasource::TableProvider, error::{DataFusionError, Result}, execution::context::SessionState, logical_expr::TableType, physical_plan::{metrics::{ExecutionPlanMetricsSet, MetricsSet}, project_schema, DisplayFormatType, ExecutionPlan, Partitioning, RecordBatchStream}, prelude::Expr};
use adaptive_hybrid_trie::TermIdx;
use futures::Stream;
use memmap::{Mmap, MmapOptions};
use rkyv::{de::deserializers::SharedDeserializeMap, ser::serializers::AllocSerializer, Archive, Deserialize, Serialize};
use roaring::RoaringBitmap;
use tracing::{debug, info};

use crate::{batch::BatchRange, physical_expr::{boolean_eval::{Chunk, TempChunk}, BooleanEvalExpr}};

use super::ExecutorWithMetadata;

#[derive(Archive, Serialize, Deserialize)]
pub struct PostingColumn {
    postings: Vec<u8>,
    offsets: Vec<u32>,
}

impl PostingColumn {
    pub fn new(postings: Vec<u8>, offsets: Vec<u32>) -> Self {
        Self {
            postings,
            offsets,
        }
    }

    pub fn value(&self, index: usize) -> &[u8] {
        let left = self.offsets[index] as usize;
        let right = self.offsets[index + 1] as usize;
        &self.postings[left..right]
    }
}

#[derive(Archive, Serialize, Deserialize)]
pub struct PostingSegment {
    pub posting_lists: Vec<PostingColumn>,
}

pub struct MmapTable {
    schema: SchemaRef,
    term_idx: Arc<TermIdx<TermMeta>>,
    // posting_lists: Arc<PostingSegment>,
    dump_file: Arc<Mmap>,
    pub partitions_num: usize,
}

impl MmapTable {
    pub fn new(
        path: &Path,
        schema: SchemaRef,
        term_idx: Arc<TermIdx<TermMeta>>,
        segment: PostingSegment,
        range: &BatchRange,
        partitions_num: usize,
    ) -> Result<Self> {
        // construct field map to index the position of the fields in schema
        debug!("{:?}", &segment.posting_lists[1].offsets[0..10]);
        let mmap = if fs::metadata(&path).is_ok() {
            unsafe { MmapOptions::new().map(&File::open(&path)?).unwrap() }
        } else {
            let mut file = File::create(path)?;
            // let mut serializer = AllocSerializer::<{ 1024 * 1024 }>::default();
            // let _ = segment.serialize(&mut serializer).unwrap();
            // let data = serializer.into_serializer().into_inner();
            let data = rkyv::to_bytes::<_, 1024>(&segment).unwrap();
            file.write_all(&data)?;
            file.flush()?;
            unsafe { MmapOptions::new().map(&File::options().read(true).open(&path)?).unwrap() }
        };
        let archived = unsafe { rkyv::archived_root::<PostingSegment>(&mmap) };
        // let shared_archived = Arc::new(archived);
        debug!("archive mapped file");
        debug!("{:?}", &archived.posting_lists[1].offsets[0..10]);
        // let posting_lists: PostingSegment = archived.deserialize(&mut SharedDeserializeMap::new()).unwrap();
        debug!("deserialize segment");
        
        Self::with_mmap(
            Arc::new(mmap),
            schema,
            term_idx,
            range,
            // posting_lists: Arc::new(posting_lists),
            partitions_num, 
        )
    }

    pub fn with_mmap(
        mmap: Arc<Mmap>,
        schema: SchemaRef,
        term_idx: Arc<TermIdx<TermMeta>>,
        _range: &BatchRange,
        partitions_num: usize,
    ) -> Result<Self> {
        Ok(Self {
            schema,
            term_idx,
            // posting_lists: Arc::new(posting_lists),
            dump_file: mmap,
            partitions_num: partitions_num,
        })
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
        todo!()
    }
}

#[async_trait]
impl TableProvider for MmapTable {
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
        match projection {
            Some(idx) => {
                // let posting_lists = idx.into_iter()
                //     .map(|i| self.posting_lists.as_ref().posting_lists[*i].clone())
                //     .collect();
                Ok(Arc::new(MmapExec::try_new(
                    // self.posting_lists.clone(), 
                    self.dump_file.clone(),
                    self.term_idx.clone(), 
                    self.schema.clone(), 
                    projection.cloned(), 
                    None, 
                    vec![], 
                    self.partitions_num,
                )?))
            }
            None => {
                Err(DataFusionError::NotImplemented("Don't support scann all posting lists.".to_string()))
            }
        }
        // Ok(Arc::new(MmapExec::try_new(
            
        // )?))
    }
}

#[derive(Clone)]
pub struct MmapExec {
    // posting_lists: Arc<PostingSegment>,
    dump_file: Arc<Mmap>,
    pub schema: SchemaRef,
    pub term_idx: Arc<TermIdx<TermMeta>>,
    pub projected_schema: SchemaRef,
    pub projection: Option<Vec<usize>>,
    pub partition_min_range: Option<Arc<RoaringBitmap>>,
    pub is_score: bool,
    pub projected_term_meta: Vec<Option<TermMeta>>,
    pub predicate: Option<BooleanEvalExpr>,
    metric: ExecutionPlanMetricsSet,
    pub distri: Vec<Option<Arc<RoaringBitmap>>>,
    pub idx: Vec<Option<u32>>,
    pub partitions_num: usize,
}

impl std::fmt::Debug for MmapExec {
   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
       write!(f, "partitions: [...]")?;
       write!(f, "schema: {:?}", self.projected_schema)?;
       write!(f, "projection: {:?}", self.projection)?;
       write!(f, "is_score: {:}", self.is_score)
   } 
}

impl ExecutionPlan for MmapExec {
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
        let batch_len = task_len / self.partitions_num;
        debug!("Start MmapExec::execute for partition {} of context session_id {} and task_id {:?}", partition, context.session_id(), context.task_id());
        
        Ok(Box::pin(PostingStream::try_new(
            // self.posting_lists.clone(),
            self.dump_file.clone(),
            self.projection.as_ref().unwrap().clone(),
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
                    "MmapExec: partition_size={:?}, is_score: {:}, predicate: {:?}",
                    self.partitions_num,
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

    // We recompute the statistics dynamically from the arrow metadata as it is pretty cheap to do so
    // fn statistics(&self) -> datafusion::physical_plan::Statistics {
    //     common::compute_record_batch_statistics(
    //         &self.partitions, &self.schema, self.projection.clone())
    // }
}

impl MmapExec {
    /// Create a new execution plan for reading in-memory record batches
    /// The provided `schema` shuold not have the projection applied.
    pub fn try_new(
        // posting_lists: Arc<PostingSegment>,
        dump_file: Arc<Mmap>,
        term_idx: Arc<TermIdx<TermMeta>>,
        schema: SchemaRef,
        projection: Option<Vec<usize>>,
        partition_min_range: Option<Arc<RoaringBitmap>>,
        projected_term_meta: Vec<Option<TermMeta>>,
        partitions_num: usize,
    ) -> Result<Self> {
        let projected_schema = project_schema(&schema, projection.as_ref())?;
        let (distris, indices ) = projected_term_meta.iter()
            .map(| v| match v {
                Some(v) => (Some(v.valid_bitmap[0].clone()), v.index[0].clone() ),
                None => (None, None),
            })
            .unzip();
        Ok(Self {
            // posting_lists,
            dump_file,
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
            distri: distris,
            idx: indices,
        })
    }
}

impl ExecutorWithMetadata for MmapExec {
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
    // posting_lists:  Arc<PostingSegment>,
    /// Dump file
    dump_file: Arc<Mmap>,
    /// Projection
    projection: Vec<usize>,
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
    empty_batch: RecordBatch,
    /// step length
    step_length: usize,
}

impl PostingStream {
    /// Create an iterator for a vector of record batches
    pub fn try_new(
        // posting_lists: Arc<PostingSegment>,
        dump_file: Arc<Mmap>,
        projection: Vec<usize>,
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
            // posting_lists,
            dump_file,
            projection,
            schema,
            min_range,
            distris,
            is_score,
            indices,
            predicate,
            // index: task_range.start,
            index: 0,
            task_range,
            empty_batch: RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new("mask", DataType::UInt64, false)]))),
            step_length: usize::MAX,
        })
    }
}

impl Stream for PostingStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, _cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        if self.min_range.len() == 0 {
            return Poll::Ready(None);
        }
        if self.index > 0 {
            return Poll::Ready(None);
        }
        // let posting_lists = self.projection.iter()
        //     .map(|p| &self.posting_lists.as_ref().posting_lists[*p])
        //     .collect();
        // let res = evaluate(&self.posting_lists.as_ref().posting_lists, &self.distris, &self.indices, &self.min_range, self.predicate.as_ref().unwrap()).unwrap();
        let res = evaluate(&self.dump_file, &self.distris, &self.indices, &self.min_range, self.predicate.as_ref().unwrap()).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("mask", DataType::UInt64, false)])),
            vec![Arc::new(UInt64Array::from(vec![res as u64]))],
        )?;
        self.index += 1;
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

fn evaluate(
    // posting_lists: &Vec<PostingColumn>,
    dump_file: &Mmap,
    distris: &[Option<Arc<RoaringBitmap>>],
    indices: &[Option<u32>],
    min_range: &[u32],
    predicate: &BooleanEvalExpr,
) -> Result<usize> {
    let posting_lists = unsafe { rkyv::archived_root::<PostingSegment>(&dump_file) };
    let posting_lists = &posting_lists.posting_lists;
    let predicate = {
        match predicate.predicate.as_ref() {
            Some(predicate) => unsafe {
                predicate.get().as_ref().unwrap()
            }
            None => return Ok(0),
        }
    };
    let mut batches: Vec<Option<Vec<Chunk>>> = Vec::with_capacity(distris.len());
    debug!("Start select valid batch.");
    for (term, i) in distris.iter().zip(indices.iter()) {
        let mut posting = Vec::with_capacity(min_range.len());
        let batch = if let Some(i) = i {
            &posting_lists[*i as usize]
        } else {
            break;
        };
        let term = term.as_ref().unwrap();
        let mut rank_iter = term.rank_iter();
        debug!("min_range: {:?}", min_range);
        for &idx in min_range {
            if term.contains(idx) {
                let bound_idx: usize = rank_iter.rank(idx);
                // let batch = batch.value(bound_idx - 1);
                let index = bound_idx - 1;
                let left = batch.offsets[index] as usize;
                let right = batch.offsets[index + 1] as usize;
                let batch = &batch.postings[left..right];
                if batch.len() == 64 {
                    let batch = unsafe {
                        from_raw_parts(batch.as_ptr() as *const u64, 8)
                    };
                    posting.push(Chunk::Bitmap(batch));
                } else {
                    let batch = unsafe {
                        from_raw_parts(batch.as_ptr() as *const u16, batch.len() / 2)
                    };
                    posting.push(Chunk::IDs(batch));
                }
            } else {
                posting.push(Chunk::N0NE);
            }
        }
        if posting.len() > 0 {
            batches.push(Some(posting));
        } else {
            batches.push(None);
        }
    }
    let eval = predicate.eval_avx512(&batches, None, true, min_range.len())?;
    if let Some(e) = eval {
        let mut accumulator = 0;
        let mut id_acc = 0;
        e.into_iter()
        .enumerate()
        .for_each(|(n, v)| unsafe {
            match v {
                TempChunk::Bitmap(b) => {
                    let popcnt = _mm512_popcnt_epi64(b);
                    let cnt = _mm512_reduce_add_epi64(popcnt);
                    debug!("num: {:}, batch0: {:?}, batch1: {:?}", cnt, batches[0].as_ref().unwrap()[n], batches[1].as_ref().unwrap()[n]);
                    accumulator += cnt;
                }
                TempChunk::IDs(i) => {
                    debug!("num: {:}, valid: {:?}, batch0: {:?}, batch1: {:?}", i.len(), i, batches[0].as_ref().unwrap()[n], batches[1].as_ref().unwrap()[n]);
                    id_acc += i.len();
                }
                TempChunk::N0NE => {},
            }
        });
        debug!("accumulator: {}", accumulator);
        // Ok(accumulator as usize + id_acc)
        Ok(accumulator as usize)
    } else {
        Ok(0)
    }
}
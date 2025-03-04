use std::{arch::x86_64::{_mm512_popcnt_epi64, _mm512_reduce_add_epi64}, collections::{BTreeMap, BTreeSet}, mem::size_of_val, ops::Index, ptr::NonNull, slice::from_raw_parts, sync::Arc};

use adaptive_hybrid_trie::TermIdx;
use datafusion::{arrow::{array::{Array, ArrayData, ArrayRef, BooleanArray, GenericBinaryArray, GenericBinaryBuilder, GenericListArray, Int64RunArray, PrimitiveRunBuilder, UInt16Array, UInt32Array}, buffer::Buffer, datatypes::{DataType, Field, Int64Type, Schema, SchemaRef, ToByteSlice, UInt8Type}, record_batch::RecordBatch}, common::TermMeta, from_slice::FromSlice};
use roaring::RoaringBitmap;
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use tracing::{debug, info};
use crate::{datasources::mmap_table::{PostingColumn, PostingSegment}, physical_expr::{boolean_eval::{freqs_filter, Chunk, TempChunk}, BooleanEvalExpr}, utils::{avx512::U64x8, FastErr}};
use crate::utils::Result;


/// The doc_id range [start, end) Batch range determines the  of relevant batch.
/// nums32 = (end - start) / 32
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchRange {
    start: u32, 
    end: u32,
    nums32: u32,
}

impl BatchRange {
    /// new a BatchRange
    pub fn new(start: u32, end: u32) -> Self {
        Self {
            start,
            end,
            nums32: (end - start + 31) / 32
        }
    }

    /// get the `start` of BatchRange
    pub fn start(&self) -> u32 {
        self.start
    }
    
    /// get the `end` of BatchRange
    pub fn end(&self) -> u32 {
        self.end
    }

    /// get the `len` of BatchRange
    pub fn len(&self) -> u32 {
        self.end - self.start
    }
}

pub type PostingList = GenericBinaryArray<i32>;
pub type TermSchemaRef = SchemaRef;
pub type Freqs = Arc<GenericBinaryArray<i32>>;
pub type BatchFreqs = Vec<Freqs>;


/// A batch of Postinglist which contain serveral terms,
/// which is in range[start, end)
#[derive(Clone, Debug)]
pub struct PostingBatch {
    postings: Vec<Arc<PostingList>>,
    term_freqs: Option<BatchFreqs>,
    range: Arc<BatchRange>,
    // tombstone bitmap for deletion operations
    valid: Arc<RwLock<RoaringBitmap>>,
    pub term_idx: Arc<TermIdx<TermMeta>>,
}

impl PostingBatch {
    pub fn term_metas_of(&self, terms: &Vec<String>) -> Vec<Option<&TermMeta>> {
        let mut metas = vec![];
        for term in terms {
            metas.push(self.term_idx.get(term));
        }
        metas
    }

    pub async fn delete(&self, position: u32) {
        self.valid.write().await.insert(position);
    }

    pub fn memory_consumption(&self) -> (usize, usize, usize) {
        let mut offset: usize = 0;
        let mut postings: usize = 0;
        self.postings.iter()
            .for_each(|v| {
                postings += v.data_ref().buffers()[0].capacity();
                offset += v.data_ref().buffers()[1].capacity();
            });
        let freqs = match &self.term_freqs {
            Some(m) => m.iter().map(|v| {
                v.get_array_memory_size()
            }).sum::<usize>(),
            None => 0,
        };
        (postings + offset + freqs, postings, offset) 
    }

    pub fn try_new(
        postings: Vec<Arc<PostingList>>,
        term_idx: Arc<TermIdx<TermMeta>>,
        range: Arc<BatchRange>,
    ) -> Result<Self> {
        Self::try_new_impl(
            postings,
            term_idx,
            None,
            range,
        )
    }

    pub fn try_new_with_freqs(
        postings: Vec<Arc<PostingList>>,
        term_idx: Arc<TermIdx<TermMeta>>,
        term_freqs: BatchFreqs,
        range: Arc<BatchRange>,
    ) -> Result<Self> {
        Self::try_new_impl(
            postings,
            term_idx,
            Some(term_freqs),
            range,
        )
    }

    pub fn try_new_impl(
        postings: Vec<Arc<PostingList>>,
        term_idx: Arc<TermIdx<TermMeta>>,
        term_freqs: Option<BatchFreqs>,
        range: Arc<BatchRange>,
    ) -> Result<Self> {
        Ok(Self {
            postings,
            term_idx,
            term_freqs,
            range,
            valid: Arc::new(RwLock::new(RoaringBitmap::new())),
        })  
    }

    /// Return the length of this partition
    pub fn batch_len(&self) -> usize {
        self.range.len() as usize
    }

    pub fn roaring_predicate(
        &self,
        distris: &[Option<Arc<RoaringBitmap>>],
        indices: &[Option<u32>],
        min_range: &[u32],
        predicate: &BooleanEvalExpr,
    ) -> Result<usize> {
        debug_assert_eq!(indices.len(), distris.len());
        if min_range.len() ==0 {
            return Ok(0);
        }
        let predicate = {
            match predicate.predicate.as_ref() {
                Some(predicate) => unsafe {
                    predicate.get().as_ref().unwrap()
                }
                None => return Ok(0),
            }
        };
        let mut batches: Vec<Option<Vec<Chunk>>> = Vec::with_capacity(distris.len());
        // let _valid = self.valid.read().unwrap().clone();
        // let _valid: u64 = self.valid.read().await.len();
        debug!("Start select valid batch.");
        for (term, i) in distris.iter().zip(indices.iter()) {
            let mut posting = Vec::with_capacity(min_range.len());
            let batch = if let Some(i) = i {
                let posting = self.postings.get(*i as usize).cloned().unwrap();
                posting
            } else {
                debug!("break, i: {:?}", i);
                break;
            };
            let term = term.as_ref().unwrap();
            let mut rank_iter = term.rank_iter();
            debug!("min_range: {:?}", min_range);
            for &idx in min_range {
                // if term.contains(idx) {
                    let bound_idx: usize = rank_iter.rank(idx);
                    // debug!("bound_idx: {:}, real: {:}, idx: {:}, i: {:}", bound_idx, term.rank(idx), idx, i.unwrap());
                    let batch = batch.value(bound_idx - 1);
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
                // } else {
                //     posting.push(Chunk::N0NE);
                // }
            }
            if posting.len() > 0 {
                batches.push(Some(posting));
            } else {
                batches.push(None);
            }
        }
        debug_assert!(batches[0].is_some());
        debug_assert!(batches[1].is_some());
        // let start_time = Instant::now();
        let eval = predicate.eval_avx512(&batches, None, true, min_range.len())?;
        // let duration = start_time.elapsed().as_micros();
        // let eval: Option<Vec<TempChunk>> = None;
        if let Some(e) = eval {
            let mut accumulator = 0;
            let mut id_acc = 0;
            e.into_iter()
            .for_each(|v| unsafe {
                match v {
                    TempChunk::Bitmap(b) => {
                        let popcnt = _mm512_popcnt_epi64(b);
                        let cnt = _mm512_reduce_add_epi64(popcnt);
                        // debug!("num: {:}, batch0: {:?}, batch1: {:?}", cnt, batches[0].as_ref().unwrap()[n], batches[1].as_ref().unwrap()[n]);
                        accumulator += cnt;
                    }
                    TempChunk::IDs(i) => {
                        // debug!("num: {:}, valid: {:?}, batch0: {:?}, batch1: {:?}", i.len(), i, batches[0].as_ref().unwrap()[n], batches[1].as_ref().unwrap()[n]);
                        id_acc += i.len();
                    }
                    TempChunk::N0NE => {},
                }
            });
            debug!("accumulator: {}", accumulator);
            // Ok(accumulator as usize + id_acc)
            Ok(accumulator as usize + id_acc)
        } else {
            Ok(0)
        }
    }

    pub fn roaring_predicate_with_score(
        &self,
        distris: &[Option<Arc<RoaringBitmap>>],
        projected_schema: &Schema,
        indices: &[Option<u32>],
        min_range: &[u32],
        predicate: &BooleanEvalExpr,
    ) -> Result<u32> {
        debug!("Roaring_predicate_with_score");
        if min_range.len() ==0 {
            return Ok(0);
        }
        let mut fields = projected_schema.fields().clone();
        let mut freqs = fields
            .iter()
            .filter(|v| v.name() != "__id__")
            .map(|v| Field::new(format!("{}_freq", v.name()), DataType::UInt8, true))
            .collect();
        fields.append(&mut freqs);
        
        debug_assert_eq!(indices.len(), distris.len());
        let predicate = {
            match predicate.predicate.as_ref() {
                Some(predicate) => unsafe {
                    predicate.get().as_ref().unwrap()
                }
                None => return Ok(0),
            }
        };
        let mut batches: Vec<Option<Vec<Chunk>>> = vec![];
        let mut freqs_batch: Vec<Option<Vec<Option<&[u8]>>>> = vec![None; indices.len()];

        debug!("Start select valid batch.");
        debug!("indices: {:?}", indices);
        debug!("term1: {:?}, term2: {:?}", distris[0],distris[1]);
        for (term, i) in distris.iter().zip(indices.iter()) {
            // let term = term.as_ref().unwrap();
            let mut posting = Vec::with_capacity(min_range.len());
            let mut freqs = Vec::with_capacity(min_range.len());
            let (batch, batch_freq) = if let Some(i) = i {
                let posting = self.postings.get(*i as usize).cloned().unwrap();
                let freq = self.term_freqs.as_ref().unwrap()[*i as usize].clone();
                (posting, freq)
            } else {
                debug!("break, i: {:?}", i);
                break;
            };
            let term = term.as_ref().unwrap();
            let mut rank_iter = term.rank_iter();
            for &idx in min_range {
                if term.contains(idx) {
                    // let bound_idx: usize = term.rank(idx) as usize;
                    let bound_idx = rank_iter.rank(idx);
                    
                    let batch = batch.value(bound_idx - 1);
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
                    let freq = batch_freq.value(bound_idx - 1);
                    freqs.push(Some(unsafe { from_raw_parts(freq.as_ptr(), freq.len())}))
                } else {
                    posting.push(Chunk::N0NE);
                    freqs.push(None);
                }
            }
            if posting.len() > 0 {
                batches.push(Some(posting));
                freqs_batch.push(Some(freqs));
            } else {
                debug!("posting len is 0");
                batches.push(None);
            }
        }
        debug_assert!(batches[0].is_some());
        debug_assert!(batches[1].is_some());
        let eval = predicate.eval_avx512(&batches, None, true, min_range.len())?;
        // let eval: Option<Vec<TempChunk>> = None;
        let mut freq_res = 0;
        if let Some(e) = eval {
            let mut accumulator = 0;
            let mut id_acc = 0;
            e.into_iter()
            .enumerate()
            .for_each(|(i, v)| unsafe {
                match v {
                    TempChunk::Bitmap(b) => {
                        let popcnt = _mm512_popcnt_epi64(b);
                        let cnt = _mm512_reduce_add_epi64(popcnt);
                        if cnt != 0 {
                            freq_res += freqs_filter(&freqs_batch, &U64x8{ vector: b }.vals, i).len();
                        }
                        accumulator += cnt;
                    }
                    TempChunk::IDs(i) => {
                        id_acc += i.len();
                    }
                    TempChunk::N0NE => {},
                }
            });
            Ok(accumulator as u32 + id_acc as u32 + freq_res as u32)
        } else {
            // unreachable!();
            Ok(0)
        }

    }

    pub fn project_fold_with_freqs(
        &self,
        indices: &[Option<usize>],
        projected_schema: SchemaRef,
        _distris: &[Option<u64>],
        _boundary_idx: usize,
        _min_range: u64,
    ) -> Result<RecordBatch> {
        debug!("project_fold_with_freqs");
        // Add freqs fields
        let mut fields = projected_schema.fields().clone();
        let mut freqs = fields
            .iter()
            .filter(|v| v.name() != "__id__")
            .map(|v| Field::new(format!("{}_freq", v.name()), DataType::UInt8, true)).collect();
        fields.append(&mut freqs);
        let mut freqs: Vec<ArrayRef> = Vec::with_capacity(indices.len());
        let bitmask_size: usize = self.range.len() as usize;
        let mut batches: Vec<ArrayRef> = Vec::new();
        for idx in indices {
            // Use customized construction for performance
            let mut bitmap = vec![0; 8];
            if idx.is_none() {
                batches.push(build_boolean_array(bitmap, bitmask_size));
                freqs.push(Arc::new(GenericListArray::<i32>::from_iter_primitive::<UInt8Type, _, _>(vec![None as Option<Vec<Option<u8>>>, None, None, None])));
                continue;
            }

            // Safety: If idx is none, this loop will continue before this unwrap_unchecked()
            let idx = unsafe { idx.unwrap_unchecked() };
            let posting = self.postings.get(idx).cloned();
            if let Some(posting) = posting {
                match posting.data_type() {
                    DataType::Boolean => batches.push(posting.clone()),
                    DataType::UInt16 => {
                        posting.as_any()
                        .downcast_ref::<UInt16Array>()
                        .unwrap()
                        .iter()
                        .for_each(|v| {
                            let v = v.unwrap();
                            bitmap[v as usize >> 6] |= 1 << (v % 64) as usize;
                        });
                        batches.push(build_boolean_array(bitmap, bitmask_size));
                    }
                    _ => {}
                }
                // add freqs array
                freqs.push(self.term_freqs.as_ref().unwrap().get(idx).cloned().ok_or_else(|| {
                    FastErr::InternalErr(format!(
                        "freqs index {} out of bounds, term_freq is none: {}",
                        idx,
                        self.term_freqs.is_none(),
                    ))
                })?);
            } else {
                batches.push(build_boolean_array(bitmap, bitmask_size));
                freqs.push(Arc::new(GenericListArray::<i32>::from_iter_primitive::<UInt8Type, _, _>(vec![None as Option<Vec<Option<u8>>>, None, None, None])));
            }
        }
        batches.insert(projected_schema.index_of("__id__").expect("Should have __id__ field"), Arc::new(UInt32Array::from_slice([])));
        batches.extend(freqs.into_iter());
        // Update: Don't add array for __id__
        let projected_schema = Arc::new(Schema::new(fields));
        Ok(RecordBatch::try_new(
            projected_schema,
            batches,
        )?)
    }

    pub fn space_usage(&self) -> usize {
        let mut space = 0;
        space += self.postings
            .iter()
            .map(|v| size_of_val(v.data().buffers()[0].as_slice()))
            .sum::<usize>();
        space
    }

    pub fn schema(&self) -> TermSchemaRef {
        unreachable!()
    }

    pub fn range(&self) -> &BatchRange {
        &self.range
    }

    pub fn posting_by_term(&self, name: &str) -> Option<&PostingList> {
        self.schema()
            .column_with_name(name)
            .map(|(index, _)| self.postings[index].as_ref())
    }
}



impl Index<&str> for PostingBatch {
    type Output = PostingList;

    /// Get a reference to a term's posting by name
    /// 
    /// # Panics
    /// 
    /// Panics if the name is not int the schema
    fn index(&self, name: &str) -> &Self::Output {
        self.posting_by_term(name).unwrap()
    }
}

#[derive(Serialize, Deserialize)]
pub struct PostingBatchBuilder {
    base: u32,
    current: u32,
    pub term_dict: Option<BTreeMap<String, Vec<(u32, u8)>>>,
    term_num: usize,
    // builder for construting term index
    pub term_idx: Option<BTreeMap<String, TermMetaBuilder>>,
}

impl PostingBatchBuilder {
    pub fn new(base: u32) -> Self {
        Self { 
            base,
            current: 0,
            term_dict: Some(BTreeMap::new()),
            term_num: 0,
            term_idx: Some(BTreeMap::new()),
        }
    }

    /// Clean the inner in-memory buffer data.
    pub fn clear(&mut self) {
        self.base += self.current + 1;
        self.current = 0;
        self.term_num = 0;
    }

    pub fn doc_len(&self) -> usize {
        self.current as usize
    }

    pub fn add_docs(&mut self, doc: Vec<String>, doc_id: usize) -> Result<()> {
        for term in doc.into_iter() {
            self.push_term(term, doc_id as u32)?;
        }
        Ok(())
    }

    pub fn push_term(&mut self, term: String, doc_id: u32) -> Result<()> {
        let off = doc_id - self.base;
        self.term_idx.as_mut().unwrap().entry(term.clone()).or_insert(TermMetaBuilder::new(0, 0));
        let entry = self.term_dict.as_mut().unwrap()
            .entry(term)
            .or_insert(vec![(off, 0)]);
        if entry.last().unwrap().0 ==  off {
            if entry.last_mut().unwrap().1 != u8::MAX {
                entry.last_mut().unwrap().1 += 1;
            }
        } else {
            entry.push((off, 1));
        }
        self.current = off;
        self.term_num += 1;
        Ok(())
    }

    pub fn build(&mut self) -> Result<PostingBatch> {
        self.build_with_idx(None)
    }

    pub fn build_with_idx(&mut self, mut with_idx: Option<&mut BTreeMap<String, TermMetaBuilder>>) -> Result<PostingBatch> {
        let term_dict = self.term_dict.take().unwrap();
        self.term_dict = Some(BTreeMap::new());

        let mut schema_list = Vec::new();
        let mut postings: Vec<Arc<PostingList>> = Vec::new();
        let mut freqs = Vec::new();
        
        let mut id_num: usize = 0;
        let mut bitmap_num: usize = 0;
        let term_num: usize = term_dict.len();
        let mut binary_num: usize = 0;
        for (i, (k, v)) in term_dict.into_iter().enumerate() {
                // let mut boundary = vec![0];
                let mut cnter = 0;
                let mut builder_len = 0;
                let mut batch_num = 0;
                let entry = self.term_idx.as_mut().unwrap().get_mut(&k).unwrap();
                entry.add_idx(i as u32, 0);
                with_idx.as_mut().map(|v| 
                    v.entry(k.clone()).and_modify(|v| v.add_idx(i as u32, 0))
                );
                if let Some(vv) = v.first() {
                   entry.set_true(vv.0 as usize/ 512, 0, vv.0);
                }
                let mut binary_buffer: Vec<Vec<u8>> = Vec::new();
                let mut buffer: Vec<u16> = Vec::with_capacity(512);
                let mut freq_buffer: Vec<u8> = Vec::with_capacity(512);
                let mut freqs_vec: Vec<Vec<u8>> = Vec::new();

                let mut byte_num = 0;
                let mut freq_num = 0;
                v.into_iter()
                .for_each(|(p, f)| {
                    if p - cnter < 512 {
                        buffer.push((p - cnter) as u16);
                        freq_buffer.push(f);
                        freq_num += 1;
                    } else {
                        entry.set_true(p as usize/ 512, 0, p);
                        let skip_num = (p - cnter) / 512;
                        cnter += skip_num * 512;
                        batch_num += 1;
                        if buffer.len() > 8 {
                            bitmap_num += 1;
                            let mut bitmap: Vec<u64> = vec![0; 8];
                            for i in &buffer {
                                let off = *i;
                                bitmap[(off >> 6) as usize] |= 1 << (off % (1 << 6));
                            }
                            byte_num += bitmap.len() * 8;
                            binary_buffer.push(bitmap.to_byte_slice().to_vec());
                            builder_len += 1;
                        } else if buffer.len() > 0 {
                            id_num += 1;
                            binary_buffer.push(unsafe {
                                let value = buffer.as_slice().align_to::<u8>();
                                assert!(value.0.len() == 0 && value.2.len() == 0);
                                byte_num += value.1.len();
                                value.1.to_vec()
                            });
                            builder_len += 1;
                        }
                        buffer.clear();
                        buffer.push((p - cnter) as u16);

                        freqs_vec.push(freq_buffer.drain(..).collect());
                        freq_buffer.clear();
                        freq_buffer.push(f % 8);
                        freq_num += 1;
                    }

                    
                });

                if buffer.len() > 0 {
                    entry.set_true(buffer[0] as usize/ 512, 0, buffer[0] as u32 + 512);
                    if buffer.len() > 8 {
                        let mut bitmap: Vec<u64> = vec![0; 8];
                        for i in &buffer {
                            let off = *i;
                            bitmap[(off >> 6) as usize] |= 1 << (off % (1 << 6));
                        }
                        byte_num += bitmap.len() * 8;
                        binary_buffer.push(bitmap.to_byte_slice().to_vec());
                    } else {
                        binary_buffer.push(unsafe {
                            let value = buffer.as_slice().align_to::<u8>();
                            assert!(value.0.len() == 0 && value.2.len() == 0);
                            byte_num += value.1.len();
                            value.1.to_vec()
                        });
                    }
                    freqs_vec.push(freq_buffer);
                }
                let mut freqs_builder = GenericBinaryBuilder::with_capacity(freqs_vec.len(), freq_num);
                for freq in freqs_vec {
                    freqs_builder.append_value(freq);
                }
                freqs.push(Arc::new(freqs_builder.finish()));

                binary_num += binary_buffer.len();
                let mut posting_builder = GenericBinaryBuilder::with_capacity(binary_buffer.len(), byte_num);
                for batch in binary_buffer {
                    posting_builder.append_value(batch);
                }
                schema_list.push(Field::new(k.clone(), DataType::UInt32, false));
                let posting = posting_builder.finish();

                postings.push(Arc::new(posting));
        }
        info!("id list size: {:}", id_num);
        info!("bitmap size: {:}", bitmap_num);
        info!("term num: {:}", term_num);
        info!("binary len: {:}", binary_num);
        schema_list.push(Field::new("__id__", DataType::UInt32, false));
        postings.push(Arc::new(PostingList::from(vec![] as Vec<&[u8]>)));
        let term_index_iter = self.term_idx.take().unwrap()
            .into_iter()
            .map(|(k, v)| {
                (k, v.build())
            });
        let term_index = TermIdx {
            term_map: BTreeMap::from_iter(term_index_iter)
        };
        self.term_idx = Some(BTreeMap::new());

        PostingBatch::try_new_with_freqs(
            postings,
            Arc::new(term_index),
            freqs,
            Arc::new(BatchRange::new(self.base, self.base + self.current + 1))
        )
    }

    pub fn build_mmap_segment(&mut self, _partition_num: usize) -> Result<PostingSegment> {
        let term_dict = self.term_dict.take().unwrap();
        self.term_dict = Some(BTreeMap::new());

        let mut schema_list = Vec::new();
        let mut postings: Vec<PostingColumn> = Vec::new();
        // let mut freqs = Vec::new();
        
        let mut id_num: usize = 0;
        let mut bitmap_num: usize = 0;
        let term_num: usize = term_dict.len();
        for  (k, v) in term_dict.into_iter() {
                // let mut boundary = vec![0];
                let mut cnter = 0;
                let mut batch_num = 0;
                let mut offset = 0;
                // let mut binary_buffer: Vec<Vec<u8>> = Vec::new();
                let mut posting_list = Vec::new();
                let mut offset_list = vec![offset];
                let mut buffer: Vec<u16> = Vec::with_capacity(512);
                // let mut freq_buffer: Vec<u8> = Vec::with_capacity(512);
                // let mut freqs_vec: Vec<Vec<u8>> = Vec::new();

                let mut byte_num = 0;
                v.into_iter()
                .for_each(|(p, _)| {
                    if p - cnter < 512 {
                        buffer.push((p - cnter) as u16);
                        // freq_buffer.push(f);
                        // freq_num += 1;
                    } else {
                        let skip_num = (p - cnter) / 512;
                        cnter += skip_num * 512;
                        batch_num += 1;
                        if buffer.len() > 8 {
                            bitmap_num += 1;
                            let mut bitmap: Vec<u64> = vec![0; 8];
                            for i in &buffer {
                                let off = *i;
                                bitmap[(off >> 6) as usize] |= 1 << (off % (1 << 6));
                            }
                            byte_num += bitmap.len() * 8;
                            posting_list.extend_from_slice(unsafe { bitmap.as_slice().align_to::<u8>().1 });
                            offset += 64;
                            offset_list.push(offset);
                        } else if buffer.len() > 0 {
                            id_num += 1;
                            posting_list.extend_from_slice(unsafe {
                                let value = buffer.as_slice().align_to::<u8>();
                                assert!(value.0.len() == 0 && value.2.len() == 0);
                                byte_num += value.1.len();
                                value.1
                            });
                            offset += buffer.len() as u32 * 2;
                            offset_list.push(offset)
                        }
                        buffer.clear();
                        buffer.push((p - cnter) as u16);
                    }
                });

                if buffer.len() > 0 {
                    if buffer.len() > 8 {
                        let mut bitmap: Vec<u64> = vec![0; 8];
                        for i in &buffer {
                            let off = *i;
                            bitmap[(off >> 6) as usize] |= 1 << (off % (1 << 6));
                        }
                        posting_list.extend_from_slice(unsafe { bitmap.as_slice().align_to::<u8>().1 });
                        offset += 64;
                        offset_list.push(offset);
                    } else {
                        posting_list.extend_from_slice(unsafe {
                            let value = buffer.as_slice().align_to::<u8>();
                            assert!(value.0.len() == 0 && value.2.len() == 0);
                            value.1
                        });
                        offset += buffer.len() as u32 * 2;
                        offset_list.push(offset);
                    }
                }
                schema_list.push(Field::new(k.clone(), DataType::UInt32, false));
                postings.push(PostingColumn::new(posting_list, offset_list));
        }
        info!("id list size: {:}", id_num);
        info!("bitmap size: {:}", bitmap_num);
        info!("term num: {:}", term_num);
        schema_list.push(Field::new("__id__", DataType::UInt32, false));
        postings.push(PostingColumn::new(vec![], vec![]));
        Ok(PostingSegment { posting_lists: postings })
    }
}

/// Core operation of merging two posting batches.
pub fn merge_segments(batches: Vec<Arc<PostingBatch>>) -> Result<Arc<PostingBatch>> {
    assert!(batches.len() > 1, "Should merge multiple segments greater than 2");
    let mut cnt = 0;
    let mut intervals = vec![];
    for i in &batches {
        intervals.push(cnt / 512);
        cnt += i.range().len();
    }
    debug!("intervals: {:?}", intervals);

    let range = Arc::new(BatchRange::new(
        batches.first().unwrap().range.start,
        batches.last().unwrap().range.end,
    ));

    let mut terms = batches.iter()
        .enumerate()
        .map(|(i, it)| {
            it.term_idx.term_map.iter()
            .map(|v| (i, v.0, v.1))
            .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();
    terms.sort_by(|a, b| 
        a.1.cmp(b.1).then_with(|| a.0.cmp(&b.0)));
    debug!("merged terms: {:?}", terms);
    let mut postings = vec![];
    let mut term_idx = BTreeMap::new();

    let mut current: (usize, &String) = (terms[0].0, terms[0].1);
    let mut merge_batches = Vec::with_capacity(batches.len());
    merge_batches.push(&terms[0]);
    let mut ii = 0;
    let mut item = &terms[1];
    let mut flush = false;
    // for (ii, item) in terms[1..].iter().enumerate() {
    loop {
        if item.1 > current.1 || flush {
            // merge batches
            if merge_batches.len() == 1  {
                let mut meta = merge_batches[0].2.clone();
                if current.0 != 0 {
                    let interval = intervals[current.0];
                    let bitmap = RoaringBitmap::from_sorted_iter(
                        meta.valid_bitmap.iter().map(|v| v + interval)).unwrap();
                    meta.valid_bitmap = Arc::new(bitmap);
                }
                meta.index = postings.len() as u32;
                term_idx.insert(current.1.clone(), meta);
                postings.push(batches[current.0].postings[merge_batches[0].2.index as usize].clone());
            } else {
                // debug!("batches when merging: {:?}", merge_batches);
                let bitmap = RoaringBitmap::from_sorted_iter(
                    merge_batches.iter()
                    .map(|v| v.2.valid_bitmap.iter().map(|i| i + intervals[v.0]))
                    .flatten()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                ).unwrap();
                let meta = TermMeta {
                    valid_bitmap: Arc::new(bitmap),
                    index: postings.len() as u32,
                    nums: 0,
                    selectivity: 0.,
                };
                // merge posting lists
                // If the last batch is smaller than 512, we should re
                let mut postings_tmp = vec![];
                let mut byte_num = 0;
                let mut item_num = 0;
                merge_batches.iter()
                .for_each(|v| {
                    let posting = batches[v.0].postings[v.2.index as usize].clone();
                    byte_num += posting.len();
                    item_num += posting.value_offsets().len();
                    postings_tmp.push(posting);
                });
                let mut builder = GenericBinaryBuilder::with_capacity(item_num, byte_num);
                for p in postings_tmp {
                    for i in p.iter() {
                        builder.append_value(i.unwrap());
                    }
                }
                postings.push(Arc::new(builder.finish()));
                term_idx.insert(current.1.clone(), meta);
            }
            current = (item.0, item.1);
            merge_batches.clear();
            merge_batches.push(&item);
        } else {
            merge_batches.push(&item);
        }
        if flush {
            break;
        }
        if ii >= terms.len() - 2 && merge_batches.len() > 0 {
            flush = true;
            continue;
        }
        ii += 1;
        item = &terms[ii + 1];
    }
    let term_idx = Arc::new(TermIdx{ term_map: term_idx });
    Ok(Arc::new(PostingBatch::try_new(postings, term_idx, range).unwrap()))
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TermMetaBuilder {
    pub distribution: Vec<u64>,
    pub valid_bitmap: RoaringBitmap,
    pub nums: u32,
    idx: u32,
    partition_num: usize,
    bounder: Option<u32>,
}

impl TermMetaBuilder {
    pub fn new(_batch_num: usize, partition_num: usize) -> Self {
        Self {
            distribution: vec![],
            valid_bitmap: RoaringBitmap::new(),
            nums: 0,
            idx: u32::MAX,
            partition_num,
            bounder: None,
        }
    }

    pub fn set_true(&mut self, i: usize, _partition_num: usize, id: u32) {
        if self.bounder.is_none() || id > self.bounder.unwrap() {
            self.nums += 1;
            self.bounder = Some(id);
        }
        // Add doc_id to valid_bitmap
        self.valid_bitmap.insert(i as u32);
    }

    pub fn add_idx(&mut self, idx: u32, _partition_num: usize) {
        self.idx = idx;
    }

    pub fn rle_usage(&self) -> usize {
        let mut builder = PrimitiveRunBuilder::<Int64Type, Int64Type>::with_capacity(self.distribution.len());
        for i in &self.distribution {
            builder.append_value(*i as i64);
        }
        let array: Int64RunArray = builder.finish();
        array.get_array_memory_size()
    }

    pub fn build(self) -> TermMeta {
        let valid_bitmap: Arc<RoaringBitmap> = Arc::new(self.valid_bitmap);
        let valid_batch_num: usize = valid_bitmap.len() as usize;
        let sel =   self.nums as f64 / (512. * valid_batch_num as f64);
        TermMeta {
            valid_bitmap,
            index: self.idx,
            nums: self.nums,
            selectivity: sel,
        }
    }
}

fn build_boolean_array(mut data: Vec<u64>, batch_len: usize) -> ArrayRef {
    let value_buffer = unsafe {
        let buf = Buffer::from_raw_parts(NonNull::new_unchecked(data.as_mut_ptr() as *mut u8), batch_len, batch_len);
        std::mem::forget(data);
        buf
    };
    let builder = ArrayData::builder(DataType::Boolean)
        .len(batch_len)
        .add_buffer(value_buffer);

    let array_data = builder.build().unwrap();
    Arc::new(BooleanArray::from(array_data))
}

#[cfg(test)]
mod test {}
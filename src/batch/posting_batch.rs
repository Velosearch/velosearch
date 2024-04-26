use std::{arch::x86_64::{_mm512_popcnt_epi64, _mm512_reduce_add_epi64}, cell::RefCell, collections::BTreeMap, mem::size_of_val, ops::Index, ptr::NonNull, slice::from_raw_parts, sync::Arc, time::Instant};

use datafusion::{arrow::{array::{Array, ArrayData, ArrayRef, BooleanArray, GenericBinaryArray, GenericBinaryBuilder, GenericListArray, Int64RunArray, PrimitiveRunBuilder, UInt16Array, UInt32Array}, buffer::Buffer, datatypes::{DataType, Field, Int64Type, Schema, SchemaRef, ToByteSlice, UInt8Type}, record_batch::RecordBatch}, common::TermMeta, from_slice::FromSlice};
use roaring::RoaringBitmap;
use serde::{Serialize, Deserialize};
use tracing::{debug, info};
use crate::{datasources::mmap_table::{PostingColumn, PostingSegment}, physical_expr::{boolean_eval::{freqs_filter, Chunk, TempChunk}, BooleanEvalExpr}, utils::{avx512::U64x8, FastErr, Result}};


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
#[derive(Clone, Debug, PartialEq)]
pub struct PostingBatch {
    schema: TermSchemaRef,
    postings: Vec<Arc<PostingList>>,
    term_freqs: Option<BatchFreqs>,
    range: Arc<BatchRange>
}

impl PostingBatch {
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
                // v.data().buffers()[0].len() + v.data().buffers()[1].len()
                v.get_array_memory_size()
            }).sum::<usize>(),
            None => 0,
        };
        (postings + offset + freqs, postings, offset) 
    }

    pub fn try_new(
        schema: TermSchemaRef,
        postings: Vec<Arc<PostingList>>,
        boundary: Vec<Vec<u16>>,
        range: Arc<BatchRange>,
    ) -> Result<Self> {
        Self::try_new_impl(schema, postings, None, boundary, range)
    }

    pub fn try_new_with_freqs(
        schema: TermSchemaRef,
        postings: Vec<Arc<PostingList>>,
        term_freqs: BatchFreqs,
        boundary: Vec<Vec<u16>>,
        range: Arc<BatchRange>,
    ) -> Result<Self> {
        Self::try_new_impl(schema, postings, Some(term_freqs), boundary, range)
    }

    pub fn try_new_impl(
        schema: TermSchemaRef,
        postings: Vec<Arc<PostingList>>,
        term_freqs: Option<BatchFreqs>,
        _boundary: Vec<Vec<u16>>,
        range: Arc<BatchRange>,
    ) -> Result<Self> {
        if schema.fields().len() != postings.len() {
            return Err(FastErr::InternalErr(format!(
                "number of columns({}) must match number of fields({}) in schema",
                postings.len(),
                schema.fields().len(),
            )));
        }
        Ok(Self {
            schema, 
            postings,
            term_freqs,
            range
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
        // let mut rank_iters: Vec<Option<RankIter>> = distris
        //     .iter()
        //     .map(|e| {
        //         match e {
        //             Some(e) => {
        //                 Some(e.rank_iter())
        //             }
        //             None => None
        //         }
        //     })
        //     .collect();
        debug!("Start select valid batch.");
        debug!("term1: {:?}, term2: {:?}", distris[0],distris[1]);
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
                if term.contains(idx) {
                    let bound_idx: usize = rank_iter.rank(idx);
                    debug!("bound_idx: {:}, real: {:}, idx: {:}, i: {:}", bound_idx, term.rank(idx), idx, i.unwrap());
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
        debug_assert!(batches[0].is_some());
        debug_assert!(batches[1].is_some());
        let start_time = Instant::now();
        let eval = predicate.eval_avx512(&batches, None, true, min_range.len())?;
        let duration = start_time.elapsed().as_micros();
        // let eval: Option<Vec<TempChunk>> = None;
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
            Ok(duration as usize)
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

    // pub fn project_fold(
    //     &self,
    //     indices: &[Option<u32>],
    //     projected_schema: SchemaRef,
    //     distris: &[Option<u64>],
    //     boundary_idx: usize,
    //     min_range: u64,
    // ) -> Result<RecordBatch> {
    //     debug!("project_fold");
    //     let valid_batch_num = min_range.count_ones() as usize;
    //     let mut batches: Vec<ArrayRef> = Vec::with_capacity(indices.len());
    //     for (idx, distri) in indices.into_iter().zip(distris.into_iter()) {
    //         // To be optimized, we can convert bitvec to BooleanArray
    //         if idx.is_none() {
    //             batches.push(Arc::new(UInt64Array::from(vec![] as Vec<u64>)));
    //             continue;
    //         }

    //         // Safety: If idx is none, this loop will continue before this unwrap_unchecked()
    //         let idx = unsafe{ idx.unwrap_unchecked() as usize };
    //         let distri = unsafe { distri.unwrap_unchecked() };

    //         let posting = if idx != usize::MAX {
    //             self.postings.get(idx).cloned().ok_or_else(|| {
    //                 FastErr::InternalErr(format!(
    //                     "project index {} out of bounds, max field {}",
    //                     idx,
    //                     self.postings.len()
    //                 ))
    //             })?
    //         } else {
    //             batches.push(Arc::new(UInt64Array::from(vec![] as Vec<u64>)));
    //             continue;
    //         };

    //         let boundary = &self.boundary[idx];
    //         if boundary_idx >= boundary.len() {
    //             batches.push(Arc::new(UInt64Array::from(vec![] as Vec<u64>)));
    //             continue;
    //         }
    //         let start_idx = boundary[boundary_idx] as usize;

    //         let boundary_len = if boundary_idx < boundary.len() - 1 {
    //             (boundary[boundary_idx + 1] - boundary[boundary_idx]) as usize
    //         } else if  boundary_idx == boundary.len() - 1 {
    //             posting.len() - boundary[boundary_idx] as usize
    //         } else {
    //             batches.push(Arc::new(UInt64Array::from(vec![] as Vec<u64>)));
    //             continue;
    //         };

    //         let mut valid_batch: Vec<u64> = vec![0; valid_batch_num * 8];
    //         let valid_mask = unsafe { _pext_u64(min_range, distri) };
    //         let mut write_mask = unsafe { _pext_u64(distri, min_range) };

    //         let mut write_pos = write_mask.trailing_zeros() as usize;
    //         write_mask = clear_lowest_set_bit(write_mask);
    //         for i in 0..boundary_len.min(64) {
    //             if valid_mask & (1 << i) == 0 {
    //                 continue
    //             }
    //             let batch = posting.value(start_idx + i);
                // if batch.len() == 64 {
                //     let batch = unsafe { from_raw_parts(batch.as_ptr() as *const u64, 8) };
                //     valid_batch[(write_pos * 8)..(write_pos * 8 + 8)].copy_from_slice(batch);
                // } else {
                //     // means this's Integer list
                //     let batch = unsafe { from_raw_parts(batch.as_ptr() as *const u16, batch.len() / 2)};
                //     //  means this's bitmap
                //     let mut bitmap = [0; CHUNK_SIZE];
                //     for off in batch {
                //         bitmap[(*off >> 6) as usize] |= 1 << (*off % (1 << 6));
                //     }
                //     valid_batch[(write_pos * 8)..(write_pos * 8 + 8)].copy_from_slice(&bitmap);
                // }
    //             valid_batch[write_pos] = batch.len() as u64;
    //             write_pos = write_mask.trailing_zeros() as usize;
    //             write_mask = clear_lowest_set_bit(write_mask);
    //         }
    //         let batch = UInt64Array::from(valid_batch);
    //         // info!("fetch true count: {:}", as_boolean_array(&batch).true_count());
    //         batches.push(Arc::new(batch));
    //     }
    //     batches.insert(projected_schema.index_of("__id__").expect("Should have __id__ field"), Arc::new(UInt32Array::from(vec![] as Vec<u32>)));
        
    //     debug!("end of project fold");
    //     let option = RecordBatchOptions::new().with_row_count(Some(valid_batch_num * 8));
    //     Ok(RecordBatch::try_new_with_options(projected_schema, batches, &option)?)
    // }

    // pub fn project_with_predicate_with_score(
    //     &self,
    //     indices: &[Option<u32>],
    //     _projected_schema: SchemaRef,
    //     distris: &[Option<u64>],
    //     boundary_idx: usize,
    //     min_range: u64,
    //     predicate: &BooleanEvalExpr,
    //     _is_encoding: &Vec<bool>,
    // ) -> Result<usize> {
    //     let predicate = predicate.predicate.as_ref().unwrap().get();
    //     let predicate_ref = unsafe {predicate.as_ref().unwrap() };
    //     let valid_num = min_range.count_ones() as usize;
        
    //     let mut batches: Vec<Option<Vec<Chunk>>> = vec![None; indices.len()];
    //     let mut freqs: Vec<Option<Vec<Option<&[u8]>>>> = vec![None; indices.len()];
    //     debug!("start select valid batch");
    //     const EMPTY: [u64; 8] = [0; 8];
    //     for (j, index) in indices.iter().enumerate() {
    //         let distri = unsafe { distris.get_unchecked(j).unwrap_unchecked() };
    //         let mut write_mask = unsafe { _pext_u64(min_range, distri )};
    //         let valid_mask = unsafe { _pext_u64(distri, min_range) };
    //         let idx = unsafe { index.unwrap_unchecked() as usize };
    //         let (bucket, batch_freq) = if idx != usize::MAX {
    //             (self.postings.get(idx).cloned().ok_or_else(|| {
    //                 FastErr::InternalErr(format!(
    //                     "project index {} out of bounds, max fields {}",
    //                     idx,
    //                     self.postings.len(),
    //                 ))
    //             })?, self.term_freqs.as_ref().unwrap()[idx].clone())
    //         } else {
    //             continue;
    //         };
    //         let boundary = &self.boundary[idx];
    //         if boundary_idx >= boundary.len() {
    //             continue;
    //         }
    //         let start_idx = unsafe { *boundary.get_unchecked(boundary_idx) as usize };
    //         let mut posting = Vec::with_capacity(valid_num);
    //         let mut posting_ptr = posting.as_mut_ptr();

    //         let mut freqs_batch: Vec<Option<&[u8]>> = Vec::with_capacity(valid_num);

    //         for i in 0..valid_num {
    //             if valid_mask & (1 << i) != 0 {
    //                 let pos = write_mask.trailing_zeros() as usize;
    //                 write_mask = clear_lowest_set_bit(write_mask);
    //                 let batch = bucket.value(start_idx + pos);
    //                 let freq = batch_freq.value(start_idx + pos);
    //                 freqs_batch.push(Some(unsafe { from_raw_parts(freq.as_ptr(), freq.len())}));
    //                 if batch.len() == 64 {
    //                     let batch = unsafe { from_raw_parts(batch.as_ptr() as *const u64, 8) };
    //                     unsafe { store_advance_aligned(Chunk::Bitmap(batch), &mut posting_ptr) };
    //                 } else {
    //                     let batch = unsafe { from_raw_parts(batch.as_ptr() as *const u16, batch.len() / 2) };
    //                     // if !encoding {
    //                         unsafe { store_advance_aligned(Chunk::IDs(batch), &mut posting_ptr) };
    //                     // } else {
    //                     //     let mut bitmap: [u64; 8] = [0; 8];
    //                     //     for off in batch {
    //                     //         unsafe {
    //                     //             *bitmap.get_unchecked_mut((*off >> 6) as usize) |= 1 << (*off % (1 << 6));
    //                     //         }
    //                     //     }
    //                     //     // leak_pool.push(bitmap);
    //                     //     // let bitmap = Arc::new(bitmap);
    //                     //     std::mem::forget(bitmap);
    //                     //     unsafe { store_advance_aligned(Chunk::Bitmap(
    //                     //         from_raw_parts(bitmap.as_ptr() as *const u64, 8)
    //                     //         // from_raw_parts(leak_pool.last().unwrap().as_ptr() as *const u64, 8)
    //                     //     ), &mut posting_ptr) };
    //                     // }
    //                 }
    //             } else {
    //                 unsafe {
    //                     store_advance_aligned(Chunk::Bitmap(&EMPTY), &mut posting_ptr);  
    //                 }
    //                 freqs_batch.push(None);
    //             }
    //         }
    //         unsafe { set_vec_len_by_ptr(&mut posting, posting_ptr)}
    //         batches[j] = Some(posting);
    //         freqs[j] = Some(freqs_batch);
    //     }
        
    //     debug!("start eval");
    //     let eval = predicate_ref.eval_avx512(&batches, None, true, valid_num)?;
    //     let mut valid_freqs = Vec::new();
    //     debug!("end eval: {:}", eval.is_some());
    //     // batches.clear();
    //     if let Some(e) = eval {
    //         let mut accumulator = 0;
    //         let mut id_acc = 0;
    //         e.into_iter()
    //         .enumerate()
    //         .for_each(|(i, v)| unsafe {
    //             match  v {
    //                 TempChunk::Bitmap(b) => {
    //                     let popcnt = _mm512_popcnt_epi64(b);
    //                     let cnt = _mm512_reduce_add_epi64(popcnt);
    //                     if cnt != 0 {
    //                         valid_freqs.push(freqs_filter(&freqs, &U64x8 { vector: b }.vals, i));
    //                     }
    //                     accumulator += cnt;
    //                 }
    //                 TempChunk::IDs(i) => {
    //                     id_acc += i.len();
    //                 }
    //                 TempChunk::N0NE => {},
    //             }
    //         });
    //         debug!("sum: {:}", accumulator);
    //         // let sum = 0;
    //         Ok(accumulator as usize + valid_freqs.len())
    //     } else {
    //         Ok(0)
    //     }
    // }

    // pub fn project_with_predicate(
    //     &self,
    //     indices: &[Option<u32>],
    //     _projected_schema: SchemaRef,
    //     distris: &[Option<u64>],
    //     boundary_idx: usize,
    //     min_range: u64,
    //     predicate: &BooleanEvalExpr,
    //     _is_encoding: &Vec<bool>,
    // ) -> Result<usize> {
    //     let predicate = predicate.predicate.as_ref().unwrap().get();
    //     let predicate_ref = unsafe {predicate.as_ref().unwrap() };
    //     let valid_num = min_range.count_ones() as usize;
        
    //     let mut batches: Vec<Option<Vec<Chunk>>> = vec![None; indices.len()];
    //     debug!("start select valid batch");
    //     const EMPTY: [u64; 8] = [0; 8];
    //     for (j, index) in indices.iter().enumerate() {
    //         let distri = unsafe { distris.get_unchecked(j).unwrap_unchecked() };
    //         let mut write_mask = unsafe { _pext_u64(min_range, distri )};
    //         let valid_mask = unsafe { _pext_u64(distri, min_range) };
    //         let idx = unsafe { index.unwrap_unchecked() as usize };
    //         let (bucket, batch_freq) = if idx != usize::MAX {
    //             (self.postings.get(idx).cloned().ok_or_else(|| {
    //                 FastErr::InternalErr(format!(
    //                     "project index {} out of bounds, max fields {}",
    //                     idx,
    //                     self.postings.len(),
    //                 ))
    //             })?, self.term_freqs.as_ref().unwrap()[idx].clone())
    //         } else {
    //             continue;
    //         };
    //         let boundary = &self.boundary[idx];
    //         if boundary_idx >= boundary.len() {
    //             continue;
    //         }
    //         let start_idx = unsafe { *boundary.get_unchecked(boundary_idx) as usize };
    //         let mut posting = Vec::with_capacity(valid_num);
    //         let mut posting_ptr = posting.as_mut_ptr();


    //         for i in 0..valid_num {
    //             if valid_mask & (1 << i) != 0 {
    //                 let pos = write_mask.trailing_zeros() as usize;
    //                 write_mask = clear_lowest_set_bit(write_mask);
    //                 let batch = bucket.value(start_idx + pos);
    //                 if batch.len() == 64 {
    //                     let batch = unsafe { from_raw_parts(batch.as_ptr() as *const u64, 8) };
    //                     unsafe { store_advance_aligned(Chunk::Bitmap(batch), &mut posting_ptr) };
    //                 } else {
    //                     let batch = unsafe { from_raw_parts(batch.as_ptr() as *const u16, batch.len() / 2) };
    //                     // if !encoding {
    //                         unsafe { store_advance_aligned(Chunk::IDs(batch), &mut posting_ptr) };
    //                     // } else {
    //                     //     let mut bitmap: [u64; 8] = [0; 8];
    //                     //     for off in batch {
    //                     //         unsafe {
    //                     //             *bitmap.get_unchecked_mut((*off >> 6) as usize) |= 1 << (*off % (1 << 6));
    //                     //         }
    //                     //     }
    //                     //     // leak_pool.push(bitmap);
    //                     //     // let bitmap = Arc::new(bitmap);
    //                     //     std::mem::forget(bitmap);
    //                     //     unsafe { store_advance_aligned(Chunk::Bitmap(
    //                     //         from_raw_parts(bitmap.as_ptr() as *const u64, 8)
    //                     //         // from_raw_parts(leak_pool.last().unwrap().as_ptr() as *const u64, 8)
    //                     //     ), &mut posting_ptr) };
    //                     // }
    //                 }
    //             } else {
    //                 unsafe {
    //                     store_advance_aligned(Chunk::Bitmap(&EMPTY), &mut posting_ptr);  
    //                 }
    //             }
    //         }
    //         unsafe { set_vec_len_by_ptr(&mut posting, posting_ptr)}
    //         batches[j] = Some(posting);
    //     }
        
    //     debug!("start eval");
    //     let eval = predicate_ref.eval_avx512(&batches, None, true, valid_num)?;
    //     debug!("end eval: {:}", eval.is_some());
    //     // batches.clear();
    //     if let Some(e) = eval {
    //         let mut accumulator = 0;
    //         let mut id_acc = 0;
    //         e.into_iter()
    //         .enumerate()
    //         .for_each(|(i, v)| unsafe {
    //             match  v {
    //                 TempChunk::Bitmap(b) => {
    //                     let popcnt = _mm512_popcnt_epi64(b);
    //                     let cnt = _mm512_reduce_add_epi64(popcnt);
    //                     accumulator += cnt;
    //                 }
    //                 TempChunk::IDs(i) => {
    //                     id_acc += i.len();
    //                 }
    //                 TempChunk::N0NE => {},
    //             }
    //         });
    //         debug!("sum: {:}", accumulator);
    //         // let sum = 0;
    //         Ok(accumulator as usize)
    //     } else {
    //         Ok(0)
    //     }
    // }
    // #[inline]
    // fn process_dense_bucket(&self, indices: &[Option<u32>], distris: &[Option<u64>], min_range: u64, boundary_idx: usize, valid_num: usize) -> Result<Vec<Option<Vec<Chunk>>>> {
    //     let mut batches: Vec<Option<Vec<Chunk>>> = vec![None; indices.len()];
    //     let compress_index = unsafe {
    //         _mm512_loadu_epi8(COMPRESS_INDEX.as_ptr() as *const i8)
    //     };
    //     for (j, index) in indices.iter().enumerate() {
    //         let distri = unsafe { distris.get_unchecked(j).unwrap_unchecked() };
    //         let write_mask = unsafe { _pext_u64(min_range, distri ) };
    //         let valid_mask = unsafe { _pext_u64(distri, min_range) };

    //         let idx = unsafe { index.unwrap_unchecked() as usize };

    //         let bucket = if idx != usize::MAX {
    //             self.postings.get(idx).cloned().ok_or_else(|| {
    //                 FastErr::InternalErr(format!(
    //                     "project index {} out of bounds, max field {}",
    //                     idx,
    //                     self.postings.len()
    //                 ))
    //             })?
    //         } else {
    //             continue;
    //         };

    //         let boundary = &self.boundary[idx];
    //         if boundary_idx >= boundary.len() {
    //             continue;
    //         }
    //         let start_idx = boundary[boundary_idx] as usize;
    //         let mut posting = vec![Chunk::N0NE; valid_num];


    //         let mut write_index: Vec<u8> = vec![0; valid_num];
    //         let mut posting_index: Vec<u8> = vec![0; valid_num];
    //         unsafe {
    //             _mm512_mask_compressstoreu_epi8(write_index.as_mut_ptr() as *mut u8, valid_mask, compress_index);
    //             _mm512_mask_compressstoreu_epi8(posting_index.as_mut_ptr() as *mut u8, write_mask, compress_index);
    //         }

    //         for (w, p) in write_index.into_iter().zip(posting_index.into_iter()){
    //             let batch = bucket.value(start_idx + p as usize);
    //             if batch.len() == 64 {
    //                 posting[w as usize] = Chunk::Bitmap(batch);
    //             } else {
    //                 // means this's Integer list
    //                 let batch = unsafe {batch.align_to::<u16>().1};
    //                 posting[w as usize] = Chunk::IDs(batch);
    //             }
    //         }
    //         batches[j] = Some(posting);
    //     }
    //     Ok(batches)
    // }

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
        space += size_of_val(self.schema.as_ref());
        space += self.postings
            .iter()
            .map(|v| size_of_val(v.data().buffers()[0].as_slice()))
            .sum::<usize>();
        space
    }

    pub fn schema(&self) -> TermSchemaRef {
        self.schema.clone()
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
    current: u32,
    pub term_dict: RefCell<BTreeMap<String, Vec<(u32, u8)>>>,
    term_num: usize,
}

impl PostingBatchBuilder {
    pub fn new() -> Self {
        Self { 
            current: 0,
            term_dict: RefCell::new(BTreeMap::new()),
            term_num: 0,
        }
    }

    pub fn doc_len(&self) -> usize {
        self.current as usize
    }

    pub fn push_term(&mut self, term: String, doc_id: u32) -> Result<()> {
        let off = doc_id;
        let entry = self.term_dict
            .get_mut()
            .entry(term)
            .or_insert(vec![(off, 0)]);
        if entry.last().unwrap().0 ==  off {
            if entry.last_mut().unwrap().1 != u8::MAX {
                entry.last_mut().unwrap().1 += 1;
            }
        } else {
            entry.push((off, 1));
        }
        self.current = doc_id;
        self.term_num += 1;
        Ok(())
    }

    pub fn build(self) -> Result<PostingBatch> {
        self.build_with_idx(None, 0)
    }

    pub fn build_with_idx(self, idx: Option<&RefCell<BTreeMap<String, TermMetaBuilder>>>, partition_num: usize) -> Result<PostingBatch> {
        let term_dict = self.term_dict
            .into_inner();
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
                if idx.is_some() {
                    idx.as_ref().unwrap().borrow_mut().get_mut(&k).unwrap().add_idx(i as u32, partition_num);
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
        PostingBatch::try_new_with_freqs(
            Arc::new(Schema::new(schema_list)),
            postings,
            freqs,
            vec![],
            Arc::new(BatchRange::new(0, 512))
        )
    }

    pub fn build_mmap_segment(self, _partition_num: usize) -> Result<PostingSegment> {
        let term_dict = self.term_dict
            .into_inner();
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

// #[inline]
// fn clear_lowest_set_bit(v: u64) -> u64 {
//     unsafe { _blsr_u64(v) }
// }


#[derive(Clone, Serialize, Deserialize)]
pub struct TermMetaBuilder {
    pub distribution: Vec<Vec<u64>>,
    pub valid_bitmap: Vec<RoaringBitmap>,
    pub nums: Vec<u32>,
    idx: Vec<Option<u32>>,
    partition_num: usize,
    bounder: Option<u32>,
}

impl TermMetaBuilder {
    pub fn new(_batch_num: usize, partition_num: usize) -> Self {
        Self {
            distribution: vec![],
            valid_bitmap: vec![RoaringBitmap::new(); partition_num],
            nums: vec![0; partition_num],
            idx: vec![None; partition_num],
            partition_num,
            bounder: None,
        }
    }

    pub fn set_true(&mut self, i: usize, partition_num: usize, id: u32) {
        if self.bounder.is_none() || id > self.bounder.unwrap() {
            self.nums[partition_num] += 1;
            self.bounder = Some(id);
        }
        // Add doc_id to valid_bitmap
        self.valid_bitmap[partition_num].insert(i as u32);
    }

    pub fn add_idx(&mut self, idx: u32, partition_num: usize) {
        self.idx[partition_num] = Some(idx);
    }

    pub fn rle_usage(&self) -> usize {
        let mut builder = PrimitiveRunBuilder::<Int64Type, Int64Type>::with_capacity(self.distribution[0].len());
        for i in &self.distribution[0] {
            builder.append_value(*i as i64);
        }
        let array: Int64RunArray = builder.finish();
        array.get_array_memory_size()
    }

    pub fn build(self) -> TermMeta {
        let valid_bitmap: Vec<Arc<RoaringBitmap>> = self.valid_bitmap
            .into_iter()
            .map(Arc::new)
            .collect();
        let valid_batch_num: usize = valid_bitmap.iter()
            .map(|d| d.len() as usize)
            .sum();
        let sel = self.nums.iter().map(|v| *v).sum::<u32>() as f64 / (512. * valid_batch_num as f64);
        TermMeta {
            valid_bitmap: Arc::new(valid_bitmap),
            index: Arc::new(self.idx),
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
mod test {
    // use std::sync::Arc;

    // use datafusion::{arrow::{datatypes::{Schema, Field, DataType, UInt8Type}, array::{UInt16Array, BooleanArray, GenericListArray, ArrayRef}}, from_slice::FromSlice};

    // use super::{BatchRange, PostingBatch};


    #[test]
    fn test_ptr() {
        // let val: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        // println!("{:?}", ptr::metadata(val.as_slic8e() as *const [u64]));
        println!("{:}", std::mem::size_of::<Option<&[u8]>>())
    }

    // fn build_batch() -> PostingBatch {
    //     let schema = Arc::new(Schema::new(vec![
    //         Field::new("test1", DataType::Boolean, true),
    //         Field::new("test2", DataType::Boolean, true),
    //         Field::new("test3", DataType::Boolean, true),
    //         Field::new("test4", DataType::Boolean, true),
    //         Field::new("__id__", DataType::UInt32, false),
    //     ]));
    //     let range = Arc::new(BatchRange {
    //         start: 0,
    //         end: 64,
    //         nums32: 1
    //     });
    //     let postings: Vec<ArrayRef> = vec![
    //        Arc::new(UInt16Array::from_slice([1, 6, 9])),
    //        Arc::new(UInt16Array::from_slice([0, 4, 16])),
    //        Arc::new(UInt16Array::from_slice([4, 6, 8])),
    //        Arc::new(UInt16Array::from_slice([6, 16, 31])),
    //        Arc::new(UInt16Array::from_slice([])),
    //     ];
    //     PostingBatch::try_new(schema, postings, range).unwrap()
    // }

    // fn build_batch_with_freqs() -> PostingBatch {
    //     let schema = Arc::new(Schema::new(vec![
    //         Field::new("test1", DataType::Boolean, true),
    //         Field::new("test2", DataType::Boolean, true),
    //         Field::new("test3", DataType::Boolean, true),
    //         Field::new("test4", DataType::Boolean, true),
    //         Field::new("__id__", DataType::UInt32, false),
    //     ]));
    //     let range = Arc::new(BatchRange {
    //         start: 0,
    //         end: 64,
    //         nums32: 1
    //     });
    //     let postings: Vec<ArrayRef> = vec![
    //        Arc::new(UInt16Array::from_slice([1, 6, 9])),
    //        Arc::new(UInt16Array::from_slice([0, 4, 16])),
    //        Arc::new(UInt16Array::from_slice([4, 6, 8])),
    //        Arc::new(UInt16Array::from_slice([6, 16, 31])),
    //        Arc::new(UInt16Array::from_slice([])),
    //     ];
    //     let freqs = vec![
    //         Arc::new(GenericListArray::from_iter_primitive::<UInt8Type, _, _>(vec![
    //             Some(vec![Some(1 as u8), Some(2), Some(3)]),
    //             None,
    //             None,
    //             Some(vec![Some(22), Some(2), Some(2)]),
    //         ])),
    //         Arc::new(GenericListArray::from_iter_primitive::<UInt8Type, _, _>(vec![
    //             Some(vec![Some(2), Some(1)]),
    //             None,
    //             Some(vec![Some(1), Some(2)]),
    //             None,
    //         ])),
    //         Arc::new(GenericListArray::from_iter_primitive::<UInt8Type, _, _>(vec![
    //             Some(vec![Some(3), Some(1), Some(1)]),
    //             None,
    //             None,
    //             None,
    //         ])),
    //         Arc::new(GenericListArray::from_iter_primitive::<UInt8Type, _, _>(vec![
    //             None,
    //             Some(vec![Some(1), Some(2), Some(2)]),
    //             None,
    //             None,
    //         ])),
    //         Arc::new(GenericListArray::from_iter_primitive::<UInt8Type, _, _>(vec![
    //             None as Option<Vec<Option<u8>>>,
    //             None,
    //             None,
    //             None,
    //         ])),
    //     ];
    //     PostingBatch::try_new_with_freqs(schema, postings, freqs, range).unwrap()
    // }

    // #[test]
    // fn postingbatch_project_fold() {
    //     let schema = Arc::new(Schema::new(vec![
    //         Field::new("test1", DataType::Boolean, false),
    //         Field::new("test2", DataType::Boolean, false),
    //         Field::new("test3", DataType::Boolean, false),
    //         Field::new("test4", DataType::Boolean, false),
    //         Field::new("__id__", DataType::UInt32, false),
    //     ]));
    //     let batch = build_batch();
    //     let res = batch.project_fold(&[Some(1), Some(2)], Arc::new(schema.clone().project(&[1, 2, 4]).unwrap())).unwrap();
    //     let mut exptected1 = vec![false; 64];
    //     exptected1[0] = true;
    //     exptected1[4] = true;
    //     exptected1[16] = true;
    //     let exptected1 = BooleanArray::from(exptected1);
    //     let mut exptected2 = vec![false; 64];
    //     exptected2[4] = true;
    //     exptected2[6] = true;
    //     exptected2[8] = true;
    //     let exptected2 = BooleanArray::from(exptected2);

    //     assert_eq!(res.column(0).as_any().downcast_ref::<BooleanArray>().unwrap(), &exptected1);
    //     assert_eq!(res.column(1).as_any().downcast_ref::<BooleanArray>().unwrap(), &exptected2);
    // }

    // #[test]
    // fn postingbatch_project_fold_with_freqs() {
    //     let schema = Arc::new(Schema::new(vec![
    //         Field::new("test1", DataType::Boolean, false),
    //         Field::new("test2", DataType::Boolean, false),
    //         Field::new("test3", DataType::Boolean, false),
    //         Field::new("test4", DataType::Boolean, false),
    //         Field::new("__id__", DataType::UInt32, false),
    //     ]));
    //     let batch = build_batch_with_freqs();
    //     let res = batch.project_fold_with_freqs(&[Some(1), Some(2)], Arc::new(schema.clone().project(&[1, 2, 4]).unwrap())).unwrap();
    //     let mut exptected1 = vec![false; 64];
    //     exptected1[0] = true;
    //     exptected1[4] = true;
    //     exptected1[16] = true;
    //     let exptected1 = BooleanArray::from(exptected1);
    //     let mut exptected2 = vec![false; 64];
    //     exptected2[4] = true;
    //     exptected2[6] = true;
    //     exptected2[8] = true;
    //     let exptected2 = BooleanArray::from(exptected2);
    //     println!("buffer: {:?}", res.column(0).data().buffers()[0].as_slice());
    //     println!("res: {:?}", res);
    //     assert_eq!(res.column(0).as_any().downcast_ref::<BooleanArray>().unwrap(), &exptected1);
    //     assert_eq!(res.column(1).as_any().downcast_ref::<BooleanArray>().unwrap(), &exptected2);
    // }
}
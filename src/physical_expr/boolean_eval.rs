use std::{any::Any, arch::x86_64::{__m256i, __m512i, _mm256_load_si256, _mm256_loadu_epi16, _mm512_and_epi64, _mm512_load_epi64, _mm512_loadu_epi64, _mm512_loadu_epi8, _mm512_mask_compressstoreu_epi8, _mm512_or_epi64, _mm_cmpestrm, _mm_extract_epi32, _mm_loadu_epi16, _mm_mask_compressstoreu_epi16, _SIDD_BIT_MASK, _SIDD_CMP_EQUAL_ANY, _SIDD_UWORD_OPS}, cell::SyncUnsafeCell, slice::from_raw_parts, sync::Arc};

use dashmap::DashSet;
use datafusion::{physical_plan::{expressions::{BinaryExpr, Column}, PhysicalExpr, ColumnarValue}, arrow::{datatypes::{Schema, DataType}, record_batch::RecordBatch, array::{UInt64Array, Int8Array, UInt8Array, Array}}, error::DataFusionError, common::{Result, cast::as_uint64_array}};
use roaring::RoaringBitmap;
use sorted_iter::{assume::AssumeSortedByItemExt, SortedIterator};
use tracing::{debug, info};

use crate::{batch::{BatchFreqs, Freqs}, utils::avx512::{bitwise_and, bitwise_and_batch, bitwise_or, bitwise_or_batch, U64x8}, ShortCircuit};

#[derive(Clone, Debug)]
pub enum Primitives {
    BitwisePrimitive(BinaryExpr),
    ShortCircuitPrimitive(ShortCircuit),
    ColumnPrimitive(Column),
}

#[derive(Clone, Debug)]
pub enum Chunk<'a> {
    Bitmap(&'a [u64]),
    IDs(&'a [u16]),
    N0NE,
}

#[derive(Clone, Debug)]
pub enum TempChunk {
    Bitmap(__m512i),
    IDs(Vec<u16>),
    N0NE,
}

impl Primitives {
    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        match self {
            Primitives::BitwisePrimitive(b) => b.evaluate(batch),
            Primitives::ShortCircuitPrimitive(s) => s.evaluate(batch),
            Primitives::ColumnPrimitive(c) => c.evaluate(batch),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubPredicate {
    /// The children nodes of physical predicate
    pub sub_predicate: PhysicalPredicate,
    /// Node num
    pub node_num: usize,
    /// Leaf num
    pub leaf_num: usize,
    /// The selectivity of this node
    pub selectivity: f64,
    /// The rank of this node, calculate by $$\frac{sel - 1}{leaf_num}$$
    pub rank: f64,
    /// The cumulative selectivity
    pub cs: f64,
}

impl SubPredicate {
    pub fn new(
        sub_predicate: PhysicalPredicate,
        node_num: usize,
        leaf_num: usize,
        selectivity: f64,
        rank: f64,
        cs: f64,
    ) -> Self {
        Self {
            sub_predicate,
            node_num,
            leaf_num,
            selectivity,
            rank,
            cs,
        }
    }

    pub fn new_with_predicate(
        sub_predicate: PhysicalPredicate
    ) -> Self {
        Self::new(sub_predicate, 0, 0, 0., 0., 0.)
    }

    pub fn node_num(&self) -> usize {
        self.node_num
    }

    pub fn leaf_num(&self) -> usize {
        self.leaf_num
    }

    pub fn sel(&self) -> f64 {
        self.selectivity
    }

    pub fn rank(&self) -> f64 {
        self.rank
    }
}

#[derive(Clone, Debug)]
pub enum PhysicalPredicate {
    /// `And` Level
    And { args: Vec<SubPredicate> },
    /// `Or` Level
    Or { args: Vec<SubPredicate> },
    /// Leaf
    Leaf { primitive: Primitives },
}

impl PhysicalPredicate {
    pub fn eval_avx512(
        &self,
        batch: &Vec<Option<Vec<Chunk>>>,
        init_v: Option<Vec<TempChunk>>,
        is_and: bool,
        valid_num: usize,
    ) -> Result<Option<Vec<TempChunk>>> {
        let mut init_v = init_v;
        match self {
            PhysicalPredicate::And { args } => {
                for predicate in args {
                    init_v = predicate.sub_predicate.eval_avx512(&batch, init_v, true, valid_num)?;
                }
                Ok(init_v)
            }
            PhysicalPredicate::Or { args } => {
                for predicate in args {
                    init_v = predicate.sub_predicate.eval_avx512(&batch, init_v, false, valid_num)?;
                }
                Ok(init_v)
            }
            PhysicalPredicate::Leaf { primitive } => {
                match primitive {
                    Primitives::BitwisePrimitive(_b) => {
                        todo!()
                    }
                    Primitives::ShortCircuitPrimitive(s) => {
                        if is_and {
                            Ok(Some(s.eval_avx512(init_v, &batch, valid_num)))
                        } else {
                            let eval = s.eval_avx512(None, &batch, valid_num);
                            match init_v {
                                Some(mut e) => {
                                    e.iter_mut()
                                    .zip(eval.iter())
                                    .for_each(|(a, b)| { 
                                        match (&a, b) {
                                            (TempChunk::Bitmap(a_bm), TempChunk::Bitmap(b)) => {
                                                *a = TempChunk::Bitmap(unsafe {
                                                    _mm512_or_epi64(*a_bm, *b)
                                                })
                                            }
                                            (TempChunk::IDs(ids), TempChunk::Bitmap(b)) => {
                                                let mut chunk = unsafe {U64x8{vector: *b}.vals};
                                                for id in ids {
                                                    chunk[(*id as usize) >> 8] |= 1 << ((*id as usize) % 64);
                                                }
                                                *a = TempChunk::Bitmap(unsafe {
                                                    U64x8{ vals: chunk }.vector
                                                })
                                            }
                                            (_, TempChunk::N0NE) => {}
                                            (_, _) => unreachable!(),
                                        }
                                    });
                                }
                                None => {}
                            }
                            Ok(Some(eval))
                        }
                    }
                    Primitives::ColumnPrimitive(c) => {
                        if is_and {
                            let eval_array = &batch[c.index()];
                            match (&mut init_v, eval_array) {
                                (Some(e), Some(i)) => {
                                    for (a, b) in e.iter_mut().zip(i.into_iter()) {
                                        match (&a, b) {
                                            (TempChunk::Bitmap(a_bm), Chunk::Bitmap(b_bm)) => {
                                                if !cfg!(feature = "scalar") {
                                                    let b_bm = unsafe { _mm512_loadu_epi64((*b_bm).as_ptr() as *const i64)};
                                                    *a = TempChunk::Bitmap(unsafe { _mm512_and_epi64(*a_bm, b_bm) })
                                                } else {
                                                    *a = TempChunk::Bitmap(bitwise_and_batch(unsafe { &U64x8 { vector: *a_bm }.vals }, &b_bm))
                                                }
                                            }
                                            (TempChunk::Bitmap(a_bm), Chunk::IDs(b)) => {
                                                let mut b_bitmap: [u64; 8] = [0; 8];

                                                for id in *b {
                                                    b_bitmap[(*id as usize) >> 8] |= 1 << ((*id as usize) % 64);
                                                }
                                                if cfg!(feature="scalar") {
                                                    *a = TempChunk::Bitmap(bitwise_and_batch(unsafe {&U64x8 {vector: *a_bm}.vals}, &b_bitmap));
                                                } else {
                                                    let b_bitmap = unsafe { _mm512_loadu_epi64(b_bitmap.as_ptr() as *const i64)};
                                                    *a = TempChunk::Bitmap(unsafe { _mm512_and_epi64(*a_bm, b_bitmap)});
                                                }
                                            }
                                            (TempChunk::IDs(a_id), Chunk::Bitmap(b_bm)) => {
                                                let mut a_bm: [u64; 8] = [0; 8];

                                                for &id in a_id {
                                                    a_bm[id as usize >> 8] |= 1 << (id as usize % 64);
                                                }
                                                *a = TempChunk::Bitmap(bitwise_and_batch(&a_bm, &b_bm));
                                            }
                                            (TempChunk::IDs(a_id), Chunk::IDs(b_id)) => {
                                                *a = TempChunk::IDs(intersect_avx(&a_id, &b_id));
                                            }
                                            (_, Chunk::N0NE) => {
                                                *a = TempChunk::N0NE
                                            }
                                            (TempChunk::N0NE, _) => {
                                                *a = TempChunk::N0NE
                                            }
                                            _ => unreachable!(),
                                        }
                                    }
                                }
                                (_, Some(i)) => {
                                    init_v = Some(
                                        i.into_iter()
                                        .map(|a| match a {
                                                Chunk::Bitmap(b) => {
                                                    if cfg!(feature="scalar") {
                                                        TempChunk::Bitmap(unsafe {std::ptr::read_unaligned((*b).as_ptr() as *const __m512i)})
                                                    } else {
                                                        TempChunk::Bitmap(unsafe { _mm512_loadu_epi64((*b).as_ptr() as *const i64)})
                                                    }
                                                }
                                                Chunk::IDs(ids) => {
                                                    TempChunk::IDs(ids.to_vec())
                                                }
                                                Chunk::N0NE => TempChunk::N0NE
                                        })
                                        .collect()
                                );},
                                (_, _) => {}
                            }
                        } else {
                            let eval_array = &batch[c.index()];
                            match (eval_array, &mut init_v) {
                                (Some(e), Some(i)) => {
                                    for (a, b) in i.iter_mut().zip(e.into_iter()) {
                                        match (b, &a) {
                                            (Chunk::Bitmap(a_bm), TempChunk::Bitmap(b_bm)) => {
                                                let a_bm = load_u64_slice(a_bm);
                                                *a = TempChunk::Bitmap(unsafe { _mm512_or_epi64(a_bm, *b_bm) })
                                                
                                            }
                                            (Chunk::Bitmap(a_bm), TempChunk::IDs(b_id)) => {
                                                let mut bitmap = (*a_bm).to_owned();
                                                for id in b_id {
                                                    bitmap[(*id as usize) >> 8] |= 1 << ((*id as usize) % 64);
                                                }
                                                *a = TempChunk::Bitmap(unsafe { _mm512_loadu_epi64(bitmap.as_ptr() as *const i64)})
                                            }
                                            (Chunk::IDs(a_id), TempChunk::Bitmap(b_bm)) => {
                                                let mut b = unsafe { U64x8{ vector: *b_bm }.vals };
                                                for &id in *a_id {
                                                    b[(id as usize) >> 8] |= 1 << ((id as usize) % 64);
                                                }
                                                *a = TempChunk::Bitmap(unsafe { U64x8 { vals: b }.vector })
                                            }
                                            (Chunk::IDs(a_id), TempChunk::IDs(b_id)) => {
                                                let a_id = a_id.into_iter().cloned().assume_sorted_by_item();
                                                let b_id = b_id.into_iter().cloned().assume_sorted_by_item();
                                                *a = TempChunk::IDs(b_id.intersection(a_id).collect())
                                            }
                                            (Chunk::Bitmap(a_bm), TempChunk::N0NE) => {
                                                *a = TempChunk::Bitmap(load_u64_slice(a_bm))
                                            }
                                            (Chunk::IDs(a_ids), TempChunk::N0NE) => {
                                                *a = TempChunk::IDs(a_ids.to_vec())
                                            }
                                            _ => *a = TempChunk::N0NE,
                                        }
                                    }
                                }
                                (Some(eval), None) => {
                                    let mut init: Vec<TempChunk> = Vec::with_capacity(eval.len());
                                    for e in eval {
                                        match e {
                                            Chunk::Bitmap(b) => {
                                                init.push(TempChunk::Bitmap(load_u64_slice(b)));
                                            }
                                            Chunk::IDs(ids) => {
                                                init.push(TempChunk::IDs(ids.to_vec()));
                                            }
                                            Chunk::N0NE => {
                                                init.push(TempChunk::N0NE);
                                            }
                                        }
                                    }
                                    init_v = Some(init);
                                }
                                (_, _) => {}
                            }
                        }
                        Ok(init_v)
                    }
                }
            }
        }

    }
    fn eval(&self, batch: &RecordBatch, init_v: Vec<u64>, is_and: bool) -> Result<Vec<u64>> {
        let mut init_v = init_v;
        match self {
            PhysicalPredicate::And { args } => {
                for predicate in args {
                    init_v = predicate.sub_predicate.eval(batch, init_v, true)?;
                }
                Ok(init_v)
            }
            PhysicalPredicate::Or { args } => {
                for predicate in args {
                    init_v = predicate.sub_predicate.eval(batch, init_v, false)?;
                }
                Ok(init_v)
            }
            PhysicalPredicate::Leaf { primitive } => {
                match  primitive {
                    Primitives::BitwisePrimitive(b) => {
                        let tmp = b.evaluate(batch)?.into_array(0);
                        let res = as_uint64_array(&tmp).unwrap();
                        if is_and {
                            if res.len() == 0 {
                                init_v.fill(0);
                                Ok(init_v)
                            } else {
                                Ok(init_v.into_iter()
                                .zip(res.into_iter())
                                .map(|(i, j)| i & unsafe { j.unwrap_unchecked() })
                                .collect())
                            }
                        } else {
                            if res.len() == 0 {
                                Ok(init_v)
                            } else {
                                Ok(init_v.into_iter()
                                .zip(res.into_iter())
                                .map(|(i, j)| i | unsafe { j.unwrap_unchecked() })
                                .collect())
                            }
                        }
                    }
                    Primitives::ShortCircuitPrimitive(s) => {
                        if is_and {
                            unsafe { s.eval(init_v.align_to_mut().1, batch).align_to::<u64>() };
                            Ok(init_v)
                        } else {
                            let mut init_v_or = vec![u8::MAX; init_v.len() * 8];
                            s.eval(&mut init_v_or, batch);
                            let init_v_or = unsafe { init_v_or.align_to::<u64>().1 };
                            for i in 0..init_v.len(){
                                init_v[i] |= init_v_or[i];
                            }
                            Ok(init_v)
                        }
                    }
                    Primitives::ColumnPrimitive(c) => {
                        if is_and {
                            let eval_array = batch.column(c.index());
                            let eval_res = as_uint64_array(&eval_array).unwrap();
                            if eval_res.len() == 0 {
                                init_v.fill(0);
                                Ok(init_v)
                            } else {
                                assert_eq!(init_v.len(), eval_res.len(), "evalue res: {:?}", eval_res);
                                bitwise_and(&mut init_v, eval_res.values());
                                Ok(init_v)
                            }
                        } else {
                            let eval_array = batch.column(c.index());
                            let eval_res = as_uint64_array(&eval_array).unwrap();
                            if eval_res.len() == 0 {
                                Ok(init_v)
                            } else {
                                bitwise_or(&mut init_v, eval_res.values());
                                Ok(init_v)
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn eval_roaring(&self, roarings: &[Arc<RoaringBitmap>]) -> Result<Arc<RoaringBitmap>> {
        match self {
            PhysicalPredicate::And { args } => {
                Ok(
                    args.iter()
                    .map(|v| v.sub_predicate.eval_roaring(&roarings).unwrap())
                    .reduce(|acc, e| {
                        Arc::new(acc.as_ref() & e.as_ref())
                    })
                    .unwrap()
                )
            },
            PhysicalPredicate::Or { args } => {
                Ok(args.iter()
                .map(|v| v.sub_predicate.eval_roaring(&roarings).unwrap())
                .reduce(|acc, e| {
                    Arc::new(acc.as_ref() | e.as_ref())
                })
                .unwrap())
            },
            PhysicalPredicate::Leaf { primitive } => {
                match primitive {
                    Primitives::ColumnPrimitive(c) => {
                        Ok(roarings[c.index()].clone())
                    }
                    _ => {unreachable!("When evaluate valid bitmap, only ColumnPrimitive branch can be reached")}
                }
            }
        }
    }

    // fn eval_tmp(&self, batch: &Vec<Option<Vec<Chunk>>>) -> Result<Option<Arc<Vec<TempChunk>>>> {
    //     debug!("evaluate batch len: {:}", batch.len());

    //     match self {
    //         PhysicalPredicate::And { args } => {
    //             let res = args.into_iter()
    //                 .map(|e| e.sub_predicate.eval_tmp(batch).unwrap())
    //                 .reduce(|acc, e| {
    //                     let (acc, e) = match (acc, e) {
    //                         (Some(acc), Some(e)) => (acc, e),
    //                         (_, _) => return None,
    //                     };
    //                     let acc = acc.into_iter()
    //                         .zip(e.into_iter())
    //                         .map(|(a, b)| {
    //                             let chunk = match (a, b) {
    //                                     (TempChunk::Bitmap(a), TempChunk::Bitmap(b)) => {
    //                                         TempChunk::Bitmap(unsafe { _mm512_and_epi64(a, b)})
    //                                     }
    //                                     (TempChunk::Bitmap(ab), TempChunk::IDs(bi)) => {
    //                                         let mut bb: [u64; 8] = [0; 8];
    //                                         for b in bi {
    //                                             bb[(b as usize) >> 8] |= 1 << ((b as usize) % 64);
    //                                         }
    //                                         let bb = unsafe { _mm512_loadu_epi64(bb.as_ptr() as *const i64) };
    //                                         TempChunk::Bitmap(unsafe { _mm512_and_epi64(ab, bb)})
    //                                     }
    //                                     (TempChunk::Bitmap(_), TempChunk::N0NE) => TempChunk::N0NE,
    //                                     (TempChunk::IDs(ai), TempChunk::Bitmap(bb)) => {
    //                                         let mut ab: [u64; 8] = [0; 8];
    //                                         for a in ai {
    //                                             ab[(a as usize) >> 8] |= 1 << ((a as usize) % 64);
    //                                         }
    //                                         let ab = unsafe { _mm512_load_epi64(ab.as_ptr() as *const i64)};
    //                                         TempChunk::Bitmap(unsafe { _mm512_and_epi64(ab, bb) })
    //                                     }
    //                                     (TempChunk::IDs(ai), TempChunk::IDs(bi)) => {
    //                                         TempChunk::IDs(intersect_avx(&ai, &bi))
    //                                     }
    //                                     (TempChunk::IDs(_), TempChunk::N0NE) => TempChunk::N0NE,
    //                                     (TempChunk::N0NE, TempChunk::Bitmap(_)) => TempChunk::N0NE,
    //                                     (TempChunk::N0NE, TempChunk::IDs(_)) => TempChunk::N0NE,
    //                                     (TempChunk::N0NE, TempChunk::N0NE) => TempChunk::N0NE,
    //                                 };
    //                             chunk
    //                         })
    //                         .collect();
    //                     Some(Arc::new(acc))
    //                 })
    //                 .unwrap();
    //             Ok(res)
    //         }
    //         PhysicalPredicate::Or { args } => {
    //             let res = args.iter()
    //                 .map(|e| e.sub_predicate.eval_tmp(&batch).unwrap())
    //                 .reduce(|acc, e| {
    //                     let (acc, e) = match (acc, e) {
    //                         (Some(acc), Some(e)) => {
    //                             (acc, e)
    //                         }
    //                         (Some(acc), None) => return Some(acc),
    //                         (None, e) => return e,
    //                     };
    //                     let acc = acc.iter()
    //                         .zip(e.iter())
    //                         .map(|(a, b)| {
    //                             match (a, b) {
    //                                 (TempChunk::Bitmap(ab), TempChunk::Bitmap(bb)) => {
    //                                     TempChunk::Bitmap(unsafe { _mm512_or_epi64(*ab, *bb)})
    //                                 }
    //                                 (TempChunk::Bitmap(ab), TempChunk::IDs(bi)) => {
    //                                     let mut ab = unsafe { U64x8 {vector: *ab }.vals };
    //                                     for &b in bi {
    //                                         ab[b as usize >> 8] |= 1 << (b as usize % 64);
    //                                     }
    //                                     let ab = unsafe { _mm512_load_epi64(ab.as_ptr() as *const i64)};
    //                                     TempChunk::Bitmap(ab)
    //                                 }
    //                                 (a, TempChunk::N0NE) => a.clone(),
    //                                 (TempChunk::IDs(ai), TempChunk::Bitmap(bb)) => {
    //                                     let mut bb = unsafe { U64x8 { vector: *bb}.vals };
    //                                     for &a in ai {
    //                                         bb[a as usize >> 8] |= 1 << (a as usize % 64);
    //                                     }
    //                                     let bb = unsafe { _mm512_load_epi64(bb.as_ptr() as *const i64) };
    //                                     TempChunk::Bitmap(bb)
    //                                 }
    //                                 (TempChunk::IDs(ai), TempChunk::IDs(bi)) => {
    //                                     let a_iter = ai.into_iter().assume_sorted_by_item();
    //                                     let b_iter = bi.into_iter().assume_sorted_by_item();
    //                                     let union_res: Vec<u16> = a_iter.union(b_iter).cloned().collect();
    //                                     TempChunk::IDs(union_res)
    //                                 }
    //                                 (TempChunk::N0NE, b) => b.clone(),
    //                             }
    //                         })
    //                         .collect();
    //                     Some(Arc::new(acc))
    //                 })
    //                 .unwrap();
    //             Ok(res)
    //         }
    //         PhysicalPredicate::Leaf { primitive } => {
    //             match primitive {
    //                 Primitives::BitwisePrimitive(_) => todo!(),
    //                 Primitives::ShortCircuitPrimitive(_) => todo!(),
    //                 Primitives::ColumnPrimitive(colum) => {
    //                     let chunk = batch[colum.index()];
    //                     match chunk {
    //                         Some(c) => {
    //                             match c {

    //                             }
    //                         }
    //                         None => Ok(None)
    //                     }
    //                 }
    //             }
    //         }
    //     }
    
    // }
}

/// Combined Primitives Expression
#[derive(Debug, Clone)]
pub struct BooleanEvalExpr {
    pub predicate: Option<Arc<SyncUnsafeCell<PhysicalPredicate>>>,
    pub valid_idx: Arc<DashSet<usize>>,
}

impl BooleanEvalExpr {
    pub fn new(predicate: Option<PhysicalPredicate>) -> Self {
        match predicate {
            Some(p) => {
                Self {
                    predicate: Some(Arc::new(SyncUnsafeCell::new(p))),
                    valid_idx: Arc::new(DashSet::new()),
                }
            }
            None => {
                Self {
                    predicate: None,
                    valid_idx: Arc::new(DashSet::new()),
                }
            }
        }
    }

    pub fn eval_bitmap(&self, roarings: &[Arc<RoaringBitmap>]) -> Result<Arc<RoaringBitmap>> {
        let predicate = {
            match self.predicate.as_ref() {
                Some(predicate) => {
                    unsafe { predicate.get().as_ref().unwrap() }
                }
                None => return Ok(Arc::new(RoaringBitmap::new())),
            }
            // let predicate = self.predicate.as_ref().unwrap().get();
            // unsafe { predicate.as_ref() }
        };
        predicate.eval_roaring(&roarings)
    }


    

}

impl std::fmt::Display for BooleanEvalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref predicate) = self.predicate {
            write!(f, "{:?}", unsafe{ &*(predicate.as_ref().get()) as &PhysicalPredicate })
        } else {
            write!(f, "None")
        }
    }
}

impl PhysicalExpr for BooleanEvalExpr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn data_type(&self, _input_schema: &Schema) -> datafusion::common::Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> datafusion::common::Result<bool> {
        Ok(false)
    }
    
    fn evaluate(&self, batch: &RecordBatch) -> datafusion::common::Result<ColumnarValue> {
        if let Some(ref predicate) = self.predicate {
            debug!("evalute batch len: {:}", batch.num_rows());
            let batch_len = batch.num_rows();
            let predicate = predicate.as_ref().get();
            let predicate_ref = unsafe {predicate.as_ref().unwrap() };
            match predicate_ref {
                PhysicalPredicate::And { .. } => {
                    let res = predicate_ref.eval(batch, vec![u64::MAX; batch_len], true)?;
                    debug!("res len: {:}", res.len());
                    let array = Arc::new(UInt64Array::from(res));
                    // debug!("evalute true count: {:}", &array.true_count());
                    Ok(ColumnarValue::Array(array))
                }
                PhysicalPredicate::Or { .. } => {
                    let res = predicate_ref.eval(batch, vec![0; batch_len], false)?;
                    let array = Arc::new(UInt64Array::from(res));
                    Ok(ColumnarValue::Array(array))
                }
                PhysicalPredicate::Leaf { primitive } => {
                    primitive.evaluate(batch)
                }
            }
        } else {
            let batch_len = batch.num_rows();
            Ok(ColumnarValue::Array(Arc::new(UInt64Array::from(vec![0; batch_len]))))
        }
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> datafusion::common::Result<Arc<dyn PhysicalExpr>> {
        Err(DataFusionError::Internal(format!(
            "Don't support with_new_children in BooleanQueryExpr"
        )))
    }
}

impl PartialEq<dyn Any> for BooleanEvalExpr {
    fn eq(&self, _other: &dyn Any) -> bool {
        false
    }
}
#[inline(always)]
fn load_u64_slice(bitmap: &&[u64]) -> __m512i {
    unsafe { _mm512_loadu_epi64((*bitmap).as_ptr() as *const i64) }
}

pub fn freqs_filter(freqs: &Vec<Option<Vec<Option<&[u8]>>>>, mask: &[u64], offset: usize) -> Vec<Arc<UInt8Array>> {
    let offs: Vec<u32> = mask.iter().map(|v| v.count_ones()).collect();
    let sum: u32 = offs.iter().map(|v| *v).sum();
    let valid_freqs: Vec<Arc<UInt8Array>> = freqs.iter()
        .map(|freq| {
            let mut valid_freqs: Vec<u8> = Vec::with_capacity(sum as usize);
            let mut cnter = 0;
            freq
                .iter()
                .zip(mask.into_iter())
                .enumerate()
                .filter(|(_, (_, m))| **m != 0)
                .for_each(|(i, (v, m))| {
                    match v[offset] {
                        Some(v) => {
                            let freqs_ptr = v.as_ptr() as *const i8;
                            unsafe {
                                let freqs = _mm512_loadu_epi8(freqs_ptr);
                                _mm512_mask_compressstoreu_epi8(valid_freqs.as_mut_ptr().offset(cnter), *m, freqs);
                                cnter += offs[i] as isize;
                            }
                        }
                        None => {  }
                    };

                });
                unsafe {
                    valid_freqs.set_len(sum as usize);
                };
                Arc::new(UInt8Array::from(valid_freqs))

        })
        .collect();
    valid_freqs
}

#[cfg(not(scalar))]
#[inline]
fn intersect_avx(l: &[u16], r: &[u16])  -> Vec<u16>{
    if !cfg!(feature = "scalar") {
        if l.len() == 0 || r.len() == 0 {
            return vec![]
        }
        // let mut i_a = 0;
        // let mut i_b = 0;
        let min_len = l.len().min(r.len());
        let mut res = Vec::with_capacity(min_len);
        const IMM: i32 = _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK;
        // while i_a < l.len() && i_b < r.len() {
            unsafe { // 1. Load the Vectors
            let v_a = _mm_loadu_epi16(l.as_ptr() as *const i16);
            let v_b = _mm_loadu_epi16(r.as_ptr() as *const i16);

            // 2. Full comparison
            let res_v = _mm_cmpestrm::<IMM>(v_b, r.len() as i32, v_a, l.len() as i32);
            let mask= _mm_extract_epi32::<0>(res_v);
            debug!("l: {:?}, r: {:?}, ", l, r);
            // let a7 = _mm_extract_epi32::<7>(v_a);
            // let b7 = _mm_extract_epi32::<7>(v_b);
            // 3. Write back common value
            _mm_mask_compressstoreu_epi16(res.as_mut_ptr() as *mut u8, mask as u8, v_a);
            res.set_len(mask.count_ones() as usize);
            }
            debug!("l: {:?}, res: {:?}", l, res);
        // }
        res
    } else {
        let l = l.into_iter().assume_sorted_by_item();
        let r = r.into_iter().assume_sorted_by_item();
        l.intersection(r).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{ptr::NonNull, sync::Arc};

    use datafusion::{arrow::{record_batch::RecordBatch, array::{BooleanArray, ArrayData}, buffer::Buffer, datatypes::{DataType, Schema, Field}}, physical_plan::{expressions::{col, BinaryExpr}, PhysicalExpr}, logical_expr::Operator};

    use crate::{ShortCircuit, jit::ast::Predicate, physical_expr::boolean_eval::SubPredicate};

    use super::{intersect_avx, BooleanEvalExpr, PhysicalPredicate, Primitives};

    #[test]
    fn test_boolean_eval_simple() {
        let predicate = Predicate::And {
            args: vec![
                Predicate::Or { 
                    args: vec![
                        Predicate::Leaf { idx: 2 },
                        Predicate::Leaf { idx: 3 },
                    ]
                },
                Predicate::Leaf { idx: 4 },
            ]
        };
        let primitive = ShortCircuit::try_new(
            vec![1, 2, 3],
            predicate,
            4,
            3,
            2,
        ).unwrap();
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("test1", DataType::Boolean, false),
                Field::new("test2", DataType::Boolean, false),
                Field::new("test3", DataType::Boolean, false),
                Field::new("test4", DataType::Boolean, false),
                Field::new("test5", DataType::Boolean, false),
                Field::new("__temp__", DataType::Boolean, false),
            ])
        );
        let t1 = vec![0b1010_1110, 0b0011_1010];
        let t2 = vec![0b1100_0100, 0b1100_1011];
        let t3 = vec![0b0110_0100, 0b1100_0011];
        let t4 = vec![0b1011_0000, 0b0111_0110];
        let t5 = vec![0b1001_0011, 0b1100_0101];
        let t6 = vec![0b1111_1111, 0b1111_1111];
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(u8_2_boolean(t1.clone())),
                Arc::new(u8_2_boolean(t2.clone())),
                Arc::new(u8_2_boolean(t3.clone())),
                Arc::new(u8_2_boolean(t4.clone())),
                Arc::new(u8_2_boolean(t5.clone())),
                Arc::new(u8_2_boolean(t6.clone())),
            ],
        ).unwrap();

        let op = |(v1, v2, v3, v4, v5): (u8, u8, u8, u8, u8)| v1 & (v2 | v3) & v4 & v5;
        let expect: Vec<u8> = (0..2).into_iter()
            .map(|v| (t1[v], t2[v], t3[v], t4[v], t5[v]))
            .map(op)
            .collect();
        let binary = BinaryExpr::new(col("test1", &schema).unwrap(), Operator::And, col("test5", &schema).unwrap());
        let sub_predicate1 = SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::BitwisePrimitive(binary) });
        let sub_predicate2 = SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ShortCircuitPrimitive(primitive) });
        let physical_predicate = PhysicalPredicate::And {
            args: vec![
                sub_predicate1,
                sub_predicate2,
            ]
        };
        let res = BooleanEvalExpr::new(Some(physical_predicate)).evaluate(&batch).unwrap().into_array(0);
        let res = unsafe { res.data().buffers()[0].align_to::<u8>().1 };
        println!("left: {:b}, right: {:b}", res[0], expect[0]);
        assert_eq!(res, &expect);
    }

    fn u8_2_boolean(mut u8s: Vec<u8>) -> BooleanArray {
        let batch_len = u8s.len() * 8;
        let value_buffer = unsafe {
            let buf = Buffer::from_raw_parts(NonNull::new_unchecked(u8s.as_mut_ptr()), u8s.len(), u8s.capacity());
            std::mem::forget(u8s);
            buf
        };
        let builder = ArrayData::builder(DataType::Boolean)
            .len(batch_len)
            .add_buffer(value_buffer);
        let array_data = builder.build().unwrap();
        BooleanArray::from(array_data)
    }

    #[test]
    fn test_intersect_simd() {
        let l = vec![136, 417, 450];
        let r = vec![303, 309];
        println!("res: {:?}", intersect_avx(&l, &r));
    }
}
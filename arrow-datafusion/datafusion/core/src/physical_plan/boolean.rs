// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! BooleanExec evaluates a boolean_query predicate against all input batches to determine
//! which rows to include in its output batches

use std::{sync::Arc, pin::Pin, task::{Poll, Context}, collections::HashMap, arch::x86_64::{__m512i, _mm512_mask_compressstoreu_epi32, _mm512_loadu_epi8, _mm512_mask_compressstoreu_epi8, _pext_u64}};

use crate::error::Result;
use arrow::{record_batch::RecordBatch, datatypes::{SchemaRef, DataType, Field, Schema}, array::{BooleanArray, UInt32Array, ArrayRef, Array, as_list_array, UInt8Array, UInt64Array}, error::ArrowError};
use datafusion_common::cast::{as_boolean_array, as_uint32_array, as_uint64_array};
use datafusion_physical_expr::{PhysicalExpr, AnalysisContext};
use futures::{StreamExt, Stream};
use itertools::Itertools;
use tracing::debug;
use datafusion_common::TermMeta;

use super::{ExecutionPlan, metrics::{ExecutionPlanMetricsSet, MetricsSet, BaselineMetrics}, DisplayFormatType, SendableRecordBatchStream, RecordBatchStream};



/// BooleanExec evaluates a boolean_query predicate against all input batches to determine
/// which rows to include in its output batches
#[derive(Debug)]
pub struct BooleanExec {
    /// The expression to filter on every partition
    pub predicate: Arc<dyn PhysicalExpr>,
    /// The input plan
    pub input: Arc<dyn ExecutionPlan>,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Terms stats for every partition
    pub terms_stats: Option<Vec<Option<TermMeta>>>,
    /// Is score
    pub is_score: bool,
    /// projected terms
    pub projected_terms: Arc<Vec<String>>,
}

impl BooleanExec {
    /// Create a BooleanExec on an input
    pub fn try_new(
        predicate: Arc<dyn PhysicalExpr>,
        input: Arc<dyn ExecutionPlan>,
        terms_stats: Option<Vec<Option<TermMeta>>>,
        is_score: bool,
        projected_terms: Arc<Vec<String>>,
    ) -> Result<Self> {
        Ok(Self {
            predicate,
            input: input.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
            terms_stats,
            is_score,
            projected_terms,
        })
    }

    /// The expression to filter on. This expression must evaluate to a boolean value.
    pub fn predicate_of(&self, _partition: usize) -> &Arc<dyn PhysicalExpr> {
        &self.predicate
    }

    /// The expression to filter on.
    pub fn predicate(&self) -> &Arc<dyn PhysicalExpr> {
        self.predicate_of(0)
    }

    /// The input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

}

impl ExecutionPlan for BooleanExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> arrow::datatypes::SchemaRef {
        // The boolean operator does not make any changes to the schema of its input
        self.input.schema()
    }

    fn output_partitioning(&self) -> super::Partitioning {
        self.input.output_partitioning()
    }

    fn output_ordering(&self) -> Option<&[datafusion_physical_expr::PhysicalSortExpr]> {
        self.input.output_ordering()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(BooleanExec::try_new(
            self.predicate.clone(),
            children[0].clone(),
            self.terms_stats.to_owned(),
            self.is_score,
            self.projected_terms.clone(),
        )?))
    }

    /// Sepcifies whether this plan generates an infinite stream of records.
    /// If the plan does not support pipelining, but it its input(s) are
    /// infinite, returns an error to indicate this.
    fn unbounded_output(&self, children: &[bool]) -> Result<bool> {
        Ok(children[0])
    }

    /// tell optimizer this operator doesn't reorder its input
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<crate::execution::context::TaskContext>,
    ) -> Result<super::SendableRecordBatchStream> {
        debug!("Start BooleanExec::execute for partition {} of context session_id {} and task_id {:?}", partition, context.session_id(), context.task_id());
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        Ok(Box::pin(BooleanExecStream {
            schema: self.input.schema(),
            predicate: self.predicate.clone(),
            input: self.input.execute(partition, context)?,
            baseline_metrics,
            is_score: self.is_score,
        }))
    }

    fn fmt_as(
        &self,
        t: super::DisplayFormatType,
        f: &mut std::fmt::Formatter
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f, "BooleanExec, predicate: {:}, is_score: {}", self.predicate(), self.is_score)
            }
        }
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    /// The output statisitcs of a boolean operation can be estimated if the
    /// predicate's selectivity value can be determined for the incoming data.
    fn statistics(&self) -> datafusion_common::Statistics {
        let input_stats = self.input.statistics();
        let _starter_ctx = 
            AnalysisContext::from_statistics(self.input.schema().as_ref(), &input_stats);
        unimplemented!()
    }
}

/// The BooleanExec streams wraps the input iterator and applies the predicate expression to
/// determine which rows to include in its ouput batches
struct BooleanExecStream {
    /// Output schema, which is the same as the input schema for this operator
    schema: SchemaRef,
    /// This expression must evaluate to boolean query
    predicate: Arc<dyn PhysicalExpr>,
    /// The input partition to boolean query
    input: SendableRecordBatchStream,
    /// runtime metrics recording
    baseline_metrics: BaselineMetrics,
    /// Is score
    is_score: bool,
}

// batch filter using vectorized intersection algorithm
fn vectorized_batch_filter(
    batch: &RecordBatch,
    predicate: &Arc<dyn PhysicalExpr>,
) -> Result<UInt64Array> {
    predicate
        .evaluate(batch)
        .map(|v| v.into_array(batch.num_rows()))
        .and_then(|array| {
            Ok(as_uint64_array(&array).unwrap().clone())
        })
}

// batch filter using vectorized intersection algorithm
fn batch_filter_with_freqs(
    batch: &RecordBatch,
    predicate: &Arc<dyn PhysicalExpr>,
) -> Result<RecordBatch> {
    debug!("start batch_filter_with_filter");
    predicate
        .evaluate(batch)
        .map(|v| v.into_array(batch.num_rows()))
        .and_then(|array| {
            Ok(as_boolean_array(&array)?)
                .and_then(|filter_array| Ok(filter_batch_with_freqs(&batch,filter_array)?))
        })
}

impl Stream for BooleanExecStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        debug!("BooleanStream poll_next");
        let poll;
        loop {
            match self.input.poll_next_unpin(cx) {
                Poll::Ready(value) => match value{
                    Some(Ok(batch)) => {
                        let timer = self.baseline_metrics.elapsed_compute().timer();
                        let boolean_batch = if true {
                            debug!("batch row count: {:}", batch.num_rows());
                            let mask = vectorized_batch_filter(&batch, &self.predicate)?;
                            debug!("end count");
                            let schema = Arc::new(Schema::new(
                                vec![Field::new("mask", DataType::UInt64, false)],
                            ));
                            debug!("array len: {:}", mask.len());
                            let batch = RecordBatch::try_new(
                                schema,
                                vec![Arc::new(mask)],
                            )?;
                            batch
                        } else {
                            if self.is_score {
                                batch_filter_with_freqs(&batch, &self.predicate)?
                            } else {
                                let mask = vectorized_batch_filter(&batch, &self.predicate)?;
                                let schema = Arc::new(
                                    Schema::new(
                                        vec![Field::new("mask", DataType::Boolean, false)]
                                    )
                                );
                                RecordBatch::try_new(
                                    schema,
                                    vec![Arc::new(mask)],
                                )?
                            }
                        };

                        timer.done();
                        poll = Poll::Ready(Some(Ok(boolean_batch)));
                        break;
                    }
                    _ => {
                        poll = Poll::Ready(value);
                        break;
                    }
                }
                Poll::Pending => {
                    poll = Poll::Pending;
                    break;
                }
            }
        }
        debug!("end poll");
        self.baseline_metrics.record_poll(poll)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // same number of record batches
        self.input.size_hint()
    }
}

impl RecordBatchStream for BooleanExecStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}



#[inline]
fn bitmap_filter(
    index: &UInt32Array,
    predicate: &BooleanArray,
) -> Arc<dyn Array> {
    let (prefix, buffer, _suffix) = unsafe { predicate.data().buffers()[0].align_to::<u16>()};
    assert!(prefix.len() == 0, "Len of prefix should be 0");
    // assert!(suffix.len() == 0, "Len of suffix should be 0");
    assert!(index.data().buffers().len() == 1, "The len of buffer should be 1");
    let (prefix, simd_idx, _suffix) = unsafe { index.data().buffers()[0].align_to::<[u32; 16]>() };
    assert!(prefix.len() == 0, "Len of prefix should be 0");
    // assert!(suffix.len() == 0, "Len of suffix should be 0");
    let mut output: Vec<u32> = Vec::with_capacity(predicate.len());
    let mut output_end = output.as_mut_ptr();
    let mut true_cnt = 0;
    buffer
        .into_iter()
        .zip(simd_idx.into_iter())
        .filter(|(v, _)| **v != 0)
        .for_each(|(v, ids)| {
            unsafe {
                _mm512_mask_compressstoreu_epi32(output_end as *mut u8, v.clone(), from_u32x16(ids.clone()));
                let cnt = v.count_ones();
                output_end = output_end.offset(cnt as isize);
                true_cnt += cnt;
            }
        });
    let (ptr, _, _) = output.into_raw_parts();
    let new_output = unsafe {
        Vec::from_raw_parts(ptr, true_cnt as usize, true_cnt as usize)
    };
    Arc::new(UInt32Array::from(new_output))
}

const fn from_u32x16(vals: [u32; 16]) -> __m512i {
    union U8x64 {
        vector: __m512i,
        vals: [u32; 16],
    }
    unsafe { U8x64 { vals }.vector }
}

#[inline]
#[allow(unused)]
fn filter_batch(
    record_batch: &RecordBatch,
    predicate: &BooleanArray,
) -> Result<RecordBatch, ArrowError> {
    let filtered_arrays = record_batch
        .columns()
        .iter()
        .map(|a| bitmap_filter(as_uint32_array(a).unwrap(), predicate))
        .collect::<Vec<ArrayRef>>();

    RecordBatch::try_new(record_batch.schema(), filtered_arrays)
}

fn freqs_filter(freqs: &ArrayRef, mask: &[u64]) -> Result<ArrayRef, ArrowError> {
    let list_arr = as_list_array(&freqs);
    let offs: Vec<u32> = mask.iter().map(|v| v.count_ones()).collect();
    let sum: u32 = offs.iter().map(|v| *v).sum();
    let mut valid_freqs: Vec<u8> = Vec::with_capacity(sum as usize);
    let mut cnter = 0;
    list_arr
        .iter()
        .zip(mask.into_iter())
        .enumerate()
        .filter(|(_, (_, m))| **m != 0)
        .for_each(|(i, (v, m))| {
            match v {
                Some(v) => {
                    let freqs = v.as_any().downcast_ref::<UInt8Array>().unwrap();
                    let freqs_ptr = freqs.data().buffers()[0].as_ptr() as *const i8;
                    unsafe {
                        let freqs = _mm512_loadu_epi8(freqs_ptr);
                        _mm512_mask_compressstoreu_epi8(valid_freqs.as_mut_ptr().offset(cnter), *m, freqs);
                        cnter += offs[i] as isize;
                    }
                }
                None => {  }
            }
        });
    unsafe {
        valid_freqs.set_len(sum as usize);
    }
    Ok(Arc::new(UInt8Array::from(valid_freqs)))
}

#[inline]
fn pext(mask: &[u64], raw: &[u64]) -> Vec<u64> {
    mask
        .into_iter()
        .zip(raw.into_iter())
        .map(|(l, r)| {
            unsafe {
                _pext_u64(*r, *l)
            }
        })
        .collect()
}

fn filter_batch_with_freqs(
    record_batch: &RecordBatch,
    predicate: &BooleanArray,
) -> Result<RecordBatch, ArrowError> {
    debug!("start filter batch with freqs");
    let bounder = record_batch.num_columns() / 2;
    let masks = unsafe {predicate.data().buffers()[0].align_to::<u64>().1 };
    // The first column is `__id__`
    let mut columns = vec![bitmap_filter(as_uint32_array(record_batch.column_by_name("__id__").expect("shoud have `__id__` column")).unwrap(), predicate)];
    record_batch
        .columns()[..bounder]
        .iter()
        .enumerate()
        .for_each(|(i, a)| {
            let posting = unsafe { a.data().buffers()[0].align_to::<u64>().1 };
            let freqs = record_batch.column(bounder + i + 1);
            let mask = pext(posting, masks);
            columns.push(freqs_filter(freqs, mask.as_slice()).unwrap());
        });
    let projected_schema = Arc::new(record_batch.schema().project((bounder..record_batch.num_columns()).collect_vec().as_slice())?);
    debug!("end filter batch with freqs");
    RecordBatch::try_new(projected_schema, columns)
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow::{array::{UInt32Array, BooleanArray, Array, UInt64Array, ArrayRef, GenericListArray}, record_batch::RecordBatch, datatypes::{Schema, Field, DataType, UInt8Type}};
    use datafusion_common::from_slice::FromSlice;
    use itertools::Itertools;

    use super::{filter_batch, pext, filter_batch_with_freqs};

    #[test]
    fn test_bitmap_filter() {
        let idx: UInt32Array = (0..32).collect();
        let mut predicate = vec![false; 32];
        predicate[2] = true;
        predicate[5] = true;
        let predicate = BooleanArray::from(predicate);
        assert_eq!(predicate.data().buffers().len(), 1);
        let record_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("__id__", DataType::UInt32, false),
            ])), 
            vec![Arc::new(idx)],
        ).unwrap();
        let res = filter_batch(&record_batch, &predicate).unwrap();
        assert_eq!(
            res.column(0).data().buffer::<u32>(0),
            &[2, 5],
        );
    }

    #[test]
    fn test_pext() {
        let raw = UInt64Array::from(vec![0b1000_1101, 0b1101_1000]);
        let mask = UInt64Array::from(vec![0b1000_1001, 0b1001_0000]);
        let res = pext(unsafe {raw.data().buffers()[0].align_to::<u64>().1 }, unsafe { mask.data().buffers()[0].align_to::<u64>().1});
        assert_eq!(res[0], 0b1101);
        assert_eq!(res[1], 0b1010);
    }

    #[test]
    fn test_filter_batch_with_freqs() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("test1", DataType::Boolean, false),
            Field::new("test2", DataType::Boolean, false),
            Field::new("__id__", DataType::UInt32, false),
            Field::new("test1_freq", DataType::UInt8, true),
            Field::new("test2_freqs", DataType::UInt8, true),
        ]));
        let mut test1 = vec![false; 512];
        let mut test2 = vec![false; 512];
        // init `test1` value
        test1[0] = true;
        test1[11] = true;
        test1[33] = true;
        test1[448] = true;
        test1[460] = true;
        test1[500] = true;

        // init `test2` value
        test2[0] = true;
        test2[5] = true;
        test2[140] = true;
        test2[145] = true;

        let postings: Vec<ArrayRef> = vec![
           Arc::new(BooleanArray::from(test1)),
           Arc::new(BooleanArray::from(test2)),
           Arc::new(UInt32Array::from_slice((0..512).collect_vec().as_slice())),
           Arc::new(GenericListArray::<i32>::from_iter_primitive::<UInt8Type, _, _>(vec![
                Some(vec![Some(1), Some(2), Some(3)]),
                None,
                None,
                Some(vec![Some(22), Some(2), Some(2)]),
            ])),
            Arc::new(GenericListArray::<i32>::from_iter_primitive::<UInt8Type, _, _>(vec![
                Some(vec![Some(2), Some(1)]),
                None,
                Some(vec![Some(1), Some(2)]),
                None,
            ])),
        ];
        let batch = RecordBatch::try_new(
            schema,
            postings,
        ).unwrap();
        let mut mask = vec![false; 512];
        mask[0] = true;
        mask[140] = true;
        mask[460] = true;
        let mask = BooleanArray::from(mask);
        let res = filter_batch_with_freqs(&batch, &mask).unwrap();
        println!("{:?}", res);
    }
}
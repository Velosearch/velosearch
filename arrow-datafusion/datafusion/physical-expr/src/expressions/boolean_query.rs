use std::{sync::Arc, any::Any, vec::IntoIter, fmt::Display, arch::x86_64::{__m512i, _mm512_or_epi64, _mm512_and_epi64, _mm512_loadu_epi64}, ptr::NonNull};

use arrow_buffer::Buffer;
use datafusion_common::{Result, cast::{as_boolean_array, as_uint32_array}, from_slice::FromSlice};
use datafusion_expr::{ColumnarValue, Operator};
use log::debug;

use crate::{PhysicalExpr, expressions::{Column, BinaryExpr}};
use arrow::{datatypes::{DataType, Schema}, array::{Array, ArrayRef, BooleanArray, UInt32Array, ArrayData}, 
    compute::cast, record_batch::RecordBatch};
use datafusion_common::DataFusionError;
use sorted_iter::*;
use sorted_iter::assume::*;

#[derive(Debug, Clone)]
pub struct Dnf {
    pub predicates: Vec<i64>,
    selectivity: Option<f64>,
}

impl IntoIterator for Dnf {
    type Item = i64;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.predicates.into_iter()
    }
}

impl Dnf {
    pub fn new(predicates: Vec<i64>) -> Self {
        Self {
            predicates,
            selectivity: None,
        }
    }

    #[inline]
    pub fn selectivity(&self) -> f64 {
        self.selectivity.unwrap_or(1.)
    }

    pub fn iter(&self) -> std::slice::Iter<i64>{
        self.predicates.iter()
    }


}

impl Display for Dnf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.predicates)
    }
}

pub struct BooleanQueryEvalFunc {
    func: fn(*const *const u8, *const i64, *const u8, i64) -> (),
    cnf: Vec<i64>
}

impl BooleanQueryEvalFunc {
    pub fn new(
        func: fn(*const *const u8, *const i64, *const u8, i64) -> (),
        cnf: Vec<i64>
    ) -> Self {
        Self {
            func,
            cnf,
        }
    }

    pub fn eval(
        &self,
        batch: &RecordBatch
    ) -> Arc<BooleanArray> {
        debug!("Eval by code_gen");
        let batch_len = batch.num_rows();
        let columns = if batch.column_by_name("__id__").is_some() {
            &batch.columns()[0..(batch.num_columns() - 1)]
        } else {
            batch.columns()
        };
        let args: Vec<*const u8> = columns
                .into_iter()
                .map(|v| v.data().buffers()[0].as_ptr())
                .collect();
        let mut res: Vec<u8> = vec![0; batch.num_rows() / 8];
        (self.func)(args.as_ptr(), self.cnf.as_ptr(), res.as_ptr(), (batch_len / 8) as i64);
        // let value_buffer = Buffer::from_slice_ref(res.as_slice());
        let value_buffer = unsafe {
            let buf = Buffer::from_raw_parts(NonNull::new_unchecked(res.as_mut_ptr()), res.len(), res.capacity());
            std::mem::forget(res);
            buf
        };
        let builder = ArrayData::builder(DataType::Boolean)
            .len(batch_len)
            .add_buffer(value_buffer);

        let array_data = unsafe { builder.build_unchecked() };
        Arc::new(BooleanArray::from(array_data))
    }
}

unsafe impl Send for BooleanQueryEvalFunc {}

unsafe impl Sync for BooleanQueryEvalFunc {}

impl std::fmt::Debug for BooleanQueryEvalFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BooleanQueryEvalFunc").finish()
    }
}

/// Binary expression
#[derive(Debug, Clone)]
pub struct BooleanQueryExpr {
    pub cnf_predicates: Option<Vec<Dnf>>,
    pub code_gen_eval: Option<Arc<BooleanQueryEvalFunc>>,
    pub predicate_tree: Arc<dyn PhysicalExpr>,
}

impl BooleanQueryExpr {
    /// Create new binary expression
    pub fn new_with_cnf(
        predicate_tree: Arc<dyn PhysicalExpr>,
        cnf_predicates: Vec<Dnf>,
    ) -> Self {
        Self { cnf_predicates: Some(cnf_predicates), code_gen_eval: None, predicate_tree }
    }

    /// 
    pub fn new_with_fn(
        predicate_tree: Arc<dyn PhysicalExpr>,
        code_gen: Arc<BooleanQueryEvalFunc>,
    ) -> Self {
        Self {
            cnf_predicates: None,
            code_gen_eval: Some(code_gen),
            predicate_tree,
        }
    }

    /// Create 
    pub fn new(
        predicate_tree: Arc<dyn PhysicalExpr>,
    ) -> Self {
        Self {
            cnf_predicates: None,
            code_gen_eval: None,
            predicate_tree,
        }
    }

    /// Evaluate 
    pub fn eval(&self, batch: RecordBatch) -> Result<Arc<BooleanArray>> {
        debug!("{:?}", self.predicate_tree);
        if let Some(ref eval) = self.code_gen_eval {
            Ok(eval.eval(&batch))
        } else {
            let eval_res = &self.predicate_tree.evaluate(&batch)?.into_array(0); 
            let eval_res = as_boolean_array(eval_res)?;
            Ok(Arc::new(eval_res.to_owned()))
        }
    }

    /// Get partition cnf
    pub fn cnf_of(&self, partition: usize) -> &Dnf {
        &self.cnf_predicates.as_ref().unwrap()[partition]
    }
}

impl std::fmt::Display for BooleanQueryExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut cnf_list = vec![];
        if let Some(ref expr) = self.cnf_predicates {
            for dnf in expr {
                let dnf_list: Vec<String> = dnf.iter().map(|p| p.to_string()).collect();
                cnf_list.push(format!("({})", dnf_list.join(" | ")));
            }
            write!(f, "{}", cnf_list.join(" & "))
        } else {
            write!(f, "Use CodeGen")
        }
    }
}

impl PhysicalExpr for BooleanQueryExpr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self, _input_schema: &arrow_schema::Schema) -> Result<arrow_schema::DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &arrow_schema::Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<datafusion_expr::ColumnarValue> {
        if let Some(ref eval) = self.code_gen_eval {
            let eval_res = eval.eval(&batch);
            Ok(ColumnarValue::Array(eval_res))
        } else {
            // self.predicate_tree.evaluate(batch)
            let BinaryExpr { left, op, right } = self.predicate_tree.as_any().downcast_ref::<BinaryExpr>().unwrap();
            let eval_res = simd_boolean_expr_eval(
                left.clone(), *op, right.clone(), &batch)?;
            Ok(ColumnarValue::Array(eval_res))
        }
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.predicate_tree.children()
    } 

    /// Append new dnf_predicates in cnf_predicates
    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        Err(DataFusionError::Internal(format!(
            "Don't supoort with_new_children in BooleanQueryExpr"
        )))
    }
}

impl PartialEq<dyn Any> for BooleanQueryExpr {
    fn eq(&self, other: &dyn Any) -> bool {
        self.predicate_tree.eq(other)
    }
}

macro_rules! _boolean_array_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        // debug!("left: {:?}, right: {:?}", $LEFT, $RIGHT);
        let left = $LEFT.as_any().downcast_ref::<BooleanArray>().unwrap();
        let right = $RIGHT.as_any().downcast_ref::<BooleanArray>().unwrap();
        $OP(left, right)
            .map(|a| Arc::new(a) as ArrayRef)
            .map_err(|e| DataFusionError::from(e))
    }}
}


#[inline]
fn _merge_pia(
    left: &ArrayRef,
    right: &ArrayRef,
) -> ArrayRef {
    // debug!("left: {:?}", left);
    // debug!("right: {:?}", right);
    let left = cast(left, &DataType::UInt32).unwrap();
    let right = cast(right, &DataType::UInt32).unwrap();
    let left = as_uint32_array(&left).expect("Can't cast to UInt16Array");
    let right = as_uint32_array(&right).expect("Can't cast to UInt16Array");
    let mut i = 0;
    let mut j = 0;
    let left_len = left.len();
    let right_len = right.len();
    let mut merge_res = Vec::new();
    while i < left_len && j < right_len {
        if left.value(i) < right.value(j) {
            merge_res.push(left.value(i));
            i += 1;
        } else if left.value(i) == right.value(j) {
            merge_res.push(left.value(i));
            i += 1;
            j += 1;
        } else {
            merge_res.push(right.value(j));
            j += 1;
        }
    }
    if i < left_len {
        merge_res.extend_from_slice(&left.values()[i..left_len]);
    }
    if j < right_len {
        merge_res.extend_from_slice(&right.values()[j..right_len]);
    }
    Arc::new(UInt32Array::from_slice(&merge_res))
}

#[inline]
fn _intersection_pia(
    left: &ArrayRef,
    right: &ArrayRef,
) -> ArrayRef {
    let left = cast(left, &DataType::UInt32).unwrap();
    let right = cast(right, &DataType::UInt32).unwrap();
    let left = as_uint32_array(&left).expect("Can't cast to UInt16Array").values().into_iter().assume_sorted_by_item();
    let right = as_uint32_array(&right).expect("Can't cast to UInt16Array").values().into_iter().assume_sorted_by_item();
    let res = left.intersection(right).map(|v| *v).collect::<Vec<_>>();
    Arc::new(UInt32Array::from_slice(&res))
}
union U8x64 {
    vector: __m512i,
    vals: [u8; 64],
}

fn simd_boolean_expr_eval(
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
    batch: &RecordBatch
) -> Result<Arc<dyn Array>> {
    let left = if let Some(c) = left.as_any().downcast_ref::<Column>() {
        c.evaluate(&batch)?.into_array(0)
    } else if let Some(BinaryExpr {left, op, right}) = left.as_any().downcast_ref::<BinaryExpr>() {
        simd_boolean_expr_eval(left.clone(), *op, right.clone(), &batch)?
    } else {
        unreachable!()
    };
    let right = if let Some(c) = right.as_any().downcast_ref::<Column>() {
        c.evaluate(&batch)?.into_array(0)
    } else if let Some(BinaryExpr {left, op, right}) = right.as_any().downcast_ref::<BinaryExpr>() {
        simd_boolean_expr_eval(left.clone(), *op, right.clone(), &batch)?
    } else {
        unreachable!()
    };
    // println!("left len: {:}", left.data().buffers()[0].len());
    let left_data = unsafe {&left.data().buffers()[0].as_slice().align_to::<i64>()};
    let right_data = unsafe { &right.data().buffers()[0].align_to::<i64>() };
    assert_eq!(left_data.0.len(), 0);
    assert_eq!(left_data.2.len(), 0);
    assert_eq!(right_data.0.len(), 0);
    assert_eq!(right_data.2.len(), 0);
    let batch_len = left.len();
    assert!(batch_len % 512 == 0, "The lenght of batch should be aligned to 512 bit");
    let mut res: Vec<u8> = match op {
        Operator::And => {
            unsafe {
                (0..(batch_len / 512))
                .map(|v| {
                    let left = _mm512_loadu_epi64(left_data.1.as_ptr().offset(8 * v as isize));
                    let right = _mm512_loadu_epi64(right_data.1.as_ptr().offset(8 * v as isize));
                    let vector = _mm512_and_epi64(left, right);
                    U8x64 { vector }.vals
                })
                .flatten()
                .collect()
            }
        }
        Operator::Or => {
            unsafe {                
                (0..(batch_len / 512))
                .map(|v| {
                    let left = _mm512_loadu_epi64(left_data.1.as_ptr().offset(8 * v as isize));
                    let right = _mm512_loadu_epi64(right_data.1.as_ptr().offset(8 * v as isize));
                    let vector = _mm512_or_epi64(left, right);
                    U8x64 { vector }.vals
                })
                .flatten()
                .collect()
            }
        }
        _ => unreachable!("Only support And and Or operation"),
    };
    let value_buffer = unsafe {
        let buf = Buffer::from_raw_parts(NonNull::new_unchecked(res.as_mut_ptr()), res.len(), res.capacity());
        std::mem::forget(res);
        buf
    };
    let builder = ArrayData::builder(DataType::Boolean)
        .len(batch_len)
        .add_buffer(value_buffer);

    let array_data = unsafe { builder.build_unchecked() };
    Ok(Arc::new(BooleanArray::from(array_data)))
}

// Create a BooleanQuery expression whose arguments are correctly coerced.
// This function errors if it is not possible to coerce the arguments
// to computational types supported by the operator.
pub fn boolean_query_with_cnf(
    cnf_predicates: Vec<Vec<i64>>,
    predicate: Arc<dyn PhysicalExpr>,
    _input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    let cnf_list = cnf_predicates
        .into_iter()
        .map(Dnf::new)
        .collect();
    Ok(Arc::new(BooleanQueryExpr::new_with_cnf(predicate, cnf_list)))
}

pub fn boolean_query(
    predicate: Arc<dyn PhysicalExpr>,
    _input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    Ok(
        Arc::new(
            BooleanQueryExpr::new(predicate)
        )
    )
}

#[cfg(test)]
mod tests {
    // use std::{sync::Arc, vec};

    // use crate::expressions::{col, literal};

    // use super::{boolean_query, merge_pia, intersection_pia};
    // use arrow::{datatypes::Schema, array::{BooleanArray, UInt32Array, ArrayRef}, record_batch::RecordBatch};
    // use arrow_schema::{Field, DataType};
    // use datafusion_common::{Result, cast::{as_boolean_array, as_uint32_array}, from_slice::FromSlice};

    // #[test]
    // fn pia_merge() {
    //     let left: ArrayRef = Arc::new(UInt32Array::from_slice(&[1, 2, 8, 9, 11, 20]));
    //     let right: ArrayRef = Arc::new(UInt32Array::from_slice(&[2, 4, 6, 7, 9, 14]));
    //     let merge_res = merge_pia(&left, &right);
    //     let merge_res = as_uint32_array(&merge_res).unwrap();
    //     assert_eq!(merge_res.values(), &[1, 2, 4, 6, 7, 8, 9, 11, 14, 20]);
    // }

    // #[test]
    // fn pia_intersection() {
    //     let left: ArrayRef = Arc::new(UInt32Array::from_slice(&[1, 2, 4, 5, 6, 7, 9, 13, 15, 20]));
    //     let right: ArrayRef = Arc::new(UInt32Array::from_slice(&[2, 4, 5, 6, 7, 8, 9, 11, 14, 16, 19]));
    //     let intersection_res = intersection_pia(&left, &right);
    //     let intersection_res = as_uint32_array(&intersection_res).unwrap();
    //     assert_eq!(intersection_res.values(), &[2, 4, 5, 6, 7, 9]);
    // }

    // #[test]
    // fn boolean_query_nested_op() -> Result<()> {
    //     let schema = Schema::new(vec![
    //         Field::new("a", DataType::Boolean, false),
    //         Field::new("b", DataType::Boolean, false),
    //         Field::new("c", DataType::Boolean, false),
    //     ]);
    //     let av = vec![true, false, false, true, true, false];
    //     let bv = vec![false, true, true, false, true, true];
    //     let cv = vec![false, false, true, true, false, false];
    //     let a = BooleanArray::from(av.clone());
    //     let b = BooleanArray::from(bv.clone());
    //     let c = BooleanArray::from(cv.clone());

    //     // expression "(a | b) & (a ^ c)"
    //     let expr = boolean_query(
    //         vec![
    //             vec![col("a", &schema)?, col("b", &schema)?],
    //             vec![col("a", &schema)?, col("c", &schema)?],
    //         ],
    //         &schema
    //     )?;

    //     let expected: Vec<bool> = av.into_iter()
    //         .zip(bv.into_iter().zip(cv.into_iter()))
    //         .map(|(a, (b, c))| {
    //             (a || b) && (a || c)
    //         }).collect();
    //     let batch = 
    //         RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b), Arc::new(c)])?;

    //     let result = expr.evaluate(&batch)?.into_array(batch.num_rows());
    //     assert_eq!(result.len(), 6);

    //     let result = result.as_any().downcast_ref::<BooleanArray>().unwrap();
    //     for (i, &expected_item) in expected.iter().enumerate() {
    //         assert_eq!(result.value(i), expected_item);
    //     }
        
    //     Ok(())
    // }

    // #[test]
    // fn boolean_query_finish() -> Result<()> {
    //     let schema = Schema::new(vec![
    //         Field::new("a", DataType::Boolean, false),
    //         Field::new("b", DataType::Boolean, false),
    //     ]);
    //     let av = vec![true, false, false, true, true, false, false, true];
    //     let bv = vec![false, true, true, false, true, false, true, true];
    //     let a = BooleanArray::from(av.clone());
    //     let b = BooleanArray::from(bv.clone());

    //     let expected: Vec<bool> = av.iter().zip(bv.iter()).map(|(a, b)| *a && *b).collect();
    //     // expression "a & b"
    //     let expr = boolean_query(
    //         vec![
    //             vec![col("a", &schema)?],
    //             vec![col("b", &schema)?]
    //         ],
    //         &schema
    //     )?;

    //     let batch = 
    //         RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)])?;
    //     let result = expr.evaluate(&batch)?.into_array(batch.num_rows() * 16);
    //     let result = as_boolean_array(&result).expect("failed to downcast to BooleanArray");
    //     assert_eq!(result.len(), 8);
    //     for (i, &expected_item) in expected.iter().enumerate() {
    //         assert_eq!(expected_item, result.value(i));
    //     }

    //     Ok(())
    // }
}
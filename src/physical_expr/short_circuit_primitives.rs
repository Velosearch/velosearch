use std::{any::Any, ptr::NonNull, sync::Arc, arch::x86_64::{_mm512_loadu_epi64, __m512i, _mm_prefetch, _MM_HINT_T1}};

use datafusion::{physical_plan::{PhysicalExpr, ColumnarValue}, arrow::{datatypes::DataType, record_batch::RecordBatch, array::{BooleanArray, ArrayData}, buffer::Buffer}, error::DataFusionError};
use tracing::debug;

use crate::{jit::{ast::{Predicate, Boolean}, jit_short_circuit, AOT_PRIMITIVES}, JIT_MAX_NODES, utils::avx512::U64x8};
use crate::utils::Result;

use super::{boolean_eval::{PhysicalPredicate, Chunk, TempChunk}, Primitives};

#[derive(Debug, Clone)]
pub struct ShortCircuit {
    pub batch_idx: Vec<usize>,
    primitive: fn(*const *const u8, *const u8, *mut u8, i64) -> (),
}

impl ShortCircuit {
    pub fn try_new(
        batch_idx: Vec<usize>,
        boolean_op_tree: Predicate,
        _node_num: usize,
        leaf_num: usize,
        start_idx: usize,
    ) -> Result<Self> {
        // let primitive = if node_num <= JIT_MAX_NODES {
            // Seek the AOT compilation
        //     unimplemented!()
        // } else {
        //     // JIT compilation the BooleanOpTree
        //     jit_short_circuit(Boolean { predicate: boolean_op_tree }, leaf_num)?
        // };
        // JIT compilation the BooleanOpTree
        let primitive = jit_short_circuit(Boolean { predicate: boolean_op_tree, start_idx}, leaf_num)?;
        Ok(Self {
            batch_idx,
            primitive,
        })
    }

    pub fn new(
        cnf: &Vec<&PhysicalPredicate>,
        node_num: usize,
        leaf_num: usize,
    ) -> Self {
        if node_num <= JIT_MAX_NODES {
            let mut builder = LoudsBuilder::new(node_num);
            predicate_2_louds(cnf, &mut builder);
            let louds = builder.build();
            debug!("louds: {:b}", louds);
            let primitive = AOT_PRIMITIVES.get(&louds).expect(&format!("Can't find louds: {:b}", louds));
            let batch_idx = convert_predicate(&cnf).1;
            return Self {
                primitive: *primitive,
                batch_idx
            };
        }
        let (predicate, batch_idx) = convert_predicate(&cnf);
        Self::try_new(batch_idx, predicate, node_num + 1, leaf_num, 0).unwrap()
    }

    pub fn eval(&self, init_v: &mut [u8], batch: &RecordBatch) -> Vec<u8> {
        let batch_len = init_v.len();
        let batch: Vec<*const u8> = self.batch_idx
            .iter()
            .map(|v| {
                unsafe {
                    batch.column(*v).data().buffers()[0].align_to::<u8>().1.as_ptr()
                }
            })
            .collect();
        let mut res: Vec<u8> = vec![0; batch_len];
        (self.primitive)(batch.as_ptr(), init_v.as_ptr(), res.as_mut_ptr(), batch_len as i64);
        res
    }

    pub fn eval_avx512(&self, init_v: Option<Vec<TempChunk>>, batches: &Vec<Option<Vec<Chunk>>>, batch_len: usize) -> Vec<TempChunk> {
        const ZEROS: [u64; 8] = [0; 8];
        debug!("Eval by short_circuit_primitives");
        let mut res: Vec<u64> = vec![0; batch_len * 8];
        
        let mut leak_list: Vec<[u64; 8]> = vec![[0; 8]; batches.len() + 1];

        let mut test: [u64; 8] = [u64::MAX; 8];
        let batch: Vec<*const u8> = self.batch_idx.iter().enumerate()
            .map(|(n, v)| unsafe { 
                if let Some(ref c) = batches[*v] {
                    match c.get_unchecked(0) {
                        Chunk::Bitmap(b) => (*b).as_ptr() as *const u8,
                        Chunk::IDs(ids) => {
                            leak_list[n].fill(0);
                            let bitmap = leak_list.get_unchecked_mut(n);
                            for off in *ids {
                                *bitmap.get_unchecked_mut((*off >> 6) as usize) |= 1 << (*off % (1 << 6));
                            }
                            let data = bitmap.as_ptr() as *const u8;
                            // _mm_prefetch::<_MM_HINT_T1>(data as *const i8);
                            data
                            // test.as_ptr() as *const u8
                        }
                        _ => test.as_ptr() as *const u8
                    }
                } else {
                    debug!("is zero");
                    test.as_ptr() as *const u8
                }
            })
            .collect();
        // for i in 0..batch_len {
            // let batch: Vec<*const u8> = batches.iter().enumerate()
            // .map(|(n, v)| unsafe { 
            //     if let Some(c) = v {
            //         match c.get_unchecked(i) {
            //             Chunk::Bitmap(b) => (*b).as_ptr() as *const u8,
            //             Chunk::IDs(ids) => {
            //                 leak_list[n].fill(0);
            //                 let bitmap = leak_list.get_unchecked_mut(n);
            //                 for off in *ids {
            //                     *bitmap.get_unchecked_mut((*off >> 6) as usize) |= 1 << (*off % (1 << 6));
            //                 }
            //                 let data = bitmap.as_ptr() as *const u8;
            //                 // _mm_prefetch::<_MM_HINT_T1>(data as *const i8);
            //                 data
            //                 // test.as_ptr() as *const u8
            //             }
            //             _ => unreachable!()
            //         }
            //     } else {
            //         ZEROS.as_ptr() as *const u8
            //     }
            // })
            // .collect();
            debug!("start init");
            let init = match init_v {
                Some(ref v) => {
                    match &v[0] {
                        TempChunk::Bitmap(b) => unsafe { U64x8 { vector: *b }.vals.as_ptr() as *const u8 },
                        TempChunk::IDs(ids) => {
                            unsafe {
                                leak_list.last_mut().unwrap_unchecked().fill(0);
                                let bitmap = leak_list.last_mut().unwrap_unchecked();
                                for &off in ids {
                                    *bitmap.get_unchecked_mut((off >> 6) as usize) |= 1 << (off % (1 << 6));
                                }
                                let data = bitmap.as_ptr() as *const u8;
                                _mm_prefetch::<_MM_HINT_T1>(data as *const i8);
                                data
                                // test.as_ptr() as *const u8
                            }
                        }
                        TempChunk::N0NE => batch[0] as *const u8,
                    }
                }
                None => batch[0] as *const u8,
            };
            debug!("start eval");
            debug!("batch len: {:}", batches.len());
                (self.primitive)(
                    batch.as_ptr() as *const *const u8,
                    init,
                    unsafe { res.as_mut_ptr().offset(0 as isize * 8) } as *mut u8,
                    8,
                );
        // }

        debug!("end eval");
        (0..batch_len).into_iter()
        .map(|v| {
            // TempChunk::Bitmap(unsafe { _mm512_loadu_epi64(res.as_ptr().add(v * 8) as *const i64)})
            TempChunk::N0NE
        })
        .collect()
    }
}

impl PhysicalExpr for ShortCircuit {
    fn as_any(&self) -> &dyn std::any::Any {
       self 
    }

    fn data_type(&self, _input_schema: &datafusion::arrow::datatypes::Schema) -> datafusion::error::Result<datafusion::arrow::datatypes::DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &datafusion::arrow::datatypes::Schema) -> datafusion::error::Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> datafusion::error::Result<ColumnarValue> {
        let init_v = batch.column_by_name("__temp__").ok_or(DataFusionError::Internal(format!("Should have `init_v` as the input param in batch")))?;
        let batch_len = init_v.len() / 8;
        let batch: Vec<*const u8> = self.batch_idx
            .iter()
            .map(|v| {
                unsafe {
                    batch.column(*v).data().buffers()[0].align_to::<u8>().1.as_ptr()
                }
            })
            .collect();
        let init_v_ptr = unsafe { init_v.data().buffers()[0].align_to::<u8>().1.as_ptr() };
        let mut res: Vec<u8> = vec![0; batch_len];
        (self.primitive)(batch.as_ptr(), init_v_ptr, res.as_mut_ptr(), batch_len as i64);
        let res = build_boolean_array(res, init_v.len());
        Ok(ColumnarValue::Array(Arc::new(res)))
    }

    fn children(&self) -> Vec<std::sync::Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(
        self: std::sync::Arc<Self>,
        _children: Vec<std::sync::Arc<dyn PhysicalExpr>>,
    ) -> datafusion::error::Result<std::sync::Arc<dyn PhysicalExpr>> {
        Err(DataFusionError::Internal(format!(
            "Don't support with_new_children in ShortCircuit"
        )))
    }
}

impl std::fmt::Display for ShortCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Use short-circuit primitive")
    }
}

impl PartialEq<dyn Any> for ShortCircuit {
    fn eq(&self, _other: &dyn Any) -> bool {
        false
    }
}

#[inline]
fn build_boolean_array(mut res: Vec<u8>, array_len: usize) -> BooleanArray {
    let value_buffer = unsafe {
        let buf = Buffer::from_raw_parts(NonNull::new_unchecked(res.as_mut_ptr()), res.len(), res.capacity());
        std::mem::forget(res);
        buf
    };
    let builder = ArrayData::builder(DataType::Boolean)
        .len(array_len)
        .add_buffer(value_buffer);

    let array_data = unsafe { builder.build_unchecked() };
    BooleanArray::from(array_data)
}

fn convert_predicate(cnf: &Vec<&PhysicalPredicate>) -> (Predicate, Vec<usize>) {
    let mut batch_idx = Vec::new();
    let mut predicates = Vec::new();
    let mut start_idx = 0;
    for forumla in cnf.into_iter().rev() {
        predicates.push(physical_2_logical(&forumla, &mut batch_idx, &mut start_idx));
    }
    let predicate = Predicate::And { args: predicates };
    (predicate, batch_idx)
}

fn physical_2_logical(
    physical_predicate: &PhysicalPredicate,
    batch_idx: &mut Vec<usize>,
    start_idx: &mut usize,
) -> Predicate {
    match physical_predicate {
        PhysicalPredicate::And { args } => {
            let mut predicates = Vec::new();
            for arg in args {
                predicates.push(physical_2_logical(&arg.sub_predicate, batch_idx, start_idx));
                *start_idx += 1;
            }
            Predicate::And { args: predicates }
        }
        PhysicalPredicate::Or { args } => {
            let mut predicates = Vec::new();
            for arg in args {
                predicates.push(physical_2_logical(&arg.sub_predicate, batch_idx, start_idx));
                *start_idx += 1;
            }
            Predicate::Or { args: predicates }
        }
        PhysicalPredicate::Leaf { primitive } => {
            let idx = if let Primitives::ColumnPrimitive(c) = primitive {
                c.index()
            } else {
                unreachable!("The subtree of a short-circuit primitive must have not other exprs");
            };
            batch_idx.push(idx);
            Predicate::Leaf { idx: *start_idx }
        }
    }
}

struct LoudsBuilder {
    node_num: usize,
    pos: usize,
    has_child: u16,
    louds: u16,
}

impl LoudsBuilder {
    fn new(node_num: usize)  -> Self {
        assert!(node_num < 14, "Only supports building LOUDS with less 14 nodes");
        Self {
            node_num,
            pos: 0,
            has_child: 0,
            louds: 0,
        }
    }

    fn set_haschild(&mut self, has_child: bool) {
        if has_child {
            self.has_child |= 1 << self.pos;
        }
    }

    fn set_louds(&mut self, louds: bool) {
        if louds {
            self.louds |= 1 << self.pos;
        }
    }

    fn next(&mut self) {
        self.pos += 1;
    }

    fn build(self) -> u32 {
        let mut louds = (self.node_num as u32) << 28;
        louds |= (self.louds as u32) << 14;
        louds |= self.has_child as u32;
        louds
    }
}

fn predicate_2_louds(cnf: &Vec<&PhysicalPredicate>, builder: &mut LoudsBuilder) {
    builder.set_louds(true);
    if let PhysicalPredicate::Leaf { .. } = cnf[0] {
        builder.set_haschild(false);
    } else {
        builder.set_haschild(true);
    }
    builder.next();
    if cnf.len() > 1 {
        for formula in &cnf[1..] {
            builder.set_louds(false);
            if let PhysicalPredicate::Leaf { .. } = formula {
                builder.set_haschild(false);
            } else {
                builder.set_haschild(true);
            }
            builder.next();
        }
    }
    
    for formula in cnf {
        recursive_predicate_2_louds(formula, builder);
    }
}

fn recursive_predicate_2_louds(predicate: &PhysicalPredicate, builder: &mut LoudsBuilder) {
    match predicate {
        PhysicalPredicate::And { args } => {
            // The LOUDS of first child node is Set bit.
            builder.set_louds(true);
            builder.next();
            for child in &args[1..] {
                builder.set_louds(false);
                if let PhysicalPredicate::Leaf { .. } = child.sub_predicate {
                    builder.set_haschild(false);
                } else {
                    builder.set_haschild(true);
                }
                builder.next();
            }
            for child in args {
                recursive_predicate_2_louds(&child.sub_predicate, builder);
            }
        }
        PhysicalPredicate::Or { args } => {
            // The LOUDS of first child node is Set.
            builder.set_louds(true);
            builder.next();
            for child in &args[1..] {
                builder.set_louds(false);
                if let PhysicalPredicate::Leaf { .. } = child.sub_predicate {
                    builder.set_haschild(false);
                } else {
                    builder.set_haschild(true);
                }
                builder.next();
            }
            for child in args {
                recursive_predicate_2_louds(&child.sub_predicate, builder);
            }
        }
        PhysicalPredicate::Leaf { .. } => {
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, ptr::NonNull};

    use datafusion::{physical_plan::{PhysicalExpr, expressions::Column}, arrow::{datatypes::{Schema, Field, DataType}, record_batch::RecordBatch, array::{BooleanArray, ArrayData}, buffer::Buffer}};

    use crate::{jit::ast::Predicate, ShortCircuit, physical_expr::{boolean_eval::{PhysicalPredicate, SubPredicate}, Primitives}};

    use super::{predicate_2_louds, LoudsBuilder};

    #[test]
    fn test_short_circuit() {
        let predicate = Predicate::And {
            args: vec![
                Predicate::Leaf { idx: 0 },
                Predicate::Or { 
                    args: vec![
                        Predicate::Leaf { idx: 1 },
                        Predicate::Leaf { idx: 2 },
                    ]
                },
                Predicate::Leaf { idx: 3 },
            ]
        };
        let primitive = ShortCircuit::try_new(
            vec![0, 1, 2, 3],
            predicate,
            5,
            4,
            0,
        ).unwrap();
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("test1", DataType::Boolean, false),
                Field::new("test2", DataType::Boolean, false),
                Field::new("test3", DataType::Boolean, false),
                Field::new("test4", DataType::Boolean, false),
                Field::new("__temp__", DataType::Boolean, false),
            ])
        );
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(u8_2_boolean(vec![0b1010_1110, 0b0011_1010])),
                Arc::new(u8_2_boolean(vec![0b1100_0100, 0b1100_1011])),
                Arc::new(u8_2_boolean(vec![0b0110_0100, 0b1100_0011])),
                Arc::new(u8_2_boolean(vec![0b1011_0000, 0b0111_0110])),
                Arc::new(u8_2_boolean(vec![0b1111_1111, 0b1111_1111])),
            ],
        ).unwrap();
        let res = primitive.evaluate(&batch).unwrap().into_array(0);
        let res = unsafe {res.data().buffers()[0].align_to::<u8>().1 };
        assert_eq!(res, &[0b1010_0000, 0b0000_0010])
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
    fn physical_2_predicate() {
        let predicate = vec![
            PhysicalPredicate::Leaf {
                primitive: Primitives::ColumnPrimitive(Column::new("test1", 0)),
            },
            PhysicalPredicate::Or{
                args: vec![
                    SubPredicate::new_with_predicate(PhysicalPredicate::Leaf {
                        primitive: Primitives::ColumnPrimitive(Column::new("test2", 1)),
                    }),
                    SubPredicate::new_with_predicate(PhysicalPredicate::Leaf {
                        primitive: Primitives::ColumnPrimitive(Column::new("test3", 2)),
                    }),
                ]
            },
            PhysicalPredicate::Leaf {
                primitive: Primitives::ColumnPrimitive(Column::new("test4", 3)),
            }
        ];
        let mut builder = LoudsBuilder::new(5);
        predicate_2_louds(&predicate.iter().collect(), &mut builder);
        let res = builder.build();
        println!("res: {:b}", res);
    }
}
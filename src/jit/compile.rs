//! Compile Expr to JIT'd function

use tracing::debug;

use crate::jit::ast::Predicate;
use crate::utils::Result;
use super::Boolean;
use super::api::Assembler;
use super::ast::JITType;
use super::{
    api::GeneratedFunction,
    ast::{Expr as JITExpr, I64},
};

pub fn build_boolean_query(
    assembler: &Assembler,
    jit_expr: JITExpr,
) -> Result<GeneratedFunction> {
    // Alias pointer type.
    // The raw pointer `R64` or `R32` is not compatible with integers
    const PTR_TYPE: JITType = I64;

    let builder = assembler.new_func_builder("eval_fn");
    // Declare in-param.
    // Each input takes one position, following by a pointer to place result,
    // and the last is the lenght of inputs/output arrays.
    let mut builder = builder
        .param("batch", PTR_TYPE)
        .param("cnf", PTR_TYPE)  // Run-Time cnf predicate
        .param("result", PTR_TYPE)
        .param("len", I64);

    // Start build function body.
    // It's loop that calculates the result one by one
    let mut fn_body = builder.enter_block();
    let cnf_nums = if let JITExpr::BooleanExpr(ref bool) = jit_expr {
        &bool.cnf
    } else {
        unreachable!()
    };
    let mut cur = 0;
    let mut cnf = Vec::new();
    for i in cnf_nums {
        cnf.push((i, cur));
        cur += *i;
    }
    for (i, n) in cnf.into_iter().enumerate() {
        let offset = fn_body.add(fn_body.id("cnf")?, fn_body.lit_i64(8 * n.1))?;
        fn_body.declare_as(format!("p{i}_1").as_str(), fn_body.load(offset, I64)?)?;
        if *n.0 == 2 {
            let offset = fn_body.add(fn_body.id("cnf")?, fn_body.lit_i64(8 * n.1 + 8))?;
            fn_body.declare_as(format!("p{i}_2").as_str(), fn_body.load(offset, I64)?)?;
        }
    }
    fn_body.declare_as("index", fn_body.lit_i64(0))?;
    fn_body.while_block(
        |cond| cond.lt(cond.id("index")?, cond.id("len")?),
        |b| {
            b.declare_as("offset", b.mul(b.id("index")?, b.lit_i64(8))?)?;
            b.declare_as("res_ptr", b.add(b.id("result")?, b.id("offset")?)?)?;
            b.declare_as("res", jit_expr.clone())?;
            b.store(b.id("res")?, b.id("res_ptr")?)?;
            b.assign("index", b.add(b.id("index")?, b.lit_i64(1))?)?;

            Ok(())
        },
    )?;

    let gen_func = fn_body.build();
    Ok(gen_func)
}

pub fn jit_short_circuit_primitive(
    assembler: &Assembler,
    jit_expr: Boolean,
    leaf_num: usize,
    louds: u32,
) -> Result<GeneratedFunction> {
    let jit_expr = JITExpr::Boolean(jit_expr);
    // Alias pointer type.
    // The raw pointer `R64` or `R32` is not compatible with integers
    const PTR_TYPE: JITType = I64;

    let builder = assembler.new_func_builder(format!("short_circuit_primitive_{:}", louds));
    // Declare in-param
    // Each input takes one position, following by a pointer to place result,
    // and the last is the length of inputs/output arrays.
    let mut builder = builder
        .param("batch", PTR_TYPE)
        .param("init_v", PTR_TYPE)
        .param("result", PTR_TYPE)
        .param("len", I64);

    // Start build function body.
    // It's loop that calculate the result one by one to enable
    // loop unrolling
    let mut fn_body = builder.enter_block();
    for i in 0..leaf_num {
        let offset = fn_body.add(fn_body.id("batch")?, fn_body.lit_i64(8 * i as i64))?;
        fn_body.declare_as(format!("p{i}").as_str(), fn_body.load(offset, I64)?)?;
    }
    fn_body.declare_as("index", fn_body.lit_i64(0))?;
    // fn_body.declare("offset", I64)?;
    fn_body.while_block(
        |cond| cond.lt(cond.id("index")?, cond.lit_i64(8)),
        |b| {
            b.declare_as("offset", b.mul(b.id("index")?, b.lit_i64(8))?)?;
            // b.declare_as("offset", b.mul(b.id("index")?, b.lit_i64(8))?)?;
            b.declare_as("res_ptr", b.add(b.id("result")?, b.id("offset")?)?)?;
            b.declare_as("res", jit_expr.clone())?;
            // b.declare_as("res", b.lit_i64(66))?;
            b.store(b.id("res")?, b.id("res_ptr")?)?;
            b.assign("index", b.add(b.id("index")?, b.lit_i64(1))?)?;
            Ok(())
        }
    )?;
    let gen_func = fn_body.build();
    Ok(gen_func)

}

#[derive(Debug)]
pub struct Louds2Boolean {
    look_up: Vec<usize>,
    louds: u16,
    has_child: u16,
    node_num: usize,
    idx: usize,
}

impl Louds2Boolean {
    pub fn new(louds: u32) -> Self {
        // Unexpected condition:
        // 1. the top level has only one node.
        let node_num = (louds >> 28) as usize;
        let has_child = (louds & ((1 << node_num) - 1)) as u16;
        let louds = (louds >> 14  & ((1 << node_num) - 1)) as u16;
        let mut look_up: Vec<usize> = Vec::new();
        for i in 0..(node_num as usize) {
            if louds & 1 << i != 0 {
                look_up.push(i);
            }
        }
        debug!("louds: {:b}", louds);
        debug!("has_child: {:b}", has_child);
        debug!("node_num: {:}", node_num);
        Self {
            look_up,
            louds,
            has_child,
            node_num,
            idx: 0,
        }
    }

    pub fn leaf_num(&self) -> usize {
        self.idx
    }

    pub fn build(&mut self) -> Option<Boolean> {
        match self.recursive_construct(0, true) {
            Ok(p) => match p {
                Some(p) => Some(Boolean { predicate: p, start_idx: 0 }),
                None => None,
            },
            Err(_) => None,
        }
    }

    pub fn recursive_construct(&mut self, pos: usize, is_and: bool) -> std::result::Result<Option<Predicate>, ()> {
        if pos >= self.node_num {
            return Ok(None);
        }
        let mut predicates: Vec<Predicate> = Vec::new();
        let mut pos = pos;
        let mut indicator = 1 << pos;
        if self.louds & indicator == 0 {
            // obey the Louds rule.
            return Err(());
        }
        let mut is_first = true;
        while is_first || pos < self.node_num && self.louds & indicator == 0 {
            if is_first {
                is_first = false;
            }
            if self.has_child & indicator != 0 {
                // r = rank(S-HasChild, pos) + 1;
                // select(louds, r);
                let r = (self.has_child & ((1 << pos + 1) - 1)).count_ones() as usize;
                if r >= self.look_up.len() {
                    return Err(());
                }
                debug!("r: {:}", r);
                let child_pos = self.look_up[r];
                debug!("child_pos: {:}", child_pos);
                match self.recursive_construct(child_pos, !is_and) {
                    Ok(Some(p)) => {
                        predicates.push(p);
                    }
                    Ok(None) => {
                        if is_and {
                            return  Ok(Some(Predicate::And { args: predicates }));
                        } else {
                            return Ok(Some(Predicate::Or { args: predicates }));
                        }
                    },
                    Err(_) => return Err(()),
                };
                pos += 1;
            } else {
                predicates.push(Predicate::Leaf { idx: self.idx });
                self.idx += 1;
                pos += 1;
            }
            indicator = 1 << pos;
        }
        if is_and {
            return Ok(Some(Predicate::And { args: predicates }));
        } else {
            return  Ok(Some(Predicate::Or { args: predicates }));
        }
    }
}

#[cfg(test)]
mod test {
    use tracing::{Level, debug};

    use crate::jit::{ast::{Expr, BooleanExpr, Boolean, Predicate}, api::Assembler};

    use super::{build_boolean_query, jit_short_circuit_primitive, Louds2Boolean};

    #[test]
    fn boolean_query_simple() {
        tracing_subscriber::fmt().with_max_level(Level::DEBUG).init();
        let jit_expr = Expr::BooleanExpr(BooleanExpr {
            cnf: vec![1, 2, 1],
        });
        // allocate memory for result
        let result: Vec<u8> = vec![0x0; 2];
        let test1 = vec![0x01, 0x0];
        let test2 = vec![0x11, 0x23];
        let test3 = vec![0x21, 0xFF];
        let test4 = vec![0x21, 0x12];
        let input = vec![
            test1.as_ptr(),
            test2.as_ptr(),
            test3.as_ptr(),
            test4.as_ptr(),
        ];
        let cnf = vec![0, 1, 2, 3 ];


        // compile and run JIT code
        let assembler = Assembler::default();
        let gen_func = build_boolean_query(&assembler, jit_expr).unwrap();

        let mut jit = assembler.create_jit();
        let gen_func = jit.compile(gen_func).unwrap();
        let code_fn = unsafe {
            core::mem::transmute::<_, fn(*const *const u8, *const i64, *const u8, i64) -> ()>(gen_func)
        };
        code_fn(input.as_ptr(), cnf.as_ptr(), result.as_ptr(), 2);
        assert_eq!(result, vec![0x01, 0]);
    }

    #[test]
    fn boolean_query_v2_simple() {
        tracing_subscriber::fmt().with_max_level(Level::DEBUG).init();
        let jit_expr = Boolean {
            predicate: Predicate::And { 
                args: vec![
                    Predicate::Leaf { idx: 0 },
                    Predicate::Or { args: vec![
                        Predicate::Leaf { idx: 1 },
                        Predicate::Leaf { idx: 2 },
                    ] },
                    Predicate::Leaf { idx: 3 },
                ] 
            },
            start_idx: 0,
        };
        // allocate memory for result
        let result: Vec<u8> = vec![0x0; 2];
        let test1 = vec![0x01, 0x0];
        let test2 = vec![0x11, 0x23];
        let test3 = vec![0x21, 0xFF];
        let test4 = vec![0x21, 0x12];
        let init_v: Vec<u8> = vec![1, 1];
        let batch = vec![
            test1.as_ptr(),
            test2.as_ptr(),
            test3.as_ptr(),
            test4.as_ptr(),
        ];

        // Compile and run JIT code
        let assembler = Assembler::default();
        let gen_func = jit_short_circuit_primitive(&assembler, jit_expr, 4, 0).unwrap();

        let mut jit = assembler.create_jit();
        let gen_func = jit.compile(gen_func).unwrap();
        debug!("start transmute");
        let code_fn = unsafe {
            core::mem::transmute::<_, fn(*const *const u8, *const u8, *const u8, i64) -> ()>(gen_func)
        };
        debug!("end transmute");
        code_fn(batch.as_ptr(), init_v.as_ptr(), result.as_ptr(), 2);
        assert_eq!(result, vec![1, 0]);
    }

    #[test]
    fn boolean_query_nested() {
        tracing_subscriber::fmt().with_max_level(Level::DEBUG).init();
        let jit_expr = Boolean {
            predicate: Predicate::And {
                args: vec![
                    Predicate::Leaf { idx: 0 },
                    Predicate::Or { args: vec![
                        Predicate::Leaf { idx: 1 },
                        Predicate::And { args: vec![
                            Predicate::Leaf { idx: 2 },
                            Predicate::Leaf { idx: 3 },
                        ] }
                    ] },
                ] 
            },
            start_idx: 0,
        };
        // allocate memory for result
        let mut result: Vec<u64> = vec![0x0; 8];
        let test1: Vec<u64> = vec![0xFF, 0xFF, 0x32, 0x22, 0x12, 0x56, 0x0, 0x53];
        let test2: Vec<u64> = vec![0x0, 0xFF, 0x34, 0x14, 0x66, 0x91, 0x12, 0x34];
        let test3: Vec<u64> = vec![0x1, 0x0, 0x11, 0x12, 0x92, 0x12, 0x93, 0x11];
        let test4: Vec<u64> = vec![0xFF, 0x0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let init_v: Vec<u64> = vec![u64::MAX; 8];
        let batch = vec![
            test1.as_ptr() as *const u8,
            test2.as_ptr() as *const u8,
            test3.as_ptr() as *const u8,
            test4.as_ptr() as *const u8,
        ];

        // Compile and run JIT code
        let assembler = Assembler::default();
        let gen_func = jit_short_circuit_primitive(&assembler, jit_expr, 4, 0).unwrap();

        let mut jit = assembler.create_jit();
        let gen_func = jit.compile(gen_func).unwrap();
        debug!("start transmute");
        let code_fn = unsafe {
            core::mem::transmute::<_, fn(*const *const u8, *const u8, *mut u8, i64) -> ()>(gen_func)
        };
        debug!("end transmute");
        code_fn(batch.as_ptr(), init_v.as_ptr() as *const u8, result.as_mut_ptr() as *mut u8, 8);
        assert_eq!(result, vec![33, 0xa1 & (0x23 | (0x12 & 0xFF)), 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_louds_2_boolean() {
        tracing_subscriber::fmt().with_max_level(Level::DEBUG).init();
        let louds = 0b0101_00000000001001_00000000000010;
        let res = Louds2Boolean::new(louds).build();
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
        assert_eq!(res.unwrap().predicate, predicate);
    }

    #[test]
    fn test_louds_2_boolean_2() {
        tracing_subscriber::fmt().with_max_level(Level::DEBUG).init();
        let predicate = Predicate::And {
            args: vec![
                Predicate::Or {
                    args: vec![
                        Predicate::And { args: vec![
                            Predicate::Leaf { idx: 0 },
                            Predicate::Leaf { idx: 1 },
                        ] },
                        Predicate::Leaf { idx: 2 },
                    ]
                },
                Predicate::Leaf { idx: 3 },
                Predicate::Leaf { idx: 4 },
            ]
        };
        let louds = 0b0111_00000000101001_00000000001001;
        let res = Louds2Boolean::new(louds).build();
        assert_eq!(predicate, res.unwrap().predicate);
    }
}
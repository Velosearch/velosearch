pub mod ast;
pub mod api;
pub mod compile;
pub mod jit;
use std::collections::HashMap;

use lazy_static::lazy_static;
use tracing::{debug, info};

pub use crate::jit::{api::Assembler, ast::{Expr, Boolean, BooleanExpr}, compile::build_boolean_query};
use crate::{utils::Result, JIT_MAX_NODES, jit::compile::Louds2Boolean};

use self::compile::jit_short_circuit_primitive;


lazy_static!{
    pub static ref AOT_PRIMITIVES: HashMap<u32, fn(*const *const u8, *const u8, *mut u8, i64)> = {
        // if let Ok(f) = File::open("aot3.bin") {
        //     info!("Loud AOT functions from `aot.bin`");
        //     let reader = std::io::BufReader::new(f);
        //     let aot_map: HashMap<u32, Vec<u8>> = bincode::deserialize_from(reader).unwrap();
        //     aot_map.into_iter()
        //     .map(|(k, v)| {
        //         let func = {
        //             let ptr = v.leak() as &'static mut [u8];
        //             let ptr = ptr.as_ptr();
        //             let code_fn = unsafe {
        //                 core::mem::transmute::<_, fn(*const *const u8, *const u8, *mut u8, i64)->()>(ptr)
        //             };
        //             code_fn
        //         };
        //         (k, func)
        //     })
        //     .collect()
        // } else {
            info!("start AOT compilation");
            let mut map = HashMap::new();
            let assembler = Assembler::default();
            let mut jit = assembler.create_jit();
            for n in 2..=JIT_MAX_NODES {
                for l in 1..(1 << n) {
                    let mut louds = (n as u32) << 28;
                    louds |= l << 14;
                    for c in 0..(1 << n) {
                        let louds = louds | c;
                        let mut builder = Louds2Boolean::new(louds);
                        let boolean = builder.build();
                        let leaf_num = builder.leaf_num();
                        if let Some(b) = boolean {
                            let gen_func = jit_short_circuit_primitive(&assembler, b, leaf_num, louds).unwrap();
                            let gen_func = jit.compile(gen_func).unwrap();
                            let code_fn = unsafe {
                                core::mem::transmute::<_, fn(*const *const u8, *const u8, *mut u8, i64)>(gen_func)
                            };
                            debug!("louds: {:b}", louds);
                            // let code_fn = unsafe {
                            //     core::mem::transmute::<_, fn(*const *const u8, *const u8, *mut u8, i64)->()>()
                            // };
                            map.insert(louds, code_fn);
                        }
                    }
                }
            }
            // let file = File::create("aot3.bin").expect("Unable to create file.");
            // let writer = std::io::BufWriter::new(file);
            // bincode::serialize_into(writer, &map).expect("Unable to serialize data");
            // map.into_iter()
            // .map(|(k, v)| {
            //     let func = {
            //         let ptr = v.as_ptr();
            //         let code_fn = unsafe {
            //             core::mem::transmute::<_, fn(*const *const u8, *const u8, *mut u8, i64)->()>(ptr)
            //         };
            //         std::mem::forget(v);
            //         code_fn
            //     };
            //     (k, func)
            // })
            // .collect()
            map
        // }
    };
}

// pub fn aot_short_circuit(expr: Boolean, leaf_num: usize) -> Result<Vec<u8>> {
//     let assembler = Assembler::default();
//     let gen_func = jit_short_circuit_primitive(&assembler, expr, leaf_num)?;

//     let mut jit = assembler.create_jit();
//     jit.compile_to_bytes(gen_func)
// }

pub fn jit_short_circuit(expr: Boolean, leaf_num: usize) -> Result<fn(*const *const u8, *const u8, *mut u8, i64)> {
    let assembler = Assembler::default();
    let gen_func = jit_short_circuit_primitive(&assembler, expr, leaf_num, 0)?;

    let mut jit = assembler.create_jit();
    let gen_func = jit.compile(gen_func)?;
    let code_fn = unsafe {
        core::mem::transmute::<_, fn(*const *const u8, *const u8, *mut u8, i64)->()>(gen_func)
    };
    Ok(code_fn)
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use tracing::{Level, info};

    use crate::{utils::Result, jit::{api::Assembler, ast::U16, AOT_PRIMITIVES}};
    use super::{jit::JIT, api::GeneratedFunction};
    
    #[test]
    fn iterative_fib() -> Result<()> {
        let expected = r#"fn iterative_fib_0(n: i16) -> r: i16 {
    if n == 0 {
        r = 0;
    } else {
        n = n - 1;
        let a: i16;
        a = 0;
        r = 1;
        while n != 0 {
            let t: i16;
            t = r;
            r = r + a;
            a = t;
            n = n - 1;
        }
    }
}"#;
        let assembler = Assembler::default();
        let mut builder = assembler
            .new_func_builder("iterative_fib")
            .param("n", U16)
            .ret("r", U16);
        let mut fn_body = builder.enter_block();

        fn_body.if_block(
            |cond| cond.eq(cond.id("n")?, cond.lit_u16(0 as u16)),
            |t| {
                t.assign("r", t.lit_u16(0 as u16))?;
                Ok(())
            },
            |e| {
                e.assign("n", e.sub(e.id("n")?, e.lit_u16(1 as u16))?)?;
                e.declare_as("a", e.lit_u16(0 as u16))?;
                e.assign("r", e.lit_u16(1 as u16))?;
                e.while_block(
                    |cond| cond.ne(cond.id("n")?, cond.lit_u16(0 as u16)),
                    |w| {
                        w.declare_as("t", w.id("r")?)?;
                        w.assign("r", w.add(w.id("r")?, w.id("a")?)?)?;
                        w.assign("a", w.id("t")?)?;
                        w.assign("n", w.sub(w.id("n")?, w.lit_u16(1 as u16))?)?;
                        Ok(())
                    },
                )?;
                Ok(())
            },
        )?;

        let gen_func = fn_body.build();
        assert_eq!(format!("{}", &gen_func), expected);
        let mut jit = assembler.create_jit();
        assert_eq!(55, run_iterative_fib_code(&mut jit, gen_func, 10)?);
        Ok(())
    }

    unsafe fn run_code<I, O>(
        jit: &mut JIT,
        code: GeneratedFunction,
        input: I,
    ) -> Result<O> {
        // Pass the string to the JIT, and it returns a raw pointer to machine code.
        let code_ptr = jit.compile(code)?;
        // Cast the raw pointer to a typed function pointer. This is unsafe, because
        // this is the critical point where you have to trust that the generated code
        // is safe to be called.
        let code_fn = core::mem::transmute::<_, fn(I) -> O>(code_ptr);
        // And now we can call it!
        Ok(code_fn(input))
    }

    fn run_iterative_fib_code(
        jit: &mut JIT,
        code: GeneratedFunction,
        input: isize,
    ) -> Result<isize> {
        unsafe { run_code(jit, code, input) }
    }

    #[test]
    fn test_aot_primitives() {
        tracing_subscriber::fmt()
        .with_max_level(Level::INFO).init();
        let timer = Instant::now();
        info!("len: {:}", AOT_PRIMITIVES.len());
        info!("space: {:} bytes", std::mem::size_of_val(&AOT_PRIMITIVES));
        info!("consume: {:}", timer.elapsed().as_secs());
        assert!(AOT_PRIMITIVES.contains_key(&0b0111_00000000101001_00000000001001));
        // AOT_PRIMITIVES.keys()
        // .for_each(|v| {
        //     println!("{:b}", v);
        // });
    }

    #[test]
    fn test_aot_res() {
        // allocate memory for result
        let test1 = vec![0x31, 0x0];
        let test2 = vec![0x11, 0x23];
        let test3 = vec![0x21, 0xFF];
        let test4 = vec![0x21, 0x12];
        let batch = vec![
            test1.as_ptr(),
            test2.as_ptr(),
            test3.as_ptr(),
            test4.as_ptr(),
        ];
        let mut res: Vec<u8> = vec![0; 2];

        let func = AOT_PRIMITIVES[&0b1000000000000000100000000000000];
        func(
            batch.as_ptr(),
            res.as_ptr(),
            res.as_mut_ptr(),
            2,
        );
        println!("res: {:?}", res);
    }
}
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::OptimizationLevel;
use inkwell::types::IntType;

use std::error::Error;

use crate::jit::Boolean;

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type SumFunc = unsafe extern "C" fn(u64, u64, u64) -> u64;

type AOTPrimitive = unsafe extern "C" fn(*const *const u64, *const u64, *mut u64, u64);

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn jit_compile_sum(&self) -> Option<JitFunction<SumFunc>> {
        let i64_type = self.context.i64_type();
        let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into(), i64_type.into()], false);
        let function = self.module.add_function("sum", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let x = function.get_nth_param(0)?.into_int_value();
        let y = function.get_nth_param(1)?.into_int_value();
        let z = function.get_nth_param(2)?.into_int_value();

        let sum = self.builder.build_int_add(x, y, "sum").unwrap();
        let sum = self.builder.build_int_add(sum, z, "sum").unwrap();

        self.builder.build_return(Some(&sum)).unwrap();

        unsafe { self.execution_engine.get_function("sum").ok() }
    }

    fn aot_primitive(
        &self,
        jit_expr: Boolean,
        leaf_num: usize,
    ) -> Option<JitFunction<AOTPrimitive>> {
        let target = self.execution_engine.get_target_data();
        let ptr_type = self.context.ptr_sized_int_type(target, None);
        let u64_type = self.context.i64_type();
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into(), u64_type.into()], false);
        let function = self.module.add_function("primitive", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let batch = function.get_nth_param(0)?.into_pointer_value();
        let init_v = function.get_nth_param(1)?.into_pointer_value();
        let result = function.get_nth_param(2)?.into_pointer_value();
        let len = function.get_nth_param(3)?.into_int_value();

        for i in 0..leaf_num {
            unsafe {
                self.builder.build_gep(
                    ptr_type,
                    batch,
                    &[u64_type.const_int(i as u64, false)], 
                    &format!("off{}", i)
                ).unwrap();
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use inkwell::{context::Context, OptimizationLevel};

    use super::CodeGen;

    #[test]
    fn text_sum() {
        let context = Context::create();
        let module = context.create_module("sum");
        let execution_engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive).unwrap();
        let code_gen = CodeGen {
            context: &context,
            module,
            builder: context.create_builder(),
            execution_engine,
        };

        let sum = code_gen.jit_compile_sum().ok_or("Unable to JIT compile `sum`").unwrap();

        let x = 1u64;
        let y = 2u64;
        let z = 3u64;

        unsafe {
            println!("{} + {} + {} = {}", x, y, z, sum.call(x, y, z));
            assert_eq!(sum.call(x, y, z), x + y + z);
        }
    }

}
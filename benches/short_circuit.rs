#![feature(sync_unsafe_cell)]
use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use datafusion::physical_plan::expressions::Column;
use velosearch::{jit::{ast::Predicate, jit_short_circuit, Boolean}, ShortCircuit, physical_expr::{boolean_eval::{SubPredicate, PhysicalPredicate, Chunk}, Primitives, BooleanEvalExpr}};
use futures::stream::iter;
use rand::seq::IteratorRandom;

#[inline]
fn gen_batch(sel: f64) -> Vec<[u64; 8]> {
    let v: Vec<usize> = (0..512).collect();
    let num: usize = (512. * sel) as usize;
    let mut rng = rand::thread_rng();
    let a1 = v.iter().cloned().choose_multiple(&mut rng, num);
    let a2 = v.iter().cloned().choose_multiple(&mut rng, num);
    let a3 = v.iter().cloned().choose_multiple(&mut rng, num);
    let mut b1: [u64; 8] = [0; 8];
    let mut b2: [u64; 8] = [0; 8];
    let mut b3: [u64; 8] = [0; 8];
    for item in a1 {
        b1[item >> 6] |= 1 << (item % 64);
    }
    for item in a2 {
        b2[item >> 6] |= 1 << (item % 64);
    }
    for item in a3 {
        b3[item >> 6] |= 1 << (item % 64);
    }
    vec![b1, b2, b3]
}

fn short_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("short_circuit");
    
    let predicate1 = SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("a0", 0)) });
    let predicate2 = SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("a1", 1)) });
    let predicate3 = SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("a2", 2)) });
    let predicate = BooleanEvalExpr::new(Some(PhysicalPredicate::And { args: vec![
        predicate1,
        predicate2,
        predicate3,
    ] }));

    let compiled_predicate = Predicate::And {
        args: vec![
            Predicate::Leaf { idx: 0 },
            Predicate::Leaf { idx: 1 },
            Predicate::Leaf { idx: 2 },
        ]
    };
    let short_circuit = jit_short_circuit(Boolean { predicate: compiled_predicate, start_idx: 0 }, 3).unwrap();
    for i in 0..=50 {
        let sel: f64 = 0.02 * i as f64;
        group.bench_with_input(BenchmarkId::new("short_circuit", format!("{:.2}", sel)), &sel, |b, sel| {
            let data = gen_batch(*sel);
            let batch: Vec<*const u8> = data.iter()
                .map(|v| v.as_ptr() as *const u8)
                .collect();
            let mut res: [u64; 8] = [0; 8];
            let init: [u64; 8] = [u64::MAX; 8];
            b.iter(|| {
                short_circuit(batch.as_ptr() as *const *const u8, init.as_ptr() as *const u8, res.as_mut_ptr() as *mut u8, 8);
            })
        });

        group.bench_with_input(BenchmarkId::new("vectorized", format!("{:.2}", sel)), &sel, |b, sel| {
            let data = gen_batch(*sel);
            let batch: Vec<Option<Vec<Chunk>>> = data.iter()
                .map(|v| {
                    Some(vec![Chunk::Bitmap(&(*v))])
                })
                .collect();
            let predicate = predicate.predicate.as_ref().unwrap().get();
            let predicate_ref = unsafe {predicate.as_ref().unwrap() };
            b.iter(|| {
                predicate_ref.eval_avx512(&batch, None, true, 1).unwrap();
            })
        });
    }
}

criterion_group!(benches, short_circuit);
criterion_main!(benches);
use arrow::compute::kernels::filter::filter;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use datafusion::arrow::{self, array::{BooleanArray, UInt32Array}, compute::and};
use rand::{Rng, distributions::Bernoulli};


#[inline]
fn bitwise_and(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrow_bitwise_and");
    for i in 1..=10 {
        let size = i * 100_000;
        let mut rng = rand::thread_rng();
        let range = Bernoulli::new(0.05).unwrap();
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let a1 = BooleanArray::from((0..size).into_iter().map(|_| (if rng.sample(&range) {true} else {false})).collect::<Vec<bool>>());
            let a2 = BooleanArray::from((0..size).into_iter().map(|_| (if rng.sample(&range) {true} else {false})).collect::<Vec<bool>>());
            let place_holder = UInt32Array::from_iter((0..size).into_iter().map(|v| Some(v as u32)));
            b.iter(|| {
                filter(&place_holder, &and(&a1, &a2).unwrap()).unwrap();
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bitwise_and);
criterion_main!(benches);

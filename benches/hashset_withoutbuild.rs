use std::collections::HashSet;

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::{seq::IteratorRandom, distributions::Uniform, Rng};

#[inline]
fn bitwise_and(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashset_withoutbuild");
    for i in 1..=10 {
        let size = i * 100_000;
        let mut rng = rand::thread_rng();
        let v: Vec<u32> = (0..size).collect();
        let range = Uniform::new(0.05, 0.3);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let a1_rate = rng.sample(&range);
            let a1 = v.iter().cloned().choose_multiple(&mut rng, (size as f64 * a1_rate) as usize);
            let a2_rate = rng.sample(&range);
            let a2 = v.iter().cloned().choose_multiple(&mut rng, (size as f64 * a2_rate) as usize);
            let a1_hash: HashSet<u32> = HashSet::from_iter(a1.iter().cloned());
            let a2_hash: HashSet<u32> = HashSet::from_iter(a2.iter().cloned());
            b.iter(|| {
                let res : HashSet<_> = a1_hash.intersection(&a2_hash).cloned().collect();
                res
        });
        });
    }
    group.finish();
}

criterion_group!(benches, bitwise_and);
criterion_main!(benches);

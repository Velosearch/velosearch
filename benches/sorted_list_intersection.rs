use sorted_iter::*;
use sorted_iter::assume::*;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::seq::IteratorRandom;

#[inline]
fn sorted_list_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorted_list_intersection");
    for i in 1..=10 {
        let size =  i * 100_000;
        let mut rng = rand::thread_rng();
        // let range = Uniform::new(0.05, 0.3);
        let v: Vec<u32> = (0..size).collect();
        let rate = 0.05;
        group.bench_with_input(BenchmarkId::from_parameter(size/10), &size, |b, &size| {
            // let a1_rate = rng.sample(&range);
            let a1 = v.iter().cloned().choose_multiple(&mut rng, (size as f64 * rate) as usize);
            // let a2_rate = rng.sample(&range);
            let a2 = v.iter().cloned().choose_multiple(&mut rng, (size as f64 * rate) as usize);
            b.iter(|| {
                let a1_iter = a1.iter().assume_sorted_by_item();
                let a2_iter = a2.iter().assume_sorted_by_item();
                let res: Vec<u32> = a1_iter.intersection(a2_iter).cloned().collect();
                res
            })
        });
    }
    group.finish();
}

criterion_group!(benches, sorted_list_intersection);
criterion_main!(benches);
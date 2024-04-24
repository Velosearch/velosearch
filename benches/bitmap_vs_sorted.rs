
#![feature(stdsimd)]
use std::{slice::Iter, arch::x86_64::{__m512i, _mm512_mask_compressstoreu_epi16, _mm512_mask_compressstoreu_epi32}};

use rayon::{prelude::*, iter};
use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use rand::seq::IteratorRandom;
use sorted_iter::*;
use sorted_iter::assume::*;
use datafusion::arrow::{array::{BooleanArray, Array}, compute::and};

const NUM_LANES: usize = 16;
const IDS: __m512i = from_u32x16([
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15,
]);

#[inline]
fn bitwise_and(b1: &BooleanArray, b2: &BooleanArray) -> Vec<u32> {
    let and_res = and(&b1, &b2).unwrap();
    let size = and_res.len();
    let (prefix, buffer, suffix) = unsafe { and_res.data().buffers()[0].align_to::<u16>() };
    assert!(prefix.len() == 0, "Len of prefix should be 0");
    assert!(suffix.len() == 0, "Len of suffix should be 0");
    let mut output: Vec<u32> = Vec::with_capacity(size);
    let mut output_end = output.as_mut_ptr();
    buffer
        .into_iter()
        .filter(|v| **v != 0)
        .for_each(|v| {
            unsafe {
                _mm512_mask_compressstoreu_epi32(output_end as *mut u8, *v, IDS);
                output_end = output_end.offset(v.count_ones() as isize);
            }
        });
    output
}

fn _split_range(r: &[u32]) -> (&[u32], Option<&[u32]>) {
    let len = r.len();
    if len <= 1 { return (r, None); }

    let midpoint = len / 2;

    (&r[0..midpoint], Some(&r[midpoint..len]))
}

#[inline]
fn _bitwise_and_parall2(b1: &BooleanArray, b2: &BooleanArray) {
    let and_res = and(&b1, &b2).unwrap();
    let size = and_res.len();
    let (prefix, buffer, suffix) = unsafe { and_res.data().buffers()[0].align_to::<u32>() };
    assert!(prefix.len() == 0, "Len of prefix should be 0");
    assert!(suffix.len() == 0, "Len of suffix should be 0");
    let _res: Vec<u16> = iter::split(buffer, _split_range)
        .map(|v| {
            let mut output: Vec<u16> = Vec::with_capacity(size / 2 + 1);
            let mut output_end = output.as_mut_ptr();
            for i in v {
                unsafe {
                    _mm512_mask_compressstoreu_epi16(output_end as *mut u8, *i, IDS);
                    output_end = output_end.offset(i.count_ones() as isize);
                }
            }
            output
        })
        .flatten()
        .collect();
}

#[inline]
fn sorted_list_intersection(a1: Iter<u32>, a2: Iter<u32>) {
    let a1_iter = a1.assume_sorted_by_item();
    let a2_iter = a2.assume_sorted_by_item();
    let _res: Vec<u32> = a1_iter.intersection(a2_iter).cloned().collect();
}

fn bench_itersection(c: &mut Criterion) {
    let num: usize = 4096;
    let mut group = c.benchmark_group("bitmap_vs_sorted");
    let v: Vec<u32> = (0..num as u32).collect();
    let mut rng = rand::thread_rng();
    for i in 1..=20 {
        let sel = i as f64 * 0.05;
        let a1 = v.iter().cloned().choose_multiple(&mut rng, (num as f64 * (i as f64 * 0.05)) as usize);
        let a2 = v.iter().cloned().choose_multiple(&mut rng, (num as f64 * (i as f64 * 0.05)) as usize);
        group.bench_with_input(BenchmarkId::new("Sorted_list", format!("{:.2}", sel)), &(&a1, &a2),
            |b, p| b.iter(|| sorted_list_intersection(p.0.iter(), p.1.iter())));
        let mut b1 = vec![false; num];
        let mut b2 = vec![false; num];
        for item in &a1 {
            b1[*item as usize] = true;
        }
        for item in &a2 {
            b2[*item as usize] = true;
        }
        let b1 = BooleanArray::from(b1);
        let b2 = BooleanArray::from(b2);
        group.bench_with_input(BenchmarkId::new("Bitmap", format!("{:.2}", sel)), &(&b1, &b2),
            |b, p| b.iter(|| bitwise_and(p.0, p.1)));
        
        // group.bench_with_input(BenchmarkId::new("Bitmap_Parall", format!("{:.2}", sel)), &(&b1, &b2),
        //     |b, p| b.iter(|| bitwise_and_parall2(p.0, p.1)));
    }
    group.finish();
}

criterion_group!(benches, bench_itersection);
criterion_main!(benches);

const fn from_u32x16(vals: [u32; NUM_LANES]) -> __m512i {
    union U8x64 {
        vector: __m512i,
        vals: [u32; NUM_LANES],
    }
    unsafe { U8x64 { vals }.vector }
}
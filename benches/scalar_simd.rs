#![feature(stdsimd)]
use std::arch::x86_64::{_pext_u64, _blsr_u64, _mm512_loadu_epi8, _mm512_mask_compressstoreu_epi8, __m512i, _mm256_mask_compressstoreu_epi8, __m256i, _mm256_loadu_epi8, _mm256_mask_compressstoreu_epi32, _mm256_mask_compressstoreu_epi16, _mm512_mask_compressstoreu_epi64};

use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
use rand::seq::IteratorRandom;
use lazy_static::lazy_static;

const COMPRESS_INDEX: [u8; 64] =  [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63,
    ];

lazy_static!{
    pub static ref compress_index: __m512i = unsafe {
        _mm512_loadu_epi8(COMPRESS_INDEX.as_ptr() as *const i8)
    };
}

fn scalar(distri: u64, mask: u64) {
    let valid_num = distri.count_ones() as usize;
    let mut write_mask = unsafe { _pext_u64(mask, distri) };
    let valid_mask = unsafe { _pext_u64(distri, mask) };
    let mut posting = Vec::with_capacity(valid_num);

    for i in 0..valid_num {
        if valid_mask & (1 << i) != 0 {
            let pos = write_mask.trailing_ones() as usize;
            write_mask = clear_lowest_set_bit(write_mask);
            posting.push(pos);
        } else {
            posting.push(0);
        }
    }
}

fn simd(distri: u64, mask: u64) {
    let valid_num = distri.count_ones() as usize;
    let write_mask = unsafe { _pext_u64(mask, distri) };
    let write_count = write_mask.count_ones() as usize;
    let valid_mask = unsafe { _pext_u64(distri, mask) };
    let mut posting = vec![0; valid_num];

    let mut write_index: Vec<u8> =  Vec::with_capacity(write_count);
    let mut posting_index: Vec<u8> = Vec::with_capacity(write_count);

    unsafe {
        _mm512_mask_compressstoreu_epi64(write_index.as_mut_ptr() as *mut u8, valid_mask as u8, *compress_index);
        _mm512_mask_compressstoreu_epi64(write_index.as_mut_ptr() as *mut u8, valid_mask as u8, *compress_index);
        // _mm512_mask_compressstoreu_epi8(posting_index.as_mut_ptr() as *mut u8, write_mask, *compress_index);
        // write_index.set_len(write_count);
        posting_index.set_len(write_count);
    }
    // for (w, p) in write_index.into_iter().zip(posting_index.into_iter()) {
    //     posting[w as usize] = p as usize;
    // }
}

#[inline]
fn clear_lowest_set_bit(v: u64) -> u64 {
    unsafe { _blsr_u64(v) }
}

fn bench_scalar_simd(c: &mut Criterion) {
    let num: usize = 64;
    let mut group = c.benchmark_group("bench_scalar_simd");
    let v: Vec<u32> = (0..num as u32).collect();
    let mut rng = rand::thread_rng();
    for i in 1..=64 {
        let a1_v = v.iter().cloned().choose_multiple(&mut rng, i);
        let a2_v = v.iter().cloned().choose_multiple(&mut rng, i);
        let mut a1: u64 = 0;
        let mut a2: u64 = 0;
        for i in a1_v {
            a1 |= 1 << i;
        }
        for i in a2_v {
            a2 |= 1 << i;
        }
        group.bench_with_input(BenchmarkId::new("scalar", format!("{:}", i)), &(a1, a2),
            |b, _| b.iter(|| scalar(a1, a2)));
        group.bench_with_input(BenchmarkId::new("simd", format!("{:}", i)), &(a1, a2),
            |b, _| b.iter(|| simd(a1, a2)));
    }
    group.finish();
}

criterion_group!(benches, bench_scalar_simd);
criterion_main!(benches);
use std::{arch::x86_64::{ __m512i, _mm512_and_epi64, _mm512_loadu_epi64, _mm512_or_epi64}, ptr::read_unaligned};


pub union U64x8 {
    pub vector: __m512i,
    pub vals: [u64; 8],
}

pub fn bitwise_and(lhs: &mut [u64], rhs: &[u64]) {
    if !cfg!(feature = "scalar") {
        let lanes_num = lhs.len() as isize / 8;
        for i in 0..lanes_num {
            unsafe {
                let left = _mm512_loadu_epi64(lhs.as_ptr().offset(8 * i) as *const i64);
                let right = _mm512_loadu_epi64(rhs.as_ptr().offset(8 * i) as *const i64);
                let vector = _mm512_and_epi64(left, right);
                *(lhs[(i as usize *8)..(i as usize*8 + 8)].as_mut_ptr() as *mut [u64; 8]) = U64x8 { vector }.vals;
            }
        }
        for i in (lanes_num as usize * 8)..lhs.len() {
            lhs[i] = unsafe { lhs.get_unchecked(i) & rhs.get_unchecked(i)};
        }
    } else {
        lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(l, h)| {
            *l = *l & *h;
        });
    }
}

pub fn bitwise_or(lhs: &mut [u64], rhs: &[u64]) {
    if !cfg!(feature = "scalar") {
        let lanes_num = lhs.len() as isize / 8;
        for i in 0..lanes_num {
            unsafe {
                let left = _mm512_loadu_epi64(lhs.as_ptr().offset(8 * i) as *const i64);
                let right = _mm512_loadu_epi64(rhs.as_ptr().offset(8 * i) as *const i64);
                let vector = _mm512_or_epi64(left, right);
                *(lhs[(i as usize *8)..(i as usize*8 + 8)].as_mut_ptr() as *mut [u64; 8]) = U64x8 { vector }.vals;
            }
        }
        for i in (lanes_num as usize * 8)..lhs.len() {
            lhs[i] = unsafe { lhs.get_unchecked(i) | rhs.get_unchecked(i)};
        }
    } else {
        lhs.iter_mut()
        .zip(rhs.iter())
        .for_each(|(l, h)| {
            *l = *l | *h;
        });
    }
}

pub fn bitwise_and_batch(lhs: &[u64], rhs: &[u64]) -> __m512i {
    if cfg!(feature = "scalar") {
        let res: Vec<u64> = lhs.into_iter()
        .zip(rhs)
        .map(|(&l, &r)| {
            l & r
        })
        .collect();
        // unsafe { _mm512_loadu_epi64(res.as_ptr() as *const i64)}
        unsafe { 
            read_unaligned(res.as_ptr() as *const __m512i)
        }
    } else {
        unsafe {
            let left = _mm512_loadu_epi64(lhs.as_ptr() as *const i64);
            let right = _mm512_loadu_epi64(rhs.as_ptr() as *const i64);
            _mm512_and_epi64(left, right)
        }
    }
}

pub fn bitwise_or_batch(lhs: &[u64], rhs: &[u64]) -> __m512i {
    if cfg!(feature = "scalar") {
        let res: Vec<u64> = lhs.into_iter()
        .zip(rhs)
        .map(|(&l, &r)| {
            l | r
        })
        .collect();
        unsafe { _mm512_loadu_epi64(res.as_ptr() as *const i64)}
    } else {
        unsafe {
            let left = _mm512_loadu_epi64(lhs.as_ptr() as *const i64);
            let right = _mm512_loadu_epi64(rhs.as_ptr() as *const i64);
            _mm512_or_epi64(left, right)
        }
    }
}
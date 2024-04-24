#![feature(portable_simd)]
#![feature(stdsimd)]

mod fst;
mod louds_dense;
mod louds_sparse;
mod fst_builder;
mod bitvector;
mod label_vector;

const K_INCLUDE_DENSE: bool = true;
const K_SPARSE_DENSE_RATIO: u32 = 1;
const K_FANOUT: usize = 256;
const K_WORD_SIZE: usize = 8;
const K_TERMINATOR: u8 = 0;

const SELECT_INTERVAL: usize = 512;


type LabelsT = u8;
type PosT = usize;
type LevelT = usize;

pub use fst_builder::FSTBuilder;
pub use bitvector::BitVector;
use label_vector::LabelVector;
pub use fst::FST;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

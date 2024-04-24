use std::{arch::x86_64::{_mm512_reduce_add_epi64, _mm512_loadu_epi64, _pdep_u32, _mm512_popcnt_epi64}, vec, fmt::Debug};

use arrow::{array::{BooleanArray, BooleanBufferBuilder, ArrayData}, datatypes::DataType};
use tracing::debug;

use crate::{PosT, LevelT, K_FANOUT, K_WORD_SIZE, SELECT_INTERVAL, LabelsT, K_TERMINATOR};

#[derive(Clone)]
pub struct BitVector {
    num_bits: usize,
    bits: BooleanArray,
    lut: Option<Vec<usize>>,
}

impl BitVector {
    pub fn new(
        bitvector_per_level: Vec<Vec<u8>>,
        num_bits_per_level: &Vec<PosT>,
        start_level: LevelT,
        end_level: LevelT,
    ) -> Self {
        let num_bits = total_num_bits(&num_bits_per_level, start_level, end_level) as usize;
        let bits = concat_bitvectors(
            BooleanBufferBuilder::new(num_bits),
            bitvector_per_level,
            num_bits_per_level,
            start_level,
            end_level,
        );
        let mut vector = Self {
            num_bits,
            bits,
            lut: None,
        };
        vector.init_select_lut();
        vector
    }

    pub fn new_with_lut(
        bitvector_per_level: Vec<Vec<u8>>,
        num_bits_per_level: &Vec<PosT>,
        start_level: LevelT,
        end_level: LevelT,
    ) -> Self {
        Self::new(
            bitvector_per_level,
            num_bits_per_level,
            start_level,
            end_level,
        )
    }

    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    pub fn read_bit(&self, pos: PosT) -> bool {
        self.bits.value(pos as usize)
    }

    /// Counts the number of 1's in the bitvector up to position pos.
    /// pos is zero-based; count is one-based.
    /// E.g., for bitvector: 100101000, rank(3) = 1
    pub fn rank(&self, pos: usize) -> usize {
        assert!(pos < self.num_bits);
        assert!(self.lut.is_some(), "Before using rank, lut must be init");
        let lut = self.lut.as_ref().unwrap();
        let idx = pos / SELECT_INTERVAL;
        return lut[idx]
            + self.bits
                .values().count_set_bits_offset(idx * SELECT_INTERVAL, (pos + 1) - idx * SELECT_INTERVAL);
    }

    /// Returns the position of the rank-1th 1 bit.
    /// position is zero-based; rank is one -based.
    /// E.g., for bitvector: 100101000, select(2) = 5
    pub fn select(&self, rank: PosT) -> PosT {
        assert!(self.lut.is_some(), "Before using select, lut must be init");
        let lut = self.lut.as_ref().unwrap();
        let idx = match lut.binary_search(&rank) {
            Ok(idx) => idx - 1,
            Err(idx) => idx - 1,
        };
        let diff = rank - lut[idx];
        let mut cnt = 0;
        {
            for (i, v) in self.bits.values().slice(idx * SELECT_INTERVAL / 8).iter().enumerate() {
                let set_num = v.count_ones() as usize;
                if cnt + set_num >= diff {
                    return unsafe {
                        _pdep_u32(1 << (diff - cnt - 1), *v as u32).trailing_zeros() as usize
                        + idx * SELECT_INTERVAL + i * 8
                    }
                }
                cnt += set_num;
            }
            unreachable!()
        }
    }

    pub fn num_set_bits_in_dense_node(&self, node_number: PosT, _label: &mut usize) -> usize {
        let mut set_bits = 0;
        for _ in 0..4 {
            set_bits += self.bits.values().count_set_bits_offset(node_number * K_FANOUT + K_WORD_SIZE, K_WORD_SIZE);

        }
        return set_bits;
    }

    pub fn distance_to_next_set_bit(&self, pos: PosT) -> PosT {
        assert!(pos < self.num_bits);
        let mut distance = 1;

        let mut byte_id = (pos + 1) / 8;
        let offset = (pos + 1) % 8; 
        let mut test_byte = self.bits.values()[byte_id] >> offset;
        if test_byte > 0 {
            return distance + test_byte.trailing_zeros() as usize;
        } else {
            if byte_id == self.num_bytes() - 1 {
                return self.num_bits - pos;
            }
            distance += 8 - offset;
        }

        while byte_id < (self.num_bits / 8) - 1 {
            byte_id += 1;
            test_byte = self.bits.values()[byte_id];
            if test_byte > 0 {
                return distance + test_byte.trailing_zeros() as usize;
            }
            distance += 8;
        }
        return distance;
    }

    fn num_bytes(&self) -> usize {
        if self.num_bits % 8 == 0 {
            self.num_bits / 8
        } else {
            self.num_bits / 8 + 1
        }
    }

    fn init_select_lut(&mut self) {
        let mut lut = vec![0];
        let mut cur = 0;
        while cur + SELECT_INTERVAL <= self.num_bits {
            let bit_slice = self.bits.values().slice_with_length(cur / 8, SELECT_INTERVAL / 8);
            let simd_slice = unsafe {
                bit_slice.align_to::<u64>()
            };

            lut.push(lut.last().unwrap() + popcnt_512(simd_slice.1));
            cur += SELECT_INTERVAL;
        }
        if cur < self.num_bits - 1 {
            let reminder_cnt = self.bits
                .values()
                .slice(cur / 8)
                .into_iter()
                .map(|b| b.count_ones() as usize)
                .sum();
            lut.push(reminder_cnt)
        }
        self.lut = Some(lut);
    }

}

impl Debug for BitVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.bits)
    }
}

#[inline]
fn popcnt_512(bit_slice: &[u64]) -> usize {
    let x = unsafe {
        let ptr = _mm512_loadu_epi64(bit_slice.as_ptr() as *const i64);
        let res = _mm512_popcnt_epi64(ptr);
        _mm512_reduce_add_epi64(res)
    };
    x as usize
}

#[inline]
fn total_num_bits(
    num_bits_per_level:&Vec<PosT>,
    start_level: LevelT,
    end_level: LevelT,
) -> PosT {
    num_bits_per_level[start_level..end_level]
        .into_iter()
        .sum()
}

#[inline]
fn concat_bitvectors(
    mut builder: BooleanBufferBuilder,
    bitvector_per_level: Vec<Vec<u8>>,
    num_bits_per_level: &Vec<PosT>,
    start_level: LevelT,
    end_level: LevelT,
) -> BooleanArray {
    let mut total_bits = 0;
    for level in start_level..end_level {
        let num_bits = num_bits_per_level[level];
        if num_bits == 0 {
            continue;
        }
        total_bits += num_bits;
        builder.append_packed_range(0.. num_bits as usize, &bitvector_per_level[level]);
    }
    let builder = ArrayData::builder(DataType::Boolean)
        .len(total_bits as usize)
        .add_buffer(builder.finish());
    let array_data = unsafe { builder.build_unchecked() };
    BooleanArray::from(array_data)
}

#[cfg(test)]
mod test {
    use std::arch::x86_64::_pdep_u64;
    use rand::{Rng, thread_rng};

    use arrow::array::{BooleanBufferBuilder, BooleanArray};

    use crate::BitVector;

    use super::{concat_bitvectors, popcnt_512};

    #[test]
    fn bitvector_concat() {
        let builder = BooleanBufferBuilder::new(32);
        let bitvector_per_level = vec![
            vec![0b0101_0011, 0b1101_0101],
            vec![0b0010_1101, 0b0010_0100],
        ];
        let num_bits_per_level = vec![16, 14];
        let start_level = 0;
        let end_level = 2;
        let vector = concat_bitvectors(builder, bitvector_per_level, &num_bits_per_level, start_level, end_level);
        assert_eq!(
            vector.values().as_slice(),
            &[0b0101_0011, 0b1101_0101, 0b0010_1101, 0b0010_0100],
        )
    }

    #[test]
    fn bitvector_rank() {
        let bitvector_per_level = vec![
            vec![0b0101_0011, 0b1101_0101],
        ];
        let num_bits_per_level = vec![16];
        let vector = BitVector::new(
            bitvector_per_level,
            &num_bits_per_level,
            0,
            1
        );
        assert_eq!(vector.rank(5), 3);
        assert_eq!(vector.rank(10), 6);
    }

    #[test]
    fn bitvector_popcnt() {
        let bit_slice = &[
            0b0010_0100, 0b0010_1101, 0b0101_0011, 0b1101_0101,
            0b1101_0011, 0b0010_0100, 0b1101_0101, 0b1101_0011,
        ];
        let res = popcnt_512(bit_slice);
        let real: usize = bit_slice.into_iter()
            .map(|b| b.count_ones() as usize)
            .sum();
        assert_eq!(res, real);
    }

    #[test]
    fn bitvector_select_simple() {
        let bitvector_per_level = vec![
            vec![0b0101_0011, 0b1101_0101],
            vec![0b0010_1101, 0b0010_0100],
        ];
        let vector = BitVector::new_with_lut(
            bitvector_per_level,
            &vec![16, 16],
            0,
            2,
        );
        assert_eq!(vector.select(3), 4);
        assert_eq!(vector.select(7), 12);
    }

    #[test]
    fn bitvector_select_bug() {
        let bit_vec = vec![vec![0b0000_1011]];
        let vec = BitVector::new_with_lut(
            bit_vec,
            &vec![5],
            0,
            1,
        );
        assert_eq!(vec.select(3), 3);
    }

    #[test]
    fn bitvector_select_large() {
        let mut rng = thread_rng();
        let v: Vec<bool> = (0..1056).map(|_| rng.gen()).collect();
        let test_vec = BooleanArray::from(v.clone());
        let bitvector = BitVector::new_with_lut(
            vec![test_vec.values().as_slice().to_vec()],
        &vec![1024],
            0,
            1
        );
        let low = test_vec.values().count_set_bits_offset(0, 64) + 1;
        let high = low + test_vec.values().count_set_bits_offset(64, 64);

        // test case for aligned bytes
        for _ in 0..10 {
            let case = rng.gen_range(low..high);
            let res = bitvector.select(case);
            let real = test_vec.values().count_set_bits_offset(0, res + 1);
            assert_eq!(case, real);
        }

        // test case for unaligned bytes
        for _ in 0..10 {
            let case = rng.gen_range(high+1..high+3);
            let res = bitvector.select(case);
            let real = test_vec.values().count_set_bits_offset(0, res + 1);
            assert_eq!(case, real);
        }
    }

    #[test]
    fn intrinsic_test() {
        println!("{:b}", 0b0010_0100 as u32);
        println!("{:b}", unsafe {
          _pdep_u64(1 << 1, 0b00100100)
        });

    }

    #[test]
    fn bitvector_distance() {
        let bitvector_per_level = vec![
            vec![0b0101_0011, 0b1101_0101],
            vec![0b0010_1101, 0b0010_0100],
        ];
        let vector = BitVector::new(
            bitvector_per_level,
            &vec![16, 16],
            0,
            2,
        );
        assert_eq!(vector.distance_to_next_set_bit(1), 3);
        assert_eq!(vector.distance_to_next_set_bit(8), 2);
        assert_eq!(vector.distance_to_next_set_bit(6), 2);
    }
}
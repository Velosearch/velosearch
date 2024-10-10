use std::time::Instant;

use datafusion::physical_plan::expressions::Column;
use velosearch::{jit::ast::Predicate, physical_expr::{boolean_eval::{Chunk, PhysicalPredicate, SubPredicate}, Primitives}, ShortCircuit};
use rand::seq::IteratorRandom;
use sorted_iter::{assume::AssumeSortedByItemExt, SortedIterator};



/// To test the benefits of pre-compiled short-circuit primitives
fn main() {
    let num: usize = 10000;
    let base_vec: Vec<u16> = Vec::from_iter((0..512 as u16).into_iter());
    let mut rng = rand::thread_rng();
    const NUM_ITER: usize = 5;

    // Build short-circuit primitive
    let predicate = Predicate::And { args: vec![
        Predicate::Leaf { idx: 0 },
        Predicate::Leaf { idx: 1 },
        Predicate::Leaf { idx: 2 },
        // Predicate::Leaf { idx: 3 },
    ] };
    let primitive = ShortCircuit::try_new(vec![0, 1, 2, 3], predicate, 2, 3, 0).unwrap();

    let physical_preidcate = PhysicalPredicate::Leaf { primitive: Primitives::ShortCircuitPrimitive(primitive) };

    let interpret = PhysicalPredicate::And {
        args: vec![
            SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test1", 0)) }),
            SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test2", 1)) }),
            SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test3", 2)) }),
            // SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test3", 3)) }),
        ]
    };

    for _ in 0..30 {
        let mut compile_cnt: u128 = 0;
        let mut interpret_cnt: u128 = 0;
        let mut handed_cnt: u128 = 0;

        for _ in 0..NUM_ITER {
            let sel: f64 = 0.02 * 20 as f64;
            let sample_num = (512. * num as f64 * sel / num as f64) as usize;
            let mut a = Vec::new();
            let mut a_raw: Vec<u32> = Vec::new();
            for i in 0..num {
                let sampled = base_vec.iter().cloned().choose_multiple(&mut rng, sample_num);
                a_raw.append(&mut sampled.iter().map(|v| i as u32 * 512 + *v as u32).collect());
                a.push(sampled);
            }
            let mut b = Vec::new();
            let mut b_raw: Vec<u32> = Vec::new();
            for i in 0..num {
                let sampled = base_vec.iter().cloned().choose_multiple(&mut rng, sample_num);
                b_raw.append(&mut sampled.iter().map(|v| i as u32 * 512 + *v as u32).collect());
                b.push(sampled);
            }
            let mut c = Vec::new();
            let mut c_raw: Vec<u32> = Vec::new();
            for i in 0..num {
                let sampled = base_vec.iter().cloned().choose_multiple(&mut rng, sample_num);
                c_raw.append(&mut sampled.iter().map(|v| i as u32 * 512 + *v as u32).collect());
                c.push(sampled);
            }
            let mut d = Vec::new();
            let mut d_raw = Vec::new();
            for i in 0..num {
                let sampled = base_vec.iter().cloned().choose_multiple(&mut rng, sample_num);
                d_raw.append(&mut sampled.iter().map(|v| i as u32 * 512 + *v as u32).collect());
                d.push(sampled);
            }

            let mut chunk: Vec<Option<Vec<Chunk>>> = Vec::new();

            let a_iter = a_raw.iter().assume_sorted_by_item();
            let b_iter = b_raw.iter().assume_sorted_by_item();
            let c_iter = c_raw.iter().assume_sorted_by_item();
            let timer = Instant::now();
            let _res: Vec<&u32> = a_iter.intersection(b_iter).intersection(c_iter).collect();
            handed_cnt += timer.elapsed().as_nanos();

            // Enter A batch
            let a_tempchunk = vec_to_bytes(a);
            let a_chunks = bytes_to_chunk(&a_tempchunk);
            chunk.push(Some(a_chunks));

            let b_tempchunk = vec_to_bytes(b);
            let b_chunk = bytes_to_chunk(&b_tempchunk);
            chunk.push(Some(b_chunk));

            let c_tempchunk = vec_to_bytes(c);
            let c_chunk = bytes_to_chunk(&c_tempchunk);
            chunk.push(Some(c_chunk));

            let d_tempchunk = vec_to_bytes(d);
            let d_chunk = bytes_to_chunk(&d_tempchunk);
            chunk.push(Some(d_chunk));

            let timer = Instant::now();
            let _res = physical_preidcate.eval_avx512(&chunk, None, true, 5).unwrap();
            compile_cnt += timer.elapsed().as_nanos();

            let timer = Instant::now();
            let _ = interpret.eval_avx512(&chunk, None, true, 10).unwrap();
            interpret_cnt += timer.elapsed().as_nanos();
        }
        println!("compile elapse: {:}, interpre elapse: {:}, hand cnt: {:}", compile_cnt / NUM_ITER as u128, interpret_cnt / NUM_ITER as u128, handed_cnt / NUM_ITER as u128);
    }
}

fn bytes_to_chunk(bytes: &Vec<Vec<u8>>) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    for byte in bytes {
        if byte.len() < 64 {
            chunks.push(Chunk::IDs(unsafe { byte.align_to::<u16>().1 }));
        } else {
            chunks.push(Chunk::Bitmap(unsafe { byte.align_to::<u64>().1 }));
        }
    }
    chunks
}

fn vec_to_bytes(id_vec: Vec<Vec<u16>>) -> Vec<Vec<u8>> {
    let mut chunks: Vec<Vec<u8>> = Vec::new();

    for batch in id_vec {
        if batch.len() > 8 {
            let mut bitmap: [u64; 8] = [0; 8];
            for id in batch {
                bitmap[id as usize >> 8] |= 1 << (id as usize % 64);
            }
            chunks.push(unsafe { bitmap.align_to::<u8>().1 }.to_vec());
        } else {
            chunks.push(unsafe { batch.align_to::<u8>().1 }.to_vec());
        }
    }
    chunks
}
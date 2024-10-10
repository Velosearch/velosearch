use std::time::Instant;

use datafusion::physical_plan::expressions::Column;
use velosearch::{jit::ast::Predicate, physical_expr::{boolean_eval::{Chunk, PhysicalPredicate, SubPredicate}, Primitives}, ShortCircuit};
use rand::seq::IteratorRandom;



/// To test the benefits of pre-compiled short-circuit primitives
fn main() {
    let num: usize = 1000_0000;
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

    let _physical_preidcate = PhysicalPredicate::Leaf { primitive: Primitives::ShortCircuitPrimitive(primitive) };

    let interpret = PhysicalPredicate::And {
        args: vec![
            SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test1", 0)) }),
            SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test2", 1)) }),
            SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test3", 2)) }),
            // SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test3", 3)) }),
        ]
    };

    let mut times = Vec::new();
    for step_len in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192, 1000_000, usize::MAX] {
        let mut interpret_cnt: u128 = 0;

        let sel: f64 = 0.02 * 5 as f64;
        let sample_num = (512. * num as f64 * sel / num as f64) as usize;
        let mut a = Vec::new();
        for _ in 0..num {
            a.push(base_vec.iter().cloned().choose_multiple(&mut rng, sample_num));
        }
        let mut b = Vec::new();
        for _ in 0..num {
            b.push(base_vec.iter().cloned().choose_multiple(&mut rng, sample_num));
        }
        let mut c = Vec::new();
        for _ in 0..num {
            c.push(base_vec.iter().cloned().choose_multiple(&mut rng, sample_num));
        }

        let mut chunk: Vec<Vec<Chunk>> = Vec::new();

        // Enter A batch
        let a_tempchunk = vec_to_bytes(a);
        let a_chunks = bytes_to_chunk(&a_tempchunk);
        chunk.push(a_chunks);

        let b_tempchunk = vec_to_bytes(b);
        let b_chunk = bytes_to_chunk(&b_tempchunk);
        chunk.push(b_chunk);

        let c_tempchunk = vec_to_bytes(c);
        let c_chunk = bytes_to_chunk(&c_tempchunk);
        let chunk_len = c_chunk.len();
        chunk.push(c_chunk);

        for _ in 0..NUM_ITER {
            let mut index: usize = 0;
            while index + step_len < chunk_len.try_into().unwrap() {
                let batch = chunk.iter().map(|v| {
                    Some(v[index..(index + step_len)].to_vec())
                })
                .collect();
                let timer = Instant::now();
                let _ = interpret.eval_avx512(&batch, None, true, step_len).unwrap();
                interpret_cnt += timer.elapsed().as_nanos();
                index += step_len;
            }
            let timer = Instant::now();
            let batch = chunk.iter().map(|v| {
                Some(v[index..].to_vec())
            })
            .collect();
            let _ = interpret.eval_avx512(&batch, None, true, chunk_len - index).unwrap();
            interpret_cnt += timer.elapsed().as_millis();
        }
        println!("interpre elapse: {:}", interpret_cnt / NUM_ITER as u128);
        times.push(interpret_cnt / NUM_ITER as u128);
    }
    println!("simulated result: {:?}", times);
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

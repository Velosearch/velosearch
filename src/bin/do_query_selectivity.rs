use std::{time::Instant, collections::{BTreeMap, BTreeSet}};

use datafusion::physical_plan::expressions::Column;
use itertools::Itertools;
use tantivy::{DocSet, query::{VecDocSet, Intersection}, TERMINATED, postings::SegmentPostings};
use velosearch::{physical_expr::{boolean_eval::{Chunk, PhysicalPredicate, SubPredicate}, Primitives, BooleanEvalExpr}, batch::{PostingBatchBuilder, PostingBatch}};
use rand::{seq::IteratorRandom, thread_rng};
use sorted_iter::{assume::AssumeSortedByItemExt, SortedIterator};



/// To test the benefits of pre-compiled short-circuit primitives
fn main() {
    // tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();
    let num: usize = 1000;
    const NUM_ITER: usize = 10;

    let interpret = PhysicalPredicate::And {
        args: vec![
            SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test1", 0)) }),
            SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test2", 1)) }),
            SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test3", 2)) }),
            // SubPredicate::new_with_predicate(PhysicalPredicate::Leaf { primitive: Primitives::ColumnPrimitive(Column::new("test3", 3)) }),
        ]
    };

    let mut interprets = vec![];
    let mut hands = vec![];
    let mut tantivys = vec![];

    let mut interpret_mem = vec![];
    // let mut hand_mem = vec![];
    let mut tantivy_mem = vec![];

    for i in 1..50 {
        let mut interpret_cnt: u128 = 0;
        let mut tantivy_cnt: u128 = 0;
        let mut hand_cnt: u128 = 0;

        let mut interpret_m = 0;
        let mut tantivy_m = 0;
        // let mut hand_m = 0;

        for _ in 0..NUM_ITER {
            let sel: f64 = 0.01 * i as f64;
            let sample_num = (512. * num as f64 * sel ) as usize;
            let (inter, batch, postings) = sample_batches(num * 512, sample_num);


            let expr = BooleanEvalExpr::new(Some(interpret.clone()));
            let map = &batch.term_idx.term_map.values().map(|v| v.valid_bitmap.clone()).collect::<Vec<_>>();
            
            let distris = map.into_iter().map(|v| Some(v.clone())).collect::<Vec<_>>();
            let min_range = (map[0].as_ref() & (map[1].as_ref() & map[2].as_ref())).into_iter().collect::<Vec<_>>();
            let timer = Instant::now();
            // let _ = interpret.eval_avx512(&chunk, None, true, 10).unwrap();
            let _res = batch.roaring_predicate(&distris, &[Some(0), Some(1), Some(2)], &min_range, &expr).unwrap();
            interpret_cnt += timer.elapsed().as_nanos();
            interpret_m += batch.memory_consumption().1 + batch.memory_consumption().2;

            let timer = Instant::now();
            let mut inter: Box<dyn DocSet> = Box::new(Intersection::new(inter));
            let mut i = 0;
            let mut docs = vec![];
            while inter.advance() != TERMINATED {
                docs.push(inter.doc());
                i += 1;
            }
            tantivy_cnt += timer.elapsed().as_nanos();
            tantivy_m += postings.iter().map(|v| v.len() * 4).sum::<usize>();

            let timer = Instant::now();
            let res = merge_based_intersection(&postings[0], &postings[1], &postings[2]).len();
            hand_cnt += timer.elapsed().as_nanos();
            // println!("res={}", res);
        }
        let tantivy = tantivy_cnt / NUM_ITER as u128;
        let interpret = interpret_cnt / NUM_ITER as u128;
        let hand = hand_cnt / NUM_ITER as u128;
        let tantivy_m = tantivy_m / NUM_ITER;
        let interpret_m = interpret_m / NUM_ITER;

        tantivys.push(tantivy);
        interprets.push(interpret);
        hands.push(hand);

        interpret_mem.push(interpret_m);
        tantivy_mem.push(tantivy_m);
        println!("iter: {:}, interpre elapse: {:}, tantivy cnt: {:}, hand cnt: {:}; interpret mem: {:}, tantivy mem: {:}", i, interpret_cnt / NUM_ITER as u128, tantivy_cnt / NUM_ITER as u128, hand, interpret_m, tantivy_m);
    }
    println!("hands = {:?}", hands);
    println!("tantivy = {:?}", tantivys);
    println!("interprert = {:?}", interprets);
    println!("tantivy_mem = {:?}", tantivy_mem);
    println!("interpret_mem = {:?}", interpret_mem);
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

fn sample_batches(num: usize, sample_num: usize) -> (Vec<Box<dyn DocSet>>, PostingBatch, Vec<Vec<u32>>) {
    let mut rng = thread_rng();
    let base: Vec<u32> = (0..num as u32).into_iter().choose_multiple(&mut rng, sample_num);
    let a_sampled: Vec<u32> = sample_posting(num, base.clone());
    let b_sampled: Vec<u32> = sample_posting(num, base.clone());
    let c_sampled: Vec<u32> = sample_posting(num, base.clone());

    let mut docs = vec![vec![]; num];
    for a in &a_sampled {
        docs[*a as usize].push("a".to_owned());
    }
    let a_p = create_posting(&a_sampled);
    for b in &b_sampled {
        docs[*b as usize].push("b".to_owned());
    }
    let b_p = create_posting(&b_sampled);
    for c in &c_sampled {
        docs[*c as usize].push("c".to_owned());
    }
    let c_p = create_posting(&c_sampled);

    
    docs = docs.into_iter()
        .map(|v| if v .len() == 0 {
            vec!["t".to_string()]
        } else {
            v
        })
        // .filter(|v| v.len() > 0)
        .collect();
    let mut builder = PostingBatchBuilder::new(0);
    for (i, doc) in docs.into_iter().enumerate() {
        builder.add_docs(doc, i).unwrap();
    }

    (vec![a_p, b_p, c_p], builder.build().unwrap(), vec![a_sampled, b_sampled, c_sampled])
}

fn create_posting(docs: &[u32]) -> Box<dyn DocSet> {
    Box::new(SegmentPostings::create_from_docs(docs))
}

fn sample_posting(num: usize, base: Vec<u32>) -> Vec<u32> {
    let mut rng = thread_rng();
    (0..num as u32)
        .into_iter()
        .choose_multiple(&mut rng, 1000)
        .into_iter()
        .chain(base.clone().into_iter())
        .collect::<BTreeSet<u32>>()
        .into_iter()
        .collect_vec()
}

fn merge_based_intersection(vec1: &Vec<u32>, vec2: &Vec<u32>, vec3: &Vec<u32>) -> Vec<u32> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;

    while i < vec1.len() && j < vec2.len() && k < vec3.len() {
        if vec1[i] == vec2[j] && vec1[i] == vec3[k] {
            // 找到交集元素
            result.push(vec1[i]);
            i += 1;
            j += 1;
            k += 1;
        } else {
            // 找到最小的元素并前进
            let min_val = vec1[i].min(vec2[j]).min(vec3[k]);
            if vec1[i] == min_val {
                i += 1;
            }
            if vec2[j] == min_val {
                j += 1;
            }
            if vec3[k] == min_val {
                k += 1;
            }
        }
    }

    result
}
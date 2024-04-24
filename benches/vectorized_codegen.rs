
// use std::sync::Arc;

// use criterion::{Criterion, BenchmarkId, criterion_group, criterion_main};
// use rand::seq::IteratorRandom;
// use datafusion::{arrow::{array::BooleanArray, record_batch::RecordBatch, datatypes::{Schema, Field, DataType}}, physical_expr::BooleanQueryExpr, physical_plan::{expressions::{binary, col, Dnf}, PhysicalExpr}, logical_expr::Operator};

// #[inline]
// fn gen_batch(sel: f64) -> RecordBatch {
//     let num = 1024;
//     let v: Vec<u32> = (0..num as u32).collect();
//     let mut rng = rand::thread_rng();
//     let input_schema = Schema::new(
//         vec![
//             Field::new(format!("test1"), DataType::Boolean, false),
//             Field::new(format!("test2"), DataType::Boolean, false),
//             Field::new(format!("test3"), DataType::Boolean, false),
//             Field::new(format!("test4"), DataType::Boolean, false),
//             Field::new(format!("test5"), DataType::Boolean, false),
//         ],
//     );
//     let a1 = v.iter().cloned().choose_multiple(&mut rng, (num as f64 * sel) as usize);
//     let a2 = v.iter().cloned().choose_multiple(&mut rng, (num as f64 * sel) as usize);
//     let a3 = v.iter().cloned().choose_multiple(&mut rng, (num as f64 * sel) as usize);
//     let a4 = v.iter().cloned().choose_multiple(&mut rng, (num as f64 * sel) as usize);
//     let a5 = v.iter().cloned().choose_multiple(&mut rng, (num as f64 * sel) as usize);
//     let mut b1 = vec![false; num];
//     let mut b2 = vec![false; num];
//     let mut b3 = vec![false; num];
//     let mut b4 = vec![false; num];
//     let mut b5 = vec![false; num];
//     for item in &a1 {
//         b1[*item as usize] = true;
//     }
//     for item in &a2 {
//         b2[*item as usize] = true;
//     }
//     for item in &a3 {
//         b3[*item as usize] = true;
//     }
//     for item in &a4 {
//         b4[*item as usize] = true;
//     }
//     for item in &a5 {
//         b5[*item as usize] = true;
//     }
//     let b1 = Arc::new(BooleanArray::from(b1));
//     let b2 = Arc::new(BooleanArray::from(b2));
//     let b3 = Arc::new(BooleanArray::from(b3));
//     let b4 = Arc::new(BooleanArray::from(b4));
//     let b5 = Arc::new(BooleanArray::from(b5));
//     let batch = RecordBatch::try_new(
//         Arc::new(input_schema.clone()),
//         vec![
//             b1, b2, b3, b4, b5,
//         ],
//     ).unwrap();
//     batch
// }
// #[inline]
// fn vectorized_codegen(c: &mut Criterion) {
//     println!("start");
//     let mut group = c.benchmark_group("vectorized_vs_codegen");
//     let input_schema = Schema::new(
//         vec![
//             Field::new(format!("test1"), DataType::Boolean, false),
//             Field::new(format!("test2"), DataType::Boolean, false),
//             Field::new(format!("test3"), DataType::Boolean, false),
//             Field::new(format!("test4"), DataType::Boolean, false),
//             Field::new(format!("test5"), DataType::Boolean, false),
//         ],
//     );
//     // let binary_expr = binary(col("test1", &input_schema).unwrap(), Operator::And, col("test2", &input_schema).unwrap(), &input_schema).unwrap();
//     // let binary_expr2 = binary(col("test4", &input_schema).unwrap(), Operator::And, col("test3", &input_schema).unwrap(), &input_schema).unwrap();
//     let binary_expr = binary(col("test1", &input_schema).unwrap(), Operator::And, col("test2", &input_schema).unwrap(), &input_schema).unwrap();
//     let binary_expr = binary(binary_expr, Operator::And, col("test3", &input_schema).unwrap(), &input_schema).unwrap();
//     let binary_expr = binary(binary_expr, Operator::And, col("test4", &input_schema).unwrap(), &input_schema).unwrap();
//     let binary_expr = binary(binary_expr, Operator::And, col("test5", &input_schema).unwrap(), &input_schema).unwrap();

//     let gen_func = create_boolean_query_fn(&vec![
//         Dnf::new(vec![0]),
//         Dnf::new(vec![1]),
//         Dnf::new(vec![2]),
//         Dnf::new(vec![3]),
//         Dnf::new(vec![4]),
//     ]);
//     let vectorized_expr = BooleanQueryExpr::new(binary_expr.clone());
//     let codegen_expr = BooleanQueryExpr::new_with_fn(binary_expr, gen_func);

//     for i in 1..=25 {
//         let sel = i as f64 * 0.02;
//         let datas: Vec<RecordBatch> = (0..1000).map(|_| gen_batch(sel)).collect();
        
//         group.bench_with_input(BenchmarkId::new("vectorized", format!("{:.2}", sel)), &datas,
//             |b, p|  b.iter(|| {
//                 for batch in p {
//                     vectorized_expr.evaluate(batch).unwrap();
//                 }
//             }));

//         group.bench_with_input(BenchmarkId::new("code_gen", format!("{:.2}", sel)), &datas,
//             |b, p| b.iter(|| {
//                 for batch in p {
//                     codegen_expr.evaluate(batch).unwrap();
//                 }
//             }));
        
//     }
//     group.finish();
// }

// criterion_group!(benches, vectorized_codegen);
// criterion_main!(benches);
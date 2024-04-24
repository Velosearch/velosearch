use std::{env, sync::Arc, fs::File, io::{self, BufRead}};
use datafusion::{prelude::*, sql::TableReference, datasource::provider_as_source,arrow::array::UInt64Array};
use velosearch::{parser, boolean_parser, utils::{Result, builder::deserialize_posting_table}, BooleanContext, jit::AOT_PRIMITIVES, query};
use jemallocator::Jemalloc;


#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[tokio::main]
async fn main() {
    // tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();
    let args: Vec<String> = env::args().collect();
    let partition_num = match args.get(2) {
        Some(p) => p.parse().unwrap(),
        None => 1,
    };
    main_inner(
        args[1].to_owned(),
        partition_num,
        &args.get(3).expect("Need query path."),
    ).await.unwrap();
}

async fn main_inner(index_dir: String, partitions_num: usize, query_file: &str) -> Result<()> {
    let posting_table = deserialize_posting_table(index_dir, partitions_num).unwrap();
    let ctx = BooleanContext::new();
    ctx.register_index(TableReference::Bare { table: "__table__".into() }, Arc::new(posting_table))?;
    let _ = AOT_PRIMITIVES.len();
    let provider = ctx.index_provider("__table__").await?;
    let schema = &provider.schema();
    let table_source = provider_as_source(Arc::clone(&provider));

    let file = File::open(query_file)?;
    let reader = io::BufReader::new(file);
    let queries: Vec<String> = reader.lines().into_iter().map(|l| l.unwrap()).collect();

    println!("Start!");
    let mut sum = 0;
    for _ in 0..1000_000_000 {
        for line in &queries{
            let predicate = if let Ok(expr) = parser::boolean(line) {
                expr
            } else {
                boolean_parser::boolean(line).unwrap()
            };
            // let predicate = boolean_parser::boolean(&words[1]).unwrap();
            let index = ctx.boolean_with_provider(table_source.clone(), &schema, predicate, false).await.unwrap();
            let res = index.collect().await.unwrap();
            if res.len() == 0 {
                println!("0");
            } else {
                let res: u64 = res.into_iter()
                .map(|v| v.column(0).as_any().downcast_ref::<UInt64Array>().unwrap().value(0))
                .sum();
                sum += 1;
            }
        }
    }
    println!("res: {:}", sum);

    Ok(())
}
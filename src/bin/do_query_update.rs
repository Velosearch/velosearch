use std::{env, sync::Arc, fs::File, io::{self, BufRead}, path::Path};
use datafusion::{sql::TableReference, datasource::provider_as_source, logical_expr::TableSource, arrow::{datatypes::Schema, array::UInt64Array}};
use serde::Deserialize;
use tantivy::tokenizer::{TextAnalyzer, SimpleTokenizer};

use tokio::time::Instant;
use tracing::debug;
use velosearch::{parser, boolean_parser, utils::{Result, builder::deserialize_posting_table}, BooleanContext, jit::AOT_PRIMITIVES, datasources::posting_table::PostingTable};
use jemallocator::Jemalloc;
use indicatif::ProgressBar;
use rand::{self, seq::SliceRandom, thread_rng};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Deserialize, Clone)]
struct Query {
    text: String,
    cmd: String,
    meta: String,
}

#[tokio::main]
async fn main() {
    // tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let args: Vec<String> = env::args().collect();
    let ratio = args.get(2).unwrap().parse::<usize>().unwrap();
    main_inner(args.get(1).cloned(), ratio).await.unwrap();
}

async fn main_inner(index_dir: Option<String>, ratio: usize) -> Result<()> {
    let ratio: f64 = 0.1 * ratio as f64;
    let posting_table = match index_dir {
        Some(dir) => Arc::new(
            deserialize_posting_table(dir, 1).unwrap()),
        None => Arc::new(PostingTable::new(
            Arc::new(Schema::empty()), vec![], 0)),
    };
    
    let ctx = BooleanContext::new();
    ctx.register_index(TableReference::Bare { table: "__table__".into() },
        posting_table.clone())?;
    let _ = AOT_PRIMITIVES.len();
    let provider = ctx.index_provider("__table__").await?;
    let schema = &provider.schema();
    let table_source = provider_as_source(Arc::clone(&provider));
    let mut tokenizer = TextAnalyzer::from(SimpleTokenizer:: default());

    let file = File::open(Path::new("/testbed/boolean-query-benchmark/cnf_query.txt"))?;
    let reader = io::BufReader::new(file);
    let queries: Vec<Query> = reader.lines().into_iter().map(|l| {
        Query { text: l.unwrap(), cmd: "SEARCH".to_string(), meta: "".to_string() }
    }).collect();

    let file = File::open(Path::new("/testbed/boolean-query-benchmark/read_write_0.25.txt"))?;
    
    let reader = io::BufReader::new(file);
    let mut write_queries: Vec<Query> = reader.lines().into_iter().map(|l| {
        serde_json::from_str::<Query>(&l.unwrap()).unwrap()
    }).collect();
    if ratio != 0. {
        let mut query_len: usize = (write_queries.len() as f64 / ratio) as usize;
        let mut i = 0;
        while query_len > 0 {
            write_queries.push(queries[i].clone());
            i = (i + 1) % queries.len();
            query_len -= 1;
        }
    } else {
        write_queries.clear();
        for _ in 0..20 {
            for query in &queries {
                write_queries.push(query.clone());
            }
        }
    }

    println!("write query len: {:}", write_queries.len());
    let query_len = write_queries.len();
    let mut rng = thread_rng();
    write_queries.shuffle(&mut rng);

    let mut time = 0;
    // let bar = ProgressBar::new(write_queries.len() as u64);
    let start = Instant::now();
    for write in &write_queries[0..3000] {
                // println!("{:?}", write.cmd);
        match write.cmd.to_uppercase().as_str() {
            "SEARCH" => {
                // let start = Instant::now();
                let predicate = if let Ok(expr) = parser::boolean(&write.text) {
                    expr
                } else {
                    boolean_parser::boolean(&write.text).unwrap()
                };
                // let predicate = boolean_parser::boolean(&words[1]).unwrap();
                let index = ctx.boolean_with_provider(table_source.clone(), &schema, predicate, false).await.unwrap();
                let _res = index.collect().await.unwrap();
                // time += start.elapsed().as_micros();
            }
            "INSERT" => {
                let mut token_stream = tokenizer.token_stream(&write.text);
                let mut doc = Vec::new();
                while let Some(token) = token_stream.next() {
                    doc.push(token.text.clone());
                }
                insert(posting_table.clone(), doc, 512).await;
            }
            "DELETE" => {

            }
            "UPDATE" => {
                // Delete and then insert new document.
                let mut token_stream = tokenizer.token_stream(&write.text);
                let mut doc = Vec::new();
                while let Some(token) = token_stream.next() {
                    doc.push(token.text.clone());
                }
                insert(posting_table.clone(), doc, 512).await;
            }
            "MERGE" => {
                posting_table.schedule_merge().await;
                println!("Merge all segments!");
            }
            "SEGLEN" => {
                println!("Segment count: {:}", posting_table.segment_num().await);
            }
            _ => {
                println!("Unregconized input: {:?}", write.cmd);
            },
        }
        // bar.inc(1);
    }
    // println!("{:} read lantecy: {:}", time, time as f64 / (2000 as f64 * (1. - ratio)));
    println!("throughout: {:}", 3000. / start.elapsed().as_millis() as f64);
    Ok(())
}

async fn insert(posting_table: Arc<PostingTable>, doc: Vec<String>, times: usize) {
    let docs = (0..times).map(|_| doc.clone()).collect::<Vec<_>>();
    posting_table.add_documents(docs).await;
    posting_table.commit().await;
    debug!("Commit one inserted document");
}

async fn search(words: Vec<&str>, ctx: BooleanContext, table_source: Arc<dyn TableSource>, schema: &Schema) {
    let predicate = if let Ok(expr) = parser::boolean(&words[1]) {
        expr
    } else {
        boolean_parser::boolean(&words[1]).unwrap()
    };
    // let predicate = boolean_parser::boolean(&words[1]).unwrap();
    let index = ctx.boolean_with_provider(table_source.clone(), &schema, predicate, false).await.unwrap();
    let res = index.collect().await.unwrap();
    if res.len() == 0 {
        println!("0");
    } else {
        let sum: usize = res.into_iter()
        .map(|v| v.column(0).as_any().downcast_ref::<UInt64Array>().unwrap().value(0) as usize)
        .sum();
        println!("{:}", sum);
    }
}

#[cfg(test)]
mod tests {
    use crate::parser;

    #[test]
    fn simple_and_test() {
        let expr = parser::boolean("ast AND ast").unwrap();
        assert_eq!(format!("{:?}", expr), "ast & ast");
    }

    #[test]
    fn somple_or_test() {
        let expr = parser::boolean("ast OR ast").unwrap();
        assert_eq!(format!("{:?}", expr), "ast | ast");
    }

    #[test]
    fn combination_test() {
        let expr = parser::boolean("ast AND ast AND (ast OR ast OR (ast AND ast))").unwrap();
        assert_eq!(format!("{:?}", expr), "ast & ast & (ast | ast | ast & ast)");
    }

    #[test]
    fn combination_test_2() {
        let expr = parser::boolean("(Receiving AND block AND blk_-666178582501318365 AND src AND 10.251.43.147 AND dest) AND (PacketResponder AND 10.250.19.102 AND for AND block AND blk_-6661785825013183656 AND Interrupted)").unwrap();
        assert_eq!(format!("{:?}", expr), "");
    }
}
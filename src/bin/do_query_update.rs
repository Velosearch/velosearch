use std::{env, sync::Arc};
use datafusion::{sql::TableReference, datasource::provider_as_source, logical_expr::TableSource, arrow::{datatypes::Schema, array::UInt64Array}};
use tantivy::tokenizer::{TextAnalyzer, SimpleTokenizer};
use velosearch::{parser, boolean_parser, utils::{Result, builder::deserialize_posting_table}, BooleanContext, jit::AOT_PRIMITIVES, datasources::posting_table::PostingTable};
use jemallocator::Jemalloc;


#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init();
    let args: Vec<String> = env::args().collect();
    let partition_num = match args.get(2) {
        Some(p) => p.parse().unwrap(),
        None => 1,
    };
    main_inner(args.get(1).cloned(), partition_num).await.unwrap();
}

async fn main_inner(index_dir: Option<String>, partitions_num: usize) -> Result<()> {
    let posting_table = match index_dir {
        Some(dir) => Arc::new(
            deserialize_posting_table(dir, partitions_num).unwrap()),
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
    let tokenizer = TextAnalyzer::from(SimpleTokenizer);

    let stdin = std::io::stdin();
    for line_res in stdin.lines() {
        let line = line_res?;
        let words: Vec<&str> = line.split("\t").collect();
        match words[0] {
            "COUNT" => search(words, ctx.clone(), table_source.clone(), schema).await,
            "INSERT" => {
                let mut token_stream = tokenizer.token_stream(&words[1]);
                let mut doc = Vec::new();
                while let Some(token) = token_stream.next() {
                    doc.push(token.text.clone());
                }
                insert_one(posting_table.clone(), doc).await;
            }
            _ => unreachable!(),
        }
        ;
    }
    Ok(())
}

async fn insert_one(posting_table: Arc<PostingTable>, doc: Vec<String>) {
    posting_table.add_document(doc).await;
    posting_table.commit().await;
    println!("Commit one inserted document");
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
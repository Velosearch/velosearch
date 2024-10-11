use core::time;
use std::{collections::{BTreeMap, HashMap}, fs::File, io::BufWriter, path::PathBuf, sync::Arc};

use datafusion::{sql::TableReference, arrow::datatypes::{Schema, DataType, Field}, datasource::provider_as_source};
use adaptive_hybrid_trie::TermIdx;
use tantivy::tokenizer::{TextAnalyzer, SimpleTokenizer};
use tracing::{info, span, Level, debug};

use crate::{utils::{json::{parse_wiki_dir, WikiItem}, builder::{deserialize_posting_table, serialize_term_meta}}, Result, batch::{PostingBatchBuilder, BatchRange, TermMetaBuilder, PostingBatch}, datasources::posting_table::PostingTable, BooleanContext, jit::AOT_PRIMITIVES, boolean_parser, parser};


pub struct PostingHandler {
    posting_table: Option<PostingTable>,
}

impl PostingHandler {
    pub fn new(base: String, path: Vec<String>, partition_nums: usize, batch_size: u32, dump_path: Option<String>) -> Self {
        let _ = AOT_PRIMITIVES.len();
        if let Some(p) = dump_path.clone() {
            if let Some(t) = deserialize_posting_table(p, partition_nums) {
                return Self {
                    posting_table: Some(t),
                };
            }
        };
        let items: Vec<WikiItem> = path
            .into_iter()
            .map(|p| parse_wiki_dir(&(base.clone() + &p)).unwrap())
            .flatten()
            .collect();

        let doc_len = items.len();
        let posting_table = to_batch(items, doc_len, partition_nums, batch_size, dump_path);
    
        Self { 
            posting_table: Some(posting_table),
        }
    }

}

impl PostingHandler {
    pub async fn execute(&mut self) ->  Result<u128> {
        rayon::ThreadPoolBuilder::new().num_threads(22).build_global().unwrap();
        let ctx = BooleanContext::new();
        let space = self.posting_table.as_ref().unwrap().memory_consumption();
        info!("space usage: {:}", space);
        ctx.register_index(TableReference::Bare { table: "__table__".into() }, Arc::new(self.posting_table.take().unwrap()))?;
        // let file = File::open("cnf_queries.txt")?;
        // let reader = io::BufReader::new(file);
        // 遍历每一行并输出
        // let queries: Vec<String> = reader.lines().into_iter().map(|l| l.unwrap()).collect();
        debug!("======================start!===========================");
        let time_sum = 0;
        debug!("start construct query");
        // let mut time_distri = Vec::new();
        let round = 1;
        let provider = ctx.index_provider("__table__").await?;
        let schema = &provider.schema();
        debug!("schema exampe: {:?}", schema.as_ref().all_fields());
        info!("term num: {:?}", schema.as_ref().all_fields().len());
        let table_source = provider_as_source(Arc::clone(&provider));
        println!("start!");
        std::thread::sleep(time::Duration::from_secs(1));
        let len = 0;
        for _ in 0..round {
            // for q in &queries {
                let q: &str = "+ccccAMERICA +ssssAMERICA +dddd1997 +ppppMFGR1";
                let predicate = if let Ok(expr) = parser::boolean(&q) {
                    expr
                } else {
                    boolean_parser::boolean(&q).unwrap()
                };
                let index = ctx.boolean_with_provider(table_source.clone(), &schema, predicate, false).await.unwrap();
                let res = index
                // .explain(false, true).unwrap()
                // .show().await.unwrap();
                // .count_agg().unwrap()
                .collect().await.unwrap();
                println!("{:} res: {:?}", q, unsafe {res[0].column(0).data().buffers()[0].align_to::<u64>().1});

            // }
            // let timer = Instant::now();
            
            // info!("Predicate{:}: {:?}", i, predicate);
            
            // let time = timer.elapsed().as_micros();
            // time_distri.push(time);
            // time_sum += time;
        }
        info!("{:}", len);
        // info!("Time distribution: {:?}", time_distri);
                // table.boolean_predicate(predicate).unwrap()
                //     .collect().await.unwrap();
                    // .explain(false, true).unwrap()
                    // .show().await.unwrap();
            // }))
        // }

        // for handle in handlers {
        //     handle.await.unwrap();
        // }
        // info!("Total time: {} us", query_time / 5);
        // info!("Total memory: {} MB", space / 1000_000);
        Ok(time_sum / round as u128)
    }
}


fn to_batch(docs: Vec<WikiItem>, length: usize, _partition_nums: usize, batch_size: u32, dump_path: Option<String>) -> PostingTable {
    let partition_nums = 1;
    let _span = span!(Level::INFO, "PostingHanlder to_batch").entered();
    let num_512 = (length as u32 + batch_size - 1) / batch_size;
    // let num_512_partition = (num_512 + partition_nums as u32 - 1) / (partition_nums as u32);
    let num_512_partition = num_512 + partition_nums as u32 - 1;

    debug!("num_512: {}, num_512_partition: {}", num_512, num_512_partition);
    let mut partition_batch = Vec::new();
    let mut term_idx: BTreeMap<String, TermMetaBuilder> = BTreeMap::new();
    for i in 0..partition_nums {
        partition_batch.push(PostingBatchBuilder::new(batch_size * num_512_partition * (i + 1)));
    }
    let mut current: (usize, usize) = (0, 0);
    let mut thredhold = batch_size;

    let tokenizer = TextAnalyzer::from(SimpleTokenizer);
    docs.into_iter()
    .enumerate()
    .for_each(|(id, e)| {
        let mut stream = tokenizer.token_stream(&e.content);
            stream.process(&mut |token| {
                let word = token.text.clone();
                let entry = term_idx.entry(word.clone()).or_insert(TermMetaBuilder::new(num_512_partition as usize, partition_nums as usize));
                if id as u32 >= thredhold {
                    info!("id: {}", id);
                    if id as u32 >= (batch_size * num_512_partition * (current.0 as u32 + 1)) {
                        current.0 += 1;
                        current.1 = 0;
                        info!("Start build ({}, {}) batch, current thredhold: {}", current.0, current.1, thredhold);
                    } else {
                        info!("Thredhold: {}, current: {:?}", thredhold, current);
                        current.1 += 1;
                    }
                    thredhold += batch_size;
                }
                partition_batch[current.0 as usize].push_term(word, id as u32).expect("Should push term correctly");
                entry.set_true(current.1 as usize, current.0.try_into().unwrap(), id as u32);
            });
    });
    // assert!(ids.is_sorted(), "ids must be sorted");
    // assert_eq!(ids.len(), words.len(), "The length of ids and words must be same");
    // words
    //     .into_iter()
    //     .zip(ids.into_iter())
    //     .for_each(|(word, id)| {
    //         let entry = term_idx.entry(word.clone()).or_insert(TermMetaBuilder::new(num_512_partition as usize, partition_nums));
    //         if id >= thredhold as u32 {
    //             debug!("id: {}", id);
    //             if id >= (batch_size * num_512_partition * (current.0 as u32 + 1)) {
    //                 current.0 += 1;
    //                 current.1 = 0;
    //                 debug!("Start build ({}, {}) batch, current thredhold: {}", current.0, current.1, thredhold);
    //             } else {
    //                 debug!("Thredhold: {}, current: {:?}", thredhold, current);
    //                 current.1 += 1;
    //             }
    //             thredhold += batch_size;
    //         }
    //         partition_batch[current.0].push_term(word, id).expect("Shoud push term correctly");
    //         entry.set_true(current.1, current.0, id);
    //     });

    for (i, p) in partition_batch.iter().enumerate() {
        debug!("The partition {} batch len: {}", i, p.doc_len())
    }

    if let Some(ref p) = dump_path {
        let path = PathBuf::from(p);
        let f = File::create(path.join(PathBuf::from("posting_batch.bin"))).unwrap();
        let writer = BufWriter::new(f);
        bincode::serialize_into(writer, &partition_batch).unwrap();
    }

    let mut term_idx = term_idx;

    let partition_batch: Vec<Arc<PostingBatch>> = partition_batch
        .into_iter()
        .map(|mut b| {
            Arc::new(b.build_with_idx(Some(&mut term_idx)).unwrap())
        })
        .collect();

    let mut keys = Vec::new();
    let mut values = Vec::new();

    term_idx
        .into_iter()
        .for_each(|m| {
            keys.push(m.0); 
            values.push(m.1.build());
        });
    // let consumption: usize = values.iter().map(|v| v..memory_consumption()).sum();
    // info!("term_idx consumption: {:}", consumption);
    
    let (fields_index, fields) = keys.iter()
        .chain([&"__id__".to_string()].into_iter())
        .enumerate()
        .map(|(i, v)| {
            let idx = (v.to_string(), i);
            let field = Field::new(v.to_string(), DataType::Boolean, false);
            (idx, field)
        })
        .unzip();
    let schema = Schema {
        fields,
        metadata: HashMap::new(),
        fields_index: Some(fields_index),
    };

    if let Some(ref p) = dump_path {
        // Serialize Batch Ranges
        let path = PathBuf::from(p);

        let f = File::create(path.join(PathBuf::from("batch_ranges.bin"))).unwrap();
        let writer = BufWriter::new(f);
        bincode::serialize_into(writer, &BatchRange::new(0, (num_512_partition * batch_size) as u32)).unwrap();

        let f = File::create(path.join(PathBuf::from("term_keys.bin"))).unwrap();
        let writer = BufWriter::new(f);
        bincode::serialize_into(writer, &keys).unwrap();

        serialize_term_meta(&values, p.to_string());
    }
    assert_eq!(keys.len(), values.len());
    #[cfg(feature = "hash_idx")]
    let _term_idx = Arc::new(TermIdx { term_map: HashMap::from_iter(
        keys.into_iter()
        .zip(values.into_iter())
    ) });
    #[cfg(all(feature = "trie_idx", not(feature = "hash_idx")))]
    let term_idx = Arc::new(TermIdx::new(keys, values, 20));


    PostingTable::new(
        Arc::new(schema),
        partition_batch,
        1,
    )
}


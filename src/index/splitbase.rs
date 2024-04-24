#![allow(unused_variables, unused_imports, dead_code)]


use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::io::Write;

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, UInt32Array, Int8Array};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::datasource::{TableProvider, MemTable};
use datafusion::from_slice::FromSlice;
use datafusion::logical_expr::Operator;
use datafusion::prelude::*;
use datafusion::sql::TableReference;
use datafusion::{arrow::datatypes::Field};
use datafusion::arrow::record_batch::RecordBatch;
use rand::seq::SliceRandom;
use tokio::time::Instant;
use tracing::{span, Level, info};

use super::HandlerT;
use crate::utils::json::parse_wiki_dir;
use crate::utils::{Result, json::WikiItem, to_hashmap};

const DEFAULT_PARTITION: usize = 4;

pub struct SplitHandler {
    doc_len: u32,
    ids: Vec<u32>,
    words: Vec<String>,
    partition_map: HashMap<String, usize>
}

impl SplitHandler {
    pub fn new(path: &str) -> Self {
        let mut cnt = 0;
        let items = parse_wiki_dir(path).unwrap();
        let mut ids: Vec<u32> = Vec::new();
        let mut words: Vec<String> = Vec::new();
        let doc_len = items.len() as u32;

        items.into_iter()
        .for_each(|e| {
            let WikiItem{id: _, text: w} = e;
            w.split([' ', ',', '.', ';', '-'])
            .into_iter()
            .filter(|w| {w.len() > 2}) 
            .for_each(|e| {
                ids.push(cnt);
                words.push(e.to_uppercase().to_string());
                cnt+=1;
            })});
        Self { doc_len, ids, words, partition_map: HashMap::new() }
    }

    fn to_recordbatch(&mut self, partition_num: usize) -> Result<Vec<Vec<RecordBatch>>> {
        let _span = span!(Level::INFO, "BaseHandler to_recordbatch").entered();

        let nums = self.doc_len as usize;
        let res = to_hashmap(&self.ids, &self.words, self.doc_len, DEFAULT_PARTITION);
        let column_nums = res.len();
        let iter_nums = (column_nums + 99) / 100 as usize;
        let mut column_iter = res.into_iter();
        let mut schema_list = Vec::with_capacity(iter_nums);
        let mut column_list: Vec<Vec<Vec<ArrayRef>>> = Vec::with_capacity(iter_nums);
        for i in 0..iter_nums {
            let mut fields= vec![Field::new("__id__", DataType::UInt32, false)];
            let mut columns: Vec<Vec<ArrayRef>> = Vec::with_capacity(DEFAULT_PARTITION);
            let mut is_init = false;
 
            column_iter.by_ref().take(100).for_each(|(field, column)| {
                self.partition_map.insert(field.clone(), i);
                if !is_init {
                    let mut cnt = 0;
                    for i in 0..DEFAULT_PARTITION {
                        columns.push(vec![Arc::new(UInt32Array::from_iter((cnt..(cnt + column[i].len() as u32)).into_iter()))]);
                        cnt += column[i].len() as u32;
                    }
                    is_init = true;
                }
                
                fields.push(Field::new(format!("{field}"), DataType::Int8, true));
                column.into_iter().enumerate() // 
                .for_each(|(i, v)| {
                    columns[i].push(Arc::new(Int8Array::from_slice(v.as_slice())))
                });
            });

            column_list.push(columns);
            schema_list.push(fields);
        }
        
        Ok(schema_list.into_iter().zip(column_list.into_iter()).map(|(s, v)| {
            let shema = Arc::new(Schema::new(s));
            v.iter()
            .map(|batch| {
                RecordBatch::try_new(shema.clone(), batch.to_vec()).unwrap()
            }).collect::<Vec<RecordBatch>>()
        }).collect())
    }
}

#[async_trait]
impl HandlerT for SplitHandler {


    fn get_words(&self, num: u32) -> Vec<String> {
        self.words.iter().take(num as usize).cloned().collect()
    }
 
    async fn execute(&mut self) -> Result<u128> {
        let batches = self.to_recordbatch(DEFAULT_PARTITION)?;
        // declare a new context.
        let ctx = SessionContext::new();

        batches.into_iter().enumerate().for_each(|(i, b)| {
            register_table(format!("_table_{i}").as_str(), b, &ctx).unwrap();
        });
        

        let mut test_keys: Vec<String> = self.get_words(100);
        test_keys.shuffle(&mut rand::thread_rng());
        let test_keys = test_keys[..2].to_vec();
        let i = &test_keys[0];
        let j = &test_keys[1];
        let ip = self.partition_map[i];
        let jp = self.partition_map[j];
        let time = Instant::now();
        let i_df = ctx.table(format!("_table_{ip}").as_str()).await?
        .select_columns(&[format!("{i}").as_str(), "__id__"])?;
        let j_df = ctx.table(format!("_table_{jp}").as_str()).await?
        .select_columns(&[format!("{j}").as_str(), "__id__"])?;
    
        i_df
        .join_on(j_df, 
            JoinType::Inner, 
            [col(i).eq(lit(1)), col(j).eq(lit(1)), 
            col(format!("_table_{ip}.__id__")).eq(col(format!("_table_{jp}.__id__")))])?
        .explain(false, true)?.show().await?;
        let query_time = time.elapsed().as_millis();
        info!("query time: {}", query_time);
        Ok(query_time)
    }
}

pub struct SplitO1 {
    base: SplitHandler,
    encode: HashMap<String, u32>,
    idx: u32
}

impl SplitO1 {
    pub fn new(path: &str) -> Self {
        Self { 
            base: SplitHandler::new(path),
            encode: HashMap::new(),
            idx: 0
        }
    }

    // use encode u32 inplace variant String
    fn recode(&mut self, batches: Vec<Vec<RecordBatch>>) -> Result<Vec<Vec<RecordBatch>>> {
        batches.into_iter().map(|batch| {
            let schema = batch[0].schema();
            let fields = schema.all_fields();
            let fields = fields.into_iter().skip(1)
            .map(|field| {
                self.encode.insert(field.name().to_string(), self.idx);
                self.idx += 1;
                Field::new(format!("{}", self.idx-1), DataType::Int8, false)
            });
            let mut fields_id = vec![Field::new("__id__", DataType::UInt32, false)];
            fields_id.extend(fields);
            let schema = Arc::new(Schema::new(fields_id));
            batch.into_iter()
            .map(|b| {
                RecordBatch::try_new(schema.clone(), b.columns().to_vec()).map_err(|e| e.into())
            }).collect::<Result<Vec<RecordBatch>>>()
        }).collect::<Result<Vec<Vec<_>>>>()
    }
}

#[async_trait]
impl HandlerT for SplitO1 {
    fn get_words(&self, num: u32) -> Vec<String> {
        self.base.get_words(num)
    }

    async fn execute(&mut self) ->Result<u128> {
        let batches = self.base.to_recordbatch(DEFAULT_PARTITION)?;
        let batches = self.recode(batches)?;
        // declare a new context.
        let ctx = SessionContext::new();

        batches.into_iter().enumerate().for_each(|(i, b)| {
            register_table(format!("_table_{i}").as_str(), b, &ctx).unwrap();
        });
        
        
        let mut test_keys: Vec<String> = self.get_words(100);
        test_keys.shuffle(&mut rand::thread_rng());
        let time = Instant::now();
        let mut handlers = Vec::with_capacity(50);
        for x in 0..50 {
            let test_keys = test_keys[2*x..2*x+2].to_vec();
            let ctx = ctx.clone();
            let i = &test_keys[0];
            let j = &test_keys[1];
            let ri = self.encode[i];
            let rj = self.encode[j];
            let ip = self.base.partition_map[i];
            let jp = self.base.partition_map[j];
                
            handlers.push(tokio::spawn(async move {
                if ri == rj {
                    return;
                }
                let query = if ip != jp {
                    let i_df = ctx.table(format!("_table_{ip}").as_str()).await.unwrap().select_columns(&[format!("{ri}").as_str(), "__id__"]).unwrap();
                    let j_df = ctx.table(format!("_table_{jp}").as_str()).await.unwrap().select_columns(&[format!("{rj}").as_str(), "__id__"]).unwrap();
                    i_df.join_on(j_df, JoinType::Inner, [col(format!("_table_{ip}.__id__")).eq(col(format!("_table_{jp}.__id__")))]).unwrap()
                    .select(vec![col(format!("_table_{ip}.__id__")), col(ri.to_string()), col(rj.to_string())]).unwrap()
                    .with_column("cache", bitwise_and(col(format!("{rj}")), col(format!("{ri}")))).unwrap().filter(col("cache").eq(lit(1))).unwrap()
                } else {
                    let i_df = ctx.table(format!("_table_{ip}").as_str()).await.unwrap().select_columns(&[format!("{rj}").as_str(), format!("{ri}").as_str(), "__id__"]).unwrap();
                    i_df.with_column("cache", bitwise_and(col(format!("{rj}")), col(format!("{ri}")))).unwrap().filter(col("cache").eq(lit(1))).unwrap()
                };
                
                query.collect().await.unwrap();
                println!("{} complete!", x);
            }));
        }

        for handle in handlers {
            handle.await.unwrap();
        }
        let query_time = time.elapsed().as_micros() / 50;
        info!("o1 query time: {} micros seconds", query_time);
        Ok(query_time)    
    }
}

pub struct SplitConstruct {
    doc_len: u32,
    ids: Vec<u32>,
    words: Vec<String>,
    idx: u32,
    encode: HashMap<String, u32>,
    partition: HashMap<String, u32>
}

impl SplitConstruct {
    pub fn new(path: &str) -> Self {
        let items = parse_wiki_dir(path).unwrap();
        let doc_len = items.len() as u32;
        info!("WikiItem len: {}", items.len());

        let mut ids: Vec<u32> = Vec::new();
        let mut words: Vec<String> = Vec::new();
        let mut cnt = 0;
        items.into_iter()
        .for_each(|e| {
            let WikiItem{id: _, text: w} = e;
            w.split([' ', ',', '.', ';', '-'])
            .into_iter()
            .filter(|w| {w.len() > 2}) 
            .for_each(|e| {
                ids.push(cnt);
                words.push(e.to_uppercase().to_string());
                cnt+=1;
            })
        });
        Self { doc_len, ids, words, idx: 0, encode: HashMap::new(), partition: HashMap::new() } 
    }

    pub async fn split(&mut self, _: &str) -> Result<()> {
        let _span = span!(Level::INFO, "SplitConstruct to_recordbatch").entered();
        // @TODO
        // let nums = self.doc_len;
        // let res = to_hashmap(&self.ids, &self.words, self.doc_len, DEFAULT_PARTITION);
        // info!("hashmap len: {}", res.len());
        // let column_nums = res.len();
        // let iter_nums = (column_nums + 99) / 100 as usize;
        // let mut column_iter = res.into_iter();

        // let ctx = SessionContext::new();
        // for i in 0..iter_nums {
        //     info!("table {} creating", i);
        //     let mut fields= vec![Field::new("__id__", DataType::UInt32, false)];
        //     let mut columns: Vec<ArrayRef> = vec![Arc::new(UInt32Array::from_iter((0..(nums as u32)).into_iter()))];
        //     column_iter.by_ref().take(100).for_each(|(field, column)| {
        //         self.partition.insert(field.clone(), i as u32);
        //         self.encode.insert(field.clone(), self.idx);
        //         self.idx += 1;
        //         fields.push(Field::new(format!("{}", self.idx-1), DataType::Int8, true));
        //         assert_eq!(nums, column.len() as u32, "the {}th", i);
        //         columns.push(Arc::new(Int8Array::from_slice(column.as_slice())));
        //     });

        //     let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        //     ctx.read_batch(batch)?.write_parquet(format!("./data/tables/table_{i}.parquet").as_str(), None).await?;
        // }


        // let mut encode_f = File::create("./data/encode.json")?;
        // let mut partition_f = File::create("./data/partition.json")?;
        // encode_f.write_all(serde_json::to_string( &self.encode)?.as_bytes())?;
        // partition_f.write_all(serde_json::to_string(&self.partition)?.as_bytes())?;

        Ok(())
    }
}

fn bitwise_and(left: Expr, right: Expr) -> Expr {
    binary_expr(left, Operator::BitwiseAnd, right)
}

pub fn register_table(
    table_name: &str,
    batch: Vec<RecordBatch>,
    ctx: &SessionContext
) -> Result<Option<Arc<dyn TableProvider>>> {
    let table = MemTable::try_new(batch[0].schema(), vec![batch])?;
    let table = ctx.register_table(
        TableReference::Bare {
            table: table_name.into(),
        },
        Arc::new(table),
    )?;
    Ok(table)
}
use std::sync::Arc;
use async_trait::async_trait;
use datafusion::{arrow::{datatypes::{DataType, Field, Schema}, array::{ArrayRef, UInt32Array, Int8Array}, record_batch::RecordBatch}, from_slice::FromSlice, datasource::{TableProvider, MemTable}, sql::TableReference};
use tokio::time::Instant;
use rand::prelude::*;
use datafusion::prelude::*;
use tracing::{span, info, Level};

use crate::utils::{json::{WikiItem, parse_wiki_dir, to_hashmap_v2}, Result};

#[async_trait]
pub trait HandlerT {
   fn get_words(&self, num: u32) -> Vec<String>; 

   async fn execute(&mut self) -> Result<u128>;
}

pub struct BaseHandler {
    doc_len: u32,
    ids: Option<Vec<u32>>,
    words: Option<Vec<String>>,
    test_case: Vec<String>
}

impl BaseHandler {
    pub fn new(path: &str) -> Self {
        let items = parse_wiki_dir(path).unwrap();
        let doc_len = items.len() as u32;
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
            })});

        let mut rng = thread_rng();
        let test_case = words.iter()
        .choose_multiple(&mut rng, 100)
        .into_iter()
        .map(|e| e.to_string()).collect();
        info!("self.doc_len = {}", doc_len);
        Self { doc_len, ids: Some(ids), words: Some(words), test_case }
    }

    fn to_recordbatch(&mut self) -> Result<Vec<RecordBatch>> {
        let _span = span!(Level::INFO, "BaseHandler to_recordbatch").entered();

        let partition_nums = 64 as usize;
        let res = to_hashmap_v2(self.ids.take().unwrap(), self.words.take().unwrap(), self.doc_len, partition_nums);
        let mut field_list = Vec::new();
        field_list.push(Field::new("__id__", DataType::UInt32, false));
        res.keys().for_each(|k| {
            field_list.push(Field::new(format!("{k}"), DataType::Int8, true));
        });

        let schema = Arc::new(Schema::new(field_list));
        info!("The term dict length: {:}", res.len());
        let mut column_list: Vec<Vec<ArrayRef>> = Vec::with_capacity(partition_nums);
        let mut is_init = false;
        res.into_iter().for_each(|(_, v)| {
            assert_eq!(v.len(), partition_nums);
            if !is_init {
                let mut cnt = 0;
                for i in 0..partition_nums {
                    column_list.push(vec![Arc::new(UInt32Array::from_iter((cnt..(cnt + v[i].len() as u32)).into_iter()))]);
                    cnt += v[i].len() as u32;
                }
                is_init = true;
            }
            (0..partition_nums).into_iter()
            .for_each(|i| {
                column_list[i].push(Arc::new(Int8Array::from_slice(v[i].as_slice())))
            })
        });
        info!("start try new RecordBatch");
        Ok(column_list.into_iter().map(|p| {
            RecordBatch::try_new(
                schema.clone(),
                p
            ).unwrap()
        }).collect())
    }
    
}

#[async_trait]
impl HandlerT for BaseHandler {


    fn get_words(&self, _: u32) -> Vec<String> {
        vec![]
    }

    async fn execute(&mut self) -> Result<u128> {

        info!("start BaseHandler executing");
        let batch = self.to_recordbatch()?;
        assert_eq!(batch.len(), 64);
        info!("End BaseHandler.to_recordbatch()");
        // declare a new context.
        let ctx = SessionContext::new();

        register_table(format!("_table_").as_str(), batch, &ctx).unwrap();
        
        info!("End register batch");
        let df = ctx.table("_table_").await?;
        info!("End cache table t");

        let mut test_iter = self.test_case.clone().into_iter();

        let mut handlers = Vec::with_capacity(50);
        let time = Instant::now();
        let mut cnt = 0;
        for x in 0..1 {
            let keys = test_iter.by_ref().take(2).collect::<Vec<String>>();
            if keys[1] == keys[0] {
                continue
            }
            cnt += 1;
            let df = df.clone();
            handlers.push(tokio::spawn(async move {
                df
                    // .select_columns(&["__id__", &keys[0], &keys[1]]).unwrap()
                    // .filter(col(&keys[0]).eq(lit(1 as i8)).and(col(&keys[1]).eq(lit(1 as i8)))).unwrap()
                    // .filter(bitwise_and(col(&keys[0]), col(&keys[1])).eq(lit(1 as i8))).unwrap()
                    .select_columns(&["__id__", "FIRST", "TIME"]).unwrap()
                    // .select_columns(&["__id__"]).unwrap()
                    // .collect().await.unwrap();
                    // .explain(false, true).unwrap()
                    .show().await.unwrap();
                println!("{} complete", x);
            }));
        }
        for handle in handlers {
            handle.await.unwrap();
        }
        let query_time = time.elapsed().as_micros();
        info!("query time: {}", query_time/cnt);
        Ok(query_time)
    }
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

#[cfg(test)]
mod test {
    // use super::BaseHandler;

    // #[test]
    // fn test_json_bashhandler() {
        // let ids = vec![1, 2, 3, 4, 4, 5];
        // let words = vec!["a".to_string(), "b".to_string(), "c".to_string(), "a".to_string(),
        // "a".to_string(), "c".to_string()];
        // let res = BaseHandler::to_hashmap(ids, words);
        // assert_eq!(res.get("a"), Some(&vec![1,0,0,1,0]));
        // assert_eq!(res.get("b"), Some(&vec![0,1,0,0,0]));
        // assert_eq!(res.get("c"), Some(&vec![0,0,1,0,1]));
//     }
}
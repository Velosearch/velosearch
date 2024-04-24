use std::{collections::HashSet, time::Instant};

use async_trait::async_trait;
use futures::executor::block_on;
use rand::{thread_rng, seq::IteratorRandom};
use tantivy::{schema::{Schema, TEXT, Field, IndexRecordOption, STORED}, Index, doc, tokenizer::{TextAnalyzer, SimpleTokenizer, RemoveLongFilter, LowerCaser, Stemmer}, query::{Query, TermQuery, BooleanQuery, PhraseQuery, Occur}, Term, collector::{Count, DocSetCollector}};
use tracing::{debug, info};
use crate::utils::{Result, json::WikiItem};

use crate::utils::json::parse_wiki_dir;

use super::HandlerT;

pub struct TantivyHandler {
    doc_len: usize,
    index: Index,
    field: Field,
    test_case: Vec<String>,
}

impl TantivyHandler {
    pub fn new(base: String, path: Vec<String>, thread_num: usize) -> Result<Self> {
        let items: Vec<WikiItem> = path
            .into_iter()
            .map(|p| parse_wiki_dir(&(base.clone() + &p)).unwrap())
            .flatten()
            .collect();
        let doc_len = items.len();

        let mut schema_builder = Schema::builder();
        let id = schema_builder.add_text_field("id", STORED);
        let content = schema_builder.add_text_field("content", TEXT);
        let schema = schema_builder.build();
        let tokenizer = TextAnalyzer::from(SimpleTokenizer)
            .filter(RemoveLongFilter::limit(40))
            .filter(LowerCaser)
            .filter(Stemmer::default());
        let mut test_case = HashSet::new();
        
        let mut index = Index::create_in_ram(schema.clone());
        index.set_multithread_executor(thread_num)?;
        let mut index_writer = index.writer(10_000_000)?;
        items
            .into_iter()
            .for_each(|e| {
                index_writer.add_document(doc!(
                    id => e.id.as_str(),
                    content => e.text.as_str(),
                )).unwrap();
                let mut stream = tokenizer.token_stream(e.text.as_str());
                while let Some(token) = stream.next() {
                    test_case.insert(token.text.clone());
                }
            });
        index_writer.commit()?;
        index_writer.wait_merging_threads()?;

        let segment_ids = index.searchable_segment_ids()?;
        let mut index_wirter = index
            .writer(1_500_000_000)
            .expect("failed to create index writer");
        block_on(index_wirter.merge(&segment_ids))?;
        block_on(index_wirter.garbage_collect_files())?;
        
        let mut rng = thread_rng();
        Ok(Self {
            doc_len,
            index,
            field: content,
            test_case: test_case.into_iter().choose_multiple(&mut rng, 100),
        })
    }
}

#[async_trait]
impl HandlerT for TantivyHandler {
    fn get_words(&self, _num:u32) -> Vec<String> {
        self.test_case.clone()
    }

    async fn execute(&mut self) -> Result<u128> {
        let reader = self.index.reader()?;
        let term_query_3: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(self.field, "book"),
            IndexRecordOption::Basic,
        ));
        let term_query_4: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(self.field, "of"),
            IndexRecordOption::Basic,
        ));
        let term_query_5: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(self.field, "life"),
            IndexRecordOption::Basic,
        ));
        let or_query_1: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(self.field, "the"),
            IndexRecordOption::Basic,
        ));
        let or_query_2 = Box::new(TermQuery::new(
            Term::from_field_text(self.field, "the"),
            IndexRecordOption::Basic,
        ));
        // let or_query = Box::new(BooleanQuery::new(vec![
        //     (Occur::Should, or_query_1),
        //     (Occur::Should, or_query_2),
        // ]));
        let boolean_query: BooleanQuery = BooleanQuery::new(vec![
            (Occur::Must, or_query_1),
            (Occur::Must, term_query_3),
            (Occur::Must, term_query_4),
            (Occur::Must, term_query_5),
        ]);
        let mut space = 0;
        let mut distri = Vec::new();
        let round = 100;
        for _ in 0..round {
            let searcher: tantivy::Searcher = reader.searcher();
            let timer = Instant::now();
            let res = searcher.search(&boolean_query, &DocSetCollector)?.len();
            let query_time  = timer.elapsed().as_micros();
            distri.push(query_time);
            // space += searcher.space_usage().unwrap().total();
            // info!("{:?}", res.len());
        }
        println!("distri: {:?}", distri);
        // info!("Tantivy took {} us", timer.elapsed().as_micros() / 1);
        // info!("Total memery: {} B", space / 1000_000);
        Ok((0) as u128)
    }
}
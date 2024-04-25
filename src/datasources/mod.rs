use datafusion::common::TermMeta;

pub mod posting_table;
pub mod mmap_table;

pub trait ExecutorWithMetadata {
    fn term_metas_of(&self, terms: &[&str]) -> Vec<Option<TermMeta>>;

    fn term_meta_of(&self, term: &str) -> Option<TermMeta>;
}
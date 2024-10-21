use std::collections::BTreeMap;

use fst::{MapBuilder, Map};
pub mod ah_trie;

#[cfg(feature = "hash_idx")]
pub type TermIdx<T> = HashTermIdx<T>;
#[cfg(all(feature = "trie_idx", not(feature = "hash_idx")))]
pub type  TermIdx<T> = ah_trie::AHTrie<T>;

#[derive(Debug)]
pub struct HashTermIdx<T> {
    pub term_map: BTreeMap<String, T>
}

impl<T> HashTermIdx<T> {
    pub fn new() -> Self {
        Self { term_map: BTreeMap::new() }
    }

    pub fn get(&self, name: &str) -> Option<&T> {
        self.term_map.get(name)
    }

    pub fn insert(&mut self, name: String, value: T) -> Option<T> {
        self.term_map.insert(name, value)
    }
}


pub struct FSTIdx {
    inner_: Map<Vec<u8>>,
}

impl FSTIdx {
    pub fn get(&self, name: &str) -> Option<u64> {
        self.inner_.get(name)
    }
}

pub struct FSTIdxBuilder {
    builder: MapBuilder<Vec<u8>>,
}

impl FSTIdxBuilder {
    pub fn new() -> Self {
        Self {
            builder: MapBuilder::memory(),
        }
    }

    pub fn insert(&mut self, term: String, index: u64) {
        self.builder.insert(term, index).unwrap();
    }
}

#[cfg(test)]
mod tests {

}

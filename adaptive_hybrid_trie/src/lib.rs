use std::collections::HashMap;

use fst::{MapBuilder, Map};
pub mod ah_trie;

#[cfg(feature = "hash_idx")]
pub type TermIdx<T> = HashTermIdx<T>;
#[cfg(all(feature = "trie_idx", not(feature = "hash_idx")))]
pub type  TermIdx<T> = ah_trie::AHTrie<T>;

pub struct HashTermIdx<T> {
    pub term_map: HashMap<String, T>
}

impl<T> HashTermIdx<T> {
    pub fn new() -> Self {
        Self { term_map: HashMap::new() }
    }

    pub fn get(&self, name: &str) -> Option<&T> {
        self.term_map.get(name)
    }

    pub fn insert(&mut self, name: String, value: T) -> Option<T> {
        self.term_map.insert(name, value)
    }
}


pub struct FSTIdx<T> {
    inner_: Map<Vec<u8>>,
    values: Vec<T>
}

impl<T> FSTIdx<T> {
    pub fn get(&self, name: &str) -> Option<&T> {
        todo!()
    }
}

pub struct FSTIdxBuilder<T> {
    builder: MapBuilder<Vec<u8>>,
    values: Vec<T>
}

impl<T> FSTIdxBuilder<T> {
    pub fn new() -> Self {
        Self {
            builder: MapBuilder::memory(),
            values: vec![],
        }
    }

    pub fn insert(&mut self, term: String, value: T) {
        self.builder.insert(term, self.values.len() as u64).unwrap();
        self.values.push(value);
    }
}

#[cfg(test)]
mod tests {

}

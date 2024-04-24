use std::collections::HashMap;
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

#[cfg(test)]
mod tests {

}

mod ah_trie;
use std::cmp::min;

use art_tree::{Art, ByteString};
use fst_rs::FST;

use self::ah_trie::Encoding;
pub use ah_trie::AHTrie;

const CUT_OFF: usize = usize::MAX;
pub enum ChildNode {
    Fst(FST),
    Offset(usize),
}

pub struct AHTrieInner<T> {
    pub root: Art<ByteString, ChildNode>,
    values: Vec<T>
}

impl<T> AHTrieInner<T> {
    pub fn new(keys: Vec<String>, values: Vec<T>, cut_off: usize) -> Self {
        assert_eq!(keys.len(), values.len(), "The length of keys and values should be same.");
        // let mut key_prefix = prefix(&keys[0].clone());
        let mut key_prefix = prefix(&keys[0]);
        let mut tmp_keys = Vec::new();
        let mut tmp_off = Vec::new();
        let mut art = Art::new();
        // let last_key_pre = prefix(&keys.last().unwrap().clone());
        let last_key_pre = prefix(&keys.last().unwrap());
        for (off, key) in keys.iter().enumerate() {
            let pre = prefix(&key);
            if key.len() <= cut_off {
                art.insert(ByteString::new(&pre), ChildNode::Offset(off));
                continue;
            }
            if key_prefix == pre {
                tmp_keys.push(key.as_bytes()[cut_off..].to_vec());
                tmp_off.push(off);
                continue;
            } else {
                if tmp_keys.len() > 0 {
                    let fst = FST::new_with_bytes(&tmp_keys, tmp_off.clone());
                    tmp_keys.clear();
                    tmp_off.clear();
                    art.insert(ByteString::new(&key_prefix), ChildNode::Fst(fst));
                }
                key_prefix = pre;
                tmp_keys.push(key.as_bytes()[cut_off..].to_vec());
                tmp_off.push(off);
            }
        }
        if tmp_keys.len() > 0 {
            let fst = FST::new_with_bytes(&tmp_keys, tmp_off);
            art.insert(ByteString::new(&last_key_pre), ChildNode::Fst(fst));
        }
        Self {
            root: art,
            values: values,
        }
    }

    pub fn get_offset(&self, key: &str) -> Option<usize> {
        let pre = prefix(key);
        let key_bytes = if key.len() <= CUT_OFF {
            &pre
        } else {
            key.as_bytes()
        };
        match self.root.get_with_length(&ByteString::new(&key_bytes)) {
            Some((child, length)) => {
                match child {
                    ChildNode::Fst(fst) => fst
                        .get(&key.as_bytes()[length..])
                        .map(|v| v as usize),
                    ChildNode::Offset(off) => Some(*off),
                }
            }
            None => None
        } 
    }

    pub fn get_mut(&mut self, offset: usize) -> &mut T {
        &mut self.values[offset]
    }

    pub fn get(&self, key: &str) -> Option<&T> {
        let pre = prefix(key);
        let key_bytes = if key.len() <= CUT_OFF {
            &pre
        } else {
            key.as_bytes()
        };
        match self.root.get_with_length(&ByteString::new(&key_bytes)) {
            Some((child, length)) => {
                match child {
                    ChildNode::Fst(fst) => fst
                        .get(&key.as_bytes()[length..])
                        .map(|v| &self.values[v as usize]),
                    ChildNode::Offset(off) => Some(&self.values[*off]),
                }
            }
            None => None
        }
    }

    pub fn get_fst(&self, key: &str) -> Option<&FST> {
        let pre = prefix(key);
        let key_bytes = if key.len() <= CUT_OFF {
            &pre
        } else {
            key.as_bytes()
        };
        match self.root.get(&ByteString::new(&key_bytes)) {
            Some(child) => match child {
                ChildNode::Fst(fst) => Some(fst),
                _ => None,
            }
            None => None,
        }
    }

    pub fn insert(&mut self, key: &str, value: ChildNode) {
        self.root.insert(ByteString::new(key.as_bytes()), value);
    }

    pub fn insert_bytes(&mut self, key: &[u8], value: ChildNode) {
        self.root.insert(ByteString::new(key), value);
    }

    pub fn remove(&mut self, key: &str) {
        self.root.remove(&ByteString::new(key.as_bytes()));
    }

    pub fn remove_bytes(&mut self, key: &[u8]) {
        self.root.remove(&ByteString::new(&key));
    }

    pub fn encoding(&self, key: &str) -> Encoding {
        let key_bytes = prefix(key);
        match self.root.get_with_length(&ByteString::new(&key_bytes)) {
            Some((child, length)) => {
                match child {
                    ChildNode::Fst(_) => Encoding::Fst(length),
                    ChildNode::Offset(_) => Encoding::Art,
                }
            }
            None => Encoding::Art,
        }
    }

    pub fn prefix_keys(&self, prefix: &str) -> Vec<(&ByteString, &ChildNode)> {
        let prefix_low_bound = ByteString::new(prefix.as_bytes());
        let mut prefix_up_bound = prefix.to_string();
        let i = prefix_up_bound.remove(prefix_up_bound.len() - 1);
        prefix_up_bound.push((i as u8 + 1) as char);
        let prefix_up_bound = ByteString::new(prefix_up_bound.as_bytes());
        self.root.range(prefix_low_bound..prefix_up_bound).collect()
    }

}

// fn prefix(key: &str) -> [u8; CUT_OFF] {
//     let mut pre: [u8; CUT_OFF] = [0xFF; CUT_OFF];
//     for i in 0..min(key.len(), CUT_OFF) {
//         pre[i] = key.as_bytes()[i].clone();
//     }
//     pre
// }

fn prefix(key: &str) -> &[u8] {
    key.as_bytes()
}

#[cfg(test)]
mod tests {
    use art_tree::ByteString;

    use crate::ah_trie::AHTrieInner;

    #[test]
    fn aht_simple_test() {
        let trie = AHTrieInner::new(
            vec!["123".to_string(), "1234".to_string(), "1235".to_string(), "12567".to_string()],
            vec![0, 1, 2, 3], 
            4);
        assert_eq!(trie.get("123"), Some(&0));
        assert_eq!(trie.get("12"), None);
        assert_eq!(trie.get("143"), None);
        assert_eq!(trie.get("1234"), Some(&1)); 
        assert_eq!(trie.get("1235"), Some(&2));
        assert_eq!(trie.get("12567"), Some(&3));
    }

    #[test]
    fn aht_under_cutoff() {
        let trie = AHTrieInner::new(
            vec!["12".to_string(), "123".to_string(), "223".to_string(), "34".to_string()],
            vec![0, 1, 2, 3],
            4,
        );
        assert_eq!(trie.get("12"), Some(&0));
        assert_eq!(trie.get("123"), Some(&1));
        assert_eq!(trie.get("12345"), None);
        assert_eq!(trie.get("34"), Some(&3))
    }

    #[test]
    fn aht_longer_test() {
        let trie = AHTrieInner::new(
            vec!["123".to_string(), "12567896".to_string(), "12567899".to_string()],
            vec![0, 1, 2],
            4, 
        );
        assert_eq!(trie.get("123"), Some(&0));
        assert_eq!(trie.get("12567896"), Some(&1));
        assert_eq!(trie.get("12567899"), Some(&2));
    }

    #[test]
    fn aht_range_simple() {
        let trie = AHTrieInner::new(
            vec!["123".to_string(), "12567896".to_string(), "12587899".to_string()],
            vec![0, 1, 2],
            4, 
        );
        println!("prefix: {:?}", trie.prefix_keys("125").into_iter().map(|v| v.0).collect::<Vec<&ByteString>>());
    }
}
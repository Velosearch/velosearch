use crate::{PosT, LevelT, BitVector, FSTBuilder, label_vector::LabelVector, K_TERMINATOR};

#[derive(Clone)]
pub struct LoudsSparse {
    positions_sparse: Vec<u64>,
    height: LevelT,
    start_level: LevelT,
    /// number of nodes in louds-dense encoding
    node_count_dense: PosT,
    /// number of children(1's in child indicator bitmap) in louds-dense encoding
    child_count_dense: PosT,

    labels: LabelVector,
    child_indicator_bits: BitVector,
    louds_bits: BitVector,
    // pointer to the original data
    // keys: &'a Vec<String>
}

impl LoudsSparse {
    pub fn new(buidler: &FSTBuilder) -> Self {
        let height = buidler.labels().len();
        let start_level = buidler.sparse_start_level();
        let node_count_dense: PosT = buidler.node_counts()[0..start_level].into_iter().sum();
        let child_count_dense = if start_level == 0 {
            0
        } else {
            node_count_dense + buidler.node_counts()[start_level] - 1
        };
        let num_items_per_level = (0..height).into_iter()
            .map(|b| buidler.labels()[b].len())
            .collect();
        let labels = LabelVector::new(
            &buidler.labels(),
            start_level,
            height,
        );
        let positions_sparse = buidler.sparse_offset().clone();
        let child_indicator_bits = BitVector::new(
            buidler.child_indicator_bits().clone(),
            &num_items_per_level,
            start_level,
            height,
        );
        let louds_bits = BitVector::new_with_lut(
            buidler.louds_bits().clone(),
            &num_items_per_level,
            start_level,
            height,
        );

        Self {
            positions_sparse,
            start_level,
            height,
            node_count_dense,
            child_count_dense,
            labels,
            child_indicator_bits,
            louds_bits,
            // keys
        }
    }

    pub fn lookup_key(
        &self,
        key: &String,
        in_node_num: PosT,
    ) -> Option<u64> {
        return self.get(key.as_bytes(), in_node_num)
    }

    pub fn get(
        &self,
        key: &[u8],
        in_node_num: PosT
    ) -> Option<u64> {
        let mut node_num = in_node_num;
        let mut pos = self.first_label_pos(node_num);
        for level in self.start_level..key.len() {
            if let Some(p) = self.labels.search(key[level], pos, self.node_size(pos)) {
                pos = p;
            } else {
                return None;
            }

            // if trie branch terminates
            if !self.child_indicator_bits.read_bit(pos) {
                if level < key.len() - 1 {
                    return None;
                }
                let value_pos = pos - self.child_indicator_bits.rank(pos);
                return Some(self.positions_sparse[value_pos]);
            }

            // move to child
            node_num = self.child_node_num(pos);
            pos = self.first_label_pos(node_num);
        }
        if self.labels.read(pos) == K_TERMINATOR && !self.child_indicator_bits.read_bit(pos) {
            let value_pos = pos - self.child_indicator_bits.rank(pos);
            return Some(self.positions_sparse[value_pos]);
        }
        None
    }

    pub fn iter(&self) -> LoudsSparseIterator {
        LoudsSparseIterator::new(&self)
    }

    fn node_size(&self, pos: PosT) -> usize {
        assert!(self.louds_bits.read_bit(pos));
        return self.louds_bits.distance_to_next_set_bit(pos)
    }

    fn first_label_pos(&self, node_num: PosT) -> PosT {
        self.louds_bits.select(node_num + 1 - self.node_count_dense)
    }

    fn child_node_num(&self, pos: PosT) -> PosT {
        self.child_indicator_bits.rank(pos) + self.child_count_dense
    }

    pub fn is_end_of_node(&self, pos: PosT) -> bool {
        pos == self.louds_bits.num_bits() - 1 || self.louds_bits.read_bit(pos + 1)
    }
}

pub struct LoudsSparseIterator<'a>{
    /// True means the iter currently points to a valid key
    is_valid: bool,
    louds_sparse: &'a LoudsSparse,
    start_level: LevelT,
    /// Passed in by the dense iterator; default = 0
    start_node_num: PosT,
    /// Start counting from start_level; does NOT include suffix
    key_len: LevelT,
    key: Vec<u8>,
    pos_in_trie: Vec<PosT>,

    /// Store the index of the  current (sparse) value
    value_pos: Vec<PosT>,
    value_pos_initialized:  Vec<bool>,
    is_at_terminator: bool,

    is_first: bool,
}

impl<'a> LoudsSparseIterator<'a> {
    pub fn new(trie: &'a LoudsSparse) -> Self {
        Self {
            is_valid: false,
            louds_sparse: trie,
            start_level: trie.start_level,
            start_node_num: 0,
            key_len: 0,
            key: vec![0; trie.height],
            pos_in_trie: vec![0; trie.height],
            value_pos: vec![0; trie.height],
            value_pos_initialized: vec![false; trie.height],
            is_at_terminator: false,
            is_first: false,
        }
    }

    pub fn init(&mut self, node_num: PosT) {
        self.is_first = true;
        self.start_node_num = node_num;
        self.move_to_leftest_key();
    }

    pub fn move_to_leftest_key(&mut self) {
        if self.key_len == 0 {
            let pos = self.louds_sparse.first_label_pos(self.start_node_num);
            let label = self.louds_sparse.labels.read(pos);
            self.append(label, pos);
        }

        let mut level = self.key_len - 1;
        let mut pos = self.pos_in_trie[level];
        let mut label = self.louds_sparse.labels.read(pos);

        if !self.louds_sparse.child_indicator_bits.read_bit(pos) {
            if label == K_TERMINATOR && !self.louds_sparse.is_end_of_node(pos) {
                self.is_at_terminator = true;
            }
            self.is_valid = true;
            self.rank_value_position(pos);
            return;
        }

        while level < self.louds_sparse.height {
            let node_num = self.louds_sparse.child_node_num(pos);
            pos = self.louds_sparse.first_label_pos(node_num);
            label = self.louds_sparse.labels.read(pos);
            // if trie branch terminates
            if !self.louds_sparse.child_indicator_bits.read_bit(pos) {
                self.append(label, pos);
                if label == K_TERMINATOR && !self.louds_sparse.is_end_of_node(pos) {
                    self.is_at_terminator = true;
                }
                self.rank_value_position(pos);
                self.is_valid = true;
                return;
            }
            self.append(label, pos);
            level += 1;
        }
        unreachable!()
    }

    fn append(&mut self, label: u8, pos: PosT) {
        assert!(self.key_len < self.key.len());
        self.key[self.key_len] = label;
        self.pos_in_trie[self.key_len] = pos;
        self.key_len += 1;
    }

    fn rank_value_position(&mut self, pos: PosT) {
        if self.value_pos_initialized[self.key_len - 1] {
            self.value_pos[self.key_len - 1] += 1;
        } else {
            self.value_pos_initialized[self.key_len - 1] = true;
            let value_pos = pos - self.louds_sparse.child_indicator_bits.rank(pos);
            self.value_pos[self.key_len - 1] = value_pos;
        }
    }

    #[inline]
    fn get_value(&self) -> (Vec<u8>, u64) {
        let cut_off = if self.is_at_terminator {
            self.key_len - 1
        } else {
            self.key_len
        };
        (self.key[..cut_off].to_vec(), self.louds_sparse.positions_sparse[self.value_pos[self.key_len - 1]])
    }

    #[inline]
    fn set(&mut self, level: LevelT, pos: PosT) {
        assert!(level < self.key.len());
        self.key[level] = self.louds_sparse.labels.read(pos);
        self.pos_in_trie[level] = pos;
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.is_valid
    }

    pub(crate) fn set_start_node_num(&mut self, node_num: PosT) {
        self.start_node_num = node_num;
    }
}

impl<'a> Iterator for LoudsSparseIterator<'a> {
    type Item = (Vec<u8>, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.key_len == 0 {
            return None;
        }
        
        if self.is_first {
            self.is_first = false;
            // self.set(self.key_len - 1, pos);
            return Some(self.get_value());
        }

        self.is_at_terminator = false;
        let mut pos = self.pos_in_trie[self.key_len - 1];
        pos += 1;
        // louds_sparse.louds_bits is set for last label in a node -> node terminates here
        while pos >= self.louds_sparse.louds_bits.num_bits() || self.louds_sparse.louds_bits.read_bit(pos) {
            self.key_len -= 1;
            if self.key_len == 0 {
                self.is_valid = false;
                return None;
            }
            pos = self.pos_in_trie[self.key_len - 1];
            pos += 1;
        }

        self.set(self.key_len - 1, pos);
        self.move_to_leftest_key();
        return Some(self.get_value());
    }
}

#[cfg(test)]
mod tests {
    use crate::{FSTBuilder, louds_sparse::LoudsSparse};

    #[test]
    fn louds_sparse_prefix() {
        let mut builder = FSTBuilder::new(true, 9999);
        let keys = vec!["123".to_string(), "1234".to_string(), "12345".to_string()];
        builder.build(&keys, &vec![0, 1, 2]);
        println!("start_level: {:}", builder.sparse_start_level());
        let sparse = LoudsSparse::new(&builder);
        assert_eq!(sparse.lookup_key(&"123".to_string(), 0), Some(0));
        assert_eq!(sparse.lookup_key(&"1234".to_string(), 0), Some(1));
        assert_eq!(sparse.lookup_key(&"1233".to_string(), 0), None);
    }

    #[test]
    fn louds_sparse_simple() {
        let mut builder = FSTBuilder::new(true, 9999);
        let keys = vec!["123".to_string(), "124556".to_string(), "231".to_string()];
        builder.build(&keys, &vec![0, 1, 2]);
        println!("sprase pos: {:?}", builder.sparse_offset());
        println!("louds bit: {:?}", builder.louds_bits());
        println!("child bit: {:?}", builder.child_indicator_bits());

        let sparse = LoudsSparse::new(&builder);
        assert_eq!(sparse.lookup_key(&"123".to_string(), 0), Some(0));
        assert_eq!(sparse.lookup_key(&"124556".to_string(), 0), Some(1));
        assert_eq!(sparse.lookup_key(&"231".to_string(), 0), Some(2));
        assert_eq!(sparse.lookup_key(&"1234".to_string(), 0), None);
    }

    macro_rules! louds_sparse_iter {
        ($KEY: expr) => {
            let keys = $KEY;
            let mut builder = FSTBuilder::new(true, 9999);
            builder.build(&keys, &vec![0, 1, 2]);
            let sparse = LoudsSparse::new(&builder);
            sparse
            .iter()
            .enumerate()
            .for_each(|(i, (k, v))| {
                // println!("k {:}: {:?}", v, k);
                assert_eq!(&k, keys[i].as_bytes());
                assert_eq!(i, v as usize);
            })
        };
    }

    #[test]
    fn louds_sparse_iter_simple() {
        let keys = vec!["123".to_string(), "124556".to_string(), "231".to_string()];
        louds_sparse_iter!(keys);
    }

    #[test]
    fn loud_sparse_iter_prefix() {
        let keys = vec!["123".to_string(), "1234".to_string(), "1235".to_string()];
        louds_sparse_iter!(keys);
    }
}
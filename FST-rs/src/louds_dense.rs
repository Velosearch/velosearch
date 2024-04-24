use crate::{FSTBuilder, K_WORD_SIZE, BitVector, LevelT, PosT, K_FANOUT, K_TERMINATOR};

#[derive(Clone)]
pub struct LoudsDense {
    positions_dense: Vec<u64>,
    height: LevelT,
    label_bitmaps: BitVector,
    child_indicator_bitmaps: BitVector,
    prefixkey_indicator_bits: BitVector,
}

impl LoudsDense {
    pub fn new(builder: &FSTBuilder,) -> Self {
        let height = builder.sparse_start_level();
        let mut num_bits_per_level = Vec::new();
        for level in 0..height {
            num_bits_per_level.push(builder.bitmap_labels()[level].len() * K_WORD_SIZE);
        }
        
        let label_bitmaps = BitVector::new(
            builder.bitmap_labels().clone(), 
            &num_bits_per_level,
            0,
            height,
        );
        let child_indicator_bitmaps = BitVector::new(
            builder.bitmap_child_indicator_bits().clone(),
            &num_bits_per_level,
            0,
            height,
        );
        let prefixkey_indicator_bits = BitVector::new(
            builder.prefixkey_indicator_bits().clone(),
            builder.node_counts(),
            0,
            height,
        );
        let positions_dense = builder.dense_offset().clone();

        Self {
            positions_dense,
            label_bitmaps,
            height,
            child_indicator_bitmaps,
            prefixkey_indicator_bits,
        }
    }

    /// Returns whether key exists in the trie so far
    /// out_node_num == 0 means search terminates in louds_dense
    /// 3 offset 
    pub fn lookup_key(&self, key: &String) -> Option<(PosT, u64)> {
        return self.get(key.as_bytes())
    }

    pub fn get(&self, key: &[u8]) -> Option<(PosT, u64)> {
        let mut node_num = 0;
        let mut pos;
        let mut out_node_num = 0;
        let mut offset = 0;
        for level in 0..self.height {
            pos = node_num * K_FANOUT;
            if level >= key.len() {
                if self.prefixkey_indicator_bits.read_bit(node_num) {
                    if !self.label_bitmaps.read_bit(pos + K_TERMINATOR as usize) {
                        return None;
                    }
                    let 
                    value_index = self.label_bitmaps.rank(pos) 
                        - self.child_indicator_bitmaps.rank(pos) - 1;
                        // + self.prefixkey_indicator_bits.rank(pos / 256) -1;
                    offset = self.positions_dense[value_index];

                    return Some((0, offset));
                }
                return None;
            }
            pos += key[level] as usize;

            if !self.label_bitmaps.read_bit(pos) { // if key byte doesn't exist
                return None;
            }

            if !self.child_indicator_bitmaps.read_bit(pos) { // if trie branch terminates
                let value_index = self.label_bitmaps.rank(pos) 
                    - self.child_indicator_bitmaps.rank(pos) - 1;
                    // + self.prefixkey_indicator_bits.rank(pos / 256) - 1;
                offset = self.positions_dense[value_index];

                // the following check must be performed by the caller
                // return (*keys_)[value] == key;
                return Some((out_node_num, offset));
            }
            node_num = self.child_node_num(pos);
        }
        // search will continue in LoudsSparse
        out_node_num = node_num;
        return Some((out_node_num, offset));
    }

    #[inline]
    pub fn lookup_key_at_node(&self, key: &str, level: LevelT, node_num: &mut usize, value: &mut u64) -> bool {
        let mut pos;
        for level in level..self.height {
            pos = *node_num * K_FANOUT;
            if level >=  key.len() {
                if self.prefixkey_indicator_bits.read_bit(pos) {
                    return true
                }
                return false
            }
            pos += key.as_bytes()[level] as usize;

            if !self.label_bitmaps.read_bit(pos) { // if key byte does note exist
                return false;
            }

            if !self.child_indicator_bitmaps.read_bit(pos) { // if trie branch terminates
                let value_index = self.label_bitmaps.rank(pos) - self.child_indicator_bitmaps.rank(pos) - 1;

                *value = self.positions_dense[value_index];

                // the following check must be performed by the caller
                // return (*key)[value] == key;
                *node_num = 0;
                return true;
            }
            *node_num = self.child_node_num(pos);
        }
        // search will continue in LoudsSparse
        return true;
    }

    pub fn iter(&self) -> LoudsDenseIterator<'_> {
        LoudsDenseIterator::new(&self)
    }

    pub(crate) fn next_pos(&self, pos: PosT) -> PosT {
        pos + self.label_bitmaps.distance_to_next_set_bit(pos)
    }

    fn memory_usage(&self) -> usize {
        self.label_bitmaps.num_bits() / 8 + self.prefixkey_indicator_bits.num_bits() / 8 + self.positions_dense.len() / 8
    }

    #[inline]
    fn child_node_num(&self, pos: usize) -> usize {
        self.child_indicator_bitmaps.rank(pos)
    }
}

pub struct LoudsDenseIterator<'a> {
    key_len: usize,
    send_out_node_num: usize,
    is_at_prefix_key: bool,
    loud_dense: &'a LoudsDense,
    key: Vec<u8>,
    pos_in_trie: Vec<PosT>,
    value_pos: Vec<PosT>,
    value_pos_init: Vec<bool>,
    is_valid: bool,
    is_complete: bool,
    is_move_left_complete: bool,
}

impl<'a> LoudsDenseIterator<'a> {
    pub fn new(louds_dense: &'a LoudsDense) -> Self {
        // default set to the first key in louds_densee
        let mut pos_in_trie = vec![0; louds_dense.height];
        let mut keys = vec![0; louds_dense.height];
        if louds_dense.label_bitmaps.read_bit(0) {
            pos_in_trie[0] = 0;
            keys[0] = 0;
        } else {
            pos_in_trie[0] = louds_dense.next_pos(0);
            keys[0] = pos_in_trie[0];
        }
        Self {
            key_len: 1,
            send_out_node_num: 0,
            is_at_prefix_key: false,
            loud_dense: &louds_dense,
            key: vec![0; louds_dense.height],
            pos_in_trie: vec![0; louds_dense.height],
            value_pos: vec![0; louds_dense.height],
            value_pos_init: vec![false; louds_dense.height],
            is_valid: false,
            is_complete: false,
            is_move_left_complete: false,
        }
    }

    pub(crate) fn set_first_label_in_root(&mut self) {
        if self.loud_dense.label_bitmaps.read_bit(0) {
            self.pos_in_trie[0] = 0;
            self.key[0] = 0;
        } else {
            self.pos_in_trie[0] = self.loud_dense.next_pos(0);
            self.key[0] = self.pos_in_trie[0] as u8;
        }
        self.key_len += 1;
    }

    fn move_to_leftest_key(&mut self) {
        assert!(self.key_len > 0, "Should init correctly");
        let mut level = self.key_len - 1;
        let mut pos = self.pos_in_trie[level];
        // Branch terminates
        if !self.loud_dense.child_indicator_bitmaps.read_bit(pos) {
            self.rank_value_position(pos);
            // valid, search complete, moveLeft, complete, moveRight complete
            return self.set_flag(true, true, true);
        }

        while level < self.loud_dense.height - 1 {
            let node_num = self.loud_dense.child_node_num(pos);
            // if the current prefix is also a key
            if self.loud_dense.prefixkey_indicator_bits.read_bit(node_num) {
                let pos = self.loud_dense.next_pos(node_num * K_FANOUT - 1);
                self.append(pos);
                self.is_at_prefix_key = true;
                self.rank_value_position(pos);
                //
                return self.set_flag(true, true, true);
            }

            pos = self.loud_dense.next_pos(node_num * K_FANOUT - 1);
            self.append(pos);

            // if trie branch terminates
            if !self.loud_dense.child_indicator_bitmaps.read_bit(pos) {
                self.rank_value_position(pos);
                return self.set_flag(true, true, true);
            }
            level += 1;
        }
        self.send_out_node_num = self.loud_dense.child_node_num(pos);
        //
        return self.set_flag(true, true, true);
    }

    fn append(&mut self, pos: PosT) {
        assert!(self.key_len < self.key.len());
        self.key[self.key_len] = (pos % K_FANOUT) as u8;
        self.pos_in_trie[self.key_len] = pos;
        self.key_len += 1;
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.is_complete && self.is_move_left_complete
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.is_valid
    }

    fn set_flag(
        &mut self,
        is_valid: bool,
        is_search_complete: bool,
        is_move_left_complete: bool,
    ) {
        self.is_valid = is_valid;
        self.is_complete = is_search_complete;
        self.is_move_left_complete = is_move_left_complete;
    }

    fn rank_value_position(&mut self, pos: usize) {
        if self.value_pos_init[self.key_len - 1] {
            self.value_pos[self.key_len - 1] += 1;
        } else { // initially rank value position here
            self.value_pos_init[self.key_len - 1] = true;
            let value_index = self.loud_dense.label_bitmaps.rank(pos)
                - self.loud_dense.child_indicator_bitmaps.rank(pos)
                // + self.loud_dense.prefixkey_indicator_bits.rank(pos / 256)
                - 1;
            self.value_pos[self.key_len - 1] = value_index;
        }
    }

    fn set(&mut self, level: LevelT, pos: PosT) {
        assert!(level < self.key.len());
        self.key[level] = (pos % K_FANOUT) as u8;
        self.pos_in_trie[level] = pos;
    }

    #[inline]
    fn get_value(&self) -> (Vec<u8>, u64) {
        (self.key[..self.key_len].to_vec(), *self.loud_dense.positions_dense.get(self.value_pos[self.key_len - 1]).unwrap_or(&0))
    }

    #[inline]
    fn get_prefix_value(&self) -> (Vec<u8>, u64) {
        (self.key[..(self.key_len - 1)].to_vec(), self.loud_dense.positions_dense[self.value_pos[self.key_len - 1]])
    }

    pub(crate) fn send_out_node_num(&self) -> usize {
        self.send_out_node_num
    }

    pub(crate) fn is_move_left_complete(&self) -> bool {
        self.is_move_left_complete
    }
}

impl<'a> Iterator for LoudsDenseIterator<'a> {
    type Item = (Vec<u8>, u64);

    fn next(&mut self) -> Option<Self::Item> {
        assert!(self.key_len > 0, "Should init correctly");
        
        let mut pos = self.pos_in_trie[self.key_len - 1];
        let mut next_pos = self.loud_dense.next_pos(pos);
        // if crossing node boundary
        while next_pos / K_FANOUT > pos / K_FANOUT {
            self.key_len -= 1;
            if self.key_len == 0 {
                self.is_valid = false;
                return None;
            }
            pos = self.pos_in_trie[self.key_len - 1];
            next_pos = self.loud_dense.next_pos(pos);
        }
        self.set(self.key_len - 1, next_pos);
        self.move_to_leftest_key();

        if self.is_at_prefix_key {
            self.is_at_prefix_key = false;
            return Some(self.get_prefix_value());
        }

        return Some(self.get_value())
    }
}

#[cfg(test)]
mod tests {
    use crate::FSTBuilder;

    use super::LoudsDense;

    #[test]
    fn louds_dense_lookup_prefix() {
        let mut builder = FSTBuilder::new(true, 0);
        let keys = vec!["123".to_string(), "12345".to_string(), "1244".to_string()];
        builder.build(&keys, &vec![0, 1, 2]);
        println!("bitmaplabels: {:?}", builder.bitmap_labels());
        println!("dense position: {:?}", builder.dense_offset());
        let dense = LoudsDense::new(&builder);
        assert_eq!(dense.lookup_key(&"1244".to_string()).unwrap(), (0, 2));
        assert_eq!(dense.lookup_key(&"1245".to_string()), None);
        assert_eq!(dense.lookup_key(&"12345".to_string()), Some((0, 1)));
        assert_eq!(dense.lookup_key(&"123".to_string()), Some((0, 0)));

        let mut builder = FSTBuilder::new(true, 0);
        let keys = vec!["123".to_string(), "124556".to_string(), "211".to_string()];
        builder.build(&keys, &vec![0, 1, 2]);
        println!("bitmaplabels: {:?}", builder.bitmap_labels());
        println!("dense position: {:?}", builder.dense_offset());
        let dense = LoudsDense::new(&builder);
        assert_eq!(dense.lookup_key(&"123".to_string()).unwrap(), (0, 0));
        assert_eq!(dense.lookup_key(&"12456".to_string()), None);
        assert_eq!(dense.lookup_key(&"124556".to_string()), Some((0, 1)));
        assert_eq!(dense.lookup_key(&"211".to_string()), Some((0, 2)));
    }

    #[test]
    fn louds_dense_lookup_simple() {
        let mut builder = FSTBuilder::new(true, 0);
        let keys = vec!["123".to_string(), "124".to_string(), "1255".to_string()];
        builder.build(&keys, &vec![0, 1, 2]);
        let dense = LoudsDense::new(&builder);
        assert_eq!(dense.lookup_key(&"123".to_string()), Some((0, 0)));
        assert_eq!(dense.lookup_key(&"124".to_string()), Some((0, 1)));
        assert_eq!(dense.lookup_key(&"1255".to_string()), Some((0, 2)));
        assert_eq!(dense.lookup_key(&"1222".to_string()), None);
    }

    macro_rules! louds_dense_iter {
        ($KEYS: expr) => {
            let mut builder = FSTBuilder::new(true, 0);
            let keys = $KEYS;
            let values = (0..keys.len()).into_iter().collect::<Vec<usize>>();
            builder.build(&keys, &values);

            let dense = LoudsDense::new(&builder);
            dense.iter().enumerate().for_each(|(i, (k, v))| {
                assert_eq!(&k, keys[i].as_bytes());
                assert_eq!(v, i as u64);
                // println!("key {:}: {:?}", v, k);
            })
        }
    }

    #[test]
    fn louds_dense_simple() {
        // simple test
        let keys = vec!["122".to_string(), "123".to_string(), "1244".to_string()];
        louds_dense_iter!(keys);

        // prefix test
        let keys = vec!["122".to_string(), "1233".to_string(), "12334".to_string()];
        louds_dense_iter!(keys);
    }
}
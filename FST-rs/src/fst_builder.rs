use arrow::ipc::KeyValue;

use crate::{LabelsT, PosT, LevelT, K_WORD_SIZE, K_FANOUT, K_TERMINATOR};

#[derive(Clone)]
pub struct FSTBuilder {
    include_dense: bool,
    sparse_dense_ratio: u32,
    sparse_start_level: usize,

    positions: Vec<Vec<u64>>,

    /// LOUDS-Sparse bit/byte vectors
    labels: Vec<Vec<LabelsT>>,
    child_indicator_bits: Vec<Vec<u8>>,
    louds_bits: Vec<Vec<u8>>,
    positions_sparse: Vec<u64>,

    /// LOUDS-Dense bit vectors
    bitmap_labels: Vec<Vec<u8>>,
    bitmap_child_indicator_bits: Vec<Vec<u8>>,
    prefixkey_indicator_bits: Vec<Vec<u8>>,
    positions_dense: Vec<u64>,

    /// auxiliary per level bookkeeping vectors
    node_counts: Vec<PosT>,
    is_last_item_terminator: Vec<bool>,
}

impl FSTBuilder {
    pub fn new(include_dense: bool, sparse_dense_ratio: u32) -> Self {
        Self {
            include_dense,
            sparse_dense_ratio,
            sparse_start_level: 0,
            positions: Vec::new(),
            labels: Vec::new(),
            child_indicator_bits: Vec::new(),
            louds_bits: Vec::new(),
            positions_sparse: Vec::new(),
            bitmap_labels: Vec::new(),
            bitmap_child_indicator_bits: Vec::new(),
            prefixkey_indicator_bits: Vec::new(),
            positions_dense: Vec::new(),
            node_counts: Vec::new(),
            is_last_item_terminator: Vec::new(),
        }
    }

    pub fn build(&mut self, keys: &Vec<String>, values: &Vec<usize>) {
        assert!(keys.len() > 0);
        self.build_sparse(keys, values);
        println!("sparse labels: {:?}", self.labels());
        if self.include_dense {
            self.determine_cutoff_level();
            self.build_dense();
        }
    }

    pub fn build_with_bytes(&mut self, keys: &Vec<Vec<u8>>, values: &Vec<usize>) {
        assert!(keys.len() > 0, "keys: {:?}", keys);
        self.build_sparse_with_bytes(keys, values);

        if self.include_dense {
            self.determine_cutoff_level();
            self.build_dense();
        }
    }

    pub fn bitmap_labels(&self) -> &Vec<Vec<u8>> {
        &self.bitmap_labels
    }
    
    pub fn labels(&self) -> &Vec<Vec<u8>> {
        &self.labels
    }

    pub fn sparse_start_level(&self) -> usize {
        self.sparse_start_level
    } 

    pub fn bitmap_child_indicator_bits(&self) -> &Vec<Vec<u8>> {
        &self.bitmap_child_indicator_bits
    }

    pub fn child_indicator_bits(&self) -> &Vec<Vec<u8>> {
        &self.child_indicator_bits
    }

    pub fn prefixkey_indicator_bits(&self) -> &Vec<Vec<u8>> {
        &self.prefixkey_indicator_bits
    }

    pub fn dense_offset(&self) -> &Vec<u64> {
        &self.positions_dense
    }

    pub fn sparse_offset(&self) -> &Vec<u64> {
        &self.positions_sparse
    }

    pub fn node_counts(&self) -> &Vec<usize> {
        &self.node_counts
    }

    pub fn louds_bits(&self) -> &Vec<Vec<u8>> {
        &self.louds_bits
    }

    fn build_dense(&mut self) {
        for level in 0..self.sparse_start_level {
            self.init_dense_vector(level);
            if self.get_num_items(level) == 0 {
                continue;
            }

            let mut node_num = 0;
            if self.is_terminator(level, 0) {
                set_bit(&mut self.prefixkey_indicator_bits[level], 0);
                set_bit(&mut self.bitmap_labels[level], K_TERMINATOR as usize);
            } else {
                self.set_label_and_child_indicator_bitmap(level, node_num, 0);
            }
            for pos in 1..self.get_num_items(level) {
                if self.is_start_of_node(level, pos) {
                    node_num += 1;
                    if self.is_terminator(level, pos) {
                        set_bit(&mut self.prefixkey_indicator_bits[level], node_num);
                        set_bit(&mut self.bitmap_labels[level], node_num * K_FANOUT + K_TERMINATOR as usize);
                        continue;
                    }
                }
                self.set_label_and_child_indicator_bitmap(level, node_num, pos);
            }
        }
    }

    fn is_start_of_node(&self, level: LevelT, pos: PosT) -> bool {
        read_bit(&self.louds_bits[level], pos)
    }

    fn set_label_and_child_indicator_bitmap(
        &mut self,
        level: LevelT,
        node_num: PosT,
        pos: PosT,
    ) {
        let label = self.labels[level][pos];
        set_bit(&mut self.bitmap_labels[level], node_num * K_FANOUT + label as usize);
        if read_bit(&self.child_indicator_bits[level], pos) {
            set_bit(&mut self.bitmap_child_indicator_bits[level], node_num * K_FANOUT + label as usize)
        }
    }

    fn is_terminator(&self, level: LevelT, pos: PosT) -> bool {
        let label: LabelsT = self.labels[level][pos as usize];
        (label == K_TERMINATOR) && !read_bit(&self.child_indicator_bits[level], pos)
    }

    fn init_dense_vector(&mut self, level: LevelT) {
        self.bitmap_labels.push(Vec::new());
        self.bitmap_child_indicator_bits.push(Vec::new());
        self.prefixkey_indicator_bits.push(Vec::new());

        for nc in 0..self.node_counts[level] {
            for _ in (0..K_FANOUT).step_by(K_WORD_SIZE as usize) {
                self.bitmap_labels[level].push(0);
                self.bitmap_child_indicator_bits[level].push(0);
            }
            if nc % K_WORD_SIZE == 0 {
                self.prefixkey_indicator_bits[level].push(0);
            }
        }
    }

    #[inline]
    fn determine_cutoff_level(&mut self) {
        let mut cutoff_level: LevelT = 0;
        let mut dense_mem: u64 = self.compute_dense_mem(cutoff_level);
        let mut sparse_mem = self.compute_sparse_mem(cutoff_level);
        if self.sparse_dense_ratio < 50 {
        while cutoff_level < self.tree_height() && (dense_mem * self.sparse_dense_ratio as u64) < sparse_mem {
            cutoff_level += 1;
            dense_mem = self.compute_dense_mem(cutoff_level);
            sparse_mem = self.compute_sparse_mem(cutoff_level);
        } 
        }
        // cutoff_level = 3;
        self.sparse_start_level = cutoff_level;
        // cutoff_level -= 1;

        // CA build dense and sparse values vectors
        for level in 0..self.sparse_start_level {
            self.positions_dense.extend_from_slice(self.positions[level as usize].as_slice());
        }

        for level in self.sparse_start_level..self.positions.len() {
            self.positions_sparse.extend_from_slice(self.positions[level as usize].as_slice());
        }
        self.positions.clear();
    }

    #[inline]
    fn compute_dense_mem(&self, downto_level: LevelT) -> u64 {
        assert!(downto_level <= self.tree_height());
        let mut mem: u64 = 0;
        for level in 0..downto_level {
            mem += 2 * K_FANOUT as u64 * self.node_counts[level] as u64;
            if level > 0 {
                mem += self.node_counts[level - 1] as u64 / 8 + 1;
            }
        }
        mem
    }

    #[inline]
    fn compute_sparse_mem(&self, start_level: LevelT) -> u64 {
        let mut mem: u64 = 0;
        for level in start_level..self.tree_height() {
            let num_items = self.labels[level].len();
            mem += (num_items + 2 * num_items / 8 + 1) as u64;
        }
        mem
    }

    fn build_sparse(&mut self, keys: &Vec<String>, _values: &Vec<usize>) {
        let mut i = 0;
        while i < keys.len() {
            let level: LevelT = self.skip_common_prefix(&keys[i].as_bytes());
            let cur_pos = i;
            while (i + 1 < keys.len()) && keys[cur_pos] == keys[i + 1] { i += 1 };
            if i < keys.len() - 1 {
                self.insert_key_to_trie(&keys[cur_pos].as_bytes(), cur_pos, keys[i + 1].as_bytes(), level);
            } else {
                self.insert_key_to_trie(&keys[cur_pos].as_bytes(), cur_pos, &[], level);
            }
            i += 1;
        }
    }

    fn build_sparse_with_bytes(&mut self, keys: &Vec<Vec<u8>>, _values: &Vec<usize>) {
        let mut i = 0;
        while i < keys.len() {
            let level: LevelT = self.skip_common_prefix(&keys[i]);
            let cur_pos = i;
            while (i + 1 < keys.len()) && keys[cur_pos] == keys[i + 1] { i += 1};
            if i < keys.len() - 1 {
                self.insert_key_to_trie(&keys[cur_pos], cur_pos, &keys[i + 1], level);
            } else {
                self.insert_key_to_trie(&keys[cur_pos], cur_pos, &[], level);
            }
            i += 1;
        }
    }

    fn insert_key_to_trie(
        &mut self,
        key: &[u8],
        posi: usize,
        next_key: &[u8],
        start_level: LevelT,
    ) -> LevelT {
        assert!(start_level < key.len(), "key: {:?}, start_level: {:}", &key, start_level);

        let mut level: LevelT = start_level;
        let mut is_start_of_node: bool = false;
        let mut is_term: bool = false;
        // If it is the start of level, the louds bit needs to be set.
        if self.is_level_empty(level) {
            is_start_of_node = true;
        }


        // After skipping the common prefix, the first following byte
        // should be in the node as the previous key.
        self.insert_key_byte(key[level], level, is_start_of_node, is_term);
        level += 1;
        // if key.len() > next_key.len() || key[..level] != next_key[..level] {
        //     if level >= key.len() {
        //         self.positions[level - 1].push(posi as u64);
        //         return level;
        //     }
        // }
        // if level > next_key.len() || &key[0..level] != &next_key[0..level] {
        //     self.positions[level - 1].push(posi as u64);
        //     return level;
        // }

        // All the following bytes inserted must be the start of a new node.
        is_start_of_node = true;
        while level < key.len() && level < next_key.len() && key[level - 1] == next_key[level - 1] {
            self.insert_key_byte(key[level], level, is_start_of_node, is_term); 
            level += 1;
        }
     
        // The last byte inserted makes key unique in the trie.
        if key.len() > next_key.len() || key != &next_key[..key.len()]  {
            while level < key.len() {
                self.insert_key_byte(key[level], level, is_start_of_node, is_term);
                // self.positions[level].push(posi as u64);
                level += 1;
            }
            self.positions[level - 1].push(posi as u64);
        } else {
            is_term = true;
            self.insert_key_byte(K_TERMINATOR, level, is_start_of_node, is_term);
            self.positions[level].push(posi as u64)
        }

        level
    }

    fn insert_key_byte(
        &mut self,
        c: u8,
        level: LevelT,
        is_start_of_node: bool,
        is_term: bool,
    ) {
        // level should be at most equal to tree height
        if level >= self.tree_height() { self.add_level() };

        assert!(level < self.tree_height());

        // sets parent node's child indicator
        if level > 0 {
            let posi = self.get_num_items(level - 1) - 1;
            set_bit(&mut self.child_indicator_bits[level as usize - 1], posi);
        }

        self.labels[level as usize].push(c);
        if is_start_of_node {
            let posi = self.get_num_items(level) - 1;
            set_bit(&mut self.louds_bits[level], posi);
            self.node_counts[level] += 1;
        }

        self.is_last_item_terminator[level] = is_term;

        self.move_to_next_slot(level);
    }

    #[inline]
    fn move_to_next_slot(&mut self, level: LevelT) {
        assert!(level < self.tree_height());
        let num_items= self.get_num_items(level);
        if num_items % K_WORD_SIZE == 0 {
            self.child_indicator_bits[level].push(0);
            self.louds_bits[level].push(0);
        }
    }

    fn add_level(&mut self) {
        self.labels.push(Vec::new());
        self.positions.push(Vec::new());
        self.child_indicator_bits.push(Vec::new());
        self.louds_bits.push(Vec::new());

        self.node_counts.push(0);
        self.is_last_item_terminator.push(false);

        let height = self.tree_height() - 1;
        self.child_indicator_bits[height].push(0);
        self.louds_bits[height].push(0);
    }

    fn is_level_empty(&self, level: LevelT) -> bool {
        (level >= self.tree_height()) || (self.labels[level as usize].is_empty())
    }

    fn skip_common_prefix(&mut self, key: &[u8]) -> LevelT  {
        let mut level = 0;
        while level < key.len() && self.is_char_common_prefix(key[level], level) {
            let posi = self.get_num_items(level) - 1;
            set_bit(&mut self.child_indicator_bits[level], posi);
            level += 1;
        }
        return level as LevelT;
    }

    fn get_num_items(&self, level: usize) -> PosT {
        self.labels[level as usize].len()
    }

    fn is_char_common_prefix(&self, c: LabelsT, level: usize) -> bool {
        return level < self.tree_height() && !self.is_last_item_terminator[level] && c == *self.labels[level].last().unwrap()
    }

    fn tree_height(&self) -> LevelT {
        return self.labels.len()
    }
}


fn set_bit(bits: &mut Vec<u8>, pos: PosT) {
    assert!(pos < (bits.len() * K_WORD_SIZE), "bits: {:?}, pos: {:}", bits, pos);
    let word_id: PosT = pos / K_WORD_SIZE;
    let offset: PosT = pos % K_WORD_SIZE;
    bits[word_id as usize] |= 1 << offset;
}

fn read_bit(bits: &Vec<u8>, pos: PosT) -> bool {
    assert!(pos < (bits.len() * K_WORD_SIZE));
    let word_id = pos / K_WORD_SIZE;
    let offset = pos % K_WORD_SIZE;
    if bits[word_id as usize] & (1 << offset) > 0 {
        true
    } else {
        false
    }
}

#[cfg(test)]
mod test {
    use crate::FSTBuilder;

    #[test]
    fn fst_builder_simple_test() {
        let mut builder = FSTBuilder::new(true, 64);
        let keys = vec![String::from("123"), String::from("1234"), String::from("125")];
        let values = vec![1, 2, 3];
        builder.build(&keys, &values);
        println!("postion: {:?}", builder.positions);
        println!("dense postion: {:?}", builder.dense_offset());
        println!("sparse position: {:?}", builder.sparse_offset());

    }

    #[test]
    fn fst_builder_dense() {
        let mut builder = FSTBuilder::new(true, 0);
        let keys = vec![String::from("123"), String::from("1234"), String::from("125")];
        let values = vec![1, 2, 3];
        builder.build(&keys, &values);
        println!("prefixkey_indicator_bits(): {:?}", builder.prefixkey_indicator_bits());
        println!("child_indicator_bits(): {:?}", builder.bitmap_child_indicator_bits());
        println!("bitmap_label: {:?}", builder.bitmap_labels());
    }
}
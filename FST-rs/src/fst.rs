use crate::{K_INCLUDE_DENSE, K_SPARSE_DENSE_RATIO, FSTBuilder, louds_sparse::{LoudsSparse, LoudsSparseIterator}, louds_dense::{LoudsDense, LoudsDenseIterator}};

#[derive(Clone)]
pub struct FST {
    builder: FSTBuilder,
    louds_dense: LoudsDense,
    louds_sparse: Option<LoudsSparse>,
    values: Vec<usize>,
}

impl FST {
    pub fn new(keys: Vec<String>, values: Vec<usize>) -> Self {
        Self::new_impl(keys, values, K_INCLUDE_DENSE, K_SPARSE_DENSE_RATIO)
    }

    pub fn new_with_bytes(keys: &Vec<Vec<u8>>, values: Vec<usize>) -> Self {
        let mut builder = FSTBuilder::new(K_INCLUDE_DENSE, K_SPARSE_DENSE_RATIO);
        builder.build_with_bytes(&keys, &values);
        let louds_dense = LoudsDense::new(&builder);
        let louds_sparse = if builder.node_counts().len() > builder.sparse_start_level() {
            Some(LoudsSparse::new(&builder))
        } else {
            None
        };

        Self {
            builder,
            louds_dense,
            louds_sparse,
            values,
        }
    }

    pub fn iter(&self) -> FSTIterator {
        FSTIterator::new(&self)
    }

    fn new_impl(
        keys: Vec<String>,
        values: Vec<usize>,
        include_dense: bool,
        sparse_dense_ratio: u32,
    ) -> Self {
        let mut builder = FSTBuilder::new(include_dense, sparse_dense_ratio);
        builder.build(&keys, &values);
        let louds_dense = LoudsDense::new(&builder);
        let louds_sparse = if builder.node_counts().len() > builder.sparse_start_level() {
            Some(LoudsSparse::new(&builder))
        } else {
            None
        };
        println!("start_level: {:}", builder.sparse_start_level());
        println!("sparse: lavel: {:?}", builder.labels());
        println!("bitmap label: {:?}", builder.bitmap_labels());
        println!("dense position: {:?}", builder.dense_offset());
        Self {
            builder,
            louds_dense,
            louds_sparse,
            values,
        }
    }

    pub fn lookup_key(&self, key: &String) -> Option<usize> {
        let connect_node_num;
        let value;
        if let Some((node_n, v)) = self.louds_dense.lookup_key(key) {
            connect_node_num = node_n;
            value = v;
        } else {
            return None;
        }
        if connect_node_num != 0 {
            return self.louds_sparse
                .as_ref()
                .unwrap()
                .lookup_key(key, connect_node_num)
                .map(|v| self.values[v as usize]);
        }
        return Some(self.values[value as usize]);
    }

    pub fn get(&self, key: &[u8]) -> Option<usize> {
        let connect_node_num;
        let value;
        if let Some((node_n, v)) = self.louds_dense.get(key) {
            connect_node_num = node_n;
            value = v;
        } else {
            return None;
        }
        if connect_node_num != 0 {
            return self.louds_sparse
                .as_ref()
                .unwrap()
                .get(key, connect_node_num)
                .map(|v| self.values[v as usize]);
        }
        return Some(self.values[value as usize])
    }
}

pub struct FSTIterator<'a> {
    louds_dense: LoudsDenseIterator<'a>,
    louds_sparse: Option<LoudsSparseIterator<'a>>,
    prefix: Option<Vec<u8>>,
}

impl<'a> FSTIterator<'a> {
    pub fn new(fst: &'a FST) -> Self {
        let mut fst_iter = Self {
            louds_dense: LoudsDenseIterator::new(&fst.louds_dense),
            louds_sparse: fst.louds_sparse
                .as_ref()
                .map(|v| LoudsSparseIterator::new(v)),
            prefix: None,
        };
        fst_iter
    }

    fn is_valid(&self) -> bool {
        self.louds_dense.is_valid() &&
        self.louds_dense.is_complete() &&
        (self.louds_sparse.is_none() || self.louds_sparse.as_ref().unwrap().is_valid())
    }

    fn increment_sparse_iter(&mut self) -> Option<(Vec<u8>, u64)> {
        if self.prefix.is_none() {
            return None;
        }
        match self.louds_sparse.as_mut() {
            Some(sparse) => sparse.next(),
            None => None,
        }
    }
}

impl<'a> Iterator for FSTIterator<'a> {
    type Item = (Vec<u8>, u64);

    fn next(&mut self) -> Option<Self::Item> {
        match self.increment_sparse_iter() {
            Some(v) => Some(([self.prefix.as_ref().unwrap().clone(), v.0].concat(), v.1)),
            None => {
                let dense_output = self.louds_dense.next();
                match dense_output {
                    Some(dense) => {
                        if self.louds_dense.send_out_node_num() == 0 {
                            return Some(dense);
                        } else {
                            self.prefix = Some(dense.0)
                        }
                    }
                    None => return None,
                }
                match self.louds_sparse.as_mut() {
                    Some(sparse) => {
                        sparse.init(self.louds_dense.send_out_node_num());
                        match sparse.next() {
                            Some(output) => Some(([self.prefix.as_ref().unwrap().clone(), output.0].concat(), output.1)),
                            None => None,
                        }
                    }
                    None => None,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::FST;

    #[test]
    fn fst_simple() {
        let fst = FST::new(vec!["123".to_string(), "1234".to_string(), "125".to_string()], vec![0, 1, 2]);
        assert_eq!(fst.lookup_key(&"1234".to_string()), Some(1));
        assert_eq!(fst.lookup_key(&"123".to_string()), Some(0));
        assert_eq!(fst.lookup_key(&"13".to_string()), None);
        assert_eq!(fst.lookup_key(&"125".to_string()), Some(2));
    }

    #[test]
    fn fst_simple_2() {
        let fst = FST::new(
            vec!["123".to_string(), "12567896".to_string(), "12567899".to_string()],
            vec![0, 1, 2],
        );
        assert_eq!(fst.get("123".as_bytes()), Some(0));
        assert_eq!(fst.get("12567896".as_bytes()), Some(1));
        assert_eq!(fst.get("12567899".as_bytes()), Some(2));
    }

    macro_rules! fst_iter {
        ($KEY: expr) => {
            let keys = $KEY;
            let fst = FST::new(
                keys.clone(),
                (0..keys.len()).collect::<Vec<usize>>(),
            );
            fst
                .iter()
                .enumerate()
                .for_each(|(i, (k, v))| {
                    assert_eq!(i, v as usize);
                    assert_eq!(&k, keys[i].as_bytes());
                })
        };
    }

    #[test]
    fn fst_iter_simple() {
        // iter with prefix
        let keys = vec!["1234".to_string(), "12345".to_string(), "225699".to_string()];
        fst_iter!(keys);

        let keys = vec!["12516".to_string(), "12566".to_string(), "13334".to_string(), "142564235".to_string()];
        fst_iter!(keys);
    }
}
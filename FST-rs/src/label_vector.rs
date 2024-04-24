use std::fmt::Debug;

use crate::{LabelsT, LevelT, PosT, K_TERMINATOR};

#[derive(Clone)]
pub(crate) struct LabelVector {
    num_bytes: usize,
    labels: Vec<u8>
}

impl LabelVector {
    pub fn new(
        labels_per_level: &Vec<Vec<LabelsT>>,
        start_level: LevelT,
        end_level: LevelT
    ) -> Self {
        let mut num_bytes = 1;
        for level in start_level..end_level {
            num_bytes += labels_per_level[level].len();
        }
        let mut labels = Vec::with_capacity(num_bytes);
        for level in start_level..end_level {
            for idx in 0..labels_per_level[level].len() {
                labels.push(labels_per_level[level][idx]);
            }
        }
        Self {
            num_bytes,
            labels,
        }
    }

    pub fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    pub fn read(&self, pos: PosT) -> LabelsT {
        self.labels[pos]
    }

    pub fn search(&self, target: LabelsT, pos: PosT, search_len: PosT) -> Option<PosT> {
        let mut pos = pos;
        let mut search_len = search_len;
        // skip terminator label
        if search_len > 1 && self.labels[pos] == K_TERMINATOR {
            pos += 1;
            search_len -= 1;
        }

        if search_len < 3 {
            return self.linear_search(target, pos, search_len);
        } else {
            return self.binary_search(target, pos, search_len);
        }
    }

    #[inline]
    fn linear_search(&self, target: LabelsT, pos: PosT,  search_len: PosT) -> Option<PosT> {
        for i in 0..search_len {
            if target == self.labels[pos + i] {
                return Some(pos + i);
            }
        }
        return None;
    }

    #[inline]
    fn binary_search(&self, target: LabelsT, pos: PosT, search_len: PosT) -> Option<PosT> {
        match self.labels[pos..(pos + search_len)]
            .binary_search(&target) {
                Ok(pos) => Some(pos),
                Err(_) => None,
        }
    }
}

impl Debug for LabelVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.labels)
    }
}
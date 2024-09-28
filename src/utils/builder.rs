use std::{collections::HashMap, fs::{self, File}, io::{BufReader, BufWriter}, path::PathBuf, sync::Arc};

use adaptive_hybrid_trie::TermIdx;
use datafusion::{arrow::datatypes::{Schema, Field, DataType}, common::TermMeta};
use memmap::MmapOptions;
use roaring::RoaringBitmap;
use tracing::info;

use crate::{batch::{BatchRange, PostingBatchBuilder}, datasources::{mmap_table::MmapTable, posting_table::PostingTable}};

#[derive(serde::Serialize, serde::Deserialize)]
struct TermMetaTemp {
    /// Which horizantal partition batches has this Term
    pub valid_bitmap: Vec<Vec<u32>>,
    /// Witch Batch has this Term
    pub index: Arc<Vec<Option<u32>>>,
    /// The number of this Term
    pub nums: Vec<u32>,
    /// Selectivity
    pub selectivity: f64,
}

impl TermMetaTemp {
    pub fn rle_usage(&self) -> usize {
        0
    }
}

pub fn serialize_term_meta(term_meta: &Vec<TermMeta>, dump_path: String) {
    let path = PathBuf::from(dump_path);
    let f = File::create(path.join(PathBuf::from("term_values.bin"))).unwrap();
    let writer = BufWriter::new(f);
    let term_metas: Vec<TermMetaTemp> = term_meta
        .iter()
        .map(|v| {
            let valid_bitmap: Vec<Vec<u32>> = v.valid_bitmap
                .as_ref()
                .iter()
                .map(|v| {
                    v.iter().collect()
                })
                .collect();
            TermMetaTemp {
                valid_bitmap,
                index: v.index.clone(),
                nums: v.nums.clone(),
                selectivity: v.selectivity.clone(),
            }
        })
        .collect();
    let consumption: usize = term_metas.iter().map(|v| v.rle_usage()).sum();
    info!("terms len: {:}", term_metas.len());
    info!("Compressed index consumption: {:}", consumption);
    bincode::serialize_into::<_, Vec<TermMetaTemp>>(writer, &term_metas).unwrap();
}

pub fn deserialize_posting_table(dump_path: String, partitions_num: usize) -> Option<PostingTable> {
    info!("Deserialize data from {:}", dump_path);
    let path = PathBuf::from(dump_path);
    let posting_batch: Vec<PostingBatchBuilder>;
    let batch_range: BatchRange;
    let keys: Vec<String>;
    let values: Vec<TermMetaTemp>;
    // batch_range.bin
    if let Ok(f) = File::open(path.join(PathBuf::from("batch_ranges.bin"))) {
        let reader = BufReader::new(f);
        batch_range = bincode::deserialize_from(reader).unwrap();
    } else {
        return None;
    }
    // term_keys.bin
    if let Ok(f) = File::open(path.join(PathBuf::from("term_keys.bin"))) {
        let reader = BufReader::new(f);
        keys = bincode::deserialize_from(reader).unwrap();
    } else {
        return None;
    }


    let (fields_index, fields) = keys
        .iter()
        .chain([&"__id__".to_string()].into_iter())
        .enumerate()
        .map(|(i, v)| {
            let idx = (v.to_string(), i);
            let field = Field::new(v.to_string(), DataType::Boolean, false);
            (idx, field)
        })
        .unzip();
    let schema = Schema {
        fields,
        metadata: HashMap::new(),
        fields_index: Some(fields_index),
    };


    let mut memory_consume = 0;
    let mut compressed_consume = 0;

    // term_values.bin
    if let Ok(f) = File::open(path.join(PathBuf::from("term_values.bin"))) {
        let reader = BufReader::new(f);
        values = bincode::deserialize_from(reader).unwrap();
    } else {
        return None;
    }
    info!("start build TermDict");
    let values: Vec<TermMeta> = values
        .into_iter()
        .map(|v| {
            compressed_consume += v.rle_usage();
         
            let valid_bitmap = v.valid_bitmap
                .into_iter()
                .map(|v| {
                    Arc::new(RoaringBitmap::from_sorted_iter(v.into_iter()).unwrap())
                })
                .collect();
            let termmeta = TermMeta {
                valid_bitmap: Arc::new(valid_bitmap),
                index: v.index,
                nums: v.nums,
                selectivity: v.selectivity,
            };
            memory_consume += termmeta.memory_consumption();
            termmeta
        })
        .collect();
    info!("term len: {:}", values.len());
    info!("term index: {:}", memory_consume);
    info!("compreed index: {:}", compressed_consume);
    let bitmap_consumption: usize = values.iter().map(|v| v.valid_bitmap.as_ref()[0].memory_consumption()).sum();
    info!("valid bitmap consumption: {:}", bitmap_consumption);
    #[cfg(all(feature = "trie_idx", not(feature = "hash_idx")))]
    let term_idx = Arc::new(TermIdx::new(keys, values, 20));
    #[cfg(feature = "hash_idx")]
    let term_idx = Arc::new(TermIdx { term_map: HashMap::from_iter(keys.into_iter().zip(values.into_iter())) });

    info!("finish deserializing index");

    // posting_batch.bin
    if let Ok(f) = File::open(path.join(PathBuf::from("posting_batch.bin"))) {
        let reader = BufReader::new(f);
        posting_batch = bincode::deserialize_from(reader).unwrap();
    } else {
        return None;
    }
    let partition_batch = posting_batch
        .into_iter()
        .map(|b| Arc::new(
            b.build().unwrap()
        ))
        .collect();

    Some(PostingTable::new(
        Arc::new(schema),
        term_idx,
        partition_batch,
        &batch_range,
        partitions_num,
    ))
}

pub fn deserialize_mmap_table(dump_path: String, _partitions_num: usize) -> Option<MmapTable> {
    info!("Deserialize data from {:}", dump_path);
    let path = PathBuf::from(dump_path);
    
    let mut posting_batch: Vec<PostingBatchBuilder>;
    let batch_range: BatchRange;
    let keys: Vec<String>;
    let values: Vec<TermMetaTemp>;
    // batch_range.bin
    if let Ok(f) = File::open(path.join(PathBuf::from("batch_ranges.bin"))) {
        let reader = BufReader::new(f);
        batch_range = bincode::deserialize_from(reader).unwrap();
    } else {
        return None;
    }
    // term_keys.bin
    if let Ok(f) = File::open(path.join(PathBuf::from("term_keys.bin"))) {
        let reader = BufReader::new(f);
        keys = bincode::deserialize_from(reader).unwrap();
    } else {
        return None;
    }


    let (fields_index, fields) = keys
        .iter()
        .chain([&"__id__".to_string()].into_iter())
        .enumerate()
        .map(|(i, v)| {
            let idx = (v.to_string(), i);
            let field = Field::new(v.to_string(), DataType::Boolean, false);
            (idx, field)
        })
        .unzip();
    let schema = Schema {
        fields,
        metadata: HashMap::new(),
        fields_index: Some(fields_index),
    };


    let mut memory_consume = 0;
    let mut compressed_consume = 0;

    // term_values.bin
    if let Ok(f) = File::open(path.join(PathBuf::from("term_values.bin"))) {
        let reader = BufReader::new(f);
        values = bincode::deserialize_from(reader).unwrap();
    } else {
        return None;
    }
    info!("start build TermDict");
    let values: Vec<TermMeta> = values
        .into_iter()
        .map(|v| {
            compressed_consume += v.rle_usage();
         
            let valid_bitmap = v.valid_bitmap
                .into_iter()
                .map(|v| {
                    Arc::new(RoaringBitmap::from_sorted_iter(v.into_iter()).unwrap())
                })
                .collect();
            let termmeta = TermMeta {
                valid_bitmap: Arc::new(valid_bitmap),
                index: v.index,
                nums: v.nums,
                selectivity: v.selectivity,
            };
            memory_consume += termmeta.memory_consumption();
            termmeta
        })
        .collect();
    info!("term len: {:}", values.len());
    info!("term index: {:}", memory_consume);
    info!("compreed index: {:}", compressed_consume);
    let bitmap_consumption: usize = values.iter().map(|v| v.valid_bitmap.as_ref()[0].memory_consumption()).sum();
    info!("valid bitmap consumption: {:}", bitmap_consumption);
    #[cfg(all(feature = "trie_idx", not(feature = "hash_idx")))]
    let term_idx = Arc::new(TermIdx::new(keys, values, 20));
    #[cfg(feature = "hash_idx")]
    let term_idx = Arc::new(TermIdx { term_map: HashMap::from_iter(keys.into_iter().zip(values.into_iter())) });

    info!("finish deserializing index");
    if fs::metadata(path.join(PathBuf::from("dump.mmap"))).is_ok() {
        let mmap = Arc::new(unsafe { MmapOptions::new().map(&File::open(path.join(PathBuf::from("dump.mmap"))).unwrap()).unwrap() });
        return Some(MmapTable::with_mmap(mmap, Arc::new(schema), term_idx, &batch_range, 1).unwrap());
    }

    // posting_batch.bin
    if let Ok(f) = File::open(path.join(PathBuf::from("posting_batch.bin"))) {
        let reader = BufReader::new(f);
        posting_batch = bincode::deserialize_from(reader).unwrap();
    } else {
        return None;
    }
    assert_eq!(posting_batch.len(), 1, "Only support 1 partition");
    let posting_segment = posting_batch.pop().unwrap().build_mmap_segment(0).unwrap();

    Some(MmapTable::new(
        &path.join(PathBuf::from("dump.mmap")), Arc::new(schema), term_idx, posting_segment, &batch_range, 1).unwrap())
}
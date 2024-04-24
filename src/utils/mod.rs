pub mod json;
pub mod err;
pub mod chunk;
pub mod array;
pub mod builder;
pub mod avx512;
pub mod vec;

pub use json::{to_hashmap, parse_wiki_file};
pub use err::{Result, FastErr};
pub use chunk::UnalignedBitChunk;
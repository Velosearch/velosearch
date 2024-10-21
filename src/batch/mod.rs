mod posting_batch;

pub use posting_batch::{PostingBatch, BatchRange, PostingBatchBuilder, TermMetaBuilder, BatchFreqs, Freqs, merge_segments};

#[cfg(test)]
mod tests {
    use std::arch::x86_64::_pext_u64;

    #[test]
    fn test_pdep() {
        let res = unsafe { _pext_u64(0b0011_0100, 0b1011_0110) };
        println!("{:b}", res);
    }
}
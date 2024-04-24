use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Iterates over an arbitrarily aligned byte buffer
///
/// Yields an iterator of aligned u64, along with the leading and trailing
/// u64 necessary to align the buffer to a 8-byte boundary
///
/// This is unlike [`BitChunkIterator`] which only exposes a trailing u64,
/// and consequently has to perform more work for each read
#[derive(Debug)]
pub struct UnalignedBitChunk<'a> {
    lead_padding: usize,
    trailing_padding: usize,

    prefix: Option<u64>,
    chunks: &'a [u64],
    suffix: Option<u64>,
}

impl<'a> UnalignedBitChunk<'a> {
    /// Create a from a byte array, and and an offset and length in bits
    pub fn new(buffer: &'a [u8], offset: usize, len: usize) -> Self {
        if len == 0 {
            return Self {
                lead_padding: 0,
                trailing_padding: 0,
                prefix: None,
                chunks: &[],
                suffix: None,
            };
        }

        let byte_offset = offset / 8;
        let offset_padding = offset % 8;

        let bytes_len = (len + offset_padding + 7) / 8;
        let buffer = &buffer[byte_offset..byte_offset + bytes_len];

        let prefix_mask = compute_prefix_mask(offset_padding);

        // If less than 8 bytes, read into prefix
        if buffer.len() <= 8 {
            let (suffix_mask, trailing_padding) =
                compute_suffix_mask(len, offset_padding);
            let prefix = read_u64(buffer) & suffix_mask & prefix_mask;

            return Self {
                lead_padding: offset_padding,
                trailing_padding,
                prefix: Some(prefix),
                chunks: &[],
                suffix: None,
            };
        }

        // If less than 16 bytes, read into prefix and suffix
        if buffer.len() <= 16 {
            let (suffix_mask, trailing_padding) =
                compute_suffix_mask(len, offset_padding);
            let prefix = read_u64(&buffer[..8]) & prefix_mask;
            let suffix = read_u64(&buffer[8..]) & suffix_mask;

            return Self {
                lead_padding: offset_padding,
                trailing_padding,
                prefix: Some(prefix),
                chunks: &[],
                suffix: Some(suffix),
            };
        }

        // Read into prefix and suffix as needed
        let (prefix, mut chunks, suffix) = unsafe { buffer.align_to::<u64>() };
        assert!(
            prefix.len() < 8 && suffix.len() < 8,
            "align_to did not return largest possible aligned slice"
        );

        let (alignment_padding, prefix) = match (offset_padding, prefix.is_empty()) {
            (0, true) => (0, None),
            (_, true) => {
                let prefix = chunks[0] & prefix_mask;
                chunks = &chunks[1..];
                (0, Some(prefix))
            }
            (_, false) => {
                let alignment_padding = (8 - prefix.len()) * 8;

                let prefix = (read_u64(prefix) & prefix_mask) << alignment_padding;
                (alignment_padding, Some(prefix))
            }
        };

        let lead_padding = offset_padding + alignment_padding;
        let (suffix_mask, trailing_padding) = compute_suffix_mask(len, lead_padding);

        let suffix = match (trailing_padding, suffix.is_empty()) {
            (0, _) => None,
            (_, true) => {
                let suffix = chunks[chunks.len() - 1] & suffix_mask;
                chunks = &chunks[..chunks.len() - 1];
                Some(suffix)
            }
            (_, false) => Some(read_u64(suffix) & suffix_mask),
        };

        Self {
            lead_padding,
            trailing_padding,
            prefix,
            chunks,
            suffix,
        }
    }

    pub fn lead_padding(&self) -> usize {
        self.lead_padding
    }

    pub fn trailing_padding(&self) -> usize {
        self.trailing_padding
    }

    pub fn prefix(&self) -> Option<u64> {
        self.prefix
    }

    pub fn suffix(&self) -> Option<u64> {
        self.suffix
    }

    pub fn chunks(&self) -> &'a [u64] {
        self.chunks
    }

    pub fn iter(&self) -> UnalignedBitChunkIterator<'a> {
        self.prefix
            .into_iter()
            .chain(self.chunks.iter().cloned())
            .chain(self.suffix.into_iter())
    }

    pub fn into_par_iter(self) -> RanyonParIter<'a> {
        self.prefix
            .into_par_iter()
            .chain(self.chunks.to_owned().into_par_iter())
            .chain(self.suffix.into_par_iter())
    }

    /// Counts the number of ones
    pub fn count_ones(&self) -> usize {
        self.iter().map(|x| x.count_ones() as usize).sum()
    }
}

pub type UnalignedBitChunkIterator<'a> = std::iter::Chain<
    std::iter::Chain<
        std::option::IntoIter<u64>,
        std::iter::Cloned<std::slice::Iter<'a, u64>>,
    >,
    std::option::IntoIter<u64>,
>;

pub type RanyonParIter<'a> = rayon::iter::Chain<
    rayon::iter::Chain<
        rayon::option::IntoIter<u64>,
        rayon::vec::IntoIter<u64>
    >,
    rayon::option::IntoIter<u64>
>;
#[inline]
fn read_u64(input: &[u8]) -> u64 {
    let len = input.len().min(8);
    let mut buf = [0_u8; 8];
    buf[..len].copy_from_slice(input);
    u64::from_le_bytes(buf)
}

#[inline]
fn compute_prefix_mask(lead_padding: usize) -> u64 {
    !((1 << lead_padding) - 1)
}

#[inline]
fn compute_suffix_mask(len: usize, lead_padding: usize) -> (u64, usize) {
    let trailing_bits = (len + lead_padding) % 64;

    if trailing_bits == 0 {
        return (u64::MAX, 0);
    }

    let trailing_padding = 64 - trailing_bits;
    let suffix_mask = (1 << trailing_bits) - 1;
    (suffix_mask, trailing_padding)
}
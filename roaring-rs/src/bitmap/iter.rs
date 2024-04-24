use alloc::vec::{self, Vec};
use core::iter::{self, FromIterator};
use core::slice;

use super::container::Container;
use crate::bitmap::util;
use crate::{NonSortedIntegers, RoaringBitmap};

/// An iterator for `RoaringBitmap`.
pub struct Iter<'a> {
    inner: iter::Flatten<slice::Iter<'a, Container>>,
    size_hint: u64,
}

/// An iterator for optimizing rank.
pub struct RankIter<'a> {
    pub(crate) inner: &'a RoaringBitmap,
    /// accumulated rank
    acc_rank: usize,
    last_value: Option<u32>,
    last_key: u16,
    last_index: usize,
    last_layer: usize,
}

impl<'a> RankIter<'a> {
    pub(crate) fn new(bitmap: &'a RoaringBitmap) -> Self {
        Self {
            inner: bitmap,
            acc_rank: 0,
            last_value: None,
            last_key: 0,
            last_index: usize::MAX,
            last_layer: 0,
        }
    }

    /// Give rank of a sequence sorted value.
    pub fn rank(&mut self, value: u32) -> usize {
        // assert the rank keys are sorted.
        debug_assert!(
            match self.last_value {
                Some(v) => value > v,
                None => true,
            }
        );
        let (key, index) = util::split(value);
        let mut base = self.last_index;
        let rank = if key == self.last_key && self.last_value.is_some() {
            unsafe {
                self.inner.containers
                    .get_unchecked(self.last_layer)
                    .rank_base(&mut base, index) as usize
                + self.acc_rank
            }
        } else {
            self.last_key = key;
            match self.inner.containers.binary_search_by_key(&key, |c| c.key) {
                Ok(i) => {
                    let rank = unsafe { self.inner.containers.get_unchecked(i) }.rank_base(&mut base, index);
                    self.last_layer = i;
                    self.inner.containers[..i].iter().rev().map(|c| c.len() as usize).sum::<usize>() + rank
                }
                Err(_i) => {
                    unreachable!()
                }
            }
        };
        self.last_value = Some(value);
        self.last_index = base;
        self.acc_rank = rank;
        rank
    }
}

/// An iterator for `RoaringBitmap`.
pub struct IntoIter {
    inner: iter::Flatten<vec::IntoIter<Container>>,
    size_hint: u64,
}

impl Iter<'_> {
    fn new(containers: &[Container]) -> Iter {
        let size_hint = containers.iter().map(|c| c.len()).sum();
        Iter { inner: containers.iter().flatten(), size_hint }
    }
}

impl IntoIter {
    fn new(containers: Vec<Container>) -> IntoIter {
        let size_hint = containers.iter().map(|c| c.len()).sum();
        IntoIter { inner: containers.into_iter().flatten(), size_hint }
    }
}

impl Iterator for Iter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        self.size_hint = self.size_hint.saturating_sub(1);
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.size_hint < usize::MAX as u64 {
            (self.size_hint as usize, Some(self.size_hint as usize))
        } else {
            (usize::MAX, None)
        }
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.size_hint = self.size_hint.saturating_sub(1);
        self.inner.next_back()
    }
}

#[cfg(target_pointer_width = "64")]
impl ExactSizeIterator for Iter<'_> {
    fn len(&self) -> usize {
        self.size_hint as usize
    }
}

impl Iterator for IntoIter {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        self.size_hint = self.size_hint.saturating_sub(1);
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.size_hint < usize::MAX as u64 {
            (self.size_hint as usize, Some(self.size_hint as usize))
        } else {
            (usize::MAX, None)
        }
    }
}

impl DoubleEndedIterator for IntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.size_hint = self.size_hint.saturating_sub(1);
        self.inner.next_back()
    }
}

#[cfg(target_pointer_width = "64")]
impl ExactSizeIterator for IntoIter {
    fn len(&self) -> usize {
        self.size_hint as usize
    }
}

impl RoaringBitmap {
    /// Iterator over each value stored in the RoaringBitmap, guarantees values are ordered by value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    /// use core::iter::FromIterator;
    ///
    /// let bitmap = (1..3).collect::<RoaringBitmap>();
    /// let mut iter = bitmap.iter();
    ///
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter {
        Iter::new(&self.containers)
    }
}

impl<'a> IntoIterator for &'a RoaringBitmap {
    type Item = u32;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl IntoIterator for RoaringBitmap {
    type Item = u32;
    type IntoIter = IntoIter;

    fn into_iter(self) -> IntoIter {
        IntoIter::new(self.containers)
    }
}

impl<const N: usize> From<[u32; N]> for RoaringBitmap {
    fn from(arr: [u32; N]) -> Self {
        RoaringBitmap::from_iter(arr)
    }
}

impl FromIterator<u32> for RoaringBitmap {
    fn from_iter<I: IntoIterator<Item = u32>>(iterator: I) -> RoaringBitmap {
        let mut rb = RoaringBitmap::new();
        rb.extend(iterator);
        rb
    }
}

impl<'a> FromIterator<&'a u32> for RoaringBitmap {
    fn from_iter<I: IntoIterator<Item = &'a u32>>(iterator: I) -> RoaringBitmap {
        let mut rb = RoaringBitmap::new();
        rb.extend(iterator);
        rb
    }
}

impl Extend<u32> for RoaringBitmap {
    fn extend<I: IntoIterator<Item = u32>>(&mut self, iterator: I) {
        for value in iterator {
            self.insert(value);
        }
    }
}

impl<'a> Extend<&'a u32> for RoaringBitmap {
    fn extend<I: IntoIterator<Item = &'a u32>>(&mut self, iterator: I) {
        for value in iterator {
            self.insert(*value);
        }
    }
}

impl RoaringBitmap {
    /// Create the set from a sorted iterator. Values must be sorted and deduplicated.
    ///
    /// The values of the iterator must be ordered and strictly greater than the greatest value
    /// in the set. If a value in the iterator doesn't satisfy this requirement, it is not added
    /// and the append operation is stopped.
    ///
    /// Returns `Ok` with the requested `RoaringBitmap`, `Err` with the number of elements
    /// that were correctly appended before failure.
    ///
    /// # Example: Create a set from an ordered list of integers.
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb = RoaringBitmap::from_sorted_iter(0..10).unwrap();
    ///
    /// assert!(rb.iter().eq(0..10));
    /// ```
    ///
    /// # Example: Try to create a set from a non-ordered list of integers.
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let integers = 0..10u32;
    /// let error = RoaringBitmap::from_sorted_iter(integers.rev()).unwrap_err();
    ///
    /// assert_eq!(error.valid_until(), 1);
    /// ```
    pub fn from_sorted_iter<I: IntoIterator<Item = u32>>(
        iterator: I,
    ) -> Result<RoaringBitmap, NonSortedIntegers> {
        let mut rb = RoaringBitmap::new();
        rb.append(iterator).map(|_| rb)
    }

    /// Extend the set with a sorted iterator.
    ///
    /// The values of the iterator must be ordered and strictly greater than the greatest value
    /// in the set. If a value in the iterator doesn't satisfy this requirement, it is not added
    /// and the append operation is stopped.
    ///
    /// Returns `Ok` with the number of elements appended to the set, `Err` with
    /// the number of elements we effectively appended before an error occurred.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use roaring::RoaringBitmap;
    ///
    /// let mut rb = RoaringBitmap::new();
    /// assert_eq!(rb.append(0..10), Ok(10));
    ///
    /// assert!(rb.iter().eq(0..10));
    /// ```
    pub fn append<I: IntoIterator<Item = u32>>(
        &mut self,
        iterator: I,
    ) -> Result<u64, NonSortedIntegers> {
        // Name shadowed to prevent accidentally referencing the param
        let mut iterator = iterator.into_iter();

        let mut prev = match (iterator.next(), self.max()) {
            (None, _) => return Ok(0),
            (Some(first), Some(max)) if first <= max => {
                return Err(NonSortedIntegers { valid_until: 0 })
            }
            (Some(first), _) => first,
        };

        // It is now guaranteed that so long as the values of the iterator are
        // monotonically increasing they must also be the greatest in the set.

        self.push_unchecked(prev);

        let mut count = 1;

        for value in iterator {
            if value <= prev {
                return Err(NonSortedIntegers { valid_until: count });
            } else {
                self.push_unchecked(value);
                prev = value;
                count += 1;
            }
        }

        Ok(count)
    }
}

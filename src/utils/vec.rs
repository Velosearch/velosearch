/// # Safety
///
/// * `ptr` must be [valid] for writes of `size_of::<T>()` bytes.
/// * The region of memory beginning at `val` with a size of `size_of::<T>()`
///   bytes must *not* overlap with the region of memory beginning at `ptr`
///   with the same size.
#[inline(always)]
pub unsafe fn store_advance<T>(val: &T, ptr: &mut *mut u8) {
    unsafe {
        std::ptr::copy_nonoverlapping(val as *const T as *const u8, *ptr, std::mem::size_of::<T>());
        *ptr = ptr.add(std::mem::size_of::<T>())
    }
}

/// # Safety
///
/// * `ptr` must be [valid] for writes.
/// * `ptr` must be properly aligned.
#[inline(always)]
pub unsafe fn store_advance_aligned<T>(val: T, ptr: &mut *mut T) {
    unsafe {
        std::ptr::write(*ptr, val);
        *ptr = ptr.add(1)
    }
}

/// # Safety
///
/// * `src` must be [valid] for writes of `count * size_of::<T>()` bytes.
/// * `ptr` must be [valid] for writes of `count * size_of::<T>()` bytes.
/// * Both `src` and `dst` must be properly aligned.
/// * The region of memory beginning at `val` with a size of `count * size_of::<T>()`
///   bytes must *not* overlap with the region of memory beginning at `ptr` with the
///   same size.
#[inline(always)]
pub unsafe fn copy_advance_aligned<T>(src: *const T, ptr: &mut *mut T, count: usize) {
    unsafe {
        std::ptr::copy_nonoverlapping(src, *ptr, count);
        *ptr = ptr.add(count);
    }
}

/// # Safety
///
/// * `(ptr as usize - vec.as_ptr() as usize) / std::mem::size_of::<T>()` must be
///    less than or equal to the capacity of Vec.
#[inline(always)]
pub unsafe fn set_vec_len_by_ptr<T>(vec: &mut Vec<T>, ptr: *const T) {
    unsafe {
        vec.set_len((ptr as usize - vec.as_ptr() as usize) / std::mem::size_of::<T>());
    }
}

//! Proven-bounds access helpers for the `pretty-safe` feature.
//!
//! Each function uses `get_unchecked` when `pretty-safe` is enabled,
//! falling back to normal indexing otherwise. All include `debug_assert!`
//! so test/debug builds catch any misuse.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use core::ops::Range;

#[inline(always)]
pub(crate) fn idx<T>(slice: &[T], i: usize) -> &T {
    debug_assert!(i < slice.len(), "proven::idx: {i} >= {}", slice.len());
    #[cfg(feature = "pretty-safe")]
    {
        unsafe { slice.get_unchecked(i) }
    }
    #[cfg(not(feature = "pretty-safe"))]
    {
        &slice[i]
    }
}

#[inline(always)]
pub(crate) fn idx_mut<T>(slice: &mut [T], i: usize) -> &mut T {
    debug_assert!(i < slice.len(), "proven::idx_mut: {i} >= {}", slice.len());
    #[cfg(feature = "pretty-safe")]
    {
        unsafe { slice.get_unchecked_mut(i) }
    }
    #[cfg(not(feature = "pretty-safe"))]
    {
        &mut slice[i]
    }
}

#[inline(always)]
pub(crate) fn sub<T>(slice: &[T], range: Range<usize>) -> &[T] {
    debug_assert!(
        range.end <= slice.len(),
        "proven::sub: {}..{} out of {}",
        range.start,
        range.end,
        slice.len()
    );
    #[cfg(feature = "pretty-safe")]
    {
        unsafe { slice.get_unchecked(range) }
    }
    #[cfg(not(feature = "pretty-safe"))]
    {
        &slice[range]
    }
}

#[inline(always)]
pub(crate) fn alloc_output<T: Copy + Default>(len: usize) -> Vec<T> {
    // vec![T::default(); len] is optimized to calloc for zero-initialized types
    // (u8, i16, f32), so there is no performance benefit to the unsafe set_len trick.
    vec![T::default(); len]
}

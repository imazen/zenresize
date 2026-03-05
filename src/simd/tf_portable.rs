//! Portable transfer function row processor using magetypes f32x4.
//!
//! Generic over any token implementing `F32x4Convert` (NEON, WASM128, etc.).
//! Transfer function implementations now live in `linear-srgb`; this module
//! provides only the row-level batch processor with alpha handling.

use magetypes::simd::backends::F32x4Convert;
use magetypes::simd::generic::f32x4;

/// Apply a 4-wide transfer function to a row of f32 values in-place.
///
/// For RGBA (4ch, has_alpha): processes 4 floats (1 pixel) at a time,
/// restoring alpha channel. For no-alpha: processes 4 floats flat.
#[inline(always)]
pub(super) fn tf_row_inplace<T: F32x4Convert>(
    t: T,
    row: &mut [f32],
    channels: usize,
    has_alpha: bool,
    tf_x4: impl Fn(T, f32x4<T>) -> f32x4<T>,
    tf_scalar: fn(f32) -> f32,
) {
    if has_alpha && channels == 4 {
        // RGBA: process 4 floats (1 pixel) at a time, restore alpha
        let (chunks, _tail) = f32x4::<T>::partition_slice_mut(t, row);
        for chunk in chunks.iter_mut() {
            let alpha = chunk[3];
            let v = f32x4::load(t, chunk);
            let converted = tf_x4(t, v);
            converted.store(chunk);
            chunk[3] = alpha;
        }
        // No tail — RGBA row length is always a multiple of 4
    } else if has_alpha && channels >= 2 {
        // Non-4ch with alpha: per-pixel, skip last channel
        for pixel in row.chunks_exact_mut(channels) {
            for v in &mut pixel[..channels - 1] {
                *v = tf_scalar(*v);
            }
        }
    } else {
        // No alpha: process all values flat, 4 at a time
        let (chunks, tail) = f32x4::<T>::partition_slice_mut(t, row);
        for chunk in chunks.iter_mut() {
            let v = f32x4::load(t, chunk);
            let converted = tf_x4(t, v);
            converted.store(chunk);
        }
        for v in tail.iter_mut() {
            *v = tf_scalar(*v);
        }
    }
}

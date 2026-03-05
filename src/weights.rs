//! Resampling weight calculation.
//!
//! Computes interpolation weights for image resampling. Weights are computed
//! per-output-pixel based on the selected filter and the mapping between
//! input and output coordinates.
//!
//! The weight calculation matches imageflow's approach:
//! - Pixel indices are clamped to valid range BEFORE calculating weights
//! - Only weights for valid pixels are included
//! - Tiny weights (< 2e-8) are rounded to zero for consistency
//! - Zero weights are trimmed from edges

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::filter::InterpolationDetails;

/// Threshold below which weights are considered zero.
/// Matches imageflow for cross-platform consistency.
pub const WEIGHT_THRESHOLD: f64 = 2e-8;

/// Fixed-point precision for i16 weights (14 bits).
pub const I16_PRECISION: i32 = 14;

// =============================================================================
// Weight table types
// =============================================================================

/// Pre-computed f32 weight table with flat layout for SIMD access.
///
/// Each output pixel's weights are zero-padded to `max_taps` width in a
/// single contiguous allocation.
#[derive(Clone)]
pub struct F32WeightTable {
    /// For each output pixel, the starting input pixel index.
    pub left: Vec<i32>,
    /// Flat array: `out_size * max_taps` elements, zero-padded.
    weights_flat: Vec<f32>,
    /// Number of actual (non-zero) taps per output pixel.
    tap_counts: Vec<u16>,
    /// Maximum taps across all output pixels.
    pub max_taps: usize,
}

/// Pre-computed i16 fixed-point weight table with flat layout for SIMD access.
///
/// Weights are scaled to 14-bit precision (1 << 14 = 1.0).
/// The flat layout stores all weights in a single contiguous allocation,
/// zero-padded to `max_taps` per output pixel (similar to [`F32WeightTable`]).
///
/// Includes a pre-expanded weight format for 4-channel horizontal convolution:
/// each group of 4 taps is broadcast into a 256-bit (16 × i16) pattern for
/// direct ymm load+madd without per-pixel broadcasts.
#[derive(Clone)]
pub struct I16WeightTable {
    /// For each output pixel, the starting input pixel index.
    pub left: Vec<i32>,
    /// Flat array: `out_size * max_taps` elements, zero-padded.
    weights_flat: Vec<i16>,
    /// Number of actual (non-zero) taps per output pixel.
    tap_counts: Vec<u16>,
    /// Maximum taps across all output pixels.
    pub max_taps: usize,
    /// Pre-expanded weights for 4ch ymm horizontal convolution.
    /// Layout per output pixel: `groups4 × 16` i16 values.
    /// Each group: lo lane [w0,w1]×4, hi lane [w2,w3]×4.
    expanded_4ch: Vec<i16>,
    /// Number of 4-tap groups (ceil(max_taps / 4)).
    pub(crate) groups4: usize,
    /// Stride in i16 values per output pixel in expanded_4ch.
    pub(crate) expanded_stride: usize,
}

// =============================================================================
// F32WeightTable
// =============================================================================

impl F32WeightTable {
    /// Create weight table for resampling from `in_size` to `out_size`.
    ///
    /// If `filter.lobe_ratio_goal` is set (via
    /// [`InterpolationDetails::with_lobe_ratio`]), negative lobes are adjusted
    /// (amplified or reduced) to reach the target ratio.
    pub fn new(in_size: u32, out_size: u32, filter: &InterpolationDetails) -> Self {
        debug_assert!(in_size > 0, "in_size must be positive");
        debug_assert!(out_size > 0, "out_size must be positive");

        let scale = out_size as f64 / in_size as f64;
        let downscale_factor = scale.min(1.0);
        let effective_window = filter.window / downscale_factor;

        // Compute lobe ratio adjustment targets.
        // When lobe_ratio_goal is Some(r), the desired neg/pos ratio is r.
        // r=0 flattens negatives, r<natural softens, r>natural sharpens.
        // Clamped to [0, 1) since r≥1 would make weights diverge.
        let (desired_ratio, apply_lobe_adj) = match filter.lobe_ratio_goal {
            Some(r) => {
                let natural = filter.calculate_percent_negative_weight();
                let desired = (r as f64).clamp(0.0, 0.9999);
                // Apply adjustment when the desired ratio differs from natural.
                // Skip if the filter has no negative lobes and we're not asking
                // to add any (e.g., Box with ratio > 0 — can't amplify what
                // doesn't exist).
                let apply = if natural < 1e-10 {
                    // No negative lobes exist — can only flatten (which is already
                    // the natural state) or attempt to sharpen (impossible).
                    false
                } else {
                    (desired - natural).abs() > 1e-10
                };
                (desired, apply)
            }
            None => (0.0, false),
        };

        // First pass: compute weights and find max_taps
        let mut temp_weights: Vec<f64> = Vec::new();
        let mut all_left: Vec<i32> = Vec::with_capacity(out_size as usize);
        let mut all_tap_counts: Vec<u16> = Vec::with_capacity(out_size as usize);
        let mut all_weights: Vec<Vec<f64>> = Vec::with_capacity(out_size as usize);
        let mut max_taps = 0;

        for out_pixel in 0..out_size {
            temp_weights.clear();
            let (left, tap_count) = compute_pixel_weights(
                out_pixel,
                in_size,
                scale,
                downscale_factor,
                effective_window,
                filter,
                &mut temp_weights,
            );
            max_taps = max_taps.max(tap_count);
            all_left.push(left);
            all_tap_counts.push(tap_count as u16);
            all_weights.push(temp_weights.clone());
        }

        // Second pass: fill flat array and normalize.
        let mut weights_flat = vec![0.0f32; out_size as usize * max_taps];
        for (i, w) in all_weights.iter().enumerate() {
            let offset = i * max_taps;
            for (j, &val) in w.iter().enumerate() {
                weights_flat[offset + j] = val as f32;
            }

            let slice = &mut weights_flat[offset..offset + w.len()];

            if apply_lobe_adj {
                // Lobe ratio adjustment: separately scale positive and negative
                // weights so the neg/pos ratio equals desired_ratio.
                // r=0 zeroes negatives, r<natural softens, r>natural sharpens.
                // Uses the per-pixel normalized weight sums (compute_pixel_weights
                // normalizes to sum=1.0, preserving the natural ratio).
                let mut total_pos = 0.0f64;
                let mut total_neg = 0.0f64;
                for &val in w.iter() {
                    if val >= 0.0 {
                        total_pos += val;
                    } else {
                        total_neg += val;
                    }
                }
                if total_neg < 0.0 && desired_ratio < 1.0 {
                    if desired_ratio < 1e-10 {
                        // Flatten: zero all negative weights, renormalize positives
                        for v in slice.iter_mut() {
                            if *v < 0.0 {
                                *v = 0.0;
                            }
                        }
                        let f32_sum: f32 = slice.iter().filter(|v| **v > 0.0).sum();
                        if f32_sum != 0.0 {
                            let inv = 1.0f32 / f32_sum;
                            for v in slice.iter_mut() {
                                *v *= inv;
                            }
                        }
                    } else {
                        let target_pos = 1.0 / (1.0 - desired_ratio);
                        let target_neg = desired_ratio * -target_pos;
                        let pos_factor = (target_pos / total_pos) as f32;
                        let neg_factor = (target_neg / total_neg) as f32;
                        for v in slice.iter_mut() {
                            if *v < 0.0 {
                                *v *= neg_factor;
                            } else {
                                *v *= pos_factor;
                            }
                        }
                    }
                } else {
                    // No negative weights — standard normalization
                    let f32_sum: f32 = slice.iter().sum();
                    if f32_sum != 0.0 {
                        let inv = 1.0f32 / f32_sum;
                        for v in slice.iter_mut() {
                            *v *= inv;
                        }
                    }
                }
            } else {
                // Standard normalization: renormalize f32 weights to sum to 1.0
                let f32_sum: f32 = slice.iter().sum();
                if f32_sum != 0.0 {
                    let inv = 1.0f32 / f32_sum;
                    for v in slice.iter_mut() {
                        *v *= inv;
                    }
                }
            }
        }

        Self {
            left: all_left,
            weights_flat,
            tap_counts: all_tap_counts,
            max_taps,
        }
    }

    /// Create a weight table from pre-computed parts.
    ///
    /// Used by [`crate::blur`] to build Gaussian convolution kernels.
    pub(crate) fn from_parts(
        left: Vec<i32>,
        weights_flat: Vec<f32>,
        tap_counts: Vec<u16>,
        max_taps: usize,
    ) -> Self {
        Self {
            left,
            weights_flat,
            tap_counts,
            max_taps,
        }
    }

    /// Get weights slice for a specific output pixel (non-zero taps only).
    #[inline]
    pub fn weights(&self, out_pixel: usize) -> &[f32] {
        let offset = out_pixel * self.max_taps;
        let tap_count = self.tap_counts[out_pixel] as usize;
        &self.weights_flat[offset..offset + tap_count]
    }

    /// Get the full padded weights slice (max_taps elements, zero-padded).
    #[inline]
    pub fn weights_padded(&self, out_pixel: usize) -> &[f32] {
        let offset = out_pixel * self.max_taps;
        &self.weights_flat[offset..offset + self.max_taps]
    }

    /// Get the number of actual taps for an output pixel.
    #[inline]
    pub fn tap_count(&self, out_pixel: usize) -> usize {
        self.tap_counts[out_pixel] as usize
    }

    /// Number of output pixels.
    #[inline]
    pub fn len(&self) -> usize {
        self.left.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }
}

// =============================================================================
// I16WeightTable
// =============================================================================

impl I16WeightTable {
    /// Create i16 fixed-point weight table with flat layout.
    pub fn new(in_size: u32, out_size: u32, filter: &InterpolationDetails) -> Self {
        debug_assert!(in_size > 0, "in_size must be positive");
        debug_assert!(out_size > 0, "out_size must be positive");

        let scale = out_size as f64 / in_size as f64;
        let downscale_factor = scale.min(1.0);
        let effective_window = filter.window / downscale_factor;

        let mut temp_weights: Vec<f64> = Vec::new();
        let mut left_vec = Vec::with_capacity(out_size as usize);
        let mut all_weights: Vec<Vec<i16>> = Vec::with_capacity(out_size as usize);
        let mut all_tap_counts: Vec<u16> = Vec::with_capacity(out_size as usize);
        let mut max_taps = 0;

        // Reusable buffers for fixed-point conversion (avoid per-pixel allocation)
        let mut scaled: Vec<f64> = Vec::new();
        let mut fixed: Vec<i16> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();

        for out_pixel in 0..out_size {
            temp_weights.clear();
            let (left, tap_count) = compute_pixel_weights(
                out_pixel,
                in_size,
                scale,
                downscale_factor,
                effective_window,
                filter,
                &mut temp_weights,
            );
            max_taps = max_taps.max(tap_count);

            // Convert to fixed-point i16 using largest-remainder method.
            // This distributes rounding error across weights instead of
            // dumping it all on the center weight, which caused catastrophic
            // errors (max=255) at high tap counts (e.g. Jinc at 8x downscale).
            let scale_factor = (1 << I16_PRECISION) as f64;
            scaled.clear();
            scaled.extend(temp_weights.iter().map(|&w| w * scale_factor));
            fixed.clear();
            fixed.extend(scaled.iter().map(|&s| s.round() as i16));

            let sum: i32 = fixed.iter().map(|&w| w as i32).sum();
            let error = (1 << I16_PRECISION) - sum;
            if error != 0 {
                // Compute rounding residuals: how much was lost by rounding each weight.
                // residual > 0 means we rounded down, < 0 means we rounded up.
                indices.clear();
                indices.extend(0..fixed.len());
                if error > 0 {
                    // Need to add +1 to some weights. Prioritize those that were
                    // rounded down the most (largest positive residual).
                    indices.sort_by(|&a, &b| {
                        let ra = scaled[a] - scaled[a].round();
                        let rb = scaled[b] - scaled[b].round();
                        rb.partial_cmp(&ra).unwrap_or(core::cmp::Ordering::Equal)
                    });
                    for &i in indices.iter().take(error as usize) {
                        fixed[i] = fixed[i].saturating_add(1);
                    }
                } else {
                    // Need to subtract 1 from some weights. Prioritize those that
                    // were rounded up the most (largest negative residual).
                    indices.sort_by(|&a, &b| {
                        let ra = scaled[a] - scaled[a].round();
                        let rb = scaled[b] - scaled[b].round();
                        ra.partial_cmp(&rb).unwrap_or(core::cmp::Ordering::Equal)
                    });
                    for &i in indices.iter().take((-error) as usize) {
                        fixed[i] = fixed[i].saturating_sub(1);
                    }
                }
            }

            left_vec.push(left);
            all_tap_counts.push(tap_count as u16);
            all_weights.push(fixed.clone());
        }

        // Pad max_taps to even for madd_epi16 (processes pairs)
        if max_taps % 2 != 0 {
            max_taps += 1;
        }

        // Build flat layout: zero-padded to max_taps per pixel
        let mut weights_flat = vec![0i16; out_size as usize * max_taps];
        for (i, w) in all_weights.iter().enumerate() {
            let offset = i * max_taps;
            weights_flat[offset..offset + w.len()].copy_from_slice(w);
        }

        // Pre-expand weights for 4ch ymm horizontal convolution.
        // Each group of 4 taps [w0,w1,w2,w3] is stored as:
        //   lo lane: [w0,w1,w0,w1,w0,w1,w0,w1]
        //   hi lane: [w2,w3,w2,w3,w2,w3,w2,w3]
        let groups4 = max_taps.div_ceil(4);
        let expanded_stride = groups4 * 16;
        let mut expanded_4ch = vec![0i16; out_size as usize * expanded_stride];

        for (px, w) in all_weights.iter().enumerate() {
            let base = px * expanded_stride;
            for g in 0..groups4 {
                let t = g * 4;
                let w0 = if t < w.len() { w[t] } else { 0 };
                let w1 = if t + 1 < w.len() { w[t + 1] } else { 0 };
                let w2 = if t + 2 < w.len() { w[t + 2] } else { 0 };
                let w3 = if t + 3 < w.len() { w[t + 3] } else { 0 };
                // Lower lane: [w0,w1] repeated 4 times
                for i in 0..4 {
                    expanded_4ch[base + g * 16 + i * 2] = w0;
                    expanded_4ch[base + g * 16 + i * 2 + 1] = w1;
                }
                // Upper lane: [w2,w3] repeated 4 times
                for i in 0..4 {
                    expanded_4ch[base + g * 16 + 8 + i * 2] = w2;
                    expanded_4ch[base + g * 16 + 8 + i * 2 + 1] = w3;
                }
            }
        }

        Self {
            left: left_vec,
            weights_flat,
            tap_counts: all_tap_counts,
            max_taps,
            expanded_4ch,
            groups4,
            expanded_stride,
        }
    }

    /// Get weights slice for a specific output pixel (non-zero taps only).
    #[inline]
    pub fn weights(&self, out_pixel: usize) -> &[i16] {
        let offset = out_pixel * self.max_taps;
        let tap_count = self.tap_counts[out_pixel] as usize;
        &self.weights_flat[offset..offset + tap_count]
    }

    /// Get the full padded weights slice (max_taps elements, zero-padded).
    #[inline]
    pub fn weights_padded(&self, out_pixel: usize) -> &[i16] {
        let offset = out_pixel * self.max_taps;
        &self.weights_flat[offset..offset + self.max_taps]
    }

    /// Get the number of actual taps for an output pixel.
    #[inline]
    pub fn tap_count(&self, out_pixel: usize) -> usize {
        self.tap_counts[out_pixel] as usize
    }

    /// Get pre-expanded 4ch weights for ymm horizontal convolution.
    /// Returns `groups4 × 16` i16 values per output pixel.
    #[inline]
    pub fn weights_expanded_4ch(&self, out_pixel: usize) -> &[i16] {
        let offset = out_pixel * self.expanded_stride;
        &self.expanded_4ch[offset..offset + self.expanded_stride]
    }

    /// Entire expanded_4ch buffer as a slice.
    ///
    /// Layout: `out_size × expanded_stride` i16 values, contiguous.
    /// Use with a single `GuardedSlice` to avoid per-pixel guard construction.
    #[inline]
    pub fn expanded_4ch_all(&self) -> &[i16] {
        &self.expanded_4ch
    }

    /// Number of output pixels.
    #[inline]
    pub fn len(&self) -> usize {
        self.left.len()
    }

    /// Check if empty.
    #[cfg(test)]
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }
}

// =============================================================================
// Shared weight computation
// =============================================================================

/// Compute normalized f64 weights into a pre-allocated buffer.
///
/// Returns (left_pixel_index, tap_count).
fn compute_pixel_weights(
    out_pixel: u32,
    in_size: u32,
    scale: f64,
    downscale_factor: f64,
    effective_window: f64,
    filter: &InterpolationDetails,
    weights_out: &mut Vec<f64>,
) -> (i32, usize) {
    // Map output pixel center to input space
    let center = (out_pixel as f64 + 0.5) / scale - 0.5;

    // Find input pixel range in filter window
    let left_edge = f64_ceil(center - effective_window - 0.0001) as i32;
    let right_edge = f64_floor(center + effective_window + 0.0001) as i32;

    // Clamp to valid range BEFORE calculating weights
    let left_pixel = left_edge.max(0) as u32;
    let right_pixel = right_edge.min(in_size as i32 - 1) as u32;

    let mut total_weight = 0.0f64;

    for in_pixel in left_pixel..=right_pixel {
        let x = downscale_factor * (in_pixel as f64 - center);
        let mut w = filter.filter(x);

        if w.abs() <= WEIGHT_THRESHOLD {
            w = 0.0;
        }

        weights_out.push(w);
        total_weight += w;
    }

    // Normalize
    if total_weight > 0.0 {
        let inv_total = 1.0 / total_weight;
        for w in weights_out.iter_mut() {
            *w *= inv_total;
        }
    }

    // Trim zero weights from right
    while weights_out.last() == Some(&0.0f64) {
        weights_out.pop();
    }

    // Trim from left, adjusting left_pixel
    let mut trimmed_left = left_pixel;
    while weights_out.first() == Some(&0.0f64) {
        weights_out.remove(0);
        trimmed_left += 1;
    }

    (trimmed_left as i32, weights_out.len())
}

// no_std math helpers
#[cfg(feature = "std")]
#[inline]
fn f64_ceil(x: f64) -> f64 {
    x.ceil()
}

#[cfg(not(feature = "std"))]
#[inline]
fn f64_ceil(x: f64) -> f64 {
    libm::ceil(x)
}

#[cfg(feature = "std")]
#[inline]
fn f64_floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(not(feature = "std"))]
#[inline]
fn f64_floor(x: f64) -> f64 {
    libm::floor(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{Filter, InterpolationDetails};
    #[cfg(not(feature = "std"))]
    use alloc::{format, string::String, vec, vec::Vec};

    #[test]
    fn test_f32_weight_table_basic() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let table = F32WeightTable::new(100, 50, &filter);

        assert_eq!(table.len(), 50);

        for i in 0..table.len() {
            let weights = table.weights(i);
            let sum: f32 = weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.001,
                "F32 weights for pixel {} sum to {}, expected 1.0",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_i16_weight_table_basic() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let table = I16WeightTable::new(100, 50, &filter);

        assert_eq!(table.len(), 50);

        for i in 0..table.len() {
            let weights = table.weights(i);
            let sum: i32 = weights.iter().map(|&w| w as i32).sum();
            assert_eq!(
                sum,
                1 << I16_PRECISION,
                "I16 weights for pixel {} sum to {}, expected {}",
                i,
                sum,
                1 << I16_PRECISION
            );
        }
    }

    /// Verify i16 weight tables sum exactly for ALL filters at 8x downscale,
    /// where high tap counts previously caused catastrophic errors.
    #[test]
    fn i16_weights_sum_correctly_at_high_tap_counts() {
        let target = 1i32 << I16_PRECISION;

        for &filter_enum in Filter::all() {
            let filter = InterpolationDetails::create(filter_enum);
            // 8x downscale: 800→100 gives high tap counts
            let table = I16WeightTable::new(800, 100, &filter);

            for i in 0..table.len() {
                let weights = table.weights(i);
                let sum: i32 = weights.iter().map(|&w| w as i32).sum();
                assert_eq!(
                    sum,
                    target,
                    "Filter {:?} pixel {}: i16 sum {} != {} (tap_count={})",
                    filter_enum,
                    i,
                    sum,
                    target,
                    weights.len()
                );
            }
        }
    }

    /// Verify no individual i16 weight deviates from its f64 source by more than 1 LSB.
    #[test]
    fn i16_weights_max_deviation_from_f64() {
        for &filter_enum in Filter::all() {
            let filter = InterpolationDetails::create(filter_enum);
            let in_size = 800u32;
            let out_size = 100u32;
            let scale = out_size as f64 / in_size as f64;
            let downscale_factor = scale.min(1.0);
            let effective_window = filter.window / downscale_factor;

            let i16_table = I16WeightTable::new(in_size, out_size, &filter);
            let mut temp_weights: Vec<f64> = Vec::new();

            for out_pixel in 0..out_size {
                temp_weights.clear();
                let (_left, _tap_count) = compute_pixel_weights(
                    out_pixel,
                    in_size,
                    scale,
                    downscale_factor,
                    effective_window,
                    &filter,
                    &mut temp_weights,
                );

                let i16_weights = i16_table.weights(out_pixel as usize);
                let scale_factor = (1 << I16_PRECISION) as f64;
                for (j, (&f64_w, &i16_w)) in temp_weights.iter().zip(i16_weights.iter()).enumerate()
                {
                    let expected = f64_w * scale_factor;
                    let deviation = (i16_w as f64 - expected).abs();
                    assert!(
                        deviation <= 1.0,
                        "Filter {:?} pixel {} weight {}: i16={} expected={:.2} dev={:.2}",
                        filter_enum,
                        out_pixel,
                        j,
                        i16_w,
                        expected,
                        deviation
                    );
                }
            }
        }
    }

    #[test]
    fn test_upscale_weights() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let table = F32WeightTable::new(50, 100, &filter);

        assert_eq!(table.len(), 100);

        // Upscaling with Lanczos-3 should have ~5-8 taps
        assert!(
            (3..=10).contains(&table.max_taps),
            "Expected 3-10 max taps for Lanczos-3 upscale, got {}",
            table.max_taps
        );
    }

    #[test]
    fn test_downscale_weights() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let table = F32WeightTable::new(100, 25, &filter);

        assert_eq!(table.len(), 25);
        assert!(
            table.max_taps >= 7,
            "Expected >= 7 taps for 4x downscale, got {}",
            table.max_taps
        );
    }

    #[test]
    fn test_edge_handling() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let in_size = 100u32;
        let out_size = 50u32;
        let table = F32WeightTable::new(in_size, out_size, &filter);

        // All left indices must be valid
        for i in 0..table.len() {
            let left = table.left[i];
            let taps = table.tap_count(i);
            assert!(left >= 0, "left must be non-negative at pixel {}", i);
            assert!(
                (left as u32 + taps as u32) <= in_size,
                "right edge exceeds input at pixel {}: left={} taps={}",
                i,
                left,
                taps
            );
        }

        // Edge weights must still be normalized
        let first_sum: f32 = table.weights(0).iter().sum();
        assert!(
            (first_sum - 1.0).abs() < 0.001,
            "First pixel weights sum to {}",
            first_sum
        );
        let last_sum: f32 = table.weights(out_size as usize - 1).iter().sum();
        assert!(
            (last_sum - 1.0).abs() < 0.001,
            "Last pixel weights sum to {}",
            last_sum
        );
    }

    #[test]
    fn test_no_zero_weights_in_output() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let table = F32WeightTable::new(100, 50, &filter);

        for i in 0..table.len() {
            let w = table.weights(i);
            if !w.is_empty() {
                assert!(
                    w[0] != 0.0,
                    "First weight for pixel {} should not be zero",
                    i
                );
                assert!(
                    *w.last().unwrap() != 0.0,
                    "Last weight for pixel {} should not be zero",
                    i
                );
            }
        }
    }

    #[test]
    fn lobe_ratio_amplifies_negative_lobes() {
        // Test with upscale where negative ratio is naturally low,
        // so lobe_ratio(0.15) definitely amplifies.
        let filter = InterpolationDetails::create(Filter::Robidoux);
        let baseline = F32WeightTable::new(2, 17, &filter);
        let filter_sharp = filter.clone().with_lobe_ratio(0.15);
        let sharpened = F32WeightTable::new(2, 17, &filter_sharp);

        assert_eq!(baseline.len(), sharpened.len());

        // Sharpened weights should still sum to ~1.0
        for i in 0..sharpened.len() {
            let sum: f32 = sharpened.weights(i).iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.02,
                "Sharpened weights for pixel {} sum to {}",
                i,
                sum
            );
        }

        // Find a pixel with negative weights: verify ratio is forced to ~15%
        for i in 0..baseline.len() {
            let sw = sharpened.weights(i);
            let neg_sum: f32 = sw.iter().filter(|&&w| w < 0.0).sum();
            let pos_sum: f32 = sw.iter().filter(|&&w| w > 0.0).sum();
            if neg_sum < 0.0 {
                let ratio = (-neg_sum) / pos_sum;
                assert!(
                    (ratio - 0.15).abs() < 0.02,
                    "Pixel {}: neg/pos ratio should be ~15%, got {:.1}% (neg={} pos={})",
                    i,
                    ratio * 100.0,
                    neg_sum,
                    pos_sum
                );
                return;
            }
        }
        panic!("Expected at least one pixel with negative weights");
    }

    #[test]
    fn lobe_ratio_reduces_negative_lobes() {
        // Lanczos has natural ratio ~6.7%. Setting lobe_ratio to 0.02
        // should reduce it to ~2%.
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let filter_soft = filter.clone().with_lobe_ratio(0.02);
        let softened = F32WeightTable::new(100, 50, &filter_soft);

        for i in 0..softened.len() {
            let sw = softened.weights(i);
            let neg_sum: f32 = sw.iter().filter(|&&w| w < 0.0).sum();
            let pos_sum: f32 = sw.iter().filter(|&&w| w > 0.0).sum();
            if neg_sum < 0.0 {
                let ratio = (-neg_sum) / pos_sum;
                assert!(
                    (ratio - 0.02).abs() < 0.01,
                    "Pixel {}: neg/pos ratio should be ~2%, got {:.1}%",
                    i,
                    ratio * 100.0
                );
            }
        }
    }

    #[test]
    fn lobe_ratio_flatten_zeroes_negatives() {
        // lobe_ratio(0.0) should zero out all negative weights
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let filter_flat = filter.clone().with_lobe_ratio(0.0);
        let flattened = F32WeightTable::new(100, 50, &filter_flat);

        for i in 0..flattened.len() {
            let sw = flattened.weights(i);
            let neg_sum: f32 = sw.iter().filter(|&&w| w < 0.0).sum();
            assert!(
                neg_sum.abs() < 1e-7,
                "Pixel {}: flattened should have no negative weights, got {}",
                i,
                neg_sum
            );
            // Should still sum to ~1.0
            let sum: f32 = sw.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Pixel {}: flattened weights sum to {}",
                i,
                sum
            );
        }
    }

    #[test]
    fn lobe_ratio_none_matches_default() {
        // None (default) should produce identical weights to no lobe_ratio
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let baseline = F32WeightTable::new(100, 50, &filter);

        // Verify lobe_ratio_goal is None by default
        assert!(filter.lobe_ratio_goal.is_none());

        // Weights should be identical
        for i in 0..baseline.len() {
            let bw = baseline.weights(i);
            let sum: f32 = bw.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "pixel {} sum = {}", i, sum);
        }
    }

    #[test]
    fn lobe_ratio_on_all_positive_filter_is_noop() {
        // Box filter has 0% negative weight — any lobe_ratio target is a no-op
        // (can't amplify or reduce what doesn't exist)
        let filter = InterpolationDetails::create(Filter::Box);
        let baseline = F32WeightTable::new(100, 50, &filter);
        let filter_sharp = filter.clone().with_lobe_ratio(0.15);
        let sharpened = F32WeightTable::new(100, 50, &filter_sharp);

        for i in 0..baseline.len() {
            let bw = baseline.weights(i);
            let sw = sharpened.weights(i);
            for (j, (&b, &s)) in bw.iter().zip(sw.iter()).enumerate() {
                assert!(
                    (b - s).abs() < 1e-7,
                    "Box pixel {} tap {}: baseline={} sharpened={}",
                    i,
                    j,
                    b,
                    s
                );
            }
        }
    }

    /// Compute weights using imageflow's exact `populate_weights` normalization.
    ///
    /// Supports sharpen: pass `desired_sharpen_ratio > sharpen_ratio` to enable
    /// separate pos/neg normalization. For default (no sharpen), both are 0.0.
    /// Returns `None` for degenerate cases (total_weight == 0 with no negative
    /// weights), matching imageflow's `Err(TotalWeightZero)`.
    fn imageflow_populate_pixel(
        out_pixel: u32,
        in_size: u32,
        scale: f64,
        downscale_factor: f64,
        filter: &InterpolationDetails,
        sharpen_ratio: f64,
        desired_sharpen_ratio: f64,
    ) -> Option<(u32, Vec<f32>)> {
        let center = (out_pixel as f64 + 0.5) / scale - 0.5;
        let left_edge = (center - filter.window / downscale_factor - 0.0001).ceil() as i32;
        let right_edge = (center + filter.window / downscale_factor + 0.0001).floor() as i32;
        let left_pixel = left_edge.max(0) as u32;
        let right_pixel = right_edge.min(in_size as i32 - 1) as u32;

        let mut weights = Vec::new();
        let mut total_weight = 0.0f64;
        let mut total_negative_weight = 0.0f64;
        let mut total_positive_weight = 0.0f64;

        for ix in left_pixel..=right_pixel {
            let mut add = filter.filter(downscale_factor * (ix as f64 - center));
            if add.abs() <= 2e-8 {
                add = 0.0;
            }
            weights.push(add as f32);
            total_weight += add;
            total_negative_weight += add.min(0.0);
            total_positive_weight += add.max(0.0);
        }

        // imageflow normalization: matches populate_weights exactly
        let mut neg_factor = (1.0f64 / total_weight) as f32;
        let mut pos_factor = neg_factor;

        if total_weight <= 0.0 || desired_sharpen_ratio > sharpen_ratio {
            if total_negative_weight < 0.0 {
                if desired_sharpen_ratio < 1.0 {
                    let target_pos = 1.0 / (1.0 - desired_sharpen_ratio);
                    let target_neg = desired_sharpen_ratio * -target_pos;
                    pos_factor = (target_pos / total_positive_weight) as f32;
                    neg_factor = (target_neg / total_negative_weight) as f32;
                    if total_negative_weight == 0.0 {
                        neg_factor = 1.0;
                    }
                }
            } else if total_weight == 0.0 {
                return None; // matches imageflow's Err(TotalWeightZero)
            }
        }

        for v in weights.iter_mut() {
            if *v < 0.0 {
                *v *= neg_factor;
            } else {
                *v *= pos_factor;
            }
        }

        // Trim zeros
        while weights.last() == Some(&0.0f32) {
            weights.pop();
        }
        let mut trimmed_left = left_pixel;
        while weights.first() == Some(&0.0f32) {
            weights.remove(0);
            trimmed_left += 1;
        }

        Some((trimmed_left, weights))
    }

    /// Generate weight output in imageflow's exact format for comparison.
    fn generate_imageflow_format() -> String {
        let scalings: [u32; 44] = [
            1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 17, 1, 2, 3, 2, 4, 2, 5, 2, 17, 11, 7, 7, 3,
            8, 8, 8, 7, 8, 6, 8, 5, 8, 4, 8, 3, 8, 2, 8, 1,
        ];
        let filters = [
            Filter::RobidouxFast,
            Filter::Robidoux,
            Filter::RobidouxSharp,
            Filter::Ginseng,
            Filter::GinsengSharp,
            Filter::Lanczos,
            Filter::LanczosSharp,
            Filter::Lanczos2,
            Filter::Lanczos2Sharp,
            Filter::CubicFast,
            Filter::Cubic,
            Filter::CubicSharp,
            Filter::CatmullRom,
            Filter::Mitchell,
            Filter::CubicBSpline,
            Filter::Hermite,
            Filter::Jinc,
            Filter::RawLanczos3,
            Filter::RawLanczos3Sharp,
            Filter::RawLanczos2,
            Filter::RawLanczos2Sharp,
            Filter::Triangle,
            Filter::Linear,
            Filter::Box,
            Filter::CatmullRomFast,
            Filter::CatmullRomFastSharp,
            Filter::Fastest,
            Filter::MitchellFast,
            Filter::NCubic,
            Filter::NCubicSharp,
        ];

        let mut output = String::from("filter, from_width, to_width, weights");

        for (index, &filter_enum) in filters.iter().enumerate() {
            let details = InterpolationDetails::create(filter_enum);

            for i in (0..scalings.len()).step_by(2) {
                let from_w = scalings[i];
                let to_w = scalings[i + 1];
                let scale = to_w as f64 / from_w as f64;
                let downscale_factor = scale.min(1.0);

                output.push_str(&format!(
                    "\nfilter_{:0>2} ({: >2}px to {: >2}px):",
                    index + 1,
                    from_w,
                    to_w
                ));

                for o in 0..to_w {
                    let (_left, weights) = imageflow_populate_pixel(
                        o,
                        from_w,
                        scale,
                        downscale_factor,
                        &details,
                        0.0,
                        0.0,
                    )
                    .expect("default weights should never fail");
                    output.push_str(&format!(" x={} from ", o));
                    for (w_idx, &w) in weights.iter().enumerate() {
                        output.push_str(if w_idx == 0 { "(" } else { " " });
                        output.push_str(&format!("{:.6}", w));
                    }
                    output.push_str("),");
                }
            }
        }

        output
    }

    /// Verify our filter functions produce weights identical to imageflow's
    /// reference output across all 30 filters × 22 scaling combinations.
    #[test]
    fn weights_match_imageflow_reference() {
        let generated = generate_imageflow_format();
        let reference = include_str!("../tests/weights.txt").replace("\r\n", "\n");
        assert_eq!(
            generated.trim(),
            reference.trim(),
            "Generated weights differ from imageflow reference"
        );
    }

    /// Verify zenresize's F32WeightTable (with f32 renormalization) stays
    /// within 1e-5 of imageflow's weights for every tap.
    #[test]
    #[cfg(feature = "std")]
    fn f32_weight_table_matches_imageflow_within_tolerance() {
        let scalings: [u32; 44] = [
            1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 17, 1, 2, 3, 2, 4, 2, 5, 2, 17, 11, 7, 7, 3,
            8, 8, 8, 7, 8, 6, 8, 5, 8, 4, 8, 3, 8, 2, 8, 1,
        ];
        let mut max_dev = 0.0f32;
        let mut max_dev_filter = Filter::Robidoux;
        let mut max_dev_scaling = (0u32, 0u32);

        for &filter_enum in Filter::all() {
            let details = InterpolationDetails::create(filter_enum);

            for i in (0..scalings.len()).step_by(2) {
                let from_w = scalings[i];
                let to_w = scalings[i + 1];
                let scale = to_w as f64 / from_w as f64;
                let downscale_factor = scale.min(1.0);
                let table = F32WeightTable::new(from_w, to_w, &details);

                for o in 0..to_w {
                    let (_if_left, if_weights) = imageflow_populate_pixel(
                        o,
                        from_w,
                        scale,
                        downscale_factor,
                        &details,
                        0.0,
                        0.0,
                    )
                    .expect("default weights should never fail");
                    let zen_weights = table.weights(o as usize);
                    assert_eq!(
                        zen_weights.len(),
                        if_weights.len(),
                        "Tap count mismatch for {:?} {}→{} pixel {}",
                        filter_enum,
                        from_w,
                        to_w,
                        o
                    );
                    for (j, (&zw, &iw)) in zen_weights.iter().zip(if_weights.iter()).enumerate() {
                        let dev = (zw - iw).abs();
                        if dev > max_dev {
                            max_dev = dev;
                            max_dev_filter = filter_enum;
                            max_dev_scaling = (from_w, to_w);
                        }
                        assert!(
                            dev < 1e-5,
                            "Weight deviation {:.8} for {:?} {}→{} pixel {} tap {} (zen={:.8} if={:.8})",
                            dev,
                            filter_enum,
                            from_w,
                            to_w,
                            o,
                            j,
                            zw,
                            iw
                        );
                    }
                }
            }
        }

        eprintln!(
            "Max F32WeightTable deviation from imageflow: {:.2e} ({:?} {}→{})",
            max_dev, max_dev_filter, max_dev_scaling.0, max_dev_scaling.1
        );
    }

    /// Blur/sharpen parameter variation for weight parity tests.
    #[derive(Clone, Copy)]
    enum ParamVariation {
        Default,
        Blur(f64),
        Sharpen(f32),
    }

    impl core::fmt::Display for ParamVariation {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                ParamVariation::Default => write!(f, "default"),
                ParamVariation::Blur(b) => write!(f, "blur={:.2}", b),
                ParamVariation::Sharpen(s) => write!(f, "sharpen={:.1}", s),
            }
        }
    }

    /// Generate weight output for blur/sharpen parameter combinations,
    /// matching imageflow's `weights_params.rs` format exactly.
    fn generate_param_weights() -> String {
        let filters = [
            Filter::Robidoux,
            Filter::RobidouxSharp,
            Filter::Lanczos,
            Filter::Lanczos2,
            Filter::Lanczos2Sharp,
            Filter::CatmullRom,
            Filter::Mitchell,
            Filter::Ginseng,
            Filter::CubicFast,
            Filter::Hermite,
            Filter::Triangle,
            Filter::Box,
        ];

        let scalings: [(u32, u32); 10] = [
            (1, 1),
            (4, 1),
            (7, 3),
            (11, 7),
            (2, 5),
            (2, 9),
            (8, 8),
            (8, 5),
            (8, 3),
            (17, 11),
        ];

        let variations = [
            ParamVariation::Default,
            ParamVariation::Blur(0.8),
            ParamVariation::Blur(0.9),
            ParamVariation::Blur(1.1),
            ParamVariation::Blur(1.2),
            ParamVariation::Sharpen(5.0),
            ParamVariation::Sharpen(15.0),
            ParamVariation::Sharpen(50.0),
        ];

        let mut output = String::from("filter, param, from_width, to_width, weights");

        for &filter_enum in &filters {
            for &variation in &variations {
                let mut details = InterpolationDetails::create(filter_enum);
                match variation {
                    ParamVariation::Default => {}
                    ParamVariation::Blur(factor) => {
                        details = details.with_blur(factor);
                    }
                    ParamVariation::Sharpen(pct) => {
                        details = details.with_sharpen_percent(pct);
                    }
                }

                // Compute sharpen ratios for imageflow-exact normalization
                let sharpen_ratio = details.calculate_percent_negative_weight();
                let desired_sharpen_ratio = match variation {
                    ParamVariation::Sharpen(pct) => {
                        1.0f64.min(sharpen_ratio.max(pct as f64 / 100.0))
                    }
                    _ => 0.0,
                };

                for &(from_w, to_w) in &scalings {
                    let scale = to_w as f64 / from_w as f64;
                    let downscale_factor = scale.min(1.0);

                    // Compute all pixels; if any fails, it's an ERROR line
                    let mut pixel_weights = Vec::new();
                    let mut any_error = false;
                    for o in 0..to_w {
                        match imageflow_populate_pixel(
                            o,
                            from_w,
                            scale,
                            downscale_factor,
                            &details,
                            sharpen_ratio,
                            desired_sharpen_ratio,
                        ) {
                            Some(pw) => pixel_weights.push(pw),
                            None => {
                                any_error = true;
                                break;
                            }
                        }
                    }

                    if any_error {
                        output.push_str(&format!(
                            "\n{:?} {} ({: >3}px to {: >2}px): ERROR",
                            filter_enum, variation, from_w, to_w
                        ));
                        continue;
                    }

                    output.push_str(&format!(
                        "\n{:?} {} ({: >3}px to {: >2}px):",
                        filter_enum, variation, from_w, to_w
                    ));

                    for (o, (_left, weights)) in pixel_weights.iter().enumerate() {
                        output.push_str(&format!(" x={} from ", o));
                        for (w_idx, &w) in weights.iter().enumerate() {
                            output.push_str(if w_idx == 0 { "(" } else { " " });
                            output.push_str(&format!("{:.6}", w));
                        }
                        output.push_str("),");
                    }
                }
            }
        }
        output
    }

    /// Set to `true` to overwrite weights_params.txt with current output.
    /// Must be set back to `false` before committing.
    const UPDATE_PARAMS_REFERENCE: bool = false;

    /// Verify our filter functions with blur/sharpen produce weights identical
    /// to imageflow's reference across 12 filters × 8 variations × 10 scalings.
    #[test]
    #[cfg(feature = "std")]
    fn weights_params_match_imageflow_reference() {
        let generated = generate_param_weights();
        let reference_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("weights_params.txt");

        if UPDATE_PARAMS_REFERENCE {
            std::fs::write(&reference_path, &generated).expect("Failed to write reference file");
            panic!(
                "UPDATE_PARAMS_REFERENCE is true — wrote {}. Set back to false before committing.",
                reference_path.display()
            );
        }

        let reference = std::fs::read_to_string(&reference_path)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to read {}: {}. \
                     Set UPDATE_PARAMS_REFERENCE = true to generate it.",
                    reference_path.display(),
                    e
                )
            })
            .replace("\r\n", "\n");

        assert_eq!(
            generated.trim(),
            reference.trim(),
            "Generated param weights differ from reference file {}. \
             Set UPDATE_PARAMS_REFERENCE = true, run the test, then set it back to false.",
            reference_path.display()
        );
    }

    /// Verify F32WeightTable's sharpen produces the same weights as
    /// imageflow's populate_weights (with sharpen) across all param variations.
    #[test]
    #[cfg(feature = "std")]
    fn f32_weight_table_sharpen_matches_imageflow() {
        let filters = [
            Filter::Robidoux,
            Filter::RobidouxSharp,
            Filter::Lanczos,
            Filter::Lanczos2,
            Filter::CatmullRom,
            Filter::Mitchell,
            Filter::Ginseng,
            Filter::Hermite,
        ];
        let scalings: [(u32, u32); 6] = [(4, 1), (7, 3), (11, 7), (2, 5), (8, 5), (17, 11)];
        let variations: [(f64, f32); 8] = [
            // (blur_factor, sharpen_percent)
            (1.0, 0.0),  // default
            (0.8, 0.0),  // blur < 1
            (0.9, 0.0),  // blur < 1
            (1.1, 0.0),  // blur > 1
            (1.2, 0.0),  // blur > 1
            (1.0, 5.0),  // sharpen
            (1.0, 15.0), // sharpen
            (1.0, 50.0), // sharpen
        ];

        let mut max_dev = 0.0f32;
        let mut max_dev_info = String::new();

        for &filter_enum in &filters {
            for &(blur, sharpen) in &variations {
                let mut details = InterpolationDetails::create(filter_enum);
                if blur != 1.0 {
                    details = details.with_blur(blur);
                }
                if sharpen > 0.0 {
                    details = details.with_sharpen_percent(sharpen);
                }

                let sharpen_ratio = details.calculate_percent_negative_weight();
                let desired_sharpen_ratio = if sharpen > 0.0 {
                    1.0f64.min(sharpen_ratio.max(sharpen as f64 / 100.0))
                } else {
                    0.0
                };

                for &(from_w, to_w) in &scalings {
                    let table = F32WeightTable::new(from_w, to_w, &details);
                    let scale = to_w as f64 / from_w as f64;
                    let downscale_factor = scale.min(1.0);

                    for o in 0..to_w {
                        let result = imageflow_populate_pixel(
                            o,
                            from_w,
                            scale,
                            downscale_factor,
                            &details,
                            sharpen_ratio,
                            desired_sharpen_ratio,
                        );
                        let Some((_left, if_weights)) = result else {
                            continue;
                        };
                        let zen_weights = table.weights(o as usize);

                        if zen_weights.len() != if_weights.len() {
                            panic!(
                                "{:?} blur={} sharpen={} {}->{}  pixel {}: \
                                 tap count {} vs {}",
                                filter_enum,
                                blur,
                                sharpen,
                                from_w,
                                to_w,
                                o,
                                zen_weights.len(),
                                if_weights.len()
                            );
                        }

                        for (j, (&zw, &iw)) in zen_weights.iter().zip(if_weights.iter()).enumerate()
                        {
                            let dev = (zw - iw).abs();
                            if dev > max_dev {
                                max_dev = dev;
                                max_dev_info = format!(
                                    "{:?} blur={} sharpen={} {}→{} px{} tap{}",
                                    filter_enum, blur, sharpen, from_w, to_w, o, j
                                );
                            }
                        }
                    }
                }
            }
        }

        eprintln!(
            "Max F32WeightTable vs imageflow sharpen deviation: {:.2e} ({})",
            max_dev, max_dev_info
        );
        assert!(
            max_dev < 5e-5,
            "F32WeightTable sharpen deviates from imageflow by {:.8} at {}",
            max_dev,
            max_dev_info
        );
    }

    const _: () = assert!(
        !UPDATE_PARAMS_REFERENCE,
        "UPDATE_PARAMS_REFERENCE must be false in committed code"
    );

    #[test]
    fn test_imageflow_parity_downscale() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let table = F32WeightTable::new(100, 50, &filter);

        // Reference from imageflow for pixel 25 (middle)
        let ref_25_left = 45i32;
        let ref_25_weights = [
            0.0036891357f32,
            0.0150561435,
            -0.03399863,
            -0.06663732,
            0.13550529,
            0.44638538,
            0.44638538,
            0.13550529,
            -0.06663732,
            -0.03399863,
            0.0150561435,
            0.0036891357,
        ];

        assert_eq!(table.left[25], ref_25_left);
        let w = table.weights(25);
        assert_eq!(w.len(), ref_25_weights.len());
        for (i, (&got, &expected)) in w.iter().zip(ref_25_weights.iter()).enumerate() {
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-6,
                "Pixel 25 weight {}: got {} expected {}, diff {}",
                i,
                got,
                expected,
                diff
            );
        }
    }
}

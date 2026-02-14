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
use alloc::vec::Vec;

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
    pub fn new(in_size: u32, out_size: u32, filter: &InterpolationDetails) -> Self {
        debug_assert!(in_size > 0, "in_size must be positive");
        debug_assert!(out_size > 0, "out_size must be positive");

        let scale = out_size as f64 / in_size as f64;
        let downscale_factor = scale.min(1.0);
        let effective_window = filter.window / downscale_factor;

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

        // Second pass: fill flat array and renormalize f32 weights.
        // The f64→f32 conversion introduces drift; renormalizing ensures
        // each pixel's weights sum to exactly 1.0f32.
        let mut weights_flat = vec![0.0f32; out_size as usize * max_taps];
        for (i, w) in all_weights.iter().enumerate() {
            let offset = i * max_taps;
            for (j, &val) in w.iter().enumerate() {
                weights_flat[offset + j] = val as f32;
            }
            // Renormalize f32 weights to sum to 1.0
            let f32_sum: f32 = weights_flat[offset..offset + w.len()].iter().sum();
            if f32_sum != 0.0 {
                let inv = 1.0f32 / f32_sum;
                for v in &mut weights_flat[offset..offset + w.len()] {
                    *v *= inv;
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
            let scaled: Vec<f64> = temp_weights.iter().map(|&w| w * scale_factor).collect();
            let mut fixed: Vec<i16> = scaled.iter().map(|&s| s.round() as i16).collect();

            let sum: i32 = fixed.iter().map(|&w| w as i32).sum();
            let error = (1 << I16_PRECISION) - sum;
            if error != 0 {
                // Compute rounding residuals: how much was lost by rounding each weight.
                // residual > 0 means we rounded down, < 0 means we rounded up.
                let mut indices: Vec<usize> = (0..fixed.len()).collect();
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
            all_weights.push(fixed);
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
        let groups4 = (max_taps + 3) / 4;
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

    /// Pointer to the start of the expanded_4ch buffer.
    #[inline]
    pub fn expanded_4ch_ptr(&self) -> *const i16 {
        self.expanded_4ch.as_ptr()
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

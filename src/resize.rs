//! Full-frame resize API.
//!
//! Uses an optimized two-pass approach for full-frame resizes:
//! 1. Horizontal pass: convert u8→f32, premultiply alpha, horizontal filter → intermediate
//! 2. Vertical pass: vertical filter → unpremultiply alpha, convert f32→u8 → output
//!
//! This eliminates the per-row allocation overhead of the streaming path.
//! The streaming API ([`crate::streaming::StreamingResize`]) is still available
//! for pipeline integration where the full image isn't in memory.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::color;
use crate::filter::InterpolationDetails;
use crate::pixel::ResizeConfig;
use crate::proven;
use crate::simd;
use crate::weights::{F32WeightTable, I16WeightTable};

/// Resize an entire u8 image. Allocates and returns output buffer.
///
/// Input may be tightly packed or strided (set `in_stride` in config).
/// Output is always tightly packed.
///
/// # Panics
/// Panics if the config is invalid or input is too short.
pub fn resize(config: &ResizeConfig, input: &[u8]) -> Vec<u8> {
    let out_row_len = config.output_row_len();
    let len = config.out_height as usize * out_row_len;
    let mut output = proven::alloc_output::<u8>(len);
    resize_into(config, input, &mut output);
    output
}

/// Resize a u8 image into a caller-provided buffer.
///
/// Output buffer must be tightly packed: `out_width * out_height * channels`.
///
/// # Panics
/// Panics if input is too short or output length doesn't match.
pub fn resize_into(config: &ResizeConfig, input: &[u8], output: &mut [u8]) {
    config.validate().expect("invalid resize config");

    let in_stride = config.effective_in_stride();
    let in_row_len = config.input_row_len();
    let in_expected = if config.in_height > 0 {
        (config.in_height as usize - 1) * in_stride + in_row_len
    } else {
        0
    };
    let out_row_len = config.output_row_len();
    let out_expected = config.out_height as usize * out_row_len;
    assert!(input.len() >= in_expected, "input too short");
    assert_eq!(output.len(), out_expected, "output length mismatch");

    let channels = config.input_format.channels() as usize;
    let has_alpha = config.input_format.has_alpha();
    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;
    let linearize = config.needs_linearization();

    // Use integer fast path for sRGB-space resize (no linearization needed).
    // This avoids u8→f32→u8 conversion and uses i16 weights with i32 accumulators,
    // doubling SIMD throughput and quartering intermediate buffer size.
    // Supports alpha premultiply/unpremultiply in u8 space.
    if !linearize && channels == 4 {
        resize_into_i16(
            config,
            input,
            output,
            in_stride,
            in_row_len,
            out_row_len,
            channels,
            in_w,
            in_h,
            out_w,
            out_h,
            has_alpha,
        );
        return;
    }

    // Integer fast path for linear-light resize (sRGB input, linear processing).
    // Uses 12-bit linear values (0-4095) stored as i16, reusing the same
    // I16WeightTable and madd_epi16 SIMD infrastructure. ~4x faster than f32.
    // Currently only for 4ch without alpha premul/unpremul.
    if linearize && channels == 4 && !has_alpha {
        resize_into_i16_linear(
            config,
            input,
            output,
            in_stride,
            in_row_len,
            out_row_len,
            channels,
            in_h,
            out_w,
            out_h,
        );
        return;
    }

    let filter = InterpolationDetails::create(config.filter);
    let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
    let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);

    let h_row_len = out_w * channels;

    // Intermediate buffer for horizontally-filtered rows (f32).
    let mut intermediate = vec![0.0f32; h_row_len * in_h];
    // Pad temp_row with max_taps extra zero elements so SIMD H-pass reads
    // beyond the valid input range hit zeros instead of uninitialized memory.
    // This prevents NaN poisoning: 0.0 * NaN = NaN, but 0.0 * 0.0 = 0.0.
    let mut temp_row = vec![0.0f32; in_w * channels + h_weights.max_taps * channels];

    // === Horizontal pass ===
    for y in 0..in_h {
        let in_start = y * in_stride;
        let in_row = &input[in_start..in_start + in_row_len];

        // u8 → f32 (writes only the first in_w*channels elements; padding stays zero)
        if linearize {
            color::srgb_u8_to_linear_f32(in_row, &mut temp_row[..in_row_len], channels, has_alpha);
        } else {
            simd::u8_to_f32_row(in_row, &mut temp_row[..in_row_len]);
        }

        // Premultiply alpha
        if has_alpha && channels == 4 {
            simd::premultiply_alpha_row(&mut temp_row);
        }

        // Horizontal filter → intermediate buffer
        let out_start = y * h_row_len;
        simd::filter_h_row_f32(
            &temp_row,
            &mut intermediate[out_start..out_start + h_row_len],
            &h_weights,
            channels,
        );
    }

    // === Vertical pass ===
    let max_taps = v_weights.max_taps;
    let mut row_ptrs: Vec<&[f32]> = Vec::with_capacity(max_taps);
    let mut temp_output = vec![0.0f32; h_row_len];

    for out_y in 0..out_h {
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        row_ptrs.clear();
        for t in 0..tap_count {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            let start = in_y * h_row_len;
            row_ptrs.push(&intermediate[start..start + h_row_len]);
        }

        simd::filter_v_row_f32(&row_ptrs, &mut temp_output[..h_row_len], weights);

        // Unpremultiply alpha
        if has_alpha && channels == 4 {
            simd::unpremultiply_alpha_row(&mut temp_output[..h_row_len]);
        }

        // f32 → u8 directly into output buffer
        let out_start = out_y * out_row_len;
        let out_slice = &mut output[out_start..out_start + out_row_len];
        if linearize {
            color::linear_f32_to_srgb_u8(&temp_output[..h_row_len], out_slice, channels, has_alpha);
        } else {
            simd::f32_to_u8_row(&temp_output[..h_row_len], out_slice);
        }
    }
}

/// Integer (i16 weights) fast path for sRGB-space resize.
///
/// Uses horizontal-first architecture with u8 intermediate buffer:
/// 1. H pass: premul → horizontal filter → intermediate (u8, 4× smaller than f32)
/// 2. V pass: vertical filter from intermediate → unpremul → output
///
/// The u8 intermediate fits in L2 cache, giving excellent V-pass locality.
#[allow(clippy::too_many_arguments)]
fn resize_into_i16(
    config: &ResizeConfig,
    input: &[u8],
    output: &mut [u8],
    in_stride: usize,
    in_row_len: usize,
    out_row_len: usize,
    channels: usize,
    _in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    has_alpha: bool,
) {
    let filter = InterpolationDetails::create(config.filter);
    let h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);
    let v_weights = I16WeightTable::new(config.in_height, config.out_height, &filter);

    let h_row_len = out_w * channels;

    // u8 intermediate: 4x smaller than f32 → fits in L2 cache.
    let mut intermediate = vec![0u8; h_row_len * in_h];

    // Temp buffer for premultiplied input row (reused per row, L1-hot).
    // Include h_padding extra bytes so the H kernel can use its efficient
    // global-guard path. Padding bytes are zero-initialized; the H kernel's
    // zero-padded weights ensure they don't affect results.
    let h_padding = h_weights.groups4 * 16; // extra bytes for SIMD overread
    let mut premul_buf = if has_alpha {
        vec![0u8; in_row_len + h_padding]
    } else {
        Vec::new()
    };

    // === Horizontal pass: u8 → i32 → u8 ===
    // Use 4-row batch for RGBA without alpha (most common case).
    //
    // Input slices are extended beyond the row pixel data to include "SIMD padding"
    // from adjacent rows in the contiguous buffer. This allows the H kernel to use
    // a single global guard (no per-pixel edge checks), since zero-padded weights
    // ensure the padding data doesn't affect results.
    if channels == 4 && !has_alpha {
        // Process 4 rows at a time
        let batch_count = in_h / 4;
        let remainder = in_h % 4;

        for batch in 0..batch_count {
            let y0 = batch * 4;
            // Extend each row slice to include padding from the buffer.
            // rows 0-2 always have a next row. Row 3 may be the last row.
            let r0 =
                &input[y0 * in_stride..(y0 * in_stride + in_row_len + h_padding).min(input.len())];
            let r1 = &input[(y0 + 1) * in_stride
                ..((y0 + 1) * in_stride + in_row_len + h_padding).min(input.len())];
            let r2 = &input[(y0 + 2) * in_stride
                ..((y0 + 2) * in_stride + in_row_len + h_padding).min(input.len())];
            let r3 = &input[(y0 + 3) * in_stride
                ..((y0 + 3) * in_stride + in_row_len + h_padding).min(input.len())];

            let out_base = y0 * h_row_len;
            // Split intermediate into 4 disjoint mutable slices
            let (o0, rest) = intermediate[out_base..].split_at_mut(h_row_len);
            let (o1, rest) = rest.split_at_mut(h_row_len);
            let (o2, o3_and_rest) = rest.split_at_mut(h_row_len);
            let o3 = &mut o3_and_rest[..h_row_len];

            simd::filter_h_u8_i16_4rows(r0, r1, r2, r3, o0, o1, o2, o3, &h_weights);
        }

        // Handle remaining rows
        for i in 0..remainder {
            let y = batch_count * 4 + i;
            let in_start = y * in_stride;
            let in_end = (in_start + in_row_len + h_padding).min(input.len());
            let in_row = &input[in_start..in_end];
            let out_start = y * h_row_len;

            simd::filter_h_u8_i16(
                in_row,
                &mut intermediate[out_start..out_start + h_row_len],
                &h_weights,
                channels,
            );
        }
    } else {
        // Single-row path: handles alpha premul and non-4ch formats
        for y in 0..in_h {
            let in_start = y * in_stride;
            let in_row = &input[in_start..in_start + in_row_len];
            let out_start = y * h_row_len;

            let src = if has_alpha {
                simd::premultiply_u8_row(in_row, &mut premul_buf[..in_row_len]);
                &premul_buf[..] // includes zero-initialized SIMD padding
            } else {
                in_row
            };

            simd::filter_h_u8_i16(
                src,
                &mut intermediate[out_start..out_start + h_row_len],
                &h_weights,
                channels,
            );
        }
    }

    // === Vertical pass: u8 → i32 → u8 ===
    // Batch V kernel: processes all output rows in one call,
    // avoiding per-row dispatch overhead and row_ptrs construction.
    simd::filter_v_all_u8_i16(&intermediate, output, h_row_len, in_h, out_h, &v_weights);

    // Unpremultiply alpha after V pass (separated from V for batch efficiency).
    if has_alpha {
        for out_y in 0..out_h {
            let out_start = out_y * out_row_len;
            simd::unpremultiply_u8_row(&mut output[out_start..out_start + out_row_len]);
        }
    }
}

/// Integer (i16) fast path for linear-light resize.
///
/// Uses 12-bit linear values (0-4095) stored as i16, with LUT-based
/// sRGB↔linear conversion. Reuses the same I16WeightTable and madd_epi16
/// SIMD infrastructure as the sRGB i16 path.
///
/// Data flow: u8 sRGB → LUT → i16 linear → H-pass → i16 intermediate
///            → V-pass → i16 output → LUT → u8 sRGB
#[allow(clippy::too_many_arguments)]
fn resize_into_i16_linear(
    config: &ResizeConfig,
    input: &[u8],
    output: &mut [u8],
    in_stride: usize,
    in_row_len: usize,
    _out_row_len: usize,
    channels: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) {
    let filter = InterpolationDetails::create(config.filter);
    let h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);
    let v_weights = I16WeightTable::new(config.in_height, config.out_height, &filter);

    let h_row_len = out_w * channels;

    // i16 intermediate: 2x smaller than f32, holds linear i12 values.
    let mut intermediate = vec![0i16; h_row_len * in_h];

    // Reusable buffer for linearized input row + SIMD padding.
    // The H-pass reads groups of 16 i16 (32 bytes), so we need padding
    // to prevent out-of-bounds reads on the last group.
    let h_padding = h_weights.groups4 * 16;
    let mut linearized_row = vec![0i16; in_row_len + h_padding];

    // === Horizontal pass: u8 sRGB → i16 linear → filter → i16 intermediate ===
    for y in 0..in_h {
        let in_start = y * in_stride;
        let in_row = &input[in_start..in_start + in_row_len];

        // LUT: sRGB u8 → 12-bit linear i16
        color::srgb_u8_to_linear_i12_row(in_row, &mut linearized_row[..in_row_len]);
        // Zero SIMD padding region
        for v in &mut linearized_row[in_row_len..] {
            *v = 0;
        }

        let out_start = y * h_row_len;
        simd::filter_h_i16_i16(
            &linearized_row,
            &mut intermediate[out_start..out_start + h_row_len],
            &h_weights,
            channels,
        );
    }

    // === Vertical pass: i16 intermediate → i16 output ===
    let mut v_output = vec![0i16; h_row_len * out_h];
    simd::filter_v_all_i16_i16(
        &intermediate,
        &mut v_output,
        h_row_len,
        in_h,
        out_h,
        &v_weights,
    );

    // === Output: i16 linear → u8 sRGB via LUT ===
    for out_y in 0..out_h {
        let v_start = out_y * h_row_len;
        let out_start = out_y * h_row_len;
        color::linear_i12_to_srgb_u8_row(
            &v_output[v_start..v_start + h_row_len],
            &mut output[out_start..out_start + h_row_len],
        );
    }
}

/// Reusable resizer with pre-computed weight tables.
///
/// When resizing many images with the same dimensions and filter,
/// `Resizer` avoids recomputing weight tables on each call. This
/// saves significant overhead for repeated resize operations.
pub struct Resizer {
    config: ResizeConfig,
    h_weights_i16: Option<I16WeightTable>,
    v_weights_i16: Option<I16WeightTable>,
    h_weights_f32: Option<F32WeightTable>,
    v_weights_f32: Option<F32WeightTable>,
    intermediate_u8: Vec<u8>,
    intermediate_f32: Vec<f32>,
    intermediate_i16: Vec<i16>,
    linearized_row: Vec<i16>,
    v_output_i16: Vec<i16>,
    premul_buf: Vec<u8>,
    temp_row_f32: Vec<f32>,
    temp_output_f32: Vec<f32>,
    /// Which path to use: 0 = sRGB i16, 1 = linear i16, 2 = f32
    path: u8,
}

impl Resizer {
    /// Create a new resizer for the given configuration.
    /// Pre-computes weight tables.
    pub fn new(config: &ResizeConfig) -> Self {
        config.validate().expect("invalid resize config");
        let filter = InterpolationDetails::create(config.filter);
        let channels = config.input_format.channels() as usize;
        let linearize = config.needs_linearization();
        let has_alpha = config.input_format.has_alpha();
        let in_h = config.in_height as usize;
        let out_w = config.out_width as usize;
        let h_row_len = out_w * channels;
        let in_row_len = config.input_row_len();

        if !linearize && channels == 4 {
            // Path 0: sRGB i16 fast path
            let h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);
            let v_weights = I16WeightTable::new(config.in_height, config.out_height, &filter);
            let intermediate = vec![0u8; h_row_len * in_h];
            let h_padding = h_weights.groups4 * 16;
            let premul_buf = if has_alpha {
                vec![0u8; in_row_len + h_padding]
            } else {
                Vec::new()
            };
            Resizer {
                config: config.clone(),
                h_weights_i16: Some(h_weights),
                v_weights_i16: Some(v_weights),
                h_weights_f32: None,
                v_weights_f32: None,
                intermediate_u8: intermediate,
                intermediate_f32: Vec::new(),
                intermediate_i16: Vec::new(),
                linearized_row: Vec::new(),
                v_output_i16: Vec::new(),
                premul_buf,
                temp_row_f32: Vec::new(),
                temp_output_f32: Vec::new(),
                path: 0,
            }
        } else if linearize && channels == 4 && !has_alpha {
            // Path 1: linear-light i16 fast path
            let h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);
            let v_weights = I16WeightTable::new(config.in_height, config.out_height, &filter);
            let h_padding = h_weights.groups4 * 16;
            let out_h = config.out_height as usize;
            Resizer {
                config: config.clone(),
                h_weights_i16: Some(h_weights),
                v_weights_i16: Some(v_weights),
                h_weights_f32: None,
                v_weights_f32: None,
                intermediate_u8: Vec::new(),
                intermediate_f32: Vec::new(),
                intermediate_i16: vec![0i16; h_row_len * in_h],
                linearized_row: vec![0i16; in_row_len + h_padding],
                v_output_i16: vec![0i16; h_row_len * out_h],
                premul_buf: Vec::new(),
                temp_row_f32: Vec::new(),
                temp_output_f32: Vec::new(),
                path: 1,
            }
        } else {
            // Path 2: f32 path
            let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
            let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);
            let in_w = config.in_width as usize;
            let h_max_taps = h_weights.max_taps;
            Resizer {
                config: config.clone(),
                h_weights_i16: None,
                v_weights_i16: None,
                h_weights_f32: Some(h_weights),
                v_weights_f32: Some(v_weights),
                intermediate_u8: Vec::new(),
                intermediate_f32: vec![0.0f32; h_row_len * in_h],
                intermediate_i16: Vec::new(),
                linearized_row: Vec::new(),
                v_output_i16: Vec::new(),
                premul_buf: Vec::new(),
                temp_row_f32: vec![0.0f32; in_w * channels + h_max_taps * channels],
                temp_output_f32: vec![0.0f32; h_row_len],
                path: 2,
            }
        }
    }

    /// Resize a u8 image, allocating and returning the output.
    pub fn resize(&mut self, input: &[u8]) -> Vec<u8> {
        let out_row_len = self.config.output_row_len();
        let len = self.config.out_height as usize * out_row_len;
        let mut output = proven::alloc_output::<u8>(len);
        self.resize_into(input, &mut output);
        output
    }

    /// Resize a u8 image into a caller-provided buffer.
    pub fn resize_into(&mut self, input: &[u8], output: &mut [u8]) {
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let channels = config.input_format.channels() as usize;
        let has_alpha = config.input_format.has_alpha();
        let in_h = config.in_height as usize;
        let out_w = config.out_width as usize;
        let out_h = config.out_height as usize;
        let h_row_len = out_w * channels;

        match self.path {
            0 => {
                // Path 0: sRGB i16 fast path
                let h_weights = self.h_weights_i16.as_ref().unwrap();
                let v_weights = self.v_weights_i16.as_ref().unwrap();
                let intermediate = &mut self.intermediate_u8;

                if channels == 4 && !has_alpha {
                    let h_padding = h_weights.groups4 * 16;
                    let batch_count = in_h / 4;
                    let remainder = in_h % 4;

                    for batch in 0..batch_count {
                        let y0 = batch * 4;
                        let r0 = &input[y0 * in_stride
                            ..(y0 * in_stride + in_row_len + h_padding).min(input.len())];
                        let r1 = &input[(y0 + 1) * in_stride
                            ..((y0 + 1) * in_stride + in_row_len + h_padding).min(input.len())];
                        let r2 = &input[(y0 + 2) * in_stride
                            ..((y0 + 2) * in_stride + in_row_len + h_padding).min(input.len())];
                        let r3 = &input[(y0 + 3) * in_stride
                            ..((y0 + 3) * in_stride + in_row_len + h_padding).min(input.len())];

                        let out_base = y0 * h_row_len;
                        let (o0, rest) = intermediate[out_base..].split_at_mut(h_row_len);
                        let (o1, rest) = rest.split_at_mut(h_row_len);
                        let (o2, o3_and_rest) = rest.split_at_mut(h_row_len);
                        let o3 = &mut o3_and_rest[..h_row_len];

                        simd::filter_h_u8_i16_4rows(r0, r1, r2, r3, o0, o1, o2, o3, h_weights);
                    }

                    for i in 0..remainder {
                        let y = batch_count * 4 + i;
                        let in_start = y * in_stride;
                        let in_end = (in_start + in_row_len + h_padding).min(input.len());
                        let in_row = &input[in_start..in_end];
                        let out_start = y * h_row_len;

                        simd::filter_h_u8_i16(
                            in_row,
                            &mut intermediate[out_start..out_start + h_row_len],
                            h_weights,
                            channels,
                        );
                    }
                } else {
                    for y in 0..in_h {
                        let in_start = y * in_stride;
                        let in_row = &input[in_start..in_start + in_row_len];
                        let out_start = y * h_row_len;

                        let src = if has_alpha {
                            simd::premultiply_u8_row(in_row, &mut self.premul_buf[..in_row_len]);
                            &self.premul_buf[..] // includes zero-initialized SIMD padding
                        } else {
                            in_row
                        };

                        simd::filter_h_u8_i16(
                            src,
                            &mut intermediate[out_start..out_start + h_row_len],
                            h_weights,
                            channels,
                        );
                    }
                }

                // V pass
                simd::filter_v_all_u8_i16(intermediate, output, h_row_len, in_h, out_h, v_weights);

                if has_alpha {
                    for out_y in 0..out_h {
                        let out_start = out_y * out_row_len;
                        simd::unpremultiply_u8_row(&mut output[out_start..out_start + out_row_len]);
                    }
                }
            }
            1 => {
                // Path 1: linear-light i16 fast path
                let h_weights = self.h_weights_i16.as_ref().unwrap();
                let v_weights = self.v_weights_i16.as_ref().unwrap();

                // H-pass: per-row sRGB→linear LUT + i16 filter
                for y in 0..in_h {
                    let in_start = y * in_stride;
                    let in_row = &input[in_start..in_start + in_row_len];

                    color::srgb_u8_to_linear_i12_row(
                        in_row,
                        &mut self.linearized_row[..in_row_len],
                    );
                    // Zero SIMD padding region
                    for v in &mut self.linearized_row[in_row_len..] {
                        *v = 0;
                    }

                    let out_start = y * h_row_len;
                    simd::filter_h_i16_i16(
                        &self.linearized_row,
                        &mut self.intermediate_i16[out_start..out_start + h_row_len],
                        h_weights,
                        channels,
                    );
                }

                // V-pass: batch kernel
                simd::filter_v_all_i16_i16(
                    &self.intermediate_i16,
                    &mut self.v_output_i16,
                    h_row_len,
                    in_h,
                    out_h,
                    v_weights,
                );

                // Output: per-row linear→sRGB LUT
                for out_y in 0..out_h {
                    let v_start = out_y * h_row_len;
                    let out_start = out_y * h_row_len;
                    color::linear_i12_to_srgb_u8_row(
                        &self.v_output_i16[v_start..v_start + h_row_len],
                        &mut output[out_start..out_start + h_row_len],
                    );
                }
            }
            _ => {
                // Path 2: f32 path (linear color space with alpha, or non-4ch formats)
                let h_weights = self.h_weights_f32.as_ref().unwrap();
                let v_weights = self.v_weights_f32.as_ref().unwrap();
                let linearize = config.needs_linearization();
                let intermediate = &mut self.intermediate_f32;
                let temp_row = &mut self.temp_row_f32;

                // === Horizontal pass ===
                for y in 0..in_h {
                    let in_start = y * in_stride;
                    let in_row = &input[in_start..in_start + in_row_len];

                    if linearize {
                        color::srgb_u8_to_linear_f32(
                            in_row,
                            &mut temp_row[..in_row_len],
                            channels,
                            has_alpha,
                        );
                    } else {
                        simd::u8_to_f32_row(in_row, &mut temp_row[..in_row_len]);
                    }

                    if has_alpha && channels == 4 {
                        simd::premultiply_alpha_row(&mut temp_row[..in_row_len]);
                    }

                    let out_start = y * h_row_len;
                    simd::filter_h_row_f32(
                        temp_row,
                        &mut intermediate[out_start..out_start + h_row_len],
                        h_weights,
                        channels,
                    );
                }

                // === Vertical pass ===
                let max_taps = v_weights.max_taps;
                let mut row_ptrs: Vec<&[f32]> = Vec::with_capacity(max_taps);
                let temp_output = &mut self.temp_output_f32;

                for out_y in 0..out_h {
                    let left = v_weights.left[out_y];
                    let tap_count = v_weights.tap_count(out_y);
                    let weights = v_weights.weights(out_y);

                    row_ptrs.clear();
                    for t in 0..tap_count {
                        let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                        let start = in_y * h_row_len;
                        row_ptrs.push(&intermediate[start..start + h_row_len]);
                    }

                    simd::filter_v_row_f32(&row_ptrs, &mut temp_output[..h_row_len], weights);

                    if has_alpha && channels == 4 {
                        simd::unpremultiply_alpha_row(&mut temp_output[..h_row_len]);
                    }

                    let out_start = out_y * out_row_len;
                    let out_slice = &mut output[out_start..out_start + out_row_len];
                    if linearize {
                        color::linear_f32_to_srgb_u8(
                            &temp_output[..h_row_len],
                            out_slice,
                            channels,
                            has_alpha,
                        );
                    } else {
                        simd::f32_to_u8_row(&temp_output[..h_row_len], out_slice);
                    }
                }
            }
        }
    }
}

/// Resize an f32 image. Allocates and returns output buffer.
pub fn resize_f32(config: &ResizeConfig, input: &[f32]) -> Vec<f32> {
    let out_row_len = config.output_row_len();
    let mut output = vec![0.0f32; config.out_height as usize * out_row_len];
    resize_f32_into(config, input, &mut output);
    output
}

/// Resize an f32 image into a caller-provided buffer.
pub fn resize_f32_into(config: &ResizeConfig, input: &[f32], output: &mut [f32]) {
    config.validate().expect("invalid resize config");

    let in_stride = config.effective_in_stride();
    let in_row_len = config.input_row_len();
    let in_expected = if config.in_height > 0 {
        (config.in_height as usize - 1) * in_stride + in_row_len
    } else {
        0
    };
    let out_row_len = config.output_row_len();
    let out_expected = config.out_height as usize * out_row_len;
    assert!(input.len() >= in_expected, "input too short");
    assert_eq!(output.len(), out_expected, "output length mismatch");

    let channels = config.input_format.channels() as usize;
    let has_alpha = config.input_format.has_alpha();
    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;

    let filter = InterpolationDetails::create(config.filter);
    let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
    let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);

    let h_row_len = out_w * channels;
    let mut intermediate = vec![0.0f32; h_row_len * in_h];
    let mut temp_row = vec![0.0f32; in_w * channels];

    // === Horizontal pass ===
    for y in 0..in_h {
        let in_start = y * in_stride;
        temp_row[..in_row_len].copy_from_slice(&input[in_start..in_start + in_row_len]);

        if has_alpha && channels == 4 {
            simd::premultiply_alpha_row(&mut temp_row[..in_row_len]);
        }

        let out_start = y * h_row_len;
        simd::filter_h_row_f32(
            &temp_row[..in_row_len],
            &mut intermediate[out_start..out_start + h_row_len],
            &h_weights,
            channels,
        );
    }

    // === Vertical pass ===
    let max_taps = v_weights.max_taps;
    let mut row_ptrs: Vec<&[f32]> = Vec::with_capacity(max_taps);

    for out_y in 0..out_h {
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        row_ptrs.clear();
        for t in 0..tap_count {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            let start = in_y * h_row_len;
            row_ptrs.push(&intermediate[start..start + h_row_len]);
        }

        let out_start = out_y * out_row_len;
        let out_slice = &mut output[out_start..out_start + out_row_len];
        simd::filter_v_row_f32(&row_ptrs, out_slice, weights);

        if has_alpha && channels == 4 {
            simd::unpremultiply_alpha_row(out_slice);
        }
    }
}

// =============================================================================
// imgref integration
// =============================================================================

// =============================================================================
// imgref + rgb crate integration
// =============================================================================
//
// All 4-channel pixel types (RGBA, BGRA, ARGB, ABGR) work identically because
// the pipeline is channel-order-agnostic. The `rgb` crate's ComponentSlice trait
// gives us zero-copy &[u8] access to any pixel type.

#[cfg(feature = "imgref")]
mod imgref_impl {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use crate::pixel::{PixelFormat, ResizeConfig};
    use crate::streaming::StreamingResize;
    use imgref::{Img, ImgRef, ImgVec};
    use rgb::ComponentSlice;

    /// Resize a 4-channel u8 image. Works with any pixel type that implements
    /// `ComponentSlice` (RGBA, BGRA, ARGB, ABGR from the `rgb` crate).
    ///
    /// Channel order is preserved — the pipeline doesn't care whether
    /// the bytes represent RGBA or BGRA.
    pub fn resize_4ch<P>(
        img: ImgRef<P>,
        out_width: u32,
        out_height: u32,
        has_alpha: bool,
        config: &ResizeConfig,
    ) -> ImgVec<P>
    where
        P: Copy + ComponentSlice<u8> + Default,
    {
        assert_eq!(core::mem::size_of::<P>(), 4, "pixel type must be 4 bytes");

        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input_format = PixelFormat::Srgb8 {
            channels: 4,
            has_alpha,
        };
        cfg.output_format = PixelFormat::Srgb8 {
            channels: 4,
            has_alpha,
        };
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        // Reusable buffer: flatten pixel row to &[u8] without per-row allocation.
        // For repr(C) pixel types (rgb::RGBA etc.) the compiler optimizes
        // the per-pixel copy into a single memcpy.
        let w = img.width();
        let mut row_buf = vec![0u8; w * 4];
        for row in img.rows() {
            for (px, chunk) in row.iter().zip(row_buf.chunks_exact_mut(4)) {
                chunk.copy_from_slice(px.as_slice());
            }
            resizer.push_row(&row_buf);
        }
        resizer.finish();

        let out_row_len = cfg.output_row_len();
        let mut out_pixels = Vec::with_capacity(out_width as usize * out_height as usize);
        while let Some(row) = resizer.next_output_row() {
            debug_assert_eq!(row.len(), out_row_len);
            for chunk in row.chunks_exact(4) {
                let mut px = P::default();
                px.as_mut_slice().copy_from_slice(chunk);
                out_pixels.push(px);
            }
        }

        Img::new(out_pixels, out_width as usize, out_height as usize)
    }

    /// Resize a 3-channel u8 image. Works with `rgb::RGB<u8>`, `rgb::BGR<u8>`, etc.
    pub fn resize_3ch<P>(
        img: ImgRef<P>,
        out_width: u32,
        out_height: u32,
        config: &ResizeConfig,
    ) -> ImgVec<P>
    where
        P: Copy + ComponentSlice<u8> + Default,
    {
        assert_eq!(core::mem::size_of::<P>(), 3, "pixel type must be 3 bytes");

        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input_format = PixelFormat::Srgb8 {
            channels: 3,
            has_alpha: false,
        };
        cfg.output_format = PixelFormat::Srgb8 {
            channels: 3,
            has_alpha: false,
        };
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        let w = img.width();
        let mut row_buf = vec![0u8; w * 3];
        for row in img.rows() {
            for (px, chunk) in row.iter().zip(row_buf.chunks_exact_mut(3)) {
                chunk.copy_from_slice(px.as_slice());
            }
            resizer.push_row(&row_buf);
        }
        resizer.finish();

        let out_row_len = cfg.output_row_len();
        let mut out_pixels = Vec::with_capacity(out_width as usize * out_height as usize);
        while let Some(row) = resizer.next_output_row() {
            debug_assert_eq!(row.len(), out_row_len);
            for chunk in row.chunks_exact(3) {
                let mut px = P::default();
                px.as_mut_slice().copy_from_slice(chunk);
                out_pixels.push(px);
            }
        }

        Img::new(out_pixels, out_width as usize, out_height as usize)
    }

    /// Resize a grayscale u8 image.
    pub fn resize_gray8(
        img: ImgRef<u8>,
        out_width: u32,
        out_height: u32,
        config: &ResizeConfig,
    ) -> ImgVec<u8> {
        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input_format = PixelFormat::Srgb8 {
            channels: 1,
            has_alpha: false,
        };
        cfg.output_format = PixelFormat::Srgb8 {
            channels: 1,
            has_alpha: false,
        };
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        for row in img.rows() {
            resizer.push_row(row);
        }
        resizer.finish();

        let mut out_buf = Vec::with_capacity(out_width as usize * out_height as usize);
        while let Some(row) = resizer.next_output_row() {
            out_buf.extend_from_slice(&row);
        }

        Img::new(out_buf, out_width as usize, out_height as usize)
    }
}

#[cfg(feature = "imgref")]
pub use imgref_impl::{resize_3ch, resize_4ch, resize_gray8};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Filter;
    use crate::pixel::PixelFormat;

    fn test_config(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
        ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(Filter::Lanczos)
            .format(PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: true,
            })
            .srgb()
            .build()
    }

    #[test]
    fn test_resize_constant_color() {
        let config = test_config(20, 20, 10, 10);
        let mut input = vec![0u8; 20 * 20 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 128;
            pixel[2] = 128;
            pixel[3] = 255;
        }

        let output = resize(&config, &input);
        assert_eq!(output.len(), 10 * 10 * 4);

        for pixel in output.chunks_exact(4) {
            assert!(
                (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                "R off: {}",
                pixel[0]
            );
            assert!(
                (pixel[3] as i16 - 255).unsigned_abs() <= 1,
                "A off: {}",
                pixel[3]
            );
        }
    }

    #[test]
    fn test_resize_into_matches_resize() {
        let config = test_config(20, 20, 10, 10);
        let input = vec![100u8; 20 * 20 * 4];

        let output_alloc = resize(&config, &input);
        let mut output_into = vec![0u8; 10 * 10 * 4];
        resize_into(&config, &input, &mut output_into);

        assert_eq!(output_alloc, output_into);
    }

    #[test]
    fn test_resize_upscale() {
        let config = test_config(10, 10, 20, 20);
        let input = vec![200u8; 10 * 10 * 4];

        let output = resize(&config, &input);
        assert_eq!(output.len(), 20 * 20 * 4);
    }

    #[test]
    fn test_resize_1x1() {
        let config = test_config(1, 1, 1, 1);
        let input = vec![128, 64, 32, 255];

        let output = resize(&config, &input);
        assert_eq!(output.len(), 4);
        // Should approximately preserve the single pixel
        assert!((output[0] as i16 - 128).unsigned_abs() <= 2);
    }

    #[test]
    fn test_resize_with_stride() {
        // Create an image with extra padding bytes per row
        let config = ResizeConfig::builder(10, 10, 5, 5)
            .format(PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: true,
            })
            .srgb()
            .in_stride(10 * 4 + 8) // 8 bytes padding per row
            .build();

        let stride = 10 * 4 + 8;
        let mut input = vec![0u8; 10 * stride];
        // Fill only the pixel data (first 40 bytes of each row)
        for y in 0..10 {
            for x in 0..10 * 4 {
                input[y * stride + x] = 128;
            }
        }

        let output = resize(&config, &input);
        assert_eq!(output.len(), 5 * 5 * 4);
    }

    #[test]
    fn test_resizer_matches_resize_srgb() {
        let config = test_config(20, 20, 10, 10);
        let input = vec![100u8; 20 * 20 * 4];

        let output_oneshot = resize(&config, &input);
        let mut resizer = Resizer::new(&config);
        let output_cached = resizer.resize(&input);

        assert_eq!(output_oneshot, output_cached);
    }

    #[test]
    fn test_resizer_matches_resize_linear() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: true,
            })
            .linear()
            .build();
        let input = vec![100u8; 20 * 20 * 4];

        let output_oneshot = resize(&config, &input);
        let mut resizer = Resizer::new(&config);
        let output_cached = resizer.resize(&input);

        assert_eq!(output_oneshot, output_cached);
    }

    #[test]
    fn test_resizer_matches_resize_3ch() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelFormat::Srgb8 {
                channels: 3,
                has_alpha: false,
            })
            .srgb()
            .build();
        let input = vec![100u8; 20 * 20 * 3];

        let output_oneshot = resize(&config, &input);
        let mut resizer = Resizer::new(&config);
        let output_cached = resizer.resize(&input);

        assert_eq!(output_oneshot, output_cached);
    }

    #[test]
    fn test_resizer_reuse() {
        let config = test_config(20, 20, 10, 10);
        let mut resizer = Resizer::new(&config);

        let input1 = vec![100u8; 20 * 20 * 4];
        let output1 = resizer.resize(&input1);

        let input2 = vec![200u8; 20 * 20 * 4];
        let output2 = resizer.resize(&input2);

        // Different inputs should produce different outputs
        assert_ne!(output1, output2);
        // Same input should produce same output
        let output1b = resizer.resize(&input1);
        assert_eq!(output1, output1b);
    }

    #[cfg(feature = "imgref")]
    #[test]
    fn test_resize_imgref_rgba() {
        use crate::resize::resize_4ch;
        use imgref::Img;
        use rgb::RGBA;

        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .srgb()
            .build();

        let pixels = vec![RGBA::new(128u8, 128, 128, 255); 20 * 20];
        let img = Img::new(pixels, 20, 20);

        let out = resize_4ch(img.as_ref(), 10, 10, true, &config);
        assert_eq!(out.width(), 10);
        assert_eq!(out.height(), 10);

        for px in out.pixels() {
            assert!((px.r as i16 - 128).unsigned_abs() <= 2, "R off: {}", px.r);
            assert!((px.a as i16 - 255).unsigned_abs() <= 1, "A off: {}", px.a);
        }
    }
}

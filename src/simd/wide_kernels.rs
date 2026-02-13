//! Portable SIMD kernels using the `wide` crate.
//!
//! These implementations work across NEON (AArch64) and WASM SIMD128,
//! with scalar fallback on other architectures. All functions are
//! `#[inline(always)]` so they inline into the archmage-dispatched callers.

use wide::{f32x4, i16x8, i32x4, i32x8, u8x16};

use crate::weights::{F32WeightTable, I16_PRECISION, I16WeightTable};

// ============================================================================
// f32 path
// ============================================================================

/// Horizontal f32 convolution — dispatch by channel count.
#[inline(always)]
pub(super) fn filter_h_row_f32(
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_4ch(input, output, weights),
        3 => filter_h_3ch(input, output, weights),
        _ => filter_h_generic(input, output, weights, channels),
    }
}

/// 4-channel horizontal f32 convolution using f32x4 FMA.
#[inline(always)]
fn filter_h_4ch(input: &[f32], output: &mut [f32], weights: &F32WeightTable) {
    let (in_pixels, _) = input.as_chunks::<4>();
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let mut acc = f32x4::ZERO;
        for (t, &weight) in w.iter().enumerate() {
            let pixel = f32x4::new(in_pixels[left + t]);
            acc = pixel.mul_add(f32x4::splat(weight), acc);
        }
        out_pixels[out_x] = acc.to_array();
    }
}

/// 3-channel horizontal f32 convolution (scalar, auto-vectorized).
#[inline(always)]
fn filter_h_3ch(input: &[f32], output: &mut [f32], weights: &F32WeightTable) {
    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * 3;

        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;

        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * 3;
            acc0 += input[in_offset] * weight;
            acc1 += input[in_offset + 1] * weight;
            acc2 += input[in_offset + 2] * weight;
        }

        output[out_offset] = acc0;
        output[out_offset + 1] = acc1;
        output[out_offset + 2] = acc2;
    }
}

/// Generic-channel horizontal f32 convolution (scalar).
#[inline(always)]
fn filter_h_generic(input: &[f32], output: &mut [f32], weights: &F32WeightTable, channels: usize) {
    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * channels;

        for c in 0..channels {
            output[out_offset + c] = 0.0;
        }

        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * channels;
            for c in 0..channels {
                output[out_offset + c] += input[in_offset + c] * weight;
            }
        }
    }
}

/// Vertical f32 convolution using f32x4 FMA.
#[inline(always)]
pub(super) fn filter_v_row_f32(rows: &[&[f32]], output: &mut [f32], weights: &[f32]) {
    let width = output.len();
    let chunks4 = width / 4;
    let base4 = chunks4 * 4;

    for chunk_idx in 0..chunks4 {
        let base = chunk_idx * 4;
        let mut acc = f32x4::ZERO;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let src = f32x4::new([row[base], row[base + 1], row[base + 2], row[base + 3]]);
            acc = src.mul_add(f32x4::splat(weight), acc);
        }
        output[base..base + 4].copy_from_slice(&acc.to_array());
    }

    // Scalar tail
    for x in base4..width {
        let mut sum = 0.0f32;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            sum += row[x] * weight;
        }
        output[x] = sum;
    }
}

/// Convert u8 → f32 (divide by 255) using f32x4.
#[inline(always)]
pub(super) fn u8_to_f32_row(input: &[u8], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = f32x4::splat(1.0 / 255.0);
    let chunks4 = input.len() / 4;
    let base4 = chunks4 * 4;

    for i in 0..chunks4 {
        let base = i * 4;
        let iv = i32x4::new([
            input[base] as i32,
            input[base + 1] as i32,
            input[base + 2] as i32,
            input[base + 3] as i32,
        ]);
        let fv = f32x4::from_i32x4(iv) * scale;
        output[base..base + 4].copy_from_slice(&fv.to_array());
    }

    for i in base4..input.len() {
        output[i] = input[i] as f32 * (1.0 / 255.0);
    }
}

/// Convert f32 → u8 (multiply by 255, round, clamp) using f32x4.
#[inline(always)]
pub(super) fn f32_to_u8_row(input: &[f32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = f32x4::splat(255.0);
    let half = f32x4::HALF;
    let zero = f32x4::ZERO;
    let max = f32x4::splat(255.0);
    let chunks4 = input.len() / 4;
    let base4 = chunks4 * 4;

    for i in 0..chunks4 {
        let base = i * 4;
        let fv = f32x4::new([
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        ]);
        // Match scalar: (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
        let scaled = fv.mul_add(scale, half);
        let clamped = scaled.max(zero).min(max);
        let iv = clamped.fast_trunc_int();
        let arr = iv.to_array();
        output[base] = arr[0] as u8;
        output[base + 1] = arr[1] as u8;
        output[base + 2] = arr[2] as u8;
        output[base + 3] = arr[3] as u8;
    }

    for i in base4..input.len() {
        output[i] = (input[i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Premultiply alpha in-place on RGBA f32 row using f32x4.
#[inline(always)]
pub(super) fn premultiply_alpha_row(row: &mut [f32]) {
    for pixel in row.chunks_exact_mut(4) {
        let a = pixel[3];
        pixel[0] *= a;
        pixel[1] *= a;
        pixel[2] *= a;
    }
}

/// Unpremultiply alpha in-place on RGBA f32 row.
#[inline(always)]
pub(super) fn unpremultiply_alpha_row(row: &mut [f32]) {
    for pixel in row.chunks_exact_mut(4) {
        let a = pixel[3];
        if a > 1.0 / 1024.0 {
            let inv_a = 1.0 / a;
            pixel[0] *= inv_a;
            pixel[1] *= inv_a;
            pixel[2] *= inv_a;
        }
    }
}

// ============================================================================
// Integer u8/i16 path
// ============================================================================

/// Integer horizontal convolution — dispatch by channel count.
#[inline(always)]
pub(super) fn filter_h_u8_i16(
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_u8_i16_4ch(input, output, weights),
        _ => filter_h_u8_i16_generic(input, output, weights, channels),
    }
}

/// 4-channel integer H kernel using i32x4.
///
/// Accumulates 4 channels in parallel per output pixel.
#[inline(always)]
fn filter_h_u8_i16_4ch(input: &[u8], output: &mut [u8], weights: &I16WeightTable) {
    let half = i32x4::splat(1 << (I16_PRECISION - 1));
    let zero = i32x4::splat(0);
    let max = i32x4::splat(255);

    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let mut acc = i32x4::splat(0);

        for (t, &weight) in w.iter().enumerate() {
            let off = (left + t) * 4;
            let pixel = i32x4::new([
                input[off] as i32,
                input[off + 1] as i32,
                input[off + 2] as i32,
                input[off + 3] as i32,
            ]);
            acc += pixel * i32x4::splat(weight as i32);
        }

        let rounded = (acc + half) >> I16_PRECISION;
        let clamped = rounded.max(zero).min(max);
        let arr = clamped.to_array();
        let out_base = out_x * 4;
        output[out_base] = arr[0] as u8;
        output[out_base + 1] = arr[1] as u8;
        output[out_base + 2] = arr[2] as u8;
        output[out_base + 3] = arr[3] as u8;
    }
}

/// Generic-channel integer H kernel (scalar).
#[inline(always)]
fn filter_h_u8_i16_generic(
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_base = out_x * channels;

        for c in 0..channels {
            let mut acc: i32 = 0;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * channels + c] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + c] = rounded.clamp(0, 255) as u8;
        }
    }
}

/// 4-row batch integer H kernel. Calls single-row 4 times.
#[inline(always)]
pub(super) fn filter_h_u8_i16_4rows(
    in0: &[u8],
    in1: &[u8],
    in2: &[u8],
    in3: &[u8],
    out0: &mut [u8],
    out1: &mut [u8],
    out2: &mut [u8],
    out3: &mut [u8],
    weights: &I16WeightTable,
) {
    filter_h_u8_i16_4ch(in0, out0, weights);
    filter_h_u8_i16_4ch(in1, out1, weights);
    filter_h_u8_i16_4ch(in2, out2, weights);
    filter_h_u8_i16_4ch(in3, out3, weights);
}

/// Integer vertical convolution: process 16 bytes at a time using i16x8.
#[inline(always)]
pub(super) fn filter_v_u8_i16(rows: &[&[u8]], output: &mut [u8], weights: &[i16]) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());
    let chunks16 = width / 16;
    let base16 = chunks16 * 16;

    for chunk_idx in 0..chunks16 {
        let x = chunk_idx * 16;
        let mut acc_lo = i32x8::splat(0);
        let mut acc_hi = i32x8::splat(0);

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let bytes: [u8; 16] = row[x..x + 16].try_into().unwrap();
            let src = u8x16::new(bytes);
            let lo = i16x8::from_u8x16_low(src);
            let hi = i16x8::from_u8x16_high(src);
            let wv = i16x8::splat(weight);
            acc_lo += lo.mul_widen(wv);
            acc_hi += hi.mul_widen(wv);
        }

        let half = i32x8::splat(1 << (I16_PRECISION - 1));
        let shifted_lo = (acc_lo + half) >> I16_PRECISION;
        let shifted_hi = (acc_hi + half) >> I16_PRECISION;
        let packed_lo = i16x8::from_i32x8_saturate(shifted_lo);
        let packed_hi = i16x8::from_i32x8_saturate(shifted_hi);
        let result = u8x16::narrow_i16x8(packed_lo, packed_hi);
        output[x..x + 16].copy_from_slice(&result.to_array());
    }

    // Scalar tail
    for x in base16..width {
        let mut acc: i32 = 0;
        for (row, &w) in rows.iter().zip(weights.iter()) {
            acc += row[x] as i32 * w as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        output[x] = rounded.clamp(0, 255) as u8;
    }
}

/// Batch vertical filter: process all output rows from the intermediate buffer.
#[inline(always)]
pub(super) fn filter_v_all_u8_i16(
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &I16WeightTable,
) {
    for out_y in 0..out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w = weights.weights(out_y);
        let out_start = out_y * h_row_len;

        let in_h_i32 = in_h as i32 - 1;
        let chunks16 = h_row_len / 16;
        let base16 = chunks16 * 16;

        for chunk_idx in 0..chunks16 {
            let x = chunk_idx * 16;
            let mut acc_lo = i32x8::splat(0);
            let mut acc_hi = i32x8::splat(0);

            for (t, &weight) in w[..tap_count].iter().enumerate() {
                let in_y = (left + t as i32).clamp(0, in_h_i32) as usize;
                let off = in_y * h_row_len + x;
                let bytes: [u8; 16] = intermediate[off..off + 16].try_into().unwrap();
                let src = u8x16::new(bytes);
                let lo = i16x8::from_u8x16_low(src);
                let hi = i16x8::from_u8x16_high(src);
                let wv = i16x8::splat(weight);
                acc_lo += lo.mul_widen(wv);
                acc_hi += hi.mul_widen(wv);
            }

            let half = i32x8::splat(1 << (I16_PRECISION - 1));
            let shifted_lo = (acc_lo + half) >> I16_PRECISION;
            let shifted_hi = (acc_hi + half) >> I16_PRECISION;
            let packed_lo = i16x8::from_i32x8_saturate(shifted_lo);
            let packed_hi = i16x8::from_i32x8_saturate(shifted_hi);
            let result = u8x16::narrow_i16x8(packed_lo, packed_hi);
            output[out_start + x..out_start + x + 16].copy_from_slice(&result.to_array());
        }

        // Scalar tail
        for x in base16..h_row_len {
            let mut acc: i32 = 0;
            for (t, &weight) in w[..tap_count].iter().enumerate() {
                let in_y = (left + t as i32).clamp(0, in_h_i32) as usize;
                acc += intermediate[in_y * h_row_len + x] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_start + x] = rounded.clamp(0, 255) as u8;
        }
    }
}

// ============================================================================
// Alpha / u8 utilities
// ============================================================================

/// Premultiply alpha on RGBA u8 row: input → output.
#[inline(always)]
pub(super) fn premultiply_u8_row(input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let a = inp[3] as u16;
        out[0] = ((inp[0] as u16 * a + 127) / 255) as u8;
        out[1] = ((inp[1] as u16 * a + 127) / 255) as u8;
        out[2] = ((inp[2] as u16 * a + 127) / 255) as u8;
        out[3] = inp[3];
    }
}

/// Unpremultiply alpha in-place on RGBA u8 row.
#[inline(always)]
pub(super) fn unpremultiply_u8_row(row: &mut [u8]) {
    for pixel in row.chunks_exact_mut(4) {
        let a = pixel[3];
        if a == 0 {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        } else if a < 255 {
            let a16 = a as u16;
            pixel[0] = ((pixel[0] as u16 * 255 + a16 / 2) / a16).min(255) as u8;
            pixel[1] = ((pixel[1] as u16 * 255 + a16 / 2) / a16).min(255) as u8;
            pixel[2] = ((pixel[2] as u16 * 255 + a16 / 2) / a16).min(255) as u8;
        }
    }
}

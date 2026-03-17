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
///
/// Processes 8 pixels (32 floats) per outer loop iteration with 8 f32x4
/// accumulators for maximum ILP and reduced loop overhead.
#[inline(always)]
pub(super) fn filter_v_row_f32(rows: &[&[f32]], output: &mut [f32], weights: &[f32]) {
    // Process 8 pixels (32 floats) at a time with 8 accumulators
    let (out_chunks32, out_remainder32) = output.as_chunks_mut::<32>();

    for (ci, out_chunk) in out_chunks32.iter_mut().enumerate() {
        let base = ci * 32;
        let mut acc0 = f32x4::ZERO;
        let mut acc1 = f32x4::ZERO;
        let mut acc2 = f32x4::ZERO;
        let mut acc3 = f32x4::ZERO;
        let mut acc4 = f32x4::ZERO;
        let mut acc5 = f32x4::ZERO;
        let mut acc6 = f32x4::ZERO;
        let mut acc7 = f32x4::ZERO;

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let wv = f32x4::splat(weight);
            let (row_chunks, _) = row.as_chunks::<4>();
            let ri = base / 4;
            acc0 = f32x4::new(row_chunks[ri]).mul_add(wv, acc0);
            acc1 = f32x4::new(row_chunks[ri + 1]).mul_add(wv, acc1);
            acc2 = f32x4::new(row_chunks[ri + 2]).mul_add(wv, acc2);
            acc3 = f32x4::new(row_chunks[ri + 3]).mul_add(wv, acc3);
            acc4 = f32x4::new(row_chunks[ri + 4]).mul_add(wv, acc4);
            acc5 = f32x4::new(row_chunks[ri + 5]).mul_add(wv, acc5);
            acc6 = f32x4::new(row_chunks[ri + 6]).mul_add(wv, acc6);
            acc7 = f32x4::new(row_chunks[ri + 7]).mul_add(wv, acc7);
        }

        out_chunk[0..4].copy_from_slice(&acc0.to_array());
        out_chunk[4..8].copy_from_slice(&acc1.to_array());
        out_chunk[8..12].copy_from_slice(&acc2.to_array());
        out_chunk[12..16].copy_from_slice(&acc3.to_array());
        out_chunk[16..20].copy_from_slice(&acc4.to_array());
        out_chunk[20..24].copy_from_slice(&acc5.to_array());
        out_chunk[24..28].copy_from_slice(&acc6.to_array());
        out_chunk[28..32].copy_from_slice(&acc7.to_array());
    }

    // Handle remainder: 1 pixel (4 floats) at a time
    let base32 = out_chunks32.len() * 32;
    let (rem_chunks4, rem_tail) = out_remainder32.as_chunks_mut::<4>();

    for (ci, out_chunk) in rem_chunks4.iter_mut().enumerate() {
        let ri = (base32 / 4) + ci;
        let mut acc = f32x4::ZERO;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let (row_chunks, _) = row.as_chunks::<4>();
            let src = f32x4::new(row_chunks[ri]);
            acc = src.mul_add(f32x4::splat(weight), acc);
        }
        *out_chunk = acc.to_array();
    }

    // Scalar tail (0-3 floats)
    let base_scalar = base32 + rem_chunks4.len() * 4;
    for (x, out) in rem_tail.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            sum += row[base_scalar + x] * weight;
        }
        *out = sum;
    }
}

/// Convert u8 → f32 (divide by 255) using f32x4.
#[inline(always)]
pub(super) fn u8_to_f32_row(input: &[u8], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let scale = f32x4::splat(1.0 / 255.0);
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let iv = i32x4::new([inp[0] as i32, inp[1] as i32, inp[2] as i32, inp[3] as i32]);
        let fv = f32x4::from_i32x4(iv) * scale;
        *out = fv.to_array();
    }

    for (i, o) in in_tail.iter().zip(out_tail.iter_mut()) {
        *o = *i as f32 * (1.0 / 255.0);
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
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let fv = f32x4::new(*inp);
        // Match scalar: (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
        let scaled = fv.mul_add(scale, half);
        let clamped = scaled.max(zero).min(max);
        let iv = clamped.fast_trunc_int();
        let arr = iv.to_array();
        *out = [arr[0] as u8, arr[1] as u8, arr[2] as u8, arr[3] as u8];
    }

    for (i, o) in in_tail.iter().zip(out_tail.iter_mut()) {
        *o = (*i * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
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
    let in_h_i32 = in_h as i32 - 1;
    let chunks16 = h_row_len / 16;
    let base16 = chunks16 * 16;

    let mut out_y = 0;
    while out_y < out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w_a = weights.weights(out_y);

        let batch2 = out_y + 1 < out_h
            && weights.left[out_y + 1] == left
            && weights.tap_count(out_y + 1) == tap_count;

        if batch2 {
            let w_b = weights.weights(out_y + 1);
            let out_start_a = out_y * h_row_len;
            let out_start_b = (out_y + 1) * h_row_len;

            for chunk_idx in 0..chunks16 {
                let x = chunk_idx * 16;
                let mut acc_a_lo = i32x8::splat(0);
                let mut acc_a_hi = i32x8::splat(0);
                let mut acc_b_lo = i32x8::splat(0);
                let mut acc_b_hi = i32x8::splat(0);

                for t in 0..tap_count {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    let off = in_y_idx * h_row_len + x;
                    let bytes: [u8; 16] = intermediate[off..off + 16].try_into().unwrap();
                    let src = u8x16::new(bytes);
                    let lo = i16x8::from_u8x16_low(src);
                    let hi = i16x8::from_u8x16_high(src);
                    let wva = i16x8::splat(w_a[t]);
                    let wvb = i16x8::splat(w_b[t]);
                    acc_a_lo += lo.mul_widen(wva);
                    acc_a_hi += hi.mul_widen(wva);
                    acc_b_lo += lo.mul_widen(wvb);
                    acc_b_hi += hi.mul_widen(wvb);
                }

                let half = i32x8::splat(1 << (I16_PRECISION - 1));
                let sa_lo = i16x8::from_i32x8_saturate((acc_a_lo + half) >> I16_PRECISION);
                let sa_hi = i16x8::from_i32x8_saturate((acc_a_hi + half) >> I16_PRECISION);
                output[out_start_a + x..out_start_a + x + 16]
                    .copy_from_slice(&u8x16::narrow_i16x8(sa_lo, sa_hi).to_array());

                let sb_lo = i16x8::from_i32x8_saturate((acc_b_lo + half) >> I16_PRECISION);
                let sb_hi = i16x8::from_i32x8_saturate((acc_b_hi + half) >> I16_PRECISION);
                output[out_start_b + x..out_start_b + x + 16]
                    .copy_from_slice(&u8x16::narrow_i16x8(sb_lo, sb_hi).to_array());
            }

            for x in base16..h_row_len {
                let mut acc_a: i32 = 0;
                let mut acc_b: i32 = 0;
                for t in 0..tap_count {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    let v = intermediate[in_y_idx * h_row_len + x] as i32;
                    acc_a += v * w_a[t] as i32;
                    acc_b += v * w_b[t] as i32;
                }
                output[out_start_a + x] =
                    ((acc_a + (1 << (I16_PRECISION - 1))) >> I16_PRECISION).clamp(0, 255) as u8;
                output[out_start_b + x] =
                    ((acc_b + (1 << (I16_PRECISION - 1))) >> I16_PRECISION).clamp(0, 255) as u8;
            }
            out_y += 2;
        } else {
            let out_start = out_y * h_row_len;

            for chunk_idx in 0..chunks16 {
                let x = chunk_idx * 16;
                let mut acc_lo = i32x8::splat(0);
                let mut acc_hi = i32x8::splat(0);

                for (t, &weight) in w_a[..tap_count].iter().enumerate() {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    let off = in_y_idx * h_row_len + x;
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
                output[out_start + x..out_start + x + 16]
                    .copy_from_slice(&u8x16::narrow_i16x8(packed_lo, packed_hi).to_array());
            }

            for x in base16..h_row_len {
                let mut acc: i32 = 0;
                for (t, &weight) in w_a[..tap_count].iter().enumerate() {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    acc += intermediate[in_y_idx * h_row_len + x] as i32 * weight as i32;
                }
                output[out_start + x] =
                    ((acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION).clamp(0, 255) as u8;
            }
            out_y += 1;
        }
    }
}

// ============================================================================
// Integer i16→i16 path (linear-light i12 values 0-4095)
// ============================================================================

/// Integer horizontal convolution: i16 input → i16 output, via wide SIMD.
/// For linear-light i12 path (values 0-4095).
#[inline(always)]
pub(super) fn filter_h_i16_i16(
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_i16_i16_4ch(input, output, weights),
        _ => filter_h_i16_i16_generic(input, output, weights, channels),
    }
}

/// 4-channel i16 H kernel using i32x4.
#[inline(always)]
fn filter_h_i16_i16_4ch(input: &[i16], output: &mut [i16], weights: &I16WeightTable) {
    let half = i32x4::splat(1 << (I16_PRECISION - 1));

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
        let arr = rounded.to_array();
        let out_base = out_x * 4;
        output[out_base] = arr[0] as i16;
        output[out_base + 1] = arr[1] as i16;
        output[out_base + 2] = arr[2] as i16;
        output[out_base + 3] = arr[3] as i16;
    }
}

/// Generic-channel i16 H kernel (scalar).
#[inline(always)]
fn filter_h_i16_i16_generic(
    input: &[i16],
    output: &mut [i16],
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
            output[out_base + c] = rounded as i16;
        }
    }
}

/// Batch vertical filter: i16 intermediate → i16 output via i32 accumulator.
/// For linear-light i12 path (values 0-4095).
#[inline(always)]
pub(super) fn filter_v_all_i16_i16(
    intermediate: &[i16],
    output: &mut [i16],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &I16WeightTable,
) {
    let in_h_i32 = in_h as i32 - 1;
    let chunks4 = h_row_len / 4;
    let base4 = chunks4 * 4;

    let mut out_y = 0;
    while out_y < out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w_a = weights.weights(out_y);

        let batch2 = out_y + 1 < out_h
            && weights.left[out_y + 1] == left
            && weights.tap_count(out_y + 1) == tap_count;

        if batch2 {
            let w_b = weights.weights(out_y + 1);
            let out_start_a = out_y * h_row_len;
            let out_start_b = (out_y + 1) * h_row_len;

            for chunk_idx in 0..chunks4 {
                let x = chunk_idx * 4;
                let mut acc_a = i32x4::splat(0);
                let mut acc_b = i32x4::splat(0);

                for t in 0..tap_count {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    let off = in_y_idx * h_row_len + x;
                    let src = i32x4::new([
                        intermediate[off] as i32,
                        intermediate[off + 1] as i32,
                        intermediate[off + 2] as i32,
                        intermediate[off + 3] as i32,
                    ]);
                    acc_a += src * i32x4::splat(w_a[t] as i32);
                    acc_b += src * i32x4::splat(w_b[t] as i32);
                }

                let half = i32x4::splat(1 << (I16_PRECISION - 1));

                let ca = ((acc_a + half) >> I16_PRECISION).to_array();
                output[out_start_a + x] = ca[0] as i16;
                output[out_start_a + x + 1] = ca[1] as i16;
                output[out_start_a + x + 2] = ca[2] as i16;
                output[out_start_a + x + 3] = ca[3] as i16;

                let cb = ((acc_b + half) >> I16_PRECISION).to_array();
                output[out_start_b + x] = cb[0] as i16;
                output[out_start_b + x + 1] = cb[1] as i16;
                output[out_start_b + x + 2] = cb[2] as i16;
                output[out_start_b + x + 3] = cb[3] as i16;
            }

            for x in base4..h_row_len {
                let mut acc_a: i32 = 0;
                let mut acc_b: i32 = 0;
                for t in 0..tap_count {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    let v = intermediate[in_y_idx * h_row_len + x] as i32;
                    acc_a += v * w_a[t] as i32;
                    acc_b += v * w_b[t] as i32;
                }
                output[out_start_a + x] =
                    ((acc_a + (1 << (I16_PRECISION - 1))) >> I16_PRECISION) as i16;
                output[out_start_b + x] =
                    ((acc_b + (1 << (I16_PRECISION - 1))) >> I16_PRECISION) as i16;
            }
            out_y += 2;
        } else {
            let out_start = out_y * h_row_len;

            for chunk_idx in 0..chunks4 {
                let x = chunk_idx * 4;
                let mut acc = i32x4::splat(0);

                for (t, &weight) in w_a[..tap_count].iter().enumerate() {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    let off = in_y_idx * h_row_len + x;
                    let src = i32x4::new([
                        intermediate[off] as i32,
                        intermediate[off + 1] as i32,
                        intermediate[off + 2] as i32,
                        intermediate[off + 3] as i32,
                    ]);
                    acc += src * i32x4::splat(weight as i32);
                }

                let half = i32x4::splat(1 << (I16_PRECISION - 1));
                let shifted = (acc + half) >> I16_PRECISION;
                let arr = shifted.to_array();
                output[out_start + x] = arr[0] as i16;
                output[out_start + x + 1] = arr[1] as i16;
                output[out_start + x + 2] = arr[2] as i16;
                output[out_start + x + 3] = arr[3] as i16;
            }

            for x in base4..h_row_len {
                let mut acc: i32 = 0;
                for (t, &weight) in w_a[..tap_count].iter().enumerate() {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    acc += intermediate[in_y_idx * h_row_len + x] as i32 * weight as i32;
                }
                output[out_start + x] =
                    ((acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION) as i16;
            }
            out_y += 1;
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

// ============================================================================
// Streaming single-row V-filter kernels (for StreamingResize i16 paths)
// ============================================================================

/// Streaming V-filter: u8 rows → u8 output via i16 weights.
#[inline(always)]
pub(super) fn filter_v_row_u8_i16(rows: &[&[u8]], output: &mut [u8], weights: &[i16]) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    for x in 0..width {
        let mut acc: i32 = 0;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            acc += row[x] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        output[x] = rounded.clamp(0, 255) as u8;
    }
}

// ============================================================================
// f16 (half-precision) support — scalar-style (no portable SIMD f16 convert)
// ============================================================================

/// Bulk convert f32 → f16 row (scalar, uses software conversion).
#[inline(always)]
pub(super) fn f32_to_f16_row(input: &[f32], output: &mut [u16]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = super::scalar::f32_to_f16_soft(*inp);
    }
}

/// Bulk convert f16 → f32 row (scalar, uses software conversion).
#[inline(always)]
pub(super) fn f16_to_f32_row(input: &[u16], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = super::scalar::f16_to_f32_soft(*inp);
    }
}

/// Horizontal f32 convolution with f16 output — dispatch by channel count.
#[inline(always)]
pub(super) fn filter_h_row_f32_to_f16(
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
    channels: usize,
) {
    let out_width = weights.len();
    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * channels;

        for c in 0..channels {
            let mut acc = 0.0f32;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * channels + c] * weight;
            }
            output[out_offset + c] = super::scalar::f32_to_f16_soft(acc);
        }
    }
}

/// Streaming V-filter: f16 rows → f32 output via f32 weights.
#[inline(always)]
pub(super) fn filter_v_row_f16(rows: &[&[u16]], output: &mut [f32], weights: &[f32]) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    for v in output.iter_mut() {
        *v = 0.0;
    }

    for (row, &weight) in rows.iter().zip(weights.iter()) {
        debug_assert!(row.len() >= width);
        for x in 0..width {
            output[x] += super::scalar::f16_to_f32_soft(row[x]) * weight;
        }
    }
}

/// Batch V-filter: f16 intermediate → f32 output.
#[inline(always)]
pub(super) fn filter_v_all_f16(
    intermediate: &[u16],
    output: &mut [f32],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &F32WeightTable,
) {
    for out_y in 0..out_h {
        let left = weights.left[out_y];
        let tap_count = weights.tap_count(out_y);
        let w = weights.weights(out_y);
        let out_start = out_y * h_row_len;

        for x in 0..h_row_len {
            let mut acc = 0.0f32;
            for (t, &weight) in w[..tap_count].iter().enumerate() {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                acc += super::scalar::f16_to_f32_soft(intermediate[in_y * h_row_len + x]) * weight;
            }
            output[out_start + x] = acc;
        }
    }
}

/// Streaming V-filter: i16 rows → i16 output via i16 weights.
#[inline(always)]
pub(super) fn filter_v_row_i16(rows: &[&[i16]], output: &mut [i16], weights: &[i16]) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    for x in 0..width {
        let mut acc: i32 = 0;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            acc += row[x] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        output[x] = rounded as i16;
    }
}

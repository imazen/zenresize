//! Portable SIMD kernels using magetypes generic types.
//!
//! All public functions use `#[magetypes(neon, wasm128)]` to generate both
//! `_neon` and `_wasm128` variants from a single source. The generic types
//! `f32x4<Token>`, `i32x4<Token>` are resolved per-tier by the macro.
//!
//! The `#[magetypes]` macro also generates `_scalar` variants automatically;
//! these are unused because `scalar.rs` provides the scalar fallbacks for `incant!`.
//!
//! Functions are `#[inline(always)]` so they inline into the archmage-dispatched callers.

use archmage::prelude::*;

use crate::weights::{F32WeightTable, I16_PRECISION, I16WeightTable};

use magetypes::simd::generic::f32x4 as GenericF32x4;
use magetypes::simd::generic::i32x4 as GenericI32x4;

// ============================================================================
// f32 path
// ============================================================================

/// Horizontal f32 convolution — dispatch by channel count.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_h_row_f32_impl(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_4ch(token, input, output, weights),
        3 => filter_h_3ch(input, output, weights),
        _ => filter_h_generic(input, output, weights, channels),
    }
}

/// 4-channel horizontal f32 convolution using f32x4.
///
/// Uses `*` + `+=` instead of `mul_add` because on wasm32 (no FMA) mul_add
/// goes through a branch to `(a * b) + c` anyway. Direct `*` + `+=` gives
/// LLVM a cleaner expression to schedule.
#[inline(always)]
fn filter_h_4ch<T: magetypes::simd::backends::F32x4Backend>(
    token: T,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
) {
    #[allow(non_camel_case_types)]
    type f32x4<U> = GenericF32x4<U>;

    let (in_pixels, _) = input.as_chunks::<4>();
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let mut acc = f32x4::zero(token);
        for (t, &weight) in w.iter().enumerate() {
            acc += f32x4::from_array(token, in_pixels[left + t]) * f32x4::splat(token, weight);
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

/// Vertical f32 convolution using f32x4.
///
/// Processes 8 pixels (32 floats) per outer loop iteration with 8 f32x4
/// accumulators for maximum ILP and reduced loop overhead.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_v_row_f32_impl(
    token: Token,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    #[allow(non_camel_case_types)]
    type f32x4<U> = GenericF32x4<U>;

    // Process 8 pixels (32 floats) at a time with 8 accumulators
    let (out_chunks32, out_remainder32) = output.as_chunks_mut::<32>();

    for (ci, out_chunk) in out_chunks32.iter_mut().enumerate() {
        let base = ci * 32;
        let mut acc0 = f32x4::zero(token);
        let mut acc1 = f32x4::zero(token);
        let mut acc2 = f32x4::zero(token);
        let mut acc3 = f32x4::zero(token);
        let mut acc4 = f32x4::zero(token);
        let mut acc5 = f32x4::zero(token);
        let mut acc6 = f32x4::zero(token);
        let mut acc7 = f32x4::zero(token);

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let wv = f32x4::splat(token, weight);
            let (row_chunks, _) = row.as_chunks::<4>();
            let ri = base / 4;
            acc0 += f32x4::from_array(token, row_chunks[ri]) * wv;
            acc1 += f32x4::from_array(token, row_chunks[ri + 1]) * wv;
            acc2 += f32x4::from_array(token, row_chunks[ri + 2]) * wv;
            acc3 += f32x4::from_array(token, row_chunks[ri + 3]) * wv;
            acc4 += f32x4::from_array(token, row_chunks[ri + 4]) * wv;
            acc5 += f32x4::from_array(token, row_chunks[ri + 5]) * wv;
            acc6 += f32x4::from_array(token, row_chunks[ri + 6]) * wv;
            acc7 += f32x4::from_array(token, row_chunks[ri + 7]) * wv;
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
        let mut acc = f32x4::zero(token);
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let (row_chunks, _) = row.as_chunks::<4>();
            acc += f32x4::from_array(token, row_chunks[ri]) * f32x4::splat(token, weight);
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
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn u8_to_f32_row_impl(token: Token, input: &[u8], output: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x4<U> = GenericF32x4<U>;
    #[allow(non_camel_case_types)]
    type i32x4<U> = GenericI32x4<U>;

    debug_assert_eq!(input.len(), output.len());
    let scale = f32x4::splat(token, 1.0 / 255.0);
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let iv = i32x4::from_array(
            token,
            [inp[0] as i32, inp[1] as i32, inp[2] as i32, inp[3] as i32],
        );
        let fv = f32x4::from_i32(token, iv) * scale;
        *out = fv.to_array();
    }

    for (i, o) in in_tail.iter().zip(out_tail.iter_mut()) {
        *o = *i as f32 * (1.0 / 255.0);
    }
}

/// Convert f32 → u8 (multiply by 255, round, clamp) using f32x4.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn f32_to_u8_row_impl(token: Token, input: &[f32], output: &mut [u8]) {
    #[allow(non_camel_case_types)]
    type f32x4<U> = GenericF32x4<U>;

    debug_assert_eq!(input.len(), output.len());
    let scale = f32x4::splat(token, 255.0);
    let half = f32x4::splat(token, 0.5);
    let zero = f32x4::zero(token);
    let max = f32x4::splat(token, 255.0);
    let (in_chunks, in_tail) = input.as_chunks::<4>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<4>();

    for (inp, out) in in_chunks.iter().zip(out_chunks.iter_mut()) {
        let fv = f32x4::from_array(token, *inp);
        // Match scalar: (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
        let scaled = fv * scale + half;
        let clamped = scaled.max(zero).min(max);
        let iv = clamped.to_i32();
        let arr = iv.to_array();
        *out = [arr[0] as u8, arr[1] as u8, arr[2] as u8, arr[3] as u8];
    }

    for (i, o) in in_tail.iter().zip(out_tail.iter_mut()) {
        *o = (*i * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Premultiply alpha in-place on RGBA f32 row using f32x4.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn premultiply_alpha_row_impl(_token: Token, row: &mut [f32]) {
    for pixel in row.chunks_exact_mut(4) {
        let a = pixel[3];
        pixel[0] *= a;
        pixel[1] *= a;
        pixel[2] *= a;
    }
}

/// Unpremultiply alpha in-place on RGBA f32 row.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn unpremultiply_alpha_row_impl(_token: Token, row: &mut [f32]) {
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

/// Load 4 contiguous u8 values as an i32x4 using a single 32-bit load.
///
/// Instead of 4 scalar byte loads + 4 lane inserts, this does one `u32` load
/// followed by shift+mask extraction. LLVM can lower this to a single
/// `v128.load32_zero` + extend on wasm128, or `ldr s` + `uxtl` on NEON.
#[inline(always)]
fn load_u8x4_as_i32x4<T: magetypes::simd::backends::I32x4Backend>(
    token: T,
    slice: &[u8],
    offset: usize,
) -> GenericI32x4<T> {
    // Single 32-bit load of 4 contiguous bytes
    let bytes: [u8; 4] = [
        slice[offset],
        slice[offset + 1],
        slice[offset + 2],
        slice[offset + 3],
    ];
    let word = u32::from_le_bytes(bytes);
    GenericI32x4::from_array(
        token,
        [
            (word & 0xFF) as i32,
            ((word >> 8) & 0xFF) as i32,
            ((word >> 16) & 0xFF) as i32,
            (word >> 24) as i32,
        ],
    )
}

/// Load 4 contiguous i16 values as an i32x4 using a contiguous slice access.
///
/// Constructs the array from a slice, enabling LLVM to see the contiguous load
/// pattern and potentially emit a single 64-bit load + widening instruction.
#[inline(always)]
fn load_i16x4_as_i32x4<T: magetypes::simd::backends::I32x4Backend>(
    token: T,
    slice: &[i16],
    offset: usize,
) -> GenericI32x4<T> {
    GenericI32x4::from_array(
        token,
        [
            slice[offset] as i32,
            slice[offset + 1] as i32,
            slice[offset + 2] as i32,
            slice[offset + 3] as i32,
        ],
    )
}

/// Integer horizontal convolution — dispatch by channel count.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_h_u8_i16_impl(
    token: Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_u8_i16_4ch(token, input, output, weights),
        _ => filter_h_u8_i16_generic(input, output, weights, channels),
    }
}

/// 4-channel integer H kernel using i32x4.
///
/// Accumulates 4 channels in parallel per output pixel.
/// Uses `weights_padded` for fixed-length iteration so LLVM can unroll,
/// and batch-loads 4 bytes via a single u32 load.
#[inline(always)]
fn filter_h_u8_i16_4ch<T: magetypes::simd::backends::I32x4Backend>(
    token: T,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
) {
    #[allow(non_camel_case_types)]
    type i32x4<U> = GenericI32x4<U>;

    let half = i32x4::splat(token, 1 << (I16_PRECISION - 1));
    let zero = i32x4::splat(token, 0);
    let max = i32x4::splat(token, 255);
    let max_taps = weights.max_taps;

    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights_padded(out_x);
        let mut acc = i32x4::zero(token);

        for t in 0..max_taps {
            let off = (left + t) * 4;
            let pixel = load_u8x4_as_i32x4(token, input, off);
            acc += pixel * w[t] as i32;
        }

        let rounded = (acc + half).shr_arithmetic_const::<{ I16_PRECISION }>();
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

/// Integer horizontal convolution: u8 input → i16 output (unclamped).
/// Preserves Lanczos ringing without [0,255] clamping.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_h_u8_to_i16_impl(
    token: Token,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_u8_to_i16_4ch(token, input, output, weights),
        _ => filter_h_u8_to_i16_generic(input, output, weights, channels),
    }
}

/// 4-channel u8→i16 H kernel using i32x4.
///
/// Same accumulation as filter_h_u8_i16_4ch but output stores i16 instead of
/// clamping to [0,255] u8. Uses batch u32 loads and fixed-length padded weights.
#[inline(always)]
fn filter_h_u8_to_i16_4ch<T: magetypes::simd::backends::I32x4Backend>(
    token: T,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
) {
    #[allow(non_camel_case_types)]
    type i32x4<U> = GenericI32x4<U>;

    let half = i32x4::splat(token, 1 << (I16_PRECISION - 1));
    let max_taps = weights.max_taps;

    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights_padded(out_x);
        let mut acc = i32x4::zero(token);

        for t in 0..max_taps {
            let off = (left + t) * 4;
            let pixel = load_u8x4_as_i32x4(token, input, off);
            acc += pixel * w[t] as i32;
        }

        let rounded = (acc + half).shr_arithmetic_const::<{ I16_PRECISION }>();
        let arr = rounded.to_array();
        let out_base = out_x * 4;
        output[out_base] = arr[0] as i16;
        output[out_base + 1] = arr[1] as i16;
        output[out_base + 2] = arr[2] as i16;
        output[out_base + 3] = arr[3] as i16;
    }
}

/// Generic-channel u8→i16 H kernel (scalar).
#[inline(always)]
fn filter_h_u8_to_i16_generic(
    input: &[u8],
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

/// 4-row batch u8→i16 H kernel. Calls single-row 4 times.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_h_u8_to_i16_4rows_impl(
    token: Token,
    in0: &[u8],
    in1: &[u8],
    in2: &[u8],
    in3: &[u8],
    out0: &mut [i16],
    out1: &mut [i16],
    out2: &mut [i16],
    out3: &mut [i16],
    weights: &I16WeightTable,
) {
    filter_h_u8_to_i16_4ch(token, in0, out0, weights);
    filter_h_u8_to_i16_4ch(token, in1, out1, weights);
    filter_h_u8_to_i16_4ch(token, in2, out2, weights);
    filter_h_u8_to_i16_4ch(token, in3, out3, weights);
}

/// 4-row batch integer H kernel. Calls single-row 4 times.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_h_u8_i16_4rows_impl(
    token: Token,
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
    filter_h_u8_i16_4ch(token, in0, out0, weights);
    filter_h_u8_i16_4ch(token, in1, out1, weights);
    filter_h_u8_i16_4ch(token, in2, out2, weights);
    filter_h_u8_i16_4ch(token, in3, out3, weights);
}

// ============================================================================
// Batch vertical filter: u8 intermediate → u8 output
// ============================================================================

/// Process one 16-byte column chunk of the V-filter using 4 i32x4 accumulators.
///
/// Each 16-byte chunk is split into 4 groups of 4 bytes, each widened to i32x4
/// via a single u32 load and shift extraction, then multiplied by the broadcast
/// weight. This structure maps directly to NEON's `vmull_s16`/`vmlal_s16` and
/// WASM128's `i32x4.extmul_low_i16x8`.
#[inline(always)]
fn v_filter_chunk_16<T: magetypes::simd::backends::I32x4Backend>(
    token: T,
    intermediate: &[u8],
    h_row_len: usize,
    x: usize,
    left: i32,
    in_h_i32: i32,
    tap_count: usize,
    weights: &[i16],
    acc: &mut [GenericI32x4<T>; 4],
) {
    for t in 0..tap_count {
        let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
        let off = in_y_idx * h_row_len + x;
        let w = weights[t] as i32;

        // Each group: single u32 load + shift extraction instead of 4 scalar loads
        acc[0] += load_u8x4_as_i32x4(token, intermediate, off) * w;
        acc[1] += load_u8x4_as_i32x4(token, intermediate, off + 4) * w;
        acc[2] += load_u8x4_as_i32x4(token, intermediate, off + 8) * w;
        acc[3] += load_u8x4_as_i32x4(token, intermediate, off + 12) * w;
    }
}

/// Pack 4 i32x4 accumulators to 16 u8 values with rounding, shift, and clamp.
#[inline(always)]
fn pack_i32x4_to_u8_16<T: magetypes::simd::backends::I32x4Backend>(
    token: T,
    acc: &[GenericI32x4<T>; 4],
    out: &mut [u8],
) {
    let half = GenericI32x4::splat(token, 1 << (I16_PRECISION - 1));
    let zero = GenericI32x4::splat(token, 0);
    let max = GenericI32x4::splat(token, 255);

    for (i, a) in acc.iter().enumerate() {
        let rounded = (*a + half).shr_arithmetic_const::<{ I16_PRECISION }>();
        let clamped = rounded.max(zero).min(max);
        let arr = clamped.to_array();
        let base = i * 4;
        out[base] = arr[0] as u8;
        out[base + 1] = arr[1] as u8;
        out[base + 2] = arr[2] as u8;
        out[base + 3] = arr[3] as u8;
    }
}

/// Batch vertical filter: process all output rows from the intermediate buffer.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_v_all_u8_i16_impl(
    token: Token,
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
                let mut acc_a = [
                    GenericI32x4::zero(token),
                    GenericI32x4::zero(token),
                    GenericI32x4::zero(token),
                    GenericI32x4::zero(token),
                ];
                let mut acc_b = [
                    GenericI32x4::zero(token),
                    GenericI32x4::zero(token),
                    GenericI32x4::zero(token),
                    GenericI32x4::zero(token),
                ];

                v_filter_chunk_16(
                    token,
                    intermediate,
                    h_row_len,
                    x,
                    left,
                    in_h_i32,
                    tap_count,
                    w_a,
                    &mut acc_a,
                );
                v_filter_chunk_16(
                    token,
                    intermediate,
                    h_row_len,
                    x,
                    left,
                    in_h_i32,
                    tap_count,
                    w_b,
                    &mut acc_b,
                );

                pack_i32x4_to_u8_16(
                    token,
                    &acc_a,
                    &mut output[out_start_a + x..out_start_a + x + 16],
                );
                pack_i32x4_to_u8_16(
                    token,
                    &acc_b,
                    &mut output[out_start_b + x..out_start_b + x + 16],
                );
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
                let mut acc = [
                    GenericI32x4::zero(token),
                    GenericI32x4::zero(token),
                    GenericI32x4::zero(token),
                    GenericI32x4::zero(token),
                ];

                v_filter_chunk_16(
                    token,
                    intermediate,
                    h_row_len,
                    x,
                    left,
                    in_h_i32,
                    tap_count,
                    w_a,
                    &mut acc,
                );

                pack_i32x4_to_u8_16(token, &acc, &mut output[out_start + x..out_start + x + 16]);
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

/// Tiled batch vertical filter: u8 intermediate → u8 output with column tiling.
///
/// `tile_chunks` is the number of 16-byte chunks per tile. Processes column tiles
/// to improve L1 cache reuse across consecutive output rows that share overlapping
/// input row windows.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_v_all_u8_i16_tiled_impl(
    token: Token,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &I16WeightTable,
    tile_chunks: usize,
) {
    let in_h_i32 = in_h as i32 - 1;
    let chunks16 = h_row_len / 16;
    let base16 = chunks16 * 16;

    // Process column tiles of 16-byte chunks.
    for tile_ci in (0..chunks16).step_by(tile_chunks) {
        let tile_ci_end = (tile_ci + tile_chunks).min(chunks16);

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

                for chunk_idx in tile_ci..tile_ci_end {
                    let x = chunk_idx * 16;
                    let mut acc_a = [
                        GenericI32x4::zero(token),
                        GenericI32x4::zero(token),
                        GenericI32x4::zero(token),
                        GenericI32x4::zero(token),
                    ];
                    let mut acc_b = [
                        GenericI32x4::zero(token),
                        GenericI32x4::zero(token),
                        GenericI32x4::zero(token),
                        GenericI32x4::zero(token),
                    ];

                    v_filter_chunk_16(
                        token,
                        intermediate,
                        h_row_len,
                        x,
                        left,
                        in_h_i32,
                        tap_count,
                        w_a,
                        &mut acc_a,
                    );
                    v_filter_chunk_16(
                        token,
                        intermediate,
                        h_row_len,
                        x,
                        left,
                        in_h_i32,
                        tap_count,
                        w_b,
                        &mut acc_b,
                    );

                    pack_i32x4_to_u8_16(
                        token,
                        &acc_a,
                        &mut output[out_start_a + x..out_start_a + x + 16],
                    );
                    pack_i32x4_to_u8_16(
                        token,
                        &acc_b,
                        &mut output[out_start_b + x..out_start_b + x + 16],
                    );
                }
                out_y += 2;
            } else {
                let out_start = out_y * h_row_len;

                for chunk_idx in tile_ci..tile_ci_end {
                    let x = chunk_idx * 16;
                    let mut acc = [
                        GenericI32x4::zero(token),
                        GenericI32x4::zero(token),
                        GenericI32x4::zero(token),
                        GenericI32x4::zero(token),
                    ];

                    v_filter_chunk_16(
                        token,
                        intermediate,
                        h_row_len,
                        x,
                        left,
                        in_h_i32,
                        tap_count,
                        w_a,
                        &mut acc,
                    );

                    pack_i32x4_to_u8_16(
                        token,
                        &acc,
                        &mut output[out_start + x..out_start + x + 16],
                    );
                }
                out_y += 1;
            }
        }
    }

    // Process remainder columns (< 16 bytes) that don't fit in any tile.
    if base16 < h_row_len {
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
}

// ============================================================================
// Integer i16→i16 path (linear-light i12 values 0-4095)
// ============================================================================

/// Integer horizontal convolution: i16 input → i16 output, via SIMD.
/// For linear-light i12 path (values 0-4095).
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_h_i16_i16_impl(
    token: Token,
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_i16_i16_4ch(token, input, output, weights),
        _ => filter_h_i16_i16_generic(input, output, weights, channels),
    }
}

/// 4-channel i16 H kernel using i32x4.
///
/// Uses `from_slice` for contiguous i16 loads and fixed-length padded weights
/// for LLVM unrolling.
#[inline(always)]
fn filter_h_i16_i16_4ch<T: magetypes::simd::backends::I32x4Backend>(
    token: T,
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
) {
    #[allow(non_camel_case_types)]
    type i32x4<U> = GenericI32x4<U>;

    let half = i32x4::splat(token, 1 << (I16_PRECISION - 1));
    let max_taps = weights.max_taps;

    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights_padded(out_x);
        let mut acc = i32x4::zero(token);

        for t in 0..max_taps {
            let off = (left + t) * 4;
            let pixel = i32x4::from_array(
                token,
                [
                    input[off] as i32,
                    input[off + 1] as i32,
                    input[off + 2] as i32,
                    input[off + 3] as i32,
                ],
            );
            acc += pixel * w[t] as i32;
        }

        let rounded = (acc + half).shr_arithmetic_const::<{ I16_PRECISION }>();
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
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_v_all_i16_i16_impl(
    token: Token,
    intermediate: &[i16],
    output: &mut [i16],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &I16WeightTable,
) {
    #[allow(non_camel_case_types)]
    type i32x4<U> = GenericI32x4<U>;

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
                let mut acc_a = i32x4::zero(token);
                let mut acc_b = i32x4::zero(token);

                for t in 0..tap_count {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    let off = in_y_idx * h_row_len + x;
                    let src = load_i16x4_as_i32x4(token, intermediate, off);
                    acc_a += src * w_a[t] as i32;
                    acc_b += src * w_b[t] as i32;
                }

                let half = i32x4::splat(token, 1 << (I16_PRECISION - 1));

                let ca = ((acc_a + half).shr_arithmetic_const::<{ I16_PRECISION }>()).to_array();
                output[out_start_a + x] = ca[0] as i16;
                output[out_start_a + x + 1] = ca[1] as i16;
                output[out_start_a + x + 2] = ca[2] as i16;
                output[out_start_a + x + 3] = ca[3] as i16;

                let cb = ((acc_b + half).shr_arithmetic_const::<{ I16_PRECISION }>()).to_array();
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
                let mut acc = i32x4::zero(token);

                for (t, &weight) in w_a[..tap_count].iter().enumerate() {
                    let in_y_idx = (left + t as i32).clamp(0, in_h_i32) as usize;
                    let off = in_y_idx * h_row_len + x;
                    let src = load_i16x4_as_i32x4(token, intermediate, off);
                    acc += src * weight as i32;
                }

                let half = i32x4::splat(token, 1 << (I16_PRECISION - 1));
                let shifted = (acc + half).shr_arithmetic_const::<{ I16_PRECISION }>();
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
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn premultiply_u8_row_impl(_token: Token, input: &[u8], output: &mut [u8]) {
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
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn unpremultiply_u8_row_impl(_token: Token, row: &mut [u8]) {
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
///
/// Uses i32x4 for the bulk of the work (4 bytes at a time), with scalar tail.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_v_row_u8_i16_impl(
    token: Token,
    rows: &[&[u8]],
    output: &mut [u8],
    weights: &[i16],
) {
    #[allow(non_camel_case_types)]
    type i32x4<U> = GenericI32x4<U>;

    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    let half = i32x4::splat(token, 1 << (I16_PRECISION - 1));
    let zero = i32x4::splat(token, 0);
    let max = i32x4::splat(token, 255);

    // Process 4 bytes at a time via i32x4
    let chunks4 = width / 4;
    let base4 = chunks4 * 4;

    for chunk_idx in 0..chunks4 {
        let x = chunk_idx * 4;
        let mut acc = i32x4::zero(token);

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let pixel = load_u8x4_as_i32x4(token, row, x);
            acc += pixel * weight as i32;
        }

        let rounded = (acc + half).shr_arithmetic_const::<{ I16_PRECISION }>();
        let clamped = rounded.max(zero).min(max);
        let arr = clamped.to_array();
        output[x] = arr[0] as u8;
        output[x + 1] = arr[1] as u8;
        output[x + 2] = arr[2] as u8;
        output[x + 3] = arr[3] as u8;
    }

    // Scalar tail
    for x in base4..width {
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
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn f32_to_f16_row_impl(_token: Token, input: &[f32], output: &mut [u16]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = super::scalar::f32_to_f16_soft(*inp);
    }
}

/// Bulk convert f16 → f32 row (scalar, uses software conversion).
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn f16_to_f32_row_impl(_token: Token, input: &[u16], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = super::scalar::f16_to_f32_soft(*inp);
    }
}

/// Horizontal f32 convolution with f16 output — dispatch by channel count.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_h_row_f32_to_f16_impl(
    _token: Token,
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
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_v_row_f16_impl(
    _token: Token,
    rows: &[&[u16]],
    output: &mut [f32],
    weights: &[f32],
) {
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
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_v_all_f16_impl(
    _token: Token,
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
///
/// Uses i32x4 for the bulk of the work (4 i16 values at a time), with scalar tail.
#[magetypes(neon, wasm128)]
#[inline(always)]
pub(super) fn filter_v_row_i16_impl(
    token: Token,
    rows: &[&[i16]],
    output: &mut [i16],
    weights: &[i16],
) {
    #[allow(non_camel_case_types)]
    type i32x4<U> = GenericI32x4<U>;

    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    let half = i32x4::splat(token, 1 << (I16_PRECISION - 1));

    // Process 4 i16 values at a time via i32x4
    let chunks4 = width / 4;
    let base4 = chunks4 * 4;

    for chunk_idx in 0..chunks4 {
        let x = chunk_idx * 4;
        let mut acc = i32x4::zero(token);

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            let src = load_i16x4_as_i32x4(token, row, x);
            acc += src * weight as i32;
        }

        let shifted = (acc + half).shr_arithmetic_const::<{ I16_PRECISION }>();
        let arr = shifted.to_array();
        output[x] = arr[0] as i16;
        output[x + 1] = arr[1] as i16;
        output[x + 2] = arr[2] as i16;
        output[x + 3] = arr[3] as i16;
    }

    // Scalar tail
    for x in base4..width {
        let mut acc: i32 = 0;
        for (row, &weight) in rows.iter().zip(weights.iter()) {
            acc += row[x] as i32 * weight as i32;
        }
        let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
        output[x] = rounded as i16;
    }
}

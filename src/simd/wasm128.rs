//! WASM SIMD128 convolution kernels.
//!
//! Most implementations delegate to `wide_kernels` (magetypes portable SIMD).
//! The 4-channel i16 horizontal filter uses native wasm128 intrinsics via
//! `#[rite(import_intrinsics)]` for widening loads and `i32x4_dot_i16x8`
//! (paired multiply-add), matching the x86 AVX2 `madd_epi16` strategy.

use crate::proven::sub;
use crate::weights::{F32WeightTable, I16_PRECISION, I16WeightTable};
use archmage::Wasm128Token;

#[archmage::arcane]
pub(crate) fn filter_h_row_f32_wasm128(
    _token: Wasm128Token,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_row_f32_impl_wasm128(_token, input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_f32_wasm128(
    _token: Wasm128Token,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    super::wide_kernels::filter_v_row_f32_impl_wasm128(_token, rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn u8_to_f32_row_wasm128(_token: Wasm128Token, input: &[u8], output: &mut [f32]) {
    super::wide_kernels::u8_to_f32_row_impl_wasm128(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn f32_to_u8_row_wasm128(_token: Wasm128Token, input: &[f32], output: &mut [u8]) {
    super::wide_kernels::f32_to_u8_row_impl_wasm128(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn premultiply_alpha_row_wasm128(_token: Wasm128Token, row: &mut [f32]) {
    super::wide_kernels::premultiply_alpha_row_impl_wasm128(_token, row)
}

#[archmage::arcane]
pub(crate) fn unpremultiply_alpha_row_wasm128(_token: Wasm128Token, row: &mut [f32]) {
    super::wide_kernels::unpremultiply_alpha_row_impl_wasm128(_token, row)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_wasm128(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
    channels: usize,
) {
    match channels {
        4 => filter_h_u8_i16_4ch_intrinsic(_token, input, output, weights),
        _ => super::wide_kernels::filter_h_u8_i16_impl_wasm128(
            _token, input, output, weights, channels,
        ),
    }
}

/// 4-channel i16 horizontal convolution using native wasm128 intrinsics.
///
/// Uses `i32x4_dot_i16x8` (paired multiply-add) with the pre-expanded
/// weight table. Each group of 4 taps is processed as two 128-bit dot
/// products — the lower 8 i16s handle taps 0,1 and the upper 8 handle
/// taps 2,3 — mirroring the AVX2 `madd_epi16` strategy at half width.
///
/// Widening loads (`u16x8_extend_low/high_u8x16`) replace the scalar
/// byte-at-a-time loads in the magetypes fallback.
#[allow(clippy::needless_range_loop)]
#[archmage::rite(import_intrinsics)]
fn filter_h_u8_i16_4ch_intrinsic(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [u8],
    weights: &I16WeightTable,
) {
    let out_width = weights.len();
    let groups4 = weights.groups4;
    let in_pixels = input.len() / 4;

    // Need at least 4 pixels for the 16-byte loads
    if in_pixels < 4 {
        filter_h_u8_i16_4ch_scalar(input, output, weights);
        return;
    }

    // Check if the input slice has enough padding for all SIMD reads.
    // The +3 accounts for the 16-byte (4-pixel) SIMD load width: the last
    // group's load at pixel max_left + (groups4-1)*4 reads 4 pixels forward.
    let max_left = weights.left.iter().map(|&l| l as usize).max().unwrap_or(0);
    let has_full_padding = max_left + groups4 * 4 + 3 < in_pixels;

    // Shuffle mask: interleave channels from two consecutive RGBA pixels.
    // Input  i16 lanes: [p0r, p0g, p0b, p0a, p1r, p1g, p1b, p1a]
    // Output i16 lanes: [p0r, p1r, p0g, p1g, p0b, p1b, p0a, p1a]
    // Byte indices:      [0,1, 8,9, 2,3, 10,11, 4,5, 12,13, 6,7, 14,15]

    let half = i32x4_splat(1 << (I16_PRECISION - 1));
    let zero = i32x4_splat(0);
    let max_val = i32x4_splat(255);

    // Weight data: contiguous stride-16 i16 array per group.
    let ew_all = weights.expanded_4ch_all();
    let (ew_chunks, _) = ew_all.as_chunks::<8>();

    // Output: contiguous stride-4 u8 pixels.
    let (out_pixels, _) = output.as_chunks_mut::<4>();

    if has_full_padding {
        // Fast path: all pixels have enough input padding for SIMD loads
        for out_x in 0..out_width {
            let left = weights.left[out_x] as usize;
            let byte_start = left * 4;

            // Pre-slice input window and weight window
            let input_window = sub(input, byte_start..byte_start + groups4 * 16);
            let (in_chunks, _) = input_window.as_chunks::<16>();
            let ew_base = out_x * groups4 * 2; // 2 chunks of 8 i16 per group

            let mut acc = i32x4_splat(0);

            for (g, chunk) in in_chunks.iter().enumerate() {
                // Load 16 bytes (4 pixels × 4 channels)
                let raw = v128_load(chunk);

                // Widen low 8 bytes to i16: pixels for taps 0,1
                let lo_i16 = u16x8_extend_low_u8x16(raw);
                // Shuffle to interleave channels: [p0r,p1r, p0g,p1g, p0b,p1b, p0a,p1a]
                let lo_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        lo_i16, lo_i16,
                    );
                // Load lower weight half: [w0,w1, w0,w1, w0,w1, w0,w1]
                let w_lo = v128_load(&ew_chunks[ew_base + g * 2]);
                // Paired multiply-add: R=p0r*w0+p1r*w1, G=p0g*w0+p1g*w1, ...
                acc = i32x4_add(acc, i32x4_dot_i16x8(lo_shuffled, w_lo));

                // Widen high 8 bytes to i16: pixels for taps 2,3
                let hi_i16 = u16x8_extend_high_u8x16(raw);
                // Same channel interleave shuffle
                let hi_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        hi_i16, hi_i16,
                    );
                // Load upper weight half: [w2,w3, w2,w3, w2,w3, w2,w3]
                let w_hi = v128_load(&ew_chunks[ew_base + g * 2 + 1]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(hi_shuffled, w_hi));
            }

            // Round, shift right, clamp to [0, 255], store
            let rounded = i32x4_add(acc, half);
            let shifted = i32x4_shr(rounded, I16_PRECISION as u32);
            let clamped = i32x4_min(i32x4_max(shifted, zero), max_val);

            // Pack i32x4 → i16x8 → u8x16, extract low 4 bytes
            let packed16 = i16x8_narrow_i32x4(clamped, zero);
            let packed8 = u8x16_narrow_i16x8(packed16, packed16);
            let pixel_val = u32x4_extract_lane::<0>(packed8);
            out_pixels[out_x].copy_from_slice(&pixel_val.to_ne_bytes());
        }
    } else {
        // Edge-aware path: per-pixel bounds check
        let safe_end = (0..out_width)
            .position(|x| (weights.left[x] as usize) + groups4 * 4 > in_pixels)
            .unwrap_or(out_width);

        // SIMD for interior pixels
        for out_x in 0..safe_end {
            let left = weights.left[out_x] as usize;
            let byte_start = left * 4;

            let input_window = &input[byte_start..byte_start + groups4 * 16];
            let (in_chunks, _) = input_window.as_chunks::<16>();
            let ew_base = out_x * groups4 * 2;

            let mut acc = i32x4_splat(0);

            for (g, chunk) in in_chunks.iter().enumerate() {
                let raw = v128_load(chunk);

                let lo_i16 = u16x8_extend_low_u8x16(raw);
                let lo_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        lo_i16, lo_i16,
                    );
                let w_lo = v128_load(&ew_chunks[ew_base + g * 2]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(lo_shuffled, w_lo));

                let hi_i16 = u16x8_extend_high_u8x16(raw);
                let hi_shuffled =
                    i8x16_shuffle::<0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15>(
                        hi_i16, hi_i16,
                    );
                let w_hi = v128_load(&ew_chunks[ew_base + g * 2 + 1]);
                acc = i32x4_add(acc, i32x4_dot_i16x8(hi_shuffled, w_hi));
            }

            let rounded = i32x4_add(acc, half);
            let shifted = i32x4_shr(rounded, I16_PRECISION as u32);
            let clamped = i32x4_min(i32x4_max(shifted, zero), max_val);

            let packed16 = i16x8_narrow_i32x4(clamped, zero);
            let packed8 = u8x16_narrow_i16x8(packed16, packed16);
            let pixel_val = u32x4_extract_lane::<0>(packed8);
            out_pixels[out_x].copy_from_slice(&pixel_val.to_ne_bytes());
        }

        // Scalar for edge pixels
        for out_x in safe_end..out_width {
            let left = weights.left[out_x] as usize;
            let w = weights.weights(out_x);
            let out_base = out_x * 4;
            for c in 0..4 {
                let mut acc: i32 = 0;
                for (t, &weight) in w.iter().enumerate() {
                    acc += input[(left + t) * 4 + c] as i32 * weight as i32;
                }
                let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
                output[out_base + c] = rounded.clamp(0, 255) as u8;
            }
        }
    }
}

/// Scalar fallback for 4ch when input is too small for SIMD.
#[inline(always)]
fn filter_h_u8_i16_4ch_scalar(input: &[u8], output: &mut [u8], weights: &I16WeightTable) {
    for out_x in 0..weights.len() {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_base = out_x * 4;
        for c in 0..4 {
            let mut acc: i32 = 0;
            for (t, &weight) in w.iter().enumerate() {
                acc += input[(left + t) * 4 + c] as i32 * weight as i32;
            }
            let rounded = (acc + (1 << (I16_PRECISION - 1))) >> I16_PRECISION;
            output[out_base + c] = rounded.clamp(0, 255) as u8;
        }
    }
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_to_i16_wasm128(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_u8_to_i16_impl_wasm128(_token, input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_to_i16_4rows_wasm128(
    _token: Wasm128Token,
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
    super::wide_kernels::filter_h_u8_to_i16_4rows_impl_wasm128(
        _token, in0, in1, in2, in3, out0, out1, out2, out3, weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_h_u8_i16_4rows_wasm128(
    _token: Wasm128Token,
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
    super::wide_kernels::filter_h_u8_i16_4rows_impl_wasm128(
        _token, in0, in1, in2, in3, out0, out1, out2, out3, weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_v_all_u8_i16_wasm128(
    _token: Wasm128Token,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    super::wide_kernels::filter_v_all_u8_i16_impl_wasm128(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_v_all_u8_i16_tiled_wasm128(
    _token: Wasm128Token,
    intermediate: &[u8],
    output: &mut [u8],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
    tile_chunks: usize,
) {
    super::wide_kernels::filter_v_all_u8_i16_tiled_impl_wasm128(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
        tile_chunks,
    )
}

#[archmage::arcane]
pub(crate) fn premultiply_u8_row_wasm128(_token: Wasm128Token, input: &[u8], output: &mut [u8]) {
    super::wide_kernels::premultiply_u8_row_impl_wasm128(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn unpremultiply_u8_row_wasm128(_token: Wasm128Token, row: &mut [u8]) {
    super::wide_kernels::unpremultiply_u8_row_impl_wasm128(_token, row)
}

#[archmage::arcane]
pub(crate) fn filter_h_i16_i16_wasm128(
    _token: Wasm128Token,
    input: &[i16],
    output: &mut [i16],
    weights: &I16WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_i16_i16_impl_wasm128(_token, input, output, weights, channels)
}

#[archmage::arcane]
pub(crate) fn filter_v_all_i16_i16_wasm128(
    _token: Wasm128Token,
    intermediate: &[i16],
    output: &mut [i16],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &crate::weights::I16WeightTable,
) {
    super::wide_kernels::filter_v_all_i16_i16_impl_wasm128(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
    )
}

#[archmage::arcane]
pub(crate) fn filter_v_row_u8_i16_wasm128(
    _token: Wasm128Token,
    rows: &[&[u8]],
    output: &mut [u8],
    weights: &[i16],
) {
    super::wide_kernels::filter_v_row_u8_i16_impl_wasm128(_token, rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn filter_v_row_i16_wasm128(
    _token: Wasm128Token,
    rows: &[&[i16]],
    output: &mut [i16],
    weights: &[i16],
) {
    super::wide_kernels::filter_v_row_i16_impl_wasm128(_token, rows, output, weights)
}

// f16 kernels — delegate to wide_kernels

#[archmage::arcane]
pub(crate) fn f32_to_f16_row_wasm128(_token: Wasm128Token, input: &[f32], output: &mut [u16]) {
    super::wide_kernels::f32_to_f16_row_impl_wasm128(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn f16_to_f32_row_wasm128(_token: Wasm128Token, input: &[u16], output: &mut [f32]) {
    super::wide_kernels::f16_to_f32_row_impl_wasm128(_token, input, output)
}

#[archmage::arcane]
pub(crate) fn filter_h_row_f32_to_f16_wasm128(
    _token: Wasm128Token,
    input: &[f32],
    output: &mut [u16],
    weights: &F32WeightTable,
    channels: usize,
) {
    super::wide_kernels::filter_h_row_f32_to_f16_impl_wasm128(
        _token, input, output, weights, channels,
    )
}

#[archmage::arcane]
pub(crate) fn filter_v_row_f16_wasm128(
    _token: Wasm128Token,
    rows: &[&[u16]],
    output: &mut [f32],
    weights: &[f32],
) {
    super::wide_kernels::filter_v_row_f16_impl_wasm128(_token, rows, output, weights)
}

#[archmage::arcane]
pub(crate) fn filter_v_all_f16_wasm128(
    _token: Wasm128Token,
    intermediate: &[u16],
    output: &mut [f32],
    h_row_len: usize,
    in_h: usize,
    out_h: usize,
    weights: &F32WeightTable,
) {
    super::wide_kernels::filter_v_all_f16_impl_wasm128(
        _token,
        intermediate,
        output,
        h_row_len,
        in_h,
        out_h,
        weights,
    )
}

// Transfer function batch processors — wrap linear-srgb rites via closures.

use magetypes::simd::generic::f32x4;

macro_rules! tf_wasm {
    ($name_wasm:ident, $rite_fn:path, $scalar_fn:path) => {
        #[archmage::arcane]
        pub(crate) fn $name_wasm(
            _token: Wasm128Token,
            row: &mut [f32],
            channels: usize,
            has_alpha: bool,
        ) {
            super::tf_portable::tf_row_inplace(
                _token,
                row,
                channels,
                has_alpha,
                |t, v: f32x4<Wasm128Token>| f32x4::from_array(t, $rite_fn(t, v.to_array())),
                $scalar_fn,
            );
        }
    };
}

tf_wasm!(
    srgb_to_linear_row_wasm128,
    linear_srgb::tokens::x4::tf_srgb_to_linear_wasm128,
    linear_srgb::tf::srgb_to_linear
);
tf_wasm!(
    srgb_from_linear_row_wasm128,
    linear_srgb::tokens::x4::tf_linear_to_srgb_wasm128,
    linear_srgb::tf::linear_to_srgb
);
tf_wasm!(
    bt709_to_linear_row_wasm128,
    linear_srgb::tokens::x4::bt709_to_linear_wasm128,
    linear_srgb::tf::bt709_to_linear
);
tf_wasm!(
    bt709_from_linear_row_wasm128,
    linear_srgb::tokens::x4::linear_to_bt709_wasm128,
    linear_srgb::tf::linear_to_bt709
);
tf_wasm!(
    pq_to_linear_row_wasm128,
    linear_srgb::tokens::x4::pq_to_linear_wasm128,
    linear_srgb::tf::pq_to_linear
);
tf_wasm!(
    pq_from_linear_row_wasm128,
    linear_srgb::tokens::x4::linear_to_pq_wasm128,
    linear_srgb::tf::linear_to_pq
);
tf_wasm!(
    hlg_to_linear_row_wasm128,
    linear_srgb::tokens::x4::hlg_to_linear_wasm128,
    linear_srgb::tf::hlg_to_linear
);
tf_wasm!(
    hlg_from_linear_row_wasm128,
    linear_srgb::tokens::x4::linear_to_hlg_wasm128,
    linear_srgb::tf::linear_to_hlg
);

#[archmage::arcane]
pub(crate) fn srgb_u8_to_linear_f32_wasm128(
    _token: Wasm128Token,
    input: &[u8],
    output: &mut [f32],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::srgb_u8_to_linear_f32_impl(input, output, channels, has_alpha);
}

#[archmage::arcane]
pub(crate) fn linear_f32_to_srgb_u8_wasm128(
    _token: Wasm128Token,
    input: &[f32],
    output: &mut [u8],
    channels: usize,
    has_alpha: bool,
) {
    crate::color::linear_f32_to_srgb_u8_impl(input, output, channels, has_alpha);
}

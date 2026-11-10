//! Fused Q14 intermediate narrowing and straight-alpha output.
use archmage::prelude::*;
use magetypes::simd::generic::{i32x4 as I32x4, i32x8 as I32x8};

#[cfg(target_arch = "x86_64")]
use encode_vector_v3 as encode_output_v3;
#[cfg(target_arch = "wasm32")]
use encode_vector_wasm128 as encode_output_wasm128;

#[inline(always)]
pub(crate) fn encode_i16_output(input: &[i16], output: &mut [u8], premul: bool) {
    if !premul {
        for (&v, dst) in input.iter().zip(output.iter_mut()) {
            *dst = v.clamp(0, 255) as u8;
        }
        return;
    }
    archmage::incant!(encode_output(input, output), [v3, neon, wasm128, scalar]);
}

// One predictable branch per four pixels, rather than per alpha lane. This
// keeps the opaque-row bypass cheap without scanning the rest of a mixed row.
#[inline(always)]
fn opaque_row(input: &[i16]) -> bool {
    let (blocks, tail) = input.as_chunks::<16>();
    for p in blocks {
        if ((p[3] ^ 255) | (p[7] ^ 255) | (p[11] ^ 255) | (p[15] ^ 255)) != 0 {
            return false;
        }
    }
    tail.as_chunks::<4>().0.iter().all(|p| p[3] == 255)
}

// Explicit vector division keeps all four RGBA lanes together; a plain loop
// with its alpha lane overwritten was scalarized into a 2-lane + scalar divide.
// Integer numerators are exact in f32. Below the 255 saturation boundary, the
// quotient's rounding error is less than 1/(2*a), for every positive i16 a.
// Truncation therefore agrees with integer floor. Test the entire i16 domain.
#[magetypes(v3, wasm128)]
#[allow(dead_code)] // The macro also emits a scalar vector model; dispatch uses the integer scalar body.
fn encode_vector(token: Token, input: &[i16], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % 4, 0);
    // Short-circuit for partial alpha, one alpha-only scan for opaque rows.
    // This replaces the old opaque unpremultiply traversal; the following
    // contiguous narrowing loop can pack whole vectors without pixel branches.
    if opaque_row(input) {
        for (&v, dst) in input.iter().zip(output.iter_mut()) {
            *dst = v.clamp(0, 255) as u8;
        }
        return;
    }
    let zero = I32x8::<Token>::zero_t(token);
    let one = I32x8::<Token>::splat_t(token, 1);
    let k255 = I32x8::<Token>::splat_t(token, 255);
    let alpha_lanes = I32x8::<Token>::from_array_t(token, [0, 0, 0, -1, 0, 0, 0, -1]);
    let (chunks, tail) = input.as_chunks::<16>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<16>();
    for (src, dst) in chunks.iter().zip(out_chunks) {
        let mut converted = [zero; 2];
        for (half, value) in converted.iter_mut().enumerate() {
            let base = half * 8;
            let pair: &[i16; 8] = src[base..base + 8].try_into().unwrap();
            let colors = I32x8::<Token>::from_array_t(token, pair.map(i32::from)).max(zero);
            let lanes = colors.to_array();
            let a = lanes[3];
            let b = lanes[7];
            let alpha = I32x8::<Token>::from_array_t(token, [a, a, a, a, b, b, b, b]);
            let numerator = colors * k255 + alpha.shr_arithmetic_const::<1>();
            let quotient = numerator.to_f32() / alpha.max(one).to_f32();
            *value =
                I32x8::<Token>::blend(alpha_lanes, alpha, quotient.to_i32()) & alpha.simd_gt(zero);
        }
        let packed16 = converted[0].narrow_saturating_i16(converted[1]);
        let packed8 = packed16.narrow_saturating_u8(packed16).to_array();
        dst.copy_from_slice(&packed8[..16]);
    }
    let zero4 = I32x4::<Token>::zero_t(token);
    let k255_4 = I32x4::<Token>::splat_t(token, 255);
    let alpha_lane = I32x4::<Token>::from_array_t(token, [0, 0, 0, -1]);
    for (src, dst) in tail
        .as_chunks::<4>()
        .0
        .iter()
        .zip(out_tail.as_chunks_mut::<4>().0)
    {
        let a = i32::from(src[3]);
        if a <= 0 {
            *dst = [0; 4];
            continue;
        }
        let colors = I32x4::<Token>::from_array_t(token, src.map(i32::from)).max(zero4);
        let alpha = I32x4::<Token>::splat_t(token, a);
        let numerator = colors * k255_4 + alpha.shr_arithmetic_const::<1>();
        let quotient = (numerator.to_f32() / alpha.to_f32()).to_i32();
        let values = I32x4::<Token>::blend(alpha_lane, alpha, quotient);
        let packed16 = values.narrow_saturating_i16(values);
        let packed8 = packed16.narrow_saturating_u8(packed16).to_array();
        dst.copy_from_slice(&packed8[..4]);
    }
}

// Scalar has integer division and no vector packing overhead. Keep its cheap
// zero/opaque cases fused into the same input/output traversal as correction.
fn encode_output_scalar(_token: ScalarToken, input: &[i16], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % 4, 0);
    if opaque_row(input) {
        for (&value, dst) in input.iter().zip(output.iter_mut()) {
            *dst = value.clamp(0, 255) as u8;
        }
        return;
    }
    for (src, dst) in input
        .as_chunks::<4>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<4>().0)
    {
        encode_pixel_integer(src, dst);
    }
}

#[inline(always)]
fn encode_pixel_integer(src: &[i16; 4], dst: &mut [u8; 4]) {
    let a = i32::from(src[3]);
    if a == 255 {
        for i in 0..4 {
            dst[i] = src[i].clamp(0, 255) as u8;
        }
    } else if a <= 0 {
        *dst = [0; 4];
    } else {
        // One 24-bit reciprocal for all three channels. floor(2^24/a)
        // underestimates 1/a by <2^-24; num <2^23, so the quotient estimate
        // is at most one below floor(num/a). One integer comparison corrects
        // it exactly. Use u32 division and widened products, without floating
        // reciprocal error or three variable divisions per pixel.
        let reciprocal = (1u32 << 24) / a as u32;
        for c in 0..3 {
            let num = (i32::from(src[c]).max(0) * 255 + a / 2) as u32;
            let q = ((u64::from(num) * u64::from(reciprocal)) >> 24) as u32;
            let exact = q + u32::from((q + 1) * a as u32 <= num);
            dst[c] = exact.min(255) as u8;
        }
        dst[3] = a.min(255) as u8;
    }
}

// NEON's structure loads deinterleave 16 RGBA pixels into channel vectors.
// This differs from AVX2's interleaved packing: doing an extra alpha divide
// and per-pixel shuffles regressed the first ARM benchmark. Only the three
// color planes need division; alpha narrows directly from its signed source.
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
fn encode_output_neon(token: NeonToken, input: &[i16], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len() % 4, 0);
    // Match the existing NEON algorithm's scalar short-row path. Routing a
    // single pixel through the wide vector dispatcher adds setup and a call.
    if input.len() < 64 {
        for (src, dst) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<4>().0)
        {
            encode_pixel_integer(src, dst);
        }
        return;
    }
    if input.as_chunks::<4>().0.iter().all(|p| p[3] == 255) {
        for (&v, dst) in input.iter().zip(output.iter_mut()) {
            *dst = v.clamp(0, 255) as u8;
        }
        return;
    }
    let (blocks, tail) = input.as_chunks::<64>();
    let (out_blocks, out_tail) = output.as_chunks_mut::<64>();
    for (src, dst) in blocks.iter().zip(out_blocks) {
        let lo = vld4q_s16(src[..32].try_into().unwrap());
        let hi = vld4q_s16(src[32..].try_into().unwrap());
        let a0 = widen_positive_neon(token, lo.3);
        let a1 = widen_positive_neon(token, hi.3);
        let low = [lo.0, lo.1, lo.2];
        let high = [hi.0, hi.1, hi.2];
        let mut planes = [vdupq_n_u8(0); 3];
        for c in 0..3 {
            let c0 = widen_positive_neon(token, low[c]);
            let c1 = widen_positive_neon(token, high[c]);
            planes[c] = vcombine_u8(
                narrow_two_groups_neon(
                    token,
                    divide_group_neon(token, c0[0], a0[0]),
                    divide_group_neon(token, c0[1], a0[1]),
                ),
                narrow_two_groups_neon(
                    token,
                    divide_group_neon(token, c1[0], a1[0]),
                    divide_group_neon(token, c1[1], a1[1]),
                ),
            );
        }
        let alpha = vcombine_u8(vqmovun_s16(lo.3), vqmovun_s16(hi.3));
        vst4q_u8(dst, uint8x16x4_t(planes[0], planes[1], planes[2], alpha));
    }
    if !tail.is_empty() {
        for (src, dst) in tail
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out_tail.as_chunks_mut::<4>().0)
        {
            encode_pixel_integer(src, dst);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn widen_positive_neon(_token: NeonToken, values: int16x8_t) -> [uint32x4_t; 2] {
    let positive = vmaxq_s16(values, vdupq_n_s16(0));
    [
        vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(positive))),
        vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(positive))),
    ]
}

#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn divide_group_neon(_token: NeonToken, color: uint32x4_t, alpha: uint32x4_t) -> uint32x4_t {
    let numerator = vaddq_u32(vmulq_u32(color, vdupq_n_u32(255)), vshrq_n_u32::<1>(alpha));
    let denominator = vmaxq_u32(alpha, vdupq_n_u32(1));
    let result = vcvtq_u32_f32(vdivq_f32(
        vcvtq_f32_u32(numerator),
        vcvtq_f32_u32(denominator),
    ));
    vbslq_u32(vceqq_u32(alpha, vdupq_n_u32(0)), vdupq_n_u32(0), result)
}

#[cfg(target_arch = "aarch64")]
#[archmage::rite]
fn narrow_two_groups_neon(_token: NeonToken, low: uint32x4_t, high: uint32x4_t) -> uint8x8_t {
    vqmovn_u16(vcombine_u16(vqmovn_u32(low), vqmovn_u32(high)))
}

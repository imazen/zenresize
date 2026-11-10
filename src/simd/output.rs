//! Fused Q14 intermediate narrowing and straight-alpha output.
use archmage::prelude::*;
use magetypes::simd::generic::f32x4 as F32x4;

pub(crate) fn encode_i16_output(input: &[i16], output: &mut [u8], premul: bool) {
    archmage::incant!(
        encode_output(input, output, premul),
        [v3, neon, wasm128, scalar]
    );
}

// Explicit vector division keeps all four RGBA lanes together; a plain loop
// with its alpha lane overwritten was scalarized into a 2-lane + scalar divide.
// Integer numerators are exact in f32. Below the 255 saturation boundary, the
// quotient's rounding error is less than 1/(2*a), for every positive i16 a.
// Truncation therefore agrees with integer floor. Test the entire i16 domain.
#[magetypes(v3, neon, wasm128, scalar)]
fn encode_output(token: Token, input: &[i16], output: &mut [u8], premul: bool) {
    assert_eq!(input.len(), output.len());
    assert_eq!(input.len() % 4, 0);
    // Short-circuit for partial alpha, one alpha-only scan for opaque rows.
    // This replaces the old opaque unpremultiply traversal; the following
    // contiguous narrowing loop can pack whole vectors without pixel branches.
    if !premul || input.as_chunks::<4>().0.iter().all(|p| p[3] == 255) {
        for (&v, dst) in input.iter().zip(output.iter_mut()) {
            *dst = v.clamp(0, 255) as u8;
        }
        return;
    }
    let zero = F32x4::<Token>::zero(token);
    let maximum = F32x4::<Token>::splat(token, 255.0);
    for (src, dst) in input
        .as_chunks::<4>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<4>().0)
    {
        let a = i32::from(src[3]).max(0);
        if a == 255 {
            for i in 0..4 {
                dst[i] = src[i].clamp(0, 255) as u8;
            }
        } else if a == 0 {
            *dst = [0; 4];
        } else {
            let numerator = F32x4::<Token>::from_array(
                token,
                src.map(|v| (i32::from(v).max(0) * 255 + a / 2) as f32),
            );
            let divisor = F32x4::<Token>::splat(token, a as f32);
            let values = (numerator / divisor)
                .max(zero)
                .min(maximum)
                .to_i32()
                .to_array();
            *dst = values.map(|v| v as u8);
            dst[3] = a.min(255) as u8;
        }
    }
}

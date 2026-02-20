//! Working-space types and conversion traits.
//!
//! The resize pipeline converts input elements to a working type for filtering,
//! then converts back to output elements. The [`WorkingType`] trait abstracts
//! over the intermediate pixel type and provides access to the appropriate
//! filter kernels.
//!
//! Three working types are provided:
//!
//! - [`U8Work`] — u8 intermediate, i16 weight tables, batch H/V. For gamma-space
//!   resize without linearization. Fastest for sRGB u8 when linear-light isn't needed.
//!
//! - [`I16Work`] — i16 intermediate, i16 weight tables, batch V. For SDR
//!   linear-light resize at 12-bit precision. Only valid when premultiply is NOT
//!   needed (Rgbx, RgbaPremul, or opaque Rgba).
//!
//! - [`F32Work`] — f32 intermediate, f32 weight tables, per-row V. For all
//!   other cases: HDR, compositing, premultiply, u16, mixed-format pipelines.
//!   The most flexible path.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::filter::InterpolationDetails;
use crate::pixel::Element;
use crate::simd;
use crate::transfer::{NoTransfer, Srgb, TransferFunction};
use crate::weights::{F32WeightTable, I16WeightTable};

// =============================================================================
// WorkingType trait
// =============================================================================

/// Abstraction over the intermediate pixel type during filtering.
///
/// Each working type owns the weight table type, filter kernel dispatch,
/// and intermediate buffer allocation.
pub(crate) trait WorkingType: Send + Sync + 'static {
    /// The intermediate element type (u8, i16, or f32).
    type Value: Copy + Default + Send + Sync;

    /// Weight table type for this working space.
    type Weights: Send + Sync;

    /// Build weight tables for the given dimensions and filter.
    fn build_weights(in_size: u32, out_size: u32, filter: &InterpolationDetails) -> Self::Weights;

    /// Horizontal filter: one input row → one output row.
    fn filter_h_row(
        input: &[Self::Value],
        output: &mut [Self::Value],
        weights: &Self::Weights,
        channels: usize,
    );

    /// Vertical filter: per-row. Used when compositing needs per-row injection.
    ///
    /// `rows` are references to horizontal-filtered intermediate rows.
    /// `weights` are the per-tap filter weights for this output row.
    fn filter_v_row(
        rows: &[&[Self::Value]],
        output: &mut [Self::Value],
        weights: &[f32],
    );

    /// Batch vertical filter: all output rows at once from the intermediate buffer.
    ///
    /// Faster when no per-row compositing is needed. Default calls `filter_v_row`
    /// in a loop. U8Work and I16Work override with optimized batch kernels.
    fn filter_v_batch(
        intermediate: &[Self::Value],
        output: &mut [Self::Value],
        h_row_len: usize,
        in_h: usize,
        out_h: usize,
        weights: &Self::Weights,
    );

    /// Whether values can represent HDR range (> 1.0 linear).
    const HDR_CAPABLE: bool;
}

// =============================================================================
// U8Work — u8 intermediate (gamma-space resize)
// =============================================================================

/// u8 working type: filter directly in gamma space.
///
/// Uses i16 weight tables for integer convolution with u8 pixel values.
/// Fastest path for sRGB u8 when linear-light isn't required.
/// Supports 4-row batch horizontal pass for RGBA without premultiply.
pub struct U8Work;

impl WorkingType for U8Work {
    type Value = u8;
    type Weights = I16WeightTable;

    fn build_weights(in_size: u32, out_size: u32, filter: &InterpolationDetails) -> Self::Weights {
        I16WeightTable::new(in_size, out_size, filter)
    }

    fn filter_h_row(
        input: &[u8],
        output: &mut [u8],
        weights: &I16WeightTable,
        channels: usize,
    ) {
        simd::filter_h_u8_i16(input, output, weights, channels);
    }

    fn filter_v_row(
        _rows: &[&[u8]],
        _output: &mut [u8],
        _weights: &[f32],
    ) {
        // U8Work v-row not used — batch only. Panic if called.
        unimplemented!("U8Work uses batch vertical pass only");
    }

    fn filter_v_batch(
        intermediate: &[u8],
        output: &mut [u8],
        h_row_len: usize,
        in_h: usize,
        out_h: usize,
        weights: &I16WeightTable,
    ) {
        simd::filter_v_all_u8_i16(intermediate, output, h_row_len, in_h, out_h, weights);
    }

    const HDR_CAPABLE: bool = false;
}

impl U8Work {
    /// 4-row batch horizontal filter (RGBA only, no premultiply).
    ///
    /// Shares weight computation across rows for better throughput.
    /// Not part of the WorkingType trait — only U8Work has this.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn filter_h_4rows(
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
        simd::filter_h_u8_i16_4rows(in0, in1, in2, in3, out0, out1, out2, out3, weights);
    }
}

// =============================================================================
// I16Work — i16 intermediate (linear-light SDR at 12-bit)
// =============================================================================

/// i16 working type: filter in linear-light at 12-bit precision.
///
/// Values are in the range [0, 4095] (i12 linear). Only suitable for SDR
/// transfer functions where linear values stay in [0, 1]. Using this with
/// HDR TFs (PQ, HLG) will silently clip values above 4095.
///
/// Does NOT support premultiply/unpremultiply. Only valid with layouts
/// that don't need premultiply (Rgbx, RgbaPremul, Gray, Rgb, Cmyk).
pub struct I16Work;

impl WorkingType for I16Work {
    type Value = i16;
    type Weights = I16WeightTable;

    fn build_weights(in_size: u32, out_size: u32, filter: &InterpolationDetails) -> Self::Weights {
        I16WeightTable::new(in_size, out_size, filter)
    }

    fn filter_h_row(
        input: &[i16],
        output: &mut [i16],
        weights: &I16WeightTable,
        channels: usize,
    ) {
        simd::filter_h_i16_i16(input, output, weights, channels);
    }

    fn filter_v_row(
        _rows: &[&[i16]],
        _output: &mut [i16],
        _weights: &[f32],
    ) {
        unimplemented!("I16Work uses batch vertical pass only");
    }

    fn filter_v_batch(
        intermediate: &[i16],
        output: &mut [i16],
        h_row_len: usize,
        in_h: usize,
        out_h: usize,
        weights: &I16WeightTable,
    ) {
        simd::filter_v_all_i16_i16(intermediate, output, h_row_len, in_h, out_h, weights);
    }

    const HDR_CAPABLE: bool = false;
}

// =============================================================================
// F32Work — f32 intermediate (universal)
// =============================================================================

/// f32 working type: filter in premultiplied linear-light f32.
///
/// The most flexible path. Supports HDR (values > 1.0), negative wide-gamut
/// values, compositing, and arbitrary transfer functions. Values are NOT
/// clamped during filtering — final quantization is the only clamping point.
///
/// Uses per-row vertical pass, enabling per-row compositing injection.
pub struct F32Work;

impl WorkingType for F32Work {
    type Value = f32;
    type Weights = F32WeightTable;

    fn build_weights(in_size: u32, out_size: u32, filter: &InterpolationDetails) -> Self::Weights {
        F32WeightTable::new(in_size, out_size, filter)
    }

    fn filter_h_row(
        input: &[f32],
        output: &mut [f32],
        weights: &F32WeightTable,
        channels: usize,
    ) {
        simd::filter_h_row_f32(input, output, weights, channels);
    }

    fn filter_v_row(
        rows: &[&[f32]],
        output: &mut [f32],
        weights: &[f32],
    ) {
        simd::filter_v_row_f32(rows, output, weights);
    }

    fn filter_v_batch(
        intermediate: &[f32],
        output: &mut [f32],
        h_row_len: usize,
        in_h: usize,
        out_h: usize,
        weights: &F32WeightTable,
    ) {
        // Default: loop over per-row V-pass
        let max_taps = weights.max_taps;
        let mut row_ptrs: Vec<&[f32]> = Vec::with_capacity(max_taps);

        for out_y in 0..out_h {
            let left = weights.left[out_y];
            let tap_count = weights.tap_count(out_y);
            let w = weights.weights(out_y);

            row_ptrs.clear();
            for t in 0..tap_count {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                let start = in_y * h_row_len;
                row_ptrs.push(&intermediate[start..start + h_row_len]);
            }

            let out_start = out_y * h_row_len;
            simd::filter_v_row_f32(
                &row_ptrs,
                &mut output[out_start..out_start + h_row_len],
                w,
            );
        }
    }

    const HDR_CAPABLE: bool = true;
}

// =============================================================================
// IntoWorking — convert input elements to working-space values
// =============================================================================

/// Convert encoded input elements to working-space values.
///
/// The conversion includes type conversion, transfer function (linearization),
/// and optionally premultiply. The specific combination of (Element, WorkingType,
/// TransferFunction) determines what operations are performed.
///
/// Not all combinations are valid — only those with an `IntoWorking` impl compile.
pub(crate) trait IntoWorking<W: WorkingType, TF: TransferFunction>: Element {
    fn convert(
        src: &[Self],
        dst: &mut [W::Value],
        tf: &TF,
        luts: &TF::Luts,
        channels: usize,
        has_alpha: bool,
        premul: bool,
    );
}

// --- u8 → U8Work (no linearization, gamma-space) ---

impl IntoWorking<U8Work, NoTransfer> for u8 {
    fn convert(
        src: &[u8],
        dst: &mut [u8],
        _tf: &NoTransfer,
        _luts: &(),
        _channels: usize,
        _has_alpha: bool,
        premul: bool,
    ) {
        if premul {
            simd::premultiply_u8_row(src, dst);
        } else {
            dst[..src.len()].copy_from_slice(src);
        }
    }
}

// --- u8 → I16Work (linearize to i12) ---

impl IntoWorking<I16Work, Srgb> for u8 {
    fn convert(
        src: &[u8],
        dst: &mut [i16],
        tf: &Srgb,
        luts: &(),
        _channels: usize,
        _has_alpha: bool,
        _premul: bool,
    ) {
        // I16Work doesn't support premul — caller must ensure layout doesn't need it
        tf.u8_to_linear_i12(src, dst, luts);
    }
}

impl IntoWorking<I16Work, NoTransfer> for u8 {
    fn convert(
        src: &[u8],
        dst: &mut [i16],
        tf: &NoTransfer,
        luts: &(),
        _channels: usize,
        _has_alpha: bool,
        _premul: bool,
    ) {
        tf.u8_to_linear_i12(src, dst, luts);
    }
}

// --- u8 → F32Work (linearize to f32, any TF) ---

impl<TF: TransferFunction> IntoWorking<F32Work, TF> for u8 {
    fn convert(
        src: &[u8],
        dst: &mut [f32],
        tf: &TF,
        luts: &TF::Luts,
        channels: usize,
        has_alpha: bool,
        premul: bool,
    ) {
        tf.u8_to_linear_f32(src, dst, luts, channels, has_alpha, premul);
    }
}

// --- u16 → F32Work (linearize to f32, any TF) ---

impl<TF: TransferFunction> IntoWorking<F32Work, TF> for u16 {
    fn convert(
        src: &[u16],
        dst: &mut [f32],
        tf: &TF,
        luts: &TF::Luts,
        channels: usize,
        has_alpha: bool,
        premul: bool,
    ) {
        tf.u16_to_linear_f32(src, dst, luts, channels, has_alpha, premul);
    }
}

// --- f32 → F32Work (in-place linearization if TF is not identity) ---

impl<TF: TransferFunction> IntoWorking<F32Work, TF> for f32 {
    fn convert(
        src: &[f32],
        dst: &mut [f32],
        tf: &TF,
        _luts: &TF::Luts,
        channels: usize,
        has_alpha: bool,
        premul: bool,
    ) {
        dst[..src.len()].copy_from_slice(src);
        tf.f32_to_linear_inplace(dst, channels, has_alpha, premul);
    }
}

// =============================================================================
// FromWorking — convert working-space values to output elements
// =============================================================================

/// Convert working-space values to encoded output elements.
///
/// The conversion includes unpremultiply (if needed), transfer function
/// (delinearization), and type conversion with quantization/clamping.
pub(crate) trait FromWorking<W: WorkingType, TF: TransferFunction>: Element {
    fn convert(
        src: &[W::Value],
        dst: &mut [Self],
        tf: &TF,
        luts: &TF::Luts,
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    );
}

// --- U8Work → u8 (no delinearization) ---

impl FromWorking<U8Work, NoTransfer> for u8 {
    fn convert(
        src: &[u8],
        dst: &mut [u8],
        _tf: &NoTransfer,
        _luts: &(),
        _channels: usize,
        _has_alpha: bool,
        unpremul: bool,
    ) {
        dst[..src.len()].copy_from_slice(src);
        if unpremul {
            simd::unpremultiply_u8_row(dst);
        }
    }
}

// --- I16Work → u8 (delinearize i12 → u8) ---

impl FromWorking<I16Work, Srgb> for u8 {
    fn convert(
        src: &[i16],
        dst: &mut [u8],
        tf: &Srgb,
        luts: &(),
        _channels: usize,
        _has_alpha: bool,
        _unpremul: bool,
    ) {
        tf.linear_i12_to_u8(src, dst, luts);
    }
}

impl FromWorking<I16Work, NoTransfer> for u8 {
    fn convert(
        src: &[i16],
        dst: &mut [u8],
        tf: &NoTransfer,
        luts: &(),
        _channels: usize,
        _has_alpha: bool,
        _unpremul: bool,
    ) {
        tf.linear_i12_to_u8(src, dst, luts);
    }
}

// --- F32Work → u8 (delinearize + quantize, any TF) ---

impl<TF: TransferFunction> FromWorking<F32Work, TF> for u8 {
    fn convert(
        src: &[f32],
        dst: &mut [u8],
        tf: &TF,
        luts: &TF::Luts,
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    ) {
        tf.linear_f32_to_u8(src, dst, luts, channels, has_alpha, unpremul);
    }
}

// --- F32Work → u16 (delinearize + quantize, any TF) ---

impl<TF: TransferFunction> FromWorking<F32Work, TF> for u16 {
    fn convert(
        src: &[f32],
        dst: &mut [u16],
        tf: &TF,
        luts: &TF::Luts,
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    ) {
        tf.linear_f32_to_u16(src, dst, luts, channels, has_alpha, unpremul);
    }
}

// --- F32Work → f32 (in-place delinearization, no clamp) ---

impl<TF: TransferFunction> FromWorking<F32Work, TF> for f32 {
    fn convert(
        src: &[f32],
        dst: &mut [f32],
        tf: &TF,
        _luts: &TF::Luts,
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    ) {
        dst[..src.len()].copy_from_slice(src);
        tf.linear_to_f32_inplace(dst, channels, has_alpha, unpremul);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Filter;

    #[test]
    fn u8work_builds_i16_weights() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let weights = U8Work::build_weights(100, 50, &filter);
        assert!(weights.max_taps > 0);
    }

    #[test]
    fn i16work_builds_i16_weights() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let weights = I16Work::build_weights(100, 50, &filter);
        assert!(weights.max_taps > 0);
    }

    #[test]
    fn f32work_builds_f32_weights() {
        let filter = InterpolationDetails::create(Filter::Lanczos);
        let weights = F32Work::build_weights(100, 50, &filter);
        assert!(weights.max_taps > 0);
    }

    #[test]
    fn into_working_u8_to_u8work() {
        let tf = NoTransfer;
        let luts = tf.build_luts();
        let src = [128u8, 64, 32, 255];
        let mut dst = [0u8; 4];

        <u8 as IntoWorking<U8Work, NoTransfer>>::convert(
            &src, &mut dst, &tf, &luts, 4, true, false,
        );
        assert_eq!(dst, src);
    }

    #[test]
    fn into_working_u8_to_i16work_srgb() {
        let tf = Srgb;
        let luts = tf.build_luts();
        let src = [128u8, 0, 255, 200];
        let mut dst = [0i16; 4];

        <u8 as IntoWorking<I16Work, Srgb>>::convert(
            &src, &mut dst, &tf, &luts, 4, true, false,
        );

        // Should match the compile-time LUT
        assert_eq!(dst[0], crate::color::SRGB_U8_TO_LINEAR_I12[128]);
        assert_eq!(dst[1], crate::color::SRGB_U8_TO_LINEAR_I12[0]);
        assert_eq!(dst[2], crate::color::SRGB_U8_TO_LINEAR_I12[255]);
    }

    #[test]
    fn into_working_u8_to_f32work_srgb() {
        let tf = Srgb;
        let luts = tf.build_luts();
        let src = [128u8, 64, 32, 200]; // RGBA
        let mut dst = [0.0f32; 4];

        <u8 as IntoWorking<F32Work, Srgb>>::convert(
            &src, &mut dst, &tf, &luts, 4, true, false,
        );

        // R should be linearized (~0.216), alpha should be linear (200/255)
        assert!(dst[0] > 0.2 && dst[0] < 0.3, "R: {}", dst[0]);
        assert!((dst[3] - 200.0 / 255.0).abs() < 0.01, "A: {}", dst[3]);
    }

    #[test]
    fn into_working_u16_to_f32work_srgb() {
        let tf = Srgb;
        let luts = tf.build_luts();
        let src = [32768u16, 0, 65535, 50000];
        let mut dst = [0.0f32; 4];

        <u16 as IntoWorking<F32Work, Srgb>>::convert(
            &src, &mut dst, &tf, &luts, 4, true, false,
        );

        // 32768/65535 ≈ 0.5 encoded → ~0.214 linear (sRGB)
        assert!(dst[0] > 0.2 && dst[0] < 0.25, "R: {}", dst[0]);
        // 65535 → 1.0 encoded → 1.0 linear
        assert!((dst[2] - 1.0).abs() < 0.001, "B: {}", dst[2]);
        // Alpha: linear scale
        assert!((dst[3] - 50000.0 / 65535.0).abs() < 0.001, "A: {}", dst[3]);
    }

    #[test]
    fn from_working_u8work_to_u8() {
        let tf = NoTransfer;
        let luts = tf.build_luts();
        let src = [128u8, 64, 32, 255];
        let mut dst = [0u8; 4];

        <u8 as FromWorking<U8Work, NoTransfer>>::convert(
            &src, &mut dst, &tf, &luts, 4, true, false,
        );
        assert_eq!(dst, src);
    }

    #[test]
    fn from_working_f32work_to_u16_srgb() {
        let tf = Srgb;
        let luts = tf.build_luts();

        // Linear 0.5 → sRGB encoded ~0.735 → u16 ~48163
        let src = [0.5f32, 0.0, 1.0, 0.8];
        let mut dst = [0u16; 4];

        <u16 as FromWorking<F32Work, Srgb>>::convert(
            &src, &mut dst, &tf, &luts, 4, true, false,
        );

        // 0.5 linear → ~0.735 sRGB → ~48163 u16
        assert!(dst[0] > 45000 && dst[0] < 50000, "R: {}", dst[0]);
        // 1.0 linear → 1.0 sRGB → 65535 u16
        assert_eq!(dst[2], 65535, "B: {}", dst[2]);
        // Alpha: linear scale → 0.8 * 65535 ≈ 52428
        assert!((dst[3] as f32 - 52428.0).abs() < 2.0, "A: {}", dst[3]);
    }

    #[test]
    fn roundtrip_u8_f32work_srgb() {
        let tf = Srgb;
        let luts = tf.build_luts();

        let src = [128u8, 64, 32, 200];
        let mut f32_buf = [0.0f32; 4];
        let mut out = [0u8; 4];

        <u8 as IntoWorking<F32Work, Srgb>>::convert(
            &src, &mut f32_buf, &tf, &luts, 4, true, false,
        );
        <u8 as FromWorking<F32Work, Srgb>>::convert(
            &f32_buf, &mut out, &tf, &luts, 4, true, false,
        );

        for i in 0..4 {
            let diff = (src[i] as i16 - out[i] as i16).unsigned_abs();
            assert!(diff <= 1, "ch {}: {} vs {}", i, src[i], out[i]);
        }
    }

    #[test]
    fn roundtrip_u16_f32work_srgb() {
        let tf = Srgb;
        let luts = tf.build_luts();

        let src = [32768u16, 16384, 65535, 50000];
        let mut f32_buf = [0.0f32; 4];
        let mut out = [0u16; 4];

        <u16 as IntoWorking<F32Work, Srgb>>::convert(
            &src, &mut f32_buf, &tf, &luts, 4, true, false,
        );
        <u16 as FromWorking<F32Work, Srgb>>::convert(
            &f32_buf, &mut out, &tf, &luts, 4, true, false,
        );

        for i in 0..4 {
            let diff = (src[i] as i32 - out[i] as i32).unsigned_abs();
            assert!(diff <= 1, "ch {}: {} vs {}", i, src[i], out[i]);
        }
    }

    #[test]
    fn hdr_capability() {
        assert!(!U8Work::HDR_CAPABLE);
        assert!(!I16Work::HDR_CAPABLE);
        assert!(F32Work::HDR_CAPABLE);
    }

    #[test]
    fn no_alpha_rgb_3ch() {
        // Verify 3-channel (no alpha) works through IntoWorking
        let tf = Srgb;
        let luts = tf.build_luts();

        let src = [128u8, 64, 32]; // single RGB pixel
        let mut dst = [0.0f32; 3];

        <u8 as IntoWorking<F32Work, Srgb>>::convert(
            &src, &mut dst, &tf, &luts, 3, false, false,
        );

        // All channels should be linearized (no alpha special-casing)
        assert!(dst[0] > 0.2, "R: {}", dst[0]);
        assert!(dst[1] > 0.04, "G: {}", dst[1]);
        assert!(dst[2] > 0.01, "B: {}", dst[2]);
    }

    #[test]
    fn cmyk_4ch_no_alpha() {
        // Verify 4-channel without alpha (like CMYK) works
        let tf = NoTransfer;
        let luts = tf.build_luts();

        let src = [200u8, 100, 50, 255]; // CMYK pixel
        let mut dst = [0.0f32; 4];

        <u8 as IntoWorking<F32Work, NoTransfer>>::convert(
            &src, &mut dst, &tf, &luts, 4, false, false,
        );

        // All 4 channels should be scaled to [0,1], no alpha treatment
        assert!((dst[0] - 200.0 / 255.0).abs() < 0.01);
        assert!((dst[3] - 255.0 / 255.0).abs() < 0.01);
    }
}

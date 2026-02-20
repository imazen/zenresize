//! Transfer function traits and implementations.
//!
//! A transfer function (TF) maps between a perceptual/encoded space and linear light.
//! sRGB, PQ (HDR10), HLG, and gamma curves are all transfer functions.
//!
//! The [`TransferFunction`] trait provides batch methods for converting pixel rows
//! between encoded and working space. Implementations fuse linearization with
//! premultiply/unpremultiply for efficiency.
//!
//! # Standard implementations
//!
//! - [`NoTransfer`] — Identity (no conversion). Zero overhead.
//! - [`Srgb`] — sRGB gamma curve. Uses compile-time LUTs for u8 and
//!   the `linear-srgb` crate's SIMD-friendly LUTs for f32.
//!
//! # LUT caching
//!
//! Standard TFs permanently cache runtime LUTs via `OnceLock` (with the `std` feature).
//! For `Srgb`, the u8↔i12 tables are compile-time constants — no runtime allocation.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::color;
use crate::simd;

// =============================================================================
// TransferFunction trait
// =============================================================================

/// Transfer function for encoding/decoding pixel values to/from linear light.
///
/// Provides batch scanline methods for converting between encoded element types
/// and the working-space type used during filtering. The `channels` parameter
/// and `premul`/`unpremul` flags handle arbitrary layouts — RGB (3ch, no alpha),
/// RGBA (4ch, alpha last), CMYK (4ch, no alpha), Gray (1ch), etc.
///
/// # Implementing a custom TF
///
/// Implement `to_linear` and `from_linear` for the scalar curve. The batch
/// methods have default implementations that call these in a loop, but
/// optimized implementations should override the batch methods with LUT-based
/// or SIMD-based versions.
pub trait TransferFunction: Send + Sync + 'static {
    /// Cached state (LUTs, etc.). Built once by `build_luts()`, passed to batch methods.
    ///
    /// Standard TFs use `&'static RuntimeLuts` (permanently cached via `OnceLock`).
    /// Custom TFs can use owned `RuntimeLuts` or `()`.
    type Luts: Send + Sync;

    // --- Scalar (for LUT building and testing) ---

    /// Encode a linear-light value to this TF's encoded space.
    /// Both input and output are in [0, 1] for SDR TFs.
    fn to_linear(&self, encoded: f32) -> f32;

    /// Decode a value from this TF's encoded space to linear light.
    fn from_linear(&self, linear: f32) -> f32;

    /// Whether this TF is the identity (no conversion needed).
    fn is_identity(&self) -> bool {
        false
    }

    /// Build or retrieve cached LUTs.
    fn build_luts(&self) -> Self::Luts;

    // --- Batch: u8 encoded → working f32 ---

    /// Convert a row of u8 encoded pixels to premultiplied linear f32.
    ///
    /// Fuses linearize + premultiply. If `premul` is false, only linearizes.
    /// `channels` is the number of components per pixel (1, 3, or 4).
    /// Alpha (if present) is assumed to be the last channel and is NOT linearized
    /// — it's scaled linearly (v/255).
    fn u8_to_linear_f32(
        &self,
        src: &[u8],
        dst: &mut [f32],
        luts: &Self::Luts,
        channels: usize,
        has_alpha: bool,
        premul: bool,
    );

    /// Convert a row of premultiplied linear f32 to u8 encoded pixels.
    ///
    /// Fuses unpremultiply + delinearize + quantize. Clamps to [0, 255].
    fn linear_f32_to_u8(
        &self,
        src: &[f32],
        dst: &mut [u8],
        luts: &Self::Luts,
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    );

    // --- Batch: u16 encoded ↔ working f32 ---

    /// Convert a row of u16 encoded pixels to premultiplied linear f32.
    ///
    /// Values span full 0-65535 range. Alpha (if present, last channel)
    /// is scaled linearly (v/65535). Non-alpha channels go through the TF.
    fn u16_to_linear_f32(
        &self,
        src: &[u16],
        dst: &mut [f32],
        luts: &Self::Luts,
        channels: usize,
        has_alpha: bool,
        premul: bool,
    );

    /// Convert premultiplied linear f32 to u16 encoded pixels.
    fn linear_f32_to_u16(
        &self,
        src: &[f32],
        dst: &mut [u16],
        luts: &Self::Luts,
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    );

    // --- Batch: u8 encoded → i12 linear (for I16Work path) ---

    /// Convert a row of u8 encoded pixels to linear i12 (0-4095).
    /// All channels converted; no premul/unpremul (I16Work doesn't support premul).
    fn u8_to_linear_i12(&self, src: &[u8], dst: &mut [i16], luts: &Self::Luts);

    /// Convert a row of linear i12 to u8 encoded pixels.
    fn linear_i12_to_u8(&self, src: &[i16], dst: &mut [u8], luts: &Self::Luts);

    // --- Batch: f32 encoded ↔ linear f32 (in-place) ---

    /// Convert f32 encoded values to premultiplied linear f32 in-place.
    fn f32_to_linear_inplace(
        &self,
        row: &mut [f32],
        channels: usize,
        has_alpha: bool,
        premul: bool,
    );

    /// Convert premultiplied linear f32 to f32 encoded in-place.
    /// Does NOT clamp — output can have values outside [0, 1] for wide gamut.
    fn linear_to_f32_inplace(
        &self,
        row: &mut [f32],
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    );
}

// =============================================================================
// NoTransfer — identity transfer function
// =============================================================================

/// Identity transfer function: no conversion.
///
/// Used when pixel data is already in the desired space, or when
/// resizing should happen in the encoded domain (gamma-space resize).
/// All batch methods are trivial copies/scales.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoTransfer;

impl TransferFunction for NoTransfer {
    type Luts = ();

    #[inline]
    fn to_linear(&self, v: f32) -> f32 {
        v
    }

    #[inline]
    fn from_linear(&self, v: f32) -> f32 {
        v
    }

    #[inline]
    fn is_identity(&self) -> bool {
        true
    }

    #[inline]
    fn build_luts(&self) -> Self::Luts {}

    fn u8_to_linear_f32(
        &self,
        src: &[u8],
        dst: &mut [f32],
        _luts: &(),
        _channels: usize,
        _has_alpha: bool,
        premul: bool,
    ) {
        simd::u8_to_f32_row(src, dst);
        if premul {
            simd::premultiply_alpha_row(dst);
        }
    }

    fn linear_f32_to_u8(
        &self,
        src: &[f32],
        dst: &mut [u8],
        _luts: &(),
        _channels: usize,
        _has_alpha: bool,
        unpremul: bool,
    ) {
        if unpremul {
            // Need a mutable copy for unpremultiply
            let mut tmp = src.to_vec();
            simd::unpremultiply_alpha_row(&mut tmp);
            simd::f32_to_u8_row(&tmp, dst);
        } else {
            simd::f32_to_u8_row(src, dst);
        }
    }

    fn u16_to_linear_f32(
        &self,
        src: &[u16],
        dst: &mut [f32],
        _luts: &(),
        _channels: usize,
        _has_alpha: bool,
        premul: bool,
    ) {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = *s as f32 / 65535.0;
        }
        if premul {
            simd::premultiply_alpha_row(dst);
        }
    }

    fn linear_f32_to_u16(
        &self,
        src: &[f32],
        dst: &mut [u16],
        _luts: &(),
        _channels: usize,
        _has_alpha: bool,
        unpremul: bool,
    ) {
        if unpremul {
            let mut tmp = src.to_vec();
            simd::unpremultiply_alpha_row(&mut tmp);
            for (s, d) in tmp.iter().zip(dst.iter_mut()) {
                *d = (*s * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            }
        } else {
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = (*s * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            }
        }
    }

    fn u8_to_linear_i12(&self, src: &[u8], dst: &mut [i16], _luts: &()) {
        // Identity: scale u8 (0-255) → i12 (0-4095)
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            // Exact: (v * 4095 + 127) / 255
            *d = ((*s as u32 * 4095 + 127) / 255) as i16;
        }
    }

    fn linear_i12_to_u8(&self, src: &[i16], dst: &mut [u8], _luts: &()) {
        // Identity: scale i12 (0-4095) → u8 (0-255)
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            let clamped = (*s).clamp(0, 4095) as u32;
            *d = ((clamped * 255 + 2047) / 4095) as u8;
        }
    }

    fn f32_to_linear_inplace(
        &self,
        row: &mut [f32],
        _channels: usize,
        _has_alpha: bool,
        premul: bool,
    ) {
        if premul {
            simd::premultiply_alpha_row(row);
        }
    }

    fn linear_to_f32_inplace(
        &self,
        row: &mut [f32],
        _channels: usize,
        _has_alpha: bool,
        unpremul: bool,
    ) {
        if unpremul {
            simd::unpremultiply_alpha_row(row);
        }
    }
}

// =============================================================================
// Srgb — sRGB transfer function
// =============================================================================

/// sRGB transfer function.
///
/// Uses the `linear-srgb` crate's LUT-based conversion for u8↔f32 and
/// compile-time constant tables for u8↔i12. The u8 path is heavily optimized
/// with SIMD-friendly LUT lookups.
///
/// For u16 input, values are normalized to [0, 1] and passed through the
/// scalar sRGB curve. This is slower than the u8 LUT path but exact.
#[derive(Debug, Clone, Copy, Default)]
pub struct Srgb;

impl TransferFunction for Srgb {
    type Luts = ();

    #[inline]
    fn to_linear(&self, encoded: f32) -> f32 {
        // Standard sRGB EOTF
        if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    }

    #[inline]
    fn from_linear(&self, linear: f32) -> f32 {
        // Standard sRGB inverse EOTF
        if linear <= 0.0031308 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        }
    }

    #[inline]
    fn build_luts(&self) -> Self::Luts {
        // Srgb uses compile-time LUTs and the linear-srgb crate — no runtime LUTs needed.
    }

    fn u8_to_linear_f32(
        &self,
        src: &[u8],
        dst: &mut [f32],
        _luts: &(),
        channels: usize,
        has_alpha: bool,
        premul: bool,
    ) {
        color::srgb_u8_to_linear_f32(src, dst, channels, has_alpha);
        if premul {
            simd::premultiply_alpha_row(dst);
        }
    }

    fn linear_f32_to_u8(
        &self,
        src: &[f32],
        dst: &mut [u8],
        _luts: &(),
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    ) {
        if unpremul {
            let mut tmp = src.to_vec();
            simd::unpremultiply_alpha_row(&mut tmp);
            color::linear_f32_to_srgb_u8(&tmp, dst, channels, has_alpha);
        } else {
            color::linear_f32_to_srgb_u8(src, dst, channels, has_alpha);
        }
    }

    fn u16_to_linear_f32(
        &self,
        src: &[u16],
        dst: &mut [f32],
        _luts: &(),
        channels: usize,
        has_alpha: bool,
        premul: bool,
    ) {
        // u16 sRGB: normalize to [0,1], apply sRGB EOTF
        if has_alpha && channels >= 2 {
            for pixel in src.chunks_exact(channels).zip(dst.chunks_exact_mut(channels)) {
                let (src_px, dst_px) = pixel;
                // Color channels: sRGB curve
                for i in 0..channels - 1 {
                    dst_px[i] = self.to_linear(src_px[i] as f32 / 65535.0);
                }
                // Alpha: linear scale
                dst_px[channels - 1] = src_px[channels - 1] as f32 / 65535.0;
            }
        } else {
            // No alpha: all channels through sRGB
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = self.to_linear(*s as f32 / 65535.0);
            }
        }
        if premul {
            simd::premultiply_alpha_row(dst);
        }
    }

    fn linear_f32_to_u16(
        &self,
        src: &[f32],
        dst: &mut [u16],
        _luts: &(),
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    ) {
        let work: Vec<f32>;
        let src = if unpremul {
            work = {
                let mut tmp = src.to_vec();
                simd::unpremultiply_alpha_row(&mut tmp);
                tmp
            };
            &work
        } else {
            src
        };

        if has_alpha && channels >= 2 {
            for (src_px, dst_px) in src.chunks_exact(channels).zip(dst.chunks_exact_mut(channels)) {
                for i in 0..channels - 1 {
                    dst_px[i] =
                        (self.from_linear(src_px[i]) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                }
                dst_px[channels - 1] =
                    (src_px[channels - 1] * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            }
        } else {
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = (self.from_linear(*s) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
            }
        }
    }

    fn u8_to_linear_i12(&self, src: &[u8], dst: &mut [i16], _luts: &()) {
        color::srgb_u8_to_linear_i12_row(src, dst);
    }

    fn linear_i12_to_u8(&self, src: &[i16], dst: &mut [u8], _luts: &()) {
        color::linear_i12_to_srgb_u8_row(src, dst);
    }

    fn f32_to_linear_inplace(
        &self,
        row: &mut [f32],
        channels: usize,
        has_alpha: bool,
        premul: bool,
    ) {
        // f32 encoded sRGB → linear
        if has_alpha && channels >= 2 {
            for pixel in row.chunks_exact_mut(channels) {
                for v in &mut pixel[..channels - 1] {
                    *v = self.to_linear(*v);
                }
                // Alpha stays as-is
            }
        } else {
            for v in row.iter_mut() {
                *v = self.to_linear(*v);
            }
        }
        if premul {
            simd::premultiply_alpha_row(row);
        }
    }

    fn linear_to_f32_inplace(
        &self,
        row: &mut [f32],
        channels: usize,
        has_alpha: bool,
        unpremul: bool,
    ) {
        if unpremul {
            simd::unpremultiply_alpha_row(row);
        }
        // linear → f32 encoded sRGB (no clamp for f32 output)
        if has_alpha && channels >= 2 {
            for pixel in row.chunks_exact_mut(channels) {
                for v in &mut pixel[..channels - 1] {
                    *v = self.from_linear(*v);
                }
            }
        } else {
            for v in row.iter_mut() {
                *v = self.from_linear(*v);
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_transfer_roundtrip_u8() {
        let tf = NoTransfer;
        let luts = tf.build_luts();
        let src: Vec<u8> = (0..=255).collect();
        let mut f32_buf = vec![0.0f32; 256];
        let mut out = vec![0u8; 256];

        tf.u8_to_linear_f32(&src, &mut f32_buf, &luts, 1, false, false);
        tf.linear_f32_to_u8(&f32_buf, &mut out, &luts, 1, false, false);

        for i in 0..256 {
            assert_eq!(src[i], out[i], "NoTransfer roundtrip mismatch at {}", i);
        }
    }

    #[test]
    fn srgb_roundtrip_u8() {
        let tf = Srgb;
        let luts = tf.build_luts();
        let src: Vec<u8> = (0..=255).collect();
        let mut f32_buf = vec![0.0f32; 256];
        let mut out = vec![0u8; 256];

        tf.u8_to_linear_f32(&src, &mut f32_buf, &luts, 1, false, false);
        tf.linear_f32_to_u8(&f32_buf, &mut out, &luts, 1, false, false);

        for i in 0..256 {
            let diff = (src[i] as i16 - out[i] as i16).unsigned_abs();
            assert!(diff <= 1, "sRGB roundtrip off by {} at {}", diff, i);
        }
    }

    #[test]
    fn srgb_roundtrip_u16() {
        let tf = Srgb;
        let luts = tf.build_luts();

        // Test a spread of u16 values
        let values: Vec<u16> = (0..=65535).step_by(257).collect(); // 256 values
        let mut f32_buf = vec![0.0f32; values.len()];
        let mut out = vec![0u16; values.len()];

        tf.u16_to_linear_f32(&values, &mut f32_buf, &luts, 1, false, false);
        tf.linear_f32_to_u16(&f32_buf, &mut out, &luts, 1, false, false);

        for i in 0..values.len() {
            let diff = (values[i] as i32 - out[i] as i32).unsigned_abs();
            // u16 roundtrip through f32 scalar sRGB curve — allow ±1 from rounding
            assert!(
                diff <= 1,
                "sRGB u16 roundtrip off by {} at value {}: {} -> {} -> {}",
                diff,
                values[i],
                values[i],
                f32_buf[i],
                out[i]
            );
        }
    }

    #[test]
    fn srgb_i12_matches_existing() {
        let tf = Srgb;
        let luts = tf.build_luts();

        let src: Vec<u8> = (0..=255).collect();
        let mut via_tf = vec![0i16; 256];
        let mut via_direct = vec![0i16; 256];

        tf.u8_to_linear_i12(&src, &mut via_tf, &luts);
        crate::color::srgb_u8_to_linear_i12_row(&src, &mut via_direct);

        assert_eq!(via_tf, via_direct, "TF i12 path should match direct LUT");
    }

    #[test]
    fn srgb_scalar_matches_lut() {
        let tf = Srgb;
        // Verify scalar to_linear matches the LUT-based path
        for i in 0..=255u8 {
            let from_scalar = tf.to_linear(i as f32 / 255.0);
            // The LUT gives us the same thing the linear-srgb crate computes
            let mut f32_buf = [0.0f32];
            crate::color::srgb_u8_to_linear_f32(&[i], &mut f32_buf, 1, false);

            let diff = (from_scalar - f32_buf[0]).abs();
            assert!(
                diff < 1e-5,
                "sRGB scalar vs LUT mismatch at {}: scalar={}, lut={}",
                i,
                from_scalar,
                f32_buf[0]
            );
        }
    }

    #[test]
    fn no_transfer_identity() {
        let tf = NoTransfer;
        assert!(tf.is_identity());
        assert_eq!(tf.to_linear(0.5), 0.5);
        assert_eq!(tf.from_linear(0.5), 0.5);
    }

    #[test]
    fn srgb_not_identity() {
        let tf = Srgb;
        assert!(!tf.is_identity());
        // sRGB mid-gray (0.5 encoded) should be ~0.214 linear
        let linear = tf.to_linear(0.5);
        assert!(
            (linear - 0.214).abs() < 0.01,
            "sRGB 0.5 → linear = {} (expected ~0.214)",
            linear
        );
    }

    #[test]
    fn srgb_u8_to_f32_with_alpha() {
        let tf = Srgb;
        let luts = tf.build_luts();

        // RGBA pixel: [128, 64, 32, 200]
        let src = [128u8, 64, 32, 200];
        let mut dst = [0.0f32; 4];

        tf.u8_to_linear_f32(&src, &mut dst, &luts, 4, true, false);

        // RGB should be linearized, alpha should be v/255
        assert!(dst[0] > 0.2 && dst[0] < 0.3, "R linear: {}", dst[0]);
        assert!((dst[3] - 200.0 / 255.0).abs() < 0.01, "A: {}", dst[3]);
    }

    #[test]
    fn srgb_u8_premul_unpremul_roundtrip() {
        let tf = Srgb;
        let luts = tf.build_luts();

        let src = [128u8, 64, 32, 200];
        let mut f32_buf = [0.0f32; 4];
        let mut out = [0u8; 4];

        // Encode → premul linear
        tf.u8_to_linear_f32(&src, &mut f32_buf, &luts, 4, true, true);
        // Premul linear → decode (with unpremul)
        tf.linear_f32_to_u8(&f32_buf, &mut out, &luts, 4, true, true);

        for i in 0..4 {
            let diff = (src[i] as i16 - out[i] as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "Premul roundtrip off by {} at channel {}: {} vs {}",
                diff,
                i,
                src[i],
                out[i]
            );
        }
    }

    #[test]
    fn no_alpha_3ch_roundtrip() {
        let tf = Srgb;
        let luts = tf.build_luts();

        let src = [128u8, 64, 32, 200, 100, 50]; // 2 RGB pixels
        let mut f32_buf = [0.0f32; 6];
        let mut out = [0u8; 6];

        tf.u8_to_linear_f32(&src, &mut f32_buf, &luts, 3, false, false);
        tf.linear_f32_to_u8(&f32_buf, &mut out, &luts, 3, false, false);

        for i in 0..6 {
            let diff = (src[i] as i16 - out[i] as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "3ch roundtrip off by {} at {}: {} vs {}",
                diff,
                i,
                src[i],
                out[i]
            );
        }
    }
}

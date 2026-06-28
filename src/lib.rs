//! High-quality image resampling with 31 filters, streaming API, and SIMD acceleration.
//!
//! `zenresize` provides a standalone resize library extracted from the zenimage/imageflow
//! image processing pipeline. It supports:
//!
//! - All 31 resampling filters from imageflow (Lanczos, Robidoux, Mitchell, etc.)
//! - Row-at-a-time streaming API with V-first pipeline for pipeline integration
//! - Built-in sRGB/linear conversion and alpha premultiply/unpremultiply
//! - u8, i16, and f32 pixel format support
//! - archmage-based SIMD (AVX2+FMA on x86, NEON on ARM)
//!
//! # Quick Start
//!
//! ```
//! use zenresize::{Resizer, ResizeConfig, Filter, PixelDescriptor};
//!
//! // Create a 4×4 RGBA test image
//! let input_pixels = vec![128u8; 4 * 4 * 4];
//!
//! let config = ResizeConfig::builder(4, 4, 2, 2)
//!     .filter(Filter::Lanczos)
//!     .format(PixelDescriptor::RGBA8_SRGB)
//!     .build();
//!
//! let output = Resizer::new(&config).resize(&input_pixels);
//! assert_eq!(output.len(), 2 * 2 * 4);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "pretty-safe"), forbid(unsafe_code))]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

whereat::define_at_crate_info!();

pub(crate) mod blur;
pub(crate) mod color;
pub mod composite;
#[doc(hidden)]
pub mod filter;
pub mod fit;
pub(crate) mod pixel;
pub mod plane;
pub(crate) mod resize;
pub(crate) mod streaming;
pub(crate) mod transfer;
#[doc(hidden)]
pub mod weights;
#[allow(dead_code)] // WIP: generic resize pipeline types
pub(crate) mod working;

#[doc(hidden)]
#[allow(clippy::excessive_precision)]
pub mod fastmath;
mod proven;
#[doc(hidden)]
pub mod simd;

// Re-exports from zenpixels
pub use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, Orientation, PixelDescriptor, TransferFunction,
};

// Re-exports: minimal public API
pub use filter::Filter;
pub use fit::{FitMode, fit_cover_source_crop, fit_dims};
pub use pixel::{
    ConfigError, Element, LobeRatio, Padding, ResizeConfig, ResizeConfigBuilder, SourceRegion,
};
pub use plane::PlaneResizer;
pub use resize::Resizer;
pub use streaming::{OrientOutput, StreamingError, StreamingResize, WorkingFormat};

pub use composite::{
    Background, BlendMode, CompositeError, NoBackground, SliceBackground, SolidBackground,
    StreamedBackground, composite_over_premul, composite_over_solid_opaque_premul,
    composite_over_solid_premul, unpremultiply_f32_row,
};

pub use zenblend::mask::{
    LinearGradientMask, MaskFill, MaskSource, MaskSpan, MaskSpans, RadialGradientMask,
    RoundedRectMask, SpanKind, mask_pixel_align,
};

pub use resize::{
    resize_3ch, resize_4ch, resize_gray8, resize_hfirst_streaming, resize_hfirst_streaming_f32,
};

pub use transfer::{Bt709, Hlg, NoTransfer, Pq, Srgb, TransferCurve};

// zennode node definitions
// #[cfg(feature = "zennode")]
// pub mod zennode_defs;

// Re-export whereat types for downstream consumers
pub use whereat::{At, ResultAtExt};

// ---------------------------------------------------------------------------
// One-shot convenience functions
//
// The shortest path for the two most common jobs. Both wrap the builder +
// `Resizer` path with sane photographic defaults (Lanczos filter, correct sRGB
// linear-light resampling). Reach for `ResizeConfig::builder` when you need a
// different filter, color space, crop, padding, `Cover`/`Stretch` fit, u16/f32
// I/O, or row-at-a-time streaming.
// ---------------------------------------------------------------------------

/// Resize an 8-bit RGBA image to exact dimensions in one call.
///
/// `src` must be exactly `in_w * in_h * 4` bytes of tightly-packed RGBA; the
/// returned buffer is `out_w * out_h * 4` bytes.
///
/// Fallible by design: the output dimensions are validated the same way
/// [`Resizer::try_new`] validates them (the default 120 MP cap, plus NaN /
/// degenerate-config rejection), and the input length is checked with overflow
/// safety — so passing an untrusted target size or a mismatched buffer returns a
/// [`ConfigError`] instead of panicking. For a different filter, color space,
/// crop, padding, `Cover`/`Stretch` fit, u16/f32 I/O, or row-streaming, build a
/// [`ResizeConfig`] and use [`Resizer`].
///
/// ```
/// let src = vec![128u8; 64 * 48 * 4];        // 64×48 RGBA
/// let out = zenresize::resize_rgba8(&src, 64, 48, 32, 24)?;
/// assert_eq!(out.len(), 32 * 24 * 4);
/// # Ok::<(), zenresize::At<zenresize::ConfigError>>(())
/// ```
pub fn resize_rgba8(
    src: &[u8],
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
) -> Result<Vec<u8>, At<ConfigError>> {
    check_rgba8_len(src, in_w, in_h)?;
    let mut resizer = Resizer::try_new(
        &ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .build(),
    )?;
    Ok(resizer.resize(src))
}

/// Aspect-preserving resize of an 8-bit RGBA image to fit within `max_w × max_h`.
///
/// Never upscales past the source ([`FitMode::Within`]): the result is at most
/// `max_w × max_h`, touches the box on its longer axis, and is returned with its
/// actual dimensions. For `Cover` (center-crop to fill), `Fit` (allow upscale),
/// or `Stretch`, build a [`ResizeConfig`] with [`ResizeConfigBuilder::fit`].
///
/// `src` must be `in_w * in_h * 4` bytes of tightly-packed RGBA; an oversized
/// target or a length mismatch returns a [`ConfigError`] (never panics).
///
/// ```
/// let src = vec![200u8; 1600 * 900 * 4];      // 1600×900 RGBA
/// let (thumb, w, h) = zenresize::resize_rgba8_to_fit(&src, 1600, 900, 320, 320)?;
/// assert_eq!((w, h), (320, 180));             // 16:9 fit inside 320×320
/// assert_eq!(thumb.len(), (w * h * 4) as usize);
/// # Ok::<(), zenresize::At<zenresize::ConfigError>>(())
/// ```
pub fn resize_rgba8_to_fit(
    src: &[u8],
    in_w: u32,
    in_h: u32,
    max_w: u32,
    max_h: u32,
) -> Result<(Vec<u8>, u32, u32), At<ConfigError>> {
    check_rgba8_len(src, in_w, in_h)?;
    let (out_w, out_h) = fit_dims(in_w, in_h, max_w, max_h, FitMode::Within);
    let mut resizer = Resizer::try_new(
        &ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .build(),
    )?;
    Ok((resizer.resize(src), out_w, out_h))
}

/// Validate that `src` is exactly `in_w * in_h * 4` tightly-packed RGBA8 bytes,
/// returning a located [`ConfigError`] (never panicking) on mismatch or on a
/// `in_w * in_h * 4` product that overflows `usize`.
fn check_rgba8_len(src: &[u8], in_w: u32, in_h: u32) -> Result<(), At<ConfigError>> {
    let expected = (in_w as usize)
        .checked_mul(in_h as usize)
        .and_then(|px| px.checked_mul(4));
    if expected != Some(src.len()) {
        return Err(whereat::at!(ConfigError(
            "src length must equal in_w * in_h * 4 (tightly-packed RGBA8)"
        )));
    }
    Ok(())
}

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

// Re-export the imgref + rgb input/output types so callers of the typed one-shot
// fns (and the `resize_4ch` / `resize_3ch` family) don't need to depend on the
// `imgref` / `rgb` crates directly.
pub use imgref::{ImgRef, ImgVec};
pub use rgb::RGBA8;

pub use transfer::{Bt709, Hlg, NoTransfer, Pq, Srgb, TransferCurve};

// zennode node definitions
// #[cfg(feature = "zennode")]
// pub mod zennode_defs;

// Re-export whereat types for downstream consumers
pub use whereat::{At, ResultAtExt};

// ---------------------------------------------------------------------------
// One-shot convenience functions
//
// The shortest path for the two most common jobs. Both take a typed
// [`ImgRef<RGBA8>`] so the source dimensions and row stride ride *with* the
// pixels — there is no separate width/height to keep in sync and no
// buffer-length-mismatch class of bug. Both default to a Lanczos filter with
// correct sRGB linear-light resampling. Reach for [`ResizeConfig::builder`] +
// [`Resizer`] when you need a different filter, color space, crop, padding,
// `Cover`/`Stretch` fit, u16/f32 I/O, or row-at-a-time streaming.
// ---------------------------------------------------------------------------

/// Resize an 8-bit RGBA image to exact `out_w × out_h` in one call.
///
/// The source dimensions and row stride travel with the pixels inside
/// [`ImgRef`], so there is no separate width/height argument to keep in sync and
/// no buffer-length mismatch to guard against. Lanczos filter, correct sRGB
/// linear-light resampling.
///
/// Fallible: the target dimensions are validated the same way
/// [`Resizer::try_new`] validates them (the default 120 MP cap, plus NaN /
/// degenerate-config rejection), so an untrusted target size returns a located
/// [`ConfigError`] instead of panicking. For a different filter, color space,
/// crop, padding, `Cover`/`Stretch` fit, u16/f32 I/O, or row-streaming, build a
/// [`ResizeConfig`] and use [`Resizer`].
///
/// ```
/// use zenresize::{ImgRef, RGBA8};
///
/// let pixels = vec![RGBA8::default(); 64 * 48]; // 64×48 RGBA
/// let img = ImgRef::new(&pixels, 64, 48);
/// let out = zenresize::resize_rgba8(img, 32, 24)?;
/// assert_eq!((out.width(), out.height()), (32, 24));
/// # Ok::<(), zenresize::At<zenresize::ConfigError>>(())
/// ```
pub fn resize_rgba8(
    img: ImgRef<RGBA8>,
    out_w: u32,
    out_h: u32,
) -> Result<ImgVec<RGBA8>, At<ConfigError>> {
    let config = ResizeConfig::builder(img.width() as u32, img.height() as u32, out_w, out_h)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBA8_SRGB)
        .build();
    // Validate the target dimensions (the 120 MP cap, NaN / degenerate-config
    // rejection) before resizing, so an untrusted target size returns a located
    // error instead of panicking inside `resize_4ch`. This is the same check
    // `Resizer::try_new` performs.
    config
        .validate()
        .map_err(|msg| whereat::at!(ConfigError(msg)))?;
    Ok(resize_4ch(
        img,
        out_w,
        out_h,
        PixelDescriptor::RGBA8_SRGB,
        &config,
    ))
}

/// Aspect-preserving resize of an 8-bit RGBA image to fit within `max_w × max_h`.
///
/// Never upscales past the source ([`FitMode::Within`]): the result is at most
/// `max_w × max_h` and touches the box on its longer axis. The output dimensions
/// ride with the returned [`ImgVec`] (`.width()` / `.height()`). For `Cover`
/// (center-crop to fill), `Fit` (allow upscale), or `Stretch`, build a
/// [`ResizeConfig`] with [`ResizeConfigBuilder::fit`].
///
/// Fallible on the same terms as [`resize_rgba8`].
///
/// ```
/// use zenresize::{ImgRef, RGBA8};
///
/// let pixels = vec![RGBA8::default(); 1600 * 900]; // 1600×900 RGBA
/// let img = ImgRef::new(&pixels, 1600, 900);
/// let thumb = zenresize::resize_rgba8_to_fit(img, 320, 320)?;
/// assert_eq!((thumb.width(), thumb.height()), (320, 180)); // 16:9 within 320×320
/// # Ok::<(), zenresize::At<zenresize::ConfigError>>(())
/// ```
pub fn resize_rgba8_to_fit(
    img: ImgRef<RGBA8>,
    max_w: u32,
    max_h: u32,
) -> Result<ImgVec<RGBA8>, At<ConfigError>> {
    let (out_w, out_h) = fit_dims(
        img.width() as u32,
        img.height() as u32,
        max_w,
        max_h,
        FitMode::Within,
    );
    resize_rgba8(img, out_w, out_h)
}

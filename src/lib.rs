//! High-quality image resampling with 31 filters, streaming API, and SIMD acceleration.
//!
//! `zenresize` provides a standalone resize library extracted from the zenimage/imageflow
//! image processing pipeline. It supports:
//!
//! - All 31 resampling filters from imageflow (Lanczos, Robidoux, Mitchell, etc.)
//! - Row-at-a-time streaming API for pipeline integration
//! - Built-in sRGB/linear conversion and alpha premultiply/unpremultiply
//! - u8, i16, and f32 pixel format support
//! - archmage-based SIMD (AVX2+FMA on x86, NEON on ARM)
//!
//! # Quick Start
//!
//! ```
//! use zenresize::{Resizer, ResizeConfig, Filter, PixelFormat, PixelLayout};
//!
//! // Create a 4×4 RGBA test image
//! let input_pixels = vec![128u8; 4 * 4 * 4];
//!
//! let config = ResizeConfig::builder(4, 4, 2, 2)
//!     .filter(Filter::Lanczos)
//!     .format(PixelFormat::Srgb8(PixelLayout::Rgba))
//!     .build();
//!
//! let output = Resizer::new(&config).resize(&input_pixels);
//! assert_eq!(output.len(), 2 * 2 * 4);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "pretty-safe"), forbid(unsafe_code))]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub(crate) mod color;
pub(crate) mod filter;
#[cfg(feature = "layout")]
pub mod layout;
pub(crate) mod pixel;
pub(crate) mod resize;
pub(crate) mod streaming;
pub(crate) mod weights;

mod proven;
mod simd;

// Re-exports: minimal public API
pub use filter::Filter;
#[cfg(feature = "layout")]
pub use layout::{
    CanvasColor, Constraint, ConstraintMode, Gravity, Layout, LayoutError, Rect, SourceCrop,
};
pub use pixel::{PixelFormat, PixelLayout, ResizeConfig, ResizeConfigBuilder};
pub use resize::Resizer;
pub use streaming::StreamingResize;

#[cfg(feature = "imgref")]
pub use resize::{resize_3ch, resize_4ch, resize_gray8};

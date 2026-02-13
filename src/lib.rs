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
//! use zenresize::{resize, ResizeConfig, Filter, PixelFormat};
//!
//! // Create a 4×4 RGBA test image
//! let input_pixels = vec![128u8; 4 * 4 * 4];
//!
//! let config = ResizeConfig::builder(4, 4, 2, 2)
//!     .filter(Filter::Lanczos)
//!     .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
//!     .linear()
//!     .build();
//!
//! let output = resize(&config, &input_pixels);
//! assert_eq!(output.len(), 2 * 2 * 4);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod color;
pub mod filter;
pub mod pixel;
pub mod resize;
pub mod streaming;
pub mod weights;

mod simd;

// Re-exports
pub use filter::{Filter, InterpolationDetails};
pub use pixel::{ColorSpace, PixelFormat, ResizeConfig, ResizeConfigBuilder};
pub use resize::{resize, resize_f32, resize_f32_into, resize_into, Resizer};
pub use streaming::StreamingResize;
pub use weights::{F32WeightTable, I16WeightTable};

#[cfg(feature = "imgref")]
pub use resize::{resize_3ch, resize_4ch, resize_gray8};

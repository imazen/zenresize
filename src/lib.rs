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
#[cfg(feature = "layout")]
pub(crate) mod execute;
#[doc(hidden)]
pub mod filter;
#[cfg(feature = "layout")]
pub mod layout;
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
pub use zenpixels::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};

// Re-exports: minimal public API
#[cfg(feature = "layout")]
pub use execute::{
    canvas_color_to_pixel, config_from_plan, execute, execute_layout,
    execute_layout_with_background, execute_secondary, execute_secondary_with_background,
    execute_with_background, execute_with_offer, fill_canvas, orient_image, place_on_canvas,
    replicate_edges, streaming_from_plan, streaming_from_plan_batched,
};
pub use filter::Filter;
#[cfg(feature = "layout")]
pub use layout::{
    Align, CanvasColor, Command, Constraint, ConstraintMode, DecoderOffer, DecoderRequest,
    DimensionEffect, ExpandEffect, FlipAxis, Gravity, IdealLayout, Layout, LayoutError, LayoutPlan,
    Orientation, OutputLimits, PadEffect, Pipeline, Rect, Region, RegionCoord, ResolutionPolicy,
    ResolvedEffect, RotateEffect, RotateMode, Rotation, Size, SourceCrop, Subsampling, TrimEffect,
    WarpEffect, compute_layout, compute_layout_sequential, expanded_canvas_dims,
    expanded_canvas_inverse, inscribed_crop_dims, inscribed_crop_inverse, warp_output_dims,
};
pub use pixel::{Element, LobeRatio, Padding, ResizeConfig, ResizeConfigBuilder, SourceRegion};
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

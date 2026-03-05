//! Pixel format descriptors, element types, and color space configuration.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use zenpixels::{AlphaMode, ChannelType, PixelDescriptor, TransferFunction};

// =============================================================================
// Element trait
// =============================================================================

/// Pixel element type.
///
/// Implemented for `u8`, `u16`, and `f32`. Determines the memory layout of
/// input and output pixel data. The resize pipeline converts elements to a
/// working type internally — see the `WorkingType` trait.
pub trait Element: Copy + Default + Send + Sync + 'static {
    /// Allocate a zeroed output buffer of the given length.
    fn alloc_output(len: usize) -> Vec<Self>;
}

impl Element for u8 {
    #[inline]
    fn alloc_output(len: usize) -> Vec<Self> {
        crate::proven::alloc_output::<u8>(len)
    }
}

impl Element for u16 {
    #[inline]
    fn alloc_output(len: usize) -> Vec<Self> {
        vec![0u16; len]
    }
}

impl Element for f32 {
    #[inline]
    fn alloc_output(len: usize) -> Vec<Self> {
        vec![0.0f32; len]
    }
}

// =============================================================================
// ResizeConfig
// =============================================================================

/// Resize configuration built with [`ResizeConfigBuilder`].
///
/// Use [`ResizeConfig::builder()`] to create one.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ResizeConfig {
    /// Filter to use for resampling.
    pub filter: crate::filter::Filter,
    /// Input image width in pixels.
    pub in_width: u32,
    /// Input image height in pixels.
    pub in_height: u32,
    /// Output image width in pixels.
    pub out_width: u32,
    /// Output image height in pixels.
    pub out_height: u32,
    /// Input pixel descriptor.
    pub input: PixelDescriptor,
    /// Output pixel descriptor.
    pub output: PixelDescriptor,
    /// Sharpening amount (0.0 = none).
    pub sharpen: f32,
    /// Gaussian blur sigma applied after resize (0.0 = none).
    ///
    /// Runs a separate Gaussian convolution pass — prefer [`filter_blur`](Self::filter_blur)
    /// which modifies the resampling kernel at zero cost. Use `post_blur_sigma`
    /// only when you need a fixed-sigma blur independent of the resize ratio.
    pub post_blur_sigma: f32,
    /// Filter blur factor (default 1.0).
    ///
    /// Scales the resampling kernel's coordinate axis. Values > 1.0 widen the
    /// kernel (blur), values < 1.0 narrow it (sharpen). This modifies the
    /// interpolation weights themselves — no extra pass, zero runtime cost.
    ///
    /// Multiplied with the filter's built-in blur value. For example,
    /// `LanczosSharp` already has blur ≈ 0.98; setting `filter_blur(1.1)`
    /// yields an effective blur of ~1.08.
    pub filter_blur: f64,
    /// Whether to resize in linear light (true) or sRGB gamma space (false).
    ///
    /// Linear light (default) converts sRGB u8 to linear f32 before resampling.
    /// Produces correct results on gradients and avoids darkening halos.
    ///
    /// sRGB mode resamples directly in gamma space. Faster (i16 integer path
    /// for 4-channel u8), but slightly incorrect on gradients.
    pub linear: bool,
    /// Input row stride in elements (0 = tightly packed, i.e., width * channels).
    pub in_stride: usize,
    /// Output row stride in elements (0 = tightly packed).
    pub out_stride: usize,
}

impl ResizeConfig {
    /// Create a builder for `ResizeConfig`.
    pub fn builder(
        in_width: u32,
        in_height: u32,
        out_width: u32,
        out_height: u32,
    ) -> ResizeConfigBuilder {
        ResizeConfigBuilder::new(in_width, in_height, out_width, out_height)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        use zenpixels::{ChannelLayout, SignalRange};

        if self.in_width == 0 || self.in_height == 0 {
            return Err("input dimensions must be positive");
        }
        if self.out_width == 0 || self.out_height == 0 {
            return Err("output dimensions must be positive");
        }
        // Channel count must match between input and output
        if self.input.channels() != self.output.channels() {
            return Err("input and output must have the same number of channels");
        }
        // Reject unsupported channel types
        if self.input.channel_type() == ChannelType::F16
            || self.output.channel_type() == ChannelType::F16
        {
            return Err("F16 channel type is not supported; convert to F32 first");
        }
        // Reject unsupported layouts
        let reject_layout = |l: ChannelLayout| {
            matches!(
                l,
                ChannelLayout::GrayAlpha | ChannelLayout::Oklab | ChannelLayout::OklabA
            )
        };
        if reject_layout(self.input.layout()) || reject_layout(self.output.layout()) {
            return Err("unsupported channel layout; convert to a supported layout first");
        }
        // Reject unknown transfer
        if self.input.transfer == TransferFunction::Unknown
            || self.output.transfer == TransferFunction::Unknown
        {
            return Err("unknown transfer function; specify Srgb, Linear, Bt709, Pq, or Hlg");
        }
        // Reject narrow range
        if self.input.signal_range == SignalRange::Narrow
            || self.output.signal_range == SignalRange::Narrow
        {
            return Err("narrow signal range is not supported; expand to full range first");
        }
        Ok(())
    }

    /// Effective input row stride in elements.
    pub fn effective_in_stride(&self) -> usize {
        if self.in_stride == 0 {
            self.in_width as usize * self.input.channels()
        } else {
            self.in_stride
        }
    }

    /// Effective output row stride in elements.
    pub fn effective_out_stride(&self) -> usize {
        if self.out_stride == 0 {
            self.out_width as usize * self.output.channels()
        } else {
            self.out_stride
        }
    }

    /// Pixel-packed input row length (no padding).
    pub fn input_row_len(&self) -> usize {
        self.in_width as usize * self.input.channels()
    }

    /// Pixel-packed output row length (no padding).
    pub fn output_row_len(&self) -> usize {
        self.out_width as usize * self.output.channels()
    }

    /// Whether linear-light processing is needed.
    ///
    /// True when `linear` is set and the input uses an integer type with
    /// a non-linear transfer function, and alpha is not premultiplied
    /// (linearizing premultiplied sRGB data is mathematically incorrect).
    pub fn needs_linearization(&self) -> bool {
        self.linear
            && self.input.channel_type().is_integer()
            && self.input.transfer != TransferFunction::Linear
            && self.input.alpha != Some(AlphaMode::Premultiplied)
    }

    /// Effective input transfer function after inference.
    ///
    /// Returns `TransferFunction::Linear` (identity) when no linearization
    /// should occur:
    /// - Premultiplied alpha → identity (linearizing premul sRGB is wrong)
    /// - Already linear transfer → identity
    /// - `linear=false` and u8 input → identity (gamma-space filtering)
    /// - f32 input with linear transfer → identity
    ///
    /// Otherwise returns the descriptor's transfer function.
    pub fn effective_input_transfer(&self) -> TransferFunction {
        // Premultiplied sRGB data must not be linearized
        if self.input.alpha == Some(AlphaMode::Premultiplied) {
            return TransferFunction::Linear;
        }
        // Already linear → identity
        if self.input.transfer == TransferFunction::Linear {
            return TransferFunction::Linear;
        }
        // For u8 sRGB: linear=false means gamma-space filtering (no linearization).
        // Non-standard transfers (PQ, HLG, Bt709) are always respected.
        if !self.linear
            && self.input.channel_type() == ChannelType::U8
            && self.input.transfer == TransferFunction::Srgb
        {
            return TransferFunction::Linear;
        }
        // Use the descriptor's transfer
        self.input.transfer
    }

    /// Effective output transfer function after inference.
    ///
    /// Same logic as [`effective_input_transfer()`](Self::effective_input_transfer)
    /// but for the output descriptor.
    pub fn effective_output_transfer(&self) -> TransferFunction {
        if self.output.alpha == Some(AlphaMode::Premultiplied) {
            return TransferFunction::Linear;
        }
        if self.output.transfer == TransferFunction::Linear {
            return TransferFunction::Linear;
        }
        if !self.linear
            && self.output.channel_type() == ChannelType::U8
            && self.output.transfer == TransferFunction::Srgb
        {
            return TransferFunction::Linear;
        }
        self.output.transfer
    }

    /// Number of channels per pixel (from input descriptor).
    #[inline]
    pub fn channels(&self) -> usize {
        self.input.channels()
    }

    /// Whether the pipeline needs to premultiply alpha.
    #[inline]
    pub fn needs_premultiply(&self) -> bool {
        self.input.alpha == Some(AlphaMode::Straight)
    }

    /// Input channel type.
    #[inline]
    pub fn input_channel_type(&self) -> ChannelType {
        self.input.channel_type()
    }

    /// Output channel type.
    #[inline]
    pub fn output_channel_type(&self) -> ChannelType {
        self.output.channel_type()
    }
}

/// Builder for [`ResizeConfig`].
///
/// # Example
/// ```
/// use zenresize::{ResizeConfig, Filter, PixelDescriptor};
///
/// let config = ResizeConfig::builder(1024, 768, 512, 384)
///     .filter(Filter::Lanczos)
///     .format(PixelDescriptor::RGBA8_SRGB)
///     .linear()
///     .build();
/// ```
pub struct ResizeConfigBuilder {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    filter: crate::filter::Filter,
    input: PixelDescriptor,
    output: Option<PixelDescriptor>,
    sharpen: f32,
    post_blur_sigma: f32,
    filter_blur: f64,
    linear: bool,
    in_stride: usize,
    out_stride: usize,
}

impl ResizeConfigBuilder {
    fn new(in_width: u32, in_height: u32, out_width: u32, out_height: u32) -> Self {
        Self {
            in_width,
            in_height,
            out_width,
            out_height,
            filter: crate::filter::Filter::default(),
            input: PixelDescriptor::RGBA8_SRGB,
            output: None,
            sharpen: 0.0,
            post_blur_sigma: 0.0,
            filter_blur: 1.0,
            linear: true,
            in_stride: 0,
            out_stride: 0,
        }
    }

    /// Set the resampling filter.
    pub fn filter(mut self, filter: crate::filter::Filter) -> Self {
        self.filter = filter;
        self
    }

    /// Set both input and output pixel descriptor.
    pub fn format(mut self, desc: PixelDescriptor) -> Self {
        self.input = desc;
        self.output = Some(desc);
        self
    }

    /// Set input pixel descriptor.
    pub fn input(mut self, desc: PixelDescriptor) -> Self {
        self.input = desc;
        self
    }

    /// Set output pixel descriptor.
    pub fn output(mut self, desc: PixelDescriptor) -> Self {
        self.output = Some(desc);
        self
    }

    /// Set sharpening amount (0.0 = none).
    pub fn sharpen(mut self, amount: f32) -> Self {
        self.sharpen = amount;
        self
    }

    /// Set post-resize Gaussian blur sigma (0.0 = none).
    ///
    /// **Prefer [`filter_blur`](Self::filter_blur) instead.** `filter_blur`
    /// modifies the resampling kernel at zero cost, while `post_blur` runs a
    /// separate Gaussian convolution pass over the output (allocates a
    /// temporary buffer and touches every pixel twice).
    ///
    /// Use `post_blur` only when you need a Gaussian blur that is independent
    /// of the resize ratio (e.g., fixed-sigma denoising).
    pub fn post_blur(mut self, sigma: f32) -> Self {
        self.post_blur_sigma = sigma;
        self
    }

    /// Set filter blur factor (default 1.0).
    ///
    /// Modifies the resampling kernel at weight-computation time — zero
    /// runtime cost (no extra pass). Values > 1.0 blur, < 1.0 sharpen.
    ///
    /// This multiplies the filter's built-in blur value, so it stacks with
    /// "Sharp" filter variants.
    pub fn filter_blur(mut self, factor: f64) -> Self {
        self.filter_blur = factor;
        self
    }

    /// Resize in linear light (correct, default).
    pub fn linear(mut self) -> Self {
        self.linear = true;
        self
    }

    /// Resize in sRGB space (fast, slight quality loss).
    pub fn srgb(mut self) -> Self {
        self.linear = false;
        self
    }

    /// Set input row stride in elements (default: tightly packed).
    pub fn in_stride(mut self, stride: usize) -> Self {
        self.in_stride = stride;
        self
    }

    /// Set output row stride in elements (default: tightly packed).
    pub fn out_stride(mut self, stride: usize) -> Self {
        self.out_stride = stride;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ResizeConfig {
        let output = self.output.unwrap_or(self.input);
        ResizeConfig {
            filter: self.filter,
            in_width: self.in_width,
            in_height: self.in_height,
            out_width: self.out_width,
            out_height: self.out_height,
            input: self.input,
            output,
            sharpen: self.sharpen,
            post_blur_sigma: self.post_blur_sigma,
            filter_blur: self.filter_blur,
            linear: self.linear,
            in_stride: self.in_stride,
            out_stride: self.out_stride,
        }
    }
}

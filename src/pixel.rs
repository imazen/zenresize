//! Pixel format descriptors and color space configuration.

/// Pixel memory layout.
///
/// Describes the number of channels and how alpha is handled.
/// Channel order doesn't matter — RGBA, BGRA, ARGB all work identically
/// because the sRGB transfer function is the same for R, G, and B,
/// and the convolution kernels just operate on N floats per pixel.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PixelLayout {
    /// Single-channel grayscale.
    Gray,
    /// 3-channel (RGB, BGR, etc.). No alpha.
    Rgb,
    /// 4-channel with padding (RGBX, BGRX). The 4th channel is processed
    /// identically to the others — it's not skipped, just not treated as alpha.
    Rgbx,
    /// 4-channel with straight (non-premultiplied) alpha.
    /// The pipeline premultiplies before filtering and unpremultiplies after.
    Rgba,
    /// 4-channel with premultiplied alpha.
    /// Skips premultiply/unpremultiply — data is filtered as-is.
    RgbaPremul,
}

impl PixelLayout {
    /// Number of channels per pixel.
    #[inline]
    pub fn channels(&self) -> u8 {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgbx | Self::Rgba | Self::RgbaPremul => 4,
        }
    }

    /// Whether the format carries meaningful alpha.
    #[inline]
    pub fn has_alpha(&self) -> bool {
        matches!(self, Self::Rgba | Self::RgbaPremul)
    }

    /// Whether the pipeline needs to premultiply/unpremultiply alpha.
    ///
    /// True only for straight alpha ([`Rgba`](Self::Rgba)).
    /// Premultiplied and non-alpha layouts skip this step.
    #[inline]
    pub fn needs_premultiply(&self) -> bool {
        matches!(self, Self::Rgba)
    }

    /// Whether the data is already premultiplied.
    #[inline]
    pub fn is_premultiplied(&self) -> bool {
        matches!(self, Self::RgbaPremul)
    }

    /// Whether the last channel is alpha (and should skip sRGB linearization).
    #[inline]
    pub fn alpha_is_last_channel(&self) -> bool {
        matches!(self, Self::Rgba | Self::RgbaPremul)
    }
}

/// Pixel format descriptor.
///
/// Combines a data type (u8 sRGB or f32 linear) with a [`PixelLayout`].
///
/// The pipeline is channel-order-agnostic: RGBA, BGRA, ARGB all work
/// identically because the sRGB transfer function is the same for R, G, and B,
/// and the convolution kernels just operate on N floats per pixel.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// sRGB gamma-encoded u8 pixels (the common case).
    ///
    /// Channel order doesn't matter — RGBA, BGRA, ARGB all work identically.
    Srgb8(PixelLayout),
    /// Linear light f32 pixels (HDR, scientific, pipeline use).
    LinearF32(PixelLayout),
}

impl PixelFormat {
    /// The pixel layout.
    #[inline]
    pub fn layout(&self) -> PixelLayout {
        match self {
            Self::Srgb8(layout) | Self::LinearF32(layout) => *layout,
        }
    }

    /// Number of channels per pixel.
    #[inline]
    pub fn channels(&self) -> u8 {
        self.layout().channels()
    }

    /// Whether the format carries meaningful alpha.
    #[inline]
    pub fn has_alpha(&self) -> bool {
        self.layout().has_alpha()
    }

    /// Components per pixel (same as channels).
    #[inline]
    pub fn components_per_pixel(&self) -> usize {
        self.channels() as usize
    }

    /// Whether this is a u8-based format.
    #[inline]
    pub fn is_u8(&self) -> bool {
        matches!(self, Self::Srgb8(..))
    }

    /// Whether this is an f32-based format.
    #[inline]
    pub fn is_f32(&self) -> bool {
        matches!(self, Self::LinearF32(..))
    }

    /// Whether this format uses sRGB gamma encoding.
    #[inline]
    pub fn is_srgb(&self) -> bool {
        matches!(self, Self::Srgb8(..))
    }

    /// Whether this format uses linear light values.
    #[inline]
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::LinearF32(..))
    }
}

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
    /// Input pixel format.
    pub input_format: PixelFormat,
    /// Output pixel format.
    pub output_format: PixelFormat,
    /// Sharpening amount (0.0 = none).
    pub sharpen: f32,
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
        if self.in_width == 0 || self.in_height == 0 {
            return Err("input dimensions must be positive");
        }
        if self.out_width == 0 || self.out_height == 0 {
            return Err("output dimensions must be positive");
        }
        if self.input_format.layout() != self.output_format.layout() {
            return Err("input and output must have same layout");
        }
        Ok(())
    }

    /// Effective input row stride in elements.
    pub fn effective_in_stride(&self) -> usize {
        if self.in_stride == 0 {
            self.in_width as usize * self.input_format.components_per_pixel()
        } else {
            self.in_stride
        }
    }

    /// Effective output row stride in elements.
    pub fn effective_out_stride(&self) -> usize {
        if self.out_stride == 0 {
            self.out_width as usize * self.output_format.components_per_pixel()
        } else {
            self.out_stride
        }
    }

    /// Pixel-packed input row length (no padding).
    pub fn input_row_len(&self) -> usize {
        self.in_width as usize * self.input_format.components_per_pixel()
    }

    /// Pixel-packed output row length (no padding).
    pub fn output_row_len(&self) -> usize {
        self.out_width as usize * self.output_format.components_per_pixel()
    }

    /// Whether linear-light processing is needed.
    ///
    /// True when `linear` is set and the input is sRGB u8.
    /// Premultiplied alpha layouts skip linearization (linearizing
    /// premultiplied sRGB data is mathematically incorrect).
    pub fn needs_linearization(&self) -> bool {
        self.linear && self.input_format.is_srgb() && !self.input_format.layout().is_premultiplied()
    }
}

/// Builder for [`ResizeConfig`].
///
/// # Example
/// ```
/// use zenresize::{ResizeConfig, Filter, PixelFormat, PixelLayout};
///
/// let config = ResizeConfig::builder(1024, 768, 512, 384)
///     .filter(Filter::Lanczos)
///     .format(PixelFormat::Srgb8(PixelLayout::Rgba))
///     .linear()
///     .build();
/// ```
pub struct ResizeConfigBuilder {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    filter: crate::filter::Filter,
    input_format: PixelFormat,
    output_format: Option<PixelFormat>,
    sharpen: f32,
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
            input_format: PixelFormat::Srgb8(PixelLayout::Rgba),
            output_format: None,
            sharpen: 0.0,
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

    /// Set both input and output pixel format.
    pub fn format(mut self, format: PixelFormat) -> Self {
        self.input_format = format;
        self.output_format = Some(format);
        self
    }

    /// Set input pixel format separately.
    pub fn input_format(mut self, format: PixelFormat) -> Self {
        self.input_format = format;
        self
    }

    /// Set output pixel format separately.
    pub fn output_format(mut self, format: PixelFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    /// Set sharpening amount (0.0 = none).
    pub fn sharpen(mut self, amount: f32) -> Self {
        self.sharpen = amount;
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
        let output_format = self.output_format.unwrap_or(self.input_format);
        ResizeConfig {
            filter: self.filter,
            in_width: self.in_width,
            in_height: self.in_height,
            out_width: self.out_width,
            out_height: self.out_height,
            input_format: self.input_format,
            output_format,
            sharpen: self.sharpen,
            linear: self.linear,
            in_stride: self.in_stride,
            out_stride: self.out_stride,
        }
    }
}

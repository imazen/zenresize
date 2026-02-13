//! Pixel format descriptors and color space configuration.

/// Pixel format descriptor.
///
/// The pipeline is channel-order-agnostic: RGBA, BGRA, ARGB all work
/// identically because the sRGB transfer function is the same for R, G, and B,
/// and the convolution kernels just operate on N floats per pixel.
///
/// For BGRA: use `Srgb8 { channels: 4, has_alpha: true }` — same as RGBA.
/// For BGRX: use `Srgb8 { channels: 4, has_alpha: false }` — X is just a padding channel.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// sRGB gamma-encoded u8 pixels.
    Srgb8 { channels: u8, has_alpha: bool },
    /// Linear light u8 pixels (rare, for pre-linearized data).
    Linear8 { channels: u8, has_alpha: bool },
    /// Linear light f32 pixels.
    LinearF32 { channels: u8, has_alpha: bool },
    /// sRGB gamma-encoded f32 pixels (for fast sRGB-space resize).
    SrgbF32 { channels: u8, has_alpha: bool },
}

impl PixelFormat {
    /// Number of channels per pixel.
    #[inline]
    pub fn channels(&self) -> u8 {
        match self {
            Self::Srgb8 { channels, .. }
            | Self::Linear8 { channels, .. }
            | Self::LinearF32 { channels, .. }
            | Self::SrgbF32 { channels, .. } => *channels,
        }
    }

    /// Whether the format has an alpha channel.
    #[inline]
    pub fn has_alpha(&self) -> bool {
        match self {
            Self::Srgb8 { has_alpha, .. }
            | Self::Linear8 { has_alpha, .. }
            | Self::LinearF32 { has_alpha, .. }
            | Self::SrgbF32 { has_alpha, .. } => *has_alpha,
        }
    }

    /// Components per pixel (same as channels).
    #[inline]
    pub fn components_per_pixel(&self) -> usize {
        self.channels() as usize
    }

    /// Whether this is a u8-based format.
    #[inline]
    pub fn is_u8(&self) -> bool {
        matches!(self, Self::Srgb8 { .. } | Self::Linear8 { .. })
    }

    /// Whether this is an f32-based format.
    #[inline]
    pub fn is_f32(&self) -> bool {
        matches!(self, Self::LinearF32 { .. } | Self::SrgbF32 { .. })
    }

    /// Whether this format uses sRGB gamma encoding.
    #[inline]
    pub fn is_srgb(&self) -> bool {
        matches!(self, Self::Srgb8 { .. } | Self::SrgbF32 { .. })
    }

    /// Whether this format uses linear light values.
    #[inline]
    pub fn is_linear(&self) -> bool {
        matches!(self, Self::Linear8 { .. } | Self::LinearF32 { .. })
    }
}

/// Color space for resize computation.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorSpace {
    /// Resize in linear light (correct, handles gamma properly).
    #[default]
    Linear,
    /// Resize in sRGB space (fast, slight quality loss on gradients).
    Srgb,
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
    /// Color space for resize computation.
    pub color_space: ColorSpace,
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
        let in_ch = self.input_format.channels();
        let out_ch = self.output_format.channels();
        if !(1..=4).contains(&in_ch) {
            return Err("input channels must be 1-4");
        }
        if !(1..=4).contains(&out_ch) {
            return Err("output channels must be 1-4");
        }
        if in_ch != out_ch {
            return Err("input and output must have same channel count");
        }
        if self.input_format.has_alpha() != self.output_format.has_alpha() {
            return Err("input and output must agree on alpha");
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
    pub fn needs_linearization(&self) -> bool {
        self.color_space == ColorSpace::Linear && self.input_format.is_srgb()
    }
}

/// Builder for [`ResizeConfig`].
///
/// # Example
/// ```
/// use zenresize::{ResizeConfig, Filter, PixelFormat};
///
/// let config = ResizeConfig::builder(1024, 768, 512, 384)
///     .filter(Filter::Lanczos)
///     .format(PixelFormat::Srgb8 { channels: 4, has_alpha: true })
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
    color_space: ColorSpace,
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
            input_format: PixelFormat::Srgb8 {
                channels: 4,
                has_alpha: true,
            },
            output_format: None,
            sharpen: 0.0,
            color_space: ColorSpace::Linear,
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
        self.color_space = ColorSpace::Linear;
        self
    }

    /// Resize in sRGB space (fast, slight quality loss).
    pub fn srgb(mut self) -> Self {
        self.color_space = ColorSpace::Srgb;
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
            color_space: self.color_space,
            in_stride: self.in_stride,
            out_stride: self.out_stride,
        }
    }
}

//! Pixel format descriptors and color space configuration.

/// Pixel format descriptor.
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

    /// Bytes per pixel for u8 formats, or floats per pixel for f32 formats.
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum ColorSpace {
    /// Resize in linear light (correct, handles gamma properly).
    #[default]
    Linear,
    /// Resize in sRGB space (fast, slight quality loss on gradients).
    Srgb,
}

/// Resize configuration.
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
}

impl ResizeConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.in_width == 0 || self.in_height == 0 {
            return Err("input dimensions must be positive");
        }
        if self.out_width == 0 || self.out_height == 0 {
            return Err("output dimensions must be positive");
        }
        if self.input_format.channels() < 1 || self.input_format.channels() > 4 {
            return Err("input channels must be 1-4");
        }
        if self.output_format.channels() < 1 || self.output_format.channels() > 4 {
            return Err("output channels must be 1-4");
        }
        if self.input_format.channels() != self.output_format.channels() {
            return Err("input and output must have same channel count");
        }
        if self.input_format.has_alpha() != self.output_format.has_alpha() {
            return Err("input and output must agree on alpha");
        }
        Ok(())
    }

    /// Bytes per input row for u8 formats.
    pub fn input_row_bytes(&self) -> usize {
        self.in_width as usize * self.input_format.components_per_pixel()
    }

    /// Bytes per output row for u8 formats.
    pub fn output_row_bytes(&self) -> usize {
        self.out_width as usize * self.output_format.components_per_pixel()
    }

    /// Whether linear-light processing is needed.
    pub fn needs_linearization(&self) -> bool {
        self.color_space == ColorSpace::Linear && self.input_format.is_srgb()
    }
}

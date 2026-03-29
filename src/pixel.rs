//! Pixel format descriptors, element types, and color space configuration.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

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
// LobeRatio
// =============================================================================

/// Negative-lobe ratio control for resampling weights.
///
/// Controls the balance between positive and negative resampling weights.
/// The ratio `r = |negative_sum| / positive_sum` determines overshoot
/// and ringing behavior. Higher ratios produce sharper edges but more
/// ringing artifacts; lower ratios produce smoother output.
///
/// # Weight profiles at different ratios (Lanczos, natural ≈ 0.067)
///
/// ```text
///    ratio=0.0 (flatten)     ratio=0.067 (natural)    ratio=0.15 (sharpen)
///
///  1.0 |    ****              1.0 |    ***              1.2 |    **
///      |   *    *                 |   *   *                 |   *  *
///  0.5 |  *      *            0.5 |  *     *            0.6 |  *    *
///      | *        *               | *       *               | *      *
///  0.0 +*----------*--        0.0 +*----*----*--         0.0 +*--*----*--*
///      |                          |     *    *               |    *    *
///      |                     -0.1 |                    -0.2  |    *    *
/// ```
///
/// # Natural ratios for common filters
///
/// | Filter       | Natural |   | Filter       | Natural |
/// |--------------|---------|---|--------------|---------|
/// | Box          |  0.000  |   | Mitchell     |  0.013  |
/// | Triangle     |  0.000  |   | CatmullRom   |  0.065  |
/// | Hermite      |  0.000  |   | Lanczos      |  0.067  |
/// | CubicBSpline |  0.000  |   | Lanczos2     |  0.043  |
/// | Robidoux     |  0.009  |   | RobidouxSharp|  0.033  |
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub enum LobeRatio {
    /// Use the filter's natural negative-lobe ratio (default).
    #[default]
    Natural,

    /// Set the exact target ratio (bidirectional).
    ///
    /// - `0.0` — flatten: zero all negative lobes (maximum smoothness)
    /// - Below natural — reduce negative lobes (softer)
    /// - Above natural — amplify negative lobes (sharper, more ringing)
    ///
    /// Range: `0.0` to `< 1.0`. Typical values: `0.0`–`0.5`.
    Exact(f32),

    /// Imageflow-compatible sharpening (amplify only, percentage 0–100).
    ///
    /// Sets an absolute target ratio of `pct / 100`. If the filter's
    /// natural ratio already meets or exceeds this target, it's a no-op.
    /// Can only amplify negative lobes, never reduce them.
    ///
    /// Matches imageflow's `f.sharpen` / `sharpen_percent_goal` exactly
    /// (see `imageflow_core/src/graphics/weights.rs:648-650`):
    /// ```text
    /// desired = min(1.0, max(natural_ratio, goal / 100))
    /// // applied only when desired > natural_ratio
    /// ```
    ///
    /// Examples with Lanczos (natural ≈ 6.7%):
    /// - `SharpenPercent(5.0)` — no-op (5% < 6.7% natural)
    /// - `SharpenPercent(15.0)` — amplifies to 15%
    /// - `SharpenPercent(50.0)` — amplifies to 50% (aggressive)
    ///
    /// Use [`Exact`](Self::Exact) for bidirectional control (can also
    /// reduce or flatten).
    SharpenPercent(f32),
}

// =============================================================================
// SourceRegion
// =============================================================================

/// Region of the source image to use for resizing.
///
/// When set on a [`ResizeConfig`], only the specified rectangle is read from the
/// input. The streaming API still accepts full-width input rows; the resizer
/// extracts the region internally. Rows outside the vertical range are skipped.
///
/// # Example
///
/// ```
/// use zenresize::SourceRegion;
///
/// // Extract a 400×300 region starting at (100, 50)
/// let region = SourceRegion { x: 100, y: 50, width: 400, height: 300 };
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRegion {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width of the region in pixels.
    pub width: u32,
    /// Height of the region in pixels.
    pub height: u32,
}

// =============================================================================
// Padding
// =============================================================================

/// Padding to add around the resized output.
///
/// The final output dimensions become:
/// - width: `left + out_width + right`
/// - height: `top + out_height + bottom`
///
/// Padding pixels are filled with [`color`](Self::color). Works with all pixel
/// formats (u8, u16, f32) and all channel counts.
///
/// # Example
///
/// ```
/// use zenresize::Padding;
///
/// // 10px black border on all sides
/// let pad = Padding::uniform(10).with_color([0.0, 0.0, 0.0, 1.0]);
///
/// // Asymmetric letterboxing
/// let pad = Padding { top: 40, bottom: 40, left: 0, right: 0, color: [0.0; 4] };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Padding {
    /// Rows of padding above the content.
    pub top: u32,
    /// Rows of padding below the content.
    pub bottom: u32,
    /// Columns of padding left of the content.
    pub left: u32,
    /// Columns of padding right of the content.
    pub right: u32,
    /// Padding color per channel, 0.0–1.0 in the output's color space.
    ///
    /// For sRGB u8 output, 0.5 → 128. For linear f32, 0.5 → 0.5.
    /// Only the first `channels` values are used.
    pub color: [f32; 4],
}

impl Default for Padding {
    fn default() -> Self {
        Self {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
            color: [0.0; 4],
        }
    }
}

impl Padding {
    /// Create uniform padding on all sides.
    pub fn uniform(px: u32) -> Self {
        Self {
            top: px,
            bottom: px,
            left: px,
            right: px,
            color: [0.0; 4],
        }
    }

    /// Set the padding color.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Whether all padding values are zero.
    pub fn is_empty(&self) -> bool {
        self.top == 0 && self.bottom == 0 && self.left == 0 && self.right == 0
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
    /// Optional override filter for upscaling. When `None`, the `filter` field
    /// is used for both up and down. When `Some`, this filter is used when
    /// the output dimension exceeds the input dimension on a given axis.
    pub up_filter: Option<crate::filter::Filter>,
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
    /// Post-resize unsharp mask amount (0.0 = none).
    ///
    /// Runs a separate unsharp-mask pass over the output.
    /// Consider [`resize_sharpen`](ResizeConfigBuilder::resize_sharpen)
    /// for zero-cost sharpening during resampling.
    pub post_sharpen: f32,
    /// Post-resize Gaussian blur sigma (0.0 = none).
    ///
    /// Runs a separate all-positive Gaussian convolution pass.
    /// Not equivalent to [`kernel_width_scale`](ResizeConfigBuilder::kernel_width_scale),
    /// which stretches the resampling kernel including its negative lobes.
    pub post_blur_sigma: f32,
    /// Kernel width scale factor (`None` = use preset default).
    ///
    /// Multiplied with the filter preset's built-in blur. `> 1.0` widens the
    /// kernel (softer), `< 1.0` narrows it (sharper, more aliasing risk).
    /// Zero cost — applied during weight computation.
    pub kernel_width_scale: Option<f64>,
    /// Negative-lobe ratio control. See [`LobeRatio`] for details.
    ///
    /// Zero cost — applied during weight computation.
    pub lobe_ratio: LobeRatio,
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
    /// Source region to crop from input before resizing.
    ///
    /// When set, only the specified rectangle of the input is used.
    /// The streaming API still accepts full-width rows and extracts the
    /// region internally. Rows outside the vertical range are skipped.
    pub source_region: Option<SourceRegion>,
    /// Padding to add around the resized output.
    ///
    /// When set, the output is padded with the specified color on all sides.
    /// The total output dimensions become `(left + out_width + right)` by
    /// `(top + out_height + bottom)`.
    pub padding: Option<Padding>,
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
        // Validate source region
        if let Some(ref r) = self.source_region {
            if r.width == 0 || r.height == 0 {
                return Err("source region dimensions must be positive");
            }
            if r.x
                .checked_add(r.width)
                .is_none_or(|end| end > self.in_width)
            {
                return Err("source region exceeds input width");
            }
            if r.y
                .checked_add(r.height)
                .is_none_or(|end| end > self.in_height)
            {
                return Err("source region exceeds input height");
            }
        }
        // Validate padding
        if let Some(ref p) = self.padding {
            if p.left
                .checked_add(self.out_width)
                .and_then(|v| v.checked_add(p.right))
                .is_none()
            {
                return Err("padded output width overflows u32");
            }
            if p.top
                .checked_add(self.out_height)
                .and_then(|v| v.checked_add(p.bottom))
                .is_none()
            {
                return Err("padded output height overflows u32");
            }
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

    /// Effective input width for the resize kernel (crop width if set, else full width).
    pub fn resize_in_width(&self) -> u32 {
        self.source_region
            .as_ref()
            .map_or(self.in_width, |r| r.width)
    }

    /// Effective input height for the resize kernel (crop height if set, else full height).
    pub fn resize_in_height(&self) -> u32 {
        self.source_region
            .as_ref()
            .map_or(self.in_height, |r| r.height)
    }

    /// Total output width including padding.
    pub fn total_output_width(&self) -> u32 {
        let base = self.out_width;
        self.padding
            .as_ref()
            .map_or(base, |p| p.left + base + p.right)
    }

    /// Total output height including padding.
    pub fn total_output_height(&self) -> u32 {
        let base = self.out_height;
        self.padding
            .as_ref()
            .map_or(base, |p| p.top + base + p.bottom)
    }

    /// Row length for the total output including padding.
    pub fn total_output_row_len(&self) -> usize {
        self.total_output_width() as usize * self.output.channels()
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
    up_filter: Option<crate::filter::Filter>,
    input: PixelDescriptor,
    output: Option<PixelDescriptor>,
    post_sharpen: f32,
    post_blur_sigma: f32,
    kernel_width_scale: Option<f64>,
    lobe_ratio: LobeRatio,
    linear: bool,
    in_stride: usize,
    out_stride: usize,
    source_region: Option<SourceRegion>,
    padding: Option<Padding>,
}

impl ResizeConfigBuilder {
    fn new(in_width: u32, in_height: u32, out_width: u32, out_height: u32) -> Self {
        Self {
            in_width,
            in_height,
            out_width,
            out_height,
            filter: crate::filter::Filter::default(),
            up_filter: None,
            input: PixelDescriptor::RGBA8_SRGB,
            output: None,
            post_sharpen: 0.0,
            post_blur_sigma: 0.0,
            kernel_width_scale: None,
            lobe_ratio: LobeRatio::Natural,
            linear: true,
            in_stride: 0,
            out_stride: 0,
            source_region: None,
            padding: None,
        }
    }

    /// Set the resampling filter.
    pub fn filter(mut self, filter: crate::filter::Filter) -> Self {
        self.filter = filter;
        self
    }

    /// Set a separate filter for upscaling. By default, the main `filter`
    /// is used for both up and down operations.
    pub fn up_filter(mut self, filter: crate::filter::Filter) -> Self {
        self.up_filter = Some(filter);
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

    /// Set post-resize unsharp mask amount (0.0 = none).
    ///
    /// Runs a separate pass after resampling. For zero-cost sharpening
    /// during resampling, use [`resize_sharpen`](Self::resize_sharpen).
    pub fn post_sharpen(mut self, amount: f32) -> Self {
        self.post_sharpen = amount;
        self
    }

    /// Deprecated alias for [`post_sharpen`](Self::post_sharpen).
    #[doc(hidden)]
    pub fn sharpen(mut self, amount: f32) -> Self {
        self.post_sharpen = amount;
        self
    }

    /// Set post-resize Gaussian blur sigma (0.0 = none).
    ///
    /// Runs a separate all-positive Gaussian convolution pass. Not equivalent
    /// to [`kernel_width_scale`](Self::kernel_width_scale) — Gaussian blur
    /// kills high frequencies more aggressively and has no negative lobes.
    pub fn post_blur(mut self, sigma: f32) -> Self {
        self.post_blur_sigma = sigma;
        self
    }

    /// Scale the resampling kernel width (zero cost, default 1.0).
    ///
    /// `> 1.0`: wider kernel, softer output. `< 1.0`: narrower, sharper.
    /// Multiplied with the filter preset's built-in blur value.
    /// Can be combined with [`lobe_ratio`](Self::lobe_ratio).
    pub fn kernel_width_scale(mut self, factor: f64) -> Self {
        self.kernel_width_scale = Some(factor);
        self
    }

    /// Set negative-lobe ratio control (zero cost).
    ///
    /// See [`LobeRatio`] for variants and documentation.
    /// Can be combined with [`kernel_width_scale`](Self::kernel_width_scale).
    pub fn lobe_ratio(mut self, ratio: LobeRatio) -> Self {
        self.lobe_ratio = ratio;
        self
    }

    /// Sharpen during resampling by amplifying negative lobes (zero cost).
    ///
    /// `pct` is a percentage (0–100). Sets an absolute target ratio of
    /// `pct / 100`. Only amplifies — values below the filter's natural
    /// negative-lobe ratio are no-ops. For example, Lanczos has a natural
    /// ratio of ~6.7%, so `resize_sharpen(5.0)` is a no-op but
    /// `resize_sharpen(15.0)` amplifies.
    ///
    /// Shorthand for `.lobe_ratio(LobeRatio::SharpenPercent(pct))`.
    /// See [`LobeRatio::SharpenPercent`] for details.
    pub fn resize_sharpen(mut self, pct: f32) -> Self {
        self.lobe_ratio = LobeRatio::SharpenPercent(pct);
        self
    }

    /// Set negative-lobe amplification as a percentage (imageflow compat).
    ///
    /// Alias for [`resize_sharpen`](Self::resize_sharpen).
    #[doc(hidden)]
    pub fn sharpen_percent(self, pct: f32) -> Self {
        self.resize_sharpen(pct)
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

    /// Crop a region from the input before resizing.
    ///
    /// Only the specified rectangle of the input is used. The streaming API
    /// still accepts full-width rows; the resizer extracts the region internally.
    ///
    /// The resize kernel maps this region to the output dimensions.
    pub fn crop(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.source_region = Some(SourceRegion {
            x,
            y,
            width,
            height,
        });
        self
    }

    /// Set the source region (alternative to [`crop`](Self::crop)).
    pub fn source_region(mut self, region: SourceRegion) -> Self {
        self.source_region = Some(region);
        self
    }

    /// Add padding around the resized output.
    ///
    /// The final output dimensions become `(left + out_width + right)` by
    /// `(top + out_height + bottom)`. Padding is filled with transparent black
    /// by default; use [`padding_color`](Self::padding_color) to change.
    pub fn padding(mut self, top: u32, right: u32, bottom: u32, left: u32) -> Self {
        let color = self.padding.as_ref().map_or([0.0; 4], |p| p.color);
        self.padding = Some(Padding {
            top,
            bottom,
            left,
            right,
            color,
        });
        self
    }

    /// Add uniform padding on all sides.
    pub fn padding_uniform(self, px: u32) -> Self {
        self.padding(px, px, px, px)
    }

    /// Set the padding color (0.0–1.0 per channel in the output's color space).
    ///
    /// Must be called after [`padding`](Self::padding) or [`padding_uniform`](Self::padding_uniform).
    pub fn padding_color(mut self, color: [f32; 4]) -> Self {
        if let Some(ref mut p) = self.padding {
            p.color = color;
        } else {
            self.padding = Some(Padding {
                top: 0,
                bottom: 0,
                left: 0,
                right: 0,
                color,
            });
        }
        self
    }

    /// Set a [`Padding`] struct directly.
    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = Some(padding);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ResizeConfig {
        let output = self.output.unwrap_or(self.input);
        ResizeConfig {
            filter: self.filter,
            up_filter: self.up_filter,
            in_width: self.in_width,
            in_height: self.in_height,
            out_width: self.out_width,
            out_height: self.out_height,
            input: self.input,
            output,
            post_sharpen: self.post_sharpen,
            post_blur_sigma: self.post_blur_sigma,
            kernel_width_scale: self.kernel_width_scale,
            lobe_ratio: self.lobe_ratio,
            linear: self.linear,
            in_stride: self.in_stride,
            out_stride: self.out_stride,
            source_region: self.source_region,
            padding: self.padding,
        }
    }
}

//! Full-frame resize via [`Resizer`].
//!
//! `Resizer` pre-computes weight tables and allocates intermediate buffers once,
//! then reuses them across calls. For a single resize, `Resizer::new(&config).resize(&input)`
//! is equivalent to the former one-shot functions.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::color;
use crate::composite::{Background, CompositeError, NoBackground};
use crate::filter::InterpolationDetails;
use crate::pixel::{ConfigError, ResizeConfig};
use crate::proven;
use crate::simd;
use crate::streaming::StreamingResize;
use crate::transfer::{Bt709, Hlg, Pq, TransferCurve};
use crate::weights::{F32WeightTable, I16WeightTable};
use whereat::At;
use zenpixels::{ChannelType, TransferFunction};

// =============================================================================
// Transfer-function-aware decode/encode helpers
// =============================================================================

/// Decode a u8 row to f32 using the specified transfer function.
fn decode_u8_row(
    src: &[u8],
    dst: &mut [f32],
    tf: TransferFunction,
    channels: usize,
    has_alpha: bool,
) {
    match tf {
        TransferFunction::Srgb => color::srgb_u8_to_linear_f32(src, dst, channels, has_alpha),
        TransferFunction::Linear => simd::u8_to_f32_row(src, dst),
        TransferFunction::Bt709 => {
            Bt709.u8_to_linear_f32(src, dst, &(), channels, has_alpha, false)
        }
        TransferFunction::Pq => Pq.u8_to_linear_f32(src, dst, &(), channels, has_alpha, false),
        TransferFunction::Hlg => Hlg.u8_to_linear_f32(src, dst, &(), channels, has_alpha, false),
        _ => simd::u8_to_f32_row(src, dst), // Unknown → identity
    }
}

/// Encode f32 to u8 row using the specified transfer function.
fn encode_u8_row(
    src: &[f32],
    dst: &mut [u8],
    tf: TransferFunction,
    channels: usize,
    has_alpha: bool,
) {
    match tf {
        TransferFunction::Srgb => color::linear_f32_to_srgb_u8(src, dst, channels, has_alpha),
        TransferFunction::Linear => simd::f32_to_u8_row(src, dst),
        TransferFunction::Bt709 => {
            Bt709.linear_f32_to_u8(src, dst, &(), channels, has_alpha, false)
        }
        TransferFunction::Pq => Pq.linear_f32_to_u8(src, dst, &(), channels, has_alpha, false),
        TransferFunction::Hlg => Hlg.linear_f32_to_u8(src, dst, &(), channels, has_alpha, false),
        _ => simd::f32_to_u8_row(src, dst), // Unknown → identity
    }
}

/// Reusable resizer with pre-computed weight tables.
///
/// When resizing many images with the same dimensions and filter,
/// `Resizer` avoids recomputing weight tables on each call. This
/// saves significant overhead for repeated resize operations.
///
/// For single-use, `Resizer::new(&config).resize(&input)` has the same
/// performance as creating and discarding a one-off resize — both compute
/// weights and allocate intermediates exactly once.
///
/// The generic parameter `B` controls background compositing. The default
/// [`NoBackground`] eliminates all composite code at compile time (zero overhead).
/// Use [`with_background()`](Self::with_background) to enable source-over compositing.
///
/// # Example
///
/// ```
/// use zenresize::{Resizer, ResizeConfig, Filter, PixelDescriptor};
///
/// let config = ResizeConfig::builder(20, 20, 10, 10)
///     .filter(Filter::Lanczos)
///     .format(PixelDescriptor::RGBA8_SRGB)
///     .build();
///
/// let mut resizer = Resizer::new(&config);
/// let input = vec![128u8; 20 * 20 * 4];
/// let output = resizer.resize(&input);
/// assert_eq!(output.len(), 10 * 10 * 4);
/// ```
pub struct Resizer<B: Background = NoBackground> {
    config: ResizeConfig,
    /// Cached streaming resizer — created once, reset per resize() call.
    stream: crate::streaming::StreamingResize<B>,
}

impl Resizer<NoBackground> {
    /// Create a new resizer for the given configuration.
    /// Pre-computes weight tables.
    ///
    /// # Panics
    ///
    /// Panics if `config` fails [`ResizeConfig::validate`]. When `config` is
    /// derived from untrusted input, use [`try_new`](Self::try_new) instead.
    pub fn new(config: &ResizeConfig) -> Self {
        Resizer {
            config: config.clone(),
            stream: StreamingResize::new(config),
        }
    }

    /// Fallible constructor: returns the [`ResizeConfig::validate`] error
    /// instead of panicking. Prefer this when dimensions come from untrusted
    /// input.
    pub fn try_new(config: &ResizeConfig) -> Result<Self, At<ConfigError>> {
        Ok(Resizer {
            config: config.clone(),
            stream: StreamingResize::try_new(config)?,
        })
    }
}

impl<B: Background> Resizer<B> {
    /// Create a resizer with background compositing.
    ///
    /// Performs source-over compositing between the resized foreground and the
    /// given background. The compositing happens in premultiplied linear f32
    /// space, between the vertical filter and unpremultiply.
    ///
    /// # Path override
    ///
    /// Compositing requires per-row f32 access to the v-filter output. If the
    /// optimal path would be i16 (sRGB or linear), this forces the f32 path
    /// instead.
    ///
    /// # Errors
    ///
    /// Returns [`CompositeError::PremultipliedInput`] if the input format
    /// is `RgbaPremul` (compositing premultiplied input is mathematically incorrect).
    pub fn with_background(
        config: &ResizeConfig,
        background: B,
    ) -> Result<Self, At<CompositeError>> {
        let stream = StreamingResize::with_background(config, background)?;
        Ok(Resizer {
            config: config.clone(),
            stream,
        })
    }

    /// Mutable reference to the background.
    pub fn background_mut(&mut self) -> &mut B {
        self.stream.background_mut()
    }

    /// Consume the resizer and return the background.
    pub fn into_background(self) -> B {
        self.stream.into_background()
    }

    /// Set the blend mode for compositing.
    ///
    /// Default is [`BlendMode::SrcOver`](crate::BlendMode::SrcOver).
    /// Only meaningful when a background is set via [`with_background`](Self::with_background).
    pub fn with_blend_mode(mut self, mode: crate::composite::BlendMode) -> Self {
        self.stream = self.stream.with_blend_mode(mode);
        self
    }

    /// Resize a u8 image, allocating and returning the output.
    ///
    /// # Panics
    /// Panics if the config uses `LinearF32` format (use [`resize_f32`](Self::resize_f32) instead).
    pub fn resize(&mut self, input: &[u8]) -> Vec<u8> {
        assert!(
            self.config.input.channel_type() == ChannelType::U8,
            "resize() requires Srgb8 format; use resize_f32() for LinearF32 or resize_u16() for Encoded16"
        );
        let out_row_len = self.config.output_row_len();
        let len = self.config.out_height as usize * out_row_len;
        let mut output = proven::alloc_output::<u8>(len);
        self.resize_into(input, &mut output);
        output
    }

    /// Resize a u8 image into a caller-provided buffer.
    ///
    /// # Panics
    /// Panics if the config uses `LinearF32` format.
    pub fn resize_into(&mut self, input: &[u8], output: &mut [u8]) {
        assert!(
            self.config.input.channel_type() == ChannelType::U8,
            "resize_into() requires Srgb8 format; use resize_f32_into() for LinearF32 or resize_u16_into() for Encoded16"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let channels = config.input.channels();
        let in_h = config.in_height as usize;
        let out_h = config.out_height as usize;

        // Reset the cached stream and push/drain all rows through it.
        self.stream.reset();
        let mut out_y = 0usize;
        for y in 0..in_h {
            self.stream
                .push_row(&input[y * in_stride..y * in_stride + in_row_len])
                .expect("push_row failed in fullframe delegation");
            while let Some(row) = self.stream.next_output_row() {
                let start = out_y * out_row_len;
                output[start..start + out_row_len].copy_from_slice(row);
                out_y += 1;
            }
        }
        let remaining = self.stream.finish();
        for _ in 0..remaining {
            let row = self
                .stream
                .next_output_row()
                .expect("finish promised remaining rows");
            let start = out_y * out_row_len;
            output[start..start + out_row_len].copy_from_slice(row);
            out_y += 1;
        }
        debug_assert_eq!(out_y, out_h);

        // Post-resize sharpening (unsharp mask).
        if config.post_sharpen > 0.0 {
            crate::blur::unsharp_mask_u8(
                output,
                config.out_width,
                config.out_height,
                channels,
                config.post_sharpen,
                config.post_sharpen * 0.5 + 0.5, // sigma scales with amount
            );
        }

        // Post-resize blur (applies after sharpening).
        if config.post_blur_sigma > 0.0 {
            crate::blur::blur_u8(
                output,
                config.out_width,
                config.out_height,
                channels,
                config.post_blur_sigma,
            );
        }
    }

    /// Resize an f32 image, allocating and returning the output.
    ///
    /// # Panics
    /// Panics if the config uses `Srgb8` format (use [`resize`](Self::resize) instead).
    pub fn resize_f32(&mut self, input: &[f32]) -> Vec<f32> {
        assert!(
            self.config.input.channel_type() == ChannelType::F32,
            "resize_f32() requires LinearF32 format; use resize() for Srgb8 or resize_u16() for Encoded16"
        );
        let out_row_len = self.config.output_row_len();
        let len = self.config.out_height as usize * out_row_len;
        let mut output = vec![0.0f32; len];
        self.resize_f32_into(input, &mut output);
        output
    }

    /// Resize an f32 image into a caller-provided buffer.
    ///
    /// # Panics
    /// Panics if the config uses `Srgb8` format.
    pub fn resize_f32_into(&mut self, input: &[f32], output: &mut [f32]) {
        assert!(
            self.config.input.channel_type() == ChannelType::F32,
            "resize_f32_into() requires LinearF32 format; use resize_into() for Srgb8 or resize_u16_into() for Encoded16"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let out_channels = config.output.channels();
        let in_h = config.in_height as usize;
        let out_h = config.out_height as usize;

        self.stream.reset();
        let mut out_y = 0usize;
        for y in 0..in_h {
            self.stream
                .push_row_f32(&input[y * in_stride..y * in_stride + in_row_len])
                .expect("push_row_f32 failed in fullframe delegation");
            while let Some(row) = self.stream.next_output_row_f32() {
                let start = out_y * out_row_len;
                output[start..start + out_row_len].copy_from_slice(row);
                out_y += 1;
            }
        }
        let remaining = self.stream.finish();
        for _ in 0..remaining {
            let row = self
                .stream
                .next_output_row_f32()
                .expect("finish promised remaining rows");
            let start = out_y * out_row_len;
            output[start..start + out_row_len].copy_from_slice(row);
            out_y += 1;
        }
        debug_assert_eq!(out_y, out_h);

        // Post-resize sharpening (unsharp mask).
        if config.post_sharpen > 0.0 {
            crate::blur::unsharp_mask_f32(
                output,
                config.out_width,
                config.out_height,
                out_channels,
                config.post_sharpen,
                config.post_sharpen * 0.5 + 0.5,
            );
        }

        // Post-resize blur.
        if config.post_blur_sigma > 0.0 {
            crate::blur::blur_f32(
                output,
                config.out_width,
                config.out_height,
                out_channels,
                config.post_blur_sigma,
            );
        }
    }

    /// Resize a u16 image, allocating and returning the output.
    ///
    /// Uses the sRGB transfer function to linearize u16 values (0-65535) before
    /// filtering in f32 linear-light space, then re-encodes to u16.
    ///
    /// # Panics
    /// Panics if the config doesn't use `Encoded16` format.
    pub fn resize_u16(&mut self, input: &[u16]) -> Vec<u16> {
        assert!(
            self.config.input.channel_type() == ChannelType::U16,
            "resize_u16() requires Encoded16 format"
        );
        let out_row_len = self.config.output_row_len();
        let len = self.config.out_height as usize * out_row_len;
        let mut output = vec![0u16; len];
        self.resize_u16_into(input, &mut output);
        output
    }

    /// Resize a u16 image into a caller-provided buffer.
    ///
    /// # Panics
    /// Panics if the config doesn't use `Encoded16` format.
    pub fn resize_u16_into(&mut self, input: &[u16], output: &mut [u16]) {
        assert!(
            self.config.input.channel_type() == ChannelType::U16,
            "resize_u16_into() requires Encoded16 format"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let in_h = config.in_height as usize;
        let out_h = config.out_height as usize;

        self.stream.reset();
        let mut out_y = 0usize;
        for y in 0..in_h {
            self.stream
                .push_row_u16(&input[y * in_stride..y * in_stride + in_row_len])
                .expect("push_row_u16 failed in fullframe delegation");
            while let Some(row) = self.stream.next_output_row_u16() {
                let start = out_y * out_row_len;
                output[start..start + out_row_len].copy_from_slice(row);
                out_y += 1;
            }
        }
        let remaining = self.stream.finish();
        for _ in 0..remaining {
            let row = self
                .stream
                .next_output_row_u16()
                .expect("finish promised remaining rows");
            let start = out_y * out_row_len;
            output[start..start + out_row_len].copy_from_slice(row);
            out_y += 1;
        }
        debug_assert_eq!(out_y, out_h);
    }

    // =========================================================================
    // Cross-format resize methods
    // =========================================================================

    /// Resize u8 input to f32 output, allocating and returning the output.
    pub fn resize_u8_to_f32(&mut self, input: &[u8]) -> Vec<f32> {
        assert!(
            self.config.input.channel_type() == ChannelType::U8,
            "input must be u8"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::F32,
            "output must be LinearF32"
        );
        let len = self.config.out_height as usize * self.config.output_row_len();
        let mut output = vec![0.0f32; len];
        self.resize_u8_to_f32_into(input, &mut output);
        output
    }

    /// Resize u8 input to f32 output into a caller-provided buffer.
    pub fn resize_u8_to_f32_into(&mut self, input: &[u8], output: &mut [f32]) {
        assert!(
            self.config.input.channel_type() == ChannelType::U8,
            "input must be u8"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::F32,
            "output must be LinearF32"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let in_h = config.in_height as usize;
        let out_h = config.out_height as usize;

        self.stream.reset();
        let mut out_y = 0usize;
        for y in 0..in_h {
            self.stream
                .push_row(&input[y * in_stride..y * in_stride + in_row_len])
                .expect("push_row failed in cross-format resize");
            while let Some(row) = self.stream.next_output_row_f32() {
                let start = out_y * out_row_len;
                output[start..start + out_row_len].copy_from_slice(row);
                out_y += 1;
            }
        }
        let remaining = self.stream.finish();
        for _ in 0..remaining {
            let row = self
                .stream
                .next_output_row_f32()
                .expect("finish promised remaining rows");
            let start = out_y * out_row_len;
            output[start..start + out_row_len].copy_from_slice(row);
            out_y += 1;
        }
        debug_assert_eq!(out_y, out_h);
    }

    /// Resize f32 input to u8 output, allocating and returning the output.
    pub fn resize_f32_to_u8(&mut self, input: &[f32]) -> Vec<u8> {
        assert!(
            self.config.input.channel_type() == ChannelType::F32,
            "input must be f32"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::U8,
            "output must be u8"
        );
        let len = self.config.out_height as usize * self.config.output_row_len();
        let mut output = proven::alloc_output::<u8>(len);
        self.resize_f32_to_u8_into(input, &mut output);
        output
    }

    /// Resize f32 input to u8 output into a caller-provided buffer.
    pub fn resize_f32_to_u8_into(&mut self, input: &[f32], output: &mut [u8]) {
        assert!(
            self.config.input.channel_type() == ChannelType::F32,
            "input must be f32"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::U8,
            "output must be u8"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let in_h = config.in_height as usize;
        let out_h = config.out_height as usize;

        self.stream.reset();
        let mut out_y = 0usize;
        for y in 0..in_h {
            self.stream
                .push_row_f32(&input[y * in_stride..y * in_stride + in_row_len])
                .expect("push_row_f32 failed in cross-format resize");
            while let Some(row) = self.stream.next_output_row() {
                let start = out_y * out_row_len;
                output[start..start + out_row_len].copy_from_slice(row);
                out_y += 1;
            }
        }
        let remaining = self.stream.finish();
        for _ in 0..remaining {
            let row = self
                .stream
                .next_output_row()
                .expect("finish promised remaining rows");
            let start = out_y * out_row_len;
            output[start..start + out_row_len].copy_from_slice(row);
            out_y += 1;
        }
        debug_assert_eq!(out_y, out_h);
    }

    /// Resize u8 input to u16 output, allocating and returning the output.
    pub fn resize_u8_to_u16(&mut self, input: &[u8]) -> Vec<u16> {
        assert!(
            self.config.input.channel_type() == ChannelType::U8,
            "input must be u8"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::U16,
            "output must be Encoded16"
        );
        let len = self.config.out_height as usize * self.config.output_row_len();
        let mut output = vec![0u16; len];
        self.resize_u8_to_u16_into(input, &mut output);
        output
    }

    /// Resize u8 input to u16 output into a caller-provided buffer.
    pub fn resize_u8_to_u16_into(&mut self, input: &[u8], output: &mut [u16]) {
        assert!(
            self.config.input.channel_type() == ChannelType::U8,
            "input must be u8"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::U16,
            "output must be Encoded16"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let in_h = config.in_height as usize;
        let out_h = config.out_height as usize;

        self.stream.reset();
        let mut out_y = 0usize;
        for y in 0..in_h {
            self.stream
                .push_row(&input[y * in_stride..y * in_stride + in_row_len])
                .expect("push_row failed in cross-format resize");
            while let Some(row) = self.stream.next_output_row_u16() {
                let start = out_y * out_row_len;
                output[start..start + out_row_len].copy_from_slice(row);
                out_y += 1;
            }
        }
        let remaining = self.stream.finish();
        for _ in 0..remaining {
            let row = self
                .stream
                .next_output_row_u16()
                .expect("finish promised remaining rows");
            let start = out_y * out_row_len;
            output[start..start + out_row_len].copy_from_slice(row);
            out_y += 1;
        }
        debug_assert_eq!(out_y, out_h);
    }

    /// Resize u16 input to u8 output, allocating and returning the output.
    pub fn resize_u16_to_u8(&mut self, input: &[u16]) -> Vec<u8> {
        assert!(
            self.config.input.channel_type() == ChannelType::U16,
            "input must be u16"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::U8,
            "output must be u8"
        );
        let len = self.config.out_height as usize * self.config.output_row_len();
        let mut output = proven::alloc_output::<u8>(len);
        self.resize_u16_to_u8_into(input, &mut output);
        output
    }

    /// Resize u16 input to u8 output into a caller-provided buffer.
    pub fn resize_u16_to_u8_into(&mut self, input: &[u16], output: &mut [u8]) {
        assert!(
            self.config.input.channel_type() == ChannelType::U16,
            "input must be u16"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::U8,
            "output must be u8"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let in_h = config.in_height as usize;
        let out_h = config.out_height as usize;

        self.stream.reset();
        let mut out_y = 0usize;
        for y in 0..in_h {
            self.stream
                .push_row_u16(&input[y * in_stride..y * in_stride + in_row_len])
                .expect("push_row_u16 failed in cross-format resize");
            while let Some(row) = self.stream.next_output_row() {
                let start = out_y * out_row_len;
                output[start..start + out_row_len].copy_from_slice(row);
                out_y += 1;
            }
        }
        let remaining = self.stream.finish();
        for _ in 0..remaining {
            let row = self
                .stream
                .next_output_row()
                .expect("finish promised remaining rows");
            let start = out_y * out_row_len;
            output[start..start + out_row_len].copy_from_slice(row);
            out_y += 1;
        }
        debug_assert_eq!(out_y, out_h);
    }

    /// Resize u16 input to f32 output, allocating and returning the output.
    pub fn resize_u16_to_f32(&mut self, input: &[u16]) -> Vec<f32> {
        assert!(
            self.config.input.channel_type() == ChannelType::U16,
            "input must be u16"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::F32,
            "output must be LinearF32"
        );
        let len = self.config.out_height as usize * self.config.output_row_len();
        let mut output = vec![0.0f32; len];
        self.resize_u16_to_f32_into(input, &mut output);
        output
    }

    /// Resize u16 input to f32 output into a caller-provided buffer.
    pub fn resize_u16_to_f32_into(&mut self, input: &[u16], output: &mut [f32]) {
        assert!(
            self.config.input.channel_type() == ChannelType::U16,
            "input must be u16"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::F32,
            "output must be LinearF32"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let in_h = config.in_height as usize;
        let out_h = config.out_height as usize;

        self.stream.reset();
        let mut out_y = 0usize;
        for y in 0..in_h {
            self.stream
                .push_row_u16(&input[y * in_stride..y * in_stride + in_row_len])
                .expect("push_row_u16 failed in cross-format resize");
            while let Some(row) = self.stream.next_output_row_f32() {
                let start = out_y * out_row_len;
                output[start..start + out_row_len].copy_from_slice(row);
                out_y += 1;
            }
        }
        let remaining = self.stream.finish();
        for _ in 0..remaining {
            let row = self
                .stream
                .next_output_row_f32()
                .expect("finish promised remaining rows");
            let start = out_y * out_row_len;
            output[start..start + out_row_len].copy_from_slice(row);
            out_y += 1;
        }
        debug_assert_eq!(out_y, out_h);
    }

    /// Resize f32 input to u16 output, allocating and returning the output.
    pub fn resize_f32_to_u16(&mut self, input: &[f32]) -> Vec<u16> {
        assert!(
            self.config.input.channel_type() == ChannelType::F32,
            "input must be f32"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::U16,
            "output must be Encoded16"
        );
        let len = self.config.out_height as usize * self.config.output_row_len();
        let mut output = vec![0u16; len];
        self.resize_f32_to_u16_into(input, &mut output);
        output
    }

    /// Resize f32 input to u16 output into a caller-provided buffer.
    pub fn resize_f32_to_u16_into(&mut self, input: &[f32], output: &mut [u16]) {
        assert!(
            self.config.input.channel_type() == ChannelType::F32,
            "input must be f32"
        );
        assert!(
            self.config.output.channel_type() == ChannelType::U16,
            "output must be Encoded16"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let in_h = config.in_height as usize;
        let out_h = config.out_height as usize;

        self.stream.reset();
        let mut out_y = 0usize;
        for y in 0..in_h {
            self.stream
                .push_row_f32(&input[y * in_stride..y * in_stride + in_row_len])
                .expect("push_row_f32 failed in cross-format resize");
            while let Some(row) = self.stream.next_output_row_u16() {
                let start = out_y * out_row_len;
                output[start..start + out_row_len].copy_from_slice(row);
                out_y += 1;
            }
        }
        let remaining = self.stream.finish();
        for _ in 0..remaining {
            let row = self
                .stream
                .next_output_row_u16()
                .expect("finish promised remaining rows");
            let start = out_y * out_row_len;
            output[start..start + out_row_len].copy_from_slice(row);
            out_y += 1;
        }
        debug_assert_eq!(out_y, out_h);
    }
}

// =============================================================================
// imgref + rgb crate integration
// =============================================================================
//
// All 4-channel pixel types (RGBA, BGRA, ARGB, ABGR) work identically because
// the pipeline is channel-order-agnostic. The `rgb` crate's ComponentSlice trait
// gives us zero-copy &[u8] access to any pixel type.

#[allow(deprecated)] // rgb::ComponentSlice deprecated but still needed for public API compat
mod imgref_impl {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    use crate::pixel::ResizeConfig;
    use crate::streaming::StreamingResize;
    use imgref::{Img, ImgRef, ImgVec};
    use rgb::ComponentSlice;
    use zenpixels::PixelDescriptor;

    /// Resize a 4-channel u8 image. Works with any pixel type that implements
    /// `ComponentSlice` (RGBA, BGRA, ARGB, ABGR from the `rgb` crate).
    ///
    /// Channel order is preserved — the pipeline doesn't care whether
    /// the bytes represent RGBA or BGRA.
    pub fn resize_4ch<P>(
        img: ImgRef<P>,
        out_width: u32,
        out_height: u32,
        desc: PixelDescriptor,
        config: &ResizeConfig,
    ) -> ImgVec<P>
    where
        P: Copy + ComponentSlice<u8> + Default,
    {
        assert_eq!(core::mem::size_of::<P>(), 4, "pixel type must be 4 bytes");
        assert_eq!(desc.channels(), 4, "descriptor must be 4-channel");

        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input = desc;
        cfg.output = desc;
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        let w = img.width();
        let mut row_buf = vec![0u8; w.checked_mul(4).expect("resize_4ch: row_buf size overflow")];
        let out_row_len = cfg.output_row_len();
        let out_pixel_count = (out_width as usize)
            .checked_mul(out_height as usize)
            .expect("resize_4ch: out_width * out_height overflows usize");
        let mut out_pixels = Vec::with_capacity(out_pixel_count);
        for row in img.rows() {
            for (px, chunk) in row.iter().zip(row_buf.chunks_exact_mut(4)) {
                chunk.copy_from_slice(px.as_slice());
            }
            resizer.push_row(&row_buf).unwrap();
            while let Some(out_row) = resizer.next_output_row() {
                debug_assert_eq!(out_row.len(), out_row_len);
                for chunk in out_row.chunks_exact(4) {
                    let mut px = P::default();
                    px.as_mut_slice().copy_from_slice(chunk);
                    out_pixels.push(px);
                }
            }
        }
        resizer.finish();
        while let Some(out_row) = resizer.next_output_row() {
            debug_assert_eq!(out_row.len(), out_row_len);
            for chunk in out_row.chunks_exact(4) {
                let mut px = P::default();
                px.as_mut_slice().copy_from_slice(chunk);
                out_pixels.push(px);
            }
        }

        Img::new(out_pixels, out_width as usize, out_height as usize)
    }

    /// Resize a 3-channel u8 image. Works with `rgb::RGB<u8>`, `rgb::BGR<u8>`, etc.
    pub fn resize_3ch<P>(
        img: ImgRef<P>,
        out_width: u32,
        out_height: u32,
        config: &ResizeConfig,
    ) -> ImgVec<P>
    where
        P: Copy + ComponentSlice<u8> + Default,
    {
        assert_eq!(core::mem::size_of::<P>(), 3, "pixel type must be 3 bytes");

        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input = PixelDescriptor::RGB8_SRGB;
        cfg.output = PixelDescriptor::RGB8_SRGB;
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        let w = img.width();
        let mut row_buf = vec![0u8; w.checked_mul(3).expect("resize_3ch: row_buf size overflow")];
        let out_row_len = cfg.output_row_len();
        let out_pixel_count = (out_width as usize)
            .checked_mul(out_height as usize)
            .expect("resize_3ch: out_width * out_height overflows usize");
        let mut out_pixels = Vec::with_capacity(out_pixel_count);
        for row in img.rows() {
            for (px, chunk) in row.iter().zip(row_buf.chunks_exact_mut(3)) {
                chunk.copy_from_slice(px.as_slice());
            }
            resizer.push_row(&row_buf).unwrap();
            while let Some(out_row) = resizer.next_output_row() {
                debug_assert_eq!(out_row.len(), out_row_len);
                for chunk in out_row.chunks_exact(3) {
                    let mut px = P::default();
                    px.as_mut_slice().copy_from_slice(chunk);
                    out_pixels.push(px);
                }
            }
        }
        resizer.finish();
        while let Some(out_row) = resizer.next_output_row() {
            debug_assert_eq!(out_row.len(), out_row_len);
            for chunk in out_row.chunks_exact(3) {
                let mut px = P::default();
                px.as_mut_slice().copy_from_slice(chunk);
                out_pixels.push(px);
            }
        }

        Img::new(out_pixels, out_width as usize, out_height as usize)
    }

    /// Resize a grayscale u8 image.
    pub fn resize_gray8(
        img: ImgRef<u8>,
        out_width: u32,
        out_height: u32,
        config: &ResizeConfig,
    ) -> ImgVec<u8> {
        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input = PixelDescriptor::GRAY8_SRGB;
        cfg.output = PixelDescriptor::GRAY8_SRGB;
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        let out_pixel_count = (out_width as usize)
            .checked_mul(out_height as usize)
            .expect("resize_gray8: out_width * out_height overflows usize");
        let mut out_buf = Vec::with_capacity(out_pixel_count);
        for row in img.rows() {
            resizer.push_row(row).unwrap();
            while let Some(out_row) = resizer.next_output_row() {
                out_buf.extend_from_slice(out_row);
            }
        }
        resizer.finish();
        while let Some(out_row) = resizer.next_output_row() {
            out_buf.extend_from_slice(out_row);
        }

        Img::new(out_buf, out_width as usize, out_height as usize)
    }
}

pub use imgref_impl::{resize_3ch, resize_4ch, resize_gray8};

// =============================================================================
// H-first streaming resize (experimental)
// =============================================================================

/// H-first streaming resize for sRGB u8 4ch.
///
/// Unlike `StreamingResize` (V-first), this applies the H-filter to each input
/// row on arrival, storing the narrowed result in a ring buffer. The V-filter
/// then operates on `out_width`-wide rows instead of `in_width`-wide rows.
///
/// For heavy downscale (large width reduction), this is significantly faster
/// because the V-filter processes 1/N the data (where N = in_width / out_width).
///
/// This is a proof-of-concept for benchmarking. It only supports:
/// - u8 RGBA/RGBX 4ch, sRGB, no linearization, no compositing, no crop, no padding.
///
/// # Errors
///
/// Returns an error if the configuration fails [`ResizeConfig::validate`], the
/// input is too small, or buffer arithmetic would overflow `usize`.
///
/// # Panics
///
/// Panics if the channel count is not 4 or if the input slice is shorter than
/// `in_width * in_height * 4`.
pub fn resize_hfirst_streaming(
    config: &ResizeConfig,
    input: &[u8],
) -> Result<Vec<u8>, &'static str> {
    config.validate()?;
    assert_eq!(config.channels(), 4, "H-first streaming only supports 4ch");

    let filter = InterpolationDetails::create(config.filter);
    let v_weights = I16WeightTable::new(config.in_height, config.out_height, &filter);
    let h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);

    // Dispatch on `v_weights.max_taps` once at function entry. Each tier picks
    // a stack-array size N tuned to the actual workload: bilinear/bicubic at
    // upscale fits N=8, Lanczos at modest downscale fits N=16, Lanczos-3 at 4×
    // downscale fits N=32, and extreme downscales fall through to the heap path.
    //
    // Compared to a single per-iteration `if tap_count <= 32` branch, this:
    // - sizes the stack array to the actual need (8/16/32 = 128/256/512 bytes)
    // - lifts the branch out of the inner loop entirely
    // - lets the compiler specialize gather-loop unrolling per N
    let max_taps = v_weights.max_taps;
    if max_taps <= 8 {
        resize_hfirst_streaming_inner::<8>(config, input, h_weights, v_weights)
    } else if max_taps <= 16 {
        resize_hfirst_streaming_inner::<16>(config, input, h_weights, v_weights)
    } else if max_taps <= 32 {
        resize_hfirst_streaming_inner::<32>(config, input, h_weights, v_weights)
    } else {
        resize_hfirst_streaming_heap(config, input, h_weights, v_weights)
    }
}

/// Inner implementation generic over the stack-array size `N`. Caller must
/// guarantee `v_weights.max_taps <= N` so every per-row `tap_count <= N` slice
/// fits in the stack buffer.
fn resize_hfirst_streaming_inner<const N: usize>(
    config: &ResizeConfig,
    input: &[u8],
    h_weights: I16WeightTable,
    v_weights: I16WeightTable,
) -> Result<Vec<u8>, &'static str> {
    let channels = config.channels();
    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;
    let in_row_len = in_w
        .checked_mul(channels)
        .ok_or("input row length overflows usize")?;
    let out_row_len = out_w
        .checked_mul(channels)
        .ok_or("output row length overflows usize")?;

    let cache_size = v_weights
        .max_taps
        .checked_add(2)
        .ok_or("v_weights cache size overflows usize")?;
    let mut ring: Vec<Vec<u8>> = (0..cache_size).map(|_| vec![0u8; out_row_len]).collect();

    let total_output = out_row_len
        .checked_mul(out_h)
        .ok_or("output buffer size overflows usize")?;
    let mut output = vec![0u8; total_output];
    let mut input_rows_pushed = 0u32;
    let mut output_rows_produced = 0u32;

    let h_padding = h_weights
        .groups4
        .checked_mul(16)
        .ok_or("h_padding overflows usize")?;
    let padded_len = in_row_len
        .checked_add(h_padding)
        .ok_or("padded row length overflows usize")?;

    for y in 0..in_h {
        let slot = y % cache_size;
        let row_off = y
            .checked_mul(in_row_len)
            .ok_or("row offset overflows usize")?;
        let in_row = &input[row_off..row_off + padded_len.min(input.len() - row_off)];
        simd::filter_h_u8_i16(in_row, &mut ring[slot], &h_weights, channels);
        input_rows_pushed += 1;

        loop {
            if output_rows_produced >= out_h as u32 {
                break;
            }
            let out_y = output_rows_produced as usize;
            let left = v_weights.left[out_y];
            let tap_count = v_weights.tap_count(out_y);
            let weights = v_weights.weights(out_y);

            let last_needed = (left + tap_count as i32 - 1).clamp(0, in_h as i32 - 1) as u32;
            if last_needed >= input_rows_pushed {
                break;
            }

            let out_start = out_y
                .checked_mul(out_row_len)
                .ok_or("output offset overflows usize")?;
            // Caller's dispatch guarantees `tap_count <= N`. Stack array
            // sized to N is allocation-free per inner iteration.
            let empty: &[u8] = &[];
            let mut stack_refs: [&[u8]; N] = [empty; N];
            for (t, slot) in stack_refs.iter_mut().enumerate().take(tap_count) {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                *slot = &ring[in_y % cache_size];
            }
            simd::filter_v_row_u8_i16(
                &stack_refs[..tap_count],
                &mut output[out_start..out_start + out_row_len],
                weights,
            );
            output_rows_produced += 1;
        }
    }

    while (output_rows_produced as usize) < out_h {
        let out_y = output_rows_produced as usize;
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        let out_start = out_y
            .checked_mul(out_row_len)
            .ok_or("output offset overflows usize")?;
        let empty: &[u8] = &[];
        let mut stack_refs: [&[u8]; N] = [empty; N];
        for (t, slot) in stack_refs.iter_mut().enumerate().take(tap_count) {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            *slot = &ring[in_y % cache_size];
        }
        simd::filter_v_row_u8_i16(
            &stack_refs[..tap_count],
            &mut output[out_start..out_start + out_row_len],
            weights,
        );
        output_rows_produced += 1;
    }

    Ok(output)
}

/// Heap-fallback implementation for extreme downscales where `max_taps > 32`
/// (Lanczos-3 at 8×+ downscale, etc.). At these tap counts the per-row
/// allocation cost is dwarfed by the SIMD work, so we accept it rather than
/// burning kilobytes of stack on the rare extreme path.
fn resize_hfirst_streaming_heap(
    config: &ResizeConfig,
    input: &[u8],
    h_weights: I16WeightTable,
    v_weights: I16WeightTable,
) -> Result<Vec<u8>, &'static str> {
    let channels = config.channels();
    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;
    let in_row_len = in_w
        .checked_mul(channels)
        .ok_or("input row length overflows usize")?;
    let out_row_len = out_w
        .checked_mul(channels)
        .ok_or("output row length overflows usize")?;

    let cache_size = v_weights
        .max_taps
        .checked_add(2)
        .ok_or("v_weights cache size overflows usize")?;
    let mut ring: Vec<Vec<u8>> = (0..cache_size).map(|_| vec![0u8; out_row_len]).collect();

    let total_output = out_row_len
        .checked_mul(out_h)
        .ok_or("output buffer size overflows usize")?;
    let mut output = vec![0u8; total_output];
    let mut input_rows_pushed = 0u32;
    let mut output_rows_produced = 0u32;

    let h_padding = h_weights
        .groups4
        .checked_mul(16)
        .ok_or("h_padding overflows usize")?;
    let padded_len = in_row_len
        .checked_add(h_padding)
        .ok_or("padded row length overflows usize")?;

    for y in 0..in_h {
        let slot = y % cache_size;
        let row_off = y
            .checked_mul(in_row_len)
            .ok_or("row offset overflows usize")?;
        let in_row = &input[row_off..row_off + padded_len.min(input.len() - row_off)];
        simd::filter_h_u8_i16(in_row, &mut ring[slot], &h_weights, channels);
        input_rows_pushed += 1;

        loop {
            if output_rows_produced >= out_h as u32 {
                break;
            }
            let out_y = output_rows_produced as usize;
            let left = v_weights.left[out_y];
            let tap_count = v_weights.tap_count(out_y);
            let weights = v_weights.weights(out_y);

            let last_needed = (left + tap_count as i32 - 1).clamp(0, in_h as i32 - 1) as u32;
            if last_needed >= input_rows_pushed {
                break;
            }

            let out_start = out_y
                .checked_mul(out_row_len)
                .ok_or("output offset overflows usize")?;
            let mut row_refs: Vec<&[u8]> = Vec::with_capacity(tap_count);
            for t in 0..tap_count {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                row_refs.push(&ring[in_y % cache_size]);
            }
            simd::filter_v_row_u8_i16(
                &row_refs,
                &mut output[out_start..out_start + out_row_len],
                weights,
            );
            output_rows_produced += 1;
        }
    }

    while (output_rows_produced as usize) < out_h {
        let out_y = output_rows_produced as usize;
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        let out_start = out_y
            .checked_mul(out_row_len)
            .ok_or("output offset overflows usize")?;
        let mut row_refs: Vec<&[u8]> = Vec::with_capacity(tap_count);
        for t in 0..tap_count {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            row_refs.push(&ring[in_y % cache_size]);
        }
        simd::filter_v_row_u8_i16(
            &row_refs,
            &mut output[out_start..out_start + out_row_len],
            weights,
        );
        output_rows_produced += 1;
    }

    Ok(output)
}

/// H-first streaming resize for the f32 path (any channel count).
///
/// Decodes each input row to f32, H-filters to f16, stores in ring buffer.
/// V-filter reads f16 ring → f32 output, then encodes back to u8.
///
/// For heavy downscale, this processes 1/N the data in the V-filter compared
/// to V-first streaming, where N = in_width / out_width.
///
/// # Errors
///
/// Returns an error if the configuration fails [`ResizeConfig::validate`] or
/// buffer arithmetic would overflow `usize`.
pub fn resize_hfirst_streaming_f32(
    config: &ResizeConfig,
    input: &[u8],
) -> Result<Vec<u8>, &'static str> {
    config.validate()?;

    let filter = InterpolationDetails::create(config.filter);
    let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
    let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);

    // Same const-generic dispatch shape as the u8 path. See
    // `resize_hfirst_streaming` for rationale.
    let max_taps = v_weights.max_taps;
    if max_taps <= 8 {
        resize_hfirst_streaming_f32_inner::<8>(config, input, h_weights, v_weights)
    } else if max_taps <= 16 {
        resize_hfirst_streaming_f32_inner::<16>(config, input, h_weights, v_weights)
    } else if max_taps <= 32 {
        resize_hfirst_streaming_f32_inner::<32>(config, input, h_weights, v_weights)
    } else {
        resize_hfirst_streaming_f32_heap(config, input, h_weights, v_weights)
    }
}

fn resize_hfirst_streaming_f32_inner<const N: usize>(
    config: &ResizeConfig,
    input: &[u8],
    h_weights: F32WeightTable,
    v_weights: F32WeightTable,
) -> Result<Vec<u8>, &'static str> {
    let channels = config.channels();
    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;
    let in_row_len = in_w
        .checked_mul(channels)
        .ok_or("input row length overflows usize")?;
    let out_row_len = out_w
        .checked_mul(channels)
        .ok_or("output row length overflows usize")?;
    let input_tf = config.effective_input_transfer();
    let output_tf = config.effective_output_transfer();
    let has_alpha = config.input.has_alpha();
    let needs_premul = config.needs_premultiply();

    let cache_size = v_weights
        .max_taps
        .checked_add(2)
        .ok_or("v_weights cache size overflows usize")?;
    let mut ring: Vec<Vec<u16>> = (0..cache_size).map(|_| vec![0u16; out_row_len]).collect();

    let h_pad = h_weights
        .max_taps
        .checked_mul(channels)
        .ok_or("h_weights padding overflows usize")?;
    let temp_f32_len = in_row_len
        .checked_add(h_pad)
        .ok_or("temp_f32 size overflows usize")?;
    let mut temp_f32 = vec![0.0f32; temp_f32_len];
    let mut temp_output = vec![0.0f32; out_row_len];
    let total_output = out_row_len
        .checked_mul(out_h)
        .ok_or("output buffer size overflows usize")?;
    let mut output = vec![0u8; total_output];

    let mut input_rows_pushed = 0u32;
    let mut output_rows_produced = 0u32;

    for y in 0..in_h {
        let row_off = y
            .checked_mul(in_row_len)
            .ok_or("row offset overflows usize")?;
        let row_end = row_off
            .checked_add(in_row_len)
            .ok_or("row end overflows usize")?;
        let in_row = &input[row_off..row_end];
        decode_u8_row(
            in_row,
            &mut temp_f32[..in_row_len],
            input_tf,
            channels,
            has_alpha,
        );
        if needs_premul {
            simd::premultiply_alpha_row(&mut temp_f32[..in_row_len]);
        }

        let slot = y % cache_size;
        simd::filter_h_row_f32_to_f16(&temp_f32, &mut ring[slot], &h_weights, channels);
        input_rows_pushed += 1;

        loop {
            if output_rows_produced >= out_h as u32 {
                break;
            }
            let out_y = output_rows_produced as usize;
            let left = v_weights.left[out_y];
            let tap_count = v_weights.tap_count(out_y);
            let weights = v_weights.weights(out_y);

            let last_needed = (left + tap_count as i32 - 1).clamp(0, in_h as i32 - 1) as u32;
            if last_needed >= input_rows_pushed {
                break;
            }

            let empty: &[u16] = &[];
            let mut stack_refs: [&[u16]; N] = [empty; N];
            for (t, slot) in stack_refs.iter_mut().enumerate().take(tap_count) {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                *slot = &ring[in_y % cache_size];
            }
            simd::filter_v_row_f16(&stack_refs[..tap_count], &mut temp_output, weights);

            if needs_premul {
                simd::unpremultiply_alpha_row(&mut temp_output);
            }

            let out_start = out_y
                .checked_mul(out_row_len)
                .ok_or("output offset overflows usize")?;
            encode_u8_row(
                &temp_output,
                &mut output[out_start..out_start + out_row_len],
                output_tf,
                channels,
                has_alpha,
            );
            output_rows_produced += 1;
        }
    }

    while (output_rows_produced as usize) < out_h {
        let out_y = output_rows_produced as usize;
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        let empty: &[u16] = &[];
        let mut stack_refs: [&[u16]; N] = [empty; N];
        for (t, slot) in stack_refs.iter_mut().enumerate().take(tap_count) {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            *slot = &ring[in_y % cache_size];
        }
        simd::filter_v_row_f16(&stack_refs[..tap_count], &mut temp_output, weights);
        if needs_premul {
            simd::unpremultiply_alpha_row(&mut temp_output);
        }
        let out_start = out_y
            .checked_mul(out_row_len)
            .ok_or("output offset overflows usize")?;
        encode_u8_row(
            &temp_output,
            &mut output[out_start..out_start + out_row_len],
            output_tf,
            channels,
            has_alpha,
        );
        output_rows_produced += 1;
    }

    Ok(output)
}

fn resize_hfirst_streaming_f32_heap(
    config: &ResizeConfig,
    input: &[u8],
    h_weights: F32WeightTable,
    v_weights: F32WeightTable,
) -> Result<Vec<u8>, &'static str> {
    let channels = config.channels();
    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;
    let in_row_len = in_w
        .checked_mul(channels)
        .ok_or("input row length overflows usize")?;
    let out_row_len = out_w
        .checked_mul(channels)
        .ok_or("output row length overflows usize")?;
    let input_tf = config.effective_input_transfer();
    let output_tf = config.effective_output_transfer();
    let has_alpha = config.input.has_alpha();
    let needs_premul = config.needs_premultiply();

    let cache_size = v_weights
        .max_taps
        .checked_add(2)
        .ok_or("v_weights cache size overflows usize")?;
    let mut ring: Vec<Vec<u16>> = (0..cache_size).map(|_| vec![0u16; out_row_len]).collect();

    let h_pad = h_weights
        .max_taps
        .checked_mul(channels)
        .ok_or("h_weights padding overflows usize")?;
    let temp_f32_len = in_row_len
        .checked_add(h_pad)
        .ok_or("temp_f32 size overflows usize")?;
    let mut temp_f32 = vec![0.0f32; temp_f32_len];
    let mut temp_output = vec![0.0f32; out_row_len];
    let total_output = out_row_len
        .checked_mul(out_h)
        .ok_or("output buffer size overflows usize")?;
    let mut output = vec![0u8; total_output];

    let mut input_rows_pushed = 0u32;
    let mut output_rows_produced = 0u32;

    for y in 0..in_h {
        let row_off = y
            .checked_mul(in_row_len)
            .ok_or("row offset overflows usize")?;
        let row_end = row_off
            .checked_add(in_row_len)
            .ok_or("row end overflows usize")?;
        let in_row = &input[row_off..row_end];
        decode_u8_row(
            in_row,
            &mut temp_f32[..in_row_len],
            input_tf,
            channels,
            has_alpha,
        );
        if needs_premul {
            simd::premultiply_alpha_row(&mut temp_f32[..in_row_len]);
        }

        let slot = y % cache_size;
        simd::filter_h_row_f32_to_f16(&temp_f32, &mut ring[slot], &h_weights, channels);
        input_rows_pushed += 1;

        loop {
            if output_rows_produced >= out_h as u32 {
                break;
            }
            let out_y = output_rows_produced as usize;
            let left = v_weights.left[out_y];
            let tap_count = v_weights.tap_count(out_y);
            let weights = v_weights.weights(out_y);

            let last_needed = (left + tap_count as i32 - 1).clamp(0, in_h as i32 - 1) as u32;
            if last_needed >= input_rows_pushed {
                break;
            }

            let mut row_refs: Vec<&[u16]> = Vec::with_capacity(tap_count);
            for t in 0..tap_count {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                row_refs.push(&ring[in_y % cache_size]);
            }
            simd::filter_v_row_f16(&row_refs, &mut temp_output, weights);

            if needs_premul {
                simd::unpremultiply_alpha_row(&mut temp_output);
            }

            let out_start = out_y
                .checked_mul(out_row_len)
                .ok_or("output offset overflows usize")?;
            encode_u8_row(
                &temp_output,
                &mut output[out_start..out_start + out_row_len],
                output_tf,
                channels,
                has_alpha,
            );
            output_rows_produced += 1;
        }
    }

    while (output_rows_produced as usize) < out_h {
        let out_y = output_rows_produced as usize;
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        let mut row_refs: Vec<&[u16]> = Vec::with_capacity(tap_count);
        for t in 0..tap_count {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            row_refs.push(&ring[in_y % cache_size]);
        }
        simd::filter_v_row_f16(&row_refs, &mut temp_output, weights);
        if needs_premul {
            simd::unpremultiply_alpha_row(&mut temp_output);
        }
        let out_start = out_y
            .checked_mul(out_row_len)
            .ok_or("output offset overflows usize")?;
        encode_u8_row(
            &temp_output,
            &mut output[out_start..out_start + out_row_len],
            output_tf,
            channels,
            has_alpha,
        );
        output_rows_produced += 1;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::SolidBackground;
    use crate::filter::Filter;
    use zenpixels::{AlphaMode, PixelDescriptor};

    fn test_config(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
        ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build()
    }

    #[test]
    fn try_new_surfaces_validate_error_instead_of_panicking() {
        // Valid config -> Ok.
        assert!(Resizer::try_new(&test_config(8, 8, 4, 4)).is_ok());

        // Invalid config (zero output width) -> Err, NOT a panic. This is the
        // path a server hits on untrusted dimensions.
        let mut bad = test_config(8, 8, 4, 4);
        bad.out_width = 0;
        assert!(
            Resizer::try_new(&bad).is_err(),
            "invalid config must be a try_new error, not a panic"
        );
    }

    #[test]
    fn streaming_try_new_surfaces_validate_error() {
        assert!(StreamingResize::try_new(&test_config(8, 8, 4, 4)).is_ok());
        let mut bad = test_config(8, 8, 4, 4);
        bad.out_height = 0;
        assert!(StreamingResize::try_new(&bad).is_err());
    }

    #[test]
    fn test_resize_constant_color() {
        let config = test_config(20, 20, 10, 10);
        let mut input = vec![0u8; 20 * 20 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 128;
            pixel[2] = 128;
            pixel[3] = 255;
        }

        let output = Resizer::new(&config).resize(&input);
        assert_eq!(output.len(), 10 * 10 * 4);

        for pixel in output.chunks_exact(4) {
            assert!(
                (pixel[0] as i16 - 128).unsigned_abs() <= 2,
                "R off: {}",
                pixel[0]
            );
            assert!(
                (pixel[3] as i16 - 255).unsigned_abs() <= 1,
                "A off: {}",
                pixel[3]
            );
        }
    }

    #[test]
    fn test_resize_into_matches_resize() {
        let config = test_config(20, 20, 10, 10);
        let input = vec![100u8; 20 * 20 * 4];

        let output_alloc = Resizer::new(&config).resize(&input);
        let mut output_into = vec![0u8; 10 * 10 * 4];
        Resizer::new(&config).resize_into(&input, &mut output_into);

        assert_eq!(output_alloc, output_into);
    }

    #[test]
    fn test_resize_upscale() {
        let config = test_config(10, 10, 20, 20);
        let input = vec![200u8; 10 * 10 * 4];

        let output = Resizer::new(&config).resize(&input);
        assert_eq!(output.len(), 20 * 20 * 4);
    }

    #[test]
    fn test_resize_1x1() {
        let config = test_config(1, 1, 1, 1);
        let input = vec![128, 64, 32, 255];

        let output = Resizer::new(&config).resize(&input);
        assert_eq!(output.len(), 4);
        assert!((output[0] as i16 - 128).unsigned_abs() <= 2);
    }

    #[test]
    fn test_resize_with_stride() {
        let config = ResizeConfig::builder(10, 10, 5, 5)
            .format(PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .in_stride(10 * 4 + 8)
            .build();

        let stride = 10 * 4 + 8;
        let mut input = vec![0u8; 10 * stride];
        for y in 0..10 {
            for x in 0..10 * 4 {
                input[y * stride + x] = 128;
            }
        }

        let output = Resizer::new(&config).resize(&input);
        assert_eq!(output.len(), 5 * 5 * 4);
    }

    #[test]
    fn test_resizer_reuse() {
        let config = test_config(20, 20, 10, 10);
        let mut resizer = Resizer::new(&config);

        let input1 = vec![100u8; 20 * 20 * 4];
        let output1 = resizer.resize(&input1);

        let input2 = vec![200u8; 20 * 20 * 4];
        let output2 = resizer.resize(&input2);

        assert_ne!(output1, output2);
        let output1b = resizer.resize(&input1);
        assert_eq!(output1, output1b);
    }

    #[test]
    fn test_resize_f32() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBAF32_LINEAR.with_alpha(Some(AlphaMode::Undefined)))
            .build();

        let input = vec![0.5f32; 20 * 20 * 4];
        let output = Resizer::new(&config).resize_f32(&input);
        assert_eq!(output.len(), 10 * 10 * 4);
        for &v in &output {
            assert!((v - 0.5).abs() < 0.05, "value off: {}", v);
        }
    }

    #[test]
    fn test_resize_imgref_rgba() {
        use crate::resize::resize_4ch;
        use imgref::Img;
        use rgb::RGBA;

        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .srgb()
            .build();

        let pixels = vec![RGBA::new(128u8, 128, 128, 255); 20 * 20];
        let img = Img::new(pixels, 20, 20);

        let out = resize_4ch(img.as_ref(), 10, 10, PixelDescriptor::RGBA8_SRGB, &config);
        assert_eq!(out.width(), 10);
        assert_eq!(out.height(), 10);

        for px in out.pixels() {
            assert!((px.r as i16 - 128).unsigned_abs() <= 2, "R off: {}", px.r);
            assert!((px.a as i16 - 255).unsigned_abs() <= 1, "A off: {}", px.a);
        }
    }

    // === Composite tests ===

    #[test]
    fn no_background_matches_new() {
        let config = test_config(20, 20, 10, 10);
        let mut input = vec![0u8; 20 * 20 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 64;
            pixel[2] = 32;
            pixel[3] = 200;
        }

        let output_new = Resizer::new(&config).resize(&input);
        let output_bg = Resizer::with_background(&config, NoBackground)
            .unwrap()
            .resize(&input);
        assert_eq!(output_new, output_bg);
    }

    #[test]
    fn solid_opaque_bg_makes_output_opaque() {
        // Use linear mode so composite path is engaged (path 2)
        let config = ResizeConfig::builder(10, 10, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build();

        // Semi-transparent input
        let mut input = vec![0u8; 10 * 10 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 128;
            pixel[1] = 64;
            pixel[2] = 32;
            pixel[3] = 128;
        }

        let bg = SolidBackground::white(PixelDescriptor::RGBA8_SRGB);
        let output = Resizer::with_background(&config, bg)
            .unwrap()
            .resize(&input);
        assert_eq!(output.len(), 10 * 10 * 4);

        for pixel in output.chunks_exact(4) {
            assert_eq!(pixel[3], 255, "output alpha must be 255 with opaque bg");
            assert!(pixel[0] > 0, "R should have content");
        }
    }

    #[test]
    fn resizer_composite_matches_streaming() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build();

        let mut input = vec![0u8; 20 * 20 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 100;
            pixel[1] = 150;
            pixel[2] = 200;
            pixel[3] = 180;
        }

        // Resizer path
        let bg1 = SolidBackground::white(PixelDescriptor::RGBA8_SRGB);
        let resizer_output = Resizer::with_background(&config, bg1)
            .unwrap()
            .resize(&input);

        // Streaming path
        use crate::streaming::StreamingResize;
        let bg2 = SolidBackground::white(PixelDescriptor::RGBA8_SRGB);
        let mut streamer = StreamingResize::with_background(&config, bg2).unwrap();
        let mut streaming_output = Vec::new();
        for y in 0..20 {
            let start = y * 20 * 4;
            let end = start + 20 * 4;
            streamer.push_row(&input[start..end]).unwrap();
            while let Some(row) = streamer.next_output_row() {
                streaming_output.extend_from_slice(row);
            }
        }
        streamer.finish();
        while let Some(row) = streamer.next_output_row() {
            streaming_output.extend_from_slice(row);
        }

        assert_eq!(resizer_output.len(), streaming_output.len());
        // Allow small differences due to path/precision differences
        for (i, (&a, &b)) in resizer_output
            .iter()
            .zip(streaming_output.iter())
            .enumerate()
        {
            assert!(
                (a as i16 - b as i16).unsigned_abs() <= 2,
                "mismatch at byte {}: resizer={}, streaming={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn rejects_premultiplied_input() {
        let config = ResizeConfig::builder(10, 10, 5, 5)
            .format(PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)))
            .build();

        let bg = SolidBackground::white(
            PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied)),
        );
        let result = Resizer::with_background(&config, bg);
        assert!(
            matches!(
                result.as_ref().map_err(|e| e.error()),
                Err(&CompositeError::PremultipliedInput)
            ),
            "expected PremultipliedInput error"
        );
    }

    #[test]
    fn f32_composite_opaque_bg() {
        let config = ResizeConfig::builder(10, 10, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBAF32_LINEAR)
            .build();

        // Semi-transparent input: RGBA = [0.5, 0.3, 0.1, 0.5]
        let mut input = vec![0.0f32; 10 * 10 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 0.5;
            pixel[1] = 0.3;
            pixel[2] = 0.1;
            pixel[3] = 0.5;
        }

        let bg = SolidBackground::from_linear(1.0, 1.0, 1.0, 1.0, PixelDescriptor::RGBAF32_LINEAR);
        let output = Resizer::with_background(&config, bg)
            .unwrap()
            .resize_f32(&input);
        assert_eq!(output.len(), 10 * 10 * 4);

        for pixel in output.chunks_exact(4) {
            // After composite + unpremultiply, alpha should be 1.0
            assert!(
                (pixel[3] - 1.0).abs() < 0.02,
                "alpha should be ~1.0, got {}",
                pixel[3]
            );
        }
    }

    // === u16 tests ===

    #[test]
    fn test_resize_u16_constant_color() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA16_SRGB)
            .build();

        let mut input = vec![0u16; 20 * 20 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 32768; // ~50% sRGB
            pixel[1] = 32768;
            pixel[2] = 32768;
            pixel[3] = 65535; // fully opaque
        }

        let output = Resizer::new(&config).resize_u16(&input);
        assert_eq!(output.len(), 10 * 10 * 4);

        for pixel in output.chunks_exact(4) {
            assert!(
                (pixel[0] as i32 - 32768).unsigned_abs() <= 100,
                "R off: {} (expected ~32768)",
                pixel[0]
            );
            assert!(
                (pixel[3] as i32 - 65535).unsigned_abs() <= 1,
                "A off: {} (expected 65535)",
                pixel[3]
            );
        }
    }

    #[test]
    fn test_resize_u16_into_matches_alloc() {
        let config = ResizeConfig::builder(20, 20, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA16_SRGB)
            .build();

        let input = vec![40000u16; 20 * 20 * 4];

        let output_alloc = Resizer::new(&config).resize_u16(&input);
        let mut output_into = vec![0u16; 10 * 10 * 4];
        Resizer::new(&config).resize_u16_into(&input, &mut output_into);

        assert_eq!(output_alloc, output_into);
    }

    #[test]
    fn test_resize_u16_upscale() {
        let config = ResizeConfig::builder(10, 10, 20, 20)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGBA16_SRGB)
            .build();

        let mut input = vec![0u16; 10 * 10 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 50000;
            pixel[1] = 30000;
            pixel[2] = 10000;
            pixel[3] = 65535;
        }

        let output = Resizer::new(&config).resize_u16(&input);
        assert_eq!(output.len(), 20 * 20 * 4);

        // Center pixels should be close to original
        let center_start = (10 * 20 + 10) * 4;
        let px = &output[center_start..center_start + 4];
        assert!(
            (px[0] as i32 - 50000).unsigned_abs() <= 200,
            "R off: {} (expected ~50000)",
            px[0]
        );
    }

    #[test]
    fn test_resize_u16_rgb_3ch() {
        let config = ResizeConfig::builder(16, 16, 8, 8)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGB16_SRGB)
            .build();

        let mut input = vec![0u16; 16 * 16 * 3];
        for pixel in input.chunks_exact_mut(3) {
            pixel[0] = 40000;
            pixel[1] = 20000;
            pixel[2] = 60000;
        }

        let output = Resizer::new(&config).resize_u16(&input);
        assert_eq!(output.len(), 8 * 8 * 3);

        for pixel in output.chunks_exact(3) {
            assert!(
                (pixel[0] as i32 - 40000).unsigned_abs() <= 200,
                "R off: {}",
                pixel[0]
            );
        }
    }

    #[test]
    fn test_resize_u16_gray() {
        let config = ResizeConfig::builder(16, 16, 8, 8)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::GRAY16_SRGB)
            .build();

        let input = vec![50000u16; 16 * 16];
        let output = Resizer::new(&config).resize_u16(&input);
        assert_eq!(output.len(), 8 * 8);

        for &v in &output {
            assert!((v as i32 - 50000).unsigned_abs() <= 200, "Gray off: {}", v);
        }
    }

    /// H4: resize_hfirst_streaming must reject adversarial configs through
    /// validate() instead of allocating tens of GB.
    #[test]
    fn hfirst_streaming_rejects_adversarial_size() {
        let mut cfg = test_config(4, 4, 2, 2);
        cfg.in_width = u32::MAX;
        cfg.in_height = 4;
        cfg.out_width = 1;
        cfg.out_height = 2;
        let input = vec![0u8; 16];
        assert!(resize_hfirst_streaming(&cfg, &input).is_err());
    }

    #[test]
    fn hfirst_streaming_f32_rejects_adversarial_size() {
        let mut cfg = ResizeConfig::builder(4, 4, 2, 2)
            .filter(Filter::Lanczos)
            .format(PixelDescriptor::RGB8_SRGB)
            .srgb()
            .build();
        cfg.in_width = u32::MAX;
        cfg.out_width = 1;
        let input = vec![0u8; 16];
        assert!(resize_hfirst_streaming_f32(&cfg, &input).is_err());
    }

    #[test]
    fn hfirst_streaming_succeeds_on_normal_input() {
        let cfg = test_config(20, 20, 10, 10);
        let mut input = vec![0u8; 20 * 20 * 4];
        for px in input.chunks_exact_mut(4) {
            px[0] = 100;
            px[1] = 100;
            px[2] = 100;
            px[3] = 255;
        }
        let output = resize_hfirst_streaming(&cfg, &input).expect("normal resize must succeed");
        assert_eq!(output.len(), 10 * 10 * 4);
    }
}

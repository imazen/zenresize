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
use crate::pixel::ResizeConfig;
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
    pub fn new(config: &ResizeConfig) -> Self {
        Resizer {
            config: config.clone(),
            stream: StreamingResize::new(config),
        }
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
        let mut row_buf = vec![0u8; w * 4];
        let out_row_len = cfg.output_row_len();
        let mut out_pixels = Vec::with_capacity(out_width as usize * out_height as usize);
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
        let mut row_buf = vec![0u8; w * 3];
        let out_row_len = cfg.output_row_len();
        let mut out_pixels = Vec::with_capacity(out_width as usize * out_height as usize);
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

        let mut out_buf = Vec::with_capacity(out_width as usize * out_height as usize);
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
pub fn resize_hfirst_streaming(config: &ResizeConfig, input: &[u8]) -> Vec<u8> {
    let channels = config.channels();
    assert_eq!(channels, 4, "H-first streaming only supports 4ch");

    let filter = InterpolationDetails::create(config.filter);
    let h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);
    let v_weights = I16WeightTable::new(config.in_height, config.out_height, &filter);

    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;
    let in_row_len = in_w * channels;
    let out_row_len = out_w * channels;

    // Ring buffer: stores H-filtered rows, each out_row_len wide.
    let cache_size = v_weights.max_taps + 2;
    let mut ring: Vec<Vec<u8>> = (0..cache_size).map(|_| vec![0u8; out_row_len]).collect();

    let mut output = vec![0u8; out_row_len * out_h];
    let mut input_rows_pushed = 0u32;
    let mut output_rows_produced = 0u32;

    // Temp buffer for rows with SIMD padding
    let h_padding = h_weights.groups4 * 16;
    let padded_len = in_row_len + h_padding;

    for y in 0..in_h {
        // H-filter this input row → ring buffer slot
        let slot = y % cache_size;
        let in_row =
            &input[y * in_row_len..y * in_row_len + padded_len.min(input.len() - y * in_row_len)];
        simd::filter_h_u8_i16(in_row, &mut ring[slot], &h_weights, channels);
        input_rows_pushed += 1;

        // Produce output rows when possible
        loop {
            if output_rows_produced >= out_h as u32 {
                break;
            }
            let out_y = output_rows_produced as usize;
            let left = v_weights.left[out_y];
            let tap_count = v_weights.tap_count(out_y);
            let weights = v_weights.weights(out_y);

            // Check if all needed input rows are available
            let last_needed = (left + tap_count as i32 - 1).clamp(0, in_h as i32 - 1) as u32;
            if last_needed >= input_rows_pushed {
                break;
            }

            // Gather row references from ring buffer
            let mut row_refs: [&[u8]; 128] = [&[]; 128];
            for (t, slot) in row_refs.iter_mut().enumerate().take(tap_count) {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                *slot = &ring[in_y % cache_size];
            }

            // V-filter directly to output (no temp buffer needed!)
            let out_start = out_y * out_row_len;
            simd::filter_v_row_u8_i16(
                &row_refs[..tap_count],
                &mut output[out_start..out_start + out_row_len],
                weights,
            );
            output_rows_produced += 1;
        }
    }

    // Produce remaining output rows (edge clamping)
    while (output_rows_produced as usize) < out_h {
        let out_y = output_rows_produced as usize;
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        let mut row_refs: [&[u8]; 128] = [&[]; 128];
        for (t, slot) in row_refs.iter_mut().enumerate().take(tap_count) {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            *slot = &ring[in_y % cache_size];
        }

        let out_start = out_y * out_row_len;
        simd::filter_v_row_u8_i16(
            &row_refs[..tap_count],
            &mut output[out_start..out_start + out_row_len],
            weights,
        );
        output_rows_produced += 1;
    }

    output
}

/// H-first streaming resize for the f32 path (any channel count).
///
/// Decodes each input row to f32, H-filters to f16, stores in ring buffer.
/// V-filter reads f16 ring → f32 output, then encodes back to u8.
///
/// For heavy downscale, this processes 1/N the data in the V-filter compared
/// to V-first streaming, where N = in_width / out_width.
pub fn resize_hfirst_streaming_f32(config: &ResizeConfig, input: &[u8]) -> Vec<u8> {
    let channels = config.channels();
    let filter = InterpolationDetails::create(config.filter);
    let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
    let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);

    let in_w = config.in_width as usize;
    let in_h = config.in_height as usize;
    let out_w = config.out_width as usize;
    let out_h = config.out_height as usize;
    let in_row_len = in_w * channels;
    let out_row_len = out_w * channels;
    let input_tf = config.effective_input_transfer();
    let output_tf = config.effective_output_transfer();
    let has_alpha = config.input.has_alpha();
    let needs_premul = config.needs_premultiply();

    // Ring buffer: stores H-filtered f16 rows, each out_row_len wide.
    let cache_size = v_weights.max_taps + 2;
    let mut ring: Vec<Vec<u16>> = (0..cache_size).map(|_| vec![0u16; out_row_len]).collect();

    // Temp buffers
    let mut temp_f32 = vec![0.0f32; in_row_len + h_weights.max_taps * channels];
    let mut temp_output = vec![0.0f32; out_row_len];
    let mut output = vec![0u8; out_row_len * out_h];

    let mut input_rows_pushed = 0u32;
    let mut output_rows_produced = 0u32;

    for y in 0..in_h {
        // Decode input row to f32
        let in_row = &input[y * in_row_len..(y + 1) * in_row_len];
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

        // H-filter f32 → f16 into ring buffer slot
        let slot = y % cache_size;
        simd::filter_h_row_f32_to_f16(&temp_f32, &mut ring[slot], &h_weights, channels);
        input_rows_pushed += 1;

        // Produce output rows
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

            // Gather row references from ring buffer
            let mut row_refs: [&[u16]; 128] = [&[]; 128];
            for (t, slot) in row_refs.iter_mut().enumerate().take(tap_count) {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                *slot = &ring[in_y % cache_size];
            }

            // V-filter f16 → f32
            simd::filter_v_row_f16(&row_refs[..tap_count], &mut temp_output, weights);

            if needs_premul {
                simd::unpremultiply_alpha_row(&mut temp_output);
            }

            // Encode f32 → u8
            let out_start = out_y * out_row_len;
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

    // Remaining output rows
    while (output_rows_produced as usize) < out_h {
        let out_y = output_rows_produced as usize;
        let left = v_weights.left[out_y];
        let tap_count = v_weights.tap_count(out_y);
        let weights = v_weights.weights(out_y);

        let mut row_refs: [&[u16]; 128] = [&[]; 128];
        for (t, slot) in row_refs.iter_mut().enumerate().take(tap_count) {
            let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
            *slot = &ring[in_y % cache_size];
        }

        simd::filter_v_row_f16(&row_refs[..tap_count], &mut temp_output, weights);
        if needs_premul {
            simd::unpremultiply_alpha_row(&mut temp_output);
        }
        let out_start = out_y * out_row_len;
        encode_u8_row(
            &temp_output,
            &mut output[out_start..out_start + out_row_len],
            output_tf,
            channels,
            has_alpha,
        );
        output_rows_produced += 1;
    }

    output
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
}

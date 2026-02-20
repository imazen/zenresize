//! Full-frame resize via [`Resizer`].
//!
//! `Resizer` pre-computes weight tables and allocates intermediate buffers once,
//! then reuses them across calls. For a single resize, `Resizer::new(&config).resize(&input)`
//! is equivalent to the former one-shot functions.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::color;
use crate::composite::{self, Background, CompositeError, NoBackground};
use crate::filter::InterpolationDetails;
use crate::pixel::ResizeConfig;
use crate::proven;
use crate::simd;
use crate::weights::{F32WeightTable, I16WeightTable};

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
/// use zenresize::{Resizer, ResizeConfig, Filter, PixelFormat, PixelLayout};
///
/// let config = ResizeConfig::builder(20, 20, 10, 10)
///     .filter(Filter::Lanczos)
///     .format(PixelFormat::Srgb8(PixelLayout::Rgba))
///     .build();
///
/// let mut resizer = Resizer::new(&config);
/// let input = vec![128u8; 20 * 20 * 4];
/// let output = resizer.resize(&input);
/// assert_eq!(output.len(), 10 * 10 * 4);
/// ```
pub struct Resizer<B: Background = NoBackground> {
    config: ResizeConfig,
    h_weights_i16: Option<I16WeightTable>,
    v_weights_i16: Option<I16WeightTable>,
    h_weights_f32: Option<F32WeightTable>,
    v_weights_f32: Option<F32WeightTable>,
    intermediate_u8: Vec<u8>,
    intermediate_f32: Vec<f32>,
    intermediate_i16: Vec<i16>,
    linearized_row: Vec<i16>,
    v_output_i16: Vec<i16>,
    premul_buf: Vec<u8>,
    temp_row_f32: Vec<f32>,
    temp_output_f32: Vec<f32>,
    /// Which path to use: 0 = sRGB i16, 1 = linear i16, 2 = f32 (u8 I/O), 3 = f32 (f32 I/O)
    path: u8,
    /// Background for compositing.
    background: B,
    /// Row buffer for non-solid backgrounds. Empty for NoBackground and SolidBackground.
    composite_bg_row: Vec<f32>,
}

impl Resizer<NoBackground> {
    /// Create a new resizer for the given configuration.
    /// Pre-computes weight tables.
    pub fn new(config: &ResizeConfig) -> Self {
        Self::new_inner(config, NoBackground, false)
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
    /// optimal path would be 0 (sRGB i16) or 1 (linear i16), this forces path 2
    /// (f32 with u8 I/O) instead. Path 3 (f32 I/O) is unaffected.
    ///
    /// # Errors
    ///
    /// Returns [`CompositeError::PremultipliedInput`] if the input format
    /// is `RgbaPremul` (compositing premultiplied input is mathematically incorrect).
    pub fn with_background(config: &ResizeConfig, background: B) -> Result<Self, CompositeError> {
        if config.input_format.layout().is_premultiplied() {
            return Err(CompositeError::PremultipliedInput);
        }
        Ok(Self::new_inner(config, background, true))
    }

    fn new_inner(config: &ResizeConfig, background: B, has_composite: bool) -> Self {
        config.validate().expect("invalid resize config");

        let mut config = config.clone();
        let force_f32 = has_composite && !background.is_transparent();

        // Compositing requires linear f32 path
        if force_f32 {
            config.linear = true;
        }

        let filter = InterpolationDetails::create(config.filter);
        let layout = config.input_format.layout();
        let channels = layout.channels() as usize;
        let needs_premul = layout.needs_premultiply();
        let linearize = config.needs_linearization();
        let in_h = config.in_height as usize;
        let out_w = config.out_width as usize;
        let h_row_len = out_w * channels;
        let in_row_len = config.input_row_len();

        // Only allocate bg row buffer for non-solid, non-transparent backgrounds
        let needs_bg_row = force_f32 && background.solid_pixel().is_none();
        let composite_bg_row = if needs_bg_row {
            vec![0.0f32; h_row_len]
        } else {
            Vec::new()
        };

        // f32 native input — always use f32 path
        if config.input_format.is_f32() {
            let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
            let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);
            let in_w = config.in_width as usize;
            let h_max_taps = h_weights.max_taps;
            return Resizer {
                config,
                h_weights_i16: None,
                v_weights_i16: None,
                h_weights_f32: Some(h_weights),
                v_weights_f32: Some(v_weights),
                intermediate_u8: Vec::new(),
                intermediate_f32: vec![0.0f32; h_row_len * in_h],
                intermediate_i16: Vec::new(),
                linearized_row: Vec::new(),
                v_output_i16: Vec::new(),
                premul_buf: Vec::new(),
                temp_row_f32: vec![0.0f32; in_w * channels + h_max_taps * channels],
                temp_output_f32: vec![0.0f32; h_row_len],
                path: 3,
                background,
                composite_bg_row,
            };
        }

        // Compositing forces f32 path — paths 0 and 1 use batch vertical passes
        // that can't inject per-row compositing.
        if !force_f32 && !linearize && channels == 4 {
            // Path 0: sRGB i16 fast path
            let h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);
            let v_weights = I16WeightTable::new(config.in_height, config.out_height, &filter);
            let intermediate = vec![0u8; h_row_len * in_h];
            let h_padding = h_weights.groups4 * 16;
            let premul_buf = if needs_premul {
                vec![0u8; in_row_len + h_padding]
            } else {
                Vec::new()
            };
            Resizer {
                config,
                h_weights_i16: Some(h_weights),
                v_weights_i16: Some(v_weights),
                h_weights_f32: None,
                v_weights_f32: None,
                intermediate_u8: intermediate,
                intermediate_f32: Vec::new(),
                intermediate_i16: Vec::new(),
                linearized_row: Vec::new(),
                v_output_i16: Vec::new(),
                premul_buf,
                temp_row_f32: Vec::new(),
                temp_output_f32: Vec::new(),
                path: 0,
                background,
                composite_bg_row,
            }
        } else if !force_f32 && linearize && channels == 4 && !needs_premul {
            // Path 1: linear-light i16 fast path (Rgbx or RgbaPremul)
            let h_weights = I16WeightTable::new(config.in_width, config.out_width, &filter);
            let v_weights = I16WeightTable::new(config.in_height, config.out_height, &filter);
            let h_padding = h_weights.groups4 * 16;
            let out_h = config.out_height as usize;
            Resizer {
                config,
                h_weights_i16: Some(h_weights),
                v_weights_i16: Some(v_weights),
                h_weights_f32: None,
                v_weights_f32: None,
                intermediate_u8: Vec::new(),
                intermediate_f32: Vec::new(),
                intermediate_i16: vec![0i16; h_row_len * in_h],
                linearized_row: vec![0i16; in_row_len + h_padding],
                v_output_i16: vec![0i16; h_row_len * out_h],
                premul_buf: Vec::new(),
                temp_row_f32: Vec::new(),
                temp_output_f32: Vec::new(),
                path: 1,
                background,
                composite_bg_row,
            }
        } else {
            // Path 2: f32 path (u8 I/O)
            let h_weights = F32WeightTable::new(config.in_width, config.out_width, &filter);
            let v_weights = F32WeightTable::new(config.in_height, config.out_height, &filter);
            let in_w = config.in_width as usize;
            let h_max_taps = h_weights.max_taps;
            Resizer {
                config,
                h_weights_i16: None,
                v_weights_i16: None,
                h_weights_f32: Some(h_weights),
                v_weights_f32: Some(v_weights),
                intermediate_u8: Vec::new(),
                intermediate_f32: vec![0.0f32; h_row_len * in_h],
                intermediate_i16: Vec::new(),
                linearized_row: Vec::new(),
                v_output_i16: Vec::new(),
                premul_buf: Vec::new(),
                temp_row_f32: vec![0.0f32; in_w * channels + h_max_taps * channels],
                temp_output_f32: vec![0.0f32; h_row_len],
                path: 2,
                background,
                composite_bg_row,
            }
        }
    }

    /// Mutable reference to the background.
    pub fn background_mut(&mut self) -> &mut B {
        &mut self.background
    }

    /// Resize a u8 image, allocating and returning the output.
    ///
    /// # Panics
    /// Panics if the config uses `LinearF32` format (use [`resize_f32`](Self::resize_f32) instead).
    pub fn resize(&mut self, input: &[u8]) -> Vec<u8> {
        assert!(
            self.config.input_format.is_u8(),
            "resize() requires Srgb8 format; use resize_f32() for LinearF32"
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
            self.config.input_format.is_u8(),
            "resize_into() requires Srgb8 format; use resize_f32_into() for LinearF32"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let layout = config.input_format.layout();
        let channels = layout.channels() as usize;
        let needs_premul = layout.needs_premultiply();
        let in_h = config.in_height as usize;
        let out_w = config.out_width as usize;
        let out_h = config.out_height as usize;
        let h_row_len = out_w * channels;

        match self.path {
            0 => {
                // Path 0: sRGB i16 fast path (no compositing possible here)
                let h_weights = self.h_weights_i16.as_ref().unwrap();
                let v_weights = self.v_weights_i16.as_ref().unwrap();
                let intermediate = &mut self.intermediate_u8;

                if channels == 4 && !needs_premul {
                    let h_padding = h_weights.groups4 * 16;
                    let batch_count = in_h / 4;
                    let remainder = in_h % 4;

                    for batch in 0..batch_count {
                        let y0 = batch * 4;
                        let r0 = &input[y0 * in_stride
                            ..(y0 * in_stride + in_row_len + h_padding).min(input.len())];
                        let r1 = &input[(y0 + 1) * in_stride
                            ..((y0 + 1) * in_stride + in_row_len + h_padding).min(input.len())];
                        let r2 = &input[(y0 + 2) * in_stride
                            ..((y0 + 2) * in_stride + in_row_len + h_padding).min(input.len())];
                        let r3 = &input[(y0 + 3) * in_stride
                            ..((y0 + 3) * in_stride + in_row_len + h_padding).min(input.len())];

                        let out_base = y0 * h_row_len;
                        let (o0, rest) = intermediate[out_base..].split_at_mut(h_row_len);
                        let (o1, rest) = rest.split_at_mut(h_row_len);
                        let (o2, o3_and_rest) = rest.split_at_mut(h_row_len);
                        let o3 = &mut o3_and_rest[..h_row_len];

                        simd::filter_h_u8_i16_4rows(r0, r1, r2, r3, o0, o1, o2, o3, h_weights);
                    }

                    for i in 0..remainder {
                        let y = batch_count * 4 + i;
                        let in_start = y * in_stride;
                        let in_end = (in_start + in_row_len + h_padding).min(input.len());
                        let in_row = &input[in_start..in_end];
                        let out_start = y * h_row_len;

                        simd::filter_h_u8_i16(
                            in_row,
                            &mut intermediate[out_start..out_start + h_row_len],
                            h_weights,
                            channels,
                        );
                    }
                } else {
                    for y in 0..in_h {
                        let in_start = y * in_stride;
                        let in_row = &input[in_start..in_start + in_row_len];
                        let out_start = y * h_row_len;

                        let src = if needs_premul {
                            simd::premultiply_u8_row(in_row, &mut self.premul_buf[..in_row_len]);
                            &self.premul_buf[..] // includes zero-initialized SIMD padding
                        } else {
                            in_row
                        };

                        simd::filter_h_u8_i16(
                            src,
                            &mut intermediate[out_start..out_start + h_row_len],
                            h_weights,
                            channels,
                        );
                    }
                }

                // V pass
                simd::filter_v_all_u8_i16(intermediate, output, h_row_len, in_h, out_h, v_weights);

                if needs_premul {
                    for out_y in 0..out_h {
                        let out_start = out_y * out_row_len;
                        simd::unpremultiply_u8_row(&mut output[out_start..out_start + out_row_len]);
                    }
                }
            }
            1 => {
                // Path 1: linear-light i16 fast path (no compositing possible here)
                let h_weights = self.h_weights_i16.as_ref().unwrap();
                let v_weights = self.v_weights_i16.as_ref().unwrap();

                // H-pass: per-row sRGB→linear LUT + i16 filter
                for y in 0..in_h {
                    let in_start = y * in_stride;
                    let in_row = &input[in_start..in_start + in_row_len];

                    color::srgb_u8_to_linear_i12_row(
                        in_row,
                        &mut self.linearized_row[..in_row_len],
                    );
                    // Zero SIMD padding region
                    for v in &mut self.linearized_row[in_row_len..] {
                        *v = 0;
                    }

                    let out_start = y * h_row_len;
                    simd::filter_h_i16_i16(
                        &self.linearized_row,
                        &mut self.intermediate_i16[out_start..out_start + h_row_len],
                        h_weights,
                        channels,
                    );
                }

                // V-pass: batch kernel
                simd::filter_v_all_i16_i16(
                    &self.intermediate_i16,
                    &mut self.v_output_i16,
                    h_row_len,
                    in_h,
                    out_h,
                    v_weights,
                );

                // Output: per-row linear→sRGB LUT
                for out_y in 0..out_h {
                    let v_start = out_y * h_row_len;
                    let out_start = out_y * h_row_len;
                    color::linear_i12_to_srgb_u8_row(
                        &self.v_output_i16[v_start..v_start + h_row_len],
                        &mut output[out_start..out_start + h_row_len],
                    );
                }
            }
            _ => {
                // Path 2: f32 path with u8 I/O
                let h_weights = self.h_weights_f32.as_ref().unwrap();
                let v_weights = self.v_weights_f32.as_ref().unwrap();
                let linearize = config.needs_linearization();
                let intermediate = &mut self.intermediate_f32;
                let temp_row = &mut self.temp_row_f32;

                // === Horizontal pass ===
                for y in 0..in_h {
                    let in_start = y * in_stride;
                    let in_row = &input[in_start..in_start + in_row_len];

                    if linearize {
                        color::srgb_u8_to_linear_f32(
                            in_row,
                            &mut temp_row[..in_row_len],
                            channels,
                            layout.alpha_is_last_channel(),
                        );
                    } else {
                        simd::u8_to_f32_row(in_row, &mut temp_row[..in_row_len]);
                    }

                    if needs_premul {
                        simd::premultiply_alpha_row(&mut temp_row[..in_row_len]);
                    }

                    let out_start = y * h_row_len;
                    simd::filter_h_row_f32(
                        temp_row,
                        &mut intermediate[out_start..out_start + h_row_len],
                        h_weights,
                        channels,
                    );
                }

                // === Vertical pass ===
                let max_taps = v_weights.max_taps;
                let mut row_ptrs: Vec<&[f32]> = Vec::with_capacity(max_taps);
                let temp_output = &mut self.temp_output_f32;

                for out_y in 0..out_h {
                    let left = v_weights.left[out_y];
                    let tap_count = v_weights.tap_count(out_y);
                    let weights = v_weights.weights(out_y);

                    row_ptrs.clear();
                    for t in 0..tap_count {
                        let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                        let start = in_y * h_row_len;
                        row_ptrs.push(&intermediate[start..start + h_row_len]);
                    }

                    simd::filter_v_row_f32(&row_ptrs, &mut temp_output[..h_row_len], weights);

                    // === Composite: source-over onto background ===
                    composite::composite_dispatch(
                        &mut temp_output[..h_row_len],
                        &mut self.background,
                        &mut self.composite_bg_row,
                        out_y as u32,
                        channels as u8,
                    );

                    if needs_premul {
                        simd::unpremultiply_alpha_row(&mut temp_output[..h_row_len]);
                    }

                    let out_start = out_y * out_row_len;
                    let out_slice = &mut output[out_start..out_start + out_row_len];
                    if linearize {
                        color::linear_f32_to_srgb_u8(
                            &temp_output[..h_row_len],
                            out_slice,
                            channels,
                            layout.alpha_is_last_channel(),
                        );
                    } else {
                        simd::f32_to_u8_row(&temp_output[..h_row_len], out_slice);
                    }
                }
            }
        }
    }

    /// Resize an f32 image, allocating and returning the output.
    ///
    /// # Panics
    /// Panics if the config uses `Srgb8` format (use [`resize`](Self::resize) instead).
    pub fn resize_f32(&mut self, input: &[f32]) -> Vec<f32> {
        assert!(
            self.config.input_format.is_f32(),
            "resize_f32() requires LinearF32 format; use resize() for Srgb8"
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
            self.config.input_format.is_f32(),
            "resize_f32_into() requires LinearF32 format; use resize_into() for Srgb8"
        );
        let config = &self.config;
        let in_stride = config.effective_in_stride();
        let in_row_len = config.input_row_len();
        let out_row_len = config.output_row_len();
        let layout = config.input_format.layout();
        let channels = layout.channels() as usize;
        let needs_premul = layout.needs_premultiply();
        let in_h = config.in_height as usize;
        let out_w = config.out_width as usize;
        let out_h = config.out_height as usize;
        let h_row_len = out_w * channels;

        let h_weights = self.h_weights_f32.as_ref().unwrap();
        let v_weights = self.v_weights_f32.as_ref().unwrap();
        let intermediate = &mut self.intermediate_f32;
        let temp_row = &mut self.temp_row_f32;

        // === Horizontal pass: f32 → f32 ===
        for y in 0..in_h {
            let in_start = y * in_stride;
            temp_row[..in_row_len].copy_from_slice(&input[in_start..in_start + in_row_len]);

            if needs_premul {
                simd::premultiply_alpha_row(&mut temp_row[..in_row_len]);
            }

            let out_start = y * h_row_len;
            simd::filter_h_row_f32(
                temp_row,
                &mut intermediate[out_start..out_start + h_row_len],
                h_weights,
                channels,
            );
        }

        // === Vertical pass: f32 → f32 ===
        let max_taps = v_weights.max_taps;
        let mut row_ptrs: Vec<&[f32]> = Vec::with_capacity(max_taps);

        for out_y in 0..out_h {
            let left = v_weights.left[out_y];
            let tap_count = v_weights.tap_count(out_y);
            let weights = v_weights.weights(out_y);

            row_ptrs.clear();
            for t in 0..tap_count {
                let in_y = (left + t as i32).clamp(0, in_h as i32 - 1) as usize;
                let start = in_y * h_row_len;
                row_ptrs.push(&intermediate[start..start + h_row_len]);
            }

            if self.background.is_transparent() {
                // No compositing — write directly to output (zero overhead for NoBackground)
                let out_start = out_y * out_row_len;
                let out_slice = &mut output[out_start..out_start + out_row_len];
                simd::filter_v_row_f32(&row_ptrs, out_slice, weights);

                if needs_premul {
                    simd::unpremultiply_alpha_row(out_slice);
                }
            } else {
                // Compositing active — use temp buffer
                let temp_output = &mut self.temp_output_f32;
                simd::filter_v_row_f32(&row_ptrs, &mut temp_output[..h_row_len], weights);

                composite::composite_dispatch(
                    &mut temp_output[..h_row_len],
                    &mut self.background,
                    &mut self.composite_bg_row,
                    out_y as u32,
                    channels as u8,
                );

                if needs_premul {
                    simd::unpremultiply_alpha_row(&mut temp_output[..h_row_len]);
                }

                let out_start = out_y * out_row_len;
                output[out_start..out_start + out_row_len]
                    .copy_from_slice(&temp_output[..h_row_len]);
            }
        }
    }
}

// =============================================================================
// imgref + rgb crate integration
// =============================================================================
//
// All 4-channel pixel types (RGBA, BGRA, ARGB, ABGR) work identically because
// the pipeline is channel-order-agnostic. The `rgb` crate's ComponentSlice trait
// gives us zero-copy &[u8] access to any pixel type.

mod imgref_impl {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use crate::pixel::{PixelFormat, PixelLayout, ResizeConfig};
    use crate::streaming::StreamingResize;
    use imgref::{Img, ImgRef, ImgVec};
    use rgb::ComponentSlice;

    /// Resize a 4-channel u8 image. Works with any pixel type that implements
    /// `ComponentSlice` (RGBA, BGRA, ARGB, ABGR from the `rgb` crate).
    ///
    /// Channel order is preserved — the pipeline doesn't care whether
    /// the bytes represent RGBA or BGRA. Use a 4-channel layout
    /// ([`Rgba`](PixelLayout::Rgba), [`Rgbx`](PixelLayout::Rgbx),
    /// or [`RgbaPremul`](PixelLayout::RgbaPremul)).
    pub fn resize_4ch<P>(
        img: ImgRef<P>,
        out_width: u32,
        out_height: u32,
        layout: PixelLayout,
        config: &ResizeConfig,
    ) -> ImgVec<P>
    where
        P: Copy + ComponentSlice<u8> + Default,
    {
        assert_eq!(core::mem::size_of::<P>(), 4, "pixel type must be 4 bytes");
        assert_eq!(layout.channels(), 4, "layout must be 4-channel");

        let mut cfg = config.clone();
        cfg.in_width = img.width() as u32;
        cfg.in_height = img.height() as u32;
        cfg.out_width = out_width;
        cfg.out_height = out_height;
        cfg.input_format = PixelFormat::Srgb8(layout);
        cfg.output_format = PixelFormat::Srgb8(layout);
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        let w = img.width();
        let mut row_buf = vec![0u8; w * 4];
        for row in img.rows() {
            for (px, chunk) in row.iter().zip(row_buf.chunks_exact_mut(4)) {
                chunk.copy_from_slice(px.as_slice());
            }
            resizer.push_row(&row_buf);
        }
        resizer.finish();

        let out_row_len = cfg.output_row_len();
        let mut out_pixels = Vec::with_capacity(out_width as usize * out_height as usize);
        while let Some(row) = resizer.next_output_row() {
            debug_assert_eq!(row.len(), out_row_len);
            for chunk in row.chunks_exact(4) {
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
        cfg.input_format = PixelFormat::Srgb8(PixelLayout::Rgb);
        cfg.output_format = PixelFormat::Srgb8(PixelLayout::Rgb);
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        let w = img.width();
        let mut row_buf = vec![0u8; w * 3];
        for row in img.rows() {
            for (px, chunk) in row.iter().zip(row_buf.chunks_exact_mut(3)) {
                chunk.copy_from_slice(px.as_slice());
            }
            resizer.push_row(&row_buf);
        }
        resizer.finish();

        let out_row_len = cfg.output_row_len();
        let mut out_pixels = Vec::with_capacity(out_width as usize * out_height as usize);
        while let Some(row) = resizer.next_output_row() {
            debug_assert_eq!(row.len(), out_row_len);
            for chunk in row.chunks_exact(3) {
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
        cfg.input_format = PixelFormat::Srgb8(PixelLayout::Gray);
        cfg.output_format = PixelFormat::Srgb8(PixelLayout::Gray);
        cfg.in_stride = 0;

        let mut resizer = StreamingResize::new(&cfg);

        for row in img.rows() {
            resizer.push_row(row);
        }
        resizer.finish();

        let mut out_buf = Vec::with_capacity(out_width as usize * out_height as usize);
        while let Some(row) = resizer.next_output_row() {
            out_buf.extend_from_slice(&row);
        }

        Img::new(out_buf, out_width as usize, out_height as usize)
    }
}

pub use imgref_impl::{resize_3ch, resize_4ch, resize_gray8};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::SolidBackground;
    use crate::filter::Filter;
    use crate::pixel::{PixelFormat, PixelLayout};

    fn test_config(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
        ResizeConfig::builder(in_w, in_h, out_w, out_h)
            .filter(Filter::Lanczos)
            .format(PixelFormat::Srgb8(PixelLayout::Rgba))
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
            .format(PixelFormat::Srgb8(PixelLayout::Rgba))
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
            .format(PixelFormat::LinearF32(PixelLayout::Rgbx))
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

        let out = resize_4ch(img.as_ref(), 10, 10, PixelLayout::Rgba, &config);
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
        let output_bg =
            Resizer::with_background(&config, NoBackground).unwrap().resize(&input);
        assert_eq!(output_new, output_bg);
    }

    #[test]
    fn solid_opaque_bg_makes_output_opaque() {
        // Use linear mode so composite path is engaged (path 2)
        let config = ResizeConfig::builder(10, 10, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelFormat::Srgb8(PixelLayout::Rgba))
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

        let bg = SolidBackground::white(PixelLayout::Rgba);
        let output = Resizer::with_background(&config, bg).unwrap().resize(&input);
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
            .format(PixelFormat::Srgb8(PixelLayout::Rgba))
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
        let bg1 = SolidBackground::white(PixelLayout::Rgba);
        let resizer_output = Resizer::with_background(&config, bg1).unwrap().resize(&input);

        // Streaming path
        use crate::streaming::StreamingResize;
        let bg2 = SolidBackground::white(PixelLayout::Rgba);
        let mut streamer = StreamingResize::with_background(&config, bg2).unwrap();
        for y in 0..20 {
            let start = y * 20 * 4;
            let end = start + 20 * 4;
            streamer.push_row(&input[start..end]);
        }
        streamer.finish();

        let mut streaming_output = Vec::new();
        while let Some(row) = streamer.next_output_row() {
            streaming_output.extend_from_slice(&row);
        }

        assert_eq!(resizer_output.len(), streaming_output.len());
        // Allow small differences due to path/precision differences
        for (i, (&a, &b)) in resizer_output.iter().zip(streaming_output.iter()).enumerate() {
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
            .format(PixelFormat::Srgb8(PixelLayout::RgbaPremul))
            .build();

        let bg = SolidBackground::white(PixelLayout::RgbaPremul);
        let result = Resizer::with_background(&config, bg);
        assert!(
            matches!(result, Err(CompositeError::PremultipliedInput)),
            "expected PremultipliedInput error"
        );
    }

    #[test]
    fn f32_composite_opaque_bg() {
        let config = ResizeConfig::builder(10, 10, 10, 10)
            .filter(Filter::Lanczos)
            .format(PixelFormat::LinearF32(PixelLayout::Rgba))
            .build();

        // Semi-transparent input: RGBA = [0.5, 0.3, 0.1, 0.5]
        let mut input = vec![0.0f32; 10 * 10 * 4];
        for pixel in input.chunks_exact_mut(4) {
            pixel[0] = 0.5;
            pixel[1] = 0.3;
            pixel[2] = 0.1;
            pixel[3] = 0.5;
        }

        let bg = SolidBackground::from_linear(1.0, 1.0, 1.0, 1.0, PixelLayout::Rgba);
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
}

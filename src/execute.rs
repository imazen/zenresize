//! Layout execution: trim → orient → resize → canvas placement → edge replication.
//!
//! Executes a [`LayoutPlan`] on decoded pixel data, handling the full pipeline
//! from decoder output to final canvas. Operates on u8 (sRGB) pixel data.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::Filter;
use crate::composite::{Background, CompositeError};
use crate::layout::{
    CanvasColor, DecoderOffer, DecoderRequest, IdealLayout, LayoutPlan, Orientation,
};
use crate::resize::Resizer;
use whereat::{At, ResultAtExt, at};
use zenpixels::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor};

// ─── Canvas-space background offset adapter ────────────────────────────

/// Wraps a canvas-space [`Background`] to translate coordinates for the Resizer.
///
/// The inner background provides rows at canvas coordinates (0..canvas_h, canvas width).
/// This wrapper maps resize-output row `y` → canvas row `y + y_offset` and extracts
/// the sub-region starting at `x_offset` columns. Used internally by
/// [`execute_layout_with_background`] when the background covers the full canvas.
struct OffsetBackground<B: Background> {
    inner: B,
    x_offset: i32,
    y_offset: i32,
    canvas_w: u32,
    canvas_h: u32,
    ch: usize,
    /// Buffer for one full canvas-width row from inner.fill_row().
    canvas_row_buf: Vec<f32>,
}

impl<B: Background> OffsetBackground<B> {
    fn new(inner: B, px: i32, py: i32, canvas_w: u32, canvas_h: u32, ch: usize) -> Self {
        Self {
            inner,
            x_offset: px,
            y_offset: py,
            canvas_w,
            canvas_h,
            ch,
            canvas_row_buf: vec![0.0f32; canvas_w as usize * ch],
        }
    }

    fn into_inner(self) -> B {
        self.inner
    }
}

impl<B: Background> Background for OffsetBackground<B> {
    fn fill_row(&mut self, dst: &mut [f32], y: u32, channels: u8) {
        let canvas_y = self.y_offset as i64 + y as i64;

        // Row outside canvas bounds → transparent
        if canvas_y < 0 || canvas_y >= self.canvas_h as i64 {
            dst.fill(0.0);
            return;
        }

        // Fetch full canvas-width row from inner background
        self.inner
            .fill_row(&mut self.canvas_row_buf, canvas_y as u32, channels);

        let ch = self.ch;
        let dst_pixels = dst.len() / ch;

        // Handle x offset with clipping
        let x_start = self.x_offset as i64;
        let x_end = x_start + dst_pixels as i64;
        let canvas_w = self.canvas_w as i64;

        if x_start >= canvas_w || x_end <= 0 {
            dst.fill(0.0);
            return;
        }

        // Zero-fill, then copy the overlapping region
        dst.fill(0.0);
        let src_col_start = x_start.max(0) as usize;
        let src_col_end = x_end.min(canvas_w) as usize;
        let dst_col_start = (src_col_start as i64 - x_start) as usize;
        let copy_cols = src_col_end - src_col_start;

        dst[dst_col_start * ch..(dst_col_start + copy_cols) * ch]
            .copy_from_slice(&self.canvas_row_buf[src_col_start * ch..src_col_end * ch]);
    }

    #[inline(always)]
    fn is_transparent(&self) -> bool {
        self.inner.is_transparent()
    }

    #[inline(always)]
    fn is_opaque(&self) -> bool {
        self.inner.is_opaque()
    }

    // Deliberately return None — force per-row path since we need coordinate translation.
    #[inline(always)]
    fn solid_pixel(&self) -> Option<&[f32; 4]> {
        None
    }
}

/// Execute a finalized [`LayoutPlan`] on decoder output.
///
/// Pipeline: trim → orient → resize → canvas placement → edge replication.
///
/// Only supports u8 pixel formats. Panics if `desc` is not `ChannelType::U8`.
pub fn execute_layout(
    decoder_output: &[u8],
    decoder_width: u32,
    decoder_height: u32,
    plan: &LayoutPlan,
    desc: PixelDescriptor,
    filter: Filter,
) -> Vec<u8> {
    assert!(
        desc.channel_type() == ChannelType::U8,
        "execute_layout only supports u8 formats"
    );
    let ch = desc.channels();

    let expected_len = decoder_width as usize * decoder_height as usize * ch;
    assert!(
        decoder_output.len() >= expected_len,
        "decoder_output too small: {} < {}",
        decoder_output.len(),
        expected_len
    );

    // --- Step 1: Trim ---
    // Determine trim dimensions and whether we can use zero-copy (stride-based) path.
    // Zero-copy: trim + Identity orient + resize needed → pass offset+stride to Resizer.
    let (trim_w, trim_h) = if let Some(trim) = plan.trim {
        (trim.width, trim.height)
    } else {
        (decoder_width, decoder_height)
    };

    let zero_copy_trim =
        plan.trim.is_some() && plan.remaining_orientation.is_identity() && !plan.resize_is_identity;

    // For non-zero-copy trim, extract the sub-rectangle into owned data.
    let trim_owned: Option<Vec<u8>> = if let Some(trim) = plan.trim {
        if zero_copy_trim {
            None // will use slice + stride instead
        } else {
            Some(extract_rect(
                decoder_output,
                decoder_width,
                trim.x,
                trim.y,
                trim.width,
                trim.height,
                ch,
            ))
        }
    } else {
        None
    };

    // Get a reference to trimmed data + stride.
    let (trimmed, trim_stride): (&[u8], usize) = if zero_copy_trim {
        let trim = plan.trim.unwrap();
        let offset = (trim.y as usize * decoder_width as usize + trim.x as usize) * ch;
        let stride = decoder_width as usize * ch;
        (&decoder_output[offset..], stride)
    } else if let Some(ref owned) = trim_owned {
        (owned.as_slice(), trim_w as usize * ch)
    } else {
        (decoder_output, decoder_width as usize * ch)
    };

    // --- Step 2: Orient ---
    let (oriented, orient_w, orient_h) = if plan.remaining_orientation.is_identity() {
        (None, trim_w, trim_h)
    } else {
        // Orient needs tightly packed data
        let packed_stride = trim_w as usize * ch;
        let packed: &[u8] = if trim_stride != packed_stride {
            // This only happens if zero_copy_trim is true, but orient is not identity,
            // which is impossible (zero_copy_trim requires identity orient). So this
            // branch is dead code, but kept for safety.
            unreachable!("zero-copy trim requires identity orientation");
        } else {
            trimmed
        };
        let (result, new_w, new_h) = orient_image(
            packed,
            trim_w,
            trim_h,
            plan.remaining_orientation,
            desc.channels() as u8,
        );
        (Some(result), new_w, new_h)
    };

    // --- Step 3: Resize ---
    let (resized, resize_w, resize_h) = if plan.resize_is_identity {
        // No resize needed — produce tightly-packed owned data
        if let Some(data) = oriented {
            (data, orient_w, orient_h)
        } else {
            let packed_stride = trim_w as usize * ch;
            if trim_stride != packed_stride {
                (
                    compact_strided(trimmed, trim_w, trim_h, trim_stride, ch),
                    orient_w,
                    orient_h,
                )
            } else {
                (trimmed.to_vec(), orient_w, orient_h)
            }
        }
    } else {
        let rw = plan.resize_to.width;
        let rh = plan.resize_to.height;

        let builder = crate::ResizeConfig::builder(orient_w, orient_h, rw, rh)
            .filter(filter)
            .format(desc);

        if let Some(ref data) = oriented {
            let config = builder.build();
            (Resizer::new(&config).resize(data), rw, rh)
        } else {
            // Zero-copy trim path or no trim: pass stride to resizer
            let config = builder.in_stride(trim_stride).build();
            (Resizer::new(&config).resize(trimmed), rw, rh)
        }
    };

    // --- Step 4: Canvas + Place ---
    let canvas_w = plan.canvas.width;
    let canvas_h = plan.canvas.height;
    let (px, py) = plan.placement;

    let placed = if canvas_w == resize_w && canvas_h == resize_h && px == 0 && py == 0 {
        resized
    } else {
        let bg = canvas_color_to_pixel(&plan.canvas_color, desc);
        let mut canvas = fill_canvas(canvas_w, canvas_h, &bg);
        place_on_canvas(
            &mut canvas,
            canvas_w,
            canvas_h,
            &resized,
            resize_w,
            resize_h,
            px,
            py,
            ch,
        );
        canvas
    };

    // --- Step 5: Edge Replicate ---
    if let Some(content) = plan.content_size {
        let mut buf = placed;
        replicate_edges(
            &mut buf,
            canvas_w,
            canvas_h,
            content.width,
            content.height,
            ch,
        );
        buf
    } else {
        placed
    }
}

/// Execute an [`IdealLayout`] assuming full decode (no decoder negotiation).
///
/// Reconstructs [`DecoderRequest`] internally, creates [`DecoderOffer::full_decode`],
/// and finalizes into a [`LayoutPlan`] before executing.
///
/// Works for both primary and secondary planes — pass whichever `IdealLayout` you have.
/// For the primary, get it from [`crate::Pipeline::plan()`]. For a secondary plane,
/// get it from [`IdealLayout::derive_secondary()`].
pub fn execute(
    source_pixels: &[u8],
    ideal: &IdealLayout,
    desc: PixelDescriptor,
    filter: Filter,
) -> Vec<u8> {
    let pre_orient = ideal
        .orientation
        .inverse()
        .transform_dimensions(ideal.layout.source.width, ideal.layout.source.height);

    let request = match ideal.source_crop {
        Some(crop) => {
            DecoderRequest::new(ideal.layout.resize_to, ideal.orientation).with_crop(crop)
        }
        None => DecoderRequest::new(ideal.layout.resize_to, ideal.orientation),
    };
    let offer = DecoderOffer::full_decode(pre_orient.width, pre_orient.height);
    let plan = ideal.finalize(&request, &offer);
    execute_layout(
        source_pixels,
        pre_orient.width,
        pre_orient.height,
        &plan,
        desc,
        filter,
    )
}

/// Execute an [`IdealLayout`] with a real [`DecoderOffer`] (decoder negotiation).
///
/// Use this when the decoder can do partial work (prescale, crop, orientation).
/// The `request` and `ideal` come from [`crate::Pipeline::plan()`] (primary) or
/// [`IdealLayout::derive_secondary()`] (secondary). The `offer` describes
/// what the decoder actually produced.
///
/// # Decoder negotiation flow
///
/// ```text
/// Pipeline::plan()              → (IdealLayout, DecoderRequest)  // or derive_secondary()
///                                                  ↓
///                                     send request to decoder
///                                                  ↓
///                                     decoder returns offer + pixels
///                                                  ↓
///                         execute_with_offer(pixels, ideal, request, offer, ...)
/// ```
pub fn execute_with_offer(
    decoder_output: &[u8],
    ideal: &IdealLayout,
    request: &DecoderRequest,
    offer: &DecoderOffer,
    desc: PixelDescriptor,
    filter: Filter,
) -> Vec<u8> {
    let plan = ideal.finalize(request, offer);
    execute_layout(
        decoder_output,
        offer.dimensions.width,
        offer.dimensions.height,
        &plan,
        desc,
        filter,
    )
}

/// Convenience: derive and execute a secondary plane assuming full decode.
///
/// Combines [`IdealLayout::derive_secondary()`] + [`execute()`] in one call.
/// For decoder negotiation, call `derive_secondary()` yourself to get the
/// [`DecoderRequest`] hints, then use [`execute_with_offer()`] with the
/// secondary's `IdealLayout`.
///
/// # Arguments
///
/// * `source_pixels` — Fully decoded secondary plane pixels.
/// * `primary_ideal` — The primary plane's `IdealLayout`.
/// * `primary_source` — Source dimensions of the primary plane (before orientation).
/// * `secondary_source` — Source dimensions of the secondary plane (before orientation).
/// * `secondary_target` — Desired output dimensions, or `None` to auto-scale
///   proportionally (e.g., if gain map is 1/4 of SDR source, output is 1/4 of SDR output).
#[allow(clippy::too_many_arguments)]
pub fn execute_secondary(
    source_pixels: &[u8],
    primary_ideal: &IdealLayout,
    primary_source: crate::layout::Size,
    secondary_source: crate::layout::Size,
    secondary_target: Option<crate::layout::Size>,
    desc: PixelDescriptor,
    filter: Filter,
) -> Vec<u8> {
    let (sec_ideal, _sec_request) =
        primary_ideal.derive_secondary(primary_source, secondary_source, secondary_target);
    execute(source_pixels, &sec_ideal, desc, filter)
}

/// Execute a [`LayoutPlan`] with background compositing.
///
/// Same pipeline as [`execute_layout`] but composites the resized foreground
/// over the given background using Porter-Duff source-over.
///
/// # Canvas padding
///
/// When the canvas is larger than the resized image (padding case):
/// - For [`crate::SolidBackground`]: the padding area is filled with the solid color
///   (automatically converted to sRGB u8).
/// - For non-solid backgrounds (e.g. [`crate::SliceBackground`]): the padding area
///   is filled from the background data. The background must cover the full canvas
///   dimensions (`canvas_w × canvas_h` rows at `canvas_w * channels` elements each).
///   The resize compositing step automatically offsets into the background at the
///   placement position.
/// - Fallback: if no background data is available, `canvas_color` is used.
///
/// # Errors
///
/// Returns [`CompositeError::PremultipliedInput`] if `desc` uses premultiplied alpha.
#[allow(clippy::too_many_arguments)]
pub fn execute_layout_with_background<B: Background>(
    decoder_output: &[u8],
    decoder_width: u32,
    decoder_height: u32,
    plan: &LayoutPlan,
    desc: PixelDescriptor,
    filter: Filter,
    background: B,
) -> Result<Vec<u8>, At<CompositeError>> {
    // Transparent background → delegate to non-composite path
    if background.is_transparent() {
        return Ok(execute_layout(
            decoder_output,
            decoder_width,
            decoder_height,
            plan,
            desc,
            filter,
        ));
    }

    assert!(
        desc.channel_type() == ChannelType::U8,
        "execute_layout_with_background only supports u8 formats"
    );
    if desc.alpha == Some(AlphaMode::Premultiplied) {
        return Err(at!(CompositeError::PremultipliedInput));
    }
    let ch = desc.channels();

    let expected_len = decoder_width as usize * decoder_height as usize * ch;
    assert!(
        decoder_output.len() >= expected_len,
        "decoder_output too small: {} < {}",
        decoder_output.len(),
        expected_len
    );

    // --- Step 1: Trim ---
    let (trim_w, trim_h) = if let Some(trim) = plan.trim {
        (trim.width, trim.height)
    } else {
        (decoder_width, decoder_height)
    };

    let zero_copy_trim =
        plan.trim.is_some() && plan.remaining_orientation.is_identity() && !plan.resize_is_identity;

    let trim_owned: Option<Vec<u8>> = if let Some(trim) = plan.trim {
        if zero_copy_trim {
            None
        } else {
            Some(extract_rect(
                decoder_output,
                decoder_width,
                trim.x,
                trim.y,
                trim.width,
                trim.height,
                ch,
            ))
        }
    } else {
        None
    };

    let (trimmed, trim_stride): (&[u8], usize) = if zero_copy_trim {
        let trim = plan.trim.unwrap();
        let offset = (trim.y as usize * decoder_width as usize + trim.x as usize) * ch;
        let stride = decoder_width as usize * ch;
        (&decoder_output[offset..], stride)
    } else if let Some(ref owned) = trim_owned {
        (owned.as_slice(), trim_w as usize * ch)
    } else {
        (decoder_output, decoder_width as usize * ch)
    };

    // --- Step 2: Orient ---
    let (oriented, orient_w, orient_h) = if plan.remaining_orientation.is_identity() {
        (None, trim_w, trim_h)
    } else {
        let packed_stride = trim_w as usize * ch;
        let packed: &[u8] = if trim_stride != packed_stride {
            unreachable!("zero-copy trim requires identity orientation");
        } else {
            trimmed
        };
        let (result, new_w, new_h) = orient_image(
            packed,
            trim_w,
            trim_h,
            plan.remaining_orientation,
            desc.channels() as u8,
        );
        (Some(result), new_w, new_h)
    };

    // --- Step 3: Resize with compositing ---
    let canvas_w = plan.canvas.width;
    let canvas_h = plan.canvas.height;
    let (px, py) = plan.placement;
    let needs_canvas =
        canvas_w != plan.resize_to.width || canvas_h != plan.resize_to.height || px != 0 || py != 0;

    // Capture solid pixel for canvas fill before moving background into Resizer.
    let solid_fill: Option<Vec<u8>> = background
        .solid_pixel()
        .map(|pixel| premul_linear_f32_to_srgb_u8_pixel(pixel, desc));

    // For non-solid backgrounds with canvas expansion, the background covers the full
    // canvas. Wrap it with OffsetBackground so the Resizer sees rows at the placement
    // offset, then recover the inner background for canvas fill.
    let use_offset = needs_canvas && solid_fill.is_none();

    let rw = plan.resize_to.width;
    let rh = plan.resize_to.height;

    let (resize_w, resize_h) = if plan.resize_is_identity {
        (orient_w, orient_h)
    } else {
        (rw, rh)
    };

    // Even for identity resize, run through Resizer to get composite applied.
    let actual_rw = if plan.resize_is_identity {
        orient_w
    } else {
        rw
    };
    let actual_rh = if plan.resize_is_identity {
        orient_h
    } else {
        rh
    };

    let (resized, mut background) = if use_offset {
        let offset_bg = OffsetBackground::new(background, px, py, canvas_w, canvas_h, ch);
        let builder = crate::ResizeConfig::builder(orient_w, orient_h, actual_rw, actual_rh)
            .filter(filter)
            .format(desc);
        if let Some(ref data) = oriented {
            let config = builder.build();
            let mut resizer = Resizer::with_background(&config, offset_bg).at()?;
            let output = resizer.resize(data);
            (output, Some(resizer.into_background().into_inner()))
        } else {
            let config = builder.in_stride(trim_stride).build();
            let mut resizer = Resizer::with_background(&config, offset_bg).at()?;
            let output = resizer.resize(trimmed);
            (output, Some(resizer.into_background().into_inner()))
        }
    } else {
        let builder = crate::ResizeConfig::builder(orient_w, orient_h, actual_rw, actual_rh)
            .filter(filter)
            .format(desc);
        let resized = if let Some(ref data) = oriented {
            let config = builder.build();
            Resizer::with_background(&config, background)
                .at()?
                .resize(data)
        } else {
            let config = builder.in_stride(trim_stride).build();
            Resizer::with_background(&config, background)
                .at()?
                .resize(trimmed)
        };
        (resized, None)
    };

    // --- Step 4: Canvas + Place ---
    let placed = if !needs_canvas {
        resized
    } else if let Some(ref pixel) = solid_fill {
        // Solid background: fill canvas with single pixel, place resized on top
        let mut canvas = fill_canvas(canvas_w, canvas_h, pixel.as_slice());
        place_on_canvas(
            &mut canvas,
            canvas_w,
            canvas_h,
            &resized,
            resize_w,
            resize_h,
            px,
            py,
            ch,
        );
        canvas
    } else if let Some(ref mut bg) = background {
        // Non-solid background: fill canvas from background rows, place resized on top
        let mut canvas = fill_canvas_from_background(bg, canvas_w, canvas_h, desc);
        place_on_canvas(
            &mut canvas,
            canvas_w,
            canvas_h,
            &resized,
            resize_w,
            resize_h,
            px,
            py,
            ch,
        );
        canvas
    } else {
        // Fallback: canvas_color (shouldn't reach here with current logic)
        let fill_pixel = canvas_color_to_pixel(&plan.canvas_color, desc);
        let mut canvas = fill_canvas(canvas_w, canvas_h, &fill_pixel);
        place_on_canvas(
            &mut canvas,
            canvas_w,
            canvas_h,
            &resized,
            resize_w,
            resize_h,
            px,
            py,
            ch,
        );
        canvas
    };

    // --- Step 5: Edge Replicate ---
    Ok(if let Some(content) = plan.content_size {
        let mut buf = placed;
        replicate_edges(
            &mut buf,
            canvas_w,
            canvas_h,
            content.width,
            content.height,
            ch,
        );
        buf
    } else {
        placed
    })
}

/// Execute an [`IdealLayout`] with background compositing, assuming full decode.
///
/// Composited variant of [`execute()`]. See [`execute_layout_with_background()`]
/// for details on how compositing and canvas padding interact.
///
/// # Errors
///
/// Returns [`CompositeError::PremultipliedInput`] if `desc` uses premultiplied alpha.
pub fn execute_with_background<B: Background>(
    source_pixels: &[u8],
    ideal: &IdealLayout,
    desc: PixelDescriptor,
    filter: Filter,
    background: B,
) -> Result<Vec<u8>, At<CompositeError>> {
    let pre_orient = ideal
        .orientation
        .inverse()
        .transform_dimensions(ideal.layout.source.width, ideal.layout.source.height);

    let request = match ideal.source_crop {
        Some(crop) => {
            DecoderRequest::new(ideal.layout.resize_to, ideal.orientation).with_crop(crop)
        }
        None => DecoderRequest::new(ideal.layout.resize_to, ideal.orientation),
    };
    let offer = DecoderOffer::full_decode(pre_orient.width, pre_orient.height);
    let plan = ideal.finalize(&request, &offer);
    execute_layout_with_background(
        source_pixels,
        pre_orient.width,
        pre_orient.height,
        &plan,
        desc,
        filter,
        background,
    )
    .at()
}

/// Convenience: derive and execute a secondary plane with background compositing.
///
/// Composited variant of [`execute_secondary()`].
///
/// # Errors
///
/// Returns [`CompositeError::PremultipliedInput`] if `desc` uses premultiplied alpha.
#[allow(clippy::too_many_arguments)]
pub fn execute_secondary_with_background<B: Background>(
    source_pixels: &[u8],
    primary_ideal: &IdealLayout,
    primary_source: crate::layout::Size,
    secondary_source: crate::layout::Size,
    secondary_target: Option<crate::layout::Size>,
    desc: PixelDescriptor,
    filter: Filter,
    background: B,
) -> Result<Vec<u8>, At<CompositeError>> {
    let (sec_ideal, _sec_request) =
        primary_ideal.derive_secondary(primary_source, secondary_source, secondary_target);
    execute_with_background(source_pixels, &sec_ideal, desc, filter, background).at()
}

/// Apply an [`Orientation`] transform to an image buffer.
///
/// Returns `(transformed_pixels, new_width, new_height)`.
///
/// Input must be tightly packed (stride = width * channels).
pub fn orient_image(
    input: &[u8],
    width: u32,
    height: u32,
    orientation: Orientation,
    channels: u8,
) -> (Vec<u8>, u32, u32) {
    if orientation.is_identity() {
        return (input.to_vec(), width, height);
    }

    let ch = channels as usize;
    let out_size = orientation.transform_dimensions(width, height);
    let ow = out_size.width;
    let oh = out_size.height;
    let mut output = vec![0u8; ow as usize * oh as usize * ch];

    let w = width;
    let h = height;

    for sy in 0..height {
        for sx in 0..width {
            let (dx, dy) = forward_map(orientation, sx, sy, w, h);
            let src_off = (sy as usize * w as usize + sx as usize) * ch;
            let dst_off = (dy as usize * ow as usize + dx as usize) * ch;
            output[dst_off..dst_off + ch].copy_from_slice(&input[src_off..src_off + ch]);
        }
    }

    (output, ow, oh)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Forward-map a source pixel (sx, sy) to destination coordinates under the given orientation.
///
/// Formulas verified against `zenlayout/src/orientation.rs:526-536`.
#[inline]
fn forward_map(o: Orientation, x: u32, y: u32, w: u32, h: u32) -> (u32, u32) {
    match o {
        Orientation::Identity => (x, y),
        Orientation::FlipH => (w - 1 - x, y),
        Orientation::Rotate90 => (h - 1 - y, x),
        Orientation::Transpose => (y, x),
        Orientation::Rotate180 => (w - 1 - x, h - 1 - y),
        Orientation::FlipV => (x, h - 1 - y),
        Orientation::Rotate270 => (y, w - 1 - x),
        Orientation::Transverse => (h - 1 - y, w - 1 - x),
        _ => (x, y), // non_exhaustive fallback
    }
}

/// Copy a sub-rectangle from a tightly-packed source buffer.
fn extract_rect(src: &[u8], src_width: u32, x: u32, y: u32, w: u32, h: u32, ch: usize) -> Vec<u8> {
    let src_stride = src_width as usize * ch;
    let dst_stride = w as usize * ch;
    let mut out = vec![0u8; dst_stride * h as usize];

    for row in 0..h as usize {
        let src_off = (y as usize + row) * src_stride + x as usize * ch;
        let dst_off = row * dst_stride;
        out[dst_off..dst_off + dst_stride].copy_from_slice(&src[src_off..src_off + dst_stride]);
    }
    out
}

/// Compact strided data into tightly packed.
fn compact_strided(src: &[u8], w: u32, h: u32, stride: usize, ch: usize) -> Vec<u8> {
    let row_len = w as usize * ch;
    let mut out = vec![0u8; row_len * h as usize];
    for row in 0..h as usize {
        let src_off = row * stride;
        let dst_off = row * row_len;
        out[dst_off..dst_off + row_len].copy_from_slice(&src[src_off..src_off + row_len]);
    }
    out
}

/// Convert a [`CanvasColor`] to pixel bytes for the given descriptor.
#[allow(unreachable_patterns)] // non_exhaustive enums require wildcard arms
pub fn canvas_color_to_pixel(color: &CanvasColor, desc: PixelDescriptor) -> Vec<u8> {
    let ch = desc.channels();
    match color {
        CanvasColor::Transparent => vec![0u8; ch],
        CanvasColor::Srgb { r, g, b, a } => match desc.layout() {
            ChannelLayout::Gray => vec![*r], // approximate: use red channel
            ChannelLayout::Rgb => vec![*r, *g, *b],
            _ => {
                // Rgba, Bgra, etc.
                vec![*r, *g, *b, *a]
            }
        },
        CanvasColor::Linear { r, g, b, a } => {
            let sr = linear_srgb::default::linear_to_srgb_u8(*r);
            let sg = linear_srgb::default::linear_to_srgb_u8(*g);
            let sb = linear_srgb::default::linear_to_srgb_u8(*b);
            let sa = (*a * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            match desc.layout() {
                ChannelLayout::Gray => vec![sr],
                ChannelLayout::Rgb => vec![sr, sg, sb],
                _ => {
                    // Rgba, Bgra, etc.
                    vec![sr, sg, sb, sa]
                }
            }
        }
        _ => vec![0u8; ch],
    }
}

/// Convert a premultiplied linear f32 pixel to sRGB u8 for canvas fill.
fn premul_linear_f32_to_srgb_u8_pixel(pixel: &[f32; 4], desc: PixelDescriptor) -> Vec<u8> {
    // Unpremultiply
    let a = pixel[3];
    let (r, g, b) = if a > 1.0 / 1024.0 {
        let inv_a = 1.0 / a;
        (pixel[0] * inv_a, pixel[1] * inv_a, pixel[2] * inv_a)
    } else {
        (0.0, 0.0, 0.0)
    };
    // Linear to sRGB
    let sr = linear_srgb::default::linear_to_srgb_u8(r);
    let sg = linear_srgb::default::linear_to_srgb_u8(g);
    let sb = linear_srgb::default::linear_to_srgb_u8(b);
    let sa = (a * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    match desc.layout() {
        ChannelLayout::Gray => vec![sr],
        ChannelLayout::Rgb => vec![sr, sg, sb],
        _ => vec![sr, sg, sb, sa],
    }
}

/// Fill a canvas buffer with a repeated pixel value.
pub fn fill_canvas(w: u32, h: u32, pixel: &[u8]) -> Vec<u8> {
    let ch = pixel.len();
    let total = w as usize * h as usize * ch;
    let mut buf = vec![0u8; total];

    if pixel.iter().all(|&b| b == 0) {
        return buf; // already zeroed
    }

    // Fill first row
    let row_len = w as usize * ch;
    for x in 0..w as usize {
        buf[x * ch..(x + 1) * ch].copy_from_slice(pixel);
    }
    // Copy first row to all subsequent rows
    for row in 1..h as usize {
        let (head, tail) = buf.split_at_mut(row * row_len);
        tail[..row_len].copy_from_slice(&head[..row_len]);
    }
    buf
}

/// Fill a canvas buffer from a non-solid [`Background`] (premul linear f32 → sRGB u8).
///
/// Each canvas row is fetched from the background via `fill_row(y)`, unpremultiplied,
/// gamma-corrected to sRGB, and written as u8 bytes.
fn fill_canvas_from_background<B: Background>(
    background: &mut B,
    w: u32,
    h: u32,
    desc: PixelDescriptor,
) -> Vec<u8> {
    let ch = desc.channels();
    let row_len_u8 = w as usize * ch;
    let row_len_f32 = w as usize * ch;
    let mut buf = vec![0u8; h as usize * row_len_u8];
    let mut row_f32 = vec![0.0f32; row_len_f32];
    let has_alpha = desc.has_alpha();

    for y in 0..h {
        background.fill_row(&mut row_f32, y, ch as u8);
        let row_u8 = &mut buf[y as usize * row_len_u8..(y as usize + 1) * row_len_u8];
        premul_f32_row_to_srgb_u8(&row_f32, row_u8, has_alpha, ch);
    }
    buf
}

/// Convert a row of premultiplied linear f32 pixels to sRGB u8.
fn premul_f32_row_to_srgb_u8(src: &[f32], dst: &mut [u8], has_alpha: bool, ch: usize) {
    if has_alpha && ch == 4 {
        for (pixel_f32, pixel_u8) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            let a = pixel_f32[3];
            let (r, g, b) = if a > 1.0 / 1024.0 {
                let inv_a = 1.0 / a;
                (
                    pixel_f32[0] * inv_a,
                    pixel_f32[1] * inv_a,
                    pixel_f32[2] * inv_a,
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            pixel_u8[0] = linear_srgb::default::linear_to_srgb_u8(r);
            pixel_u8[1] = linear_srgb::default::linear_to_srgb_u8(g);
            pixel_u8[2] = linear_srgb::default::linear_to_srgb_u8(b);
            pixel_u8[3] = (a * 255.0 + 0.5).min(255.0).max(0.0) as u8;
        }
    } else {
        for (pixel_f32, pixel_u8) in src.chunks_exact(ch).zip(dst.chunks_exact_mut(ch)) {
            for i in 0..ch {
                pixel_u8[i] = linear_srgb::default::linear_to_srgb_u8(pixel_f32[i]);
            }
        }
    }
}

/// Blit an image onto a canvas at the given placement offset.
///
/// Handles negative placement (content extends past top-left) and clipping.
#[allow(clippy::too_many_arguments)]
pub fn place_on_canvas(
    canvas: &mut [u8],
    canvas_w: u32,
    canvas_h: u32,
    image: &[u8],
    img_w: u32,
    img_h: u32,
    px: i32,
    py: i32,
    ch: usize,
) {
    let canvas_stride = canvas_w as usize * ch;
    let img_stride = img_w as usize * ch;

    // Source region (clipped to canvas bounds)
    let src_x0 = if px < 0 { (-px) as u32 } else { 0 };
    let src_y0 = if py < 0 { (-py) as u32 } else { 0 };

    let dst_x0 = px.max(0) as u32;
    let dst_y0 = py.max(0) as u32;

    let copy_w = img_w
        .saturating_sub(src_x0)
        .min(canvas_w.saturating_sub(dst_x0));
    let copy_h = img_h
        .saturating_sub(src_y0)
        .min(canvas_h.saturating_sub(dst_y0));

    if copy_w == 0 || copy_h == 0 {
        return;
    }

    let copy_bytes = copy_w as usize * ch;

    for row in 0..copy_h as usize {
        let src_off = (src_y0 as usize + row) * img_stride + src_x0 as usize * ch;
        let dst_off = (dst_y0 as usize + row) * canvas_stride + dst_x0 as usize * ch;
        canvas[dst_off..dst_off + copy_bytes]
            .copy_from_slice(&image[src_off..src_off + copy_bytes]);
    }
}

/// Replicate right and bottom edges for MCU alignment padding.
///
/// For content rows: replicate the rightmost content pixel across extension columns.
/// Then copy the last full row (already right-extended) into all rows below.
pub fn replicate_edges(
    buf: &mut [u8],
    canvas_w: u32,
    canvas_h: u32,
    content_w: u32,
    content_h: u32,
    ch: usize,
) {
    if content_w >= canvas_w && content_h >= canvas_h {
        return; // nothing to replicate
    }

    let stride = canvas_w as usize * ch;

    // Right-extend: for each content row, replicate last content pixel
    if content_w < canvas_w {
        for row in 0..content_h.min(canvas_h) as usize {
            let last_pixel_off = row * stride + (content_w as usize - 1) * ch;
            // Copy the last content pixel to a temp buffer
            let mut pixel = [0u8; 16]; // max 4 channels * 4 bytes, but we're u8 so max 4
            pixel[..ch].copy_from_slice(&buf[last_pixel_off..last_pixel_off + ch]);
            // Fill extension columns
            for x in content_w as usize..canvas_w as usize {
                let off = row * stride + x * ch;
                buf[off..off + ch].copy_from_slice(&pixel[..ch]);
            }
        }
    }

    // Bottom-extend: copy the last content row (already right-extended) to all rows below
    if content_h < canvas_h {
        let last_row = (content_h as usize - 1) * stride;
        for row in content_h as usize..canvas_h as usize {
            let dst = row * stride;
            buf.copy_within(last_row..last_row + stride, dst);
        }
    }
}

// ─── LayoutPlan → ResizeConfig / StreamingResize helpers ─────────────

/// Build a [`ResizeConfig`] from a [`LayoutPlan`] with crop and padding.
///
/// Converts the plan's trim (→ [`crate::SourceRegion`]), resize target,
/// and canvas placement (→ [`crate::Padding`]) into a single config
/// suitable for [`crate::StreamingResize`] or [`crate::Resizer`].
///
/// Returns `None` if the plan requires a non-identity orientation
/// (rotations can't be expressed as row-at-a-time crop/pad).
///
/// **Edge replication** (`content_size`) is not handled — call
/// [`replicate_edges()`] on the final output if needed.
///
/// # Arguments
///
/// * `decoder_width`, `decoder_height` — Dimensions of the decoder output
///   (before trim/orient/resize).
/// * `plan` — The finalized layout plan.
/// * `desc` — Pixel format for both input and output.
/// * `filter` — Resampling filter.
pub fn config_from_plan(
    decoder_width: u32,
    decoder_height: u32,
    plan: &LayoutPlan,
    desc: PixelDescriptor,
    filter: Filter,
) -> Option<crate::ResizeConfig> {
    if !plan.remaining_orientation.is_identity() {
        return None;
    }

    let (trim_x, trim_y, trim_w, trim_h) = if let Some(trim) = plan.trim {
        (trim.x, trim.y, trim.width, trim.height)
    } else {
        (0, 0, decoder_width, decoder_height)
    };

    let rw = plan.resize_to.width;
    let rh = plan.resize_to.height;

    let mut builder = crate::ResizeConfig::builder(decoder_width, decoder_height, rw, rh)
        .filter(filter)
        .format(desc);

    // Source region from trim
    if plan.trim.is_some() {
        builder = builder.crop(trim_x, trim_y, trim_w, trim_h);
    }

    // Padding from canvas placement
    let canvas_w = plan.canvas.width;
    let canvas_h = plan.canvas.height;
    let (px, py) = plan.placement;

    let needs_canvas = canvas_w != rw || canvas_h != rh || px != 0 || py != 0;
    if needs_canvas {
        let pad_left = px.max(0) as u32;
        let pad_top = py.max(0) as u32;
        let pad_right = canvas_w.saturating_sub(rw + pad_left);
        let pad_bottom = canvas_h.saturating_sub(rh + pad_top);

        let color = canvas_color_to_f32(&plan.canvas_color, desc);
        builder = builder
            .padding(pad_top, pad_right, pad_bottom, pad_left)
            .padding_color(color);
    }

    Some(builder.build())
}

/// Create a [`crate::StreamingResize`] from a [`LayoutPlan`].
///
/// Returns `None` if the plan requires a non-identity orientation.
/// See [`config_from_plan()`] for details.
///
/// # Example
///
/// ```ignore
/// let plan = ideal.finalize(&request, &offer);
/// if let Some(mut stream) = streaming_from_plan(w, h, &plan, desc, filter) {
///     for y in 0..h {
///         stream.push_row(&rows[y]).unwrap();
///         while let Some(out) = stream.next_output_row() {
///             encoder.write_row(out);
///         }
///     }
///     stream.finish();
///     while let Some(out) = stream.next_output_row() {
///         encoder.write_row(out);
///     }
/// }
/// ```
pub fn streaming_from_plan(
    decoder_width: u32,
    decoder_height: u32,
    plan: &LayoutPlan,
    desc: PixelDescriptor,
    filter: Filter,
) -> Option<crate::StreamingResize> {
    let config = config_from_plan(decoder_width, decoder_height, plan, desc, filter)?;
    Some(crate::StreamingResize::new(&config))
}

/// Convert a [`CanvasColor`] to `[f32; 4]` in the output's color space (0.0–1.0).
fn canvas_color_to_f32(color: &CanvasColor, desc: PixelDescriptor) -> [f32; 4] {
    match color {
        CanvasColor::Transparent => [0.0, 0.0, 0.0, 0.0],
        CanvasColor::Srgb { r, g, b, a } => [
            *r as f32 / 255.0,
            *g as f32 / 255.0,
            *b as f32 / 255.0,
            *a as f32 / 255.0,
        ],
        CanvasColor::Linear { r, g, b, a } => {
            // Convert linear to output space
            if desc.transfer == zenpixels::TransferFunction::Linear {
                [*r, *g, *b, *a]
            } else {
                // Linear → sRGB (0.0–1.0 range)
                [
                    linear_srgb::default::linear_to_srgb_u8(*r) as f32 / 255.0,
                    linear_srgb::default::linear_to_srgb_u8(*g) as f32 / 255.0,
                    linear_srgb::default::linear_to_srgb_u8(*b) as f32 / 255.0,
                    *a,
                ]
            }
        }
        _ => [0.0, 0.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{
        CanvasColor, Constraint, ConstraintMode, DecoderOffer, LayoutPlan, Orientation, Pipeline,
        Rect, Size,
    };

    /// Create a test image where each pixel has a unique value based on position.
    fn make_test_image(w: u32, h: u32, ch: usize) -> Vec<u8> {
        let mut img = vec![0u8; w as usize * h as usize * ch];
        for y in 0..h {
            for x in 0..w {
                let off = (y as usize * w as usize + x as usize) * ch;
                let val = ((y * w + x) % 251) as u8; // prime mod to avoid collisions
                for c in 0..ch {
                    img[off + c] = val.wrapping_add(c as u8 * 60);
                }
            }
        }
        img
    }

    /// Get pixel at (x, y) from a tightly-packed buffer.
    fn get_pixel(buf: &[u8], w: u32, x: u32, y: u32, ch: usize) -> &[u8] {
        let off = (y as usize * w as usize + x as usize) * ch;
        &buf[off..off + ch]
    }

    // -----------------------------------------------------------------------
    // Test 1: orient_all_8 — 4×3 image with unique pixels, verify all 8
    // -----------------------------------------------------------------------
    #[test]
    fn orient_all_8() {
        let w = 4u32;
        let h = 3u32;
        let ch = 3usize;
        let img = make_test_image(w, h, ch);

        // Identity
        let (out, ow, oh) = orient_image(&img, w, h, Orientation::Identity, ch as u8);
        assert_eq!((ow, oh), (4, 3));
        assert_eq!(out, img);

        // FlipH: (x,y) → (w-1-x, y)
        let (out, ow, oh) = orient_image(&img, w, h, Orientation::FlipH, ch as u8);
        assert_eq!((ow, oh), (4, 3));
        assert_eq!(get_pixel(&out, ow, 3, 0, ch), get_pixel(&img, w, 0, 0, ch));
        assert_eq!(get_pixel(&out, ow, 0, 0, ch), get_pixel(&img, w, 3, 0, ch));

        // FlipV: (x,y) → (x, h-1-y)
        let (out, ow, oh) = orient_image(&img, w, h, Orientation::FlipV, ch as u8);
        assert_eq!((ow, oh), (4, 3));
        assert_eq!(get_pixel(&out, ow, 0, 2, ch), get_pixel(&img, w, 0, 0, ch));
        assert_eq!(get_pixel(&out, ow, 0, 0, ch), get_pixel(&img, w, 0, 2, ch));

        // Rotate90: (x,y) → (h-1-y, x), output is h×w
        let (out, ow, oh) = orient_image(&img, w, h, Orientation::Rotate90, ch as u8);
        assert_eq!((ow, oh), (3, 4)); // swapped
        assert_eq!(get_pixel(&out, ow, 2, 0, ch), get_pixel(&img, w, 0, 0, ch));
        assert_eq!(get_pixel(&out, ow, 0, 0, ch), get_pixel(&img, w, 0, 2, ch));

        // Rotate180: (x,y) → (w-1-x, h-1-y)
        let (out, ow, oh) = orient_image(&img, w, h, Orientation::Rotate180, ch as u8);
        assert_eq!((ow, oh), (4, 3));
        assert_eq!(get_pixel(&out, ow, 3, 2, ch), get_pixel(&img, w, 0, 0, ch));
        assert_eq!(get_pixel(&out, ow, 0, 0, ch), get_pixel(&img, w, 3, 2, ch));

        // Rotate270: (x,y) → (y, w-1-x), output is h×w
        let (out, ow, oh) = orient_image(&img, w, h, Orientation::Rotate270, ch as u8);
        assert_eq!((ow, oh), (3, 4));
        // source (0,0) → dest (0, 3)
        assert_eq!(get_pixel(&out, ow, 0, 3, ch), get_pixel(&img, w, 0, 0, ch));
        // source (0,2) → dest (2, 3)
        assert_eq!(get_pixel(&out, ow, 2, 3, ch), get_pixel(&img, w, 0, 2, ch));

        // Transpose: (x,y) → (y, x), output is h×w
        let (out, ow, oh) = orient_image(&img, w, h, Orientation::Transpose, ch as u8);
        assert_eq!((ow, oh), (3, 4));
        assert_eq!(get_pixel(&out, ow, 0, 0, ch), get_pixel(&img, w, 0, 0, ch));
        assert_eq!(get_pixel(&out, ow, 1, 0, ch), get_pixel(&img, w, 0, 1, ch));
        assert_eq!(get_pixel(&out, ow, 0, 1, ch), get_pixel(&img, w, 1, 0, ch));

        // Transverse: (x,y) → (h-1-y, w-1-x), output is h×w
        let (out, ow, oh) = orient_image(&img, w, h, Orientation::Transverse, ch as u8);
        assert_eq!((ow, oh), (3, 4));
        assert_eq!(get_pixel(&out, ow, 2, 3, ch), get_pixel(&img, w, 0, 0, ch));
    }

    // -----------------------------------------------------------------------
    // Test 2: orient_roundtrip — apply then inverse → identity
    // -----------------------------------------------------------------------
    #[test]
    fn orient_roundtrip() {
        let w = 5u32;
        let h = 7u32;
        let ch = 4usize;
        let img = make_test_image(w, h, ch);

        for orient in [
            Orientation::FlipH,
            Orientation::FlipV,
            Orientation::Rotate90,
            Orientation::Rotate180,
            Orientation::Rotate270,
            Orientation::Transpose,
            Orientation::Transverse,
        ] {
            let (fwd, fw, fh) = orient_image(&img, w, h, orient, ch as u8);
            let inv = orient.inverse();
            let (back, bw, bh) = orient_image(&fwd, fw, fh, inv, ch as u8);
            assert_eq!((bw, bh), (w, h), "roundtrip dims for {orient:?}");
            assert_eq!(back, img, "roundtrip pixels for {orient:?}");
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: extract_rect_basic — 10×10 gradient, extract 5×5 sub-rect
    // -----------------------------------------------------------------------
    #[test]
    fn extract_rect_basic() {
        let w = 10u32;
        let ch = 3usize;
        let img = make_test_image(w, 10, ch);

        let sub = extract_rect(&img, w, 2, 3, 5, 5, ch);
        assert_eq!(sub.len(), 5 * 5 * ch);

        // Verify a few pixels
        for dy in 0..5u32 {
            for dx in 0..5u32 {
                let expected = get_pixel(&img, w, 2 + dx, 3 + dy, ch);
                let got = get_pixel(&sub, 5, dx, dy, ch);
                assert_eq!(got, expected, "pixel ({dx},{dy})");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: canvas_fill_and_place — 5×5 red on 10×10 white at (2,2)
    // -----------------------------------------------------------------------
    #[test]
    fn canvas_fill_and_place() {
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let white = canvas_color_to_pixel(&CanvasColor::white(), format);
        assert_eq!(white, vec![255, 255, 255, 255]);

        let mut canvas = fill_canvas(10, 10, &white);
        assert_eq!(canvas.len(), 10 * 10 * 4);

        // 5×5 red image
        let red_pixel = [255u8, 0, 0, 255];
        let red_img: Vec<u8> = red_pixel.iter().copied().cycle().take(5 * 5 * 4).collect();

        place_on_canvas(&mut canvas, 10, 10, &red_img, 5, 5, 2, 2, ch);

        // Check a white pixel outside the placed area
        assert_eq!(get_pixel(&canvas, 10, 0, 0, ch), &[255, 255, 255, 255]);
        assert_eq!(get_pixel(&canvas, 10, 1, 1, ch), &[255, 255, 255, 255]);

        // Check a red pixel inside the placed area
        assert_eq!(get_pixel(&canvas, 10, 2, 2, ch), &[255, 0, 0, 255]);
        assert_eq!(get_pixel(&canvas, 10, 6, 6, ch), &[255, 0, 0, 255]);

        // Check boundary
        assert_eq!(get_pixel(&canvas, 10, 7, 2, ch), &[255, 255, 255, 255]);
    }

    // -----------------------------------------------------------------------
    // Test 5: canvas_negative_placement — partial offscreen
    // -----------------------------------------------------------------------
    #[test]
    fn canvas_negative_placement() {
        let ch = 3usize;
        let bg = vec![0u8; ch];
        let mut canvas = fill_canvas(10, 10, &bg);

        // 6×6 image with unique pixels
        let img = make_test_image(6, 6, ch);

        // Place at (-2, -3): only the bottom-right 4×3 of the image is visible
        place_on_canvas(&mut canvas, 10, 10, &img, 6, 6, -2, -3, ch);

        // Canvas pixel (0,0) should be image pixel (2,3)
        assert_eq!(
            get_pixel(&canvas, 10, 0, 0, ch),
            get_pixel(&img, 6, 2, 3, ch)
        );
        // Canvas pixel (3,2) should be image pixel (5,5)
        assert_eq!(
            get_pixel(&canvas, 10, 3, 2, ch),
            get_pixel(&img, 6, 5, 5, ch)
        );
        // Canvas pixel (4,0) should be background (image only 4 wide visible)
        assert_eq!(get_pixel(&canvas, 10, 4, 0, ch), &[0, 0, 0]);
    }

    // -----------------------------------------------------------------------
    // Test 6: edge_replicate_basic — 6×6 canvas with 4×4 content
    // -----------------------------------------------------------------------
    #[test]
    fn edge_replicate_basic() {
        let ch = 3usize;
        let cw = 6u32;
        let ch_ = 6u32;
        let content_w = 4u32;
        let content_h = 4u32;
        let img = make_test_image(cw, ch_, ch);
        let mut buf = img.clone();

        replicate_edges(&mut buf, cw, ch_, content_w, content_h, ch);

        // Right extension: pixel (4,0) and (5,0) should match (3,0)
        assert_eq!(get_pixel(&buf, cw, 4, 0, ch), get_pixel(&buf, cw, 3, 0, ch));
        assert_eq!(get_pixel(&buf, cw, 5, 0, ch), get_pixel(&buf, cw, 3, 0, ch));

        // Right extension on row 2
        assert_eq!(get_pixel(&buf, cw, 5, 2, ch), get_pixel(&buf, cw, 3, 2, ch));

        // Bottom extension: row 4 and 5 should match row 3 (after right-extend)
        let stride = cw as usize * ch;
        let row3 = &buf[3 * stride..4 * stride].to_vec();
        assert_eq!(&buf[4 * stride..5 * stride], row3.as_slice());
        assert_eq!(&buf[5 * stride..6 * stride], row3.as_slice());
    }

    // -----------------------------------------------------------------------
    // Test 7: identity_plan_passthrough — no-op plan returns identical pixels
    // -----------------------------------------------------------------------
    #[test]
    fn identity_plan_passthrough() {
        let w = 8u32;
        let h = 6u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let img = make_test_image(w, h, ch);

        let plan = LayoutPlan::identity(Size::new(w, h));

        let result = execute_layout(&img, w, h, &plan, format, Filter::Lanczos);
        assert_eq!(result, img);
    }

    // -----------------------------------------------------------------------
    // Test 8: full_pipeline_fit_pad — Pipeline FitPad → execute, verify dims + padding
    // -----------------------------------------------------------------------
    #[test]
    fn full_pipeline_fit_pad() {
        let src_w = 100u32;
        let src_h = 50u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let img = make_test_image(src_w, src_h, ch);

        let (ideal, request) = Pipeline::new(src_w, src_h).fit_pad(80, 80).plan().unwrap();

        let offer = DecoderOffer::full_decode(src_w, src_h);
        let plan = ideal.finalize(&request, &offer);

        // FitPad 100×50 into 80×80: resize to 80×40, pad to 80×80
        assert_eq!(plan.canvas, Size::new(80, 80));
        assert_eq!(plan.resize_to, Size::new(80, 40));

        let result = execute_layout(&img, src_w, src_h, &plan, format, Filter::Lanczos);
        assert_eq!(result.len(), 80 * 80 * ch);

        // The padding area (top rows) should be transparent (default canvas color)
        // Placement should center the 80×40 image vertically: y = (80-40)/2 = 20
        assert_eq!(plan.placement.1, 20);
        // Top-left corner should be transparent
        assert_eq!(get_pixel(&result, 80, 0, 0, ch), &[0, 0, 0, 0]);
        // A pixel in the image area should be non-zero (most likely)
        let mid_pixel = get_pixel(&result, 80, 40, 40, ch);
        assert!(
            mid_pixel.iter().any(|&b| b != 0),
            "center pixel should have content"
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: full_pipeline_orientation — auto_orient + resize
    // -----------------------------------------------------------------------
    #[test]
    fn full_pipeline_orientation() {
        // Source is 60×40 but stored as EXIF 6 (Rotate90), so logical is 40×60
        let stored_w = 60u32;
        let stored_h = 40u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let img = make_test_image(stored_w, stored_h, ch);

        let (ideal, request) = Pipeline::new(stored_w, stored_h)
            .auto_orient(6) // Rotate90: logical becomes 40×60
            .within(20, 30) // Fit 40×60 within 20×30 → 20×30
            .plan()
            .unwrap();

        let offer = DecoderOffer::full_decode(stored_w, stored_h);
        let plan = ideal.finalize(&request, &offer);

        let result = execute_layout(&img, stored_w, stored_h, &plan, format, Filter::Lanczos);
        assert_eq!(
            result.len(),
            plan.canvas.width as usize * plan.canvas.height as usize * ch
        );
        assert_eq!(plan.canvas, Size::new(20, 30));
    }

    // -----------------------------------------------------------------------
    // Test 10: zero_copy_trim_matches_explicit — stride-based trim == extract + resize
    // -----------------------------------------------------------------------
    #[test]
    fn zero_copy_trim_matches_explicit() {
        let src_w = 20u32;
        let src_h = 20u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let img = make_test_image(src_w, src_h, ch);

        let trim = Rect::new(5, 5, 10, 10);

        // Method A: explicit extract + resize
        let extracted = extract_rect(&img, src_w, trim.x, trim.y, trim.width, trim.height, ch);
        let config_a = crate::ResizeConfig::builder(10, 10, 6, 6)
            .filter(Filter::Lanczos)
            .format(format)
            .build();
        let result_a = Resizer::new(&config_a).resize(&extracted);

        // Method B: via execute_layout with trim (uses zero-copy stride path)
        let target = Size::new(6, 6);
        let plan = LayoutPlan::identity(target)
            .with_trim(trim)
            .with_resize_to(target)
            .with_canvas(target);

        let result_b = execute_layout(&img, src_w, src_h, &plan, format, Filter::Lanczos);

        assert_eq!(result_a.len(), result_b.len());
        assert_eq!(
            result_a, result_b,
            "zero-copy trim must match explicit extract+resize"
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: execute_with_offer — decoder handles orientation
    // -----------------------------------------------------------------------
    #[test]
    fn execute_with_offer_decoder_orients() {
        // Source is 60×40, EXIF 6 (Rotate90). Logical = 40×60.
        let stored_w = 60u32;
        let stored_h = 40u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let stored_img = make_test_image(stored_w, stored_h, ch);

        let (ideal, request) = Pipeline::new(stored_w, stored_h)
            .auto_orient(6)
            .within(20, 30)
            .plan()
            .unwrap();

        // Simulate decoder applying orientation: output is 40×60 (already rotated)
        let (oriented_img, ow, oh) = orient_image(
            &stored_img,
            stored_w,
            stored_h,
            Orientation::Rotate90,
            ch as u8,
        );
        assert_eq!((ow, oh), (40, 60));

        let offer =
            DecoderOffer::full_decode(ow, oh).with_orientation_applied(Orientation::Rotate90);

        let result = execute_with_offer(
            &oriented_img,
            &ideal,
            &request,
            &offer,
            format,
            Filter::Lanczos,
        );
        assert_eq!(result.len(), 20 * 30 * ch);

        // Compare against full_decode path where we orient ourselves
        let offer_full = DecoderOffer::full_decode(stored_w, stored_h);
        let result_full = execute_with_offer(
            &stored_img,
            &ideal,
            &request,
            &offer_full,
            format,
            Filter::Lanczos,
        );
        assert_eq!(result_full.len(), 20 * 30 * ch);

        // Both paths should produce the same output
        assert_eq!(result, result_full);
    }

    // -----------------------------------------------------------------------
    // Test 12: execute_secondary — gain map full decode
    // -----------------------------------------------------------------------
    #[test]
    fn execute_secondary_gain_map() {
        let sdr_w = 400u32;
        let sdr_h = 300u32;
        let gm_w = 100u32; // 1/4 scale gain map
        let gm_h = 75u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let gm_img = make_test_image(gm_w, gm_h, ch);

        // Primary: auto_orient(6) + fit(200, 200)
        // EXIF 6 = Rotate90, so logical SDR = 300×400, fit → 150×200
        let (sdr_ideal, sdr_req) = Pipeline::new(sdr_w, sdr_h)
            .auto_orient(6)
            .fit(200, 200)
            .plan()
            .unwrap();

        // Full decode convenience path
        let result = execute_secondary(
            &gm_img,
            &sdr_ideal,
            Size::new(sdr_w, sdr_h),
            Size::new(gm_w, gm_h),
            None,
            format,
            Filter::Lanczos,
        );

        let result_pixels = result.len() / ch;
        assert!(result_pixels > 0, "secondary should produce output");

        // Verify orientation matches primary
        let (gm_ideal, gm_req) =
            sdr_ideal.derive_secondary(Size::new(sdr_w, sdr_h), Size::new(gm_w, gm_h), None);
        let gm_offer = DecoderOffer::full_decode(gm_w, gm_h);
        let gm_plan = gm_ideal.finalize(&gm_req, &gm_offer);
        let sdr_offer = DecoderOffer::full_decode(sdr_w, sdr_h);
        let sdr_plan = sdr_ideal.finalize(&sdr_req, &sdr_offer);
        assert_eq!(
            sdr_plan.remaining_orientation, gm_plan.remaining_orientation,
            "secondary must have same orientation as primary"
        );

        assert_eq!(
            result.len(),
            gm_plan.canvas.width as usize * gm_plan.canvas.height as usize * ch
        );
    }

    // -----------------------------------------------------------------------
    // Test 12b: secondary negotiation via derive_secondary + execute_with_offer
    // -----------------------------------------------------------------------
    #[test]
    fn secondary_negotiated_decoder_orients() {
        let sdr_w = 400u32;
        let sdr_h = 300u32;
        let gm_w = 100u32;
        let gm_h = 75u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let gm_stored = make_test_image(gm_w, gm_h, ch);

        let (sdr_ideal, _sdr_req) = Pipeline::new(sdr_w, sdr_h)
            .auto_orient(6) // Rotate90
            .fit(200, 200)
            .plan()
            .unwrap();

        // Step 1: derive secondary → get request hints
        let (gm_ideal, gm_request) =
            sdr_ideal.derive_secondary(Size::new(sdr_w, sdr_h), Size::new(gm_w, gm_h), None);

        // Step 2: decoder applies orientation using the hints
        let (gm_oriented, ow, oh) =
            orient_image(&gm_stored, gm_w, gm_h, Orientation::Rotate90, ch as u8);

        let gm_offer =
            DecoderOffer::full_decode(ow, oh).with_orientation_applied(Orientation::Rotate90);

        // Step 3: execute with offer — same function as primary
        let result_negotiated = execute_with_offer(
            &gm_oriented,
            &gm_ideal,
            &gm_request,
            &gm_offer,
            format,
            Filter::Lanczos,
        );

        // Compare against full decode convenience path
        let result_full = execute_secondary(
            &gm_stored,
            &sdr_ideal,
            Size::new(sdr_w, sdr_h),
            Size::new(gm_w, gm_h),
            None,
            format,
            Filter::Lanczos,
        );

        assert_eq!(result_negotiated.len(), result_full.len());
        assert_eq!(
            result_negotiated, result_full,
            "negotiated secondary must match full-decode secondary"
        );
    }

    // -----------------------------------------------------------------------
    // Test 13: execute_with_offer matches execute for full decode
    // -----------------------------------------------------------------------
    #[test]
    fn execute_with_offer_full_decode_matches_execute() {
        let w = 80u32;
        let h = 60u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let img = make_test_image(w, h, ch);

        let (ideal, request) = Pipeline::new(w, h).fit(40, 40).plan().unwrap();

        let result_exec = execute(&img, &ideal, format, Filter::Lanczos);
        let offer = DecoderOffer::full_decode(w, h);
        let result_offer =
            execute_with_offer(&img, &ideal, &request, &offer, format, Filter::Lanczos);

        assert_eq!(result_exec, result_offer);
    }

    // -----------------------------------------------------------------------
    // Test 14: execute works for secondary plane directly
    // -----------------------------------------------------------------------
    #[test]
    fn execute_works_for_secondary_ideal() {
        let sdr_w = 400u32;
        let sdr_h = 300u32;
        let gm_w = 100u32;
        let gm_h = 75u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let gm_img = make_test_image(gm_w, gm_h, ch);

        let (sdr_ideal, _sdr_req) = Pipeline::new(sdr_w, sdr_h)
            .auto_orient(6)
            .fit(200, 200)
            .plan()
            .unwrap();

        // derive_secondary + execute should match execute_secondary
        let (gm_ideal, _gm_req) =
            sdr_ideal.derive_secondary(Size::new(sdr_w, sdr_h), Size::new(gm_w, gm_h), None);

        let result_direct = execute(&gm_img, &gm_ideal, format, Filter::Lanczos);
        let result_convenience = execute_secondary(
            &gm_img,
            &sdr_ideal,
            Size::new(sdr_w, sdr_h),
            Size::new(gm_w, gm_h),
            None,
            format,
            Filter::Lanczos,
        );

        assert_eq!(result_direct, result_convenience);
    }

    // -----------------------------------------------------------------------
    // Test 15: no background matches execute_layout
    // -----------------------------------------------------------------------
    #[test]
    fn execute_with_no_background_matches_plain() {
        let w = 80u32;
        let h = 60u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let img = make_test_image(w, h, ch);

        let (ideal, request) = Pipeline::new(w, h).fit(40, 40).plan().unwrap();
        let offer = DecoderOffer::full_decode(w, h);
        let plan = ideal.finalize(&request, &offer);

        let result_plain = execute_layout(&img, w, h, &plan, format, Filter::Lanczos);
        let result_bg = execute_layout_with_background(
            &img,
            w,
            h,
            &plan,
            format,
            Filter::Lanczos,
            crate::composite::NoBackground,
        )
        .unwrap();

        assert_eq!(result_plain, result_bg);
    }

    // -----------------------------------------------------------------------
    // Test 16: solid opaque white bg + FitPad → padding is white, content composited
    // -----------------------------------------------------------------------
    #[test]
    fn execute_with_solid_white_bg_fit_pad() {
        let src_w = 100u32;
        let src_h = 50u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        // Semi-transparent RGBA image
        let mut img = vec![0u8; src_w as usize * src_h as usize * ch];
        for pixel in img.chunks_exact_mut(ch) {
            pixel[0] = 200; // R
            pixel[1] = 100; // G
            pixel[2] = 50; // B
            pixel[3] = 128; // A = 50%
        }

        let (ideal, request) = Pipeline::new(src_w, src_h)
            .constrain(
                Constraint::new(ConstraintMode::FitPad, 80, 80).canvas_color(CanvasColor::white()),
            )
            .plan()
            .unwrap();

        let offer = DecoderOffer::full_decode(src_w, src_h);
        let plan = ideal.finalize(&request, &offer);

        // FitPad 100×50 → 80×80: resize to 80×40, pad to 80×80
        assert_eq!(plan.canvas, Size::new(80, 80));
        assert_eq!(plan.resize_to, Size::new(80, 40));
        let py = plan.placement.1; // vertical offset (should be 20)

        let bg = crate::composite::SolidBackground::white(format);
        let result =
            execute_layout_with_background(&img, src_w, src_h, &plan, format, Filter::Lanczos, bg)
                .unwrap();

        assert_eq!(result.len(), 80 * 80 * ch);

        // Padding area (top rows before py) should be white (255,255,255,255)
        let top_pixel = get_pixel(&result, 80, 0, 0, ch);
        assert_eq!(top_pixel, &[255, 255, 255, 255], "padding should be white");

        // Bottom padding should also be white
        let bottom_pixel = get_pixel(&result, 80, 0, 79, ch);
        assert_eq!(bottom_pixel, &[255, 255, 255, 255], "bottom padding white");

        // Content area should be fully opaque (composited over white)
        let content_pixel = get_pixel(&result, 80, 40, (py + 20) as u32, ch);
        assert_eq!(content_pixel[3], 255, "composited content should be opaque");

        // Content RGB should be blended (not raw input values)
        // With 50% alpha over white: out ≈ fg * 0.5 + white * 0.5
        // Not testing exact values due to sRGB conversion, just that it's not pure white
        assert!(content_pixel[0] > 200, "red channel should be bright");
        assert!(
            content_pixel[0] < 255,
            "red channel should not be pure white"
        );
    }

    // -----------------------------------------------------------------------
    // Test 17: transparent background matches no background
    // -----------------------------------------------------------------------
    #[test]
    fn execute_transparent_bg_matches_no_bg() {
        let w = 60u32;
        let h = 40u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let img = make_test_image(w, h, ch);

        let (ideal, request) = Pipeline::new(w, h).fit(30, 30).plan().unwrap();
        let offer = DecoderOffer::full_decode(w, h);
        let plan = ideal.finalize(&request, &offer);

        let result_no_bg = execute_layout_with_background(
            &img,
            w,
            h,
            &plan,
            format,
            Filter::Lanczos,
            crate::composite::NoBackground,
        )
        .unwrap();

        let transparent_bg = crate::composite::SolidBackground::transparent(format);
        let result_transparent = execute_layout_with_background(
            &img,
            w,
            h,
            &plan,
            format,
            Filter::Lanczos,
            transparent_bg,
        )
        .unwrap();

        assert_eq!(result_no_bg, result_transparent);
    }

    // -----------------------------------------------------------------------
    // Test 18: rejects premultiplied input
    // -----------------------------------------------------------------------
    #[test]
    fn execute_rejects_premultiplied() {
        let w = 10u32;
        let h = 10u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied));
        let img = vec![128u8; w as usize * h as usize * ch];

        let (ideal, request) = Pipeline::new(w, h).fit(5, 5).plan().unwrap();
        let offer = DecoderOffer::full_decode(w, h);
        let plan = ideal.finalize(&request, &offer);

        let bg = crate::composite::SolidBackground::white(format);
        let result = execute_layout_with_background(&img, w, h, &plan, format, Filter::Lanczos, bg);

        assert!(matches!(
            result.as_ref().map_err(|e| e.error()),
            Err(&CompositeError::PremultipliedInput)
        ));
    }

    // -----------------------------------------------------------------------
    // Test 19: config_from_plan with trim → SourceRegion
    // -----------------------------------------------------------------------
    #[test]
    fn config_from_plan_trim_to_source_region() {
        let src_w = 100u32;
        let src_h = 80u32;
        let format = PixelDescriptor::RGBA8_SRGB;

        // Trim 20,10,60,50 → resize to 30×25
        let plan = LayoutPlan::identity(Size::new(30, 25))
            .with_trim(Rect::new(20, 10, 60, 50))
            .with_resize_to(Size::new(30, 25))
            .with_canvas(Size::new(30, 25));

        let config = config_from_plan(src_w, src_h, &plan, format, Filter::Lanczos)
            .expect("identity orientation should produce config");

        // Source region should match trim
        let region = config.source_region.as_ref().expect("should have source region");
        assert_eq!(region.x, 20);
        assert_eq!(region.y, 10);
        assert_eq!(region.width, 60);
        assert_eq!(region.height, 50);

        // Resize dimensions
        assert_eq!(config.out_width, 30);
        assert_eq!(config.out_height, 25);

        // No padding (canvas == resize_to)
        assert!(config.padding.is_none());
    }

    // -----------------------------------------------------------------------
    // Test 20: config_from_plan with FitPad → Padding
    // -----------------------------------------------------------------------
    #[test]
    fn config_from_plan_fit_pad_to_padding() {
        let src_w = 100u32;
        let src_h = 50u32;
        let format = PixelDescriptor::RGBA8_SRGB;

        let (ideal, request) = Pipeline::new(src_w, src_h)
            .constrain(
                Constraint::new(ConstraintMode::FitPad, 80, 80).canvas_color(CanvasColor::white()),
            )
            .plan()
            .unwrap();

        let offer = DecoderOffer::full_decode(src_w, src_h);
        let plan = ideal.finalize(&request, &offer);

        // FitPad 100×50 into 80×80: resize to 80×40, pad to 80×80
        assert_eq!(plan.resize_to, Size::new(80, 40));
        assert_eq!(plan.canvas, Size::new(80, 80));

        let config = config_from_plan(src_w, src_h, &plan, format, Filter::Lanczos)
            .expect("identity orientation should produce config");

        // Resize target
        assert_eq!(config.out_width, 80);
        assert_eq!(config.out_height, 40);

        // Padding: centered vertically → 20px top, 20px bottom
        let padding = config.padding.as_ref().expect("should have padding");
        assert_eq!(padding.top, 20);
        assert_eq!(padding.bottom, 20);
        assert_eq!(padding.left, 0);
        assert_eq!(padding.right, 0);

        // Canvas color is white sRGB → [1.0, 1.0, 1.0, 1.0]
        assert_eq!(padding.color, [1.0, 1.0, 1.0, 1.0]);

        // Total output dimensions
        assert_eq!(config.total_output_width(), 80);
        assert_eq!(config.total_output_height(), 80);
    }

    // -----------------------------------------------------------------------
    // Test 21: config_from_plan returns None for non-identity orientation
    // -----------------------------------------------------------------------
    #[test]
    fn config_from_plan_none_for_rotation() {
        let format = PixelDescriptor::RGBA8_SRGB;
        let plan = LayoutPlan::identity(Size::new(10, 10))
            .with_remaining_orientation(Orientation::Rotate90);

        assert!(config_from_plan(10, 10, &plan, format, Filter::Lanczos).is_none());
    }

    // -----------------------------------------------------------------------
    // Test 22: streaming_from_plan matches execute_layout for identity plans
    // -----------------------------------------------------------------------
    #[test]
    fn streaming_from_plan_matches_execute_layout() {
        let src_w = 100u32;
        let src_h = 50u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let img = make_test_image(src_w, src_h, ch);

        let (ideal, request) = Pipeline::new(src_w, src_h)
            .constrain(
                Constraint::new(ConstraintMode::FitPad, 80, 80).canvas_color(CanvasColor::white()),
            )
            .plan()
            .unwrap();

        let offer = DecoderOffer::full_decode(src_w, src_h);
        let plan = ideal.finalize(&request, &offer);

        // Fullframe path
        let result_fullframe = execute_layout(&img, src_w, src_h, &plan, format, Filter::Lanczos);

        // Streaming path
        let mut stream = streaming_from_plan(src_w, src_h, &plan, format, Filter::Lanczos)
            .expect("identity orientation");

        let row_len = src_w as usize * ch;
        let mut output_rows: Vec<Vec<u8>> = Vec::new();

        for y in 0..src_h {
            let row = &img[y as usize * row_len..(y as usize + 1) * row_len];
            stream.push_row(row).unwrap();
            while let Some(out) = stream.next_output_row() {
                output_rows.push(out.to_vec());
            }
        }
        stream.finish();
        while let Some(out) = stream.next_output_row() {
            output_rows.push(out.to_vec());
        }

        let streaming_result: Vec<u8> = output_rows.into_iter().flatten().collect();
        assert_eq!(streaming_result.len(), result_fullframe.len());

        // The fullframe and streaming paths use different pipeline order (H-first vs V-first)
        // and different internal precision, so exact match isn't expected. But dimensions and
        // padding area should match.
        let out_w = plan.canvas.width as usize;
        let out_h = plan.canvas.height as usize;
        assert_eq!(streaming_result.len(), out_w * out_h * ch);
        assert_eq!(result_fullframe.len(), out_w * out_h * ch);

        // Padding rows (top 20) should be white in both
        let py = plan.placement.1 as usize;
        for y in 0..py {
            for x in 0..out_w {
                let off = (y * out_w + x) * ch;
                let sf = &streaming_result[off..off + ch];
                let ff = &result_fullframe[off..off + ch];
                assert_eq!(sf, ff, "padding pixel ({x},{y}) should match");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 23: streaming_from_plan with trim+pad combined
    // -----------------------------------------------------------------------
    #[test]
    fn streaming_from_plan_trim_and_pad() {
        let src_w = 80u32;
        let src_h = 60u32;
        let ch = 4usize;
        let format = PixelDescriptor::RGBA8_SRGB;
        let img = make_test_image(src_w, src_h, ch);

        // Trim center 40×30, resize to 20×15, place on 30×30 canvas with white bg
        let plan = LayoutPlan::identity(Size::new(20, 15))
            .with_trim(Rect::new(20, 15, 40, 30))
            .with_resize_to(Size::new(20, 15))
            .with_canvas(Size::new(30, 30))
            .with_placement(5, 7)
            .with_canvas_color(CanvasColor::white());

        let mut stream = streaming_from_plan(src_w, src_h, &plan, format, Filter::Lanczos)
            .expect("identity orientation");

        let row_len = src_w as usize * ch;
        let mut output_rows: Vec<Vec<u8>> = Vec::new();

        for y in 0..src_h {
            let row = &img[y as usize * row_len..(y as usize + 1) * row_len];
            stream.push_row(row).unwrap();
            while let Some(out) = stream.next_output_row() {
                output_rows.push(out.to_vec());
            }
        }
        stream.finish();
        while let Some(out) = stream.next_output_row() {
            output_rows.push(out.to_vec());
        }

        // Total output: 30×30
        assert_eq!(output_rows.len(), 30);
        assert_eq!(output_rows[0].len(), 30 * ch);

        // Top 7 rows should be white padding
        for y in 0..7 {
            for x in 0..30 {
                let px = &output_rows[y][x * ch..(x + 1) * ch];
                assert_eq!(px, &[255, 255, 255, 255], "top pad row {y}, col {x}");
            }
        }

        // Bottom 8 rows (30 - 7 - 15 = 8) should be white padding
        for y in 22..30 {
            for x in 0..30 {
                let px = &output_rows[y][x * ch..(x + 1) * ch];
                assert_eq!(px, &[255, 255, 255, 255], "bottom pad row {y}, col {x}");
            }
        }

        // Left padding (cols 0..5) in content rows should be white
        for y in 7..22 {
            for x in 0..5 {
                let px = &output_rows[y][x * ch..(x + 1) * ch];
                assert_eq!(
                    px,
                    &[255, 255, 255, 255],
                    "left pad row {y}, col {x}"
                );
            }
        }

        // Right padding (cols 25..30) in content rows should be white
        for y in 7..22 {
            for x in 25..30 {
                let px = &output_rows[y][x * ch..(x + 1) * ch];
                assert_eq!(
                    px,
                    &[255, 255, 255, 255],
                    "right pad row {y}, col {x}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 24: canvas_color_to_f32 conversion
    // -----------------------------------------------------------------------
    #[test]
    fn canvas_color_to_f32_conversions() {
        let srgb_format = PixelDescriptor::RGBA8_SRGB;
        let linear_format = PixelDescriptor::RGBAF32_LINEAR;

        // Transparent → [0, 0, 0, 0]
        assert_eq!(
            canvas_color_to_f32(&CanvasColor::Transparent, srgb_format),
            [0.0, 0.0, 0.0, 0.0]
        );

        // White sRGB → [1.0, 1.0, 1.0, 1.0]
        assert_eq!(
            canvas_color_to_f32(&CanvasColor::white(), srgb_format),
            [1.0, 1.0, 1.0, 1.0]
        );

        // Black sRGB → [0.0, 0.0, 0.0, 1.0]
        let black = CanvasColor::Srgb {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        };
        assert_eq!(
            canvas_color_to_f32(&black, srgb_format),
            [0.0, 0.0, 0.0, 1.0]
        );

        // Linear white on linear format → passthrough
        let linear_white = CanvasColor::Linear {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        assert_eq!(
            canvas_color_to_f32(&linear_white, linear_format),
            [1.0, 1.0, 1.0, 1.0]
        );

        // Linear white on sRGB format → converted to sRGB
        let result = canvas_color_to_f32(&linear_white, srgb_format);
        assert_eq!(result, [1.0, 1.0, 1.0, 1.0]); // 1.0 linear → 255 sRGB → 1.0
    }
}

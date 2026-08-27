//! Regression: the allocating `Resizer::resize()` must honor canvas padding.
//! Before the fix it sized the output buffer at inner dims (out_height *
//! output_row_len) and panicked / produced a too-small buffer when `.padding()`
//! was set. Padding worked only via `StreamingResize`. These tests pin the
//! allocating path to the streaming path's behavior.

use zenresize::{PixelDescriptor, ResizeConfig, Resizer};

fn at(out: &[u8], w: usize, x: usize, y: usize) -> [u8; 4] {
    let i = (y * w + x) * 4;
    [out[i], out[i + 1], out[i + 2], out[i + 3]]
}

#[test]
fn allocating_resize_applies_uniform_padding() {
    // 4x4 solid red -> resize to 2x2 -> pad 1px each side (green) -> 4x4 canvas.
    let red: Vec<u8> = [255u8, 0, 0, 255]
        .iter()
        .copied()
        .cycle()
        .take(4 * 4 * 4)
        .collect();
    let cfg = ResizeConfig::builder(4, 4, 2, 2)
        .format(PixelDescriptor::RGBA8_SRGB)
        .padding(1, 1, 1, 1)
        .padding_color([0.0, 1.0, 0.0, 1.0]) // green, sRGB space
        .build();

    let out = Resizer::new(&cfg).resize(&red);

    assert_eq!(out.len(), 4 * 4 * 4, "padded canvas is 4x4 RGBA8");
    // border is green padding
    assert_eq!(at(&out, 4, 0, 0), [0, 255, 0, 255], "top-left pad");
    assert_eq!(at(&out, 4, 3, 0), [0, 255, 0, 255], "top-right pad");
    assert_eq!(at(&out, 4, 0, 3), [0, 255, 0, 255], "bottom-left pad");
    assert_eq!(at(&out, 4, 3, 3), [0, 255, 0, 255], "bottom-right pad");
    // inner 2x2 is the (solid red) content
    assert_eq!(at(&out, 4, 1, 1), [255, 0, 0, 255], "content");
    assert_eq!(at(&out, 4, 2, 2), [255, 0, 0, 255], "content");
}

#[test]
fn allocating_resize_asymmetric_padding_dims() {
    // 10x10 -> 4x4, pad top=2 right=3 bottom=1 left=0 -> (0+4+3) x (2+4+1) = 7x7.
    let src: Vec<u8> = [128u8, 64, 32, 255]
        .iter()
        .copied()
        .cycle()
        .take(10 * 10 * 4)
        .collect();
    let cfg = ResizeConfig::builder(10, 10, 4, 4)
        .format(PixelDescriptor::RGBA8_SRGB)
        .padding(2, 3, 1, 0)
        .padding_color([0.0, 0.0, 0.0, 0.0])
        .build();

    let out = Resizer::new(&cfg).resize(&src);
    assert_eq!(
        out.len(),
        7 * 7 * 4,
        "canvas = (left+ow+right) x (top+oh+bottom)"
    );
}

#[test]
fn allocating_resize_no_padding_unchanged() {
    // Sanity: without padding, the allocating path is unaffected by the fix.
    let src: Vec<u8> = [10u8, 20, 30, 255]
        .iter()
        .copied()
        .cycle()
        .take(8 * 8 * 4)
        .collect();
    let cfg = ResizeConfig::builder(8, 8, 4, 4)
        .format(PixelDescriptor::RGBA8_SRGB)
        .build();
    let out = Resizer::new(&cfg).resize(&src);
    assert_eq!(out.len(), 4 * 4 * 4);
    assert_eq!(
        at(&out, 4, 0, 0),
        [10, 20, 30, 255],
        "solid color preserved"
    );
}

/// Regression for #10 (part 3): padding can inflate the canvas past the
/// output-pixel cap and past `usize`. `try_new` must surface the validation
/// error; before the fix the config validated and the allocating `resize()`
/// computed `total_output_height as usize * total_output_row_len` unchecked.
#[test]
fn try_new_rejects_padded_canvas_that_overflows_usize() {
    let cfg = ResizeConfig::builder(1, 1, 1, 1)
        .format(PixelDescriptor::RGBA8_SRGB)
        .padding(1 << 31, 0, 0, 1 << 31)
        .max_output_pixels(None)
        .build();
    assert!(
        Resizer::try_new(&cfg).is_err(),
        "a padded canvas whose byte size overflows usize must be rejected"
    );
}

#[test]
fn try_new_rejects_padded_canvas_over_pixel_cap() {
    // Inner 1 MP passes the cap on its own; the 400 MP padded canvas must not.
    let cfg = ResizeConfig::builder(100, 100, 1000, 1000)
        .format(PixelDescriptor::RGBA8_SRGB)
        .padding(9500, 9500, 9500, 9500)
        .build();
    assert!(
        Resizer::try_new(&cfg).is_err(),
        "the output-pixel cap must cover the padded canvas"
    );
}

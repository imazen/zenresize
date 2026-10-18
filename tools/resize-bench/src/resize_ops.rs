//! Whole public resize calls, instantiated once per zenresize copy
//! (`old_ops` = baseline, `new_ops` = candidate), plus deterministic input
//! derivation. Everything here runs inside or around the timed callback
//! exactly as a caller would write it: configuration is built once, then each
//! call constructs the resizer, allocates its output and returns it.

use crate::plan::{Engine, Format};

/// A typed pixel buffer (a group input or a resize output).
pub(crate) enum Buf {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl Buf {
    /// Canonical little-endian bytes, used for hashing and the artifact store.
    pub(crate) fn le_bytes(&self) -> Vec<u8> {
        match self {
            Buf::U8(v) => v.clone(),
            Buf::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Buf::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Buf::U8(_) => "u8",
            Buf::U16(_) => "u16-le",
            Buf::F32(_) => "f32-le",
        }
    }

    /// Exact equality: integers by value, floats by bit pattern.
    pub(crate) fn bit_equal(&self, other: &Buf) -> bool {
        match (self, other) {
            (Buf::U8(a), Buf::U8(b)) => a == b,
            (Buf::U16(a), Buf::U16(b)) => a == b,
            (Buf::F32(a), Buf::F32(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            _ => false,
        }
    }
}

/// Derive a group's input from the cropped RGB8 rows (outside timing).
/// RGB16 = v * 257, RGBF32 = v / 255, 4-channel formats append 255. Stride
/// padding elements are 0xA5 / 0xA5A5 / NaN so reading them would show.
pub(crate) fn derive_input(format: Format, w: usize, h: usize, pad: usize, rgb8: &[u8]) -> Buf {
    assert_eq!(rgb8.len(), w * h * 3, "raw RGB8 size");
    let ch = format.channels();
    let row = w * ch;
    let px = |i: usize| &rgb8[i * 3..i * 3 + 3];
    match format {
        Format::Rgb8Linear
        | Format::Rgb8Srgb
        | Format::Rgba8Srgb
        | Format::Rgbx8Linear
        | Format::Rgba8Linear => {
            let mut v = Vec::with_capacity((row + pad) * h);
            for y in 0..h {
                for x in 0..w {
                    v.extend_from_slice(px(y * w + x));
                    if ch == 4 {
                        v.push(255);
                    }
                }
                v.extend(std::iter::repeat_n(0xA5u8, pad));
            }
            Buf::U8(v)
        }
        Format::Rgb16Linear => {
            let mut v = Vec::with_capacity((row + pad) * h);
            for y in 0..h {
                for x in 0..w {
                    v.extend(px(y * w + x).iter().map(|&c| c as u16 * 257));
                }
                v.extend(std::iter::repeat_n(0xA5A5u16, pad));
            }
            Buf::U16(v)
        }
        Format::RgbF32Linear => Buf::F32(f32_rows(rgb8, w, h, 3, pad)),
    }
}

/// Raw-f32 twin input (RGB or RGBA f32, v / 255, packed) used to check old/new
/// parity in f32 output too, not only after u8/u16 quantization. None where
/// the group already is f32 or the engine has no f32 entry point (h-first).
pub(crate) fn derive_f32_twin(
    format: Format,
    engine: Engine,
    w: usize,
    h: usize,
    rgb8: &[u8],
) -> Option<Buf> {
    if engine == Engine::Hfirst || format == Format::RgbF32Linear {
        return None;
    }
    Some(Buf::F32(f32_rows(rgb8, w, h, format.channels(), 0)))
}

fn f32_rows(rgb8: &[u8], w: usize, h: usize, ch: usize, pad: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity((w * ch + pad) * h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            v.extend(rgb8[i..i + 3].iter().map(|&c| c as f32 / 255.0));
            if ch == 4 {
                v.push(1.0);
            }
        }
        v.extend(std::iter::repeat_n(f32::NAN, pad));
    }
    v
}

macro_rules! resize_ops {
    ($module:ident, $zr:ident) => {
        pub(crate) mod $module {
            use super::Buf;
            use crate::plan::{Engine, Format};
            use $zr as z;

            /// The configuration a caller builds once for this group.
            pub(crate) fn config(
                format: Format,
                in_w: u32,
                in_h: u32,
                out_w: u32,
                out_h: u32,
                pad: usize,
            ) -> z::ResizeConfig {
                let desc = match format {
                    Format::Rgb8Linear | Format::Rgb8Srgb => z::PixelDescriptor::RGB8_SRGB,
                    Format::Rgb16Linear => z::PixelDescriptor::RGB16_SRGB,
                    Format::RgbF32Linear => z::PixelDescriptor::RGBF32_LINEAR,
                    Format::Rgba8Srgb | Format::Rgba8Linear => z::PixelDescriptor::RGBA8_SRGB,
                    Format::Rgbx8Linear => z::PixelDescriptor::RGBX8_SRGB,
                };
                let mut b = z::ResizeConfig::builder(in_w, in_h, out_w, out_h)
                    .filter(z::Filter::Lanczos)
                    .format(desc);
                b = match format {
                    Format::Rgb8Srgb | Format::Rgba8Srgb => b.srgb(),
                    _ => b.linear(),
                };
                if pad > 0 {
                    b = b.in_stride(in_w as usize * format.channels() + pad);
                }
                b.build()
            }

            /// RGB(A)F32_LINEAR configuration of the same geometry (parity twin).
            pub(crate) fn twin_config(
                channels: usize,
                in_w: u32,
                in_h: u32,
                out_w: u32,
                out_h: u32,
            ) -> z::ResizeConfig {
                let desc = if channels == 4 {
                    z::PixelDescriptor::RGBAF32_LINEAR
                } else {
                    z::PixelDescriptor::RGBF32_LINEAR
                };
                z::ResizeConfig::builder(in_w, in_h, out_w, out_h)
                    .filter(z::Filter::Lanczos)
                    .format(desc)
                    .linear()
                    .build()
            }

            /// One whole public resize: construction, every row, the output
            /// allocation; the returned buffer is dropped by the caller.
            pub(crate) fn run(engine: Engine, cfg: &z::ResizeConfig, input: &Buf) -> Buf {
                let stride = cfg.effective_in_stride();
                let row = cfg.input_row_len();
                let in_h = cfg.in_height as usize;
                match (engine, input) {
                    (Engine::Fullframe, Buf::U8(i)) => Buf::U8(z::Resizer::new(cfg).resize(i)),
                    (Engine::Fullframe, Buf::U16(i)) => {
                        Buf::U16(z::Resizer::new(cfg).resize_u16(i))
                    }
                    (Engine::Fullframe, Buf::F32(i)) => {
                        Buf::F32(z::Resizer::new(cfg).resize_f32(i))
                    }
                    (Engine::Stream, Buf::U8(i)) => {
                        let mut s = z::StreamingResize::new(cfg);
                        let mut out = Vec::with_capacity(
                            cfg.out_width as usize * cfg.out_height as usize * cfg.channels(),
                        );
                        for y in 0..in_h {
                            s.push_row(&i[y * stride..y * stride + row])
                                .expect("push_row");
                            while let Some(r) = s.next_output_row() {
                                out.extend_from_slice(r);
                            }
                        }
                        s.finish();
                        while let Some(r) = s.next_output_row() {
                            out.extend_from_slice(r);
                        }
                        Buf::U8(out)
                    }
                    (Engine::Stream, Buf::U16(i)) => {
                        let mut s = z::StreamingResize::new(cfg);
                        let mut out = Vec::with_capacity(
                            cfg.out_width as usize * cfg.out_height as usize * cfg.channels(),
                        );
                        for y in 0..in_h {
                            s.push_row_u16(&i[y * stride..y * stride + row])
                                .expect("push_row_u16");
                            while let Some(r) = s.next_output_row_u16() {
                                out.extend_from_slice(r);
                            }
                        }
                        s.finish();
                        while let Some(r) = s.next_output_row_u16() {
                            out.extend_from_slice(r);
                        }
                        Buf::U16(out)
                    }
                    (Engine::Stream, Buf::F32(i)) => {
                        let mut s = z::StreamingResize::new(cfg);
                        let mut out = Vec::with_capacity(
                            cfg.out_width as usize * cfg.out_height as usize * cfg.channels(),
                        );
                        for y in 0..in_h {
                            s.push_row_f32(&i[y * stride..y * stride + row])
                                .expect("push_row_f32");
                            while let Some(r) = s.next_output_row_f32() {
                                out.extend_from_slice(r);
                            }
                        }
                        s.finish();
                        while let Some(r) = s.next_output_row_f32() {
                            out.extend_from_slice(r);
                        }
                        Buf::F32(out)
                    }
                    (Engine::Hfirst, Buf::U8(i)) => Buf::U8(if cfg.channels() == 4 {
                        z::resize_hfirst_streaming(cfg, i).expect("resize_hfirst_streaming")
                    } else {
                        z::resize_hfirst_streaming_f32(cfg, i).expect("resize_hfirst_streaming_f32")
                    }),
                    (engine, buf) => panic!(
                        "plan error: engine {engine:?} has no {} entry point",
                        buf.kind()
                    ),
                }
            }
        }
    };
}

resize_ops!(old_ops, zr_old);
resize_ops!(new_ops, zr_new);

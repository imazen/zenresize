//! Independent dense image oracle for every sample-type conversion and working path.
use zenresize::{
    ChannelType, Filter, PixelDescriptor, ResizeConfig, Resizer, StreamingResize, WorkingFormat,
};

fn kernel(x: f64, newer: bool) -> f64 {
    let spline = |x: f64| {
        let t = x.abs();
        if t < 0.5 {
            0.75 - t * t
        } else if t < 1.5 {
            (1.5 - t).powi(2) / 2.0
        } else {
            0.0
        }
    };
    if newer {
        [-1.0, 6.0, -35.0, 204.0, -35.0, 6.0, -1.0]
            .iter()
            .enumerate()
            .map(|(i, w)| w * spline(x + i as f64 - 3.0) / 144.0)
            .sum()
    } else {
        (6.0 * spline(x) - spline(x - 1.0) - spline(x + 1.0)) / 4.0
    }
}
fn weights(n: usize, out: usize, at: usize, newer: bool) -> Vec<f64> {
    let scale = out as f64 / n as f64;
    let center = (at as f64 + 0.5) / scale - 0.5;
    let mut w: Vec<_> = (0..n)
        .map(|i| kernel((i as f64 - center) * scale.min(1.0), newer))
        .collect();
    let sum: f64 = w.iter().sum();
    for v in &mut w {
        *v /= sum;
    }
    w
}
fn linear(x: f64) -> f64 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
fn srgb(x: f64) -> f64 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}
fn run(
    input_fmt: PixelDescriptor,
    output_fmt: PixelDescriptor,
    channels: usize,
    integer_linear: bool,
    ow: usize,
    oh: usize,
    newer: bool,
) {
    let (w, h) = (13, 9);
    let stride = w * channels + 7;
    let config_builder = ResizeConfig::builder(w as u32, h as u32, ow as u32, oh as u32)
        .input(input_fmt)
        .output(output_fmt)
        .in_stride(stride)
        .filter(if newer {
            Filter::MagicKernelSharp2021
        } else {
            Filter::MagicKernelSharp
        });
    let config = if integer_linear {
        config_builder.linear().build()
    } else {
        config_builder.srgb().build()
    };
    let mut stream = StreamingResize::new(&config);
    let expected_path = if integer_linear && newer && ow == 7 {
        // Existing >14-tap precision guard selects f32 for this reduction.
        WorkingFormat::F32
    } else if integer_linear {
        WorkingFormat::I16Linear
    } else if input_fmt.channel_type() == ChannelType::U8
        && output_fmt.channel_type() == ChannelType::U8
        && channels == 4
    {
        WorkingFormat::I16Srgb
    } else {
        WorkingFormat::F32
    };
    assert_eq!(stream.working_format(), expected_path);
    // Padded rows use a sentinel; the reference only sees actual pixels.
    let mut bytes = vec![231u8; stride * h];
    let mut words = vec![59367u16; stride * h];
    let mut floats = vec![9.0f32; stride * h];
    let mut source = vec![0.0; w * h * channels];
    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let v = if c == 3 {
                    if input_fmt.alpha == Some(zenresize::AlphaMode::Straight) {
                        128 + ((x * 17 + y * 11) % 128) as u8
                    } else {
                        255
                    }
                } else {
                    64 + ((x * 19 + y * 23 + c * 31) % 128) as u8
                };
                let offset = y * stride + x * channels + c;
                bytes[offset] = v;
                words[offset] = v as u16 * 257;
                floats[offset] = v as f32 / 255.0;
                let value = match input_fmt.channel_type() {
                    ChannelType::F32 => floats[offset] as f64,
                    _ => v as f64 / 255.0,
                };
                source[(y * w + x) * channels + c] = if config.effective_input_transfer()
                    == zenresize::TransferFunction::Srgb
                    && c < 3
                {
                    linear(value)
                } else {
                    value
                };
            }
        }
    }
    if config.needs_premultiply() {
        for pixel in source.chunks_exact_mut(channels) {
            let alpha = pixel[3];
            for v in &mut pixel[..3] {
                *v *= alpha;
            }
        }
    }
    let mut resizer = Resizer::new(&config);
    let full: Vec<f64> = match (input_fmt.channel_type(), output_fmt.channel_type()) {
        (ChannelType::U8, ChannelType::U8) => resizer
            .resize(&bytes)
            .into_iter()
            .map(|v| v as f64 / 255.0)
            .collect(),
        (ChannelType::U8, ChannelType::U16) => resizer
            .resize_u8_to_u16(&bytes)
            .into_iter()
            .map(|v| v as f64 / 65535.0)
            .collect(),
        (ChannelType::U8, _) => resizer
            .resize_u8_to_f32(&bytes)
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        (ChannelType::U16, ChannelType::U8) => resizer
            .resize_u16_to_u8(&words)
            .into_iter()
            .map(|v| v as f64 / 255.0)
            .collect(),
        (ChannelType::U16, ChannelType::U16) => resizer
            .resize_u16(&words)
            .into_iter()
            .map(|v| v as f64 / 65535.0)
            .collect(),
        (ChannelType::U16, _) => resizer
            .resize_u16_to_f32(&words)
            .into_iter()
            .map(|v| v as f64)
            .collect(),
        (_, ChannelType::U8) => resizer
            .resize_f32_to_u8(&floats)
            .into_iter()
            .map(|v| v as f64 / 255.0)
            .collect(),
        (_, ChannelType::U16) => resizer
            .resize_f32_to_u16(&floats)
            .into_iter()
            .map(|v| v as f64 / 65535.0)
            .collect(),
        _ => resizer
            .resize_f32(&floats)
            .into_iter()
            .map(|v| v as f64)
            .collect(),
    };
    let drain =
        |stream: &mut StreamingResize, actual: &mut Vec<f64>| match output_fmt.channel_type() {
            ChannelType::U8 => {
                while let Some(row) = stream.next_output_row() {
                    actual.extend(row.iter().map(|&v| v as f64 / 255.0));
                }
            }
            ChannelType::U16 => {
                while let Some(row) = stream.next_output_row_u16() {
                    actual.extend(row.iter().map(|&v| v as f64 / 65535.0));
                }
            }
            _ => {
                while let Some(row) = stream.next_output_row_f32() {
                    actual.extend(row.iter().map(|&v| v as f64));
                }
            }
        };
    let mut actual = Vec::new();
    for y in 0..h {
        let range = y * stride..y * stride + w * channels;
        match input_fmt.channel_type() {
            ChannelType::U8 => stream.push_row(&bytes[range]).unwrap(),
            ChannelType::U16 => stream.push_row_u16(&words[range]).unwrap(),
            _ => stream.push_row_f32(&floats[range]).unwrap(),
        }
        drain(&mut stream, &mut actual);
    }
    stream.finish();
    drain(&mut stream, &mut actual);
    assert_eq!(full, actual, "fullframe/streaming parity");
    assert_eq!(actual.len(), ow * oh * channels);
    // Integer stages round premultiplied samples and intermediate rows.
    // Alpha >= 0.5 can amplify these errors by two on unpremultiply: allow
    // five u8 codes for straight alpha, two for opaque. Floating paths store
    // input in f16: 0.001 absolute allowance.
    let tolerance = if expected_path != WorkingFormat::F32 {
        if config.needs_premultiply() {
            5.0 / 255.0
        } else {
            2.0 / 255.0
        }
    } else if output_fmt.channel_type() == ChannelType::U8 {
        1.0 / 255.0
    } else {
        0.001
    };
    for y in 0..oh {
        let wy = weights(h, oh, y, newer);
        for x in 0..ow {
            let wx = weights(w, ow, x, newer);
            for c in 0..channels {
                let mut value = 0.0;
                for iy in 0..h {
                    for ix in 0..w {
                        value += wy[iy] * wx[ix] * source[(iy * w + ix) * channels + c];
                    }
                }
                if config.needs_premultiply() && c < 3 {
                    let mut alpha = 0.0;
                    for iy in 0..h {
                        for ix in 0..w {
                            alpha += wy[iy] * wx[ix] * source[(iy * w + ix) * channels + 3];
                        }
                    }
                    if alpha > 1.0 / 1024.0 {
                        value /= alpha;
                    }
                }
                if config.effective_output_transfer() == zenresize::TransferFunction::Srgb && c < 3
                {
                    value = srgb(value);
                }
                if output_fmt.channel_type() != ChannelType::F32 {
                    value = value.clamp(0.0, 1.0);
                }
                let got = actual[(y * ow + x) * channels + c];
                assert!(
                    (value - got).abs() <= tolerance,
                    "newer={newer} {input_fmt:?}->{output_fmt:?} {expected_path:?} {ow}x{oh} ({x},{y},{c}): {got} vs {value}"
                );
            }
        }
    }
}

#[test]
fn all_sample_pairs_and_working_paths_match_independent_oracle() {
    // This binary contains one test; token overrides cannot race another test.
    if std::env::var_os("ZENRESIZE_MAGIC_SCALAR").is_some() {
        #[cfg(target_arch = "x86_64")]
        archmage::X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
        #[cfg(target_arch = "aarch64")]
        archmage::NeonToken::dangerously_disable_token_process_wide(true).unwrap();
    }
    let formats = [
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGBA16_SRGB,
        PixelDescriptor::RGBAF32_LINEAR,
    ];
    for newer in [false, true] {
        for (ow, oh) in [(23, 15), (11, 8), (7, 5)] {
            for input in formats {
                for output in formats {
                    run(input, output, 4, false, ow, oh, newer);
                }
            }
            run(
                PixelDescriptor::RGBX8_SRGB,
                PixelDescriptor::RGBX8_SRGB,
                4,
                true,
                ow,
                oh,
                newer,
            );
            run(
                PixelDescriptor::RGB8_SRGB,
                PixelDescriptor::RGB8_SRGB,
                3,
                false,
                ow,
                oh,
                newer,
            );
        }
    }
}

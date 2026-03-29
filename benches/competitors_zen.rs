//! Cross-library resize benchmark using zenbench interleaved measurement.
//!
//! Resource-gated, interleaved, paired comparison — reliable results on busy systems.
//! Compares zenresize against pic-scale, pic-scale-safe, fast_image_resize, and resize crate.
//!
//! Usage:
//!   cargo bench --bench competitors_zen --release
//!   cargo bench --bench competitors_zen --release --features bench-simd-competitors

// ---------------------------------------------------------------------------
// Image generation
// ---------------------------------------------------------------------------

fn make_gradient(w: u32, h: u32, channels: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w as usize * h as usize * channels];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * channels;
            buf[i] = (x % 256) as u8;
            if channels >= 2 {
                buf[i + 1] = (y % 256) as u8;
            }
            if channels >= 3 {
                buf[i + 2] = ((x + y) % 256) as u8;
            }
            if channels >= 4 {
                buf[i + 3] = 255;
            }
        }
    }
    buf
}

fn make_f32(u8_data: &[u8]) -> Vec<f32> {
    u8_data.iter().map(|&b| b as f32 / 255.0).collect()
}

// ---------------------------------------------------------------------------
// zenresize wrappers
// ---------------------------------------------------------------------------

fn zen_srgb(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(iw, ih, ow, oh)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();
    zenresize::Resizer::new(&config).resize(src)
}

fn zen_srgb_i16(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(iw, ih, ow, oh)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelDescriptor::RGBX8_SRGB)
        .srgb()
        .build();
    zenresize::Resizer::new(&config).resize(src)
}

fn zen_linear(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(iw, ih, ow, oh)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelDescriptor::RGBA8_SRGB)
        .linear()
        .build();
    zenresize::Resizer::new(&config).resize(src)
}

fn zen_linear_i16(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(iw, ih, ow, oh)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelDescriptor::RGBX8_SRGB)
        .linear()
        .build();
    zenresize::Resizer::new(&config).resize(src)
}

fn zen_f32(src_f32: &[f32], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<f32> {
    let config = zenresize::ResizeConfig::builder(iw, ih, ow, oh)
        .filter(zenresize::Filter::Lanczos)
        .format(
            zenresize::PixelDescriptor::RGBAF32_LINEAR
                .with_alpha(Some(zenresize::AlphaMode::Undefined)),
        )
        .build();
    zenresize::Resizer::new(&config).resize_f32(src_f32)
}

fn zen_stream_srgb(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(iw, ih, ow, oh)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();
    let row_len = iw as usize * 4;
    let mut resizer = zenresize::StreamingResize::new(&config);
    let mut output = Vec::with_capacity(ow as usize * oh as usize * 4);
    for y in 0..ih as usize {
        resizer
            .push_row(&src[y * row_len..(y + 1) * row_len])
            .unwrap();
        while let Some(row) = resizer.next_output_row() {
            output.extend_from_slice(row);
        }
    }
    resizer.finish();
    while let Some(row) = resizer.next_output_row() {
        output.extend_from_slice(row);
    }
    output
}

// ---------------------------------------------------------------------------
// pic-scale wrappers
// ---------------------------------------------------------------------------

fn ps_srgb(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    use pic_scale::*;
    let mut scaler = Scaler::new(ResamplingFunction::Lanczos3);
    scaler.set_threading_policy(ThreadingPolicy::Single);
    let src_size = ImageSize::new(iw as usize, ih as usize);
    let dst_size = ImageSize::new(ow as usize, oh as usize);
    let store = ImageStore::<u8, 4>::from_slice(src, iw as usize, ih as usize).unwrap();
    let mut dst = ImageStoreMut::<u8, 4>::alloc(ow as usize, oh as usize);
    let plan = scaler.plan_rgba_resampling(src_size, dst_size, true).unwrap();
    plan.resample(&store, &mut dst).unwrap();
    dst.as_bytes().to_vec()
}

fn ps_linear(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    use pic_scale::*;
    let mut scaler = LinearScaler::new(ResamplingFunction::Lanczos3);
    scaler.set_threading_policy(ThreadingPolicy::Single);
    let src_size = ImageSize::new(iw as usize, ih as usize);
    let dst_size = ImageSize::new(ow as usize, oh as usize);
    let store = ImageStore::<u8, 4>::from_slice(src, iw as usize, ih as usize).unwrap();
    let mut dst = ImageStoreMut::<u8, 4>::alloc(ow as usize, oh as usize);
    let plan = scaler.plan_rgba_resampling(src_size, dst_size, true).unwrap();
    plan.resample(&store, &mut dst).unwrap();
    dst.as_bytes().to_vec()
}

// ---------------------------------------------------------------------------
// pic-scale-safe wrappers
// ---------------------------------------------------------------------------

fn pss_srgb(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    use pic_scale_safe::*;
    let src_size = ImageSize::new(iw as usize, ih as usize);
    let dst_size = ImageSize::new(ow as usize, oh as usize);
    let mut buf = src.to_vec();
    image_to_linear::<4>(&mut buf, TransferFunction::Srgb);
    let mut result = resize_rgba8(&buf, src_size, dst_size, ResamplingFunction::Lanczos3).unwrap();
    linear_to_gamma_image::<4>(&mut result, TransferFunction::Srgb);
    result
}

// ---------------------------------------------------------------------------
// fast_image_resize wrappers
// ---------------------------------------------------------------------------

fn fir_srgb(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    use fast_image_resize as fir;
    use fir::images::{Image, ImageRef};
    use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
    let src_img = ImageRef::new(iw, ih, src, PixelType::U8x4).unwrap();
    let mut dst = Image::new(ow, oh, PixelType::U8x4);
    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer.resize(&src_img, &mut dst, &opts).unwrap();
    dst.into_vec()
}

fn fir_linear(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    use fast_image_resize as fir;
    use fir::images::{Image, ImageRef};
    use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
    let mapper = fir::create_srgb_mapper();
    let src_img = ImageRef::new(iw, ih, src, PixelType::U8x4).unwrap();
    let mut linear_src = Image::new(iw, ih, PixelType::U16x4);
    mapper.forward_map(&src_img, &mut linear_src).unwrap();
    let mut linear_dst = Image::new(ow, oh, PixelType::U16x4);
    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer.resize(&linear_src, &mut linear_dst, &opts).unwrap();
    let mut dst = Image::new(ow, oh, PixelType::U8x4);
    mapper.backward_map(&linear_dst, &mut dst).unwrap();
    dst.into_vec()
}

fn fir_f32(src_f32: &[f32], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<f32> {
    use fast_image_resize as fir;
    use fir::images::{Image, ImageRef};
    use fir::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(src_f32.as_ptr() as *const u8, src_f32.len() * 4) };
    let src_img = ImageRef::new(iw, ih, f32_bytes, PixelType::F32x4).unwrap();
    let mut dst = Image::new(ow, oh, PixelType::F32x4);
    let mut resizer = Resizer::new();
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3));
    resizer.resize(&src_img, &mut dst, &opts).unwrap();
    let dst_bytes = dst.into_vec();
    let dst_f32: Vec<f32> = dst_bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    dst_f32
}

// ---------------------------------------------------------------------------
// resize crate wrapper
// ---------------------------------------------------------------------------

fn resize_crate(src: &[u8], iw: u32, ih: u32, ow: u32, oh: u32) -> Vec<u8> {
    use rgb::FromSlice;
    let mut dst = vec![0u8; ow as usize * oh as usize * 4];
    let mut resizer = resize::new(
        iw as usize,
        ih as usize,
        ow as usize,
        oh as usize,
        resize::Pixel::RGBA8P,
        resize::Type::Lanczos3,
    )
    .unwrap();
    resizer.resize(src.as_rgba(), dst.as_rgba_mut()).unwrap();
    dst
}

// ---------------------------------------------------------------------------
// Scenario definitions
// ---------------------------------------------------------------------------

struct Scenario {
    label: &'static str,
    iw: u32,
    ih: u32,
    ow: u32,
    oh: u32,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            label: "1024_2x_down",
            iw: 1024,
            ih: 1024,
            ow: 512,
            oh: 512,
        },
        Scenario {
            label: "4k_2x_down",
            iw: 3840,
            ih: 2160,
            ow: 1920,
            oh: 1080,
        },
        Scenario {
            label: "4k_10x_down",
            iw: 4000,
            ih: 3000,
            ow: 400,
            oh: 300,
        },
        Scenario {
            label: "576_2x_up",
            iw: 576,
            ih: 576,
            ow: 1152,
            oh: 1152,
        },
    ]
}

// ---------------------------------------------------------------------------
// zenbench harness
// ---------------------------------------------------------------------------

zenbench::main!(|suite| {
    let ps_label = if cfg!(feature = "bench-simd-competitors") {
        "ps_simd"
    } else {
        "ps_scalar"
    };

    for s in &scenarios() {
        let (iw, ih, ow, oh) = (s.iw, s.ih, s.ow, s.oh);
        let rgba = make_gradient(iw, ih, 4);
        let rgba_f32 = make_f32(&rgba);
        let in_bytes = rgba.len() as u64;

        // --- sRGB comparison ---
        {
            let group_name = format!("srgb_{}", s.label);
            let rgba = rgba.clone();
            let rgba2 = rgba.clone();
            let rgba3 = rgba.clone();
            let rgba4 = rgba.clone();
            let rgba5 = rgba.clone();
            let rgba6 = rgba.clone();
            let rgba7 = rgba.clone();

            suite.compare(&group_name, |group| {
                group.throughput(zenbench::Throughput::Bytes(in_bytes));

                group.bench("zenresize_srgb", move |b| {
                    let src = rgba.clone();
                    b.iter(|| zen_srgb(&src, iw, ih, ow, oh));
                });

                group.bench("zenresize_srgb_i16", move |b| {
                    let src = rgba2.clone();
                    b.iter(|| zen_srgb_i16(&src, iw, ih, ow, oh));
                });

                group.bench("zen_stream_srgb", move |b| {
                    let src = rgba3.clone();
                    b.iter(|| zen_stream_srgb(&src, iw, ih, ow, oh));
                });

                {
                    let ps_name = format!("{ps_label}_srgb");
                    group.bench(&ps_name, move |b| {
                        let src = rgba4.clone();
                        b.iter(|| ps_srgb(&src, iw, ih, ow, oh));
                    });
                }

                group.bench("pss_srgb", move |b| {
                    let src = rgba5.clone();
                    b.iter(|| pss_srgb(&src, iw, ih, ow, oh));
                });

                group.bench("fir_srgb", move |b| {
                    let src = rgba6.clone();
                    b.iter(|| fir_srgb(&src, iw, ih, ow, oh));
                });

                group.bench("resize_crate", move |b| {
                    let src = rgba7.clone();
                    b.iter(|| resize_crate(&src, iw, ih, ow, oh));
                });
            });
        }

        // --- linear comparison ---
        {
            let group_name = format!("linear_{}", s.label);
            let rgba = rgba.clone();
            let rgba2 = rgba.clone();
            let rgba3 = rgba.clone();
            let rgba4 = rgba.clone();

            suite.compare(&group_name, |group| {
                group.throughput(zenbench::Throughput::Bytes(in_bytes));

                group.bench("zenresize_linear", move |b| {
                    let src = rgba.clone();
                    b.iter(|| zen_linear(&src, iw, ih, ow, oh));
                });

                group.bench("zenresize_linear_i16", move |b| {
                    let src = rgba2.clone();
                    b.iter(|| zen_linear_i16(&src, iw, ih, ow, oh));
                });

                {
                    let ps_name = format!("{ps_label}_linear");
                    group.bench(&ps_name, move |b| {
                        let src = rgba3.clone();
                        b.iter(|| ps_linear(&src, iw, ih, ow, oh));
                    });
                }

                group.bench("fir_linear", move |b| {
                    let src = rgba4.clone();
                    b.iter(|| fir_linear(&src, iw, ih, ow, oh));
                });
            });
        }

        // --- f32 comparison ---
        {
            let group_name = format!("f32_{}", s.label);
            let f32_data = rgba_f32.clone();
            let f32_data2 = rgba_f32.clone();

            suite.compare(&group_name, |group| {
                group.throughput(zenbench::Throughput::Bytes(in_bytes * 4)); // f32 = 4x bytes

                group.bench("zenresize_f32", move |b| {
                    let src = f32_data.clone();
                    b.iter(|| zen_f32(&src, iw, ih, ow, oh));
                });

                group.bench("fir_f32", move |b| {
                    let src = f32_data2.clone();
                    b.iter(|| fir_f32(&src, iw, ih, ow, oh));
                });
            });
        }
    }
});

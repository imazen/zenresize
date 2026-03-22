//! Fullframe vs streaming benchmark using zenbench interleaved measurement.
//!
//! Resource-gated, interleaved, paired comparison — reliable results on busy systems.

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

fn fullframe_resize(config: &zenresize::ResizeConfig, input: &[u8]) -> Vec<u8> {
    zenresize::Resizer::new(config).resize(input)
}

fn streaming_resize(config: &zenresize::ResizeConfig, input: &[u8]) -> Vec<u8> {
    let in_w = config.in_width as usize;
    let channels = config.input.channels();
    let row_len = in_w * channels;
    let in_h = config.in_height as usize;
    let mut resizer = zenresize::StreamingResize::new(config);
    let mut output =
        Vec::with_capacity(config.out_width as usize * config.out_height as usize * channels);
    for y in 0..in_h {
        resizer
            .push_row(&input[y * row_len..(y + 1) * row_len])
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

fn hfirst_resize(config: &zenresize::ResizeConfig, input: &[u8]) -> Vec<u8> {
    let ch = config.input.channels();
    if ch == 4 {
        zenresize::resize_hfirst_streaming(config, input)
    } else {
        zenresize::resize_hfirst_streaming_f32(config, input)
    }
}

#[allow(dead_code)]
struct Scenario {
    label: &'static str,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    config: zenresize::ResizeConfig,
    input: Vec<u8>,
}

fn scenarios() -> Vec<Scenario> {
    let mut out = Vec::new();

    let mut add = |label,
                   in_w,
                   in_h,
                   out_w,
                   out_h,
                   build: fn(u32, u32, u32, u32) -> zenresize::ResizeConfig| {
        let config = build(in_w, in_h, out_w, out_h);
        let channels = config.input.channels();
        let input = make_gradient(in_w, in_h, channels);
        out.push(Scenario {
            label,
            in_w,
            in_h,
            out_w,
            out_h,
            config,
            input,
        });
    };

    fn srgb4(iw: u32, ih: u32, ow: u32, oh: u32) -> zenresize::ResizeConfig {
        zenresize::ResizeConfig::builder(iw, ih, ow, oh)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build()
    }
    fn linear4(iw: u32, ih: u32, ow: u32, oh: u32) -> zenresize::ResizeConfig {
        zenresize::ResizeConfig::builder(iw, ih, ow, oh)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGBX8_SRGB)
            .linear()
            .build()
    }
    fn f32_3ch(iw: u32, ih: u32, ow: u32, oh: u32) -> zenresize::ResizeConfig {
        zenresize::ResizeConfig::builder(iw, ih, ow, oh)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGB8_SRGB)
            .linear()
            .build()
    }
    fn alpha4(iw: u32, ih: u32, ow: u32, oh: u32) -> zenresize::ResizeConfig {
        zenresize::ResizeConfig::builder(iw, ih, ow, oh)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build()
    }

    add("srgb4_1024_2x", 1024, 1024, 512, 512, srgb4);
    add("linear4_1024_2x", 1024, 1024, 512, 512, linear4);
    add("f32_3ch_1024_2x", 1024, 1024, 512, 512, f32_3ch);
    add("srgb4_4k_2x", 3840, 2160, 1920, 1080, srgb4);
    add("linear4_4k_2x", 3840, 2160, 1920, 1080, linear4);
    add("f32_3ch_4k_2x", 3840, 2160, 1920, 1080, f32_3ch);
    add("srgb4_4k_10x", 4000, 3000, 400, 300, srgb4);
    add("linear4_4k_10x", 4000, 3000, 400, 300, linear4);
    add("f32_3ch_4k_10x", 4000, 3000, 400, 300, f32_3ch);
    add("srgb4_512_up2x", 512, 512, 1024, 1024, srgb4);
    add("alpha4_1024_2x", 1024, 1024, 512, 512, alpha4);
    add("alpha4_4k_2x", 3840, 2160, 1920, 1080, alpha4);

    out
}

zenbench::main!(|suite| {
    let all = scenarios();

    for s in &all {
        let config = s.config.clone();
        let input = s.input.clone();
        let label = s.label;
        let in_bytes = input.len() as u64;

        suite.compare(label, |group| {
            group.throughput(zenbench::Throughput::Bytes(in_bytes));

            {
                let config = config.clone();
                let input = input.clone();
                group.bench("fullframe", move |b| {
                    let c = config.clone();
                    let i = input.clone();
                    b.iter(|| fullframe_resize(&c, &i));
                });
            }

            {
                let config = config.clone();
                let input = input.clone();
                group.bench("v-first-stream", move |b| {
                    let c = config.clone();
                    let i = input.clone();
                    b.iter(|| streaming_resize(&c, &i));
                });
            }

            {
                let config = config.clone();
                let input = input.clone();
                group.bench("h-first-stream", move |b| {
                    let c = config.clone();
                    let i = input.clone();
                    b.iter(|| hfirst_resize(&c, &i));
                });
            }
        });
    }
});

//! End-to-end resize + composite benchmark.
//!
//! Measures the full streaming pipeline with solid background compositing
//! at realistic sizes. This captures the real-world impact of SIMD compositing
//! within the fused V-filter → H-filter → composite → unpremultiply pipeline.

use std::hint::black_box;

fn make_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; w as usize * h as usize * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * 4;
            buf[i] = (x % 256) as u8;
            buf[i + 1] = (y % 256) as u8;
            buf[i + 2] = ((x + y) % 256) as u8;
            buf[i + 3] = 128; // 50% alpha — forces composite to do real work
        }
    }
    buf
}

fn streaming_resize_composite(
    config: &zenresize::ResizeConfig,
    input: &[u8],
    bg: zenresize::SolidBackground,
) -> Vec<u8> {
    let in_w = config.in_width as usize;
    let channels = config.input.channels();
    let row_len = in_w * channels;
    let in_h = config.in_height as usize;
    let mut resizer = zenresize::StreamingResize::with_background(config, bg).unwrap();
    let mut output = Vec::with_capacity(
        config.out_width as usize * config.out_height as usize * channels,
    );
    for y in 0..in_h {
        resizer.push_row(&input[y * row_len..(y + 1) * row_len]).unwrap();
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

fn streaming_resize_no_bg(
    config: &zenresize::ResizeConfig,
    input: &[u8],
) -> Vec<u8> {
    let in_w = config.in_width as usize;
    let channels = config.input.channels();
    let row_len = in_w * channels;
    let in_h = config.in_height as usize;
    let mut resizer = zenresize::StreamingResize::new(config);
    let mut output = Vec::with_capacity(
        config.out_width as usize * config.out_height as usize * channels,
    );
    for y in 0..in_h {
        resizer.push_row(&input[y * row_len..(y + 1) * row_len]).unwrap();
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

struct Scenario {
    label: &'static str,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
}

zenbench::main!(|suite| {
    let scenarios = [
        Scenario { label: "4K→1080p", in_w: 3840, in_h: 2160, out_w: 1920, out_h: 1080 },
        Scenario { label: "4K→800x600", in_w: 4000, in_h: 3000, out_w: 800, out_h: 600 },
        Scenario { label: "1080p→4K up", in_w: 1920, in_h: 1080, out_w: 3840, out_h: 2160 },
    ];

    for s in &scenarios {
        let config = zenresize::ResizeConfig::builder(s.in_w, s.in_h, s.out_w, s.out_h)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGBA8_SRGB)
            .linear()
            .build();

        let input = make_gradient(s.in_w, s.in_h);
        let in_bytes = input.len() as u64;

        // Compare: no background (baseline) vs opaque white bg vs semi-transparent bg
        let label = s.label;
        suite.compare(label, |group| {
            group.throughput(zenbench::Throughput::Bytes(in_bytes));

            {
                let config = config.clone();
                let input = input.clone();
                group.bench("no_bg", move |b| {
                    let c = config.clone();
                    let i = input.clone();
                    b.iter(|| {
                        black_box(streaming_resize_no_bg(&c, &i));
                    });
                });
            }

            {
                let config = config.clone();
                let input = input.clone();
                group.bench("white_bg", move |b| {
                    let c = config.clone();
                    let i = input.clone();
                    b.iter(|| {
                        let bg = zenresize::SolidBackground::white(zenresize::PixelDescriptor::RGBA8_SRGB);
                        black_box(streaming_resize_composite(&c, &i, bg));
                    });
                });
            }

            {
                let config = config.clone();
                let input = input.clone();
                group.bench("semi_bg", move |b| {
                    let c = config.clone();
                    let i = input.clone();
                    b.iter(|| {
                        let bg = zenresize::SolidBackground::from_srgb_u8(128, 128, 128, 128, zenresize::PixelDescriptor::RGBA8_SRGB);
                        black_box(streaming_resize_composite(&c, &i, bg));
                    });
                });
            }
        });
    }
});

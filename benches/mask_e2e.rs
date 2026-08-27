//! End-to-end resize + output-mask benchmark (issue #3).
//!
//! Measures the cost of `StreamingResize::with_mask()` on the default
//! `RGBA8_SRGB` (non-linear) config, which without a mask takes the I16Srgb
//! path. The interesting number is the in-run ratio `rounded_mask / no_mask`:
//! it is ~2x while `with_mask` forces the f32 path, and should collapse to
//! ~1x once the mask is applied on the i16 path. Absolute timings drift
//! between runs on this machine; only the paired ratio is comparable.

use std::hint::black_box;

fn make_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut buf = vec![0u8; w as usize * h as usize * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * 4;
            buf[i] = (x % 256) as u8;
            buf[i + 1] = (y % 256) as u8;
            buf[i + 2] = ((x + y) % 256) as u8;
            buf[i + 3] = 255; // opaque photo: the mask is the only alpha source
        }
    }
    buf
}

fn run(
    resizer: &mut zenresize::StreamingResize,
    config: &zenresize::ResizeConfig,
    input: &[u8],
) -> Vec<u8> {
    let channels = config.input.channels();
    let row_len = config.in_width as usize * channels;
    let in_h = config.in_height as usize;
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

struct Scenario {
    label: &'static str,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
}

zenbench::main!(|suite| {
    let scenarios = [
        Scenario {
            label: "4K→1080p",
            in_w: 3840,
            in_h: 2160,
            out_w: 1920,
            out_h: 1080,
        },
        Scenario {
            label: "1080p→4K up",
            in_w: 1920,
            in_h: 1080,
            out_w: 3840,
            out_h: 2160,
        },
        Scenario {
            label: "800x600→400x300",
            in_w: 800,
            in_h: 600,
            out_w: 400,
            out_h: 300,
        },
    ];

    for s in &scenarios {
        // Non-linear sRGB: the no-mask arm takes the I16Srgb path.
        let config = zenresize::ResizeConfig::builder(s.in_w, s.in_h, s.out_w, s.out_h)
            .filter(zenresize::Filter::Lanczos)
            .format(zenresize::PixelDescriptor::RGBA8_SRGB)
            .srgb()
            .build();

        let input = make_gradient(s.in_w, s.in_h);
        let in_bytes = input.len() as u64;
        let radius = (s.out_h / 8) as f32;

        let label = s.label;
        {
            // Report which internal path each arm takes so a 1.00x ratio can't
            // be mistaken for "the mask is free" when both arms are on f32.
            let plain = zenresize::StreamingResize::new(&config);
            let masked = zenresize::StreamingResize::new(&config).with_mask(
                zenresize::RoundedRectMask::uniform(s.out_w, s.out_h, radius),
            );
            eprintln!(
                "[mask_e2e] {label}: no_mask={:?} rounded_mask={:?}",
                plain.working_format(),
                masked.working_format()
            );
        }
        suite.compare(label, |group| {
            group.throughput(zenbench::Throughput::Bytes(in_bytes));

            {
                let config = config.clone();
                let input = input.clone();
                group.bench("no_mask", move |b| {
                    let c = config.clone();
                    let i = input.clone();
                    b.iter(|| {
                        let mut r = zenresize::StreamingResize::new(&c);
                        black_box(run(&mut r, &c, &i));
                    });
                });
            }

            {
                let config = config.clone();
                let input = input.clone();
                group.bench("rounded_mask", move |b| {
                    let c = config.clone();
                    let i = input.clone();
                    b.iter(|| {
                        let mask =
                            zenresize::RoundedRectMask::uniform(c.out_width, c.out_height, radius);
                        let mut r = zenresize::StreamingResize::new(&c).with_mask(mask);
                        black_box(run(&mut r, &c, &i));
                    });
                });
            }
        });
    }
});

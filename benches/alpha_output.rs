//! Cross-revision paired benchmark. Same file is built in each source export.
use std::hint::black_box;
use tango_bench::{IntoBenchmarks, benchmark_fn, tango_benchmarks, tango_main};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

fn cases() -> impl IntoBenchmarks {
    if std::env::var_os("ZENRESIZE_MAGIC_SCALAR").is_some() {
        #[cfg(target_arch = "x86_64")]
        archmage::X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
        #[cfg(target_arch = "aarch64")]
        archmage::NeonToken::dangerously_disable_token_process_wide(true).unwrap();
    }
    let filter = if std::env::var_os("ZENRESIZE_MAGIC_2021").is_some() {
        Filter::MagicKernelSharp2021
    } else {
        Filter::MagicKernelSharp
    };
    [
        (1, 1),
        (64, 32),
        (65, 33),
        (512, 256),
        (255, 511),
        (256, 512),
        (1920, 960),
    ]
    .into_iter()
    .flat_map(move |(w, ow)| {
        ["opaque", "partial", "ringing", "rgbx", "linear_rgbx"]
            .into_iter()
            .map(move |alpha| {
                benchmark_fn(format!("alpha/{w}to{ow}/{alpha}"), move |b| {
                    let input: Vec<u8> = (0..w * w)
                        .flat_map(|i| {
                            let x = i % w;
                            let a = match alpha {
                                "opaque" | "rgbx" | "linear_rgbx" => 255,
                                "partial" => 160,
                                _ => {
                                    if x % 32 < 16 {
                                        128
                                    } else {
                                        255
                                    }
                                }
                            };
                            [64 + (x % 96) as u8, 96, 128, a]
                        })
                        .collect();
                    let builder = ResizeConfig::builder(w, w, ow, ow).filter(filter).format(
                        if alpha.contains("rgbx") {
                            PixelDescriptor::RGBX8_SRGB
                        } else {
                            PixelDescriptor::RGBA8_SRGB
                        },
                    );
                    let config = if alpha == "linear_rgbx" {
                        builder.linear().build()
                    } else {
                        builder.srgb().build()
                    };
                    let mut resizer = Resizer::new(&config);
                    let mut output = vec![0u8; (ow * ow * 4) as usize];
                    b.iter(move || {
                        resizer.resize_into(black_box(&input), black_box(&mut output));
                    })
                })
            })
    })
    .collect::<Vec<_>>()
}
tango_benchmarks!(cases());
tango_main!();

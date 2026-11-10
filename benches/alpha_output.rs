//! Cross-revision paired benchmark. Same file is built in each source export.
use std::hint::black_box;
use tango_bench::{IntoBenchmarks, benchmark_fn, tango_benchmarks, tango_main};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

fn cases() -> impl IntoBenchmarks {
    [(64, 32), (512, 256), (256, 512), (1920, 960)]
        .into_iter()
        .flat_map(|(w, ow)| {
            ["opaque", "partial", "ringing"]
                .into_iter()
                .map(move |alpha| {
                    benchmark_fn(format!("alpha/{w}to{ow}/{alpha}"), move |b| {
                        let input: Vec<u8> = (0..w * w)
                            .flat_map(|i| {
                                let x = i % w;
                                let a = match alpha {
                                    "opaque" => 255,
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
                        let config = ResizeConfig::builder(w, w, ow, ow)
                            .filter(Filter::MagicKernelSharp)
                            .format(PixelDescriptor::RGBA8_SRGB)
                            .srgb()
                            .build();
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

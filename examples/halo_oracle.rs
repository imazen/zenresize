//! Differential check of `zenresize::halo` against GEGL's NoHalo/LoHalo.
//!
//! ```text
//! halo_oracle resize <in.png> <width> <height> <nohalo|lohalo> <out.png>
//! halo_oracle diff <a.png> <b.png>
//! ```
//!
//! `just halo-oracle <in.png> <width> <height>` runs both samplers through
//! this and through `gegl ... gegl:scale-size` and prints the diffs. GEGL is
//! an independent implementation, so it is a test oracle only: agreement
//! within a few levels on photos says the samplers match in substance.

use std::process::ExitCode;

use zenresize::halo::{HaloSampler, resize_halo_srgba8};

fn load(path: &str) -> image::RgbaImage {
    image::open(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .to_rgba8()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["resize", input, w, h, sampler, output] => {
            let sampler = match *sampler {
                "nohalo" => HaloSampler::NoHalo,
                "lohalo" => HaloSampler::LoHalo,
                other => {
                    eprintln!("unknown sampler {other}");
                    return ExitCode::FAILURE;
                }
            };
            let src = load(input);
            let (iw, ih) = (src.width() as usize, src.height() as usize);
            let px: Vec<[u8; 4]> = src.pixels().map(|p| p.0).collect();
            let img = imgref::ImgRef::new(&px, iw, ih);
            let out = resize_halo_srgba8(img, w.parse().unwrap(), h.parse().unwrap(), sampler);
            let bytes: Vec<u8> = out.pixels().flatten().collect();
            image::RgbaImage::from_raw(out.width() as u32, out.height() as u32, bytes)
                .unwrap()
                .save(output)
                .unwrap();
            ExitCode::SUCCESS
        }
        ["diff", a, b] => {
            let (a, b) = (load(a), load(b));
            assert_eq!(a.dimensions(), b.dimensions(), "size mismatch");
            let mut hist = [0u64; 256];
            let mut sum = 0u64;
            let mut worst = (0u8, 0u32, 0u32);
            for ((x, y, pa), pb) in a.enumerate_pixels().zip(b.pixels()) {
                for c in 0..3 {
                    let d = pa.0[c].abs_diff(pb.0[c]);
                    hist[d as usize] += 1;
                    sum += d as u64;
                    if d > worst.0 {
                        worst = (d, x, y);
                    }
                }
            }
            let n: u64 = hist.iter().sum();
            let pct = |q: f64| {
                let mut acc = 0u64;
                for (d, &c) in hist.iter().enumerate() {
                    acc += c;
                    if acc as f64 >= q * n as f64 {
                        return d;
                    }
                }
                255
            };
            let max = hist.iter().rposition(|&c| c > 0).unwrap_or(0);
            println!(
                "rgb |diff|: mean {:.3}  p50 {}  p99 {}  p99.9 {}  max {} at ({}, {}) of {}x{}",
                sum as f64 / n as f64,
                pct(0.5),
                pct(0.99),
                pct(0.999),
                max,
                worst.1,
                worst.2,
                a.width(),
                a.height()
            );
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage: halo_oracle resize <in> <w> <h> <nohalo|lohalo> <out> | diff <a> <b>"
            );
            ExitCode::FAILURE
        }
    }
}

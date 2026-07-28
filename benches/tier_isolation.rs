//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! Every other bench here measures zenresize against pic-scale /
//! fast_image_resize / resize. That answers "are we competitive", not "is our
//! SIMD worth anything" — a kernel slower than its own scalar fallback still
//! looks fine next to a competitor. This bench runs the identical zenresize
//! pipeline with the native SIMD token disabled, which is the comparison that
//! can expose a bad kernel.
//!
//! Covers both the sRGB (u8 direct) and linear-light (f32 transfer) paths, and
//! both downscale and upscale, because they exercise different kernels.
//!
//! Run: `cargo bench --bench tier_isolation`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

fn make_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize * 4;
            rgba[i] = (x % 256) as u8;
            rgba[i + 1] = (y % 256) as u8;
            rgba[i + 2] = ((x + y) % 256) as u8;
            rgba[i + 3] = 255;
        }
    }
    rgba
}

fn resize_srgb(src: &[u8], w: u32, h: u32, ow: u32, oh: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(w, h, ow, oh)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelDescriptor::RGBA8_SRGB)
        .srgb()
        .build();
    zenresize::Resizer::new(&config).resize(src)
}

fn resize_linear(src: &[u8], w: u32, h: u32, ow: u32, oh: u32) -> Vec<u8> {
    let config = zenresize::ResizeConfig::builder(w, h, ow, oh)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelDescriptor::RGBA8_SRGB)
        .linear()
        .build();
    zenresize::Resizer::new(&config).resize(src)
}

/// (label, in_w, in_h, out_w, out_h). Downscale and upscale hit different
/// kernel shapes (accumulate-many vs interpolate-few), so both are measured.
const CASES: &[(&str, u32, u32, u32, u32)] = &[
    ("down_4k_to_1080p", 3840, 2160, 1920, 1080),
    ("down_1024_to_256", 1024, 1024, 256, 256),
    ("up_256_to_1024", 256, 256, 1024, 1024),
];

fn bench_tiers(c: &mut Criterion) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native, ensure \
             archmage/testable_dispatch). Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar");

    for &(label, w, h, ow, oh) in CASES {
        let src = make_gradient(w, h);
        for (mode, f) in [
            ("srgb", resize_srgb as fn(&[u8], u32, u32, u32, u32) -> Vec<u8>),
            ("linear", resize_linear),
        ] {
            let mut group = c.benchmark_group(format!("{label}/{mode}"));
            group.bench_function(TIER_NAME, |b| {
                set_simd(true);
                b.iter(|| f(&src, w, h, ow, oh))
            });
            group.bench_function("scalar", |b| {
                set_simd(false);
                b.iter(|| f(&src, w, h, ow, oh))
            });
            set_simd(true);
            group.finish();
        }
    }
    set_simd(true);
}

criterion_group!(benches, bench_tiers);
criterion_main!(benches);

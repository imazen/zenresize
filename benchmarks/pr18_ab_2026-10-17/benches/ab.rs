//! Default-configuration cost of the resampling-conventions change: every
//! case runs the same config through `old` (main), `same` (main again, the
//! A/A control) and `new` (this checkout).
use zenbench::prelude::*;

#[derive(Clone, Copy)]
enum Path {
    /// RGBA8 `.srgb()`: I16Srgb.
    SrgbI16,
    /// RGBX8 linear light: I16Linear (≤2x) or F32.
    LinearI16,
    /// RGBA8 linear light, straight alpha: F32 with the f16 row cache.
    LinearF32,
    /// RGBA f32 in and out: F32 with the f16 row cache.
    F32Io,
}

macro_rules! config {
    ($krate:ident, $path:expr, $w:expr, $h:expr, $ow:expr, $oh:expr) => {{
        use $krate::{Filter, PixelDescriptor, ResizeConfig};
        let b = ResizeConfig::builder($w, $h, $ow, $oh).filter(Filter::Lanczos);
        match $path {
            Path::SrgbI16 => b.format(PixelDescriptor::RGBA8_SRGB).srgb(),
            Path::LinearI16 => b.format(PixelDescriptor::RGBX8_SRGB).linear(),
            Path::LinearF32 => b.format(PixelDescriptor::RGBA8_SRGB).linear(),
            Path::F32Io => b.format(PixelDescriptor::RGBAF32_LINEAR),
        }
        .build()
    }};
}

fn input_u8(w: u32, h: u32) -> Vec<u8> {
    (0..w as usize * h as usize * 4)
        .map(|i| ((i * 73 + i / 7 * 19) % 256) as u8)
        .collect()
}

fn case(suite: &mut Suite, name: &str, path: Path, (w, h, ow, oh): (u32, u32, u32, u32)) {
    let old_cfg = config!(old, path, w, h, ow, oh);
    let same_cfg = config!(same, path, w, h, ow, oh);
    let new_cfg = config!(new, path, w, h, ow, oh);
    let bytes = input_u8(w, h);
    let floats: Vec<f32> = bytes.iter().map(|&v| f32::from(v) / 255.0).collect();
    suite.group(name, move |g| {
        g.throughput(Throughput::Elements(u64::from(w) * u64::from(h)));
        let (bytes_old, floats_old) = (bytes.clone(), floats.clone());
        g.bench("main", move |b| match path {
            Path::F32Io => b.iter(|| old::Resizer::new(&old_cfg).resize_f32(&floats_old)),
            _ => b.iter(|| old::Resizer::new(&old_cfg).resize(&bytes_old)),
        });
        let (bytes_same, floats_same) = (bytes.clone(), floats.clone());
        g.bench("main_aa", move |b| match path {
            Path::F32Io => b.iter(|| same::Resizer::new(&same_cfg).resize_f32(&floats_same)),
            _ => b.iter(|| same::Resizer::new(&same_cfg).resize(&bytes_same)),
        });
        g.bench("pr18", move |b| match path {
            Path::F32Io => b.iter(|| new::Resizer::new(&new_cfg).resize_f32(&floats)),
            _ => b.iter(|| new::Resizer::new(&new_cfg).resize(&bytes)),
        });
    });
}

fn construct(suite: &mut Suite, name: &str, (w, h, ow, oh): (u32, u32, u32, u32)) {
    let old_cfg = config!(old, Path::LinearF32, w, h, ow, oh);
    let same_cfg = config!(same, Path::LinearF32, w, h, ow, oh);
    let new_cfg = config!(new, Path::LinearF32, w, h, ow, oh);
    suite.group(name, move |g| {
        g.bench("main", move |b| b.iter(|| old::Resizer::new(&old_cfg)));
        g.bench("main_aa", move |b| b.iter(|| same::Resizer::new(&same_cfg)));
        g.bench("pr18", move |b| b.iter(|| new::Resizer::new(&new_cfg)));
    });
}

fn ab(suite: &mut Suite) {
    let sizes = [
        ("tiny_down", (32, 32, 16, 16)),
        ("tiny_up", (48, 48, 64, 64)),
        ("small_down", (256, 256, 128, 128)),
        ("medium_down", (1024, 1024, 512, 512)),
        ("medium_up", (512, 512, 1024, 1024)),
        ("large_down", (3840, 2160, 1920, 1080)),
    ];
    let paths = [
        ("srgb_i16", Path::SrgbI16),
        ("linear_i16", Path::LinearI16),
        ("linear_f32", Path::LinearF32),
        ("f32_io", Path::F32Io),
    ];
    // AB_REVERSE=1 runs the cases in reverse order, to separate code effects
    // from heap/cache state left by earlier groups.
    let mut cases: Vec<_> = sizes.into_iter().collect();
    if std::env::var_os("AB_REVERSE").is_some() {
        cases.reverse();
    }
    for (size_name, dims) in cases {
        for (path_name, path) in paths {
            case(suite, &format!("{path_name}/{size_name}"), path, dims);
        }
    }
    construct(suite, "construct/tiny", (32, 32, 16, 16));
    construct(suite, "construct/large_down", (3840, 2160, 960, 540));
}

zenbench::main!(ab);

//! Sampling conventions (`antialias`, `sample_mapping`, `boundary`), the
//! convention presets, and the `Arithmetic` modes, through the public API.
use zenresize::filter::InterpolationDetails;
use zenresize::{
    Arithmetic, BoundaryMode, Filter, PixelDescriptor, ResizeConfig, Resizer, SampleMapping,
};

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(14695981039346656037u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
    })
}

fn texture(len: u32) -> Vec<u8> {
    (0..len)
        .map(|i| ((i * 73 + i / 7 * 19) % 256) as u8)
        .collect()
}

/// One-channel f32 resize along x only, in f32 arithmetic.
fn resize_row(
    input: &[f32],
    out: u32,
    filter: Filter,
    tweak: impl Fn(ResizeConfigB) -> ResizeConfigB,
) -> Vec<f32> {
    let builder = ResizeConfig::builder(input.len() as u32, 1, out, 1)
        .format(PixelDescriptor::GRAYF32_LINEAR)
        .filter(filter)
        .arithmetic(Arithmetic::F32);
    Resizer::new(&tweak(builder).build()).resize_f32(input)
}
type ResizeConfigB = zenresize::ResizeConfigBuilder;

/// Index of the sample `Filter::Nearest` picks for each output.
fn nearest_indices(n: u32, m: u32, mapping: SampleMapping) -> Vec<usize> {
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    resize_row(&input, m, Filter::Nearest, |b| b.sample_mapping(mapping))
        .iter()
        .map(|&v| v as usize)
        .collect()
}

/// Input samples with nonzero weight in output `row` (impulse responses).
fn support(
    n: u32,
    m: u32,
    row: usize,
    filter: Filter,
    tweak: impl Fn(ResizeConfigB) -> ResizeConfigB,
) -> Vec<usize> {
    (0..n as usize)
        .filter(|&j| {
            let mut impulse = vec![0.0f32; n as usize];
            impulse[j] = 1.0;
            resize_row(&impulse, m, filter, &tweak)[row] != 0.0
        })
        .collect()
}

#[test]
fn preset_kernels_match_their_formulas() {
    let keys = InterpolationDetails::create(Filter::CubicA075);
    let gaussian = InterpolationDetails::create(Filter::Gaussian);
    let a = -0.75;
    for i in -4096..=4096 {
        let x = (i as f64 / 1024.0).abs();
        let expected = if x < 1.0 {
            (a + 2.0) * x.powi(3) - (a + 3.0) * x * x + 1.0
        } else if x < 2.0 {
            a * x.powi(3) - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
        } else {
            0.0
        };
        assert!((keys.filter(x) - expected).abs() < 1e-13, "keys x={x}");
        let expected = if x < 1.5 { (-2.0 * x * x).exp() } else { 0.0 };
        assert!(
            (gaussian.filter(x) - expected).abs() < 1e-15,
            "gaussian x={x}"
        );
    }
}

#[test]
fn fixed_support_clamped_cubic_matches_direct_convolution() {
    // antialias(false) + Clamp: out[i] = Σ_k w(k − c)·in[clamp(k)], c = (i+½)·n/m − ½,
    // over the kernel's fixed radius of 2 regardless of scale.
    let keys = |x: f64| InterpolationDetails::create(Filter::CubicA075).filter(x);
    let input: Vec<f32> = (0..37).map(|i| ((i * 73 % 101) as f32) / 7.0).collect();
    for m in [5u32, 11, 36, 37, 80] {
        let got = resize_row(&input, m, Filter::CubicA075, |b| {
            b.antialias(false).boundary(BoundaryMode::Clamp)
        });
        for (i, &value) in got.iter().enumerate() {
            let c = (i as f64 + 0.5) * input.len() as f64 / m as f64 - 0.5;
            let expected: f64 = (c.floor() as i64 - 1..=c.floor() as i64 + 2)
                .map(|k| {
                    keys(k as f64 - c)
                        * f64::from(input[k.clamp(0, input.len() as i64 - 1) as usize])
                })
                .sum();
            assert!((f64::from(value) - expected).abs() < 1e-4, "m={m} i={i}");
        }
    }
}

#[test]
fn antialias_widens_support_only_when_enabled() {
    let widened = support(127, 11, 5, Filter::CubicA075, |b| b);
    let fixed = support(127, 11, 5, Filter::CubicA075, |b| b.antialias(false));
    assert!(widened.len() > 4, "{widened:?}");
    assert!(fixed.len() <= 4, "{fixed:?}");
}

#[test]
fn antialias_off_and_clamp_give_exact_linear_values() {
    let input = [0.0, 10.0, 20.0, 30.0];
    for (m, expected) in [
        (2, vec![5.0, 25.0]),
        (8, vec![0.0, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 30.0]),
    ] {
        let got = resize_row(&input, m, Filter::Triangle, |b| {
            b.antialias(false).boundary(BoundaryMode::Clamp)
        });
        assert_eq!(got, expected);
    }
}

#[test]
fn align_corners_hits_endpoints_and_single_output_origin() {
    let input = [0.0, 1.0, 2.0, 3.0, 4.0];
    for m in [1, 2, 7, 19] {
        let got = resize_row(&input, m, Filter::Triangle, |b| {
            b.antialias(false)
                .sample_mapping(SampleMapping::AlignCorners)
                .boundary(BoundaryMode::Clamp)
        });
        assert_eq!(got[0], 0.0);
        if m > 1 {
            assert_eq!(got[got.len() - 1], 4.0);
        }
    }
}

#[test]
fn align_corners_with_widened_weighted_support_is_refused() {
    let base = || ResizeConfig::builder(7, 5, 3, 2).sample_mapping(SampleMapping::AlignCorners);
    assert!(base().build().validate().is_err());
    assert!(base().antialias(false).build().validate().is_ok());
    assert!(base().filter(Filter::Nearest).build().validate().is_ok());
}

#[test]
fn nearest_index_rules_at_integer_boundaries() {
    // Exact-integer half-pixel: floor((2i+1)·n / 2m); 38→21, i=10 lands on 19.
    assert_eq!(nearest_indices(38, 21, SampleMapping::HalfPixel)[10], 19);
    // 26→22, i=11: i·n/m = 13 exactly; the f32 product rounds just below it
    // (PyTorch `nearest`).
    assert_eq!(nearest_indices(26, 22, SampleMapping::Asymmetric)[11], 13);
    assert_eq!(
        nearest_indices(26, 22, SampleMapping::AsymmetricFloat32)[11],
        12
    );
    // 6→34, i=17: i·n/m = 3; the f64 reciprocal product floors to 2 (OpenCV).
    assert_eq!(
        nearest_indices(6, 34, SampleMapping::AsymmetricFloat64)[17],
        2
    );
    // AlignCorners rounds half up: 5→3 maps 0, 2, 4; 4→3 maps 0, 1.5→2, 3.
    assert_eq!(
        nearest_indices(5, 3, SampleMapping::AlignCorners),
        [0, 2, 4]
    );
    assert_eq!(
        nearest_indices(4, 3, SampleMapping::AlignCorners),
        [0, 2, 3]
    );
}

#[test]
fn nearest_copies_strided_samples() {
    for (w, h, ow, oh) in [(1, 7, 9, 3), (9, 1, 3, 7), (7, 5, 3, 2), (13, 17, 9, 8)] {
        let stride = w as usize * 3 + 5;
        let mut input = vec![0.0f32; stride * h as usize];
        for y in 0..h as usize {
            for x in 0..w as usize * 3 {
                input[y * stride + x] = (y * 101 + x) as f32 / 113.0;
            }
        }
        let config = ResizeConfig::builder(w, h, ow, oh)
            .format(PixelDescriptor::RGBF32_LINEAR)
            .in_stride(stride)
            .filter(Filter::Nearest)
            .sample_mapping(SampleMapping::Asymmetric)
            .arithmetic(Arithmetic::F32)
            .build();
        let output = Resizer::new(&config).resize_f32(&input);
        for y in 0..oh as usize {
            for x in 0..ow as usize * 3 {
                let source = (y * h as usize / oh as usize) * stride
                    + (x / 3 * w as usize / ow as usize) * 3
                    + x % 3;
                assert_eq!(
                    output[y * ow as usize * 3 + x].to_bits(),
                    input[source].to_bits()
                );
            }
        }
    }
}

#[test]
fn f32_arithmetic_keeps_samples_that_f16_would_round() {
    let input: Vec<f32> = (0..64).map(|i| 1.0 + i as f32 * 1e-4).collect();
    let exact = resize_row(&input, 64, Filter::Nearest, |b| b);
    assert_eq!(exact, input);
    let config = ResizeConfig::builder(64, 1, 64, 1)
        .format(PixelDescriptor::GRAYF32_LINEAR)
        .filter(Filter::Nearest)
        .build();
    assert_ne!(Resizer::new(&config).resize_f32(&input), input);
}

#[test]
fn half_pixel_float32_gaussian_support_matches_tensorflow_pins() {
    // TF 2.21.0 public impulse probes (benchmarks/conventions_2026-10-15/
    // pin_gaussian_support.py). Exact support sets catch f32 rounding at the
    // radius-1.5 cutoff.
    for (n, m, row, aa, expected) in [
        (18, 39, 6, false, vec![1, 2, 3]),
        (14, 29, 14, false, vec![6, 7, 8]),
        (17, 7, 0, true, vec![0, 1, 2, 3, 4]),
    ] {
        let got = support(n, m, row, Filter::Gaussian, |b| {
            b.antialias(aa)
                .sample_mapping(SampleMapping::HalfPixelFloat32)
        });
        assert_eq!(got, expected, "{n}->{m} row {row} aa={aa}");
    }
}

#[test]
fn every_convention_preserves_flat_fields() {
    let mappings = [
        SampleMapping::HalfPixel,
        SampleMapping::HalfPixelFloat32,
        SampleMapping::AlignCorners,
        SampleMapping::Asymmetric,
        SampleMapping::AsymmetricFloat32,
        SampleMapping::AsymmetricFloat64,
    ];
    let filters = [
        Filter::Nearest,
        Filter::CubicA075,
        Filter::Gaussian,
        Filter::Triangle,
    ];
    for (w, h, ow, oh) in [
        (1, 1, 1, 1),
        (1, 1, 5, 3),
        (5, 3, 1, 1),
        (1, 7, 9, 1),
        (9, 1, 1, 7),
        (2, 2, 3, 3),
    ] {
        for mapping in mappings {
            for aa in [true, false] {
                for boundary in [BoundaryMode::Truncate, BoundaryMode::Clamp] {
                    for filter in filters {
                        let builder = || {
                            ResizeConfig::builder(w, h, ow, oh)
                                .filter(filter)
                                .sample_mapping(mapping)
                                .antialias(aa)
                                .boundary(boundary)
                        };
                        let cfg = builder().format(PixelDescriptor::RGBF32_LINEAR).build();
                        if mapping == SampleMapping::AlignCorners && aa && filter != Filter::Nearest
                        {
                            assert!(cfg.validate().is_err());
                            continue;
                        }
                        let out = Resizer::new(&cfg).resize_f32(&vec![100.0; (w * h * 3) as usize]);
                        assert!(out.iter().all(|v| (*v - 100.0).abs() < 0.001), "{cfg:?}");
                        let cfg8 = builder().format(PixelDescriptor::RGBA8_SRGB).srgb().build();
                        let out = Resizer::new(&cfg8)
                            .resize(&[100, 100, 100, 255].repeat((w * h) as usize));
                        assert!(
                            out.as_chunks::<4>()
                                .0
                                .iter()
                                .all(|p| *p == [100, 100, 100, 255]),
                            "{cfg8:?}"
                        );
                    }
                }
            }
        }
    }
}

fn fixed22(
    w: u32,
    h: u32,
    ow: u32,
    oh: u32,
    format: PixelDescriptor,
) -> zenresize::ResizeConfigBuilder {
    ResizeConfig::builder(w, h, ow, oh)
        .format(format)
        .srgb()
        .filter(Filter::CatmullRom)
        .arithmetic(Arithmetic::Fixed22)
}

#[test]
fn fixed22_catmull_rom_matches_pillow_bicubic_pins() {
    // Pillow 12.3.0 public RGB bicubic API (benchmarks/conventions_2026-10-15/
    // pins.py). The larger case distinguishes 16-bit from 22-bit coefficients.
    for (w, h, ow, oh, expected) in [
        (7, 5, 3, 2, 0x35fa4ed06962600au64),
        (16, 12, 23, 9, 0xada71d758df83c6fu64),
    ] {
        let cfg = fixed22(w, h, ow, oh, PixelDescriptor::RGB8_SRGB).build();
        let output = Resizer::new(&cfg).resize(&texture(w * h * 3));
        assert_eq!(fnv(&output), expected, "{w}x{h} -> {ow}x{oh}");
    }
}

#[test]
fn fixed22_is_per_channel_and_stride_independent() {
    let (w, h, ow, oh) = (13, 9, 6, 17);
    let rgb = texture(w * h * 3);
    let packed =
        Resizer::new(&fixed22(w, h, ow, oh, PixelDescriptor::RGB8_SRGB).build()).resize(&rgb);

    let stride = w as usize * 3 + 7;
    let mut strided = vec![211u8; stride * h as usize];
    for y in 0..h as usize {
        strided[y * stride..][..w as usize * 3]
            .copy_from_slice(&rgb[y * w as usize * 3..][..w as usize * 3]);
    }
    let cfg = fixed22(w, h, ow, oh, PixelDescriptor::RGB8_SRGB)
        .in_stride(stride)
        .build();
    assert_eq!(Resizer::new(&cfg).resize(&strided), packed);

    for c in 0..3 {
        let gray: Vec<u8> = rgb.iter().skip(c).step_by(3).copied().collect();
        let cfg = fixed22(w, h, ow, oh, PixelDescriptor::GRAY8_SRGB).build();
        let out = Resizer::new(&cfg).resize(&gray);
        let expected: Vec<u8> = packed.iter().skip(c).step_by(3).copied().collect();
        assert_eq!(out, expected, "channel {c}");
    }
}

#[test]
fn fixed22_refuses_conversions_it_cannot_express() {
    let f = || fixed22(8, 8, 4, 4, PixelDescriptor::RGB8_SRGB);
    assert!(f().build().validate().is_ok());
    assert!(f().linear().build().validate().is_err());
    assert!(
        f().format(PixelDescriptor::RGBA8_SRGB)
            .build()
            .validate()
            .is_err()
    );
    assert!(
        f().format(PixelDescriptor::RGBX8_SRGB)
            .build()
            .validate()
            .is_ok()
    );
    assert!(
        f().format(PixelDescriptor::RGBF32_LINEAR)
            .build()
            .validate()
            .is_err()
    );
    let cfg = f().format(PixelDescriptor::RGBX8_SRGB).build();
    let opaque = zenresize::composite::SolidBackground::from_srgb_u8(
        255,
        255,
        255,
        255,
        PixelDescriptor::RGBX8_SRGB,
    );
    assert_eq!(
        zenresize::StreamingResize::with_background(&cfg, opaque)
            .err()
            .map(|e| e.error().clone()),
        Some(zenresize::composite::CompositeError::UnsupportedArithmetic)
    );
    assert!(zenresize::resize_hfirst_streaming(&cfg, &[0; 256]).is_err());
}

#[test]
fn extreme_downscale_keeps_full_tap_count() {
    let details = InterpolationDetails::create(Filter::Lanczos);
    let table = zenresize::weights::F32WeightTable::new(2_000_000, 100, &details);
    assert_eq!(table.tap_count(50), 119_984);
    let sum: f64 = table.weights(50).iter().map(|&w| f64::from(w)).sum();
    assert!((sum - 1.0).abs() < 1e-6, "weight sum {sum}");
    let table = zenresize::weights::I16WeightTable::new(2_000_000, 100, &details);
    assert!(table.tap_count(50) > 65_535);
}

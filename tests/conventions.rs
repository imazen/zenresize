use zenresize::weights::F32WeightTable;
use zenresize::{
    BoundaryMode, Filter, InterpolationDetails, PixelDescriptor, ResizeConfig, Resizer,
    SampleMapping,
};

#[test]
fn fixed_support_does_not_expand_on_downscale() {
    let base = InterpolationDetails::cubic_bc(0.0, 0.75);
    let aa = F32WeightTable::new(127, 11, &base);
    let fixed = F32WeightTable::new(127, 11, &base.with_antialias(false));
    assert!(aa.max_taps > 4);
    assert!(fixed.max_taps <= 4);
}
#[test]
fn public_cubic_matches_independent_mitchell_netravali_formula() {
    for (b, c) in [(0.0, 0.5), (1.0 / 3.0, 1.0 / 3.0), (0.0, 0.75), (0.5, 0.25)] {
        let custom = InterpolationDetails::cubic_bc(b, c);
        for i in -4096..=4096 {
            let x = (i as f64 / 1024.0).abs();
            let expected = if x < 1.0 {
                ((12.0 - 9.0 * b - 6.0 * c) * x.powi(3)
                    + (-18.0 + 12.0 * b + 6.0 * c) * x * x
                    + 6.0
                    - 2.0 * b)
                    / 6.0
            } else if x < 2.0 {
                ((-b - 6.0 * c) * x.powi(3)
                    + (6.0 * b + 30.0 * c) * x * x
                    + (-12.0 * b - 48.0 * c) * x
                    + 8.0 * b
                    + 24.0 * c)
                    / 6.0
            } else {
                0.0
            };
            assert!(
                (custom.filter(x) - expected).abs() < 1e-13,
                "b={b} c={c} x={x}"
            );
        }
    }
}
#[test]
fn nearest_asymmetric_is_integer_floor_and_strided() {
    for (w, h, ow, oh) in [
        (1, 7, 9, 3),
        (9, 1, 3, 7),
        (7, 5, 3, 2),
        (5, 3, 11, 7),
        (13, 17, 9, 8),
    ] {
        let stride = w as usize * 3 + 5;
        let mut input = vec![0.0f32; stride * h as usize];
        for y in 0..h as usize {
            for x in 0..w as usize * 3 {
                input[y * stride + x] = (y * 101 + x) as f32;
            }
        }
        let config = ResizeConfig::builder(w, h, ow, oh)
            .format(PixelDescriptor::RGBF32_LINEAR)
            .in_stride(stride)
            .interpolation(
                InterpolationDetails::nearest().with_sample_mapping(SampleMapping::Asymmetric),
            )
            .build();
        let output = Resizer::new(&config).resize_f32(&input);
        for y in 0..oh as usize {
            for x in 0..ow as usize {
                for c in 0..3 {
                    assert_eq!(
                        output[(y * ow as usize + x) * 3 + c],
                        input[(y * h as usize / oh as usize) * stride
                            + (x * w as usize / ow as usize) * 3
                            + c]
                    );
                }
            }
        }
    }
}
#[test]
fn align_corners_hits_endpoints_and_single_output_origin() {
    for ow in [1, 2, 7, 19] {
        let details = InterpolationDetails::create(Filter::Triangle)
            .with_antialias(false)
            .with_sample_mapping(SampleMapping::AlignCorners)
            .with_boundary_mode(BoundaryMode::Clamp);
        let input: Vec<f32> = (0..5).flat_map(|x| [x as f32; 3]).collect();
        let config = ResizeConfig::builder(5, 1, ow, 1)
            .format(PixelDescriptor::RGBF32_LINEAR)
            .interpolation(details)
            .build();
        let output = Resizer::new(&config).resize_f32(&input);
        assert_eq!(output[0], 0.0);
        if ow > 1 {
            assert_eq!(output[output.len() - 1], 4.0);
        }
    }
}
#[test]
fn invalid_custom_kernel_is_refused() {
    for (b, c) in [(f64::NAN, 0.5), (0.0, f64::INFINITY)] {
        let config = ResizeConfig::builder(7, 5, 3, 2)
            .interpolation(InterpolationDetails::cubic_bc(b, c))
            .build();
        assert!(config.validate().is_err());
    }
}

#[test]
fn nearest_preserves_non_half_f32_bits() {
    let input: Vec<f32> = (0..7 * 5 * 3)
        .map(|i| if i == 0 { -0.0 } else { i as f32 / 113.0 })
        .collect();
    let config = ResizeConfig::builder(7, 5, 11, 9)
        .format(PixelDescriptor::RGBF32_LINEAR)
        .interpolation(
            InterpolationDetails::nearest().with_sample_mapping(SampleMapping::Asymmetric),
        )
        .build();
    let output = Resizer::new(&config).resize_f32(&input);
    for y in 0..9 {
        for x in 0..11 {
            for c in 0..3 {
                assert_eq!(
                    output[(y * 11 + x) * 3 + c].to_bits(),
                    input[(y * 5 / 9 * 7 + x * 7 / 11) * 3 + c].to_bits()
                );
            }
        }
    }
}

#[test]
fn pillow_rgb_stride_and_validation() {
    let resizer = zenresize::PillowCubicResizer::new(7, 5, 11, 3).unwrap();
    let input: Vec<u8> = (0..7 * 5 * 3).map(|i| (i * 73 % 256) as u8).collect();
    let mut packed = vec![0; 11 * 3 * 3];
    resizer
        .resize_rgb8(&input, PixelDescriptor::RGB8_SRGB, 21, &mut packed, 33)
        .unwrap();
    let mut strided = vec![211u8; 27 * 5];
    for y in 0..5 {
        strided[y * 27..y * 27 + 21].copy_from_slice(&input[y * 21..y * 21 + 21]);
    }
    let mut output = vec![197u8; 40 * 3];
    resizer
        .resize_rgb8(&strided, PixelDescriptor::RGB8_SRGB, 27, &mut output, 40)
        .unwrap();
    for y in 0..3 {
        assert_eq!(&output[y * 40..y * 40 + 33], &packed[y * 33..y * 33 + 33]);
        assert!(output[y * 40 + 33..(y + 1) * 40].iter().all(|&x| x == 197));
    }
    assert!(
        resizer
            .resize_rgb8(&input, PixelDescriptor::RGB8_SRGB, 20, &mut packed, 33)
            .is_err()
    );
    assert!(
        resizer
            .resize_rgb8(
                &input[..10],
                PixelDescriptor::RGB8_SRGB,
                21,
                &mut packed,
                33
            )
            .is_err()
    );
    assert!(zenresize::PillowCubicResizer::new(1, 0, 2, 2).is_err());
    assert!(zenresize::PillowCubicResizer::new(1, 40_000, 40_000, 1).is_err());
}

#[test]
fn nearest_half_pixel_avoids_float_cancellation_at_exact_boundaries() {
    let details = InterpolationDetails::nearest();
    let weights = F32WeightTable::new(38, 21, &details);
    assert_eq!(weights.left[10], 19);
}
#[test]
fn torch_legacy_nearest_keeps_its_f32_boundary_rounding() {
    let exact = F32WeightTable::new(
        26,
        22,
        &InterpolationDetails::nearest().with_sample_mapping(SampleMapping::Asymmetric),
    );
    let torch = F32WeightTable::new(
        26,
        22,
        &InterpolationDetails::nearest().with_sample_mapping(SampleMapping::AsymmetricFloat32),
    );
    assert_eq!(exact.left[11], 13);
    assert_eq!(torch.left[11], 12);
}

#[test]
fn opencv_nearest_keeps_reciprocal_f64_boundary_rounding() {
    let cv = F32WeightTable::new(
        6,
        34,
        &InterpolationDetails::nearest().with_sample_mapping(SampleMapping::AsymmetricFloat64),
    );
    assert_eq!(cv.left[17], 2);
}

#[test]
fn pillow_public_oracle_pins_are_exact() {
    // Pillow 12.3.0 public RGB bicubic API. Generator: conventions pins.py.
    // The larger fixture distinguishes 16-bit from 22-bit coefficients in
    // the independent arithmetic study; both pins require exact byte equality.
    for (w, h, ow, oh, expected_hash) in [
        (7, 5, 3, 2, 0x35fa4ed06962600au64),
        (16, 12, 23, 9, 0xada71d758df83c6fu64),
    ] {
        let input: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 73 + i / 7 * 19) % 256) as u8)
            .collect();
        let mut output = vec![0; ow as usize * oh as usize * 3];
        zenresize::PillowCubicResizer::new(w, h, ow, oh)
            .unwrap()
            .resize_rgb8(
                &input,
                PixelDescriptor::RGB8_SRGB,
                w as usize * 3,
                &mut output,
                ow as usize * 3,
            )
            .unwrap();
        let hash = output.iter().fold(14695981039346656037u64, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
        });
        assert_eq!(hash, expected_hash, "{w}x{h} -> {ow}x{oh}");
    }
}

#[test]
fn custom_support_and_modifiers_are_included_in_memory_bound() {
    let mut large = InterpolationDetails::cubic_bc(0.0, 0.5).with_blur(1e-20);
    large.window = 1e20;
    assert!(
        ResizeConfig::builder(7, 5, 3, 2)
            .interpolation(large)
            .build()
            .validate()
            .is_err()
    );
    assert!(
        ResizeConfig::builder(7, 5, 3, 2)
            .kernel_width_scale(f64::INFINITY)
            .build()
            .validate()
            .is_err()
    );
    let ordinary = InterpolationDetails::cubic_bc(0.0, 0.5).with_blur(5.0);
    assert!(
        ResizeConfig::builder(7, 5, 3, 2)
            .interpolation(ordinary)
            .build()
            .validate()
            .is_ok()
    );
}

#[test]
fn tensorflow_public_gaussian_support_pins() {
    // Public TF 2.21.0 impulse probes; generator pin_gaussian_support.py.
    // Exact support masks catch reciprocal/direct-ratio changes at the hard
    // cutoff. Coefficient normalization is separately measured in the panel.
    for (n, m, row, aa, expected) in [
        (18, 39, 6, false, vec![1, 2, 3]),
        (14, 29, 14, false, vec![6, 7, 8]),
        (17, 7, 0, true, vec![0, 1, 2, 3, 4]),
    ] {
        let table = F32WeightTable::new(
            n,
            m,
            &InterpolationDetails::gaussian_tf().with_antialias(aa),
        );
        let left = table.left[row] as usize;
        let support: Vec<_> = table
            .weights(row)
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| (w > 0.0).then_some(left + i))
            .collect();
        assert_eq!(support, expected);
    }
}

#[test]
fn extreme_downscale_keeps_full_tap_count() {
    let details = InterpolationDetails::create(Filter::Lanczos);
    let table = F32WeightTable::new(2_000_000, 100, &details);
    assert_eq!(table.tap_count(50), 119_984);
    let sum: f64 = table.weights(50).iter().map(|&w| f64::from(w)).sum();
    assert!((sum - 1.0).abs() < 1e-6, "weight sum {sum}");
    let table = zenresize::weights::I16WeightTable::new(2_000_000, 100, &details);
    assert!(table.tap_count(50) > 65_535);
}

#[test]
fn reviewer_invalid_windows_and_coefficients_are_refused() {
    for boundary in [BoundaryMode::Truncate, BoundaryMode::Clamp] {
        for window in [0.25, 0.0, -1.0, f64::INFINITY, f64::NAN] {
            let mut d = InterpolationDetails::cubic_bc(0.0, 0.5).with_boundary_mode(boundary);
            d.window = window;
            assert!(
                ResizeConfig::builder(4, 1, 16, 1)
                    .interpolation(d)
                    .build()
                    .validate()
                    .is_err()
            );
        }
    }
    for field in 0..10 {
        let mut d = InterpolationDetails::cubic_bc(0.0, 0.5);
        match field {
            0 => d.blur = f64::NAN,
            1 => d.p1 = f64::NAN,
            2 => d.p2 = f64::NAN,
            3 => d.p3 = f64::NAN,
            4 => d.q1 = f64::NAN,
            5 => d.q2 = f64::NAN,
            6 => d.q3 = f64::NAN,
            7 => d.q4 = f64::NAN,
            8 => d.lobe_ratio_goal = Some(f32::NAN),
            _ => {
                d.p1 = 0.0;
                d.p2 = 0.0;
                d.p3 = 0.0;
                d.q1 = 0.0;
                d.q2 = 0.0;
                d.q3 = 0.0;
                d.q4 = 0.0;
            }
        }
        assert!(
            ResizeConfig::builder(4, 1, 16, 1)
                .interpolation(d)
                .build()
                .validate()
                .is_err()
        );
    }
    // Direct low-level callers cannot turn an empty signed range into 2^32 taps.
    let mut d = InterpolationDetails::cubic_bc(0.0, 0.5);
    d.window = 0.25;
    let t = F32WeightTable::new(4, 16, &d);
    assert!(t.max_taps <= 1);
}
#[test]
fn memory_guard_uses_support_not_curve_blur() {
    assert!(
        ResizeConfig::builder(4_000_000, 1, 1, 1)
            .kernel_width_scale(2.0)
            .build()
            .validate()
            .is_ok()
    );
    let mut d = InterpolationDetails::cubic_bc(0.0, 0.5)
        .with_blur(1e-5)
        .with_boundary_mode(BoundaryMode::Clamp);
    d.window = 1e5;
    assert!(
        ResizeConfig::builder(4000, 1, 2000, 1)
            .interpolation(d)
            .build()
            .validate()
            .is_err()
    );
}
#[test]
fn align_corners_antialias_is_refused() {
    assert!(
        ResizeConfig::builder(7, 5, 3, 2)
            .interpolation(
                InterpolationDetails::cubic_bc(0.0, 0.5)
                    .with_sample_mapping(SampleMapping::AlignCorners)
            )
            .build()
            .validate()
            .is_err()
    );
}
#[test]
fn pillow_refuses_rgba_descriptor() {
    let r = zenresize::PillowCubicResizer::new(4, 4, 2, 2)
        .unwrap()
        .clone();
    assert!(
        r.resize_rgb8(&[255; 64], PixelDescriptor::RGBA8_SRGB, 16, &mut [0; 12], 6)
            .is_err()
    );
    assert!(!format!("{r:?}").is_empty());
}
#[test]
fn builder_custom_precedence_is_independent_of_up_filter_order() {
    let d = InterpolationDetails::cubic_bc(0.0, 0.5);
    let a = ResizeConfig::builder(4, 3, 7, 5)
        .interpolation(d.clone())
        .up_filter(Filter::Box)
        .build();
    let b = ResizeConfig::builder(4, 3, 7, 5)
        .up_filter(Filter::Box)
        .interpolation(d)
        .build();
    let input: Vec<u8> = (0..4 * 3 * 4)
        .map(|i| ((i * 73 + i / 7 * 19) % 256) as u8)
        .collect();
    let mut reference = a.clone();
    reference.up_filter = None;
    assert_eq!(
        Resizer::new(&a).resize(&input),
        Resizer::new(&reference).resize(&input)
    );
    let named_box = ResizeConfig::builder(4, 3, 7, 5)
        .filter(Filter::Box)
        .build();
    assert_ne!(
        Resizer::new(&a).resize(&input),
        Resizer::new(&named_box).resize(&input)
    );
    assert_eq!(
        Resizer::new(&a).resize(&input),
        Resizer::new(&b).resize(&input)
    );
    assert!(a.interpolation.is_some());
    assert!(
        ResizeConfig::builder(4, 3, 7, 5)
            .interpolation(InterpolationDetails::nearest())
            .filter(Filter::Box)
            .build()
            .interpolation
            .is_none()
    );
}
#[test]
fn antialias_off_and_clamp_have_independent_linear_values() {
    let input: Vec<f32> = [0.0, 10.0, 20.0, 30.0]
        .into_iter()
        .flat_map(|v| [v; 3])
        .collect();
    for (ow, expected) in [
        (2, vec![5.0, 25.0]),
        (8, vec![0.0, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 30.0]),
    ] {
        let cfg = ResizeConfig::builder(4, 1, ow, 1)
            .format(PixelDescriptor::RGBF32_LINEAR)
            .interpolation(
                InterpolationDetails::create(Filter::Triangle)
                    .with_antialias(false)
                    .with_boundary_mode(BoundaryMode::Clamp),
            )
            .build();
        let out = Resizer::new(&cfg).resize_f32(&input);
        for (px, expected) in out.as_chunks::<3>().0.iter().zip(expected) {
            assert_eq!(px, &[expected; 3]);
        }
    }
}

#[test]
fn reviewer_tiny_shapes_preserve_finite_positive_pixels() {
    for (w, h, ow, oh) in [
        (1, 1, 1, 1),
        (1, 1, 5, 3),
        (5, 3, 1, 1),
        (1, 7, 9, 1),
        (9, 1, 1, 7),
        (2, 2, 3, 3),
    ] {
        for mapping in [
            SampleMapping::HalfPixel,
            SampleMapping::AlignCorners,
            SampleMapping::Asymmetric,
            SampleMapping::AsymmetricFloat32,
            SampleMapping::AsymmetricFloat64,
        ] {
            for aa in [true, false] {
                for boundary in [BoundaryMode::Truncate, BoundaryMode::Clamp] {
                    for base in [
                        InterpolationDetails::nearest(),
                        InterpolationDetails::cubic_bc(0.0, 0.75),
                        InterpolationDetails::gaussian_tf(),
                        InterpolationDetails::create(Filter::Triangle),
                    ] {
                        let nearest = base.window == 0.5;
                        let details = base
                            .with_sample_mapping(mapping)
                            .with_antialias(aa)
                            .with_boundary_mode(boundary);
                        let cfg = ResizeConfig::builder(w, h, ow, oh)
                            .format(PixelDescriptor::RGBF32_LINEAR)
                            .interpolation(details.clone())
                            .build();
                        if mapping == SampleMapping::AlignCorners && aa && !nearest {
                            assert!(cfg.validate().is_err());
                        } else {
                            cfg.validate().unwrap();
                            let output =
                                Resizer::new(&cfg).resize_f32(&vec![100.0; (w * h * 3) as usize]);
                            assert!(
                                output
                                    .iter()
                                    .all(|v| v.is_finite() && (*v - 100.0).abs() < 0.001)
                            );
                            let cfg8 = ResizeConfig::builder(w, h, ow, oh)
                                .format(PixelDescriptor::RGBA8_SRGB)
                                .srgb()
                                .interpolation(details)
                                .build();
                            assert!(
                                Resizer::new(&cfg8)
                                    // Opaque RGB isolates weight normalization from
                                    // the existing quantized alpha round trip.
                                    .resize(&[100, 100, 100, 255].repeat((w * h) as usize))
                                    .as_chunks::<4>()
                                    .0
                                    .iter()
                                    .all(|pixel| *pixel == [100, 100, 100, 255])
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn clamp_support_beyond_signed_input_edge_keeps_complete_kernel() {
    let details = InterpolationDetails::cubic_bc(0.0, 0.5)
        .with_antialias(false)
        .with_sample_mapping(SampleMapping::AlignCorners)
        .with_boundary_mode(BoundaryMode::Clamp);
    let table = F32WeightTable::new(i32::MAX as u32, 2, &details);
    assert_eq!(table.weights(0).iter().sum::<f32>(), 1.0);
    assert_eq!(table.weights(1).iter().sum::<f32>(), 1.0);
    assert_eq!(
        table.left[1] as usize + table.tap_count(1),
        i32::MAX as usize
    );
}

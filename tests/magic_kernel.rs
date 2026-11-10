//! Clean-room oracle: compose a quadratic B-spline with discrete sharpening.
//! Production instead evaluates expanded piecewise quadratics.
use zenresize::filter::InterpolationDetails;
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer, StreamingResize};

fn spline(x: f64) -> f64 {
    let t = x.abs();
    if t <= 0.5 {
        0.75 - t * t
    } else if t < 1.5 {
        (1.5 - t).powi(2) / 2.0
    } else {
        0.0
    }
}
fn oracle(x: f64) -> f64 {
    [-1.0, 6.0, -1.0]
        .iter()
        .enumerate()
        .map(|(i, w)| w * spline(x + i as f64 - 1.0) / 4.0)
        .sum()
}

#[test]
fn magic_formula_and_partition_of_unity() {
    let f = InterpolationDetails::create(Filter::MagicKernelSharp2013);
    for i in -4096..=4096 {
        let x = i as f64 / 1024.0;
        assert!((f.filter(x) - oracle(x)).abs() < 1e-14, "x={x}");
        let sum: f64 = (-8..=8).map(|j| f.filter(x - j as f64)).sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }
    for x in [0.5, 1.5, 2.5] {
        for delta in [-1e-9, 0.0, 1e-9] {
            assert!((f.filter(x + delta) - oracle(x + delta)).abs() < 1e-14);
        }
    }
    assert_eq!(f.filter(0.0), 17.0 / 16.0);
    assert_eq!(f.filter(1.0), 0.0);
    assert_eq!(f.filter(2.0), -1.0 / 32.0);
}

// The f32 APIs use a streaming f16 row cache; allow two f16 ULPs
// near unity (1e-3), rather than claiming f32 accumulation-only precision.
// Independent dense 2D reference, including truncated-and-renormalized edges.
fn weights(n: usize, out: usize, at: usize) -> Vec<f64> {
    let scale = out as f64 / n as f64;
    let center = (at as f64 + 0.5) / scale - 0.5;
    let mut w: Vec<_> = (0..n)
        .map(|i| oracle((i as f64 - center) * scale.min(1.0)))
        .collect();
    let sum: f64 = w.iter().sum();
    for v in &mut w {
        *v /= sum;
    }
    w
}

#[test]
fn magic_fullframe_and_streaming_match_dense_oracle() {
    for (w, h, ow, oh) in [
        (1, 1, 7, 5),
        (2, 3, 1, 1),
        (7, 5, 19, 13),
        (19, 13, 7, 5),
        (9, 7, 9, 7),
        (17, 3, 3, 11),
    ] {
        let input: Vec<f32> = (0..w * h * 4)
            .map(|i| {
                if i % 4 == 3 {
                    1.0
                } else {
                    0.2 + 0.6 * ((i * 37) % 101) as f32 / 100.0
                }
            })
            .collect();
        let config = ResizeConfig::builder(w as u32, h as u32, ow as u32, oh as u32)
            .filter(Filter::MagicKernelSharp2013)
            .format(PixelDescriptor::RGBAF32_LINEAR)
            .build();
        let actual = Resizer::new(&config).resize_f32(&input);
        let mut stream = StreamingResize::new(&config);
        let mut streamed = Vec::new();
        for row in input.chunks_exact(w * 4) {
            stream.push_row_f32(row).unwrap();
            while let Some(row) = stream.next_output_row_f32() {
                streamed.extend_from_slice(row);
            }
        }
        stream.finish();
        while let Some(row) = stream.next_output_row_f32() {
            streamed.extend_from_slice(row);
        }
        assert_eq!(streamed.len(), actual.len());
        for y in 0..oh {
            let wy = weights(h, oh, y);
            for x in 0..ow {
                let wx = weights(w, ow, x);
                for c in 0..4 {
                    let mut expected = 0.0;
                    for iy in 0..h {
                        for ix in 0..w {
                            expected += wy[iy] * wx[ix] * input[(iy * w + ix) * 4 + c] as f64;
                        }
                    }
                    let index = (y * ow + x) * 4 + c;
                    for value in [actual[index], streamed[index]] {
                        assert!(
                            (value as f64 - expected).abs() < 1e-3,
                            "{w}x{h}->{ow}x{oh} ({x},{y},{c}): {value} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mitchell_comparison_and_integer_constant_preservation() {
    for filter in [Filter::Mitchell, Filter::MagicKernelSharp2013] {
        let details = InterpolationDetails::create(filter);
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        // Reconstruct a unit step over subpixel phases, away from image edges.
        for phase in -4096..=4096 {
            let x = phase as f64 / 1024.0;
            let step: f64 = (0..16).map(|i| details.filter(x - i as f64)).sum();
            minimum = minimum.min(step);
            maximum = maximum.max(step);
        }
        println!(
            "{} step_min={minimum:.9} step_max={maximum:.9}",
            filter.name()
        );
        for (w, h, ow, oh) in [(1, 1, 11, 9), (17, 13, 5, 3), (5, 3, 17, 13)] {
            let config = ResizeConfig::builder(w, h, ow, oh)
                .filter(filter)
                .format(PixelDescriptor::RGBA8_SRGB)
                .srgb()
                .build();
            let input: Vec<u8> = (0..w * h).flat_map(|_| [80, 120, 160, 255]).collect();
            let expected: Vec<u8> = (0..ow * oh).flat_map(|_| [80, 120, 160, 255]).collect();
            assert_eq!(Resizer::new(&config).resize(&input), expected);
            let mut stream = StreamingResize::new(&config);
            let mut output = Vec::new();
            for row in input.chunks_exact(w as usize * 4) {
                stream.push_row(row).unwrap();
                while let Some(row) = stream.next_output_row() {
                    output.extend_from_slice(row);
                }
            }
            stream.finish();
            while let Some(row) = stream.next_output_row() {
                output.extend_from_slice(row);
            }
            assert_eq!(output, expected);
        }
    }
}

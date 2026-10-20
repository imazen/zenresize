//! NoHalo / LoHalo behaviour tests (issue #17).

use imgref::{ImgRef, ImgVec};
use zenresize::halo::{HaloSampler, resize_halo, resize_halo_srgba8};

const SAMPLERS: [HaloSampler; 2] = [HaloSampler::NoHalo, HaloSampler::LoHalo];

/// (in_w, in_h, out_w, out_h): up, down, mixed, identity, tiny.
const SIZES: [(usize, usize, u32, u32); 6] = [
    (17, 13, 61, 47),
    (64, 48, 19, 11),
    (40, 10, 13, 37),
    (23, 19, 23, 19),
    (1, 1, 5, 3),
    (3, 2, 1, 1),
];

/// Deterministic pseudo-random premultiplied pixels in [0, 1].
fn noise(w: usize, h: usize, seed: u32) -> ImgVec<[f32; 4]> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f32 / (1u32 << 24) as f32
    };
    let px = (0..w * h)
        .map(|_| {
            let a = next();
            [next() * a, next() * a, next() * a, a]
        })
        .collect();
    ImgVec::new(px, w, h)
}

#[test]
fn constant_image_stays_constant() {
    let c = [0.25f32, 0.5, 0.125, 0.75];
    for sampler in SAMPLERS {
        for (iw, ih, ow, oh) in SIZES {
            let src = ImgVec::new(vec![c; iw * ih], iw, ih);
            let out = resize_halo(src.as_ref(), ow, oh, sampler);
            for p in out.pixels() {
                for k in 0..4 {
                    assert!(
                        (p[k] - c[k]).abs() < 1e-5,
                        "{sampler:?} {iw}x{ih}->{ow}x{oh}: {p:?} != {c:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn nohalo_identity_is_exact() {
    // At scale 1 every sample lands on an original pixel, where LBB returns
    // the pixel value unchanged.
    let src = noise(29, 17, 1);
    let out = resize_halo(src.as_ref(), 29, 17, HaloSampler::NoHalo);
    assert_eq!(out.buf(), src.buf());
}

#[test]
fn nohalo_upsampling_is_locally_bounded() {
    // Nohalo values lie in the hull of their nearest originals and LBB stays
    // inside the range of its 4×4 fine-grid stencil, so every upsampled value
    // lies within the min/max of the 4×4 original pixels around it.
    for (iw, ih, ow, oh, seed) in [(17, 13, 61, 47, 2), (8, 8, 64, 64, 3), (31, 7, 40, 29, 4)] {
        let src = noise(iw, ih, seed);
        let out = resize_halo(src.as_ref(), ow, oh, HaloSampler::NoHalo);
        let (sx, sy) = (iw as f64 / ow as f64, ih as f64 / oh as f64);
        for (oy, row) in out.rows().enumerate() {
            let y = (oy as f64 + 0.5) * sy - 0.5;
            for (ox, p) in row.iter().enumerate() {
                let x = (ox as f64 + 0.5) * sx - 0.5;
                let (x0, y0) = (x.floor() as isize, y.floor() as isize);
                for k in 0..4 {
                    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                    for j in y0 - 1..=y0 + 2 {
                        for i in x0 - 1..=x0 + 2 {
                            let v = src[(
                                i.clamp(0, iw as isize - 1) as usize,
                                j.clamp(0, ih as isize - 1) as usize,
                            )][k];
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                    assert!(
                        p[k] >= lo - 1e-6 && p[k] <= hi + 1e-6,
                        "{iw}x{ih}->{ow}x{oh} ({ox},{oy}) ch{k}: {} outside [{lo}, {hi}]",
                        p[k]
                    );
                }
            }
        }
    }
}

#[test]
fn nohalo_step_edge_has_no_overshoot() {
    // A hard vertical edge upsampled 5x: plain cubics ring here; NoHalo must
    // stay within [0, 1] and be monotone across the edge.
    let w = 12;
    let src: Vec<[f32; 4]> = (0..w * 4)
        .map(|i| if i % w < w / 2 { [0.0; 4] } else { [1.0; 4] })
        .collect();
    let src = ImgVec::new(src, w, 4);
    let out = resize_halo(src.as_ref(), 60, 20, HaloSampler::NoHalo);
    for row in out.rows() {
        for pair in row.windows(2) {
            // 1e-6: f32 rounding in the Hermite sum.
            assert!(pair[1][0] >= pair[0][0] - 1e-6, "not monotone: {pair:?}");
        }
        for p in row {
            assert!((-1e-6..=1.0 + 1e-6).contains(&p[0]), "overshoot: {p:?}");
        }
    }
}

#[test]
fn nohalo_is_exact_on_linear_ramps() {
    // Nohalo subdivision and LBB are both exact on linear data, so away from
    // the clamped borders an upsampled ramp is the ramp.
    let (iw, ih) = (20, 16);
    let src: Vec<[f32; 4]> = (0..iw * ih)
        .map(|i| {
            let (x, y) = ((i % iw) as f32, (i / iw) as f32);
            let v = 0.01 * x + 0.02 * y + 0.1;
            [v, v, v, 1.0]
        })
        .collect();
    let src = ImgVec::new(src, iw, ih);
    let (ow, oh) = (60u32, 48u32);
    let out = resize_halo(src.as_ref(), ow, oh, HaloSampler::NoHalo);
    let s = iw as f64 / ow as f64;
    for (oy, row) in out.rows().enumerate() {
        let y = (oy as f64 + 0.5) * s - 0.5;
        for (ox, p) in row.iter().enumerate() {
            let x = (ox as f64 + 0.5) * s - 0.5;
            if x < 2.0 || y < 2.0 || x > iw as f64 - 3.0 || y > ih as f64 - 3.0 {
                continue;
            }
            let expected = (0.01 * x + 0.02 * y + 0.1) as f32;
            assert!(
                (p[0] - expected).abs() < 1e-5,
                "({ox},{oy}): {} != {expected}",
                p[0]
            );
        }
    }
}

#[test]
fn downscaling_antialiases_fine_checkerboard() {
    // A one-pixel checkerboard has nothing below Nyquist except its mean;
    // 8x downscaling must average it to grey, not alias it.
    let n = 128;
    let src: Vec<[f32; 4]> = (0..n * n)
        .map(|i| {
            let v = ((i % n + i / n) % 2) as f32;
            [v, v, v, 1.0]
        })
        .collect();
    let src = ImgVec::new(src, n, n);
    for sampler in SAMPLERS {
        let out = resize_halo(src.as_ref(), 16, 16, sampler);
        for p in out.pixels() {
            assert!((p[0] - 0.5).abs() < 0.02, "{sampler:?}: {p:?}");
            assert!((p[3] - 1.0).abs() < 1e-5, "{sampler:?}: {p:?}");
        }
    }
}

#[test]
fn strided_input_matches_packed() {
    let src = noise(21, 15, 5);
    let stride = 27;
    let mut padded = vec![[9.0f32; 4]; stride * 15];
    for (y, row) in src.rows().enumerate() {
        padded[y * stride..y * stride + 21].copy_from_slice(row);
    }
    let strided = ImgRef::new_stride(&padded, 21, 15, stride);
    for sampler in SAMPLERS {
        for (ow, oh) in [(50, 33), (7, 5), (21, 15)] {
            let a = resize_halo(src.as_ref(), ow, oh, sampler);
            let b = resize_halo(strided, ow, oh, sampler);
            assert_eq!(a.buf(), b.buf(), "{sampler:?} {ow}x{oh}");
        }
    }
}

#[test]
fn lohalo_sigmoidization_shrinks_overshoot() {
    // Upsample a hard 0 -> 1 edge. Plain Mitchell-Netravali overshoots both
    // ends; sigmoidized light maps 0 and 1 to themselves and continues the
    // curve linearly beyond them with slope (C/4)(1/s1 - s1), s1 = tanh(C/4),
    // so LoHalo's overshoot is exactly the plain cubic's times that slope.
    let w = 12usize;
    let src: Vec<[f32; 4]> = (0..w * 4)
        .map(|i| {
            if i % w < w / 2 {
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [1.0; 4]
            }
        })
        .collect();
    let src = ImgVec::new(src, w, 4);
    let out_w = 96u32;
    let out = resize_halo(src.as_ref(), out_w, 32, HaloSampler::LoHalo);

    // Plain Mitchell-Netravali at the same sample positions.
    let mn = |x: f64| -> f64 {
        let (b, c) = (1.0 / 3.0, 1.0 / 3.0);
        let x = x.abs();
        if x < 1.0 {
            ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
                + (-18.0 + 12.0 * b + 6.0 * c) * x * x
                + (6.0 - 2.0 * b))
                / 6.0
        } else if x < 2.0 {
            ((-b - 6.0 * c) * x * x * x
                + (6.0 * b + 30.0 * c) * x * x
                + (-12.0 * b - 48.0 * c) * x
                + (8.0 * b + 24.0 * c))
                / 6.0
        } else {
            0.0
        }
    };
    let scale = w as f64 / out_w as f64;
    let mut plain_hi = f64::MIN;
    for ox in 0..out_w {
        let x = (ox as f64 + 0.5) * scale - 0.5;
        let x0 = x.floor() as isize;
        let v: f64 = (x0 - 1..=x0 + 2)
            .map(|i| {
                let ic = i.clamp(0, w as isize - 1);
                if ic >= (w / 2) as isize {
                    mn(x - i as f64)
                } else {
                    0.0
                }
            })
            .sum();
        plain_hi = plain_hi.max(v);
    }
    let c = 3.38589f64;
    let s1 = (c / 4.0).tanh();
    let slope = 0.25 * c * (1.0 / s1 - s1);

    let lohalo_hi = out.pixels().map(|p| p[0]).fold(f32::MIN, f32::max) as f64;
    let lohalo_lo = out.pixels().map(|p| p[0]).fold(f32::MAX, f32::min) as f64;
    assert!(plain_hi > 1.01, "plain cubic should overshoot: {plain_hi}");
    let expected = (plain_hi - 1.0) * slope;
    assert!(
        ((lohalo_hi - 1.0) - expected).abs() < 1e-4,
        "LoHalo overshoot {} != {expected} (plain {})",
        lohalo_hi - 1.0,
        plain_hi - 1.0
    );
    // The step is symmetric about 0.5, and so is the sigmoid.
    assert!(
        (lohalo_lo + expected).abs() < 1e-4,
        "undershoot {lohalo_lo}"
    );
}

#[test]
fn srgba8_wrapper_preserves_flat_colours() {
    for sampler in SAMPLERS {
        for c in [
            [0u8, 0, 0, 255],
            [255, 255, 255, 255],
            [200, 100, 50, 255],
            [10, 20, 30, 128],
        ] {
            let src = ImgVec::new(vec![c; 9 * 7], 9, 7);
            for (ow, oh) in [(31, 22), (3, 2)] {
                let out = resize_halo_srgba8(src.as_ref(), ow, oh, sampler);
                assert!(
                    out.pixels().all(|p| p == c),
                    "{sampler:?} {c:?} -> {:?}",
                    out.pixels().find(|&p| p != c)
                );
            }
        }
    }
}

#[test]
fn srgba8_wrapper_transparent_stays_zero() {
    let src = ImgVec::new(vec![[0u8; 4]; 16], 4, 4);
    for sampler in SAMPLERS {
        let out = resize_halo_srgba8(src.as_ref(), 9, 2, sampler);
        assert!(out.pixels().all(|p| p == [0, 0, 0, 0]));
    }
}

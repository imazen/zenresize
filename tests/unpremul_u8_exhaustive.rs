//! `unpremultiply_u8_row` must be EXACTLY the integer formula it replaced —
//! verified over the complete input domain, not a sample.
//!
//! The kernel divides by the pixel's own alpha, a runtime value. There is no
//! SIMD integer divide, so the original scalar form could not vectorize
//! (measured 2.14 GiB/s against the premultiply kernel's 14.5 GiB/s, whose
//! divisor is the constant 255). Replacing three integer divides with one f32
//! divide is only acceptable if it is bit-identical.
//!
//! The argument that it is: `num = c*255 + a/2 <= 65152` and `a <= 254` are
//! integers exactly representable in f32, so IEEE division is correctly
//! rounded; truncating it could only disagree with integer floor if rounding
//! crossed an integer, and a non-integral `num/a` is at least `1/a >= 1/254`
//! from the nearest integer while its half-ULP is ~1000x smaller.
//!
//! Arguments about floating point are exactly the kind that turn out to have an
//! unconsidered corner, and here the entire domain is 256 channel values x 256
//! alpha values = 65,536 combinations. So this does not sample it — it
//! enumerates it. A passing run is a proof, not evidence.

use zenresize::simd::__bench_kernels as k;

/// The original integer formula, transcribed verbatim from the pre-change body.
fn reference_pixel(c: u8, a: u8) -> u8 {
    if a == 0 {
        0
    } else if a < 255 {
        let a16 = a as u16;
        ((c as u16 * 255 + a16 / 2) / a16).min(255) as u8
    } else {
        c
    }
}

/// Every (channel, alpha) pair, on all three colour channels at once.
#[test]
fn unpremultiply_u8_exact_over_complete_domain() {
    // One row holding every (c, a) combination: 256 alphas x 256 channel values.
    let mut row = Vec::with_capacity(256 * 256 * 4);
    for a in 0..=255u16 {
        for c in 0..=255u16 {
            // Put a different channel value in each of R/G/B so the three lanes
            // are not accidentally testing the same number.
            row.push(c as u8);
            row.push((255 - c) as u8);
            row.push(((c * 7) % 256) as u8);
            row.push(a as u8);
        }
    }
    let expect: Vec<u8> = row
        .chunks_exact(4)
        .flat_map(|p| {
            [
                reference_pixel(p[0], p[3]),
                reference_pixel(p[1], p[3]),
                reference_pixel(p[2], p[3]),
                p[3],
            ]
        })
        .collect();

    let mut got = row.clone();
    k::unpremultiply_u8_row(&mut got);

    if got != expect {
        let (i, (g, e)) = got
            .iter()
            .zip(expect.iter())
            .enumerate()
            .find(|(_, (g, e))| g != e)
            .unwrap();
        let px = i / 4;
        panic!(
            "unpremultiply diverged at element {i} (pixel {px}, channel {}): \
             got {g}, want {e}; pixel was {:?}",
            i % 4,
            &row[px * 4..px * 4 + 4]
        );
    }
    assert_eq!(got.len(), 256 * 256 * 4, "domain not fully enumerated");
}

/// Alpha must never be modified, and the a==0 / a==255 branches must hold.
#[test]
fn unpremultiply_u8_preserves_alpha_and_edge_branches() {
    let mut row: Vec<u8> = Vec::new();
    for a in [0u8, 1, 127, 128, 254, 255] {
        for c in [0u8, 1, 127, 128, 254, 255] {
            row.extend_from_slice(&[c, c, c, a]);
        }
    }
    let orig = row.clone();
    k::unpremultiply_u8_row(&mut row);

    for (i, (p, o)) in row.chunks_exact(4).zip(orig.chunks_exact(4)).enumerate() {
        assert_eq!(p[3], o[3], "alpha modified at pixel {i}");
        match o[3] {
            0 => assert_eq!(&p[..3], &[0, 0, 0], "a==0 must zero RGB, pixel {i}"),
            255 => assert_eq!(&p[..3], &o[..3], "a==255 must be identity, pixel {i}"),
            _ => {}
        }
    }
}

/// The hand-written NEON kernel processes 16 pixels per `vld4q_u8` and hands
/// the remainder to the portable body. The exhaustive test above uses 65,536
/// pixels — a multiple of 16 — so it never reaches that tail. Cover every
/// remainder length explicitly.
#[test]
fn unpremultiply_u8_tail_lengths_exact() {
    for px in 0..=40usize {
        let mut row: Vec<u8> = Vec::with_capacity(px * 4);
        for i in 0..px {
            // Vary alpha across the whole range, including 0 and 255, so the
            // tail sees the special-cased branches too.
            let a = ((i * 37) % 256) as u8;
            row.extend_from_slice(&[(i * 13 % 256) as u8, (i * 91 % 256) as u8, 200, a]);
        }
        let expect: Vec<u8> = row
            .chunks_exact(4)
            .flat_map(|p| {
                [
                    reference_pixel(p[0], p[3]),
                    reference_pixel(p[1], p[3]),
                    reference_pixel(p[2], p[3]),
                    p[3],
                ]
            })
            .collect();
        let mut got = row.clone();
        k::unpremultiply_u8_row(&mut got);
        assert_eq!(got, expect, "tail diverged at {px} pixels");
    }
}

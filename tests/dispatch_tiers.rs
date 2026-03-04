//! Verify SIMD dispatch correctness at all available tiers.
//!
//! Uses archmage's `for_each_token_permutation` to disable SIMD tokens
//! one by one, ensuring scalar fallback produces correct results.
//!
//! Run with `cargo test --test dispatch_tiers -- --test-threads=1`
//! for accurate results (token disabling is process-wide).

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

fn config_srgb(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> ResizeConfig {
    ResizeConfig::builder(in_w, in_h, out_w, out_h)
        .format(PixelDescriptor::RGBA8_SRGB)
        .filter(Filter::Lanczos)
        .srgb()
        .build()
}

/// Resize a small constant-color image at each dispatch tier and verify
/// the output matches the reference (all-tiers-enabled) result.
#[test]
fn dispatch_all_tiers_constant_color() {
    let in_w = 64u32;
    let in_h = 64u32;
    let out_w = 32u32;
    let out_h = 32u32;

    // Solid mid-gray RGBA
    let input = vec![128u8; (in_w * in_h * 4) as usize];
    let cfg = config_srgb(in_w, in_h, out_w, out_h);

    // Reference result with all tiers enabled
    let reference = Resizer::new(&cfg).resize(&input);

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let result = Resizer::new(&cfg).resize(&input);
        assert_eq!(
            result.len(),
            reference.len(),
            "output length mismatch at tier: {perm}"
        );
        for (i, (&a, &b)) in reference.iter().zip(result.iter()).enumerate() {
            assert!(
                (a as i16 - b as i16).unsigned_abs() <= 1,
                "pixel mismatch at byte {i}: ref={a}, got={b}, tier: {perm}"
            );
        }
    });

    eprintln!("dispatch_all_tiers_constant_color: {report}");
}

/// Resize a gradient image at each dispatch tier and verify consistency.
#[test]
fn dispatch_all_tiers_gradient() {
    let in_w = 100u32;
    let in_h = 80u32;
    let out_w = 50u32;
    let out_h = 40u32;

    // Horizontal gradient
    let mut input = vec![0u8; (in_w * in_h * 4) as usize];
    for y in 0..in_h as usize {
        for x in 0..in_w as usize {
            let v = (x * 255 / (in_w as usize - 1)) as u8;
            let off = (y * in_w as usize + x) * 4;
            input[off] = v;
            input[off + 1] = v;
            input[off + 2] = v;
            input[off + 3] = 255;
        }
    }

    let cfg = config_srgb(in_w, in_h, out_w, out_h);
    let reference = Resizer::new(&cfg).resize(&input);

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let result = Resizer::new(&cfg).resize(&input);
        for (i, (&a, &b)) in reference.iter().zip(result.iter()).enumerate() {
            assert!(
                (a as i16 - b as i16).unsigned_abs() <= 1,
                "pixel mismatch at byte {i}: ref={a}, got={b}, tier: {perm}"
            );
        }
    });

    eprintln!("dispatch_all_tiers_gradient: {report}");
}

/// Upscale test — verify dispatch tiers produce consistent results.
#[test]
fn dispatch_all_tiers_upscale() {
    let in_w = 32u32;
    let in_h = 32u32;
    let out_w = 96u32;
    let out_h = 96u32;

    let mut input = vec![0u8; (in_w * in_h * 4) as usize];
    // Checkerboard pattern
    for y in 0..in_h as usize {
        for x in 0..in_w as usize {
            let v = if (x + y) % 2 == 0 { 200u8 } else { 50u8 };
            let off = (y * in_w as usize + x) * 4;
            input[off] = v;
            input[off + 1] = v;
            input[off + 2] = v;
            input[off + 3] = 255;
        }
    }

    let cfg = config_srgb(in_w, in_h, out_w, out_h);
    let reference = Resizer::new(&cfg).resize(&input);

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let result = Resizer::new(&cfg).resize(&input);
        for (i, (&a, &b)) in reference.iter().zip(result.iter()).enumerate() {
            assert!(
                (a as i16 - b as i16).unsigned_abs() <= 1,
                "pixel mismatch at byte {i}: ref={a}, got={b}, tier: {perm}"
            );
        }
    });

    eprintln!("dispatch_all_tiers_upscale: {report}");
}

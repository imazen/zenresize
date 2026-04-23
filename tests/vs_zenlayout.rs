//! Brute-force parity test: zenresize's [`fit_dims`] + [`fit_cover_source_crop`]
//! vs `zenlayout::Constraint::compute` across a wide sweep of inputs.
//!
//! zenresize's constraint solver is a port of zenlayout's. This test guarantees
//! they produce byte-identical output so downstream callers migrating from
//! zenlayout to zenresize see no behavioral drift (pixel-level resize results
//! stay bit-identical on every image).
//!
//! Covers:
//! - `Fit` ↔ `ConstraintMode::Fit`
//! - `Within` ↔ `ConstraintMode::Within`
//! - `Cover` (dims + source crop) ↔ `ConstraintMode::FitCrop`
//! - `Stretch` ↔ `ConstraintMode::Distort`
//!
//! Run: `cargo test --release --test vs_zenlayout`

use zenlayout::{Constraint, ConstraintMode};
use zenresize::{FitMode, fit_cover_source_crop, fit_dims};

/// Sizes chosen to stress the rounding-snap logic: prime-ish, pathological
/// aspect ratios, matched aspects, round numbers, odd numbers, edges.
fn sweep_sizes() -> Vec<u32> {
    vec![
        1, 2, 3, 5, 7, 8, 9, 10, 15, 16, 17,
        32, 33, 64, 99, 100, 101, 127, 128, 129,
        250, 256, 257, 300, 333, 400, 401, 499, 500, 501,
        600, 750, 800, 801, 1000, 1023, 1024, 1025,
        1200, 1366, 1600, 1920, 2048, 2560, 3000, 3840, 4096, 7681,
    ]
}

fn zenlayout_dims(in_w: u32, in_h: u32, max_w: u32, max_h: u32, mode: ConstraintMode) -> (u32, u32) {
    let layout = Constraint::new(mode, max_w, max_h)
        .compute(in_w, in_h)
        .expect("zenlayout compute failed on non-zero input");
    (layout.canvas.width, layout.canvas.height)
}

fn zenlayout_source_crop(in_w: u32, in_h: u32, max_w: u32, max_h: u32) -> (u32, u32, u32, u32) {
    let layout = Constraint::new(ConstraintMode::FitCrop, max_w, max_h)
        .compute(in_w, in_h)
        .expect("zenlayout compute failed");
    match layout.source_crop {
        Some(r) => (r.x, r.y, r.width, r.height),
        None => (0, 0, in_w, in_h),
    }
}

fn check_parity<F, G>(name: &str, ours: F, theirs: G)
where
    F: Fn(u32, u32, u32, u32) -> (u32, u32),
    G: Fn(u32, u32, u32, u32) -> (u32, u32),
{
    let sizes = sweep_sizes();
    let mut mismatches = Vec::new();
    let mut total = 0usize;
    for &iw in &sizes {
        for &ih in &sizes {
            for &mw in &sizes {
                for &mh in &sizes {
                    total += 1;
                    let ours = ours(iw, ih, mw, mh);
                    let theirs = theirs(iw, ih, mw, mh);
                    if ours != theirs {
                        mismatches.push(format!(
                            "  {name}: in={iw}x{ih} max={mw}x{mh} — ours={:?} vs zenlayout={:?}",
                            ours, theirs
                        ));
                    }
                }
            }
        }
    }
    if !mismatches.is_empty() {
        let shown = mismatches.len().min(20);
        panic!(
            "{name}: {}/{total} mismatches (showing first {shown})\n{}",
            mismatches.len(),
            mismatches[..shown].join("\n")
        );
    }
    eprintln!("{name}: {total} cases, all match zenlayout");
}

#[test]
fn fit_matches_zenlayout() {
    check_parity(
        "Fit",
        |iw, ih, mw, mh| fit_dims(iw, ih, mw, mh, FitMode::Fit),
        |iw, ih, mw, mh| zenlayout_dims(iw, ih, mw, mh, ConstraintMode::Fit),
    );
}

#[test]
fn within_matches_zenlayout() {
    check_parity(
        "Within",
        |iw, ih, mw, mh| fit_dims(iw, ih, mw, mh, FitMode::Within),
        |iw, ih, mw, mh| zenlayout_dims(iw, ih, mw, mh, ConstraintMode::Within),
    );
}

#[test]
fn cover_dims_matches_zenlayout() {
    // zenlayout FitCrop's canvas is always the target (after crop + resize).
    check_parity(
        "Cover (dims)",
        |iw, ih, mw, mh| fit_dims(iw, ih, mw, mh, FitMode::Cover),
        |iw, ih, mw, mh| zenlayout_dims(iw, ih, mw, mh, ConstraintMode::FitCrop),
    );
}

#[test]
fn stretch_matches_zenlayout() {
    check_parity(
        "Stretch",
        |iw, ih, mw, mh| fit_dims(iw, ih, mw, mh, FitMode::Stretch),
        |iw, ih, mw, mh| zenlayout_dims(iw, ih, mw, mh, ConstraintMode::Distort),
    );
}

#[test]
fn cover_source_crop_matches_zenlayout() {
    let sizes = sweep_sizes();
    let mut mismatches = Vec::new();
    let mut total = 0usize;
    for &iw in &sizes {
        for &ih in &sizes {
            for &mw in &sizes {
                for &mh in &sizes {
                    total += 1;
                    let ours = fit_cover_source_crop(iw, ih, mw, mh);
                    let theirs = zenlayout_source_crop(iw, ih, mw, mh);
                    if ours != theirs {
                        mismatches.push(format!(
                            "  cover-crop: in={iw}x{ih} max={mw}x{mh} — ours={:?} zen={:?}",
                            ours, theirs
                        ));
                    }
                }
            }
        }
    }
    if !mismatches.is_empty() {
        let shown = mismatches.len().min(20);
        panic!(
            "cover-source-crop: {}/{total} mismatches (showing first {shown})\n{}",
            mismatches.len(),
            mismatches[..shown].join("\n")
        );
    }
    eprintln!("cover-source-crop: {total} cases, all match zenlayout");
}

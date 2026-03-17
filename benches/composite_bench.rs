//! Benchmark: old scalar compositing vs new zenblend SIMD compositing.
//!
//! Measures three hot paths:
//! 1. blend_row (fg over row-based bg)
//! 2. blend_row_solid (fg over solid bg pixel)
//! 3. blend_row_solid_opaque (fg over opaque solid bg)
//!
//! Each at multiple row widths to capture cache behavior.

use std::hint::black_box;

/// Old scalar source-over: `fg[i] += bg[i] * (1 - fg_alpha)`.
/// Copied verbatim from pre-zenblend `composite.rs`.
#[inline(never)]
fn old_composite_over_premul(src: &mut [f32], bg: &[f32]) {
    for (s, b) in src.chunks_exact_mut(4).zip(bg.chunks_exact(4)) {
        let inv_a = 1.0 - s[3];
        s[0] += b[0] * inv_a;
        s[1] += b[1] * inv_a;
        s[2] += b[2] * inv_a;
        s[3] += b[3] * inv_a;
    }
}

/// Old scalar solid source-over.
#[inline(never)]
fn old_composite_over_solid_premul(src: &mut [f32], pixel: &[f32; 4]) {
    for s in src.chunks_exact_mut(4) {
        let inv_a = 1.0 - s[3];
        s[0] += pixel[0] * inv_a;
        s[1] += pixel[1] * inv_a;
        s[2] += pixel[2] * inv_a;
        s[3] += pixel[3] * inv_a;
    }
}

/// Old scalar solid opaque source-over.
#[inline(never)]
fn old_composite_over_solid_opaque_premul(src: &mut [f32], pixel: &[f32; 4]) {
    for s in src.chunks_exact_mut(4) {
        let inv_a = 1.0 - s[3];
        s[0] += pixel[0] * inv_a;
        s[1] += pixel[1] * inv_a;
        s[2] += pixel[2] * inv_a;
        s[3] = 1.0;
    }
}

fn make_fg(width: usize) -> Vec<f32> {
    let mut fg = vec![0.0f32; width * 4];
    for px in 0..width {
        let i = px * 4;
        let t = px as f32 / width as f32;
        fg[i] = t * 0.5;
        fg[i + 1] = (1.0 - t) * 0.3;
        fg[i + 2] = 0.1;
        fg[i + 3] = 0.5 + t * 0.3;
    }
    fg
}

fn make_bg(width: usize) -> Vec<f32> {
    let mut bg = vec![0.0f32; width * 4];
    for px in 0..width {
        let i = px * 4;
        let t = px as f32 / width as f32;
        bg[i] = 0.2 + t * 0.3;
        bg[i + 1] = 0.4;
        bg[i + 2] = (1.0 - t) * 0.6;
        bg[i + 3] = 0.8 + t * 0.2;
    }
    bg
}

zenbench::main!(|suite| {
    let widths: &[(usize, &str)] = &[
        (256, "256px"),
        (1920, "1920px"),
        (4096, "4096px"),
    ];

    for &(width, label) in widths {
        let fg_template = make_fg(width);
        let bg = make_bg(width);
        let pixel: [f32; 4] = [0.3, 0.5, 0.2, 0.8];
        let opaque_pixel: [f32; 4] = [0.3, 0.5, 0.2, 1.0];

        // --- blend_row: fg over row bg ---
        {
            let name = format!("blend_row {label}");
            let fg_t = fg_template.clone();
            let bg_c = bg.clone();
            suite.compare(&name, |group| {
                group.throughput(zenbench::Throughput::Bytes((width * 4 * 4) as u64));

                {
                    let fg_t = fg_t.clone();
                    let bg_c = bg_c.clone();
                    group.bench("old_scalar", move |b| {
                        let mut fg = fg_t.clone();
                        b.iter(|| {
                            fg.copy_from_slice(&fg_t);
                            old_composite_over_premul(&mut fg, &bg_c);
                            black_box(fg[0]);
                        });
                    });
                }

                {
                    let fg_t = fg_t.clone();
                    let bg_c = bg_c.clone();
                    group.bench("zenblend", move |b| {
                        let mut fg = fg_t.clone();
                        b.iter(|| {
                            fg.copy_from_slice(&fg_t);
                            zenblend::blend_row(&mut fg, &bg_c, zenblend::BlendMode::SrcOver);
                            black_box(fg[0]);
                        });
                    });
                }
            });
        }

        // --- blend_row_solid: fg over solid bg ---
        {
            let name = format!("blend_solid {label}");
            let fg_t = fg_template.clone();
            suite.compare(&name, |group| {
                group.throughput(zenbench::Throughput::Bytes((width * 4 * 4) as u64));

                {
                    let fg_t = fg_t.clone();
                    group.bench("old_scalar", move |b| {
                        let mut fg = fg_t.clone();
                        b.iter(|| {
                            fg.copy_from_slice(&fg_t);
                            old_composite_over_solid_premul(&mut fg, &pixel);
                            black_box(fg[0]);
                        });
                    });
                }

                {
                    let fg_t = fg_t.clone();
                    group.bench("zenblend", move |b| {
                        let mut fg = fg_t.clone();
                        b.iter(|| {
                            fg.copy_from_slice(&fg_t);
                            zenblend::blend_row_solid(&mut fg, &pixel, zenblend::BlendMode::SrcOver);
                            black_box(fg[0]);
                        });
                    });
                }
            });
        }

        // --- blend_row_solid_opaque: fg over opaque solid bg ---
        {
            let name = format!("blend_opaque {label}");
            let fg_t = fg_template.clone();
            suite.compare(&name, |group| {
                group.throughput(zenbench::Throughput::Bytes((width * 4 * 4) as u64));

                {
                    let fg_t = fg_t.clone();
                    group.bench("old_scalar", move |b| {
                        let mut fg = fg_t.clone();
                        b.iter(|| {
                            fg.copy_from_slice(&fg_t);
                            old_composite_over_solid_opaque_premul(&mut fg, &opaque_pixel);
                            black_box(fg[0]);
                        });
                    });
                }

                {
                    let fg_t = fg_t.clone();
                    group.bench("zenblend", move |b| {
                        let mut fg = fg_t.clone();
                        b.iter(|| {
                            fg.copy_from_slice(&fg_t);
                            zenblend::blend_row_solid_opaque(&mut fg, &opaque_pixel, zenblend::BlendMode::SrcOver);
                            black_box(fg[0]);
                        });
                    });
                }
            });
        }
    }
});

//! Per-kernel NEON-vs-scalar for the resize pipeline's row kernels.
//!
//! zenresize was previously only measured end-to-end (whole-image resize:
//! 1.56x sRGB / 4.29x linear on aarch64). An end-to-end number cannot reveal a
//! single kernel that is SLOWER than its own scalar fallback — that failure
//! mode was found in garb, zensim, zentone and zenpng during the same sweep,
//! and in zenpng six of eight filter paths were losing.
//!
//! Run: `cargo bench --bench kernel_tiers`
//! Do NOT pass `-C target-cpu=native` (the tier then cannot be disabled).
//!
//! # READING THESE RATIOS: they are biased AGAINST the SIMD arm
//!
//! Measured 2026-08-01 while fixing `premultiply_alpha_row`. Point BOTH arms at
//! byte-identical source and the ratio is still **0.92x**, not 1.00x (597/550,
//! 596/552 ns, reproduced). The forced-scalar arm resolves to a plain
//! `pub(crate) fn` that INLINES into this bench's loop; the dispatched arm is a
//! real function carrying `target_feature` and cannot. The gap is the boundary,
//! not the kernel.
//!
//! Consequences when reading any number here:
//!
//! - **1.00x is a PASS, not a tie.** The vector body has already paid back the
//!   ~8% boundary cost. `premultiply_alpha_row` and `u8_to_f32_row` sit here.
//! - **Below ~0.95x is the finding** — the body is not covering the boundary.
//! - A ratio cannot be improved past the boundary by editing the kernel alone;
//!   that needs a caller-side fix (hoist the token, widen the `#[arcane]` region
//!   to cover more work). Same conclusion zenwebp reached for `dequantize_block`.
//!
//! # The trap that produced a false verdict here
//!
//! An earlier revision cloned the 30 KB row INSIDE the timed region. A large
//! constant added to both arms compresses every ratio toward 1.00x, which does
//! not merely add noise — it MANUFACTURES "no gain" verdicts that then get
//! written into source as decisions. That is exactly what happened to
//! `premultiply_alpha_row`: the comment said a hand-written `vld4q_f32` version
//! had been measured at 1.00x and rejected, while the shipped path was actually
//! 0.92x and that rejected kernel was the fix. Keep every buffer setup inside
//! `with_input` (untimed) — see the `ab!` / `ab_inplace!` macros below.

use zenbench::prelude::*;
use zenresize::simd::__bench_kernels as k;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") { "neon" } else { "v3(avx2)" };

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(on: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!on).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_on: bool) -> bool { false }

const W: usize = 1920;

fn bench_kernels(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    let n = W * 4;
    const PAD: usize = 512; // H kernels over-read to `max_taps`; see the note below.
    let u8src: &'static [u8] = Box::leak((0..n + PAD).map(|i| if i < n { (i % 251) as u8 } else { 0 }).collect::<Vec<_>>().into_boxed_slice());
    let fsrc: &'static [f32] = Box::leak((0..n + PAD).map(|i| if i < n { (i % 251) as f32 / 251.0 } else { 0.0 }).collect::<Vec<_>>().into_boxed_slice());

    // In-place kernels: the buffer clone is a 30 KB allocation + copy, which
    // dominates a sub-microsecond kernel and made both arms move ~30% between
    // runs. `with_input` is not timed, so build the buffer there.
    macro_rules! ab_inplace {
        ($name:expr, $src:expr, $call:expr) => {
            suite.compare($name, |g| {
                g.throughput(Throughput::Bytes(n as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        b.with_input(move || {
                            set_simd(simd);
                            $src.to_vec()
                        })
                        .run(move |mut r| {
                            $call(&mut r);
                            r
                        })
                    });
                }
            });
        };
    }

    // Out-of-place kernels. The output buffer is built in `with_input`
    // (untimed) for the same reason `ab_inplace!` does it: a 30 KB alloc
    // inside the timed body is a large constant on both arms, and that is
    // exactly what manufactured the false `premultiply_alpha_row` verdict
    // documented at the top of this file. (Fixed 2026-08-01 — this macro had
    // kept allocating in the body after `ab_inplace!` was corrected.)
    // `ab!` kept as a thin alias of `ab_out!` so every call site gets the
    // untimed allocation. Same signature: (name, output-expr, call).
    macro_rules! ab {
        ($name:expr, $out:expr, $call:expr) => { ab_out!($name, $out, $call) };
    }

    macro_rules! ab_out {
        ($name:expr, $out:expr, $call:expr) => {
            suite.compare($name, |g| {
                g.throughput(Throughput::Bytes(n as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        b.with_input(move || {
                            set_simd(simd);
                            $out
                        })
                        .run(move |mut o| {
                            $call(&mut o);
                            o
                        })
                    });
                }
            });
        };
    }

    let u8src2 = u8src.clone();
    let u8src3 = u8src.clone();
    ab_out!("u8_to_f32_row", vec![0f32; n], |o: &mut Vec<f32>| k::u8_to_f32_row(&u8src[..n], o));
    ab_out!("f32_to_u8_row", vec![0u8; n], |o: &mut Vec<u8>| k::f32_to_u8_row(&fsrc[..n], o));
    // The FUSION, against the two-kernel sequence it replaces. Not a tier A/B:
    // both arms run the shipped dispatch. This measures whether widening the
    // #[arcane] region (and eliminating a 120 KB write+read round trip through
    // cache) is worth it — the structural lever, not a kernel rewrite.
    suite.compare("u8_to_f32 + premultiply", |g| {
        g.throughput(Throughput::Bytes(n as u64));
        g.bench("fused", move |b| {
            b.with_input(move || vec![0f32; n]).run(move |mut o| {
                k::u8_to_f32_premultiply_row(&u8src2[..n], &mut o);
                o
            })
        });
        g.bench("sequence", move |b| {
            b.with_input(move || vec![0f32; n]).run(move |mut o| {
                k::u8_to_f32_row(&u8src3[..n], &mut o);
                k::premultiply_alpha_row(&mut o);
                o
            })
        });
    });

    // The sRGB fusion, against the three-pass sequence it replaces. sRGB is the
    // default transfer function for web images, so this is the highest-traffic
    // pairing in the crate.
    let u8src4 = u8src;
    let u8src5 = u8src;
    suite.compare("srgb_u8_to_linear + premultiply", |g| {
        g.throughput(Throughput::Bytes(n as u64));
        g.bench("fused", move |b| {
            b.with_input(move || vec![0f32; n]).run(move |mut o| {
                k::srgb_u8_to_linear_premultiply_f32(&u8src4[..n], &mut o);
                o
            })
        });
        g.bench("sequence", move |b| {
            b.with_input(move || vec![0f32; n]).run(move |mut o| {
                k::srgb_u8_to_linear_f32(&u8src5[..n], &mut o, 4, true);
                k::premultiply_alpha_row(&mut o);
                o
            })
        });
    });

    ab_inplace!("premultiply_alpha_row", fsrc[..n], k::premultiply_alpha_row);
    // 64x the row size: the #[arcane] boundary is a per-CALL cost, so at this
    // size it amortizes to nothing and the ratio reflects the BODY alone.
    // That is what decides whether a hand-written body is worth keeping, which
    // a single-row measurement cannot answer when the effect is ~8% and this
    // host's run-to-run drift is comparable.
    let big: &'static [f32] = Box::leak((0..n * 64).map(|i| (i % 251) as f32 / 251.0).collect::<Vec<_>>().into_boxed_slice());
    suite.compare("premultiply_alpha_row/64rows", |g| {
        g.throughput(Throughput::Bytes((n * 64) as u64));
        for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
            g.bench(arm, move |b| {
                b.with_input(move || { set_simd(simd); big.to_vec() })
                    .run(move |mut r| { k::premultiply_alpha_row(&mut r); r })
            });
        }
    });
    ab_inplace!("unpremultiply_alpha_row", fsrc[..n], k::unpremultiply_alpha_row);
    ab_out!("premultiply_u8_row", vec![0u8; n], |o: &mut Vec<u8>| k::premultiply_u8_row(&u8src[..n], o));
    ab_inplace!("unpremultiply_u8_row", u8src[..n], k::unpremultiply_u8_row);

    // ── The convolutions. A 1920 -> 960 downscale (the common web case) with
    // Lanczos3, RGBA. These run once per output row of every resize, so they
    // dominate; the per-pixel kernels above are the cheap part.
    use zenresize::filter::{Filter, InterpolationDetails};
    use zenresize::weights::{F32WeightTable, I16WeightTable};
    const OUT_W: usize = 960;
    let det = InterpolationDetails::create(Filter::Lanczos);
    let wf: &'static F32WeightTable =
        Box::leak(Box::new(F32WeightTable::new(W as u32, OUT_W as u32, &det)));
    let wi: &'static I16WeightTable =
        Box::leak(Box::new(I16WeightTable::new(W as u32, OUT_W as u32, &det)));
    let on = OUT_W * 4;

    // The 4ch H kernels loop to `max_taps` and rely on zero-padded weights, so
    // they read up to `groups4 * 16` elements past the row — exactly what
    // streaming.rs allocates (`in_row_len + h_padding_i16`, zeroed). Match that
    // contract here; without it the bench panics rather than measuring.
    // groups4 is private; PAD (512) covers groups4*16 for these sizes with margin.
    let pad = PAD;
    let i16src: &'static [i16] = Box::leak(
        (0..n + pad).map(|i| if i < n { (i % 4095) as i16 } else { 0 }).collect::<Vec<_>>().into_boxed_slice(),
    );
    let f16src: &'static [u16] = Box::leak(
        (0..n + PAD).map(|i| if i < n { (i % 251) as u16 + 0x3800 } else { 0 }).collect::<Vec<_>>().into_boxed_slice(),
    );

    ab!("filter_h_row_f32", vec![0f32; on], |o: &mut _| k::filter_h_row_f32(fsrc, o, wf, 4));
    ab!("filter_h_u8_i16", vec![0u8; on], |o: &mut _| k::filter_h_u8_i16(u8src, o, wi, 4));
    ab!("filter_h_u8_to_i16", vec![0i16; on], |o: &mut _| k::filter_h_u8_to_i16(u8src, o, wi, 4));
    ab!("filter_h_i16_i16", vec![0i16; on], |o: &mut _| k::filter_h_i16_i16(i16src, o, wi, 4));
    ab!("filter_h_row_f32_to_f16", vec![0u16; on], |o: &mut _| k::filter_h_row_f32_to_f16(fsrc, o, wf, 4));
    ab!(
        "filter_h_u8_i16_4rows",
        (vec![0u8; on], vec![0u8; on], vec![0u8; on], vec![0u8; on]),
        |o: &mut (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)| k::filter_h_u8_i16_4rows(
            u8src, u8src, u8src, u8src, &mut o.0, &mut o.1, &mut o.2, &mut o.3, wi
        )
    );
    ab!(
        "filter_h_u8_to_i16_4rows",
        (vec![0i16; on], vec![0i16; on], vec![0i16; on], vec![0i16; on]),
        |o: &mut (Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>)| k::filter_h_u8_to_i16_4rows(
            u8src, u8src, u8src, u8src, &mut o.0, &mut o.1, &mut o.2, &mut o.3, wi
        )
    );

    // V-filters: a 6-tap Lanczos3 window over full-width rows.
    let vw_f32: &'static [f32] = Box::leak(vec![0.16f32; 6].into_boxed_slice());
    let vw_i16: &'static [i16] = Box::leak(vec![1200i16; 6].into_boxed_slice());
    // `rows` is built in `with_input` too: the original built it inside the
    // timed body, so every V-filter number also charged a 6-element Vec alloc.
    ab!(
        "filter_v_row_f32",
        ((0..6).map(|_| &fsrc[..n]).collect::<Vec<&[f32]>>(), vec![0f32; n]),
        |o: &mut (Vec<&[f32]>, Vec<f32>)| k::filter_v_row_f32(&o.0, &mut o.1, vw_f32)
    );
    ab!(
        "filter_v_row_u8_i16",
        ((0..6).map(|_| &u8src[..n]).collect::<Vec<&[u8]>>(), vec![0u8; n]),
        |o: &mut (Vec<&[u8]>, Vec<u8>)| k::filter_v_row_u8_i16(&o.0, &mut o.1, vw_i16)
    );
    ab!(
        "filter_v_row_i16",
        ((0..6).map(|_| &i16src[..n]).collect::<Vec<&[i16]>>(), vec![0i16; n]),
        |o: &mut (Vec<&[i16]>, Vec<i16>)| k::filter_v_row_i16(&o.0, &mut o.1, vw_i16)
    );
    ab!(
        "filter_v_row_f16",
        ((0..6).map(|_| &f16src[..n]).collect::<Vec<&[u16]>>(), vec![0f32; n]),
        |o: &mut (Vec<&[u16]>, Vec<f32>)| k::filter_v_row_f16(&o.0, &mut o.1, vw_f32)
    );

    // ── f16 conversions
    ab!("f32_to_f16_row", vec![0u16; n], |o: &mut _| k::f32_to_f16_row(&fsrc[..n], o));
    ab!("f16_to_f32_row", vec![0f32; n], |o: &mut _| k::f16_to_f32_row(&f16src[..n], o));

    // ── colour / transfer
    ab!("srgb_u8_to_linear_f32", vec![0f32; n], |o: &mut _| k::srgb_u8_to_linear_f32(&u8src[..n], o, 4, true));
    ab!("linear_f32_to_srgb_u8", vec![0u8; n], |o: &mut _| k::linear_f32_to_srgb_u8(&fsrc[..n], o, 4, true));
    ab!("srgb_to_linear_row", fsrc[..n].to_vec(), |r: &mut _| k::srgb_to_linear_row(r, 4, true));
    ab!("srgb_from_linear_row", fsrc[..n].to_vec(), |r: &mut _| k::srgb_from_linear_row(r, 4, true));
    ab!("pq_to_linear_row", fsrc[..n].to_vec(), |r: &mut _| k::pq_to_linear_row(r, 4, true));
    ab!("hlg_to_linear_row", fsrc[..n].to_vec(), |r: &mut _| k::hlg_to_linear_row(r, 4, true));

    set_simd(true);
}

zenbench::main!(bench_kernels);

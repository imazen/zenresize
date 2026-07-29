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

    macro_rules! ab {
        ($name:expr, $body:expr) => {
            suite.compare($name, |g| {
                g.throughput(Throughput::Bytes(n as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        b.with_input(move || set_simd(simd)).run(move |_| $body)
                    });
                }
            });
        };
    }

    ab!("u8_to_f32_row", { let mut o = vec![0f32; n]; k::u8_to_f32_row(&u8src[..n], &mut o); o });
    ab!("f32_to_u8_row", { let mut o = vec![0u8; n]; k::f32_to_u8_row(&fsrc[..n], &mut o); o });
    ab!("premultiply_alpha_row", { let mut r = fsrc[..n].to_vec(); k::premultiply_alpha_row(&mut r); r });
    ab!("unpremultiply_alpha_row", { let mut r = fsrc[..n].to_vec(); k::unpremultiply_alpha_row(&mut r); r });
    ab!("premultiply_u8_row", { let mut o = vec![0u8; n]; k::premultiply_u8_row(&u8src[..n], &mut o); o });
    ab!("unpremultiply_u8_row", { let mut r = u8src[..n].to_vec(); k::unpremultiply_u8_row(&mut r); r });

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

    ab!("filter_h_row_f32", { let mut o = vec![0f32; on]; k::filter_h_row_f32(fsrc, &mut o, wf, 4); o });
    ab!("filter_h_u8_i16", { let mut o = vec![0u8; on]; k::filter_h_u8_i16(u8src, &mut o, wi, 4); o });
    ab!("filter_h_u8_to_i16", { let mut o = vec![0i16; on]; k::filter_h_u8_to_i16(u8src, &mut o, wi, 4); o });
    ab!("filter_h_i16_i16", { let mut o = vec![0i16; on]; k::filter_h_i16_i16(i16src, &mut o, wi, 4); o });
    ab!("filter_h_row_f32_to_f16", { let mut o = vec![0u16; on]; k::filter_h_row_f32_to_f16(fsrc, &mut o, wf, 4); o });
    ab!("filter_h_u8_i16_4rows", {
        let (mut o0, mut o1, mut o2, mut o3) = (vec![0u8; on], vec![0u8; on], vec![0u8; on], vec![0u8; on]);
        k::filter_h_u8_i16_4rows(u8src, u8src, u8src, u8src, &mut o0, &mut o1, &mut o2, &mut o3, wi); o0
    });
    ab!("filter_h_u8_to_i16_4rows", {
        let (mut o0, mut o1, mut o2, mut o3) = (vec![0i16; on], vec![0i16; on], vec![0i16; on], vec![0i16; on]);
        k::filter_h_u8_to_i16_4rows(u8src, u8src, u8src, u8src, &mut o0, &mut o1, &mut o2, &mut o3, wi); o0
    });

    // V-filters: a 6-tap Lanczos3 window over full-width rows.
    let vw_f32: &'static [f32] = Box::leak(vec![0.16f32; 6].into_boxed_slice());
    let vw_i16: &'static [i16] = Box::leak(vec![1200i16; 6].into_boxed_slice());
    ab!("filter_v_row_f32", {
        let rows: Vec<&[f32]> = (0..6).map(|_| &fsrc[..n]).collect();
        let mut o = vec![0f32; n]; k::filter_v_row_f32(&rows, &mut o, vw_f32); o
    });
    ab!("filter_v_row_u8_i16", {
        let rows: Vec<&[u8]> = (0..6).map(|_| &u8src[..n]).collect();
        let mut o = vec![0u8; n]; k::filter_v_row_u8_i16(&rows, &mut o, vw_i16); o
    });
    ab!("filter_v_row_i16", {
        let rows: Vec<&[i16]> = (0..6).map(|_| &i16src[..n]).collect();
        let mut o = vec![0i16; n]; k::filter_v_row_i16(&rows, &mut o, vw_i16); o
    });
    ab!("filter_v_row_f16", {
        let rows: Vec<&[u16]> = (0..6).map(|_| &f16src[..n]).collect();
        let mut o = vec![0f32; n]; k::filter_v_row_f16(&rows, &mut o, vw_f32); o
    });

    // ── f16 conversions
    ab!("f32_to_f16_row", { let mut o = vec![0u16; n]; k::f32_to_f16_row(&fsrc[..n], &mut o); o });
    ab!("f16_to_f32_row", { let mut o = vec![0f32; n]; k::f16_to_f32_row(&f16src[..n], &mut o); o });

    // ── colour / transfer
    ab!("srgb_u8_to_linear_f32", { let mut o = vec![0f32; n]; k::srgb_u8_to_linear_f32(&u8src[..n], &mut o, 4, true); o });
    ab!("linear_f32_to_srgb_u8", { let mut o = vec![0u8; n]; k::linear_f32_to_srgb_u8(&fsrc[..n], &mut o, 4, true); o });
    ab!("srgb_to_linear_row", { let mut r = fsrc[..n].to_vec(); k::srgb_to_linear_row(&mut r, 4, true); r });
    ab!("srgb_from_linear_row", { let mut r = fsrc[..n].to_vec(); k::srgb_from_linear_row(&mut r, 4, true); r });
    ab!("pq_to_linear_row", { let mut r = fsrc[..n].to_vec(); k::pq_to_linear_row(&mut r, 4, true); r });
    ab!("hlg_to_linear_row", { let mut r = fsrc[..n].to_vec(); k::hlg_to_linear_row(&mut r, 4, true); r });

    set_simd(true);
}

zenbench::main!(bench_kernels);

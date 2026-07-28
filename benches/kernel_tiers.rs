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
    let u8src: &'static [u8] = Box::leak((0..n).map(|i| (i % 251) as u8).collect::<Vec<_>>().into_boxed_slice());
    let fsrc: &'static [f32] = Box::leak((0..n).map(|i| (i % 251) as f32 / 251.0).collect::<Vec<_>>().into_boxed_slice());

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

    ab!("u8_to_f32_row", { let mut o = vec![0f32; n]; k::u8_to_f32_row(u8src, &mut o); o });
    ab!("f32_to_u8_row", { let mut o = vec![0u8; n]; k::f32_to_u8_row(fsrc, &mut o); o });
    ab!("premultiply_alpha_row", { let mut r = fsrc.to_vec(); k::premultiply_alpha_row(&mut r); r });
    ab!("unpremultiply_alpha_row", { let mut r = fsrc.to_vec(); k::unpremultiply_alpha_row(&mut r); r });
    ab!("premultiply_u8_row", { let mut o = vec![0u8; n]; k::premultiply_u8_row(u8src, &mut o); o });
    ab!("unpremultiply_u8_row", { let mut r = u8src.to_vec(); k::unpremultiply_u8_row(&mut r); r });

    set_simd(true);
}

zenbench::main!(bench_kernels);

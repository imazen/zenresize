//! End-to-end differential of the baseline and candidate builds on RGB resizes.
//!
//! Every RGB resize that reaches the F32 horizontal filter must be bit-identical
//! between the two builds within one dispatch tier and process. f32 I/O is
//! compared on raw bits (NaN class-equal; payload agreement counted), u8/u16 on
//! exact values. Covers fullframe `Resizer` (incl. post_blur / post_sharpen,
//! which run blur_f32 -> filter_h_row_f32(.., 3) on a padded row), streaming
//! `StreamingResize`, crop, padding, input stride, output stride
//! (`resize_into`), lobe-ratio and kernel-width-scale weight variants, every
//! filter, RGB8 (linear and sRGB-space), RGB16 and RGBF32_LINEAR.

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

#[path = "../common.rs"]
mod common;
use common::{Rng, Scope, same_class};

const SEED: u64 = 0xE2E_5EED;

#[derive(Clone, Debug)]
struct Case {
    iw: u32,
    ih: u32,
    ow: u32,
    oh: u32,
    fi: usize,
    fmt: u32, // 0 RGB8 linear, 1 RGB8 srgb(), 2 RGB16, 3 RGBF32_LINEAR
    crop: Option<(u32, u32, u32, u32)>,
    pad: Option<(u32, u32, u32, u32)>,
    in_stride_extra: usize,
    out_stride_extra: usize,
    blur: f32,
    sharpen: f32,
    lobe: Option<f32>,
    kws: Option<f64>,
    fmode: u32,
}

/// (stream name, is_f32, values as u32) per output stream.
type Outs = Vec<(&'static str, bool, Vec<u32>)>;

macro_rules! run_e2e {
    ($z:ident, $c:expr, $u8in:expr, $u16in:expr, $f32in:expr) => {{
        use $z as z;
        let c: &Case = $c;
        let fmt = match c.fmt {
            0 | 1 => z::PixelDescriptor::RGB8_SRGB,
            2 => z::PixelDescriptor::RGB16_SRGB,
            _ => z::PixelDescriptor::RGBF32_LINEAR,
        };
        let in_stride = c.iw as usize * 3 + c.in_stride_extra;
        let builder = || {
            let mut b = z::ResizeConfig::builder(c.iw, c.ih, c.ow, c.oh)
                .filter(z::Filter::all()[c.fi])
                .format(fmt);
            if c.fmt == 1 {
                b = b.srgb();
            } else if c.fmt == 0 {
                b = b.linear();
            }
            if let Some((x, y, w, h)) = c.crop {
                b = b.crop(x, y, w, h);
            }
            if let Some((t, r, bt, l)) = c.pad {
                b = b.padding(t, r, bt, l).padding_color([0.25, 0.5, 0.75, 1.0]);
            }
            if c.in_stride_extra > 0 {
                b = b.in_stride(in_stride);
            }
            b
        };
        let mut b = builder();
        if c.blur > 0.0 {
            b = b.post_blur(c.blur);
        }
        if c.sharpen > 0.0 {
            b = b.post_sharpen(c.sharpen);
        }
        if let Some(r) = c.lobe {
            b = b.lobe_ratio(z::LobeRatio::Exact(r));
        }
        if let Some(k) = c.kws {
            b = b.kernel_width_scale(k);
        }
        let cfg = b.build();
        let row_len = c.iw as usize * 3;
        let mut outs: Outs = Vec::new();
        match c.fmt {
            0 | 1 => {
                let input: &[u8] = $u8in;
                let ff = z::Resizer::new(&cfg).resize(input);
                outs.push(("ff", false, ff.iter().map(|&v| v as u32).collect()));
                if c.out_stride_extra > 0 {
                    let ostride = cfg.total_output_width() as usize * 3 + c.out_stride_extra;
                    let th = cfg.total_output_height() as usize;
                    let cfg2 = builder().out_stride(ostride).build();
                    let mut dst = vec![0xA5u8; ostride * th];
                    z::Resizer::new(&cfg2).resize_into(input, &mut dst);
                    outs.push((
                        "ff_into_ostride",
                        false,
                        dst.iter().map(|&v| v as u32).collect(),
                    ));
                }
                let mut s = z::StreamingResize::new(&cfg);
                let mut acc = Vec::new();
                for y in 0..c.ih as usize {
                    s.push_row(&input[y * in_stride..y * in_stride + row_len])
                        .unwrap();
                    while let Some(r) = s.next_output_row() {
                        acc.extend_from_slice(r);
                    }
                }
                s.finish();
                while let Some(r) = s.next_output_row() {
                    acc.extend_from_slice(r);
                }
                outs.push(("st", false, acc.iter().map(|&v| v as u32).collect()));
            }
            2 => {
                let input: &[u16] = $u16in;
                let ff = z::Resizer::new(&cfg).resize_u16(input);
                outs.push(("ff", false, ff.iter().map(|&v| v as u32).collect()));
                let mut s = z::StreamingResize::new(&cfg);
                let mut acc = Vec::new();
                for y in 0..c.ih as usize {
                    s.push_row_u16(&input[y * in_stride..y * in_stride + row_len])
                        .unwrap();
                    while let Some(r) = s.next_output_row_u16() {
                        acc.extend_from_slice(r);
                    }
                }
                s.finish();
                while let Some(r) = s.next_output_row_u16() {
                    acc.extend_from_slice(r);
                }
                outs.push(("st", false, acc.iter().map(|&v| v as u32).collect()));
            }
            _ => {
                let input: &[f32] = $f32in;
                let ff = z::Resizer::new(&cfg).resize_f32(input);
                outs.push(("ff", true, ff.iter().map(|v| v.to_bits()).collect()));
                let mut s = z::StreamingResize::new(&cfg);
                let mut acc = Vec::new();
                for y in 0..c.ih as usize {
                    s.push_row_f32(&input[y * in_stride..y * in_stride + row_len])
                        .unwrap();
                    while let Some(r) = s.next_output_row_f32() {
                        acc.extend_from_slice(r);
                    }
                }
                s.finish();
                while let Some(r) = s.next_output_row_f32() {
                    acc.extend_from_slice(r);
                }
                outs.push(("st", true, acc.iter().map(|v| v.to_bits()).collect()));
            }
        }
        outs
    }};
}

fn gen_f32(rng: &mut Rng, mode: u32) -> f32 {
    match mode {
        0 => rng.unit(),
        1 => rng.unit() * 5.0 - 2.0, // HDR / negative wide-gamut
        2 => match rng.below(6) {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(rng.below(0x0080_0000) as u32),
            3 => -f32::from_bits(rng.below(0x0080_0000) as u32),
            _ => rng.unit(),
        },
        _ => match rng.below(40) {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => 3.0e38,
            _ => rng.unit(),
        },
    }
}

fn random_case(rng: &mut Rng, nf: usize, big: bool) -> Case {
    let lim = if big { 1400 } else { 96 };
    let iw = 1 + rng.below(lim) as u32;
    let ih = 1 + rng.below(if big { 64 } else { 48 }) as u32;
    let ow = 1 + rng.below(lim) as u32;
    let oh = 1 + rng.below(if big { 64 } else { 48 }) as u32;
    let crop = if rng.below(4) == 0 {
        let w = 1 + rng.below(iw as u64) as u32;
        let h = 1 + rng.below(ih as u64) as u32;
        let x = rng.below((iw - w + 1) as u64) as u32;
        let y = rng.below((ih - h + 1) as u64) as u32;
        Some((x, y, w, h))
    } else {
        None
    };
    let pad = if rng.below(4) == 0 {
        Some((
            rng.below(4) as u32,
            rng.below(4) as u32,
            rng.below(4) as u32,
            rng.below(4) as u32,
        ))
    } else {
        None
    };
    Case {
        iw,
        ih,
        ow,
        oh,
        fi: rng.below(nf as u64) as usize,
        fmt: rng.below(4) as u32,
        crop,
        pad,
        in_stride_extra: if rng.below(3) == 0 {
            1 + rng.below(7) as usize
        } else {
            0
        },
        out_stride_extra: if rng.below(3) == 0 {
            1 + rng.below(7) as usize
        } else {
            0
        },
        blur: if rng.below(5) == 0 {
            0.3 + rng.unit() * 3.0
        } else {
            0.0
        },
        sharpen: if rng.below(6) == 0 {
            0.2 + rng.unit()
        } else {
            0.0
        },
        lobe: match rng.below(6) {
            0 => Some(0.0),
            1 => Some(0.25),
            _ => None,
        },
        kws: if rng.below(6) == 0 {
            Some(0.6 + rng.unit() as f64 * 2.0)
        } else {
            None
        },
        fmode: rng.below(4) as u32,
    }
}

fn main() {
    let scope = common::parse_scope("e2e_diff");
    let nf = zr_new::Filter::all().len();
    assert_eq!(nf, zr_old::Filter::all().len(), "filter lists differ");
    let (n_small, n_big) = match scope {
        Scope::Quick => (200usize, 12usize),
        Scope::Full => (1500, 120),
    };
    println!(
        "e2e_diff scope={} features={} seed={SEED:#x}: {n_small} cases up to 96 px wide + {n_big} up to 1400 px, \
         formats RGB8 linear / RGB8 srgb / RGB16 / RGBF32_LINEAR",
        scope.name(),
        common::features()
    );

    let (mut cases, mut streams, mut elems, mut nanp) = (0u64, 0u64, 0u64, 0u64);
    let mut perms = Vec::new();
    let report = for_each_token_permutation(CompileTimePolicy::Fail, |perm| {
        perms.push(perm.label.clone());
        let mut rng = Rng(SEED);
        let (mut p_cases, mut p_streams, mut p_elems, mut p_nanp) = (0u64, 0u64, 0u64, 0u64);
        let mut fmt_counts = [0u64; 4];
        for i in 0..(n_small + n_big) {
            let c = random_case(&mut rng, nf, i >= n_small);
            let in_stride = c.iw as usize * 3 + c.in_stride_extra;
            let n_in = in_stride * c.ih as usize;
            let in_u8: Vec<u8> = (0..n_in).map(|_| rng.next() as u8).collect();
            let in_u16: Vec<u16> = (0..n_in).map(|_| rng.next() as u16).collect();
            let in_f32: Vec<f32> = (0..n_in).map(|_| gen_f32(&mut rng, c.fmode)).collect();
            let a = run_e2e!(zr_new, &c, &in_u8, &in_u16, &in_f32);
            let b = run_e2e!(zr_old, &c, &in_u8, &in_u16, &in_f32);
            assert_eq!(a.len(), b.len(), "stream count differs {c:?}");
            for ((na, fa, va), (nb, _, vb)) in a.iter().zip(b.iter()) {
                assert_eq!(na, nb);
                assert_eq!(va.len(), vb.len(), "length differs in {na} {c:?}");
                for (k, (&x, &y)) in va.iter().zip(vb.iter()).enumerate() {
                    if x == y {
                        continue;
                    }
                    if *fa && same_class(f32::from_bits(x), f32::from_bits(y)) {
                        p_nanp += 1;
                        continue;
                    }
                    panic!(
                        "E2E MISMATCH perm[{}] stream {na} elem {k}: new={x:#010x} old={y:#010x} case={c:?}",
                        perm.label
                    );
                }
                p_streams += 1;
                p_elems += va.len() as u64;
            }
            fmt_counts[c.fmt as usize] += 1;
            p_cases += 1;
        }
        eprintln!(
            "perm [{}]: cases={p_cases} streams={p_streams} elems={p_elems} nan_payload_diffs={p_nanp} \
             fmt_counts(rgb8lin,rgb8srgb,rgb16,rgbf32)={fmt_counts:?}",
            perm.label
        );
        cases += p_cases;
        streams += p_streams;
        elems += p_elems;
        nanp += p_nanp;
    });
    assert!(report.permutations_run > 0, "no dispatch permutation ran");
    println!("e2e_diff OK scope={} permutations={perms:?}", scope.name());
    println!("report: {report}");
    println!("totals: cases={cases} streams={streams} elems={elems} nan_payload_diffs={nanp}");
}

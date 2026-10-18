//! Kernel-level differential for `filter_h_row_f32(.., channels = 3)`.
//!
//! Three-way comparison per call, under every archmage token permutation:
//!   new = candidate build (`zr_new`)
//!   old = baseline build (`zr_old`)
//!   ref = the baseline's scalar `filter_h_3ch` body, transcribed below
//!         (strict IEEE, compiled without target features)
//!
//! Input classes include signed zeros, subnormals, overflow-prone magnitudes,
//! inf and NaN (quiet, signalling, payload-carrying, negative). Physical input
//! tails (0..=7 extra elements, filled with garbage/NaN/inf/sNaN) and physical
//! output tails (0..=5 extra, sentinel) are swept independently of the logical
//! row. Every non-NaN output must match bit-for-bit; NaN must meet NaN (payload
//! agreement is counted, not required); every sentinel must survive.

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zr_new::filter::{Filter as FN, InterpolationDetails as IDN};
use zr_new::simd::__bench_kernels as kn;
use zr_new::weights::F32WeightTable as WN;
use zr_old::filter::{Filter as FO, InterpolationDetails as IDO};
use zr_old::simd::__bench_kernels as ko;
use zr_old::weights::F32WeightTable as WO;

#[path = "../common.rs"]
mod common;
use common::{Rng, Scope, same_class};

const SENTINEL: u32 = 0x5E17_1E11;
const SEED: u64 = 0x3C4_7A11_D00D;

const SPECIALS: &[u32] = &[
    0x0000_0000,
    0x8000_0000, // +-0
    0x0000_0001,
    0x8000_0001, // +-min subnormal
    0x007f_ffff,
    0x807f_ffff, // +-max subnormal
    0x0080_0000,
    0x8080_0000, // +-min normal
    0x3f7f_ffff,
    0x3f80_0000,
    0x3f80_0001, // 1-ulp, 1, 1+ulp
    0xbf7f_ffff,
    0xbf80_0000,
    0xbf80_0001,
    0x7f7f_ffff,
    0xff7f_ffff, // +-max finite
    0x7f80_0000,
    0xff80_0000, // +-inf
    0x7fc0_0000,
    0xffc0_0000, // +-qNaN
    0x7fa0_0001,
    0xffa0_0001, // +-sNaN with payload
    0x7fc1_2345, // qNaN with payload
];

fn gen_value(rng: &mut Rng, mode: u32) -> f32 {
    match mode {
        // ordinary bounded data
        0 => rng.unit() * 4.0 - 1.0,
        // random finite bit patterns over the whole exponent range
        1 => loop {
            let v = f32::from_bits(rng.next() as u32);
            if v.is_finite() {
                break v;
            }
        },
        // specials incl. inf/NaN
        2 => f32::from_bits(SPECIALS[rng.below(SPECIALS.len() as u64) as usize]),
        // sign-of-zero / subnormal stress, finite only
        3 => match rng.below(8) {
            0 => 0.0,
            1 => -0.0,
            2 => f32::from_bits(rng.below(0x0080_0000) as u32),
            3 => f32::from_bits(0x8000_0000 | rng.below(0x0080_0000) as u32),
            4 => f32::from_bits(0x0080_0000),
            5 => -f32::from_bits(0x0080_0000),
            _ => (rng.unit() - 0.5) * 1e-37,
        },
        // overflow-prone magnitudes, finite inputs (sums may overflow to inf / NaN)
        _ => (rng.unit() * 2.0 - 1.0) * 3.0e38,
    }
}

fn gen_tail(rng: &mut Rng, kind: u32) -> f32 {
    match kind {
        0 => rng.unit() * 1000.0 - 500.0,
        1 => f32::NAN,
        2 => f32::INFINITY,
        3 => f32::from_bits(0x7fa0_0001),
        _ => f32::NEG_INFINITY,
    }
}

/// Transcription of the baseline x86 v3 `filter_h_3ch` body (the same
/// arithmetic as `filter_h_row_f32_scalar` for channels == 3).
fn reference(input: &[f32], out_width: usize, weights: &WN) -> Vec<f32> {
    let mut out = vec![0.0f32; out_width * 3];
    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * 3;
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * 3;
            acc0 += input[in_offset] * weight;
            acc1 += input[in_offset + 1] * weight;
            acc2 += input[in_offset + 2] * weight;
        }
        out[out_offset] = acc0;
        out[out_offset + 1] = acc1;
        out[out_offset + 2] = acc2;
    }
    out
}

#[derive(Default, Debug, Clone)]
struct Stats {
    calls: u64,
    elems: u64,
    nan_elems: u64,
    nan_payload_diff_new_vs_ref: u64,
    nan_payload_diff_new_vs_old: u64,
    tables: u64,
    padded_last_tap_pixels: u64,
    zero_tap_pixels: u64,
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        self.calls += o.calls;
        self.elems += o.elems;
        self.nan_elems += o.nan_elems;
        self.nan_payload_diff_new_vs_ref += o.nan_payload_diff_new_vs_ref;
        self.nan_payload_diff_new_vs_old += o.nan_payload_diff_new_vs_old;
        self.tables += o.tables;
        self.padded_last_tap_pixels += o.padded_last_tap_pixels;
        self.zero_tap_pixels += o.zero_tap_pixels;
    }
}

struct Case<'a> {
    wn: &'a WN,
    wo: &'a WO,
    in_w: usize,
    out_w: usize,
    mode: u32,
    in_extra: usize,
    tail_kind: u32,
    out_extra: usize,
}

fn check_case(st: &mut Stats, rng: &mut Rng, c: &Case, label: &str, perm: &str) {
    let in_n = c.in_w * 3;
    let out_n = c.out_w * 3;
    let mut input: Vec<f32> = (0..in_n).map(|_| gen_value(rng, c.mode)).collect();
    for _ in 0..c.in_extra {
        input.push(gen_tail(rng, c.tail_kind));
    }
    let expect = reference(&input[..in_n], c.out_w, c.wn);

    let mut got_new = vec![f32::from_bits(SENTINEL); out_n + c.out_extra];
    let mut got_old = vec![f32::from_bits(SENTINEL); out_n + c.out_extra];
    kn::filter_h_row_f32(&input, &mut got_new, c.wn, 3);
    ko::filter_h_row_f32(&input, &mut got_old, c.wo, 3);

    // Which output pixels take the zero-tap branch or the physically padded
    // last-tap load (counted so the report states real coverage).
    for x in 0..c.out_w {
        let n = c.wn.weights(x).len();
        if n == 0 {
            st.zero_tap_pixels += 1;
            continue;
        }
        let off = (c.wn.left[x] as usize + n - 1) * 3;
        if off + 4 > input.len() {
            st.padded_last_tap_pixels += 1;
        }
    }

    for i in 0..out_n {
        let (n, o, r) = (got_new[i], got_old[i], expect[i]);
        if !same_class(n, r) || !same_class(n, o) {
            panic!(
                "MISMATCH {label} perm[{perm}] in_w={} out_w={} mode={} in_extra={} tail={} out_extra={} \
                 elem={i} (px {} ch {}): new={:#010x} old={:#010x} ref={:#010x}",
                c.in_w,
                c.out_w,
                c.mode,
                c.in_extra,
                c.tail_kind,
                c.out_extra,
                i / 3,
                i % 3,
                n.to_bits(),
                o.to_bits(),
                r.to_bits()
            );
        }
        if n.is_nan() {
            st.nan_elems += 1;
            st.nan_payload_diff_new_vs_ref += u64::from(n.to_bits() != r.to_bits());
            st.nan_payload_diff_new_vs_old += u64::from(n.to_bits() != o.to_bits());
        }
    }
    for i in out_n..out_n + c.out_extra {
        assert_eq!(
            got_new[i].to_bits(),
            SENTINEL,
            "NEW clobbered output tail +{} {label} perm[{perm}] in_w={} out_w={} out_extra={}",
            i - out_n,
            c.in_w,
            c.out_w,
            c.out_extra
        );
        assert_eq!(
            got_old[i].to_bits(),
            SENTINEL,
            "OLD clobbered output tail +{} {label} perm[{perm}]",
            i - out_n
        );
    }
    st.calls += 1;
    st.elems += out_n as u64;
}

const VARIANTS: [&str; 5] = [
    "plain",
    "with_blur(3.0)",
    "with_blur(0.4)",
    "with_lobe_ratio(0.0)",
    "with_lobe_ratio(0.3)",
];

fn tables(in_w: u32, out_w: u32, fi: usize, variant: usize) -> (WN, WO) {
    let (mut dn, mut dold) = (IDN::create(FN::all()[fi]), IDO::create(FO::all()[fi]));
    match variant {
        0 => {}
        1 => (dn, dold) = (dn.with_blur(3.0), dold.with_blur(3.0)),
        2 => (dn, dold) = (dn.with_blur(0.4), dold.with_blur(0.4)),
        3 => (dn, dold) = (dn.with_lobe_ratio(0.0), dold.with_lobe_ratio(0.0)),
        _ => (dn, dold) = (dn.with_lobe_ratio(0.3), dold.with_lobe_ratio(0.3)),
    }
    let wn = WN::new(in_w, out_w, &dn);
    let wo = WO::new(in_w, out_w, &dold);
    // Weight construction is shared code; a table mismatch would make the
    // old/new comparison meaningless, so prove the tables are identical.
    assert_eq!(wn.len(), wo.len(), "table length differs");
    for x in 0..wn.len() {
        assert_eq!(wn.left[x], wo.left[x], "table left differs at {x}");
        let (a, b) = (wn.weights(x), wo.weights(x));
        assert_eq!(a.len(), b.len(), "tap count differs at {x}");
        for (p, q) in a.iter().zip(b) {
            assert_eq!(p.to_bits(), q.to_bits(), "weight differs at {x}");
        }
    }
    (wn, wo)
}

fn main() {
    let scope = common::parse_scope("kernel_diff");
    let nf = FN::all().len();
    assert_eq!(nf, FO::all().len(), "filter lists differ");

    // Lanczos, CatmullRom, Jinc, Triangle, Box (indices into Filter::all()).
    let small_filters = [5usize, 12, 16, 21, 23];
    let (small_max, sweep_filters, random_cases): (u32, Vec<usize>, usize) = match scope {
        Scope::Quick => (16, small_filters.to_vec(), 60),
        Scope::Full => (40, (0..nf).collect(), 600),
    };
    let ratios: &[(u32, u32)] = &[
        (1, 1),
        (1, 7),
        (2, 1),
        (3, 2),
        (5, 13),
        (17, 16),
        (64, 1),
        (64, 3),
        (100, 33),
        (333, 1000),
        (1920, 960),
        (4000, 800),
        (1000, 4000),
        (7, 700),
        (4096, 3),
    ];
    println!(
        "kernel_diff scope={} features={} seed={SEED:#x}: (A) in_w,out_w in 1..={small_max} x {} filters; \
         (B) {} filters x {} weight variants x {} ratios x 5 input classes x 5 tail layouts; (C) {random_cases} random geometries",
        scope.name(),
        common::features(),
        small_filters.len(),
        sweep_filters.len(),
        VARIANTS.len(),
        ratios.len()
    );

    let mut total = Stats::default();
    let mut perms = Vec::new();
    let report = for_each_token_permutation(CompileTimePolicy::Fail, |perm| {
        let label = perm.label.clone();
        perms.push(label.clone());
        let mut st = Stats::default();
        let mut rng = Rng(SEED);

        for in_w in 1..=small_max {
            for out_w in 1..=small_max {
                for (k, &fi) in small_filters.iter().enumerate() {
                    let (wn, wo) = tables(in_w, out_w, fi, 0);
                    st.tables += 1;
                    let mode = (in_w + out_w + k as u32) % 5;
                    let mut c = Case {
                        wn: &wn,
                        wo: &wo,
                        in_w: in_w as usize,
                        out_w: out_w as usize,
                        mode,
                        in_extra: ((in_w * 7 + out_w) % 8) as usize,
                        tail_kind: (in_w + 3 * out_w) % 5,
                        out_extra: ((in_w + out_w * 5) % 6) as usize,
                    };
                    check_case(&mut st, &mut rng, &c, "small", &label);
                    // and always the physically exact layout (padded last tap)
                    (c.in_extra, c.tail_kind, c.out_extra) = (0, 0, 0);
                    check_case(&mut st, &mut rng, &c, "small-exact", &label);
                }
            }
        }

        for &fi in &sweep_filters {
            for variant in 0..VARIANTS.len() {
                for &(iw, ow) in ratios {
                    let (wn, wo) = tables(iw, ow, fi, variant);
                    st.tables += 1;
                    for mode in 0..5 {
                        for &(in_extra, tail_kind, out_extra) in &[
                            (0usize, 0u32, 0usize),
                            (1, 1, 1),
                            (2, 2, 3),
                            (3, 3, 5),
                            (64, 0, 64),
                        ] {
                            let c = Case {
                                wn: &wn,
                                wo: &wo,
                                in_w: iw as usize,
                                out_w: ow as usize,
                                mode,
                                in_extra,
                                tail_kind,
                                out_extra,
                            };
                            check_case(&mut st, &mut rng, &c, "ratios", &label);
                        }
                    }
                }
            }
        }

        for _ in 0..random_cases {
            let iw = 1 + rng.below(3000) as u32;
            let ow = 1 + rng.below(3000) as u32;
            let fi = rng.below(nf as u64) as usize;
            let variant = rng.below(VARIANTS.len() as u64) as usize;
            let (wn, wo) = tables(iw, ow, fi, variant);
            st.tables += 1;
            let c = Case {
                wn: &wn,
                wo: &wo,
                in_w: iw as usize,
                out_w: ow as usize,
                mode: rng.below(5) as u32,
                in_extra: rng.below(8) as usize,
                tail_kind: rng.below(5) as u32,
                out_extra: rng.below(6) as usize,
            };
            check_case(&mut st, &mut rng, &c, "random", &label);
        }

        eprintln!("perm [{label}]: {st:?}");
        total.add(&st);
    });
    assert!(report.permutations_run > 0, "no dispatch permutation ran");
    println!(
        "kernel_diff OK scope={} permutations={perms:?}",
        scope.name()
    );
    println!("report: {report}");
    println!("totals: {total:?}");
}

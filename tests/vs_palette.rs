//! Compare zenresize transfer functions against the `palette` crate (sRGB)
//! and f64 reference implementations (BT.709, PQ, HLG).

use zenresize::{Bt709, Hlg, NoTransfer, Pq, Srgb, TransferCurve};

// ============================================================================
// f64 reference implementations — same formulas, double precision
// ============================================================================

mod reference {
    /// sRGB EOTF (IEC 61966-2-1)
    pub fn srgb_to_linear(v: f64) -> f64 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    pub fn srgb_from_linear(v: f64) -> f64 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    /// BT.709 OETF (ITU-R BT.709) — exact constants for C0/C1 continuity
    const BT709_ALPHA: f64 = 0.09929682680944;
    const BT709_BETA: f64 = 0.018053968510807;

    pub fn bt709_to_linear(v: f64) -> f64 {
        if v < 4.5 * BT709_BETA {
            v / 4.5
        } else {
            ((v + BT709_ALPHA) / (1.0 + BT709_ALPHA)).powf(1.0 / 0.45)
        }
    }

    pub fn bt709_from_linear(v: f64) -> f64 {
        if v < BT709_BETA {
            4.5 * v
        } else {
            (1.0 + BT709_ALPHA) * v.powf(0.45) - BT709_ALPHA
        }
    }

    /// PQ / SMPTE ST 2084
    const PQ_M1: f64 = 0.1593017578125;
    const PQ_M2: f64 = 78.84375;
    const PQ_C1: f64 = 0.8359375;
    const PQ_C2: f64 = 18.8515625;
    const PQ_C3: f64 = 18.6875;

    pub fn pq_to_linear(v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let vp = v.powf(1.0 / PQ_M2);
        let num = (vp - PQ_C1).max(0.0);
        let den = PQ_C2 - PQ_C3 * vp;
        if den <= 0.0 {
            return 1.0;
        }
        (num / den).powf(1.0 / PQ_M1)
    }

    pub fn pq_from_linear(v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let vp = v.powf(PQ_M1);
        let num = PQ_C1 + PQ_C2 * vp;
        let den = 1.0 + PQ_C3 * vp;
        (num / den).powf(PQ_M2)
    }

    /// HLG / ARIB STD-B67
    const HLG_A: f64 = 0.17883277;
    const HLG_B: f64 = 0.28466892; // 1 - 4*A
    const HLG_C: f64 = 0.55991073; // 0.5 - A*ln(4*A)

    pub fn hlg_to_linear(v: f64) -> f64 {
        if v <= 0.0 {
            0.0
        } else if v <= 0.5 {
            (v * v) / 3.0
        } else {
            (((v - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
        }
    }

    pub fn hlg_from_linear(v: f64) -> f64 {
        if v <= 0.0 {
            0.0
        } else if v <= 1.0 / 12.0 {
            (3.0 * v).sqrt()
        } else {
            HLG_A * (12.0 * v - HLG_B).ln() + HLG_C
        }
    }
}

// ============================================================================
// palette sRGB comparison
// ============================================================================

#[test]
fn srgb_to_linear_vs_palette() {
    use palette::encoding::{IntoLinear, Srgb as PaletteSrgb};

    let tf = Srgb;
    let mut max_diff: f32 = 0.0;
    let mut worst_input: f32 = 0.0;

    // Test all u8 values (normalized to f32)
    for i in 0..=255u8 {
        let v = i as f32 / 255.0;
        let ours = tf.to_linear(v);
        let palette_val: f32 = <PaletteSrgb as IntoLinear<f32, f32>>::into_linear(v);

        let diff = (ours - palette_val).abs();
        if diff > max_diff {
            max_diff = diff;
            worst_input = v;
        }
        assert!(
            diff < 1e-6,
            "sRGB to_linear({v}): ours={ours}, palette={palette_val}, diff={diff}"
        );
    }
    eprintln!("sRGB to_linear vs palette: max_diff={max_diff} at {worst_input}");
}

#[test]
fn srgb_from_linear_vs_palette() {
    use palette::encoding::{FromLinear, Srgb as PaletteSrgb};

    let tf = Srgb;
    let mut max_diff: f32 = 0.0;
    let mut worst_input: f32 = 0.0;

    // Test fine grid
    for i in 0..=10000 {
        let v = i as f32 / 10000.0;
        let ours = tf.from_linear(v);
        let palette_val: f32 = <PaletteSrgb as FromLinear<f32, f32>>::from_linear(v);

        let diff = (ours - palette_val).abs();
        if diff > max_diff {
            max_diff = diff;
            worst_input = v;
        }
        assert!(
            diff < 1e-6,
            "sRGB from_linear({v}): ours={ours}, palette={palette_val}, diff={diff}"
        );
    }
    eprintln!("sRGB from_linear vs palette: max_diff={max_diff} at {worst_input}");
}

#[test]
fn srgb_u8_to_linear_vs_palette_lut() {
    use palette::encoding::{IntoLinear, Srgb as PaletteSrgb};

    let tf = Srgb;
    let luts = tf.build_luts();

    // Our u8→f32 LUT (linear-srgb crate) vs palette's (fast_srgb8 crate).
    // Both are precomputed tables with different generation methods; diffs up to
    // ~2e-6 are expected and well within 1/2 ULP at these magnitudes.
    let mut max_diff: f32 = 0.0;
    for i in 0..=255u8 {
        let src = [i];
        let mut dst = [0.0f32];
        tf.u8_to_linear_f32(&src, &mut dst, &luts, 1, false, false);
        let palette_val: f32 = <PaletteSrgb as IntoLinear<f32, u8>>::into_linear(i);

        let diff = (dst[0] - palette_val).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 5e-6,
            "sRGB LUT u8({i}): ours={}, palette={palette_val}, diff={diff}",
            dst[0]
        );
    }
    eprintln!("sRGB u8->f32 LUT: max diff vs palette = {max_diff:.2e}");
}

#[test]
fn srgb_linear_to_u8_vs_palette_lut() {
    use palette::encoding::{FromLinear, Srgb as PaletteSrgb};

    let tf = Srgb;
    let luts = tf.build_luts();

    // Test the LUT-based batch path against palette
    for i in 0..=10000 {
        let linear = i as f32 / 10000.0;
        let src = [linear];
        let mut dst = [0u8];
        tf.linear_f32_to_u8(&src, &mut dst, &luts, 1, false, false);
        let palette_val: u8 = <PaletteSrgb as FromLinear<f32, u8>>::from_linear(linear);

        let diff = (dst[0] as i16 - palette_val as i16).unsigned_abs();
        assert!(
            diff <= 1,
            "sRGB LUT linear({linear})->u8: ours={}, palette={palette_val}",
            dst[0]
        );
    }
}

// ============================================================================
// sRGB vs f64 reference
// ============================================================================

#[test]
fn srgb_to_linear_vs_f64() {
    let tf = Srgb;
    let mut max_err: f64 = 0.0;

    for i in 0..=65535u32 {
        let v = i as f64 / 65535.0;
        let ours = tf.to_linear(v as f32) as f64;
        let ref_val = reference::srgb_to_linear(v);

        let err = (ours - ref_val).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < 5e-7,
            "sRGB to_linear f64 ref at {v:.6}: ours={ours}, ref={ref_val}, err={err}"
        );
    }
    eprintln!("sRGB to_linear vs f64: max_err={max_err:.2e}");
}

#[test]
fn srgb_from_linear_vs_f64() {
    let tf = Srgb;
    let mut max_err: f64 = 0.0;

    for i in 0..=65535u32 {
        let v = i as f64 / 65535.0;
        let ours = tf.from_linear(v as f32) as f64;
        let ref_val = reference::srgb_from_linear(v);

        let err = (ours - ref_val).abs();
        if err > max_err {
            max_err = err;
        }
        assert!(
            err < 5e-7,
            "sRGB from_linear f64 ref at {v:.6}: ours={ours}, ref={ref_val}, err={err}"
        );
    }
    eprintln!("sRGB from_linear vs f64: max_err={max_err:.2e}");
}

// ============================================================================
// BT.709 vs f64 reference
// ============================================================================

#[test]
fn bt709_to_linear_vs_f64() {
    let tf = Bt709;
    let mut max_err: f64 = 0.0;
    let mut worst_input: f64 = 0.0;

    for i in 0..=65535u32 {
        let v = i as f64 / 65535.0;
        let ours = tf.to_linear(v as f32) as f64;
        let ref_val = reference::bt709_to_linear(v);

        let err = (ours - ref_val).abs();
        if err > max_err {
            max_err = err;
            worst_input = v;
        }
        assert!(
            err < 5e-7,
            "BT.709 to_linear at {v:.6}: ours={ours}, ref={ref_val}, err={err}"
        );
    }
    eprintln!("BT.709 to_linear vs f64: max_err={max_err:.2e} at {worst_input:.6}");
}

#[test]
fn bt709_from_linear_vs_f64() {
    let tf = Bt709;
    let mut max_err: f64 = 0.0;
    let mut worst_input: f64 = 0.0;

    for i in 0..=65535u32 {
        let v = i as f64 / 65535.0;
        let ours = tf.from_linear(v as f32) as f64;
        let ref_val = reference::bt709_from_linear(v);

        let err = (ours - ref_val).abs();
        if err > max_err {
            max_err = err;
            worst_input = v;
        }
        assert!(
            err < 5e-7,
            "BT.709 from_linear at {v:.6}: ours={ours}, ref={ref_val}, err={err}"
        );
    }
    eprintln!("BT.709 from_linear vs f64: max_err={max_err:.2e} at {worst_input:.6}");
}

// ============================================================================
// PQ vs f64 reference
// ============================================================================

#[test]
fn pq_to_linear_vs_f64() {
    let tf = Pq;
    let mut max_err: f64 = 0.0;
    let mut worst_input: f64 = 0.0;

    for i in 0..=65535u32 {
        let v = i as f64 / 65535.0;
        let ours = tf.to_linear(v as f32) as f64;
        let ref_val = reference::pq_to_linear(v);

        let err = (ours - ref_val).abs();
        if err > max_err {
            max_err = err;
            worst_input = v;
        }
        // PQ chains multiple powf() calls so f32 accumulates more error near extremes
        assert!(
            err < 1e-4,
            "PQ to_linear at {v:.6}: ours={ours}, ref={ref_val}, err={err}"
        );
    }
    eprintln!("PQ to_linear vs f64: max_err={max_err:.2e} at {worst_input:.6}");
}

#[test]
fn pq_from_linear_vs_f64() {
    let tf = Pq;
    let mut max_err: f64 = 0.0;
    let mut worst_input: f64 = 0.0;

    for i in 0..=65535u32 {
        let v = i as f64 / 65535.0;
        let ours = tf.from_linear(v as f32) as f64;
        let ref_val = reference::pq_from_linear(v);

        let err = (ours - ref_val).abs();
        if err > max_err {
            max_err = err;
            worst_input = v;
        }
        assert!(
            err < 5e-5,
            "PQ from_linear at {v:.6}: ours={ours}, ref={ref_val}, err={err}"
        );
    }
    eprintln!("PQ from_linear vs f64: max_err={max_err:.2e} at {worst_input:.6}");
}

// ============================================================================
// HLG vs f64 reference
// ============================================================================

#[test]
fn hlg_to_linear_vs_f64() {
    let tf = Hlg;
    let mut max_err: f64 = 0.0;
    let mut worst_input: f64 = 0.0;

    for i in 0..=65535u32 {
        let v = i as f64 / 65535.0;
        let ours = tf.to_linear(v as f32) as f64;
        let ref_val = reference::hlg_to_linear(v);

        let err = (ours - ref_val).abs();
        if err > max_err {
            max_err = err;
            worst_input = v;
        }
        assert!(
            err < 5e-6,
            "HLG to_linear at {v:.6}: ours={ours}, ref={ref_val}, err={err}"
        );
    }
    eprintln!("HLG to_linear vs f64: max_err={max_err:.2e} at {worst_input:.6}");
}

#[test]
fn hlg_from_linear_vs_f64() {
    let tf = Hlg;
    let mut max_err: f64 = 0.0;
    let mut worst_input: f64 = 0.0;

    // HLG scene linear range is typically [0, 1]
    for i in 0..=65535u32 {
        let v = i as f64 / 65535.0;
        let ours = tf.from_linear(v as f32) as f64;
        let ref_val = reference::hlg_from_linear(v);

        let err = (ours - ref_val).abs();
        if err > max_err {
            max_err = err;
            worst_input = v;
        }
        assert!(
            err < 5e-6,
            "HLG from_linear at {v:.6}: ours={ours}, ref={ref_val}, err={err}"
        );
    }
    eprintln!("HLG from_linear vs f64: max_err={max_err:.2e} at {worst_input:.6}");
}

// ============================================================================
// Cross-TF: u8 batch paths vs scalar for all TFs
// ============================================================================

fn test_batch_vs_scalar<T: TransferCurve>(tf: &T, name: &str)
where
    T::Luts: Sized,
{
    let luts = tf.build_luts();

    // u8 -> linear f32 (1ch, no alpha)
    let src: Vec<u8> = (0..=255).collect();
    let mut batch_out = vec![0.0f32; 256];
    tf.u8_to_linear_f32(&src, &mut batch_out, &luts, 1, false, false);

    for i in 0..=255u8 {
        let scalar = tf.to_linear(i as f32 / 255.0);
        let diff = (batch_out[i as usize] - scalar).abs();
        assert!(
            diff < 1e-5,
            "{name} u8_to_linear batch vs scalar at {i}: batch={}, scalar={scalar}, diff={diff}",
            batch_out[i as usize]
        );
    }

    // linear f32 -> u8 (1ch, no alpha)
    let linear_inputs: Vec<f32> = (0..=1000).map(|i| i as f32 / 1000.0).collect();
    let mut batch_u8 = vec![0u8; 1001];
    tf.linear_f32_to_u8(&linear_inputs, &mut batch_u8, &luts, 1, false, false);

    for i in 0..=1000 {
        let v = i as f32 / 1000.0;
        let scalar = (tf.from_linear(v) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        let diff = (batch_u8[i] as i16 - scalar as i16).unsigned_abs();
        assert!(
            diff <= 1,
            "{name} linear_to_u8 batch vs scalar at {v}: batch={}, scalar={scalar}",
            batch_u8[i]
        );
    }
}

#[test]
fn batch_vs_scalar_srgb() {
    test_batch_vs_scalar(&Srgb, "sRGB");
}

#[test]
fn batch_vs_scalar_bt709() {
    test_batch_vs_scalar(&Bt709, "BT.709");
}

#[test]
fn batch_vs_scalar_pq() {
    test_batch_vs_scalar(&Pq, "PQ");
}

#[test]
fn batch_vs_scalar_hlg() {
    test_batch_vs_scalar(&Hlg, "HLG");
}

#[test]
fn batch_vs_scalar_no_transfer() {
    test_batch_vs_scalar(&NoTransfer, "NoTransfer");
}

// ============================================================================
// i12 LUT paths vs f64 reference
// ============================================================================

fn test_i12_vs_f64<T: TransferCurve>(
    tf: &T,
    name: &str,
    to_linear_f64: fn(f64) -> f64,
    from_linear_f64: fn(f64) -> f64,
) where
    T::Luts: Sized,
{
    let luts = tf.build_luts();

    // u8 -> i12
    let src: Vec<u8> = (0..=255).collect();
    let mut i12_out = vec![0i16; 256];
    tf.u8_to_linear_i12(&src, &mut i12_out, &luts);

    for i in 0..=255u8 {
        let ref_linear = to_linear_f64(i as f64 / 255.0);
        let ref_i12 = (ref_linear * 4095.0 + 0.5).clamp(0.0, 4095.0) as i16;
        let diff = (i12_out[i as usize] - ref_i12).unsigned_abs();
        assert!(
            diff <= 1,
            "{name} u8_to_i12 at {i}: ours={}, ref={ref_i12}, diff={diff}",
            i12_out[i as usize]
        );
    }

    // i12 -> u8
    let i12_src: Vec<i16> = (0..=4095).collect();
    let mut u8_out = vec![0u8; 4096];
    tf.linear_i12_to_u8(&i12_src, &mut u8_out, &luts);

    for i in 0..=4095i16 {
        let ref_linear = i as f64 / 4095.0;
        let ref_u8 = (from_linear_f64(ref_linear) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        let diff = (u8_out[i as usize] as i16 - ref_u8 as i16).unsigned_abs();
        assert!(
            diff <= 1,
            "{name} i12_to_u8 at {i}: ours={}, ref={ref_u8}, diff={diff}",
            u8_out[i as usize]
        );
    }
}

#[test]
fn srgb_i12_vs_f64() {
    test_i12_vs_f64(
        &Srgb,
        "sRGB",
        reference::srgb_to_linear,
        reference::srgb_from_linear,
    );
}

#[test]
fn bt709_i12_vs_f64() {
    test_i12_vs_f64(
        &Bt709,
        "BT.709",
        reference::bt709_to_linear,
        reference::bt709_from_linear,
    );
}

#[test]
fn pq_i12_vs_f64() {
    test_i12_vs_f64(
        &Pq,
        "PQ",
        reference::pq_to_linear,
        reference::pq_from_linear,
    );
}

#[test]
fn hlg_i12_vs_f64() {
    test_i12_vs_f64(
        &Hlg,
        "HLG",
        reference::hlg_to_linear,
        reference::hlg_from_linear,
    );
}

// ============================================================================
// u16 batch paths vs f64 reference
// ============================================================================

fn test_u16_vs_f64<T: TransferCurve>(
    tf: &T,
    name: &str,
    to_linear_f64: fn(f64) -> f64,
    from_linear_f64: fn(f64) -> f64,
) where
    T::Luts: Sized,
{
    let luts = tf.build_luts();

    // u16 -> linear f32 (test a spread of values, not all 65536)
    let values: Vec<u16> = (0..=65535u32).step_by(37).map(|v| v as u16).collect();
    let mut f32_out = vec![0.0f32; values.len()];
    tf.u16_to_linear_f32(&values, &mut f32_out, &luts, 1, false, false);

    for (idx, &v) in values.iter().enumerate() {
        let ref_val = to_linear_f64(v as f64 / 65535.0);
        let err = (f32_out[idx] as f64 - ref_val).abs();
        assert!(
            err < 5e-5,
            "{name} u16_to_linear at {v}: ours={}, ref={ref_val}, err={err}",
            f32_out[idx]
        );
    }

    // linear f32 -> u16 (test a spread)
    let linear_inputs: Vec<f32> = (0..=10000).map(|i| i as f32 / 10000.0).collect();
    let mut u16_out = vec![0u16; linear_inputs.len()];
    tf.linear_f32_to_u16(&linear_inputs, &mut u16_out, &luts, 1, false, false);

    let mut max_diff: u32 = 0;
    for (idx, &v) in linear_inputs.iter().enumerate() {
        let ref_encoded = from_linear_f64(v as f64);
        let ref_u16 = (ref_encoded * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        let diff = (u16_out[idx] as i32 - ref_u16 as i32).unsigned_abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff <= 2,
            "{name} linear_to_u16 at {v}: ours={}, ref={ref_u16}, diff={diff}",
            u16_out[idx]
        );
    }
    eprintln!("{name} u16: max_diff={max_diff}");
}

#[test]
fn srgb_u16_vs_f64() {
    test_u16_vs_f64(
        &Srgb,
        "sRGB",
        reference::srgb_to_linear,
        reference::srgb_from_linear,
    );
}

#[test]
fn bt709_u16_vs_f64() {
    test_u16_vs_f64(
        &Bt709,
        "BT.709",
        reference::bt709_to_linear,
        reference::bt709_from_linear,
    );
}

#[test]
fn pq_u16_vs_f64() {
    test_u16_vs_f64(
        &Pq,
        "PQ",
        reference::pq_to_linear,
        reference::pq_from_linear,
    );
}

#[test]
fn hlg_u16_vs_f64() {
    test_u16_vs_f64(
        &Hlg,
        "HLG",
        reference::hlg_to_linear,
        reference::hlg_from_linear,
    );
}

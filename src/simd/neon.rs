//! AArch64 NEON convolution kernels.

use crate::weights::F32WeightTable;
use archmage::NeonToken;

/// Horizontal convolution using NEON.
#[archmage::arcane]
pub(crate) fn filter_h_row_f32_neon(
    _token: NeonToken,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    // NEON implementation: start with scalar logic, optimize later
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let tap_count = w.len();
        let out_offset = out_x * channels;

        for c in 0..channels {
            output[out_offset + c] = 0.0;
        }

        for t in 0..tap_count {
            let in_offset = (left + t) * channels;
            let weight = w[t];
            for c in 0..channels {
                output[out_offset + c] += input[in_offset + c] * weight;
            }
        }
    }
}

/// Vertical convolution using NEON.
#[archmage::arcane]
pub(crate) fn filter_v_row_f32_neon(
    _token: NeonToken,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    for v in output.iter_mut() {
        *v = 0.0;
    }

    for (row, &weight) in rows.iter().zip(weights.iter()) {
        for x in 0..width {
            output[x] += row[x] * weight;
        }
    }
}

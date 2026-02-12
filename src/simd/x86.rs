//! x86-64 AVX2+FMA convolution kernels.

use crate::weights::F32WeightTable;
use archmage::X64V3Token;

/// Horizontal convolution using AVX2+FMA.
#[archmage::arcane]
pub(crate) fn filter_h_row_f32_v3(
    _token: X64V3Token,
    input: &[f32],
    output: &mut [f32],
    weights: &F32WeightTable,
    channels: usize,
) {
    let out_width = weights.len();

    for out_x in 0..out_width {
        let left = weights.left[out_x] as usize;
        let w = weights.weights(out_x);
        let out_offset = out_x * channels;

        // Initialize accumulators
        let mut acc = [0.0f32; 4];

        for (t, &weight) in w.iter().enumerate() {
            let in_offset = (left + t) * channels;
            for c in 0..channels.min(4) {
                acc[c] += input[in_offset + c] * weight;
            }
        }

        output[out_offset..out_offset + channels.min(4)].copy_from_slice(&acc[..channels.min(4)]);
    }
}

/// Vertical convolution using AVX2+FMA.
#[archmage::arcane]
pub(crate) fn filter_v_row_f32_v3(
    _token: X64V3Token,
    rows: &[&[f32]],
    output: &mut [f32],
    weights: &[f32],
) {
    let width = output.len();
    debug_assert_eq!(rows.len(), weights.len());

    // Process 8 floats at a time
    let chunks = width / 8;
    let remainder = width % 8;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 8;
        let mut acc = [0.0f32; 8];

        for (row, &weight) in rows.iter().zip(weights.iter()) {
            for i in 0..8 {
                acc[i] += row[base + i] * weight;
            }
        }

        output[base..base + 8].copy_from_slice(&acc);
    }

    // Handle remainder
    if remainder > 0 {
        let base = chunks * 8;
        for x in 0..remainder {
            let mut acc = 0.0f32;
            for (row, &weight) in rows.iter().zip(weights.iter()) {
                acc += row[base + x] * weight;
            }
            output[base + x] = acc;
        }
    }
}

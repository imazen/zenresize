//! [`Arithmetic::Fixed22`](crate::Arithmetic::Fixed22): u8 resampling with
//! 22-bit fixed-point coefficients and a rounded, clamped 8-bit intermediate.
//!
//! Runs inside the streaming H-first u8 path: each input row is filtered
//! horizontally into the ring buffer (u8 values held in i16), then output rows
//! are filtered vertically from it. Products accumulate in i64.

use crate::filter::InterpolationDetails;
use crate::weights::fixed22_rows;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

const PRECISION: u32 = 22;

/// Horizontal and vertical `(left, coefficients)` rows for one resize.
pub(crate) struct Fixed22Tables {
    pub(crate) h: Vec<(usize, Vec<i32>)>,
    pub(crate) v: Vec<(usize, Vec<i32>)>,
}

impl Fixed22Tables {
    pub(crate) fn new(
        in_width: u32,
        in_height: u32,
        out_width: u32,
        out_height: u32,
        filter: &InterpolationDetails,
    ) -> Self {
        Self {
            h: fixed22_rows(in_width, out_width, filter),
            v: fixed22_rows(in_height, out_height, filter),
        }
    }
}

/// Round half up and clamp to the u8 range.
#[inline]
fn round_to_u8(sum: i64) -> i16 {
    ((sum + (1 << (PRECISION - 1))) >> PRECISION).clamp(0, 255) as i16
}

/// Horizontal pass of one row: `input` holds `in_width * channels` samples,
/// `output` receives `h.len() * channels` rounded u8 values.
pub(crate) fn filter_h<T: Copy + Into<i64>>(
    input: &[T],
    output: &mut [i16],
    h: &[(usize, Vec<i32>)],
    channels: usize,
) {
    for (out, (left, weights)) in output.chunks_exact_mut(channels).zip(h) {
        let taps = &input[left * channels..(left + weights.len()) * channels];
        for (c, o) in out.iter_mut().enumerate() {
            let sum: i64 = taps
                .iter()
                .skip(c)
                .step_by(channels)
                .zip(weights)
                .map(|(&v, &w)| v.into() * i64::from(w))
                .sum();
            *o = round_to_u8(sum);
        }
    }
}

/// Vertical pass producing one output row from the rows its taps cover.
pub(crate) fn filter_v(rows: &[&[i16]], output: &mut [u8], weights: &[i32]) {
    for (x, o) in output.iter_mut().enumerate() {
        let sum: i64 = rows
            .iter()
            .zip(weights)
            .map(|(row, &w)| i64::from(row[x]) * i64::from(w))
            .sum();
        *o = round_to_u8(sum) as u8;
    }
}

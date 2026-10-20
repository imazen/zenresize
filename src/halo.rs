//! NoHalo and LoHalo: halo-suppressing resamplers (issue #17).
//!
//! These are the two samplers GIMP offers as "NoHalo" and "LoHalo", designed
//! by Nicolas Robidoux and collaborators. They are **not separable**, so they
//! do not run through the H/V convolution pipeline the rest of this crate
//! uses. They are full-frame only: [`resize_halo`] reads the whole source
//! image and writes the whole output.
//!
//! This is an independent implementation from the published definitions,
//! not a translation of GEGL's source:
//!
//! - **NoHalo** upsamples with one level of Nohalo subdivision (Racette 2012,
//!   eq. 4.6–4.8: minmod-limited, co-monotone, locally bounded) finished by
//!   LBB, the locally bounded bicubic (Racette 2012, §12.4.2). The result
//!   never leaves the range of the nearby input values, so upsampling
//!   creates no halos.
//! - **LoHalo** upsamples with a Mitchell-Netravali tensor cubic (B = C = 1/3)
//!   in *sigmoidized* light: colour values pass through an inverse sigmoidal
//!   contrast curve (tanh, contrast 3.38589, linear beyond [0, 1]) before
//!   filtering and back through the curve after, which shrinks overshoots
//!   near black and white. Alpha is filtered without sigmoidization.
//!
//! When either axis shrinks, both blend their upsampler with an elliptical
//! weighted average (EWA) over the input footprint, in plain (not
//! sigmoidized) values: a radial tent of radius 1 for NoHalo, and for LoHalo
//! a radial Keys cubic of radius 2 with B = 1656 / (1592 + 597√2). The ellipse
//! axes are the per-axis scale factors clamped up to 1, and the blend weight
//! of the upsampler is `1 / (sx · sy)`, so heavy downscaling is almost pure EWA.
//!
//! References:
//! - C. Racette, *Numerical Analysis of Diagonal-Preserving, Ripple-Minimizing
//!   and Low-Pass Image Resampling Methods*, MSc thesis, Laurentian
//!   University, 2012 (arXiv:1204.4734).
//! - N. Robidoux, M. Gong, J. Cupitt, A. Turcotte, K. Martinez, *CPU, SMP and
//!   GPU implementations of Nohalo level 1*, C3S2E 2009.
//!
//! # Pixel format
//!
//! [`resize_halo`] takes and returns 4-channel f32 pixels with **alpha last
//! and premultiplied**. It is otherwise colour-agnostic, but the samplers are
//! designed for linear light, and LoHalo's sigmoid assumes colour values in
//! [0, 1]. [`resize_halo_srgba8`] wraps it for straight-alpha sRGB u8 pixels:
//! it decodes to linear light, premultiplies, resamples, and encodes back.
//!
//! Edges: pixels outside the image read as the nearest edge pixel.
//!
//! # Example
//!
//! ```
//! use zenresize::halo::{HaloSampler, resize_halo_srgba8};
//!
//! let src = vec![[200u8, 100, 50, 255]; 16 * 16];
//! let img = imgref::ImgVec::new(src, 16, 16);
//! let out = resize_halo_srgba8(img.as_ref(), 40, 40, HaloSampler::NoHalo);
//! assert_eq!((out.width(), out.height()), (40, 40));
//! assert!(out.pixels().all(|p| p == [200, 100, 50, 255]));
//! ```

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use imgref::{ImgRef, ImgRefMut, ImgVec};

/// Which halo-suppressing sampler [`resize_halo`] uses.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HaloSampler {
    /// Nohalo subdivision + LBB upsampling (no halos at all), blended with a
    /// tent EWA when downscaling. Sharp, but can look slightly "painted".
    NoHalo,
    /// Sigmoidized Mitchell-Netravali upsampling (few, small halos), blended
    /// with a Keys-cubic EWA when downscaling. Smoother than NoHalo.
    LoHalo,
}

/// Resize premultiplied 4-channel f32 pixels with NoHalo or LoHalo.
///
/// See the [module docs](self) for the pixel format and the algorithms.
///
/// # Panics
///
/// Panics if any source or output dimension is zero.
pub fn resize_halo(
    src: ImgRef<'_, [f32; 4]>,
    out_width: u32,
    out_height: u32,
    sampler: HaloSampler,
) -> ImgVec<[f32; 4]> {
    let (w, h) = (out_width as usize, out_height as usize);
    let len = w
        .checked_mul(h)
        .expect("resize_halo: out_width * out_height overflows usize");
    let mut out = vec![[0.0f32; 4]; len];
    resize_halo_into(src, ImgRefMut::new(&mut out, w, h), sampler);
    ImgVec::new(out, w, h)
}

/// [`resize_halo`] into a caller-owned (possibly strided) output image.
///
/// The output size is `dst`'s size.
///
/// # Panics
///
/// Panics if any source or output dimension is zero.
pub fn resize_halo_into(
    src: ImgRef<'_, [f32; 4]>,
    mut dst: ImgRefMut<'_, [f32; 4]>,
    sampler: HaloSampler,
) {
    let (in_w, in_h) = (src.width(), src.height());
    let (out_w, out_h) = (dst.width(), dst.height());
    assert!(in_w > 0 && in_h > 0, "source dimensions must be positive");
    assert!(out_w > 0 && out_h > 0, "output dimensions must be positive");

    let scale_x = in_w as f64 / out_w as f64;
    let scale_y = in_h as f64 / out_h as f64;
    let ewa = Ewa::new(scale_x, scale_y, sampler);

    // LoHalo filters colour in sigmoidized light; convert the source once.
    let sigmoidized: Vec<[f32; 4]>;
    let up_src = match sampler {
        HaloSampler::NoHalo => src,
        HaloSampler::LoHalo => {
            sigmoidized = src
                .pixels()
                .map(|p| {
                    [
                        inverse_sigmoidal(p[0]),
                        inverse_sigmoidal(p[1]),
                        inverse_sigmoidal(p[2]),
                        p[3],
                    ]
                })
                .collect();
            ImgRef::new(&sigmoidized, in_w, in_h)
        }
    };

    for (oy, row) in dst.rows_mut().enumerate() {
        let y = (oy as f64 + 0.5) * scale_y - 0.5;
        for (ox, out) in row.iter_mut().enumerate() {
            let x = (ox as f64 + 0.5) * scale_x - 0.5;
            let up = match sampler {
                HaloSampler::NoHalo => nohalo_lbb(&up_src, x as f32, y as f32),
                HaloSampler::LoHalo => {
                    let s = mitchell_tensor(&up_src, x as f32, y as f32);
                    [sigmoidal(s[0]), sigmoidal(s[1]), sigmoidal(s[2]), s[3]]
                }
            };
            *out = match &ewa {
                None => up,
                Some(e) => {
                    let avg = e.average(&src, x, y);
                    let t = e.theta;
                    core::array::from_fn(|c| t * up[c] + (1.0 - t) * avg[c])
                }
            };
        }
    }
}

/// [`resize_halo`] for straight-alpha sRGB-encoded RGBA u8 pixels.
///
/// Decodes sRGB to linear light, premultiplies, resamples, then
/// unpremultiplies, clamps to [0, 1] and encodes back to sRGB with rounding.
///
/// # Panics
///
/// Panics if any source or output dimension is zero.
pub fn resize_halo_srgba8(
    src: ImgRef<'_, [u8; 4]>,
    out_width: u32,
    out_height: u32,
    sampler: HaloSampler,
) -> ImgVec<[u8; 4]> {
    let to_linear: [f32; 256] =
        core::array::from_fn(|v| crate::fastmath::srgb_to_linear(v as f32 / 255.0));
    let linear: Vec<[f32; 4]> = src
        .pixels()
        .map(|p| {
            let a = p[3] as f32 / 255.0;
            [
                to_linear[p[0] as usize] * a,
                to_linear[p[1] as usize] * a,
                to_linear[p[2] as usize] * a,
                a,
            ]
        })
        .collect();
    let linear = ImgRef::new(&linear, src.width(), src.height());
    let out = resize_halo(linear, out_width, out_height, sampler);

    let encode = |v: f32| -> u8 {
        let s = crate::fastmath::srgb_from_linear(v.clamp(0.0, 1.0));
        libm::floorf(s * 255.0 + 0.5) as u8
    };
    let pixels = out
        .pixels()
        .map(|p| {
            let a = p[3].clamp(0.0, 1.0);
            if a <= 0.0 {
                return [0, 0, 0, 0];
            }
            [
                encode(p[0] / a),
                encode(p[1] / a),
                encode(p[2] / a),
                libm::floorf(a * 255.0 + 0.5) as u8,
            ]
        })
        .collect();
    ImgVec::new(pixels, out.width(), out.height())
}

/// Edge-clamped pixel read.
#[inline]
fn px(src: &ImgRef<'_, [f32; 4]>, x: isize, y: isize) -> [f32; 4] {
    let xi = x.clamp(0, src.width() as isize - 1) as usize;
    let yi = y.clamp(0, src.height() as isize - 1) as usize;
    src[(xi, yi)]
}

// =============================================================================
// NoHalo: one Nohalo subdivision, then LBB
// =============================================================================

/// The argument closest to zero if both have the same sign, else 0.
#[inline]
fn minmod(s: f32, t: f32) -> f32 {
    if s * t > 0.0 {
        if s.abs() <= t.abs() { s } else { t }
    } else {
        0.0
    }
}

/// Horizontal minmod slope at original pixel (i, j).
#[inline]
fn slope_x(src: &ImgRef<'_, [f32; 4]>, i: isize, j: isize) -> [f32; 4] {
    let (l, c, r) = (px(src, i - 1, j), px(src, i, j), px(src, i + 1, j));
    core::array::from_fn(|k| minmod(r[k] - c[k], c[k] - l[k]))
}

/// Vertical minmod slope at original pixel (i, j).
#[inline]
fn slope_y(src: &ImgRef<'_, [f32; 4]>, i: isize, j: isize) -> [f32; 4] {
    let (u, c, d) = (px(src, i, j - 1), px(src, i, j), px(src, i, j + 1));
    core::array::from_fn(|k| minmod(d[k] - c[k], c[k] - u[k]))
}

/// Value of the once-subdivided grid at fine index (p, q); original pixel
/// (i, j) sits at fine index (2i, 2j). Racette 2012, eq. 4.6–4.8.
fn nohalo_fine(src: &ImgRef<'_, [f32; 4]>, p: isize, q: isize) -> [f32; 4] {
    let (i, j) = (p.div_euclid(2), q.div_euclid(2));
    match (p.rem_euclid(2), q.rem_euclid(2)) {
        (0, 0) => px(src, i, j),
        (1, 0) => {
            let (a, b) = (px(src, i, j), px(src, i + 1, j));
            let (ma, mb) = (slope_x(src, i, j), slope_x(src, i + 1, j));
            core::array::from_fn(|k| 0.5 * (a[k] + b[k]) + 0.25 * (ma[k] - mb[k]))
        }
        (0, 1) => {
            let (a, b) = (px(src, i, j), px(src, i, j + 1));
            let (ma, mb) = (slope_y(src, i, j), slope_y(src, i, j + 1));
            core::array::from_fn(|k| 0.5 * (a[k] + b[k]) + 0.25 * (ma[k] - mb[k]))
        }
        _ => {
            let z = [
                px(src, i, j),
                px(src, i + 1, j),
                px(src, i, j + 1),
                px(src, i + 1, j + 1),
            ];
            let mx = [
                slope_x(src, i, j),
                slope_x(src, i + 1, j),
                slope_x(src, i, j + 1),
                slope_x(src, i + 1, j + 1),
            ];
            let my = [
                slope_y(src, i, j),
                slope_y(src, i + 1, j),
                slope_y(src, i, j + 1),
                slope_y(src, i + 1, j + 1),
            ];
            core::array::from_fn(|k| {
                0.25 * (z[0][k] + z[1][k] + z[2][k] + z[3][k])
                    + 0.125 * (mx[0][k] - mx[1][k] + mx[2][k] - mx[3][k])
                    + 0.125 * (my[0][k] + my[1][k] - my[2][k] - my[3][k])
            })
        }
    }
}

/// LBB value, x/y derivatives and cross derivative at the centre of a 3×3
/// block `n[row][col]` (unit spacing), limited so the bicubic stays between
/// the block's min `m` and max `M` (Racette 2012, §12.4.2).
///
/// First derivatives are clamped to `±3·min(z - m, M - z)`. The cross
/// derivative bounds come from the patch's Bézier form: the corner's interior
/// control point is `z ± dx/3 ± dy/3 ± dxy/9`, with signs set by the
/// quadrant the patch lies in, and must stay in [m, M] for each of the four
/// quadrants that share the corner. The thesis lists four of these eight
/// conditions; with only those four the result left the local range by up
/// to 3.4e-5 on random data (`nohalo_upsampling_is_locally_bounded`), so all
/// eight are enforced. They never cross: `|dx|, |dy| ≤ (M - m) / 2` for
/// centred differences inside the block, and `≤ 3·min(z - m, M - z)` after
/// the clamp.
#[inline]
fn lbb_corner(n: [[f32; 3]; 3]) -> [f32; 4] {
    let z = n[1][1];
    let mut m = z;
    let mut big_m = z;
    for row in &n {
        for &v in row {
            m = m.min(v);
            big_m = big_m.max(v);
        }
    }
    let lo = z - m;
    let hi = big_m - z;
    let d3 = 3.0 * lo.min(hi);
    let dx = (0.5 * (n[1][2] - n[1][0])).clamp(-d3, d3);
    let dy = (0.5 * (n[2][1] - n[0][1])).clamp(-d3, d3);
    let dxy = 0.25 * (n[2][2] - n[0][2] - n[2][0] + n[0][0]);
    let sum = 3.0 * (dx + dy).abs();
    let diff = 3.0 * (dx - dy).abs();
    let lower = (sum - 9.0 * lo).max(diff - 9.0 * hi);
    let upper = (9.0 * lo - diff).min(9.0 * hi - sum);
    [z, dx, dy, dxy.max(lower).min(upper)]
}

/// Cubic Hermite basis at t: (value at 0, slope at 0, value at 1, slope at 1).
#[inline]
fn hermite(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        2.0 * t3 - 3.0 * t2 + 1.0,
        t3 - 2.0 * t2 + t,
        -2.0 * t3 + 3.0 * t2,
        t3 - t2,
    ]
}

/// Nohalo-LBB at input position (x, y), pixel centres at integers.
fn nohalo_lbb(src: &ImgRef<'_, [f32; 4]>, x: f32, y: f32) -> [f32; 4] {
    // Fine grid coordinates: original pixel i is fine index 2i.
    let fx = 2.0 * x;
    let fy = 2.0 * y;
    let x0 = libm::floorf(fx);
    let y0 = libm::floorf(fy);
    let (t, u) = (fx - x0, fy - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);

    // 4×4 fine block covering the 3×3 neighbourhoods of the 2×2 corners.
    let mut f = [[[0.0f32; 4]; 4]; 4];
    for (r, frow) in f.iter_mut().enumerate() {
        for (c, v) in frow.iter_mut().enumerate() {
            *v = nohalo_fine(src, x0 - 1 + c as isize, y0 - 1 + r as isize);
        }
    }

    let hx = hermite(t);
    let hy = hermite(u);
    core::array::from_fn(|k| {
        let mut acc = 0.0f32;
        for (cy, wy) in [(0usize, (hy[0], hy[1])), (1, (hy[2], hy[3]))] {
            for (cx, wx) in [(0usize, (hx[0], hx[1])), (1, (hx[2], hx[3]))] {
                let n: [[f32; 3]; 3] =
                    core::array::from_fn(|r| core::array::from_fn(|c| f[cy + r][cx + c][k]));
                let [z, dx, dy, dxy] = lbb_corner(n);
                acc += z * wx.0 * wy.0 + dx * wx.1 * wy.0 + dy * wx.0 * wy.1 + dxy * wx.1 * wy.1;
            }
        }
        acc
    })
}

// =============================================================================
// LoHalo: sigmoidized Mitchell-Netravali
// =============================================================================

/// Sigmoidal contrast used by LoHalo.
const LOHALO_CONTRAST: f32 = 3.38589;

/// `tanh(contrast / 4)`: the unnormalized sigmoid's value at 1.
fn sig1() -> f32 {
    libm::tanhf(0.25 * LOHALO_CONTRAST)
}

/// Slope of the normalized sigmoid at 0 and at 1, used past [0, 1].
fn sigmoid_edge_slope() -> f32 {
    let s1 = sig1();
    0.25 * LOHALO_CONTRAST * (1.0 / s1 - s1)
}

/// Normalized sigmoidal contrast curve mapping [0, 1] onto [0, 1],
/// continued linearly (matching value and slope) outside it.
fn sigmoidal(q: f32) -> f32 {
    if q < 0.0 {
        q * sigmoid_edge_slope()
    } else if q > 1.0 {
        1.0 + (q - 1.0) * sigmoid_edge_slope()
    } else {
        0.5 + 0.5 * libm::tanhf(0.5 * LOHALO_CONTRAST * (q - 0.5)) / sig1()
    }
}

/// Inverse of [`sigmoidal`].
fn inverse_sigmoidal(p: f32) -> f32 {
    if p < 0.0 {
        p / sigmoid_edge_slope()
    } else if p > 1.0 {
        1.0 + (p - 1.0) / sigmoid_edge_slope()
    } else {
        0.5 + (2.0 / LOHALO_CONTRAST) * libm::atanhf(sig1() * (2.0 * p - 1.0))
    }
}

/// Mitchell-Netravali-family BC-spline kernel at distance `x` (support 2).
#[inline]
fn bc_cubic(b: f64, c: f64, x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
            + (-18.0 + 12.0 * b + 6.0 * c) * x * x
            + (6.0 - 2.0 * b))
            / 6.0
    } else if x < 2.0 {
        ((-b - 6.0 * c) * x * x * x
            + (6.0 * b + 30.0 * c) * x * x
            + (-12.0 * b - 48.0 * c) * x
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// Mitchell-Netravali (B = C = 1/3) tensor interpolation at (x, y).
fn mitchell_tensor(src: &ImgRef<'_, [f32; 4]>, x: f32, y: f32) -> [f32; 4] {
    const THIRD: f64 = 1.0 / 3.0;
    let x0 = libm::floorf(x);
    let y0 = libm::floorf(y);
    let (t, u) = ((x - x0) as f64, (y - y0) as f64);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let wx: [f32; 4] =
        core::array::from_fn(|i| bc_cubic(THIRD, THIRD, t - (i as f64 - 1.0)) as f32);
    let wy: [f32; 4] =
        core::array::from_fn(|i| bc_cubic(THIRD, THIRD, u - (i as f64 - 1.0)) as f32);
    let mut acc = [0.0f32; 4];
    for (r, &wr) in wy.iter().enumerate() {
        for (c, &wc) in wx.iter().enumerate() {
            let p = px(src, x0 - 1 + c as isize, y0 - 1 + r as isize);
            let w = wr * wc;
            for k in 0..4 {
                acc[k] += w * p[k];
            }
        }
    }
    acc
}

// =============================================================================
// EWA for downscaling
// =============================================================================

/// Elliptical weighted average over the input footprint of one output pixel.
struct Ewa {
    /// Ellipse half-axes in input pixels (scale factors clamped up to 1).
    sx: f64,
    sy: f64,
    /// Kernel support in units of the clamped axes.
    radius: f64,
    /// Upsampler blend weight, `1 / (sx * sy)`.
    theta: f32,
    sampler: HaloSampler,
}

impl Ewa {
    /// `None` when neither axis shrinks: the upsampler is used alone.
    fn new(scale_x: f64, scale_y: f64, sampler: HaloSampler) -> Option<Self> {
        if scale_x <= 1.0 && scale_y <= 1.0 {
            return None;
        }
        let sx = scale_x.max(1.0);
        let sy = scale_y.max(1.0);
        let radius = match sampler {
            HaloSampler::NoHalo => 1.0,
            HaloSampler::LoHalo => 2.0,
        };
        Some(Self {
            sx,
            sy,
            radius,
            theta: (1.0 / (sx * sy)) as f32,
            sampler,
        })
    }

    /// Kernel weight at elliptical radius `r` (< `radius`).
    #[inline]
    fn weight(&self, r: f64) -> f64 {
        match self.sampler {
            // Radial tent ("teepee").
            HaloSampler::NoHalo => 1.0 - r,
            // Radial Keys cubic (B + 2C = 1).
            HaloSampler::LoHalo => {
                let b = 1656.0 / (1592.0 + 597.0 * core::f64::consts::SQRT_2);
                bc_cubic(b, 0.5 * (1.0 - b), r)
            }
        }
    }

    fn average(&self, src: &ImgRef<'_, [f32; 4]>, x: f64, y: f64) -> [f32; 4] {
        let rx = self.radius * self.sx;
        let ry = self.radius * self.sy;
        let (x_lo, x_hi) = (libm::ceil(x - rx) as isize, libm::floor(x + rx) as isize);
        let (y_lo, y_hi) = (libm::ceil(y - ry) as isize, libm::floor(y + ry) as isize);
        let r2_max = self.radius * self.radius;
        let mut acc = [0.0f64; 4];
        let mut total = 0.0f64;
        for j in y_lo..=y_hi {
            let dy = (j as f64 - y) / self.sy;
            for i in x_lo..=x_hi {
                let dx = (i as f64 - x) / self.sx;
                let r2 = dx * dx + dy * dy;
                if r2 >= r2_max {
                    continue;
                }
                let w = self.weight(libm::sqrt(r2));
                let p = px(src, i, j);
                total += w;
                for k in 0..4 {
                    acc[k] += w * p[k] as f64;
                }
            }
        }
        // The nearest pixel is within 1/√2 clamped units of (x, y), where
        // both kernels are positive, so `total` is positive.
        core::array::from_fn(|k| (acc[k] / total) as f32)
    }
}

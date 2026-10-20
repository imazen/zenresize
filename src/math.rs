//! Float math that builds with and without `std`.
//!
//! `core` lacks these `f32`/`f64` methods, so without `std` they route
//! through `libm`. `ceil`, `floor`, `round` and `sqrt` are exact either way;
//! `sin`, `cos` and `exp` may differ by an ULP between the two builds.

#[cfg(feature = "std")]
#[inline]
pub(crate) fn f32_ceil(x: f32) -> f32 {
    x.ceil()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn f32_ceil(x: f32) -> f32 {
    libm::ceilf(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn f64_ceil(x: f64) -> f64 {
    x.ceil()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn f64_ceil(x: f64) -> f64 {
    libm::ceil(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn f64_floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn f64_floor(x: f64) -> f64 {
    libm::floor(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn f64_round(x: f64) -> f64 {
    x.round()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn f64_round(x: f64) -> f64 {
    libm::round(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn f64_sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn f64_sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn f64_sin(x: f64) -> f64 {
    x.sin()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn f64_sin(x: f64) -> f64 {
    libm::sin(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn f64_cos(x: f64) -> f64 {
    x.cos()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn f64_cos(x: f64) -> f64 {
    libm::cos(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn f64_exp(x: f64) -> f64 {
    x.exp()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn f64_exp(x: f64) -> f64 {
    libm::exp(x)
}

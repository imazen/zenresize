//! Transfer function re-exports from `linear-srgb`.
//!
//! All implementations now live in `linear_srgb::tf`. This module provides
//! the old names for backward compatibility within the crate.

// Scalar transfer functions
pub use linear_srgb::tf::bt709_to_linear;
pub use linear_srgb::tf::hlg_to_linear;
pub use linear_srgb::tf::linear_to_bt709 as bt709_from_linear;
pub use linear_srgb::tf::linear_to_hlg as hlg_from_linear;
pub use linear_srgb::tf::linear_to_pq as pq_from_linear;
pub use linear_srgb::tf::linear_to_srgb as srgb_from_linear;
pub use linear_srgb::tf::pq_to_linear;
pub use linear_srgb::tf::srgb_to_linear;

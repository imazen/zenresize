//! Layout constraint computation and pipeline planning for resize operations.
//!
//! This module re-exports types from [`zenlayout`]. See that crate for
//! full documentation.
//!
//! # Example
//!
//! ```
//! use zenresize::layout::{Constraint, ConstraintMode, Size};
//!
//! let layout = Constraint::new(ConstraintMode::FitCrop, 400, 300)
//!     .compute(1000, 500)
//!     .unwrap();
//!
//! // Source cropped to 4:3 aspect ratio, then resized to 400×300
//! assert_eq!(layout.resize_to, Size::new(400, 300));
//! assert!(layout.source_crop.is_some());
//! ```

// Re-export everything from zenlayout
pub use zenlayout::*;

//! The frozen benchmark plan: groups, arms, sampling and acceptance rules
//! (the inputs are frozen in inputs.rs). Private module of the bench binary.
//! Anything here that changes what is measured requires a new
//! `inputs::WORKLOAD_ID`.

/// zenbench native Confirm plan. Fresh processes, fixed rounds, no adaptation.
pub(crate) const PROCESSES: usize = 5;
pub(crate) const ROUNDS: usize = 30;
pub(crate) const MARGIN_PCT: f64 = 1.0;
pub(crate) const CONFIDENCE: f64 = 0.95;
/// Supervisor deadline for all five child processes together (a budget, not a
/// runtime prediction). Each child runs every group: 30 rounds x 3 arms.
pub(crate) const TIMEOUT_SECS: u64 = 7200;
/// Per-group benchmark-time budget. In Confirm, zenbench forces
/// min_rounds = max_rounds, so this cannot cut the 30 fixed rounds short.
pub(crate) const GROUP_MAX_TIME_SECS: u64 = 120;
/// Per-group hard wall-clock limit (zenbench default is 120 s); sized so 30
/// rounds of the largest whole resize cannot hit it.
pub(crate) const GROUP_MAX_WALL_SECS: u64 = 600;

/// Every group has exactly these arms. `old` is the comparison baseline;
/// `old-aa` runs the identical baseline function again as the A/A control.
pub(crate) const ARM_OLD: &str = "old";
pub(crate) const ARM_NEW: &str = "new";
pub(crate) const ARM_AA: &str = "old-aa";

/// What a group's old-vs-new comparison must show to be accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    /// Reaches the changed RGB f32 H-filter: old -> new must be an Improvement
    /// (the whole interval below -MARGIN_PCT).
    Target,
    /// Reaches the changed filter, but fixed per-call cost (weight tables,
    /// allocation) dominates at this size: non-inferiority within MARGIN_PCT.
    StartupTarget,
    /// Does not reach the changed filter: old -> new must be Equivalent
    /// (the whole interval inside +-MARGIN_PCT).
    Control,
}

impl Role {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Role::Target => "target",
            Role::StartupTarget => "startup-target",
            Role::Control => "control",
        }
    }
}

/// Public entry point timed by every arm, from the same call shape a caller uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Engine {
    /// `StreamingResize`: push every input row, collect every output row.
    Stream,
    /// `Resizer::new(&cfg).resize*(input)`.
    Fullframe,
    /// `resize_hfirst_streaming[_f32]` (does not use filter_h_row_f32).
    Hfirst,
}

impl Engine {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Engine::Stream => "stream",
            Engine::Fullframe => "fullframe",
            Engine::Hfirst => "hfirst",
        }
    }
}

/// Pixel format and processing space. RGB formats reach the F32 path's
/// 3-channel H-filter; the 4-channel formats are controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    /// RGB8_SRGB, `.linear()`: F32 path, 3 channels.
    Rgb8Linear,
    /// RGB8_SRGB, `.srgb()`: F32 path, 3 channels, no linearization.
    Rgb8Srgb,
    /// RGB16_SRGB, `.linear()`: F32 path, 3 channels (input = RGB8 x 257).
    Rgb16Linear,
    /// RGBF32_LINEAR: F32 path, 3 channels (input = RGB8 / 255).
    RgbF32Linear,
    /// RGBA8_SRGB, `.srgb()`: I16Srgb path (alpha 255 appended).
    Rgba8Srgb,
    /// RGBX8_SRGB, `.linear()`: I16Linear path (pad byte 255).
    Rgbx8Linear,
    /// RGBA8_SRGB, `.linear()`: F32 path, 4 channels with premultiply.
    Rgba8Linear,
}

impl Format {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Format::Rgb8Linear => "rgb8-linear",
            Format::Rgb8Srgb => "rgb8-srgb",
            Format::Rgb16Linear => "rgb16-linear",
            Format::RgbF32Linear => "rgbf32-linear",
            Format::Rgba8Srgb => "rgba8-srgb",
            Format::Rgbx8Linear => "rgbx8-linear",
            Format::Rgba8Linear => "rgba8-linear",
        }
    }

    pub(crate) fn channels(self) -> usize {
        match self {
            Format::Rgb8Linear | Format::Rgb8Srgb | Format::Rgb16Linear | Format::RgbF32Linear => 3,
            Format::Rgba8Srgb | Format::Rgbx8Linear | Format::Rgba8Linear => 4,
        }
    }
}

pub(crate) struct GroupSpec {
    pub(crate) id: &'static str,
    pub(crate) role: Role,
    pub(crate) engine: Engine,
    pub(crate) format: Format,
    pub(crate) input: &'static str,
    pub(crate) out_w: u32,
    pub(crate) out_h: u32,
    /// Extra elements per input row (strided input); 0 = tightly packed.
    pub(crate) in_stride_pad: usize,
}

const fn g(
    id: &'static str,
    role: Role,
    engine: Engine,
    format: Format,
    input: &'static str,
    (out_w, out_h): (u32, u32),
    in_stride_pad: usize,
) -> GroupSpec {
    GroupSpec {
        id,
        role,
        engine,
        format,
        input,
        out_w,
        out_h,
        in_stride_pad,
    }
}

use crate::inputs::{INPUTS, InputSpec};
use Engine::{Fullframe, Hfirst, Stream};
use Format::*;
use Role::{Control, StartupTarget, Target};

const C64: &str = "cid22-1025469-c64";
const C256: &str = "cid22-1044329-c256";
const N512: &str = "cid22-1189261-512";
const C1024: &str = "clic-097cb4-c1024";
const N2048: &str = "clic-0c49a5-2048x1358";
const N4000: &str = "imazen26-1001-4000x3000";

/// Lanczos throughout. Sizes: <= 64 (startup boundary), 256, 512, ~1 MP crop,
/// a 2.78 MP source and the 12 MP native photograph. Every group runs
/// old / new / old-aa.
pub(crate) const GROUPS: &[GroupSpec] = &[
    // Targets: RGB, F32 path, the changed 3-channel H-filter.
    g(
        "t-rgb8lin-stream-64to32",
        StartupTarget,
        Stream,
        Rgb8Linear,
        C64,
        (32, 32),
        0,
    ),
    g(
        "t-rgb8lin-full-64to32",
        StartupTarget,
        Fullframe,
        Rgb8Linear,
        C64,
        (32, 32),
        0,
    ),
    g(
        "t-rgb8lin-stream-256to128",
        Target,
        Stream,
        Rgb8Linear,
        C256,
        (128, 128),
        0,
    ),
    g(
        "t-rgb8lin-full-256to128",
        Target,
        Fullframe,
        Rgb8Linear,
        C256,
        (128, 128),
        0,
    ),
    g(
        "t-rgb8lin-stream-256to512",
        Target,
        Stream,
        Rgb8Linear,
        C256,
        (512, 512),
        0,
    ),
    g(
        "t-rgb8lin-stream-1024to512",
        Target,
        Stream,
        Rgb8Linear,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "t-rgb8lin-full-1024to512",
        Target,
        Fullframe,
        Rgb8Linear,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "t-rgb8lin-fullstride-1024to512",
        Target,
        Fullframe,
        Rgb8Linear,
        C1024,
        (512, 512),
        48,
    ),
    g(
        "t-rgb8srgb-stream-1024to512",
        Target,
        Stream,
        Rgb8Srgb,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "t-rgb16lin-stream-1024to512",
        Target,
        Stream,
        Rgb16Linear,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "t-rgbf32lin-stream-1024to512",
        Target,
        Stream,
        RgbF32Linear,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "t-rgb8lin-stream-2048to1024",
        Target,
        Stream,
        Rgb8Linear,
        N2048,
        (1024, 679),
        0,
    ),
    g(
        "t-rgb8lin-full-2048to1024",
        Target,
        Fullframe,
        Rgb8Linear,
        N2048,
        (1024, 679),
        0,
    ),
    g(
        "t-rgb8lin-stream-2048to512",
        Target,
        Stream,
        Rgb8Linear,
        N2048,
        (512, 340),
        0,
    ),
    // Controls: unchanged code paths, must be equivalent.
    g(
        "c-hfirst-rgb8lin-512to1024",
        Control,
        Hfirst,
        Rgb8Linear,
        N512,
        (1024, 1024),
        0,
    ),
    g(
        "c-hfirst-rgb8lin-1024to512",
        Control,
        Hfirst,
        Rgb8Linear,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "c-rgba8srgb-stream-1024to512",
        Control,
        Stream,
        Rgba8Srgb,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "c-rgba8srgb-full-1024to512",
        Control,
        Fullframe,
        Rgba8Srgb,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "c-rgbx8lin-stream-1024to512",
        Control,
        Stream,
        Rgbx8Linear,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "c-rgba8lin-stream-1024to512",
        Control,
        Stream,
        Rgba8Linear,
        C1024,
        (512, 512),
        0,
    ),
    g(
        "c-rgba8srgb-stream-2048to1024",
        Control,
        Stream,
        Rgba8Srgb,
        N2048,
        (1024, 679),
        0,
    ),
    // The 12 MP native cell: the changed filter at real 4k-class scale, plus
    // an I16Srgb control at the same geometry.
    g(
        "t-rgb8lin-stream-4000to2000",
        Target,
        Stream,
        Rgb8Linear,
        N4000,
        (2000, 1500),
        0,
    ),
    g(
        "t-rgb8lin-full-4000to2000",
        Target,
        Fullframe,
        Rgb8Linear,
        N4000,
        (2000, 1500),
        0,
    ),
    g(
        "t-rgb8lin-stream-4000to1024",
        Target,
        Stream,
        Rgb8Linear,
        N4000,
        (1024, 768),
        0,
    ),
    g(
        "c-rgba8srgb-stream-4000to2000",
        Control,
        Stream,
        Rgba8Srgb,
        N4000,
        (2000, 1500),
        0,
    ),
];

/// The frozen input a group reads.
pub(crate) fn input(id: &str) -> &'static InputSpec {
    INPUTS
        .iter()
        .find(|i| i.id == id)
        .unwrap_or_else(|| panic!("plan error: group references unknown input {id}"))
}

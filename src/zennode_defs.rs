//! Zennode definitions for image resizing/constraining.
//!
//! Defines [`Constrain`] with full resize-engine control using `Option<T>`
//! for optional parameters — `None` means "auto" or "not specified."

extern crate alloc;
use alloc::string::String;

use zennode::*;

/// Constrain image dimensions with resize, crop, or pad modes.
///
/// The primary resize/layout node. Computes how to map source dimensions
/// to target dimensions using the selected constraint mode, gravity, and
/// resampling hints.
///
/// Optional parameters (`Option<T>`) use `None` for "not specified" —
/// the engine picks sensible defaults based on the operation.
///
/// JSON API:
/// ```json
/// {
///   "w": 800,
///   "h": 600,
///   "mode": "fit_crop",
///   "gravity": "top_left",
///   "down_filter": "lanczos",
///   "up_filter": "ginseng",
///   "sharpen": 15.0,
///   "scaling_colorspace": "linear"
/// }
/// ```
///
/// RIAPI: `?w=800&h=600&mode=fit_crop&anchor=top_left&down.filter=lanczos`
#[derive(Node, Clone, Debug)]
#[node(id = "zenresize.constrain", group = Geometry, role = Resize)]
#[node(coalesce = "layout_plan")]
#[node(changes_dimensions)]
#[node(format(preferred = LinearF32))]
#[node(tags("resize", "geometry", "scale"))]
pub struct Constrain {
    // ─── Dimensions ───
    /// Target width in pixels. None = unconstrained (derive from height + aspect ratio).
    #[param(range(0..=65535), default = 0, step = 1)]
    #[param(unit = "px", section = "Dimensions", label = "Width")]
    #[kv("w", "width")]
    pub w: Option<u32>,

    /// Target height in pixels. None = unconstrained (derive from width + aspect ratio).
    #[param(range(0..=65535), default = 0, step = 1)]
    #[param(unit = "px", section = "Dimensions", label = "Height")]
    #[kv("h", "height")]
    pub h: Option<u32>,

    // ─── Layout ───
    /// Constraint mode controlling how the image fits the target dimensions.
    ///
    /// - "distort" — stretch to exact dimensions, ignoring aspect ratio
    /// - "within" — fit inside target, never upscale (default)
    /// - "fit" — fit inside target, may upscale
    /// - "within_crop" — fill target by cropping, never upscale
    /// - "fit_crop" — fill target by cropping, may upscale
    /// - "fit_pad" — fit inside target, pad to exact dimensions
    /// - "within_pad" — fit inside target without upscale, pad to exact dimensions
    /// - "pad_within" — never upscale, always pad to exact canvas
    /// - "aspect_crop" — crop to target aspect ratio without resizing
    #[param(default = "within")]
    #[param(section = "Layout", label = "Mode")]
    #[kv("mode")]
    pub mode: String,

    /// Anchor/gravity point for crop and pad operations.
    ///
    /// Controls which part of the image is preserved when cropping,
    /// or where the image is positioned when padding.
    /// Values: "center", "top_left", "top", "top_right",
    /// "left", "right", "bottom_left", "bottom", "bottom_right".
    #[param(default = "center")]
    #[param(section = "Layout", label = "Anchor")]
    #[kv("anchor")]
    pub gravity: String,

    /// Background color for pad modes.
    ///
    /// None = transparent. Accepts CSS-style hex or named colors:
    /// "white", "#FF0000", "000000FF".
    #[param(default = "")]
    #[param(section = "Layout", label = "Background Color")]
    #[kv("bgcolor")]
    pub bgcolor: Option<String>,

    // ─── Resampling ───
    /// Downscale resampling filter. None = auto (Robidoux).
    ///
    /// 31 filters available: "robidoux", "robidoux_sharp", "robidoux_fast",
    /// "lanczos", "lanczos_sharp", "lanczos2", "lanczos2_sharp",
    /// "ginseng", "ginseng_sharp", "mitchell", "catmull_rom", "cubic",
    /// "cubic_sharp", "cubic_b_spline", "hermite", "triangle", "linear",
    /// "box", "fastest", "jinc", "n_cubic", "n_cubic_sharp", etc.
    #[param(default = "")]
    #[param(section = "Quality", label = "Downscale Filter")]
    #[kv("down.filter")]
    pub down_filter: Option<String>,

    /// Upscale resampling filter. None = auto (Ginseng).
    ///
    /// Same filter names as down_filter.
    #[param(default = "")]
    #[param(section = "Quality", label = "Upscale Filter")]
    #[kv("up.filter")]
    pub up_filter: Option<String>,

    /// Color space for resampling math. None = auto ("linear" for most operations).
    ///
    /// - "linear" — resize in linear light (gamma-correct, default)
    /// - "srgb" — resize in sRGB gamma space (faster, less correct)
    #[param(default = "")]
    #[param(section = "Quality", label = "Scaling Colorspace")]
    pub scaling_colorspace: Option<String>,

    // ─── Sharpening & Blur ───
    /// Post-resize sharpening amount (0 = none, 100 = maximum).
    ///
    /// None = no sharpening. Applied as an unsharp mask after resize.
    #[param(range(0.0..=100.0), default = 0.0, step = 1.0)]
    #[param(unit = "%", section = "Quality", label = "Sharpen")]
    #[kv("f.sharpen")]
    pub sharpen: Option<f32>,

    /// Negative-lobe ratio for in-kernel sharpening.
    ///
    /// None = use filter's natural ratio. 0.0 = flatten (maximum smoothness).
    /// Above natural = sharpen. This is zero-cost (applied during weight computation).
    #[param(range(0.0..=1.0), default = 0.0, step = 0.01)]
    #[param(section = "Advanced", label = "Lobe Ratio")]
    pub lobe_ratio: Option<f32>,

    /// Kernel width scale factor. None = auto (1.0).
    ///
    /// > 1.0 = softer (widens filter window), < 1.0 = sharper (aliasing risk).
    /// Combined with blur. This is zero-cost (applied during weight computation).
    #[param(range(0.1..=4.0), default = 1.0, step = 0.01)]
    #[param(section = "Advanced", label = "Kernel Width Scale")]
    pub kernel_width_scale: Option<f32>,

    /// Post-resize Gaussian blur sigma. None = no blur.
    ///
    /// Applied as a separable H+V pass after resize. Not equivalent to
    /// kernel_width_scale (which changes the resampling kernel itself).
    #[param(range(0.0..=100.0), default = 0.0, step = 0.1)]
    #[param(section = "Advanced", label = "Post Blur")]
    pub post_blur: Option<f32>,
}

impl Default for Constrain {
    fn default() -> Self {
        Self {
            w: None,
            h: None,
            mode: String::from("within"),
            gravity: String::from("center"),
            bgcolor: None,
            down_filter: None,
            up_filter: None,
            scaling_colorspace: None,
            sharpen: None,
            lobe_ratio: None,
            kernel_width_scale: None,
            post_blur: None,
        }
    }
}

/// Registration function for aggregating crates.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&CONSTRAIN_NODE);
}

/// All zenresize zennode definitions.
pub static ALL: &[&dyn NodeDef] = &[&CONSTRAIN_NODE];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_metadata() {
        let schema = CONSTRAIN_NODE.schema();
        assert_eq!(schema.id, "zenresize.constrain");
        assert_eq!(schema.group, NodeGroup::Geometry);
        assert_eq!(schema.role, NodeRole::Resize);
        assert!(schema.tags.contains(&"resize"));
        assert!(schema.tags.contains(&"geometry"));
        assert!(schema.tags.contains(&"scale"));
    }

    #[test]
    fn schema_format_hint() {
        let schema = CONSTRAIN_NODE.schema();
        assert_eq!(schema.format.preferred, PixelFormatPreference::LinearF32);
        assert!(schema.format.changes_dimensions);
    }

    #[test]
    fn schema_coalesce() {
        let schema = CONSTRAIN_NODE.schema();
        let coalesce = schema.coalesce.as_ref().expect("coalesce should be set");
        assert_eq!(coalesce.group, "layout_plan");
    }

    #[test]
    fn param_count() {
        let schema = CONSTRAIN_NODE.schema();
        assert_eq!(schema.params.len(), 12);
    }

    #[test]
    fn optional_params_marked() {
        let schema = CONSTRAIN_NODE.schema();
        let optional_names: Vec<&str> = schema
            .params
            .iter()
            .filter(|p| p.optional)
            .map(|p| p.name)
            .collect();
        assert!(optional_names.contains(&"w"));
        assert!(optional_names.contains(&"h"));
        assert!(optional_names.contains(&"bgcolor"));
        assert!(optional_names.contains(&"down_filter"));
        assert!(optional_names.contains(&"up_filter"));
        assert!(optional_names.contains(&"scaling_colorspace"));
        assert!(optional_names.contains(&"sharpen"));
        assert!(optional_names.contains(&"lobe_ratio"));
        assert!(optional_names.contains(&"kernel_width_scale"));
        assert!(optional_names.contains(&"post_blur"));

        let required_names: Vec<&str> = schema
            .params
            .iter()
            .filter(|p| !p.optional)
            .map(|p| p.name)
            .collect();
        assert!(required_names.contains(&"mode"));
        assert!(required_names.contains(&"gravity"));
    }

    #[test]
    fn defaults_are_none() {
        let node = CONSTRAIN_NODE.create_default().unwrap();
        assert_eq!(node.get_param("w"), Some(ParamValue::None));
        assert_eq!(node.get_param("h"), Some(ParamValue::None));
        assert_eq!(node.get_param("down_filter"), Some(ParamValue::None));
        assert_eq!(node.get_param("up_filter"), Some(ParamValue::None));
        assert_eq!(node.get_param("sharpen"), Some(ParamValue::None));
        assert_eq!(node.get_param("lobe_ratio"), Some(ParamValue::None));
        assert_eq!(node.get_param("kernel_width_scale"), Some(ParamValue::None));
        assert_eq!(node.get_param("post_blur"), Some(ParamValue::None));
        assert_eq!(node.get_param("bgcolor"), Some(ParamValue::None));
        assert_eq!(node.get_param("scaling_colorspace"), Some(ParamValue::None));
    }

    #[test]
    fn required_defaults() {
        let node = CONSTRAIN_NODE.create_default().unwrap();
        assert_eq!(
            node.get_param("mode"),
            Some(ParamValue::Str("within".into()))
        );
        assert_eq!(
            node.get_param("gravity"),
            Some(ParamValue::Str("center".into()))
        );
    }

    #[test]
    fn from_kv_dimensions() {
        let mut kv = KvPairs::from_querystring("w=800&h=600");
        let node = CONSTRAIN_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("w"), Some(ParamValue::U32(800)));
        assert_eq!(node.get_param("h"), Some(ParamValue::U32(600)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn from_kv_dimension_aliases() {
        let mut kv = KvPairs::from_querystring("width=1024&height=768");
        let node = CONSTRAIN_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("w"), Some(ParamValue::U32(1024)));
        assert_eq!(node.get_param("h"), Some(ParamValue::U32(768)));
    }

    #[test]
    fn from_kv_mode_and_filters() {
        let mut kv =
            KvPairs::from_querystring("w=400&mode=fit_crop&down.filter=lanczos&up.filter=ginseng");
        let node = CONSTRAIN_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(
            node.get_param("mode"),
            Some(ParamValue::Str("fit_crop".into()))
        );
        assert_eq!(
            node.get_param("down_filter"),
            Some(ParamValue::Str("lanczos".into()))
        );
        assert_eq!(
            node.get_param("up_filter"),
            Some(ParamValue::Str("ginseng".into()))
        );
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn from_kv_sharpen() {
        let mut kv = KvPairs::from_querystring("w=200&f.sharpen=15");
        let node = CONSTRAIN_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("sharpen"), Some(ParamValue::F32(15.0)));
    }

    #[test]
    fn from_kv_anchor_and_bgcolor() {
        let mut kv =
            KvPairs::from_querystring("w=400&h=400&mode=fit_pad&anchor=top_left&bgcolor=white");
        let node = CONSTRAIN_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(
            node.get_param("gravity"),
            Some(ParamValue::Str("top_left".into()))
        );
        assert_eq!(
            node.get_param("bgcolor"),
            Some(ParamValue::Str("white".into()))
        );
    }

    #[test]
    fn from_kv_no_match() {
        let mut kv = KvPairs::from_querystring("quality=85&jpeg.progressive=true");
        let result = CONSTRAIN_NODE.from_kv(&mut kv).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn set_and_clear_optional() {
        let mut node = Constrain::default();
        assert_eq!(node.w, None);

        // Set via set_param
        let boxed: &mut dyn NodeInstance = &mut node;
        assert!(boxed.set_param("w", ParamValue::U32(800)));
        assert_eq!(boxed.get_param("w"), Some(ParamValue::U32(800)));

        // Clear with ParamValue::None
        assert!(boxed.set_param("w", ParamValue::None));
        assert_eq!(boxed.get_param("w"), Some(ParamValue::None));
    }

    #[test]
    fn round_trip() {
        let c = Constrain {
            w: Some(1920),
            h: Some(1080),
            mode: String::from("fit_crop"),
            gravity: String::from("top_left"),
            bgcolor: Some(String::from("white")),
            down_filter: Some(String::from("lanczos")),
            up_filter: Some(String::from("ginseng")),
            scaling_colorspace: Some(String::from("linear")),
            sharpen: Some(15.0),
            lobe_ratio: None,
            kernel_width_scale: None,
            post_blur: Some(0.5),
        };
        let params = c.to_params();
        let node = CONSTRAIN_NODE.create(&params).unwrap();
        let restored = node.as_any().downcast_ref::<Constrain>().unwrap();

        assert_eq!(restored.w, Some(1920));
        assert_eq!(restored.h, Some(1080));
        assert_eq!(restored.mode, "fit_crop");
        assert_eq!(restored.gravity, "top_left");
        assert_eq!(restored.bgcolor.as_deref(), Some("white"));
        assert_eq!(restored.down_filter.as_deref(), Some("lanczos"));
        assert_eq!(restored.up_filter.as_deref(), Some("ginseng"));
        assert_eq!(restored.scaling_colorspace.as_deref(), Some("linear"));
        assert_eq!(restored.sharpen, Some(15.0));
        assert_eq!(restored.lobe_ratio, None);
        assert_eq!(restored.kernel_width_scale, None);
        assert_eq!(restored.post_blur, Some(0.5));
    }

    #[test]
    fn downcast_default() {
        let node = CONSTRAIN_NODE.create_default().unwrap();
        let c = node.as_any().downcast_ref::<Constrain>().unwrap();
        assert_eq!(c.w, None);
        assert_eq!(c.h, None);
        assert_eq!(c.mode, "within");
        assert_eq!(c.gravity, "center");
        assert_eq!(c.down_filter, None);
        assert_eq!(c.sharpen, None);
    }

    #[test]
    fn registry_integration() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);
        assert!(registry.get("zenresize.constrain").is_some());

        let result = registry.from_querystring("w=800&h=600&mode=within");
        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.instances[0].schema().id, "zenresize.constrain");
    }
}

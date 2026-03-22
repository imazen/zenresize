//! zenode node definitions for image resizing/constraining.
//!
//! Defines [`Constrain`] with RIAPI-compatible querystring keys matching
//! imageflow's established resize parameters.

extern crate alloc;
use alloc::string::String;

use zennode::*;

/// Constrain image dimensions with resize, crop, or pad modes.
///
/// The primary resize/layout node. Computes how to map source dimensions
/// to target dimensions using the selected constraint mode and gravity.
///
/// JSON API: `{ "w": 800, "h": 600, "mode": "within", "filter": "lanczos" }`
/// RIAPI: `?w=800&h=600&mode=within&down.filter=lanczos`
#[derive(Node, Clone, Debug)]
#[node(id = "zenresize.constrain", group = Geometry, role = Resize)]
#[node(coalesce = "layout_plan")]
#[node(changes_dimensions)]
#[node(format(preferred = LinearF32))]
#[node(tags("resize", "geometry", "scale"))]
pub struct Constrain {
    /// Target width in pixels. 0 means unconstrained (derive from height + aspect ratio).
    #[param(range(0..=65535), default = 0, step = 1)]
    #[param(unit = "px", section = "Dimensions", label = "Width")]
    #[kv("w", "width")]
    pub w: u32,

    /// Target height in pixels. 0 means unconstrained (derive from width + aspect ratio).
    #[param(range(0..=65535), default = 0, step = 1)]
    #[param(unit = "px", section = "Dimensions", label = "Height")]
    #[kv("h", "height")]
    pub h: u32,

    /// Constraint mode controlling how the image fits the target dimensions.
    ///
    /// - `"distort"` — stretch to exact dimensions, ignoring aspect ratio
    /// - `"within"` — fit inside target, never upscale (default)
    /// - `"fit"` — fit inside target, may upscale
    /// - `"within_crop"` — fill target by cropping, never upscale
    /// - `"fit_crop"` — fill target by cropping, may upscale
    /// - `"fit_pad"` — fit inside target, pad to exact dimensions
    /// - `"within_pad"` — fit inside target without upscale, pad to exact dimensions
    /// - `"aspect_crop"` — crop to target aspect ratio without resizing
    #[param(default = "within")]
    #[param(section = "Layout", label = "Mode")]
    #[kv("mode")]
    pub mode: String,

    /// Anchor/gravity point for crop and pad operations.
    ///
    /// Controls which part of the image is preserved when cropping,
    /// or where the image is positioned when padding.
    /// Values: `"center"`, `"top_left"`, `"top"`, `"top_right"`,
    /// `"left"`, `"right"`, `"bottom_left"`, `"bottom"`, `"bottom_right"`.
    #[param(default = "center")]
    #[param(section = "Layout", label = "Anchor")]
    #[kv("anchor")]
    pub gravity: String,

    /// Resampling filter for downscaling and upscaling.
    ///
    /// Empty string means auto-select (Robidoux for downscale, Ginseng for upscale).
    /// Values: `"robidoux"`, `"lanczos"`, `"mitchell"`, `"catmull_rom"`,
    /// `"cubic"`, `"ginseng"`, `"hermite"`, `"box"`, `"triangle"`, `"linear"`, etc.
    #[param(default = "")]
    #[param(section = "Quality", label = "Filter")]
    #[kv("down.filter", "up.filter")]
    pub filter: String,

    /// Post-resize sharpening amount (0 = none, 100 = maximum).
    #[param(range(0.0..=100.0), default = 0.0, identity = 0.0, step = 1.0)]
    #[param(unit = "%", section = "Quality", label = "Sharpen")]
    #[kv("f.sharpen")]
    pub sharpen: f32,

    /// Background color for pad modes (CSS-style hex or named color).
    ///
    /// Empty string means transparent. Examples: `"white"`, `"#FF0000"`, `"000000FF"`.
    #[param(default = "")]
    #[param(section = "Layout", label = "Background Color")]
    #[kv("bgcolor")]
    pub bgcolor: String,
}

impl Default for Constrain {
    fn default() -> Self {
        Self {
            w: 0,
            h: 0,
            mode: String::from("within"),
            gravity: String::from("center"),
            filter: String::new(),
            sharpen: 0.0,
            bgcolor: String::new(),
        }
    }
}

/// Registration function for aggregating crates.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&CONSTRAIN_NODE);
}

/// All zenresize zenode definitions.
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
    fn param_count_and_names() {
        let schema = CONSTRAIN_NODE.schema();
        let names: Vec<&str> = schema.params.iter().map(|p| p.name).collect();
        assert!(names.contains(&"w"));
        assert!(names.contains(&"h"));
        assert!(names.contains(&"mode"));
        assert!(names.contains(&"gravity"));
        assert!(names.contains(&"filter"));
        assert!(names.contains(&"sharpen"));
        assert!(names.contains(&"bgcolor"));
        assert_eq!(names.len(), 7);
    }

    #[test]
    fn defaults() {
        let node = CONSTRAIN_NODE.create_default().unwrap();
        assert_eq!(node.get_param("w"), Some(ParamValue::U32(0)));
        assert_eq!(node.get_param("h"), Some(ParamValue::U32(0)));
        assert_eq!(
            node.get_param("mode"),
            Some(ParamValue::Str(String::from("within")))
        );
        assert_eq!(
            node.get_param("gravity"),
            Some(ParamValue::Str(String::from("center")))
        );
        assert_eq!(
            node.get_param("filter"),
            Some(ParamValue::Str(String::new()))
        );
        assert_eq!(node.get_param("sharpen"), Some(ParamValue::F32(0.0)));
        assert_eq!(
            node.get_param("bgcolor"),
            Some(ParamValue::Str(String::new()))
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
    fn from_kv_mode_and_filter() {
        let mut kv = KvPairs::from_querystring("w=400&mode=fit_crop&down.filter=lanczos");
        let node = CONSTRAIN_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(
            node.get_param("mode"),
            Some(ParamValue::Str("fit_crop".into()))
        );
        assert_eq!(
            node.get_param("filter"),
            Some(ParamValue::Str("lanczos".into()))
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
    fn json_round_trip() {
        let mut params = ParamMap::new();
        params.insert("w".into(), ParamValue::U32(1920));
        params.insert("h".into(), ParamValue::U32(1080));
        params.insert("mode".into(), ParamValue::Str("fit_crop".into()));
        params.insert("filter".into(), ParamValue::Str("lanczos".into()));
        params.insert("sharpen".into(), ParamValue::F32(5.0));

        let node = CONSTRAIN_NODE.create(&params).unwrap();
        assert_eq!(node.get_param("w"), Some(ParamValue::U32(1920)));
        assert_eq!(node.get_param("h"), Some(ParamValue::U32(1080)));
        assert_eq!(
            node.get_param("mode"),
            Some(ParamValue::Str("fit_crop".into()))
        );

        // Round-trip through to_params/create
        let exported = node.to_params();
        let node2 = CONSTRAIN_NODE.create(&exported).unwrap();
        assert_eq!(node2.get_param("w"), Some(ParamValue::U32(1920)));
        assert_eq!(
            node2.get_param("filter"),
            Some(ParamValue::Str("lanczos".into()))
        );
        assert_eq!(node2.get_param("sharpen"), Some(ParamValue::F32(5.0)));
    }

    #[test]
    fn downcast_to_concrete() {
        let node = CONSTRAIN_NODE.create_default().unwrap();
        let c = node.as_any().downcast_ref::<Constrain>().unwrap();
        assert_eq!(c.w, 0);
        assert_eq!(c.h, 0);
        assert_eq!(c.mode, "within");
        assert_eq!(c.gravity, "center");
        assert_eq!(c.filter, "");
        assert_eq!(c.sharpen, 0.0);
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

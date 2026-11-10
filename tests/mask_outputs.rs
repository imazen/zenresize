//! Output masks apply exactly once, before caller-side compositing.
#[path = "support/mask_outputs.rs"]
mod oracle;

#[test]
fn mask_applies_to_every_premultiplied_channel_and_external_composite() {
    oracle::run(|case, components, bytes| {
        assert_eq!(components, 0, "premultiplied mask mismatch: {case}");
        assert_eq!(bytes, 0, "external composite mask mismatch: {case}");
    });
}

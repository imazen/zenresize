fn main() {
    let (w, h) = (1024u32, 1024u32);
    let (ow, oh) = (512u32, 512u32);

    let rgba_f32: Vec<f32> = (0..w * h * 4).map(|i| (i % 256) as f32 / 255.0).collect();

    let config = zenresize::ResizeConfig::builder(w, h, ow, oh)
        .filter(zenresize::Filter::Lanczos)
        .format(zenresize::PixelFormat::LinearF32(
            zenresize::PixelLayout::Rgbx,
        ))
        .build();

    for _ in 0..10 {
        let result = zenresize::Resizer::new(&config).resize_f32(&rgba_f32);
        std::hint::black_box(&result);
    }
}

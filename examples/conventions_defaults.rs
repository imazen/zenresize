use std::io::Write;
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};
fn main() {
    let path = std::env::args().nth(1).expect("output path");
    let mut out = std::fs::File::create(path).unwrap();
    for &filter in Filter::all().iter().filter(|f| (**f as u8) <= 31) {
        for (w, h, ow, oh) in [
            (1, 7, 9, 3),
            (17, 13, 7, 5),
            (7, 5, 19, 11),
            (23, 19, 23, 19),
        ] {
            let input: Vec<u8> = (0..w * h * 4)
                .map(|i| ((i * 73 + i / 7 * 19) % 256) as u8)
                .collect();
            let config = ResizeConfig::builder(w, h, ow, oh)
                .filter(filter)
                .format(PixelDescriptor::RGBA8_SRGB)
                .srgb()
                .build();
            out.write_all(&Resizer::new(&config).resize(&input))
                .unwrap();
            let input: Vec<f32> = input.iter().map(|&x| f32::from(x) / 255.0).collect();
            let config = ResizeConfig::builder(w, h, ow, oh)
                .filter(filter)
                .format(PixelDescriptor::RGBAF32_LINEAR)
                .build();
            for x in Resizer::new(&config).resize_f32(&input) {
                out.write_all(&x.to_le_bytes()).unwrap();
            }
        }
    }
}

use zenresize::filter::{Filter, InterpolationDetails};
use zenresize::pixel::{PixelFormat, ResizeConfig};
use zenresize::resize::resize;

fn main() {
    // Test 1: Direct i16 resize (no alpha)
    let config = ResizeConfig::builder(20, 20, 20, 20)
        .format(PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: false,
        })
        .srgb()
        .build();

    let mut input = Vec::with_capacity(20 * 20 * 4);
    for y in 0..20u32 {
        for x in 0..20u32 {
            let r = ((x as f32 / 20.0) * 255.0) as u8;
            let g = ((y as f32 / 20.0) * 255.0) as u8;
            let b = (((x + y) as f32 / 40.0) * 255.0) as u8;
            input.extend_from_slice(&[r, g, b, 255]);
        }
    }

    let output = resize(&config, &input);

    // Print first few pixels
    eprintln!("=== Same-size i16 (no alpha) ===");
    for x in 0..5 {
        let i = x * 4;
        eprintln!(
            "  pixel ({},0): in=[{},{},{},{}] out=[{},{},{},{}]",
            x,
            input[i],
            input[i + 1],
            input[i + 2],
            input[i + 3],
            output[i],
            output[i + 1],
            output[i + 2],
            output[i + 3]
        );
    }

    // Test 2: Downscale
    let config2 = ResizeConfig::builder(20, 20, 10, 10)
        .format(PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: false,
        })
        .filter(Filter::Lanczos)
        .srgb()
        .build();
    let output2 = resize(&config2, &input);
    eprintln!("\n=== Downscale 20→10 (no alpha) ===");
    for x in 0..5 {
        let i = x * 4;
        eprintln!(
            "  out pixel ({},0): [{},{},{},{}]",
            x,
            output2[i],
            output2[i + 1],
            output2[i + 2],
            output2[i + 3]
        );
    }

    // Test 3: f32 path same size (with alpha, forces f32 on old code)
    // Actually now also i16. Force f32 by using linear
    let config3 = ResizeConfig::builder(20, 20, 20, 20)
        .format(PixelFormat::Srgb8 {
            channels: 4,
            has_alpha: true,
        })
        .linear()
        .build();
    let output3 = resize(&config3, &input);
    eprintln!("\n=== Same-size f32 (linear) ===");
    for x in 0..5 {
        let i = x * 4;
        eprintln!(
            "  pixel ({},0): in=[{},{},{},{}] out=[{},{},{},{}]",
            x,
            input[i],
            input[i + 1],
            input[i + 2],
            input[i + 3],
            output3[i],
            output3[i + 1],
            output3[i + 2],
            output3[i + 3]
        );
    }

    // Test 4: f32 path same size (sRGB, but force by using channels=3)
    // Can't do this easily, let me use a different approach
    // Use a single constant row to isolate horizontal vs vertical
    let mut const_input = vec![0u8; 20 * 20 * 4];
    for pixel in const_input.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[100, 50, 25, 255]);
    }
    let output4 = resize(&config, &const_input);
    eprintln!("\n=== Same-size constant [100,50,25,255] ===");
    for x in 0..5 {
        let i = x * 4;
        eprintln!(
            "  pixel ({},0): [{},{},{},{}]",
            x,
            output4[i],
            output4[i + 1],
            output4[i + 2],
            output4[i + 3]
        );
    }

    // Check weights
    let filter = InterpolationDetails::create(Filter::Lanczos);
    let i16w = zenresize::I16WeightTable::new(20, 20, &filter);
    let f32w = zenresize::F32WeightTable::new(20, 20, &filter);
    eprintln!("\nI16 weights: max_taps={}", i16w.max_taps);
    eprintln!("F32 weights: max_taps={}", f32w.max_taps);
    for i in [0, 1, 10, 19] {
        eprintln!(
            "  i16 pixel {}: left={}, taps={:?}",
            i,
            i16w.left[i],
            i16w.weights(i)
        );
        eprintln!(
            "  f32 pixel {}: left={}, taps={:?}",
            i,
            f32w.left[i],
            f32w.weights(i)
        );
    }
}

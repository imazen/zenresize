//! Raw numerical oracle driver; three channels, explicit input/output files.
use std::io::Write;
use zenresize::{
    BoundaryMode, Filter, InterpolationDetails, PixelDescriptor, ResizeConfig, Resizer,
    SampleMapping,
};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let [w, h, ow, oh]: [u32; 4] = core::array::from_fn(|i| args[i + 2].parse().unwrap());
    if args[1] == "pillow22-u8" {
        let input = std::fs::read(&args[6]).unwrap();
        let mut output = vec![0u8; ow as usize * oh as usize * 3];
        zenresize::PillowCubicResizer::new(w, h, ow, oh)
            .unwrap()
            .resize_rgb8(&input, w as usize * 3, &mut output, ow as usize * 3)
            .unwrap();
        std::fs::write(&args[7], output).unwrap();
        return;
    }
    let mode = args[1].trim_end_matches("-u8");
    let mut details = match mode {
        "cubic075" | "cubic075-align" => InterpolationDetails::cubic_bc(0.0, 0.75)
            .with_antialias(false)
            .with_boundary_mode(BoundaryMode::Clamp),
        "cubic05-aa" => InterpolationDetails::cubic_bc(0.0, 0.5),
        "linear" | "linear-align" => InterpolationDetails::create(Filter::Triangle)
            .with_antialias(false)
            .with_boundary_mode(BoundaryMode::Clamp),
        "gaussian" => InterpolationDetails::gaussian_tf().with_antialias(false),
        "gaussian-aa" => InterpolationDetails::gaussian_tf(),
        "nearest-cv" => {
            InterpolationDetails::nearest().with_sample_mapping(SampleMapping::AsymmetricFloat64)
        }
        "nearest-torch" => {
            InterpolationDetails::nearest().with_sample_mapping(SampleMapping::AsymmetricFloat32)
        }
        "nearest" => InterpolationDetails::nearest().with_sample_mapping(SampleMapping::Asymmetric),
        "nearest-half" => InterpolationDetails::nearest(),
        _ => panic!("unknown convention {mode}"),
    };
    if mode.ends_with("-align") {
        details = details.with_sample_mapping(SampleMapping::AlignCorners);
    }
    let bytes = std::fs::read(&args[6]).unwrap();
    let mut output = std::fs::File::create(&args[7]).unwrap();
    if args[1].ends_with("-u8") {
        let config = ResizeConfig::builder(w, h, ow, oh)
            .interpolation(details)
            .format(PixelDescriptor::RGB8_SRGB)
            .srgb()
            .build();
        output
            .write_all(&Resizer::new(&config).resize(&bytes))
            .unwrap();
    } else {
        let input: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        let config = ResizeConfig::builder(w, h, ow, oh)
            .interpolation(details)
            .format(PixelDescriptor::RGBF32_LINEAR)
            .build();
        for x in Resizer::new(&config).resize_f32(&input) {
            output.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}

//! Input preparation for resize-bench: decode the frozen corpus PNGs, crop
//! (never resample) to the planned sizes and store both the encoded source
//! bytes and the raw RGB8 crops in the artifact CAS. Every source and crop is
//! checked against the sha256 frozen in plan.rs; any mismatch or missing file
//! is a hard failure. Pixel values are used as stored (no colour management),
//! the same bytes the resize benchmark consumes.
//!
//! usage: resize-bench-prep --corpus=CODEC_CORPUS_DIR --artifacts=DIR \
//!         [--imazen26=IMAZEN26_PNG_V3_CHECKOUT_DIR]

mod cas;
mod inputs;

use cas::{Cas, sha256_hex, unique_suffix, write_new};
use serde_json::json;
use std::path::PathBuf;

fn fail(msg: &str) -> ! {
    eprintln!("resize-bench-prep: error: {msg}");
    std::process::exit(2);
}

fn main() {
    let mut corpus = None;
    let mut imazen26 = None;
    let mut artifacts = None;
    for arg in std::env::args().skip(1) {
        match arg.split_once('=') {
            Some(("--corpus", v)) if corpus.is_none() && !v.is_empty() => {
                corpus = Some(PathBuf::from(v))
            }
            Some(("--imazen26", v)) if imazen26.is_none() && !v.is_empty() => {
                imazen26 = Some(PathBuf::from(v))
            }
            Some(("--artifacts", v)) if artifacts.is_none() && !v.is_empty() => {
                artifacts = Some(PathBuf::from(v))
            }
            _ => fail(&format!(
                "unexpected argument {arg:?}; usage: --corpus=DIR --artifacts=DIR [--imazen26=DIR]"
            )),
        }
    }
    let (Some(corpus), Some(artifacts)) = (corpus, artifacts) else {
        fail("usage: --corpus=CODEC_CORPUS_DIR --artifacts=DIR [--imazen26=DIR]")
    };
    if !corpus.is_dir() {
        fail(&format!("corpus dir {} does not exist", corpus.display()));
    }
    if !artifacts.is_dir() {
        fail(&format!(
            "artifacts dir {} does not exist",
            artifacts.display()
        ));
    }
    if inputs::INPUTS
        .iter()
        .any(|s| s.corpus == inputs::Corpus::Imazen26PngV3)
    {
        match &imazen26 {
            Some(dir) if dir.is_dir() => {}
            Some(dir) => fail(&format!("imazen26 dir {} does not exist", dir.display())),
            None => fail("the plan needs an imazen-26 png-v3 checkout; pass --imazen26=DIR"),
        }
    }
    let cas = Cas::open(&artifacts).unwrap_or_else(|e| fail(&e));
    let mut records = Vec::new();
    let mut unfrozen = Vec::new();
    for spec in inputs::INPUTS {
        let root = match spec.corpus {
            inputs::Corpus::CodecCorpus => &corpus,
            inputs::Corpus::Imazen26PngV3 => imazen26
                .as_ref()
                .unwrap_or_else(|| fail("--imazen26=DIR is required")),
        };
        let path = root.join(spec.corpus_path);
        let encoded = cas::read_all(&path).unwrap_or_else(|e| fail(&e));
        let source_sha = sha256_hex(&encoded);
        if source_sha != spec.source_sha256 {
            fail(&format!(
                "{}: source sha256 {source_sha} != frozen {}",
                path.display(),
                spec.source_sha256
            ));
        }
        cas.put(&encoded).unwrap_or_else(|e| fail(&e));

        let mut reader = png::Decoder::new(std::io::Cursor::new(&encoded))
            .read_info()
            .unwrap_or_else(|e| fail(&format!("{}: {e}", path.display())));
        let size = reader
            .output_buffer_size()
            .unwrap_or_else(|| fail(&format!("{}: image too large", path.display())));
        let mut buf = vec![0u8; size];
        let info = reader
            .next_frame(&mut buf)
            .unwrap_or_else(|e| fail(&format!("{}: {e}", path.display())));
        if info.color_type != png::ColorType::Rgb || info.bit_depth != png::BitDepth::Eight {
            fail(&format!(
                "{}: {:?} {:?}, expected 8-bit RGB",
                path.display(),
                info.color_type,
                info.bit_depth
            ));
        }
        if (info.width, info.height) != (spec.source_w, spec.source_h) {
            fail(&format!(
                "{}: {}x{}, expected {}x{}",
                path.display(),
                info.width,
                info.height,
                spec.source_w,
                spec.source_h
            ));
        }
        let (x, y, w, h) = spec.crop;
        if x + w > info.width || y + h > info.height {
            fail(&format!(
                "{}: crop {:?} outside the image",
                spec.id, spec.crop
            ));
        }
        let mut raw = Vec::with_capacity(w as usize * h as usize * 3);
        for row in y..y + h {
            let start = row as usize * info.line_size + x as usize * 3;
            raw.extend_from_slice(&buf[start..start + w as usize * 3]);
        }
        let raw_sha = sha256_hex(&raw);
        if spec.raw_rgb8_sha256 == "TBD" {
            unfrozen.push(format!("{}: raw_rgb8_sha256 = {raw_sha}", spec.id));
        } else if raw_sha != spec.raw_rgb8_sha256 {
            fail(&format!(
                "{}: raw crop sha256 {raw_sha} != frozen {}",
                spec.id, spec.raw_rgb8_sha256
            ));
        }
        let stored = cas.put(&raw).unwrap_or_else(|e| fail(&e));
        // Read the stored object back and prove it still hashes to its name.
        cas.get_verified(&stored).unwrap_or_else(|e| fail(&e));
        records.push(
            json!({"id": spec.id, "corpus": format!("{:?}", spec.corpus), "root": root.display().to_string(),
            "corpus_path": spec.corpus_path, "source_sha256": source_sha,
            "source_bytes": encoded.len(), "source_w": info.width, "source_h": info.height,
            "crop": [x, y, w, h], "raw_rgb8_sha256": raw_sha, "raw_bytes": raw.len()}),
        );
    }
    if !unfrozen.is_empty() {
        fail(&format!(
            "plan inputs are not frozen; computed:\n  {}",
            unfrozen.join("\n  ")
        ));
    }
    let dir = artifacts.join("inputs");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| fail(&format!("{}: {e}", dir.display())));
    let manifest = dir.join(format!("manifest-{}.json", unique_suffix()));
    let body = json!({"workload_id": inputs::WORKLOAD_ID, "corpus": corpus.display().to_string(),
        "imazen26": imazen26.as_ref().map(|d| d.display().to_string()), "inputs": records});
    write_new(
        &manifest,
        &serde_json::to_vec_pretty(&body).expect("manifest serializes"),
    )
    .unwrap_or_else(|e| fail(&e));
    println!("{}", manifest.display());
}

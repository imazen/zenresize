//! Frozen benchmark inputs and workload identity, shared by the bench and
//! prep binaries as a private module. Changing anything here, or anything in
//! plan.rs that changes what is measured, requires a new WORKLOAD_ID.

/// Identifies inputs, case inventory and sampling plan. Bump on any change.
pub(crate) const WORKLOAD_ID: &str = "zenresize-rgb-hfilter-whole-resize-v2";

/// The dataset checkout a `corpus_path` is resolved against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Corpus {
    /// imazen/codec-corpus checkout (the `--corpus` directory).
    CodecCorpus,
    /// imazen-26 checkout on the `variant/png-v3` branch (the optional
    /// `--imazen26` directory): SDR PNG renders of the camera originals at
    /// their native dimensions — never resampled.
    Imazen26PngV3,
}

/// A real corpus image, cropped (never resampled) to the benchmark size.
pub(crate) struct InputSpec {
    pub(crate) id: &'static str,
    /// Which checkout `corpus_path` is relative to.
    pub(crate) corpus: Corpus,
    /// Relative to the selected checkout's root.
    pub(crate) corpus_path: &'static str,
    pub(crate) source_sha256: &'static str,
    pub(crate) source_w: u32,
    pub(crate) source_h: u32,
    pub(crate) crop: (u32, u32, u32, u32),
    /// sha256 of the cropped, tightly packed RGB8 rows.
    pub(crate) raw_rgb8_sha256: &'static str,
}

/// codec-corpus 28205bbc (2026-07-23): CID22-512 validation and CLIC2025
/// training PNGs, all 8-bit RGB — plus one imazen-26 `variant/png-v3` SDR
/// render at its native camera size. No source in CID22/CLIC reaches 4k, so
/// the large cell comes from the pulled png-v3 checkout (imazen/imazen-26
/// b1faae55): a real 4000x3000 photograph, decoded once and stored raw.
pub(crate) const INPUTS: &[InputSpec] = &[
    InputSpec {
        id: "cid22-1025469-c64",
        corpus: Corpus::CodecCorpus,
        corpus_path: "CID22/CID22-512/validation/1025469.png",
        source_sha256: "c7b8f38aa309b7e32d7387dff4fb82aeeba3ede32116130600a05825b9966f18",
        source_w: 512,
        source_h: 512,
        crop: (224, 224, 64, 64),
        raw_rgb8_sha256: "a103c45e03be2be112535ede2089e2e08f597865c44faa3d33f1cbdd56d794e6",
    },
    InputSpec {
        id: "cid22-1044329-c256",
        corpus: Corpus::CodecCorpus,
        corpus_path: "CID22/CID22-512/validation/1044329.png",
        source_sha256: "a383f96f0634f2431e4490966acff583d78297622aff834a37f0f38c4683eb24",
        source_w: 512,
        source_h: 512,
        crop: (128, 128, 256, 256),
        raw_rgb8_sha256: "dd3007e77aa2cceae3f72351e9d7f915e8bd1ead62d0e5818329546847d7bee1",
    },
    InputSpec {
        id: "cid22-1189261-512",
        corpus: Corpus::CodecCorpus,
        corpus_path: "CID22/CID22-512/validation/1189261.png",
        source_sha256: "245f6563c03d0d2728d679c5195b80e7d1ccdff3ee118648e910a42cadec7525",
        source_w: 512,
        source_h: 512,
        crop: (0, 0, 512, 512),
        raw_rgb8_sha256: "c7d2a571de4b74fce880127a3f30e6964dd9cc6502159d8e8a7e38abebad1a18",
    },
    InputSpec {
        id: "clic-097cb4-c1024",
        corpus: Corpus::CodecCorpus,
        corpus_path: "clic2025/training/097cb426910ba8ce2525dd8bb7fb1777.png",
        source_sha256: "7995fc6c3653a0cfa9cfa6ee207983532f2c52d430262243cc917fd214ae186c",
        source_w: 1507,
        source_h: 2048,
        crop: (241, 512, 1024, 1024),
        raw_rgb8_sha256: "9da5f1a80c8d584b88101632674673504238a7427ba39b5b1253d4e68c4114e6",
    },
    InputSpec {
        id: "clic-0c49a5-2048x1358",
        corpus: Corpus::CodecCorpus,
        corpus_path: "clic2025/training/0c49a5cce349020bbba2f97ae41e90ba.png",
        source_sha256: "2d74383267345922b4afb34f04d6249ca9a39869d9954008e24a7794263b9415",
        source_w: 2048,
        source_h: 1358,
        crop: (0, 0, 2048, 1358),
        raw_rgb8_sha256: "212131e9fafcc687b06734592a49d61e2dfed087bc7c6f6ff259eb90d0e5fa71",
    },
    // Native 4000x3000 photograph (Samsung S21 Ultra); the .sdr.png render
    // carries the camera pixels 1:1 — cropped to the full frame, never
    // resampled. This is the previously missing 4k-class cell.
    InputSpec {
        id: "imazen26-1001-4000x3000",
        corpus: Corpus::Imazen26PngV3,
        corpus_path: "png-v3/1000-lilith-photos-general/1001_general_oceanfront-resort-view_sheraton-waikiki-east-honolulu_s21u_iso32-f1p8_20210602-055442_4000x3000.sdr.png",
        source_sha256: "15cd2f7c2648e3238f41456ceedc8330f6c2bf53a61a1a2f560e61b96d1e7ee6",
        source_w: 4000,
        source_h: 3000,
        crop: (0, 0, 4000, 3000),
        raw_rgb8_sha256: "df7a242ea5728cc9c5024f83aaecb9db91dc50c835909d2a4e3f2a4ab9b42777",
    },
];

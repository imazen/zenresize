//! Private helpers shared by the hfilter-parity binaries. Each binary includes
//! this file with `#[path]`; nothing here is library API.

/// splitmix64; fixed seeds make every run reproducible.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub(crate) fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
}

/// How much of the sweep to run. Both scopes are explicit on the command line;
/// counts are printed so a report can state exactly what ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    Quick,
    Full,
}

impl Scope {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Scope::Quick => "quick",
            Scope::Full => "full",
        }
    }
}

/// Accepts exactly `--scope=quick|full` (or `--scope quick|full`). Anything
/// else, including no argument, exits 2 before any work starts.
pub(crate) fn parse_scope(bin: &str) -> Scope {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = match args.as_slice() {
        [one] => one.strip_prefix("--scope=").map(str::to_owned),
        [flag, value] if flag == "--scope" => Some(value.clone()),
        _ => None,
    };
    match value.as_deref() {
        Some("quick") => Scope::Quick,
        Some("full") => Scope::Full,
        _ => {
            eprintln!("usage: {bin} --scope=quick|full   (got {args:?})");
            std::process::exit(2);
        }
    }
}

/// Feature set this binary was built with (forwarded to both zenresize copies).
pub(crate) fn features() -> String {
    let mut f = Vec::new();
    if cfg!(feature = "avx512") {
        f.push("avx512");
    }
    if cfg!(feature = "pretty-safe") {
        f.push("pretty-safe");
    }
    if f.is_empty() {
        "none".into()
    } else {
        f.join(",")
    }
}

/// Bit-exact for every non-NaN value; any NaN matches any NaN. Rust does not
/// specify NaN payloads for arithmetic, so payload agreement is counted
/// separately and never treated as a parity requirement.
pub(crate) fn same_class(a: f32, b: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        a.is_nan() && b.is_nan()
    } else {
        a.to_bits() == b.to_bits()
    }
}

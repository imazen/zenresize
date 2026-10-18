//! Content-addressed artifact store shared by the bench and prep binaries
//! (private module). Objects live at `<artifacts>/cas/sha256/<aa>/<sha256>`,
//! are written once through a `create_new` temporary file and published with a
//! no-clobber hard link, and are never overwritten: storing bytes whose object
//! already exists re-reads and byte-compares the existing object instead.

use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest.iter() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

pub(crate) fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{nanos}", std::process::id())
}

/// Write a new file; fails if it already exists (never overwrites).
pub(crate) fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    f.write_all(bytes)
        .and_then(|()| f.sync_all())
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

pub(crate) struct Cas {
    root: PathBuf,
}

impl Cas {
    pub(crate) fn open(artifacts: &Path) -> Result<Cas, String> {
        let root = artifacts.join("cas");
        for d in [root.join("sha256"), root.join("tmp")] {
            fs::create_dir_all(&d).map_err(|e| format!("cannot create {}: {e}", d.display()))?;
        }
        Ok(Cas { root })
    }

    pub(crate) fn object_path(&self, hash: &str) -> PathBuf {
        self.root.join("sha256").join(&hash[..2]).join(hash)
    }

    /// Store `bytes`; returns their sha256. An existing object must match byte
    /// for byte or this fails.
    pub(crate) fn put(&self, bytes: &[u8]) -> Result<String, String> {
        let hash = sha256_hex(bytes);
        let dest = self.object_path(&hash);
        if dest.exists() {
            self.verify_existing(&dest, bytes)?;
            return Ok(hash);
        }
        let dir = dest.parent().expect("object path has a parent");
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let tmp = self
            .root
            .join("tmp")
            .join(format!("{hash}.{}.part", unique_suffix()));
        write_new(&tmp, bytes)?;
        let linked = fs::hard_link(&tmp, &dest);
        let removed = fs::remove_file(&tmp);
        match linked {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::AlreadyExists => self.verify_existing(&dest, bytes)?,
            Err(e) => return Err(format!("cannot publish {}: {e}", dest.display())),
        }
        removed.map_err(|e| format!("cannot remove {}: {e}", tmp.display()))?;
        Ok(hash)
    }

    fn verify_existing(&self, dest: &Path, bytes: &[u8]) -> Result<(), String> {
        let existing = read_all(dest)?;
        if existing != bytes {
            return Err(format!(
                "CAS object {} exists with different bytes; refusing to overwrite",
                dest.display()
            ));
        }
        Ok(())
    }

    /// Read an object and prove its content still hashes to its name.
    pub(crate) fn get_verified(&self, hash: &str) -> Result<Vec<u8>, String> {
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("invalid sha256 {hash:?}"));
        }
        let path = self.object_path(hash);
        let bytes = read_all(&path)?;
        let actual = sha256_hex(&bytes);
        if actual != hash {
            return Err(format!(
                "CAS object {} is corrupt: content hashes to {actual}",
                path.display()
            ));
        }
        Ok(bytes)
    }
}

pub(crate) fn read_all(path: &Path) -> Result<Vec<u8>, String> {
    let mut f = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut v = Vec::new();
    f.read_to_end(&mut v)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(v)
}

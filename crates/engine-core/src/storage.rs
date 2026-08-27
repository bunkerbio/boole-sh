use anyhow::Result;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Backing storage for tensor data — either mmap'd from disk or owned in memory.
pub enum Storage {
    Mmap(Arc<Mmap>),
    Owned(Vec<u8>),
}

impl Storage {
    pub fn mmap(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Storage::Mmap(Arc::new(mmap)))
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Storage::Mmap(m) => m.as_ref(),
            Storage::Owned(v) => v.as_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

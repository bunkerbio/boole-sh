use anyhow::{Result, Context};
use engine_core::{DType, Storage, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Loads safetensors weight files via mmap.
pub struct SafetensorsLoader {
    files: Vec<(Arc<Mmap>, PathBuf)>,
}

impl SafetensorsLoader {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let mut files = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let path = entry.path();
            let file = File::open(&path)
                .with_context(|| format!("opening {}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file)? };
            files.push((Arc::new(mmap), path));
        }

        anyhow::ensure!(!files.is_empty(), "no .safetensors files in {}", model_dir.display());
        Ok(Self { files })
    }

    pub fn load_tensors(&self) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();

        for (mmap, path) in &self.files {
            let st = SafeTensors::deserialize(mmap.as_ref())
                .with_context(|| format!("parsing {}", path.display()))?;

            for (name, view) in st.tensors() {
                let dtype = convert_dtype(view.dtype());
                let shape: Vec<usize> = view.shape().to_vec();
                let offset = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
                let storage = Arc::new(Storage::Mmap(Arc::clone(mmap)));
                tensors.insert(name.to_string(), Tensor::new(storage, offset, shape, dtype));
            }
        }

        Ok(tensors)
    }
}

fn convert_dtype(dt: safetensors::Dtype) -> DType {
    match dt {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        _ => DType::F32, // fallback
    }
}

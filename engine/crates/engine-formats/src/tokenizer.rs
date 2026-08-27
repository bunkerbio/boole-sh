use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

/// Wraps the HuggingFace tokenizers crate for BPE encoding/decoding.
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let text = self.inner.decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode error: {e}"))?;
        Ok(text)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

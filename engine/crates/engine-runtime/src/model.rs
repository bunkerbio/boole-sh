use anyhow::Result;
use candle_core::{Device, Tensor, IndexOp};
use engine_formats::ModelConfig;
use engine_kernels::rms_norm;
use std::cell::RefCell;

use crate::block::TransformerBlock;
use crate::kv_cache::KvCache;
use crate::weights::Weights;

pub struct Model {
    pub config: ModelConfig,
    embed_tokens: Tensor,
    blocks: Vec<TransformerBlock>,
    final_norm: Tensor,
    lm_head: Tensor,
    cache: RefCell<KvCache>,
}

impl Model {
    pub fn load(config: ModelConfig, weights: Weights) -> Result<Self> {
        let embed_tokens = weights.get("model.embed_tokens.weight")?;

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = TransformerBlock::load(
                i,
                &weights,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim(),
                config.rms_norm_eps,
                config.rope_theta,
            )?;
            blocks.push(block);
        }

        let final_norm = weights.get("model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            weights.get("model.embed_tokens.weight")?
        } else {
            weights.get("lm_head.weight")?
        };

        let cache = RefCell::new(KvCache::new(config.num_hidden_layers));

        Ok(Self { config, embed_tokens, blocks, final_norm, lm_head, cache })
    }

    pub fn forward(&self, token_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        let seq_len = token_ids.len();
        let device = self.embed_tokens.device().clone();

        // Embedding lookup
        let ids_vec: Vec<u32> = token_ids.to_vec();
        let ids = Tensor::from_slice(&ids_vec, &[seq_len], &device)?;
        let mut hidden = self.embed_tokens.index_select(&ids, 0)?;
        hidden = hidden.reshape(&[1, seq_len, self.config.hidden_size])?;

        // Position indices
        let positions: Vec<usize> = (start_pos..start_pos + seq_len).collect();

        // Causal mask for the visible window
        let mask = if seq_len > 1 {
            let total_len = start_pos + seq_len;
            Some(Self::causal_mask(seq_len, total_len, start_pos, &device)?)
        } else {
            None
        };

        // Forward through transformer blocks
        let mut cache = self.cache.borrow_mut();
        for block in &self.blocks {
            hidden = block.forward(&hidden, &positions, mask.as_ref(), Some(&mut cache))?;
        }

        // Final norm
        hidden = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;

        // Project to vocab logits
        let logits = hidden.matmul(&self.lm_head.t()?)?;

        // Return logits for the last token: (1, vocab)
        let last_logits = logits.i((.., seq_len - 1, ..))?;
        Ok(last_logits)
    }

    fn causal_mask(
        query_len: usize,
        kv_len: usize,
        start_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut mask_data = vec![0f32; query_len * kv_len];
        for q in 0..query_len {
            let q_pos = start_pos + q;
            for k in 0..kv_len {
                if k > q_pos {
                    mask_data[q * kv_len + k] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_slice(&mask_data, &[1, 1, query_len, kv_len], device)?;
        Ok(mask)
    }

    pub fn reset_cache(&self) {
        self.cache.borrow_mut().reset();
    }

    pub fn device(&self) -> &Device {
        self.embed_tokens.device()
    }
}

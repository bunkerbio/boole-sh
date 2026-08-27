use anyhow::Result;
use candle_core::{Device, Tensor, IndexOp};
use engine_formats::ModelConfig;
use engine_kernels::rms_norm;

use crate::block::TransformerBlock;
use crate::weights::Weights;

pub struct Model {
    pub config: ModelConfig,
    embed_tokens: Tensor,
    blocks: Vec<TransformerBlock>,
    final_norm: Tensor,
    lm_head: Tensor,
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

        Ok(Self { config, embed_tokens, blocks, final_norm, lm_head })
    }

    pub fn forward(&self, token_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        let seq_len = token_ids.len();
        let device = self.embed_tokens.device();

        // Embedding lookup
        let ids = Tensor::from_slice(
            &token_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
            &[1, seq_len],
            device,
        )?;
        let mut hidden = self.embed_tokens.index_select(&ids.flatten_all()?, 0)?;
        hidden = hidden.reshape(&[1, seq_len, self.config.hidden_size])?;

        // Position indices
        let positions: Vec<usize> = (start_pos..start_pos + seq_len).collect();

        // Causal mask (only needed for seq_len > 1)
        let mask = if seq_len > 1 {
            Some(Self::causal_mask(seq_len, device)?)
        } else {
            None
        };

        // Forward through transformer blocks
        for block in &self.blocks {
            hidden = block.forward(&hidden, &positions, mask.as_ref())?;
        }

        // Final norm
        hidden = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;

        // Project to logits: (1, seq_len, hidden) @ (vocab, hidden)^T -> (1, seq_len, vocab)
        let logits = hidden.matmul(&self.lm_head.t()?)?;

        // Return logits for the last token only: (1, vocab)
        let last_logits = logits.i((.., seq_len - 1, ..))?;
        Ok(last_logits)
    }

    fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        let mut mask_data = vec![0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_slice(&mask_data, &[1, 1, seq_len, seq_len], device)?;
        Ok(mask)
    }

    pub fn device(&self) -> &Device {
        self.embed_tokens.device()
    }
}

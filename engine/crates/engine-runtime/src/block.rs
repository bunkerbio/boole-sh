use anyhow::Result;
use candle_core::Tensor;
use engine_kernels::{rms_norm, grouped_query_attention, apply_rope, matmul, silu};

pub struct TransformerBlock {
    pub layer_idx: usize,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_eps: f64,
    rope_theta: f64,
}

impl TransformerBlock {
    pub fn load(
        layer_idx: usize,
        weights: &crate::weights::Weights,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_eps: f64,
        rope_theta: f64,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");

        Ok(Self {
            layer_idx,
            attn_norm: weights.get(&format!("{prefix}.input_layernorm.weight"))?,
            ffn_norm: weights.get(&format!("{prefix}.post_attention_layernorm.weight"))?,
            wq: weights.get(&format!("{prefix}.self_attn.q_proj.weight"))?,
            wk: weights.get(&format!("{prefix}.self_attn.k_proj.weight"))?,
            wv: weights.get(&format!("{prefix}.self_attn.v_proj.weight"))?,
            wo: weights.get(&format!("{prefix}.self_attn.o_proj.weight"))?,
            w_gate: weights.get(&format!("{prefix}.mlp.gate_proj.weight"))?,
            w_up: weights.get(&format!("{prefix}.mlp.up_proj.weight"))?,
            w_down: weights.get(&format!("{prefix}.mlp.down_proj.weight"))?,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_eps,
            rope_theta,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3()?;

        // Pre-attention norm
        let normed = rms_norm(x, &self.attn_norm, self.rms_eps)?;

        // QKV projections: (batch, seq, hidden) @ (hidden, proj_dim)^T
        let q = matmul(&normed, &self.wq.t()?)?;
        let k = matmul(&normed, &self.wk.t()?)?;
        let v = matmul(&normed, &self.wv.t()?)?;

        // Reshape to (batch, seq, num_heads, head_dim)
        let q = q.reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;

        // Apply RoPE
        let q = apply_rope(&q, positions, self.rope_theta)?;
        let k = apply_rope(&k, positions, self.rope_theta)?;

        // Grouped-query attention
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let attn_out = grouped_query_attention(&q, &k, &v, num_kv_groups, mask)?;

        // Output projection: (batch, seq, num_heads * head_dim) -> (batch, seq, hidden)
        let attn_out = attn_out.reshape(&[batch, seq_len, self.num_heads * self.head_dim])?;
        let attn_out = matmul(&attn_out, &self.wo.t()?)?;

        // Residual
        let x = (x + attn_out)?;

        // Pre-FFN norm
        let normed = rms_norm(&x, &self.ffn_norm, self.rms_eps)?;

        // SwiGLU MLP: gate * up then down
        let gate = matmul(&normed, &self.w_gate.t()?)?;
        let up = matmul(&normed, &self.w_up.t()?)?;
        let gate = silu(&gate)?;
        let ffn_out = (gate * up)?;
        let ffn_out = matmul(&ffn_out, &self.w_down.t()?)?;

        // Residual
        let out = (x + ffn_out)?;
        Ok(out)
    }
}

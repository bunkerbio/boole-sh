use anyhow::Result;
use candle_core::Tensor;

/// Grouped-query attention (GQA).
/// q: (batch, seq_len, num_heads, head_dim)
/// k: (batch, seq_len, num_kv_heads, head_dim)
/// v: (batch, seq_len, num_kv_heads, head_dim)
/// mask: optional causal mask
pub fn grouped_query_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_kv_groups: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (_batch, _seq_len, _num_heads, head_dim) = q.dims4()?;

    let k_expanded = expand_kv(k, num_kv_groups)?;
    let v_expanded = expand_kv(v, num_kv_groups)?;

    let q_t = q.transpose(1, 2)?;
    let k_t = k_expanded.transpose(1, 2)?.transpose(2, 3)?;
    let v_t = v_expanded.transpose(1, 2)?;

    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut scores = q_t.matmul(&k_t)?;
    scores = (scores * scale)?;

    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }

    let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
    let output = attn_weights.matmul(&v_t)?;
    let output = output.transpose(1, 2)?;

    Ok(output)
}

fn expand_kv(kv: &Tensor, num_groups: usize) -> Result<Tensor> {
    if num_groups == 1 {
        return Ok(kv.clone());
    }
    let (batch, seq_len, num_kv_heads, head_dim) = kv.dims4()?;
    let expanded = kv
        .unsqueeze(3)?
        .expand(&[batch, seq_len, num_kv_heads, num_groups, head_dim])?
        .reshape(&[batch, seq_len, num_kv_heads * num_groups, head_dim])?;
    Ok(expanded)
}

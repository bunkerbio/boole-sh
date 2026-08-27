use anyhow::Result;
use candle_core::Tensor;

/// Apply Rotary Position Embeddings (RoPE).
/// x: (..., seq_len, head_dim)
/// positions: the position indices for each token
pub fn apply_rope(x: &Tensor, positions: &[usize], theta: f64) -> Result<Tensor> {
    let (_batch, seq_len, _heads, head_dim) = x.dims4()?;
    let half = head_dim / 2;

    let mut cos_vals = vec![0f32; seq_len * half];
    let mut sin_vals = vec![0f32; seq_len * half];

    for (s, &pos) in positions.iter().enumerate() {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            cos_vals[s * half + i] = angle.cos() as f32;
            sin_vals[s * half + i] = angle.sin() as f32;
        }
    }

    let device = x.device();
    let cos = Tensor::from_vec(cos_vals, &[1, seq_len, 1, half], device)?;
    let sin = Tensor::from_vec(sin_vals, &[1, seq_len, 1, half], device)?;

    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;

    let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

    let result = Tensor::cat(&[&rotated_x1, &rotated_x2], 3)?;
    Ok(result)
}

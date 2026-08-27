use anyhow::Result;
use candle_core::Tensor;

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps)
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
    let eps_tensor = mean_sq.ones_like()? * eps;
    let norm_factor = (mean_sq + eps_tensor)?.sqrt()?.recip()?;
    let normalized = x.broadcast_mul(&norm_factor)?;
    let result = normalized.broadcast_mul(weight)?;
    Ok(result)
}

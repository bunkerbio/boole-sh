use anyhow::Result;
use candle_core::Tensor;

/// Matrix multiplication: (M, K) x (K, N) -> (M, N).
/// Wraps candle for correctness — will be swappable later.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let result = a.matmul(b)?;
    Ok(result)
}

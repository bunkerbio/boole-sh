use anyhow::Result;
use candle_core::Tensor;

/// SiLU (Swish): x * sigmoid(x)
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let result = candle_nn::ops::silu(x)?;
    Ok(result)
}

/// Softmax over the last dimension.
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    let result = candle_nn::ops::softmax_last_dim(x)?;
    Ok(result)
}

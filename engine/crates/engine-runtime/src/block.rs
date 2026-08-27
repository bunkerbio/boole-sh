/// A single transformer block (Llama-family):
/// RMSNorm → GQA attention w/ RoPE → residual → RMSNorm → SwiGLU MLP → residual
///
/// Implementation deferred to milestone 3.
pub struct TransformerBlock {
    pub layer_idx: usize,
}

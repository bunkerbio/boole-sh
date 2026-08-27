use anyhow::Result;

/// Trait for weight quantization schemes.
/// Implementors encode f32 weights into a compressed representation
/// and decode them back. The trait is designed to be reimplemented
/// without touching callers.
pub trait QuantScheme: Send + Sync {
    /// Human-readable name for this scheme.
    fn name(&self) -> &str;

    /// Bits per weight element.
    fn bits(&self) -> u8;

    /// Encode a slice of f32 weights into quantized bytes.
    fn encode(&self, weights: &[f32]) -> Result<QuantizedTensor>;

    /// Decode quantized bytes back to f32 weights.
    fn decode(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>>;
}

/// A quantized tensor: the compressed data plus metadata needed for decoding.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub scale: Vec<f32>,
    pub zero_point: Vec<i32>,
    pub shape: Vec<usize>,
    pub bits: u8,
}

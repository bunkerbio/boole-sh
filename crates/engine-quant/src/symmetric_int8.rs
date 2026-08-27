use crate::scheme::{QuantScheme, QuantizedTensor};
use anyhow::Result;

/// Naive symmetric per-tensor int8 quantization.
/// Exists to prove the abstraction boundary — not optimized.
pub struct SymmetricInt8;

impl QuantScheme for SymmetricInt8 {
    fn name(&self) -> &str {
        "symmetric-int8"
    }

    fn bits(&self) -> u8 {
        8
    }

    fn encode(&self, weights: &[f32]) -> Result<QuantizedTensor> {
        let abs_max = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };

        let data: Vec<u8> = weights
            .iter()
            .map(|&w| {
                let q = (w / scale).round().clamp(-128.0, 127.0) as i8;
                q as u8
            })
            .collect();

        Ok(QuantizedTensor {
            data,
            scale: vec![scale],
            zero_point: vec![0],
            shape: vec![weights.len()],
            bits: 8,
        })
    }

    fn decode(&self, quantized: &QuantizedTensor) -> Result<Vec<f32>> {
        let scale = quantized.scale[0];
        let weights: Vec<f32> = quantized
            .data
            .iter()
            .map(|&b| (b as i8) as f32 * scale)
            .collect();
        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let scheme = SymmetricInt8;
        let weights = vec![0.5, -0.3, 1.0, -1.0, 0.0, 0.7];
        let quantized = scheme.encode(&weights).unwrap();
        let decoded = scheme.decode(&quantized).unwrap();

        assert_eq!(decoded.len(), weights.len());
        for (orig, dec) in weights.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.02, "orig={orig}, decoded={dec}");
        }
    }
}

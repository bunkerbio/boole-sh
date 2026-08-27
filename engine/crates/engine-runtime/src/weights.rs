use anyhow::{Result, bail};
use candle_core::{DType as CandleDType, Device, Tensor as CandleTensor};
use engine_core::{DType, Tensor};
use half::{bf16, f16};
use std::collections::HashMap;

pub struct Weights {
    tensors: HashMap<String, Tensor>,
    device: Device,
}

impl Weights {
    pub fn new(tensors: HashMap<String, Tensor>, device: Device) -> Self {
        Self { tensors, device }
    }

    pub fn get(&self, name: &str) -> Result<CandleTensor> {
        let t = self.tensors.get(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        to_candle(t, &self.device)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

fn to_candle(t: &Tensor, device: &Device) -> Result<CandleTensor> {
    let shape = t.shape();
    let bytes = t.data_bytes();

    match t.dtype() {
        DType::F32 => {
            let data: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, t.numel())
            };
            Ok(CandleTensor::from_slice(data, shape, device)?)
        }
        DType::F16 => {
            let data: &[f16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f16, t.numel())
            };
            let ct = CandleTensor::from_slice(data, shape, device)?;
            Ok(ct.to_dtype(CandleDType::F32)?)
        }
        DType::BF16 => {
            let data: &[bf16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const bf16, t.numel())
            };
            let ct = CandleTensor::from_slice(data, shape, device)?;
            Ok(ct.to_dtype(CandleDType::F32)?)
        }
        DType::Quantized { .. } => {
            bail!("direct candle conversion for quantized tensors not supported — use engine-quant decode first");
        }
    }
}

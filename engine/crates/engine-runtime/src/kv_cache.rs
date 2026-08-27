use anyhow::Result;
use candle_core::{Device, Tensor};

pub struct KvCache {
    layers: Vec<LayerCache>,
}

struct LayerCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerCache { k: None, v: None })
            .collect();
        Self { layers }
    }

    pub fn append_and_get(
        &mut self,
        layer: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let cache = &mut self.layers[layer];

        let full_k = match &cache.k {
            Some(prev) => Tensor::cat(&[prev, new_k], 1)?,
            None => new_k.clone(),
        };
        let full_v = match &cache.v {
            Some(prev) => Tensor::cat(&[prev, new_v], 1)?,
            None => new_v.clone(),
        };

        cache.k = Some(full_k.clone());
        cache.v = Some(full_v.clone());

        Ok((full_k, full_v))
    }

    pub fn seq_len(&self) -> usize {
        self.layers[0]
            .k
            .as_ref()
            .map(|t| t.dim(1).unwrap_or(0))
            .unwrap_or(0)
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.k = None;
            layer.v = None;
        }
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        for layer in &mut self.layers {
            if let Some(k) = &layer.k {
                layer.k = Some(k.to_device(device)?);
            }
            if let Some(v) = &layer.v {
                layer.v = Some(v.to_device(device)?);
            }
        }
        Ok(())
    }
}

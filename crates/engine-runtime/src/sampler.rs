/// Token sampling strategies.
#[derive(Debug, Clone)]
pub enum Sampler {
    Greedy,
    Temperature { temp: f32, top_p: f32 },
}

impl Sampler {
    pub fn sample(&self, logits: &[f32]) -> u32 {
        match self {
            Sampler::Greedy => {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            }
            Sampler::Temperature { .. } => {
                // TODO: implement temperature + top-p sampling in milestone 4
                self.greedy_fallback(logits)
            }
        }
    }

    fn greedy_fallback(&self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }
}

use rand::Rng;

#[derive(Debug, Clone)]
pub enum Sampler {
    Greedy,
    Temperature { temp: f32, top_p: f32 },
}

impl Sampler {
    pub fn sample(&self, logits: &[f32]) -> u32 {
        match self {
            Sampler::Greedy => argmax(logits),
            Sampler::Temperature { temp, top_p } => {
                let mut probs = softmax_with_temperature(logits, *temp);
                top_p_filter(&mut probs, *top_p);
                sample_from_distribution(&probs)
            }
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

fn softmax_with_temperature(logits: &[f32], temp: f32) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| ((x - max_val) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    probs
}

fn top_p_filter(probs: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumulative = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumulative += p;
        if cumulative > top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    let kept: std::collections::HashSet<usize> = indexed[..cutoff_idx].iter().map(|(i, _)| *i).collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !kept.contains(&i) {
            *p = 0.0;
        }
    }

    // Renormalize
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

fn sample_from_distribution(probs: &[f32]) -> u32 {
    let mut rng = rand::rng();
    let r: f32 = rng.random();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

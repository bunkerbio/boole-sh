/// KV cache for incremental token generation.
/// Stores key/value tensors per layer for O(1) per-token decode cost.
///
/// Implementation deferred to milestone 4.
pub struct KvCache {
    pub num_layers: usize,
    pub max_seq_len: usize,
}

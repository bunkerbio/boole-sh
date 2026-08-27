use engine_formats::ModelConfig;

/// Top-level model: holds config + layer weights + manages forward pass.
/// Implementation deferred to milestone 3.
pub struct Model {
    pub config: ModelConfig,
}

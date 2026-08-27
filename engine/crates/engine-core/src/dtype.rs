#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Quantized { bits: u8, group_size: usize },
}

impl DType {
    pub fn size_in_bytes(&self) -> Option<usize> {
        match self {
            DType::F32 => Some(4),
            DType::F16 | DType::BF16 => Some(2),
            DType::Quantized { .. } => None,
        }
    }
}

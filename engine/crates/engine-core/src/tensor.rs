use crate::dtype::DType;
use crate::storage::Storage;
use std::sync::Arc;

/// A tensor view over a contiguous storage region.
pub struct Tensor {
    storage: Arc<Storage>,
    offset: usize,
    shape: Vec<usize>,
    dtype: DType,
}

impl Tensor {
    pub fn new(storage: Arc<Storage>, offset: usize, shape: Vec<usize>, dtype: DType) -> Self {
        Self { storage, offset, shape, dtype }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn data_bytes(&self) -> &[u8] {
        let size = self.numel() * self.dtype.size_in_bytes().unwrap_or(1);
        &self.storage.as_bytes()[self.offset..self.offset + size]
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32);
        let bytes = self.data_bytes();
        unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.numel())
        }
    }
}

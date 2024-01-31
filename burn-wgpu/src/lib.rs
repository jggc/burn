#![warn(missing_docs)]

//! Burn WGPU Backend

#[macro_use]
extern crate derive_new;
extern crate alloc;

mod ops;

/// Compute related module.
pub mod compute;
/// Kernel module
pub mod kernel;
/// Tensor module.
pub mod tensor;

pub(crate) mod codegen;
pub(crate) mod tune;

mod element;
pub use element::{FloatElement, IntElement};

mod device;
pub use device::*;

mod backend;
pub use backend::*;

mod graphics;
pub use graphics::*;

#[cfg(any(feature = "fusion", test))]
mod fusion;

#[cfg(test)]
mod tests {
    use super::*;

    pub type TestBackend = Wgpu;
    pub type ReferenceBackend = burn_ndarray::NdArray<f32>;

    pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> =
        burn_tensor::Tensor<TestBackend, D, burn_tensor::Bool>;

    pub type ReferenceTensor<const D: usize> = burn_tensor::Tensor<ReferenceBackend, D>;

    burn_tensor::testgen_all!();
    burn_autodiff::testgen_all!();
}

//! Abstraction layer for OpenCL and CUDA.
//!
//! Feature flags
//! -------------
//!
//! There are two [feature flags], `cuda` and `opencl`. By default `opencl` is enabled. You can
//! enable both at the same time. At least one of them needs to be enabled at any time.
//!
//! [feature flags]: https://doc.rust-lang.org/cargo/reference/manifest.html#the-features-section

#![warn(missing_docs)]

mod device;
mod error;
#[cfg(any(feature = "cuda", feature = "opencl"))]
mod program;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;

pub use device::{Device, DeviceUuid, Framework, PciId, UniqueId, Vendor};
pub use error::GPUError;
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use program::Program;

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
compile_error!("At least one of the features `cuda` or `opencl` must be enabled.");

/// A buffer on the GPU.
///
/// The concept of a local buffer is from OpenCL. In CUDA you don't allocate a buffer directly
/// via API call. Instead you pass in the amount of shared memory that should be used.
///
/// There can be at most a single local buffer per kernel. On CUDA a null pointer will be passed
/// in, instead of an actual value. The memory that should get allocated is then passed into the
/// kernel call automatically.
#[derive(Debug)]
pub struct LocalBuffer<T> {
    /// The number of T sized elements.
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> LocalBuffer<T> {
    /// Returns a new buffer of the specified `length`.
    pub fn new(length: usize) -> Self {
        LocalBuffer::<T> {
            length,
            _phantom: std::marker::PhantomData,
        }
    }
}

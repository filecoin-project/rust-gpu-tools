//! Abstraction layer for OpenCL, CUDA, and Metal.
//!
//! Feature flags
//! -------------
//!
//! There are three [feature flags], `cuda`, `opencl`, and `metal`. By default `opencl` is enabled. You can
//! enable all of them at the same time. At least one of them needs to be enabled at any time.
//!
//! [feature flags]: https://doc.rust-lang.org/cargo/reference/manifest.html#the-features-section

#![warn(missing_docs)]

mod device;
mod error;
#[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
mod program;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "metal")]
pub mod metal;

pub use device::{Device, DeviceUuid, Framework, PciId, UniqueId, Vendor};
pub use error::GPUError;
#[cfg(any(feature = "cuda", feature = "opencl", feature = "metal"))]
pub use program::Program;

#[cfg(not(any(feature = "cuda", feature = "opencl", feature = "metal")))]
compile_error!("At least one of the features `cuda`, `opencl`, or `metal` must be enabled.");

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

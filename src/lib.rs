//! Abstraction layer for OpenCL and CUDA.
//!
//! Feature Flags
//! -------------
//!
//! There are two [feature flags], `cuda` and `opencl`. By default `opencl` is enabled. You can
//! enable both at the same time. At least one of them needs to be enabled at any time.
//!
//! [feature flags]: https://doc.rust-lang.org/cargo/reference/manifest.html#the-features-section

mod device;
mod error;
#[cfg(all(feature = "cuda", feature = "opencl"))]
mod program;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;

pub use device::{Device, DeviceUuid, Framework, PciId, UniqueId, Vendor};
pub use error::GPUError;
#[cfg(all(feature = "cuda", feature = "opencl"))]
pub use program::Program;

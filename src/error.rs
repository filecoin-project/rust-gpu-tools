#[cfg(feature = "opencl")]
use opencl3::{device::DeviceInfo, error_codes::ClError, program::ProgramInfo};
#[cfg(feature = "cuda")]
use rustacuda::error::CudaError;

#[derive(thiserror::Error, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum GPUError {
    #[cfg(feature = "opencl")]
    #[error("Opencl3 Error: {0}{}", match .1 {
       Some(message) => format!(" {}", message),
       None => "".to_string(),
    })]
    Opencl3(ClError, Option<String>),

    #[cfg(feature = "opencl")]
    #[error("Program info not available!")]
    ProgramInfoNotAvailable(ProgramInfo),

    #[cfg(feature = "opencl")]
    #[error("Device info not available!")]
    DeviceInfoNotAvailable(DeviceInfo),

    #[cfg(feature = "cuda")]
    #[error("Cuda Error: {0}")]
    Cuda(#[from] CudaError),

    #[error("Device not found!")]
    DeviceNotFound,

    #[error("Kernel with name {0} not found!")]
    KernelNotFound(String),

    #[error("IO Error: {0}")]
    IO(#[from] std::io::Error),

    #[error("Vendor {0} is not supported.")]
    UnsupportedVendor(String),

    #[error("{0}")]
    InvalidId(String),
}

#[allow(clippy::upper_case_acronyms)]
pub type GPUResult<T> = std::result::Result<T, GPUError>;

#[cfg(feature = "opencl")]
impl From<ClError> for GPUError {
    fn from(error: ClError) -> Self {
        GPUError::Opencl3(error, None)
    }
}

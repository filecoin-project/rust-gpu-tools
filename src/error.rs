#[cfg(feature = "opencl")]
use opencl3::error_codes::ClError;
#[cfg(feature = "cuda")]
use rustacuda::error::CudaError;

/// Error types of this library.
#[derive(thiserror::Error, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum GPUError {
    /// Error from the underlying `opencl3` library, e.g. a memory allocation failure.
    #[cfg(feature = "opencl")]
    #[error("Opencl3 Error: {0}{}", match .1 {
       Some(message) => format!(" {}", message),
       None => "".to_string(),
    })]
    Opencl3(ClError, Option<String>),

    /// Error for OpenCL `clGetProgramInfo()` call failures.
    #[cfg(feature = "opencl")]
    #[error("Program info not available!")]
    ProgramInfoNotAvailable(ClError),

    /// Error for OpenCL `clGetDeviceInfo()` call failures.
    #[cfg(feature = "opencl")]
    #[error("Device info not available!")]
    DeviceInfoNotAvailable(ClError),

    /// Error from the underlying `RustaCUDA` library, e.g. a memory allocation failure.
    #[cfg(feature = "cuda")]
    #[error("Cuda Error: {0}")]
    Cuda(#[from] CudaError),

    /// Error when a device cannot be found.
    #[error("Device not found!")]
    DeviceNotFound,

    /// Error when a kernel with the given name cannot be found.
    #[error("Kernel with name {0} not found!")]
    KernelNotFound(String),

    /// Error when standard I/O fails.
    #[error("IO Error: {0}")]
    IO(#[from] std::io::Error),

    /// Error when the device is from an unsupported vendor.
    #[error("Vendor {0} is not supported.")]
    UnsupportedVendor(String),

    /// Error when the string representation of a unique identifier (PCI-ID or UUID) cannot be
    /// parsed.
    #[error("{0}")]
    InvalidId(String),

    /// Errors that rarely happen and don't deserve their own error type.
    #[error("{0}")]
    Generic(String),
}

/// Convenience type alias for [`GPUError`] based [`Result`]s.
#[allow(clippy::upper_case_acronyms)]
pub type GPUResult<T> = std::result::Result<T, GPUError>;

#[cfg(feature = "opencl")]
impl From<ClError> for GPUError {
    fn from(error: ClError) -> Self {
        GPUError::Opencl3(error, None)
    }
}

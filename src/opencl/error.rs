use opencl3::{device::DeviceInfo, error_codes::ClError, program::ProgramInfo};

#[derive(thiserror::Error, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum GPUError {
    #[error("Opencl3 Error: {0}{}", match .1 {
       Some(message) => format!(" {}", message),
       None => "".to_string(),
    })]
    Opencl3(ClError, Option<String>),
    #[error("Device not found!")]
    DeviceNotFound,
    #[error("Device info not available!")]
    DeviceInfoNotAvailable(DeviceInfo),
    #[error("Program info not available!")]
    ProgramInfoNotAvailable(ProgramInfo),
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
#[allow(dead_code)]
pub type GPUResult<T> = std::result::Result<T, GPUError>;

impl From<ClError> for GPUError {
    fn from(error: ClError) -> Self {
        GPUError::Opencl3(error, None)
    }
}

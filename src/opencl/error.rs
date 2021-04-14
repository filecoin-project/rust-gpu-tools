#[derive(thiserror::Error, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum GPUError {
    #[error("Ocl Error: {0}")]
    Ocl(ocl::Error),
    #[error("Device not found!")]
    DeviceNotFound,
    #[error("Device info not available!")]
    DeviceInfoNotAvailable(ocl::enums::DeviceInfo),
    #[error("Program info not available!")]
    ProgramInfoNotAvailable(ocl::enums::ProgramInfo),
    #[error("IO Error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Cannot parse UUID, expected hex-encoded string formated as aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, got {0}.")]
    Uuid(String),
}

#[allow(clippy::upper_case_acronyms)]
#[allow(dead_code)]
pub type GPUResult<T> = std::result::Result<T, GPUError>;

impl From<ocl::Error> for GPUError {
    fn from(error: ocl::Error) -> Self {
        GPUError::Ocl(error)
    }
}

impl From<ocl::core::Error> for GPUError {
    fn from(error: ocl::core::Error) -> Self {
        GPUError::Ocl(error.into())
    }
}

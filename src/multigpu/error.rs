#[derive(thiserror::Error, Debug)]
pub enum SchedulerError {
    #[error("Cannot parse task file!")]
    ParseError,
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type SchedulerResult<T> = std::result::Result<T, SchedulerError>;

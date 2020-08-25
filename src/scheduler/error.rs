#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Channel Error: {0}")]
    Channel(#[from] std::sync::mpsc::RecvError),
    #[error("IO Error: {0}")]
    IO(#[from] std::io::Error),
}

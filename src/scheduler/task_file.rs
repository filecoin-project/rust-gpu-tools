use log::debug;
use std::fs::{remove_file, File};
use std::hash::Hasher;
use std::io::Error;
use std::path::PathBuf;

#[derive(Debug)]
pub(crate) struct TaskFile {
    pub(crate) file: File,
    pub(crate) path: PathBuf,
}

impl PartialEq for TaskFile {
    fn eq(&self, other: &Self) -> bool {
        self.path.eq(&other.path)
    }
}

impl Eq for TaskFile {}

impl std::hash::Hash for TaskFile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state)
    }
}

impl TaskFile {
    pub(crate) fn destroy(&self) -> Result<(), Error> {
        remove_file(self.path.clone())?;
        debug!("Removing TaskFile from queue");
        Ok(())
    }
}

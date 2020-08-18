use fs2::FileExt;
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
    /// Destroy the underlying file if it can be locked immediately.
    pub(crate) fn try_destroy(&self) -> Result<(), Error> {
        let file = File::open(self.path.clone())?;
        if file.try_lock_exclusive().is_err() {
            debug!(
                "Not removing TaskFile from queue: {:?}! (Could not acquire lock.)",
                file
            );
        } else {
            remove_file(self.path.clone())?;
            debug!("Removing TaskFile from queue: {:?}", file);
        }
        Ok(())
    }

    pub(crate) fn destroy(&self) -> Result<(), Error> {
        if self.path.exists() {
            remove_file(self.path.clone())?;
        }
        Ok(())
    }
}

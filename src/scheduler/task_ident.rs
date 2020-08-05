use std::fs::{remove_file, File};
use std::io::Error;
use std::path::PathBuf;
use std::str::FromStr;

use fs2::FileExt;
use log::debug;

use super::task_file::TaskFile;
use super::{Priority, PROCESS_ID};

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) struct TaskIdent {
    pub(crate) priority: Priority,
    pub(crate) name: String,
    id: usize,
}

/// `TaskIdent`s must uniquely identify `Task`s, so must be created with
/// `SchedulerRoot::new_ident` — which manages the id counter.
impl TaskIdent {
    pub(crate) fn new(priority: Priority, name: String, id: usize) -> Self {
        Self { priority, name, id }
    }

    fn path(&self, dir: &PathBuf) -> PathBuf {
        let filename = self.to_string();
        dir.join(filename)
    }

    pub(crate) fn enqueue_in_dir(&self, dir: &PathBuf) -> Result<TaskFile, Error> {
        debug!(
            "Enqueueing TaskFile in queue {:?}: {}",
            dir,
            self.to_string()
        );
        let path = self.path(dir);
        let file = File::create(path.clone())?;
        file.lock_exclusive()?;
        debug!("Enqueued TaskFile in queue {:?}: {}", dir, self.to_string());
        Ok(TaskFile {
            file,
            path: path.to_path_buf(),
        })
    }
    pub(crate) fn try_destroy(&self, dir: &PathBuf) -> Result<(), Error> {
        let path = self.path(dir);
        let file = File::open(path.clone())?;
        if file.try_lock_exclusive().is_err() {
            debug!(
                "Not removing TaskFile from queue {:?}: {}",
                dir,
                self.to_string()
            );
        } else {
            remove_file(path)?;
            debug!(
                "Removing TaskFile from queue {:?}: {}",
                dir,
                self.to_string()
            );
        };
        Ok(())
    }
}

impl ToString for TaskIdent {
    fn to_string(&self) -> String {
        format!(
            "{priority}-{process}-{name}-{id}",
            priority = self.priority,
            process = *PROCESS_ID,
            name = self.name,
            id = self.id,
        )
    }
}

impl FromStr for TaskIdent {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = s.split("-").collect();
        let priority: Priority = parts.get(0).map(|s| s.parse().unwrap()).unwrap();
        // Ignore the 'process' segment.
        let name = parts.get(2).unwrap_or(&"").to_string();
        let id = parts.get(3).unwrap_or(&"0").parse().unwrap_or(0); // FIXME: How should we actually handle a bad identifier string?
        Ok(Self { priority, name, id })
    }
}

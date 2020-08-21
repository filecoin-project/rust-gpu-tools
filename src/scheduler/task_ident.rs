use fs2::FileExt;
use log::debug;
use std::fs::{remove_file, File};
use std::io::Error;
use std::path::PathBuf;
use std::str::FromStr;

use super::task_file::TaskFile;
use super::{Priority, PROCESS_ID};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TaskIdent {
    pub(crate) priority: Priority,
    pub(crate) name: String,
    id: usize,
    is_preemptible: bool,
}

/// `TaskIdent`s must uniquely identify `Task`s, so must be created with
/// `SchedulerRoot::new_ident` — which manages the id counter.
impl TaskIdent {
    pub(crate) fn new(priority: Priority, name: &str, id: usize, is_preemptible: bool) -> Self {
        Self {
            priority,
            name: name.to_string(),
            id,
            is_preemptible,
        }
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

    pub(crate) fn has_priority_to_preempt(&self) -> bool {
        self.priority == 0
    }

    pub(crate) fn try_destroy(&self, dir: &PathBuf) -> Result<(), Error> {
        let path = self.path(dir);
        if path.exists() {
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
        }
        Ok(())
    }
    pub(crate) fn destroy(&self, dir: &PathBuf) -> Result<(), Error> {
        let path = self.path(dir);
        if path.exists() {
            let file = File::open(path.clone())?;
            file.lock_exclusive()?;
            remove_file(path)?;
            debug!(
                "Removing TaskFile from queue {:?}: {}",
                dir,
                self.to_string()
            );
        }
        Ok(())
    }
}

impl ToString for TaskIdent {
    fn to_string(&self) -> String {
        format!(
            "{priority}-{process}-{name}-{id}-{preemptible}",
            priority = self.priority,
            process = *PROCESS_ID,
            name = self.name,
            id = self.id,
            preemptible = if self.is_preemptible { "P" } else { "X" },
        )
    }
}

impl FromStr for TaskIdent {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // TODO: Refactor this error handling (but preserve the behavior).
        let parts: Vec<_> = s.split("-").collect();
        let priority: Priority = if let Some(p) = parts.get(0).map(|s| s.parse().ok()) {
            if let Some(p2) = p {
                p2
            } else {
                return Err(Error::new(std::io::ErrorKind::InvalidInput, s));
            }
        } else {
            return Err(Error::new(std::io::ErrorKind::InvalidInput, s));
        };
        let name = parts.get(2).unwrap_or(&"").to_string();
        let is_preemptible = if let Some(p) = parts.get(4) {
            if p == &"P" {
                true
            } else if p == &"X" {
                false
            } else {
                return Err(Error::new(std::io::ErrorKind::InvalidInput, s));
            }
        } else {
            return Err(Error::new(std::io::ErrorKind::InvalidInput, s));
        };
        if let Some(id1) = parts.get(3).map(|s| s.parse().ok()) {
            if let Some(id) = id1 {
                Ok(Self {
                    priority,
                    name,
                    id,
                    is_preemptible,
                })
            } else {
                return Err(Error::new(std::io::ErrorKind::InvalidInput, s));
            }
        } else {
            return Err(Error::new(std::io::ErrorKind::InvalidInput, s));
        }
    }
}

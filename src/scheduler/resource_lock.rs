use std::fs::File;
use std::io::Error;
use std::path::PathBuf;

use fs2::FileExt;
use log::debug;

use super::Resource;

const LOCK_NAME: &str = "resource.lock";
const PREEMPTION_LOCK_NAME: &str = "resource-preemption.lock";

#[derive(Debug)]
pub(crate) struct ResourceLock {
    /// ResourceLock holds a reference to lockfile.
    file: File,
    resource_name: String,
    /// If `preemption` is true, this lock signals the presence of a request to preempt.
    /// The resulting lock file should then be held until the primary lock file is released.
    preemption: bool,
}

impl ResourceLock {
    pub(crate) fn acquire<R: Resource>(
        dir: &PathBuf,
        resource: &R,
        preemption: bool,
    ) -> Result<ResourceLock, Error> {
        debug!("Acquiring lock for {}...", resource.name());
        let lockfile_path = Self::lockfile_path(dir, preemption);
        let file = File::create(lockfile_path)?;
        file.lock_exclusive()?;
        debug!("Resource lock acquired for {}!", resource.name());
        Ok(Self {
            file,
            resource_name: resource.name(),
            preemption,
        })
    }

    pub(crate) fn maybe_acquire<R: Resource>(
        dir: &PathBuf,
        resource: &R,
        preemption: bool,
    ) -> Result<Option<ResourceLock>, Error> {
        debug!("Acquiring lock for {}...", resource.name());
        let lockfile_path = Self::lockfile_path(dir, preemption);
        let file = File::create(lockfile_path)?;
        if file.try_lock_exclusive().is_err() {
            debug!("Could not acquire lock for {}.", resource.name());
            return Ok(None);
        }

        debug!("Resource lock acquired for {}!", resource.name());
        Ok(Some(Self {
            file,
            resource_name: resource.name(),
            preemption,
        }))
    }

    pub(crate) fn is_held(dir: &PathBuf, preemption: bool) -> bool {
        let lockfile_path = Self::lockfile_path(dir, preemption);
        let file = File::create(lockfile_path).unwrap();
        if file.try_lock_exclusive().is_err() {
            true
        } else {
            false
        }
    }

    fn lockfile_path(dir: &PathBuf, preemption: bool) -> PathBuf {
        dir.join(if preemption {
            PREEMPTION_LOCK_NAME
        } else {
            LOCK_NAME
        })
    }
}

impl Drop for ResourceLock {
    fn drop(&mut self) {
        // Lock will have been released when `file` is dropped.
        debug!("Resource lock for {} released!", self.resource_name);
    }
}

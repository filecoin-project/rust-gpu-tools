use std::fs::File;
use std::io::Error;
use std::path::PathBuf;

use fs2::FileExt;
use log::debug;

use super::Resource;

const LOCK_NAME: &str = "resource.lock";

#[derive(Debug)]
pub(crate) struct ResourceLock {
    /// ResourceLock holds a reference to lockfile.
    file: File,
    resource_name: String,
}

impl ResourceLock {
    pub(crate) fn acquire(dir: &PathBuf, resource: &dyn Resource) -> Result<ResourceLock, Error> {
        debug!("Acquiring lock for {}...", resource.name());
        let lockfile_path = dir.join(LOCK_NAME);
        let file = File::create(lockfile_path)?;
        file.lock_exclusive()?;
        debug!("Resource lock acquired for {}!", resource.name());
        Ok(Self {
            file,
            resource_name: resource.name(),
        })
    }

    pub(crate) fn maybe_acquire(
        dir: &PathBuf,
        resource: &dyn Resource,
    ) -> Result<Option<ResourceLock>, Error> {
        debug!("Acquiring lock for {}...", resource.name());
        let lockfile_path = dir.join(LOCK_NAME);
        let file = File::create(lockfile_path)?;
        if file.try_lock_exclusive().is_err() {
            debug!("Could not acquire lock for {}.", resource.name());
            return Ok(None);
        }

        debug!("Resource lock acquired for {}!", resource.name());
        Ok(Some(Self {
            file,
            resource_name: resource.name(),
        }))
    }
}

impl Drop for ResourceLock {
    fn drop(&mut self) {
        // Lock will have been released when `file` is dropped.
        debug!("Resource lock for {} released!", self.resource_name);
    }
}

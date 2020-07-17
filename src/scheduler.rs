extern crate rand;

use ::lazy_static::lazy_static;

use fs2::FileExt;
use log::debug;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, create_dir_all, remove_file, File};
use std::hash::{Hash, Hasher};
use std::io::Error;
use std::iter;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Mutex;

/// How often, in milliseconds, should we poll?
const POLL_INTERVAL_MS: usize = 1000;
const LOCK_NAME: &str = "resource.lock";

lazy_static! {
    static ref PROCESS_ID: String = iter::repeat(())
        .map(|()| thread_rng().sample(Alphanumeric))
        .take(10)
        .collect();
}

pub trait Resource {
    fn id(&self) -> String;
    fn name(&self) -> String {
        format!("Resource #{}", self.id())
    }
}

#[derive(Debug)]
pub struct ResourceLock {
    file: File,
    name: String,
}

impl ResourceLock {
    pub fn lock(path: &PathBuf, resource: &dyn Resource) -> Result<ResourceLock, Error> {
        debug!("Acquiring lock for {}...", resource.name());
        let file = File::create(path.join(LOCK_NAME))?;
        file.lock_exclusive()?;
        debug!("Resource lock acquired for {}!", resource.name());
        Ok(Self {
            file,
            name: resource.name(),
        })
    }
}

impl Drop for ResourceLock {
    fn drop(&mut self) {
        // Lock will have been released when `file` was dropped.
        debug!("Resource lock released!");
    }
}

#[derive(Debug)]
pub struct TaskFile {
    file: File,
    path: PathBuf,
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

impl Clone for TaskFile {
    fn clone(&self) -> Self {
        Self {
            file: File::try_clone(&self.file).unwrap(),
            path: self.path.clone(),
        }
    }
}

impl TaskFile {
    fn path(dir: &PathBuf, ident: &TaskIdent) -> PathBuf {
        let filename = ident.to_string();
        dir.join(filename)
    }
    fn create(dir: &PathBuf, ident: &TaskIdent) -> Result<TaskFile, Error> {
        debug!("Enqueueing TaskFile");
        let path = Self::path(dir, ident);
        let file = File::create(path.clone())?;
        file.lock_exclusive()?;
        debug!("Enqueued TaskFile");
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }
    fn try_destroy_for_ident(dir: &PathBuf, ident: &TaskIdent) -> Result<(), Error> {
        let path = Self::path(dir, ident);
        let file = File::open(path.clone())?;
        file.try_lock_exclusive()?;
        remove_file(path)?;
        debug!("Removing TaskFile from queue");
        Ok(())
    }

    fn destroy(&self) -> Result<(), Error> {
        remove_file(self.path.clone())?;
        debug!("Removing TaskFile from queue");
        Ok(())
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
pub struct Priority(usize);

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
enum Process {
    Own,
    Other,
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct TaskIdent {
    priority: Priority,
    name: String,
    process: Process,
}

impl ToString for TaskIdent {
    fn to_string(&self) -> String {
        match self.process {
            Process::Own => format!(
                "{priority}-{process}-{name}",
                priority = self.priority.0,
                process = *PROCESS_ID,
                name = self.name
            ),
            Process::Other => panic!("cannot format ident for other proceses"),
        }
    }
}

impl FromStr for TaskIdent {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = s.split("-").collect();
        let priority: Priority = Priority(parts.get(0).map(|s| s.parse().unwrap()).unwrap());
        let process = parts
            .get(1)
            .map(|p| {
                if *p == &*PROCESS_ID {
                    Process::Own
                } else {
                    Process::Other
                }
            })
            .unwrap();
        let name = parts.get(2).unwrap_or(&"");

        Ok(Self {
            priority,
            process,
            name: name.to_string(),
        })
    }
}

/// A TaskExecutor acts as a callback which executes the job associated with a task.
trait TaskExecutor<R: Resource> {
    /// `execute` executes a task's job. `preempt.preempt_now()` should be polled as appropriate,
    /// and execution should terminate if it returns true. Tasks which are no preemptible need not
    /// ever check for preemption.
    fn execute(&self, preempt: &dyn Preemption<R>);

    /// Returns true if the job associated with this `TaskExecutor` can be preempted. Executors
    /// which return `true` should periodically poll for preemption while executing.
    fn preemptible(&self) -> bool;
}

// #[derive(Clone)]
pub struct Task<'a, R: Resource> {
    ident: TaskIdent,
    /// These are the resources for which the `Task` has been requested to be scheduled,
    /// in order of preference. It is guaranteed that the `Task` will be scheduled on only one of these.
    resources: Option<Vec<R>>,
    executor: &'a dyn TaskExecutor<R>,
}

impl<'a, R: Resource> Task<'a, R> {
    fn perform(&self, path: &PathBuf, resource: &R, scheduler: &FSResourceScheduler<R>) {
        let lock = ResourceLock::lock(path, resource);
        self.executor.execute(scheduler)
    }
}

pub struct FSScheduler<'a, R: Resource> {
    root: PathBuf,
    task_files: HashMap<TaskIdent, HashSet<TaskFile>>,
    own_tasks: HashMap<TaskIdent, Mutex<&'a Task<'a, R>>>,
    _r: PhantomData<R>,
}

impl<'a, R: Resource> FSScheduler<'a, R> {
    fn new(root: PathBuf) -> Result<Self, Error> {
        create_dir_all(&root)?;
        Ok(Self {
            root,
            task_files: Default::default(),
            own_tasks: Default::default(),
            _r: PhantomData,
        })
    }
    pub fn schedule(&mut self, task: &'a Task<'a, R>, resources: &[R]) -> Result<(), Error> {
        for resource in resources {
            self.schedule_for_resource(&task, resource);
        }
        Ok(())
    }
    pub fn schedule_for_resource(
        &mut self,
        task: &'a Task<'a, R>,
        resource: &R,
    ) -> Result<(), Error> {
        let dir = self.root.join(resource.id());
        create_dir_all(&dir);
        let task_file = TaskFile::create(&dir, &task.ident.clone())?;
        self.task_files
            .entry(task.ident.clone())
            .or_insert(Default::default())
            .insert(task_file);
        self.own_tasks.insert(task.ident.clone(), Mutex::new(&task));
        Ok(())
    }
}

pub struct FSResourceScheduler<'a, R: Resource> {
    root_scheduler: FSScheduler<'a, R>,
    dir: PathBuf,
    resource: R,
    previous: Option<TaskIdent>,
}

impl<'a, R: Resource> FSResourceScheduler<'a, R> {
    pub fn lock(&self) -> Result<ResourceLock, Error> {
        ResourceLock::lock(&self.dir, &self.resource)
    }
}

trait Preemption<R: Resource> {
    // Return true if task should be preempted now.
    fn preempt_now(&self, _task: &Task<R>) -> bool;
}

impl<'a, R: Resource> FSResourceScheduler<'a, R> {
    pub fn handle_next(&mut self) -> Result<(), Error> {
        assert!(self.dir.is_dir(), "scheduler dir is not a directory.");
        let mut ident_creations = fs::read_dir(&self.dir)?
            .map(|res| {
                res.map(|e| {
                    // FIXME: unwraps
                    let metadata = e.metadata().unwrap();
                    let task_ident = TaskIdent::from_str(&e.file_name().to_str().unwrap()).unwrap();
                    (task_ident, metadata.created().unwrap())
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        ident_creations.sort_by(|a, b| {
            let priority_ordering = a.0.priority.partial_cmp(&b.0.priority).unwrap();
            match priority_ordering {
                Ordering::Equal => a.1.partial_cmp(&b.1).unwrap(),
                _ => priority_ordering,
            }
        });

        let ident = if let Some((ident, _)) = ident_creations.get(0) {
            ident
        } else {
            // If there was no `TaskIdent` found, nothing to do.
            self.previous = None;
            return Ok(());
        };

        match ident.process {
            Process::Own => {
                // Lock the task so a sibling won't remove it.

                let mut performed_task = false;

                {
                    let mut lock = self
                        .root_scheduler
                        .own_tasks
                        .get(ident)
                        .expect("own task missing")
                        .try_lock();

                    if let Ok(ref mut mutex) = lock {
                        let task = &*mutex; // TODO: clean it up.
                        self.previous = None;

                        // We have the lock for this task, so we may destroy the sibling TaskFiles.
                        if let Some(all_task_files) = self.root_scheduler.task_files.get(ident) {
                            // FIXME: unwrap
                            all_task_files.iter().for_each(|task_file| {
                                // Don't destroy this directory's task file until we are done performing the task
                                if !task_file.path.starts_with(self.dir.clone()) {
                                    task_file.destroy().unwrap();
                                }
                            });
                        }

                        task.perform(&self.dir, &self.resource, &self);
                        performed_task = true;
                        // Finally, destroy this taskfile too.

                        // TODO: Destroy it.

                        // And remove the task
                        self.root_scheduler.task_files.remove(&ident);
                    } else {
                        // Task was already locked, which means this process has already assigned it to a different resource.
                        // Do nothing and allow it to be cleaned up (removed from this queue) as part of that assignment.
                    }

                    // lock is dropped here
                }

                if performed_task {
                    self.root_scheduler.own_tasks.remove(&ident);
                }
            }
            Process::Other => {
                if let Some(previous) = &self.previous {
                    // The same task has been 'next up' for two turns, so it has forfeited its turn.
                    // Since we discovered this, it is our job to destroy it.
                    // TODO: Should require one more round, since different will be on different schedules.
                    //       worst-case behavior of out-of-sync schedules gives no time for the actual winner to act.
                    let _ = TaskFile::try_destroy_for_ident(&self.dir, previous);
                // If this fails, someone else may have seized the lock and done it for us.
                } else {
                    // Remember this ident, so we can destroy it if it's still next when we check again.
                    self.previous = Some(ident.clone());
                }
            }
        }

        Ok(())
    }
}

impl<'a, R: Resource> Preemption<R> for FSResourceScheduler<'a, R> {
    fn preempt_now(&self, _task: &Task<R>) -> bool {
        todo!();
    }
}

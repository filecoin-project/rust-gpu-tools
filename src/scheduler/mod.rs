extern crate rand;

use ::lazy_static::lazy_static;

use self::task_ident::TaskIdent;
use fs2::FileExt;
use log::debug;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, create_dir_all, remove_file, File};
use std::hash::{Hash, Hasher};
use std::io::Error;
use std::iter;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::{mpsc, Mutex};
use std::thread;

use crate::scheduler::task_file::TaskFile;

mod task_file;
mod task_ident;

/// How often, in milliseconds, should we poll by default?
const POLL_INTERVAL_MS: usize = 1000;
const LOCK_NAME: &str = "resource.lock";

/// Lower values have 'higher' priority.
type Priority = usize;

lazy_static! {
    static ref PROCESS_ID: String = iter::repeat(())
        .map(|()| thread_rng().sample(Alphanumeric))
        .take(10)
        .collect();
}

pub trait Resource {
    /// `dir_id` uniquely identifies the directory associated with the resource.
    fn dir_id(&self) -> String;
    /// `name` is the descriptive name of the resource and defaults to wrapping `dir_id`.
    fn name(&self) -> String {
        format!("Resource #{}", self.dir_id())
    }
}

/// Implementers of `Executable` acts as a callback which executes the job associated with a task.
pub trait Executable<R: Resource> {
    /// `execute` executes a task's job. `preempt.should_preempt_now()` should be polled as appropriate,
    /// and execution should terminate if it returns true. Tasks which are not preemptible need not
    /// ever check for preemption.
    fn execute(&self, preempt: &dyn Preemption<R>);

    /// Returns true if the job associated with this `Executable` can be preempted. `Executable`s
    /// which return `true` should periodically poll for preemption while executing.
    fn is_preemptible(&self) -> bool;
}

pub trait Preemption<R: Resource> {
    // Return true if task should be preempted now.
    // `Executable`s which are preemptible, must call this method.
    fn should_preempt_now(&self, _task: &Task<R>) -> bool;
}

impl<'a, R: Resource> Preemption<R> for ResourceScheduler<'a, R> {
    /// The current `Task` should be preempted if the high-priority lock has been acquired
    /// by another `Task`.
    fn should_preempt_now(&self, _task: &Task<R>) -> bool {
        todo!();
    }
}

#[derive(Clone)]
pub struct Task<'a, R: Resource> {
    /// These are the resources for which the `Task` has been requested to be scheduled,
    /// in order of preference. It is guaranteed that the `Task` will be scheduled on only one of these.
    executable: &'a dyn Executable<R>,
}

impl<'a, R: Resource> Task<'a, R> {
    pub fn new(executable: &'a dyn Executable<R>) -> Self {
        Self { executable }
    }
}

pub struct Scheduler<'a, R: Resource> {
    scheduler_root: Rc<RefCell<SchedulerRoot<'a, R>>>,
    resource_schedulers: Vec<ResourceScheduler<'a, R>>,
    poll_interval: usize,
}

impl<'a, R: 'a + Resource + Copy> Scheduler<'a, R> {
    pub fn new(root: PathBuf) -> Result<Self, Error> {
        Self::new_with_poll_interval(root, POLL_INTERVAL_MS)
    }

    pub fn new_with_poll_interval(root: PathBuf, poll_interval: usize) -> Result<Self, Error> {
        let scheduler = SchedulerRoot::new(root)?;
        Ok(Self {
            scheduler_root: Rc::new(RefCell::new(scheduler)),
            resource_schedulers: Default::default(),
            poll_interval,
        })
    }

    pub fn schedule(
        &mut self,
        priority: usize,
        name: String,
        task: &'a Task<'a, R>,
        resources: &[R],
    ) -> Result<(), Error> {
        resources.iter().for_each(|r| {
            self.ensure_resource_scheduler(*r);
        });
        let task_ident = self.scheduler_root.borrow_mut().new_ident(priority, name);
        self.scheduler_root
            .borrow_mut()
            .schedule(task_ident, task, resources)
    }

    pub fn start(root: PathBuf) -> Result<mpsc::Sender<bool>, Error> {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut scheduler = Self::new(root).unwrap();

            loop {
                for s in scheduler.resource_schedulers.iter_mut() {
                    if !rx.try_recv().is_err() {
                        break;
                    };
                    s.handle_next().unwrap(); // FIXME
                }
            }
        });

        Ok(tx)
    }

    pub fn stop(control_chan: mpsc::Sender<bool>) -> Result<(), mpsc::SendError<bool>> {
        Ok(control_chan.send(true)?)
    }

    fn ensure_resource_scheduler(&mut self, resource: R) -> ResourceScheduler<'a, R> {
        let dir = self.scheduler_root.borrow().root.join(resource.dir_id());
        ResourceScheduler::new(self.scheduler_root.clone(), dir, resource)
    }
}

#[derive(Debug)]
struct ResourceLock {
    /// ResourceLock holds a reference to lockfile.
    file: File,
    resource_name: String,
}

impl ResourceLock {
    fn acquire(dir: &PathBuf, resource: &dyn Resource) -> Result<ResourceLock, Error> {
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
}

impl Drop for ResourceLock {
    fn drop(&mut self) {
        // Lock will have been released when `file` is dropped.
        debug!("Resource lock for {} released!", self.resource_name);
    }
}

struct SchedulerRoot<'a, R: Resource> {
    root: PathBuf,
    /// A given `Task` (identified uniquely by a `TaskIdent`) may have multiple `TaskFile`s associated,
    /// one per `Resource` for which it is currently scheduled (but only one `Resource` will eventually be assigned).
    task_files: HashMap<TaskIdent, HashSet<TaskFile>>,
    /// Each `Task` (identified uniquely by a `TaskIdent`) is protected by a `Mutex`.
    own_tasks: HashMap<TaskIdent, Mutex<&'a Task<'a, R>>>,
    children: RefCell<HashMap<String, ResourceScheduler<'a, R>>>,
    ident_counter: usize,
    _r: PhantomData<R>,
}

impl<'a, R: Resource> SchedulerRoot<'a, R> {
    fn new(root: PathBuf) -> Result<Self, Error> {
        create_dir_all(&root)?;
        Ok(Self {
            root,
            task_files: Default::default(),
            own_tasks: Default::default(),
            children: Default::default(),
            ident_counter: 0,
            _r: PhantomData,
        })
    }
    fn new_ident(&mut self, priority: Priority, name: String) -> TaskIdent {
        let id = self.ident_counter;
        self.ident_counter += 1;
        TaskIdent::new(priority, name, id)
    }
    fn schedule(
        &mut self,
        task_ident: TaskIdent,
        task: &'a Task<'a, R>,
        resources: &[R],
    ) -> Result<(), Error> {
        for resource in resources.iter() {
            let dir = self.root.join(resource.dir_id());
            create_dir_all(&dir)?;
            let task_file = task_ident.enqueue_in_dir(&dir)?;
            self.task_files
                .entry(task_ident.clone())
                .or_insert(Default::default())
                .insert(task_file);
            self.own_tasks.insert(task_ident.clone(), Mutex::new(&task));
        }
        Ok(())
    }
}

struct ResourceScheduler<'a, R: Resource> {
    root_scheduler: Rc<RefCell<SchedulerRoot<'a, R>>>,
    // root_scheduler: RefCell<Scheduler<'a, R>>,
    dir: PathBuf,
    resource: R,
    /// The previous 'next', and a count of how many times we have seen it as such.
    previous: Option<(TaskIdent, usize)>,
}

impl<'a, R: Resource> ResourceScheduler<'a, R> {
    fn new(
        root_scheduler: Rc<RefCell<SchedulerRoot<'a, R>>>,
        // root_scheduler: RefCell<Scheduler<'a, R>>,
        dir: PathBuf,
        resource: R,
    ) -> Self {
        Self {
            root_scheduler,
            dir,
            resource,
            previous: None,
        }
    }
    fn lock(&self) -> Result<ResourceLock, Error> {
        ResourceLock::acquire(&self.dir, &self.resource)
    }

    fn handle_next(&mut self) -> Result<(), Error> {
        assert!(self.dir.is_dir(), "scheduler dir is not a directory.");
        let mut ident_data = fs::read_dir(&self.dir)?
            .map(|res| {
                res.map(|e| {
                    // FIXME: unwraps
                    let metadata = e.metadata().unwrap();
                    let task_ident = TaskIdent::from_str(&e.file_name().to_str().unwrap()).unwrap();
                    let file = File::open(e.path()).unwrap();
                    let locked = file.try_lock_exclusive().is_err();
                    (task_ident, metadata.created().unwrap(), locked)
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        ident_data.sort_by(|(a_ident, a_create_date, _), (b_ident, b_create_date, _)| {
            // Sort first by (priority, creation date).
            let priority_ordering = a_ident.priority.partial_cmp(&b_ident.priority).unwrap();
            match priority_ordering {
                Ordering::Equal => a_create_date.partial_cmp(&b_create_date).unwrap(),
                _ => priority_ordering,
            }
        });

        let (ident, locked) = if let Some((ident, _, locked)) = ident_data.get(0) {
            (ident, *locked)
        } else {
            // If there was no `TaskIdent` found, nothing to do.
            // Forget about anything we saw before.
            self.previous = None;
            return Ok(());
        };
        let is_own = self.root_scheduler.borrow().own_tasks.get(ident).is_some();

        if is_own {
            // Task is owned by this process.

            let mut performed_task = false;
            {
                let root_scheduler = self.root_scheduler.borrow();
                // Lock the task so a sibling won't remove it.
                let mut guard_result = root_scheduler
                    .own_tasks
                    .get(ident)
                    .expect("own task missing")
                    .try_lock();

                if let Ok(ref mut guard) = guard_result {
                    let task = &*guard;
                    self.previous = None;

                    let mut to_destroy_later = None;

                    // We have the lock for this task, so we may destroy the sibling TaskFiles.
                    if let Some(all_task_files) = root_scheduler.task_files.get(ident) {
                        // FIXME: unwrap
                        all_task_files.iter().for_each(|task_file| {
                            // Don't destroy this directory's task file until we are done performing the task
                            if !task_file.path.starts_with(self.dir.clone()) {
                                task_file.destroy().unwrap();
                            // TODO: check that destroy fails gracefully if already gone.
                            } else {
                                to_destroy_later = Some(task_file);
                            }
                        });
                    }

                    self.perform_task(&task)?;
                    // NOTE: We must defer removing from `self.own_tasks` because the map is borrowed in this scope above.
                    performed_task = true;

                    // Finally, destroy this `TaskFile`, too — assuming it is necessary.
                    if let Some(task_file) = to_destroy_later {
                        task_file.destroy().unwrap()
                    };

                    // And remove the task
                    self.root_scheduler.borrow_mut().task_files.remove(&ident);
                } else {
                    // Task `Mutex` was already locked, which means this process has already assigned it to a different resource.
                    // Do nothing and allow it to be cleaned up (removed from this queue) as part of that assignment.
                }

                // lock is dropped here
            }

            if performed_task {
                // Now we can remove (see NOTE above).
                self.root_scheduler.borrow_mut().own_tasks.remove(&ident);
            }
        } else {
            // Task is owned by another process.
            if locked {
                self.previous = None;
            } else {
                self.previous = match &self.previous {
                    // The same unlocked task has been 'next up' for three turns, so it has forfeited its turn.
                    // Since we discovered this, it is our job to destroy it.
                    // We need to see it three times, since different processes will be on different schedules.
                    // Worst-case behavior of out-of-sync schedules gives no time for the actual winner to act.
                    Some((previous, n)) if previous == ident && *n >= 2 => {
                        // If this fails, someone else may have seized the lock and done it for us.
                        previous.try_destroy(&self.dir)?;
                        None
                    }

                    // Increment the count, so we can destroy this if we see it on top next time we check.
                    Some((previous, n)) if previous == ident => Some((previous.clone(), n + 1)),

                    // No match, forget.
                    Some(_) => None,

                    // Remember this ident,
                    None => Some((ident.clone(), 1)),
                }
            }
        }
        Ok(())
    }

    fn perform_task(&self, task: &Task<R>) -> Result<(), Error> {
        let _lock = self.lock()?;
        // Pass `self` so `Executable` can call `should_preempt_now` on it if needed.
        task.executable.execute(self);
        Ok(())
        // Lock is dropped, and therefore released here, at end of scope.
    }
}

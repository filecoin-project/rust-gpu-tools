extern crate futures;
extern crate rand;

use ::lazy_static::lazy_static;

use self::task_ident::TaskIdent;
use fs2::FileExt;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, create_dir_all, File};
use std::iter;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use self::{
    resource_lock::ResourceLock,
    task::{Preemption, Task},
    task_file::TaskFile,
};

mod error;
mod resource_lock;
mod task;
mod task_file;
mod task_ident;
pub use error::Error;

/// How long should we wait between polling by default?
const POLL_INTERVAL_MS: Duration = Duration::from_millis(100);

/// Sometimes we want to yield and let something else happen first,
/// if possible. How long we should do that?
const BRIEF_SLEEP: Duration = Duration::from_millis(10);

fn sleep_briefly() {
    thread::sleep(BRIEF_SLEEP);
}

/// Lower values have 'higher' priority.
type Priority = usize;

lazy_static! {
    static ref PROCESS_ID: String = iter::repeat(())
        .map(|()| thread_rng().sample(Alphanumeric))
        .take(10)
        .collect();
}

pub trait Resource: Clone {
    /// `dir_id` uniquely identifies the directory associated with the resource.
    fn dir_id(&self) -> String;
    /// `name` is the descriptive name of the resource and defaults to wrapping `dir_id`.
    fn name(&self) -> String {
        format!("Resource #{}", self.dir_id())
    }
}

pub struct Scheduler<R: Resource + Send + Sync + 'static> {
    scheduler_root: Arc<Mutex<SchedulerRoot<R>>>,
    resource_schedulers: HashMap<PathBuf, ResourceScheduler<R>>,
    root_dir: PathBuf,
    control_chan: Option<mpsc::Sender<Control>>,
    poll_interval: Duration,
}

pub enum Control {
    Stop,
    Finished(PathBuf, TaskIdent),
}

impl<'a, R: 'a + Resource + Send + Sync> Scheduler<R> {
    pub fn new(root: PathBuf) -> Result<Self, Error> {
        Self::new_with_poll_interval(root, POLL_INTERVAL_MS)
    }

    pub fn new_with_poll_interval(root: PathBuf, poll_interval: Duration) -> Result<Self, Error> {
        let scheduler = SchedulerRoot::new(root)?;
        let root_dir = scheduler.root.clone();
        Ok(Self {
            scheduler_root: Arc::new(Mutex::new(scheduler)),
            resource_schedulers: Default::default(),
            root_dir,
            control_chan: None,
            poll_interval,
        })
    }

    pub fn start(scheduler: &'static Mutex<Self>) -> Result<SchedulerHandle, Error> {
        let (control_tx, control_rx) = mpsc::channel::<Control>();
        let control_tx = Arc::new(Mutex::new(control_tx));
        let control_tx_move = control_tx.clone();
        let handle = thread::spawn(move || {
            let poll_interval = scheduler.lock().unwrap().poll_interval;
            loop {
                for (_, s) in scheduler.lock().unwrap().resource_schedulers.iter_mut() {
                    s.handle_next(control_tx_move.clone())
                        .expect("failed to schedule next task in handle_next");
                }
                match control_rx.try_recv() {
                    Ok(Control::Stop) => {
                        scheduler.lock().unwrap().cleanup();
                        break;
                    }
                    Ok(Control::Finished(dir, task_ident)) => {
                        let _ = scheduler.lock().map(|scheduler| {
                            let _ = scheduler.scheduler_root.lock().map(|mut scheduler_root| {
                                scheduler_root.own_tasks.remove(&task_ident);
                                scheduler_root
                                    .task_files
                                    .remove(&task_ident)
                                    .map(|task_set| {
                                        task_set
                                            .iter()
                                            .find(|t| (**t).path.starts_with(dir.clone()));
                                    });
                            });
                        });
                        let _ = scheduler.lock().map(|mut scheduler| {
                            scheduler
                                .resource_schedulers
                                .get_mut(&dir)
                                .map(|resource_scheduler| {
                                    resource_scheduler.dequeue_task(&task_ident)
                                });
                        });
                    }
                    Err(_) => (),
                }
                thread::sleep(poll_interval);
            }
        });
        Ok(SchedulerHandle {
            handle,
            control_chan: control_tx,
        })
    }

    /// Schedule `task_function` and return a `Receiver` for the result. If priority is 0, scheduler will attempt to
    /// preempt any tasks with higher priority on any of `resources`.
    pub fn schedule<F: 'static, T: 'static>(
        &mut self,
        priority: usize,
        name: &str,
        resources: &Vec<R>,
        task_function: F,
        is_preemptible: bool,
    ) -> Result<mpsc::Receiver<T>, Error>
    where
        F: FnOnce(&R) -> T + Sync + Send,
        T: Sync + Send,
    {
        let (tx, rx) = mpsc::channel();
        let tx_mutex = Mutex::new(tx);
        self.schedule_aux(
            priority,
            name,
            Box::new(move |r, _| {
                let result = task_function(r);
                let _ = (*tx_mutex.lock().unwrap()).send(result);
            }),
            resources,
            is_preemptible,
        )?;
        Ok(rx)
    }

    /// Schedule `task_function` and block until result is available, then return it.
    pub fn schedule_wait<F: 'static, T: 'static>(
        &mut self,
        priority: usize,
        name: &str,
        resources: &Vec<R>,
        task_function: F,
        is_preemptible: bool,
    ) -> Result<T, Error>
    where
        F: FnOnce(&R) -> T + Sync + Send,
        T: Sync + Send,
    {
        let rx = self.schedule(priority, name, resources, task_function, is_preemptible)?;

        Ok(rx.recv()?)
    }

    /// Schedule `task_function` and return a `Future` for the value.
    pub fn schedule_future<F: 'static, T: 'static>(
        &mut self,
        priority: usize,
        name: &str,
        resources: &Vec<R>,
        task_function: F,
        is_preemptible: bool,
    ) -> Result<futures::channel::oneshot::Receiver<T>, Error>
    where
        F: FnOnce(&R) -> T + Sync + Send,
        T: Sync + Send,
    {
        let (tx, rx) = futures::channel::oneshot::channel();
        let tx_mutex = Arc::new(Mutex::new(Some(tx)));
        self.schedule_aux(
            priority,
            name,
            Box::new(move |r, _| {
                let result = task_function(r);

                if let Some(tx) = tx_mutex.lock().unwrap().take() {
                    let _ = tx.send(result);
                }
            }),
            resources,
            is_preemptible,
        )?;
        Ok(rx)
    }

    pub fn stop(&self) -> Result<(), mpsc::SendError<Control>> {
        if let Some(c) = self.control_chan.as_ref() {
            c.send(Control::Stop)?
        };
        Ok(())
    }

    fn schedule_aux(
        &mut self,
        priority: usize,
        name: &str,
        task: Task<R>,
        resources: &Vec<R>,
        is_preemptible: bool,
    ) -> Result<(), Error> {
        let task_ident =
            self.scheduler_root
                .lock()
                .unwrap()
                .new_ident(priority, name, is_preemptible);

        self.scheduler_root
            .lock()
            .unwrap()
            .own_tasks
            .insert(task_ident.clone(), Assignment::Unassigned(task));

        let mut preempt = task_ident.has_priority_to_preempt();
        // If this task has preemption priority, we need to choose *one* resource on which to schedule preemption.
        // This resource must not already have a preempting task assigned.

        for resource in resources.iter() {
            let resource_dir = self.ensure_resource_scheduler(resource.clone())?;
            let resource_scheduler =
                self.resource_schedulers
                    .get_mut(&resource_dir)
                    .expect(&format!(
                        "resource_scheduler for {:?} missing",
                        resource_dir
                    ));

            let status = self
                .scheduler_root
                .lock()
                .unwrap()
                .enqueue_task_for_resource(&task_ident, preempt, resource_scheduler)?;

            match status {
                // Only try to preempt once. Preemption is assumed not to fail, and we do not want to disrupt more than
                // one running task.
                EnqueueStatus::EnqueuedWithPreemption => {
                    preempt = false;
                }
                _ => (),
            }

            match status {
                // Give the just-enqueued task time to be assigned to a resource – so a less-preferred resource is not
                // assigned if a more-preferred resource is available. Note that we are not holding a lock on
                // `self.scheduler_root`, so the scheduler loop will have an opportunity to work, if possible.
                EnqueueStatus::Enqueued | EnqueueStatus::EnqueuedWithPreemption => {
                    thread::sleep(self.poll_interval)
                }
                _ => (),
            }
        }

        Ok(())
    }

    /// `cleanup` performs any (currently none) cleanup required when a scheduler terminates.
    fn cleanup(&self) {}

    fn ensure_resource_scheduler(&mut self, resource: R) -> Result<PathBuf, Error> {
        let dir = self.resource_dir(&resource);

        if !self.resource_schedulers.contains_key(&dir) {
            let rs = ResourceScheduler::new(self.scheduler_root.clone(), dir.clone(), resource)?;
            self.resource_schedulers.insert(dir.clone(), rs);
        }

        Ok(dir)
    }

    fn resource_dir(&self, resource: &R) -> PathBuf {
        self.root_dir.join(resource.dir_id())
    }
}

/// Scheduler will be terminated when `SchedulerHandle` is dropped.
pub struct SchedulerHandle {
    control_chan: Arc<Mutex<mpsc::Sender<Control>>>,
    #[allow(dead_code)]
    handle: thread::JoinHandle<()>,
}

impl Drop for SchedulerHandle {
    fn drop(&mut self) {
        if let Err(e) = self.control_chan.lock().unwrap().send(Control::Stop) {
            log::error!("Error on sending Stop signal to the scheduler loop: {}", e);
        }
    }
}

enum Assignment<R> {
    Unassigned(Task<R>),
    Assigned(R),
}

struct SchedulerRoot<R: Resource + 'static> {
    root: PathBuf,
    /// A given `Task` (identified uniquely by a `TaskIdent`) may have multiple `TaskFile`s associated,
    /// one per `Resource` for which it is currently scheduled (but only one `Resource` will eventually be assigned).
    task_files: HashMap<TaskIdent, HashSet<TaskFile>>,
    own_tasks: HashMap<TaskIdent, Assignment<R>>,
    ident_counter: usize,
}

unsafe impl<'a, R: Resource> Send for SchedulerRoot<R> {}

enum EnqueueStatus {
    Enqueued,
    EnqueuedWithPreemption,
    NotEnqueued,
}

impl<'a, R: Resource + Sync + Send> SchedulerRoot<R> {
    fn new(root: PathBuf) -> Result<Self, Error> {
        create_dir_all(&root)?;
        Ok(Self {
            root,
            task_files: Default::default(),
            own_tasks: Default::default(),
            ident_counter: 0,
        })
    }

    fn new_ident(&mut self, priority: Priority, name: &str, is_preemptible: bool) -> TaskIdent {
        let id = self.ident_counter;
        self.ident_counter += 1;
        TaskIdent::new(priority, name, id, is_preemptible)
    }

    // Returns true if task was enqueued by this call.
    fn enqueue_task_for_resource(
        &mut self,
        task_ident: &TaskIdent,
        preempt: bool,
        resource_scheduler: &mut ResourceScheduler<R>,
    ) -> Result<EnqueueStatus, Error> {
        match self.own_tasks.get(&task_ident) {
            // This isn't our own task. Do nothing and report that.
            None => Ok(EnqueueStatus::NotEnqueued),
            // If the task has been assigned to a resource already, then no further enqueueing is required.
            Some(Assignment::Assigned(_)) => Ok(EnqueueStatus::NotEnqueued),
            Some(Assignment::Unassigned(_)) => {
                let preempting = preempt && resource_scheduler.maybe_preempt_with(task_ident)?;
                // `preempting` is true if we successfully scheduled the task for preemption for this resource. In that
                // case, we should not schedule it for preemption on another. We should, however, still schedule it
                // without preemption on any remaining resources.
                let status = if preempting {
                    EnqueueStatus::EnqueuedWithPreemption
                } else {
                    EnqueueStatus::Enqueued
                };

                let task_file = resource_scheduler.enqueue_task(&task_ident)?;
                self.task_files
                    .entry(task_ident.clone())
                    .or_insert_with(Default::default)
                    .insert(task_file);

                Ok(status)
            }
        }
    }
}

struct ResourceScheduler<R: Resource + 'static> {
    root_scheduler: Arc<Mutex<SchedulerRoot<R>>>,
    dir: PathBuf,
    resource: R,
    preempting: Mutex<Option<(TaskIdent, ResourceLock)>>,
}

impl<'a, R: Resource + Sync + Send + 'static> ResourceScheduler<R> {
    fn new(
        root_scheduler: Arc<Mutex<SchedulerRoot<R>>>,
        dir: PathBuf,
        resource: R,
    ) -> Result<Self, Error> {
        create_dir_all(&dir)?;
        Ok(Self {
            root_scheduler,
            dir,
            resource,
            preempting: Mutex::new(None),
        })
    }

    fn enqueue_task(&self, task_ident: &TaskIdent) -> Result<TaskFile, Error> {
        Ok(task_ident.enqueue_in_dir(&self.dir)?)
    }

    fn dequeue_task(&mut self, task_ident: &TaskIdent) -> Result<(), Error> {
        let mut guard = self.preempting.lock().unwrap();

        match &*guard {
            // NOTE: `_preemption_lock` will be moved out of `self`, dropped, and therefore released if currently held.
            Some((_task_ident, _preemption_lock)) => *guard = None,
            None => (),
        };

        task_ident.destroy(&self.dir)?;

        Ok(())
    }

    /// Try to acquire the preemption lock and record the results.
    /// Return true if the preemption lock was successfully acquired.
    fn maybe_preempt_with(&mut self, task_ident: &TaskIdent) -> Result<bool, Error> {
        if !self.current_task_is_preemptible()? {
            return Ok(false);
        }

        let mut guard = self.preempting.lock().unwrap();

        if guard.is_none() {
            match self.try_preempt_lock()? {
                Some(lock) => {
                    *guard = Some((task_ident.clone(), lock));
                    Ok(true)
                }
                None => Ok(false), // Another process holds the preemption lock.
            }
        } else {
            Ok(false) // Another of this process' own tasks is preempting.
        }
    }

    fn lock(&self) -> Result<ResourceLock, Error> {
        Ok(ResourceLock::acquire(&self.dir, &self.resource, false)?)
    }

    fn try_lock(&self) -> Result<Option<ResourceLock>, Error> {
        Ok(ResourceLock::maybe_acquire(
            &self.dir,
            &self.resource,
            false,
        )?)
    }

    #[allow(dead_code)]
    fn preempt_lock(&self) -> Result<ResourceLock, Error> {
        Ok(ResourceLock::acquire(&self.dir, &self.resource, true)?)
    }

    fn try_preempt_lock(&self) -> Result<Option<ResourceLock>, Error> {
        Ok(ResourceLock::maybe_acquire(
            &self.dir,
            &self.resource,
            true,
        )?)
    }

    // Returns next-up `TaskIdent` and true if locked.
    fn next_task_ident(&self) -> Result<Option<(TaskIdent, bool)>, Error> {
        Ok(Self::sorted_task_idents_and_locked_status(&self.dir)?
            .get(0)
            .map(|x| x.clone()))
    }

    fn current_task_is_preemptible(&self) -> Result<bool, Error> {
        Ok(Self::sorted_task_idents_and_locked_status(&self.dir)?
            .iter()
            .filter(|(_, locked)| *locked)
            .take(1)
            .map(|x| x.1)
            .last()
            .unwrap_or(false))
    }

    fn sorted_task_idents_and_locked_status(
        dir: &PathBuf,
    ) -> Result<Vec<(TaskIdent, bool)>, Error> {
        let mut ident_data = Vec::new();
        let _ = fs::read_dir(&dir)?
            .map(|res| -> Result<(), Error> {
                res.map(|e| -> Result<(), Error> {
                    let metadata = e.metadata()?;
                    let task_ident = TaskIdent::from_str(
                        &e.file_name()
                            .to_str()
                            .expect("failed to create TaskIdent from string"),
                    )
                    .ok();
                    let file = File::open(e.path())?;
                    let locked = file.try_lock_exclusive().is_err();
                    if let Some(ident) = task_ident {
                        ident_data.push((
                            ident,
                            metadata.created().expect("failed to create metadata"),
                            locked,
                        ))
                    };
                    Ok(())
                })?
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

        Ok(ident_data
            .iter()
            .map(|(ident, _create_date, locked)| (ident.clone(), *locked))
            .collect())
    }

    fn handle_next(
        &mut self,
        control_chan: Arc<Mutex<mpsc::Sender<Control>>>,
    ) -> Result<(), Error> {
        assert!(self.dir.is_dir(), "scheduler dir is not a directory.");

        // If resource is locked, there is nothing to do now.
        if self.try_lock()?.is_none() {
            return Ok(());
        }

        let (ident, locked) = match self.next_task_ident()? {
            Some(res) => res,
            None => return Ok(()),
        };

        let is_own = self
            .root_scheduler
            .lock()
            .unwrap()
            .own_tasks
            .get(&ident)
            .is_some();

        if is_own {
            // Task is owned by this process.
            self.handle_own(
                &ident,
                &mut self.root_scheduler.lock().unwrap(),
                control_chan,
            );
        } else {
            if !locked {
                // The next-up task is unlocked, so it can be destroyed —
                // unless it has *just* been created and not yet locked.
                // In that case, sleep briefly to be safe.
                sleep_briefly();
                ident.try_destroy(&self.dir)?;
            }
        }
        Ok(())
    }

    fn handle_own(
        &self,
        ident: &TaskIdent,
        // `root_scheduler` is `self.root_scheduler` — with the lock held.
        root_scheduler: &mut SchedulerRoot<R>,
        control_chan: Arc<Mutex<mpsc::Sender<Control>>>,
    ) {
        let assignment = root_scheduler
            .own_tasks
            .get_mut(&ident)
            .expect("own task missing");

        match std::mem::replace(assignment, Assignment::Assigned(self.resource.clone())) {
            Assignment::Unassigned(task) => {
                if let Some(all_task_files) = root_scheduler.task_files.get(&ident) {
                    all_task_files.iter().for_each(|task_file| {
                        // FIXME: If this task has priority zero, it may be scheduled for preemption on
                        // other resources. We need to ensure only one resource is preempted.

                        // Don't destroy this directory's task file until we are done performing the task
                        if !task_file.path.starts_with(self.dir.clone()) {
                            task_file.destroy().unwrap();
                        };
                    });
                }
                self.perform_task(task, control_chan, ident, self.dir.clone());
            }
            // If `task` is `None`, that means we own it, and it has already been assigned, so there is nothing to do now.
            Assignment::Assigned(_) => (),
        }
    }

    fn perform_task(
        &self,
        task: Task<R>,
        control_chan: Arc<Mutex<mpsc::Sender<Control>>>,
        ident: &TaskIdent,
        dir: PathBuf,
    ) {
        let resource_lock = self.lock().unwrap();
        let task_is_preempting = match &*self.preempting.lock().unwrap() {
            Some((task_ident, _preemption_lock)) => task_ident == ident,
            None => false,
        };

        let preemption_checker = PreemptionChecker {
            dir: self.dir.clone(),
            task_is_preempting,
        };

        let resource = self.resource.clone();
        let ident = ident.clone();
        thread::spawn(move || {
            {
                let _captured_lock = resource_lock;
                task(&resource, &preemption_checker);
                // Lock is dropped, and therefore released here, at end of scope after task has been performed.
            }
            if let Err(e) = control_chan
                .lock()
                .unwrap()
                .send(Control::Finished(dir, ident))
            {
                log::error!(
                    "Error on sending Finished signal to the scheduler loop: {}",
                    e
                );
            }
        });
    }
}

struct PreemptionChecker {
    dir: PathBuf,
    task_is_preempting: bool,
}

impl<R: Resource> Preemption<R> for PreemptionChecker {
    fn should_preempt_now(&self) -> bool {
        if self.task_is_preempting {
            // A task should never preempt itself.
            false
        } else {
            ResourceLock::is_held(&self.dir, true)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Clone, Debug)]
    struct Rsrc {
        id: usize,
    }

    impl Resource for Rsrc {
        fn dir_id(&self) -> String {
            self.id.to_string()
        }
    }

    const TEST_POLL_INTERVAL: Duration = Duration::from_millis(5);
    macro_rules! new_scheduler {
        ($name:ident) => {
            lazy_static! {
                static ref $name: Mutex<Scheduler::<Rsrc>> = Mutex::new(
                    Scheduler::<Rsrc>::new_with_poll_interval(
                        tempfile::tempdir().unwrap().into_path(),
                        TEST_POLL_INTERVAL,
                    )
                    .expect("Failed to create scheduler"),
                );
            }
        };
    }

    new_scheduler!(SCHEDULER1);
    #[test]
    fn test_scheduler() {
        let scheduler = &*SCHEDULER1;
        let _handle = Scheduler::start(scheduler).unwrap();

        let result_state = Arc::new(Mutex::new(Vec::<usize>::new()));
        let num_resources = 3;
        let resources = (0..num_resources).map(|id| Rsrc { id }).collect::<Vec<_>>();

        let mut expected = Vec::new();

        let mut chans = Vec::new();
        for i in 0..num_resources {
            let result_state = Arc::clone(&result_state);
            // Schedule a slow task to tie up all the resources
            // while the next batch of `Task1`s is scheduled.
            let priority = 0;
            expected.push(0);
            chans.push(
                scheduler
                    .lock()
                    .unwrap()
                    .schedule(
                        priority,
                        &format!("Task0[{}]", i),
                        &resources,
                        move |_| {
                            result_state.lock().unwrap().push(0);
                            thread::sleep(Duration::from_millis(100));
                        },
                        false,
                    )
                    .unwrap(),
            );
        }
        chans.into_iter().for_each(|ch| ch.recv().unwrap());

        let tasks1_len = 5;
        let mut chans = Vec::new();
        for id in 0..tasks1_len {
            let result_state = Arc::clone(&result_state);
            // When tasks are added very quickly (relative to the poll interval),
            // they should be performed in order of their priority.
            // In this group, we set priority to be the 'inverse' of task id.
            // So task 0 has a high-numbered priority and should be performed last.
            // Therefore, we push the highest id onto `expected` first.
            let priority = tasks1_len - id - 1;
            expected.push(priority);
            chans.push(
                scheduler
                    .lock()
                    .unwrap()
                    .schedule(
                        priority,
                        &format!("Task1[{}]", id),
                        &resources,
                        move |_| {
                            result_state.lock().unwrap().push(id);
                        },
                        false,
                    )
                    .unwrap(),
            );
        }
        chans.into_iter().for_each(|ch| ch.recv().unwrap());

        let mut chans = Vec::new();
        for id in 0..tasks1_len {
            let result_state = Arc::clone(&result_state);
            // This example is like the previous, except that we sleep for twice the length of the poll interval
            // between each call to `schedule`. TODO: set the poll interval explicitly in the test.
            // Because each task is fully processed, they are performed in the order scheduled.
            let priority = tasks1_len - id - 1;
            expected.push(id);
            thread::sleep(Duration::from_millis(200));
            chans.push(
                scheduler
                    .lock()
                    .unwrap()
                    .schedule(
                        priority,
                        &format!("Task1[{}]", id),
                        &resources,
                        move |_| {
                            result_state.lock().unwrap().push(id);
                        },
                        false,
                    )
                    .unwrap(),
            );
        }
        chans.into_iter().for_each(|ch| ch.recv().unwrap());

        let mut chans = Vec::new();
        for id in 0..tasks1_len {
            let result_state = Arc::clone(&result_state);
            // In this example, tasks are added quickly and with priority matching id.
            // We therefore expect them to be performed in the order scheduled.
            // This case is somewhat trivial.
            expected.push(id);
            chans.push(
                scheduler
                    .lock()
                    .unwrap()
                    .schedule(
                        id,
                        &format!("Task1[{}]", id),
                        &resources,
                        move |_| {
                            result_state.lock().unwrap().push(id);
                        },
                        false,
                    )
                    .unwrap(),
            );
        }
        chans.into_iter().for_each(|ch| ch.recv().unwrap());

        let tasks2_len = 5;
        let mut out_rxs = Vec::with_capacity(tasks2_len);
        for i in 0..tasks2_len {
            // This example does not exercise the scheduler as such,
            // since results are harvested from the output channels
            // in the expected order (so scheduling does not come into play).
            // However, it does demonstrate and ensure use and usability of channels
            // to provide input to and receive output from tasks.
            let (tx, rx) = mpsc::channel();
            out_rxs.push(rx);
            let tx = Mutex::new(tx);

            let priority = tasks2_len - i - 1;
            expected.push(i * i);
            scheduler
                .lock()
                .unwrap()
                .schedule(
                    priority,
                    &format!("Task2[{}]", i),
                    &resources,
                    move |_| {
                        let input = i;
                        let result = input * input;
                        tx.lock().unwrap().send(result).unwrap();
                    },
                    false,
                )
                .unwrap();
        }

        for rx in out_rxs.iter() {
            let result = rx.recv().unwrap();
            result_state.lock().unwrap().push(result);
        }

        scheduler.lock().unwrap().stop().unwrap();

        let tasks1_len = 5;
        assert_eq!(
            num_resources + tasks1_len * 4,
            result_state.lock().unwrap().len()
        );

        assert_eq!(expected, result_state.lock().unwrap().clone());
    }

    new_scheduler!(SCHEDULER2);
    #[test]
    fn test_guards() {
        let scheduler = &*SCHEDULER2;
        let _handle = Scheduler::start(scheduler).unwrap();

        let guard_failure = Arc::new(Mutex::new(false));

        let resources = (0..3).map(|id| Rsrc { id }).collect::<Vec<_>>();
        let resource_locks = Arc::new({
            let mut map = HashMap::new();
            for rsrc in resources.iter() {
                map.insert(rsrc.name(), Mutex::new(()));
            }
            map
        });

        let mut chans = Vec::new();
        for id in 0..10 {
            let resource_locks = Arc::clone(&resource_locks);
            let guard_failure = Arc::clone(&guard_failure);
            chans.push(
                scheduler
                    .lock()
                    .unwrap()
                    .schedule(
                        0,
                        &format!("MyTask[{}]", id),
                        &resources,
                        move |r| {
                            let mutex = &resource_locks[&r.name()];
                            // No more than one task should be able to run on a single resource at a time!
                            let _lock = match mutex.try_lock() {
                                Ok(lock) => lock,
                                Err(_) => {
                                    *guard_failure.lock().unwrap() = true;
                                    return;
                                }
                            };
                            thread::sleep(Duration::from_millis(100));
                        },
                        false,
                    )
                    .unwrap(),
            );
        }
        chans.into_iter().for_each(|ch| ch.recv().unwrap());

        assert!(!*guard_failure.lock().unwrap());
    }

    new_scheduler!(SCHEDULER3);
    #[test]
    fn test_schedule_fn() {
        let scheduler = &*SCHEDULER3;
        let _handle = Scheduler::start(scheduler).unwrap();

        thread::sleep(Duration::from_millis(300));

        let n = 10;
        let resources = (0..n).map(|id| Rsrc { id }).collect::<Vec<_>>();
        let mut futures = (0..n)
            .map(|i| {
                scheduler
                    .lock()
                    .unwrap()
                    .schedule_future(n - i, &format!("task {}", i), &resources, move |_| i, false)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        // This only tests that every function is run and returns its expected value.
        // The order of esults is determined by the order in which the task functions are scheduled,
        // not on that in which they are performed. Actual scheduling is not exercised here.
        let mut results = Vec::new();

        thread::sleep(Duration::from_millis(100));

        for future in futures.iter_mut() {
            results.push(tokio_test::block_on(future).unwrap());
        }

        let expected = (0..n).collect::<Vec<_>>();

        assert_eq!(expected, results);
    }
}

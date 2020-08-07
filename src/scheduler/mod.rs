extern crate rand;

// TODO: respect resource order preference for scheduled tasks.

use ::lazy_static::lazy_static;

use self::task_ident::TaskIdent;
use fs2::FileExt;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, create_dir_all, File};
use std::io::Error;
use std::iter;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use self::{
    resource_lock::ResourceLock,
    task::{Executable, Preemption, Task},
    task_file::TaskFile,
};

mod resource_lock;
mod task;
mod task_file;
mod task_ident;

/// How often, in milliseconds, should we poll by default?
const POLL_INTERVAL_MS: u64 = 100;

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

pub struct Scheduler<R: Resource + 'static> {
    scheduler_root: Arc<Mutex<SchedulerRoot<R>>>,
    resource_schedulers: HashMap<PathBuf, ResourceScheduler<R>>,
    control_chan: Option<mpsc::Sender<()>>,
    poll_interval: u64,
}

impl<'a, R: 'a + Resource + Copy + Send + Sync> Scheduler<R> {
    pub fn new(root: PathBuf) -> Result<Self, Error> {
        Self::new_with_poll_interval(root, POLL_INTERVAL_MS)
    }

    pub fn new_with_poll_interval(root: PathBuf, poll_interval: u64) -> Result<Self, Error> {
        let scheduler = SchedulerRoot::new(root)?;
        Ok(Self {
            scheduler_root: Arc::new(Mutex::new(scheduler)),
            resource_schedulers: Default::default(),
            control_chan: None,
            poll_interval,
        })
    }

    pub fn start(scheduler: &'static Mutex<Self>) -> Result<(), Error> {
        let (control_tx, control_rx) = mpsc::channel();
        thread::spawn(move || {
            let should_stop = || !control_rx.try_recv().is_err();
            let poll_interval = scheduler.lock().unwrap().poll_interval;
            loop {
                if should_stop() {
                    break;
                };
                for (_, s) in scheduler.lock().unwrap().resource_schedulers.iter_mut() {
                    if should_stop() {
                        break;
                    };

                    s.handle_next().expect("failed in handle_next"); // FIXME
                }
                thread::sleep(Duration::from_millis(poll_interval));
            }
        });
        scheduler.lock().unwrap().control_chan = Some(control_tx);
        Ok(())
    }

    pub fn schedule(
        &mut self,
        priority: usize,
        name: &str,
        task: &'static (dyn Executable<R> + Sync),
        resources: &[R],
    ) -> Result<(), Error> {
        resources.iter().for_each(|r| {
            self.ensure_resource_scheduler(*r);
        });
        let task_ident = self
            .scheduler_root
            .lock()
            .unwrap()
            .new_ident(priority, name);
        let task = Task::new(Box::new(task));
        self.scheduler_root.lock().unwrap().schedule(
            task_ident,
            task,
            resources,
            &self.resource_schedulers,
        )
    }

    pub fn stop(&self) -> Result<(), mpsc::SendError<()>> {
        if let Some(c) = self.control_chan.as_ref() {
            c.send(())?
        };
        Ok(())
    }

    fn ensure_resource_scheduler(&mut self, resource: R) {
        let dir = self
            .scheduler_root
            .lock()
            .unwrap()
            .root
            .join(resource.dir_id());
        if !self.resource_schedulers.contains_key(&dir) {
            let rs = ResourceScheduler::new(self.scheduler_root.clone(), dir.clone(), resource);
            self.resource_schedulers.insert(dir, rs);
        }
    }
}

struct SchedulerRoot<R: Resource + 'static> {
    root: PathBuf,
    /// A given `Task` (identified uniquely by a `TaskIdent`) may have multiple `TaskFile`s associated,
    /// one per `Resource` for which it is currently scheduled (but only one `Resource` will eventually be assigned).
    task_files: HashMap<TaskIdent, HashSet<TaskFile>>,
    /// Each `Task` (identified uniquely by a `TaskIdent`) is protected by a `Mutex`.
    own_tasks: HashMap<TaskIdent, Mutex<Task<R>>>,
    ident_counter: usize,
}

unsafe impl<'a, R: Resource> Send for SchedulerRoot<R> {}

impl<'a, R: Resource + Sync> SchedulerRoot<R> {
    fn new(root: PathBuf) -> Result<Self, Error> {
        create_dir_all(&root)?;
        Ok(Self {
            root,
            task_files: Default::default(),
            own_tasks: Default::default(),
            ident_counter: 0,
        })
    }
    fn new_ident(&mut self, priority: Priority, name: &str) -> TaskIdent {
        let id = self.ident_counter;
        self.ident_counter += 1;
        TaskIdent::new(priority, name, id)
    }
    fn schedule(
        &mut self,
        task_ident: TaskIdent,
        task: Task<R>,
        resources: &[R],
        resource_schedulers: &HashMap<PathBuf, ResourceScheduler<R>>,
    ) -> Result<(), Error> {
        for resource in resources.iter() {
            let dir = self.root.join(resource.dir_id());
            create_dir_all(&dir)?;
            let task_file = task_ident.enqueue_in_dir(&dir)?;

            self.task_files
                .entry(task_ident.clone())
                .or_insert(Default::default())
                .insert(task_file);
            self.own_tasks
                .insert(task_ident.clone(), Mutex::new(task.clone()));

            // FIXME: Refactor to break handle_next up further, so we can use some of its
            // decomposed parts here. In particular, after performing the task,
            // we need to remove it and any siblings already added (on earlier iterations of this loop).
            // if let Some((next, locked)) = ResourceScheduler::<R>::next_task_ident(&dir) {
            //     if next == task_ident {
            //         if let Some(resource_scheduler) = resource_schedulers.get(&dir) {
            //             resource_scheduler.perform_task(&task);
            //         }
            //     }
            // }
        }
        Ok(())
    }
}

struct ResourceScheduler<R: Resource + 'static> {
    root_scheduler: Arc<Mutex<SchedulerRoot<R>>>,
    dir: PathBuf,
    resource: R,
    /// The previous 'next', and a count of how many times we have seen it as such.
    previous: Option<(TaskIdent, usize)>,
}

impl<'a, R: Resource + Sync> ResourceScheduler<R> {
    fn new(root_scheduler: Arc<Mutex<SchedulerRoot<R>>>, dir: PathBuf, resource: R) -> Self {
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

    fn next_task_ident(dir: &PathBuf) -> Option<(TaskIdent, bool)> {
        let mut ident_data = Vec::new();
        let _ = fs::read_dir(&dir)
            .unwrap()
            .map(|res| {
                res.map(|e| {
                    // FIXME: unwraps
                    let metadata = e.metadata().unwrap();
                    let task_ident = TaskIdent::from_str(
                        &e.file_name()
                            .to_str()
                            .expect("failed to create TaskIdent from string"),
                    )
                    .ok();
                    let file = File::open(e.path()).unwrap();
                    let locked = file.try_lock_exclusive().is_err();
                    if let Some(ident) = task_ident {
                        ident_data.push((
                            ident,
                            metadata.created().expect("failed to create metadata"),
                            locked,
                        ))
                    };
                })
            })
            .collect::<Result<Vec<_>, Error>>()
            .unwrap();
        ident_data.sort_by(|(a_ident, a_create_date, _), (b_ident, b_create_date, _)| {
            // Sort first by (priority, creation date).
            let priority_ordering = a_ident.priority.partial_cmp(&b_ident.priority).unwrap();
            match priority_ordering {
                Ordering::Equal => a_create_date.partial_cmp(&b_create_date).unwrap(),
                _ => priority_ordering,
            }
        });
        if let Some((ident, _, locked)) = ident_data.get(0) {
            return Some((ident.clone(), *locked));
        } else {
            return None;
        };
    }

    fn handle_next(&mut self) -> Result<(), Error> {
        assert!(self.dir.is_dir(), "scheduler dir is not a directory.");

        // FIXME: Clean this up.
        let (ident, locked) = if let Some((ident, locked)) = Self::next_task_ident(&self.dir) {
            (ident, locked)
        } else {
            // If there was no `TaskIdent` found, nothing to do.
            // Forget about anything we saw before.
            self.previous = None;
            return Ok(());
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

            let mut performed_task = false;
            {
                let root_scheduler = self.root_scheduler.lock().unwrap();
                // Lock the task so a sibling won't remove it.
                let mut guard_result = root_scheduler
                    .own_tasks
                    .get(&ident)
                    .expect("own task missing")
                    .try_lock();

                if let Ok(ref mut guard) = guard_result {
                    let task = &*guard;
                    self.previous = None;

                    let mut to_destroy_later = None;

                    // We have the lock for this task, so we may destroy the sibling TaskFiles.
                    if let Some(all_task_files) = root_scheduler.task_files.get(&ident) {
                        // FIXME: unwrap
                        all_task_files.iter().for_each(|task_file| {
                            // Don't destroy this directory's task file until we are done performing the task
                            if !task_file.path.starts_with(self.dir.clone()) {
                                // We already hold the lock for all of our task files, so this is okay.
                                task_file.destroy().unwrap();
                            // TODO: check that destroy fails gracefully if already gone.
                            } else {
                                to_destroy_later = Some(task_file);
                            }
                        });
                    }

                    self.perform_task(&**task)?;
                    // NOTE: We must defer removing from `self.own_tasks` because the map is borrowed in this scope above.
                    performed_task = true;

                    // Finally, destroy this `TaskFile`, too — assuming it is necessary.
                    if let Some(task_file) = to_destroy_later {
                        // We already hold the lock for this task file, so this is okay.
                        task_file.destroy().unwrap()
                    };
                } else {
                    // Task `Mutex` was already locked, which means this process has already assigned it to a different resource.
                    // Do nothing and allow it to be cleaned up (removed from this queue) as part of that assignment.
                }

                // lock is dropped here
            }

            if performed_task {
                // Now we can remove (see NOTE above).
                self.root_scheduler.lock().unwrap().own_tasks.remove(&ident);
            }
        } else {
            // Task is owned by another process.
            if locked {
                self.previous = None;
            } else {
                self.previous = match &self.previous {
                    // The same unlocked task has been 'next up' for turns threevmx, so it has forfeited its turn.
                    // Since we discovered this, it is our job to destroy it.
                    // We need to see it three times, since different processes will be on different schedules.
                    // Worst-case behavior of out-of-sync schedules gives no time for the actual winner to act.
                    Some((previous, n)) if previous == &ident && *n >= 2 => {
                        // If this fails, someone else may have seized the lock and done it for us.
                        previous.try_destroy(&self.dir)?;
                        None
                    }

                    // Increment the count, so we can destroy this if we see it on top next time we check.
                    Some((previous, n)) if previous == &ident => Some((previous.clone(), n + 1)),

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
        // TOOD: Pass `self` so `Executable` can call `should_preempt_now` on it if needed.

        task.execute(
            &self.resource,
            &Dummy {
                _r: PhantomData::<R>,
            },
        );

        Ok(())
        // Lock is dropped, and therefore released here, at end of scope.
    }
}

struct Dummy<R> {
    _r: PhantomData<R>,
}
impl<R: Resource> Preemption<R> for Dummy<R> {
    fn should_preempt_now(&self, _task: &Task<R>) -> bool {
        false
    }
}

mod test {
    use super::*;

    /// `Scheduler` requires that resources be `Copy`.
    #[derive(Copy, Clone, Debug)]
    struct Rsrc {
        id: usize,
    }

    impl Resource for Rsrc {
        fn dir_id(&self) -> String {
            self.id.to_string()
        }
    }

    // struct Dummy<R> {
    //     _r: PhantomData<R>,
    // }
    // impl<R: Resource> Preemption<R> for Dummy<R> {
    //     fn should_preempt_now(&self, _task: &Task<R>) -> bool {
    //         false
    //     }
    // }

    #[derive(Debug)]
    struct Task1 {
        id: usize,
    }

    impl<R: Resource> Executable<R> for Task1 {
        fn execute(&self, resource: &R, _p: &dyn Preemption<R>) {
            (*RESULT_STATE).lock().unwrap().push(self.id);
        }
    }

    #[derive(Debug)]
    struct Task2 {
        id: usize,
        in_chan: Mutex<mpsc::Sender<usize>>,
        in_chan_internal: Mutex<mpsc::Receiver<usize>>,
        out_chan: Mutex<mpsc::Receiver<usize>>,
        out_chan_internal: Mutex<mpsc::Sender<usize>>,
    }

    impl Task2 {
        fn new(id: usize) -> Self {
            let (in_tx, in_rx) = mpsc::channel();
            let (out_tx, out_rx) = mpsc::channel();
            Self {
                id,
                in_chan: Mutex::new(in_tx),
                in_chan_internal: Mutex::new(in_rx),
                out_chan: Mutex::new(out_rx),
                out_chan_internal: Mutex::new(out_tx),
            }
        }
        fn set_input(&self, x: usize) {
            self.in_chan.lock().unwrap().send(x);
        }
    }

    impl<R: Resource> Executable<R> for Task2 {
        fn execute(&self, resource: &R, _p: &dyn Preemption<R>) {
            let input = self.in_chan_internal.lock().unwrap().recv().unwrap_or(999);
            let result = input * input;
            self.out_chan_internal.lock().unwrap().send(result);
        }
    }

    lazy_static! {
        static ref RESULT_STATE: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        static ref SCHEDULER: Mutex<Scheduler::<Rsrc>> = Mutex::new(
            Scheduler::<Rsrc>::new(tempfile::tempdir().unwrap().into_path())
                .expect("Failed to create scheduler")
        );
        static ref TASKS1: Vec<Task1> = (0..5).map(|id| Task1 { id }).collect::<Vec<_>>();
        static ref TASKS2: Vec<Task2> = (0..5).map(|id| Task2::new(id)).collect::<Vec<_>>();
    }

    #[test]
    fn test_scheduler() {
        let scheduler = &*SCHEDULER;
        let root_dir = scheduler
            .lock()
            .unwrap()
            .scheduler_root
            .lock()
            .unwrap()
            .root
            .clone();

        let resources = (0..3).map(|id| Rsrc { id }).collect::<Vec<_>>();

        let mut expected = Vec::new();

        let control_chan = Scheduler::start(scheduler).expect("Failed to start scheduler.");
        for (i, task) in TASKS1.iter().enumerate() {
            // When tasks are added very quickly (relative to the poll interval),
            // they should be performed in order of their priority.
            // In this group, we set priority to be the 'inverse' of task id.
            // So task 0 has a high-numbered priority and should be performed last.
            // Therefore, we push the highest id onto `expected` first.
            let priority = TASKS1.len() - i - 1;
            expected.push(priority);
            scheduler
                .lock()
                .unwrap()
                .schedule(priority, &format!("{:?}", task), task, &resources);
        }
        thread::sleep(Duration::from_millis(1000));
        for (i, task) in TASKS1.iter().enumerate() {
            // This example is like the previous, except that we sleep for twice the length of the poll interval
            // between each call to `schedule`. TODO: set the poll interval explicitly in the test.
            // Because each task is fully processed, they are performed in the order scheduled.
            let priority = TASKS1.len() - i - 1;
            expected.push(i);
            thread::sleep(Duration::from_millis(200));
            scheduler
                .lock()
                .unwrap()
                .schedule(priority, &format!("{:?}", task), task, &resources);
        }
        thread::sleep(Duration::from_millis(1000));
        for (i, task) in TASKS1.iter().enumerate() {
            // In this example, tasks are added quickly and with priority matching id.
            // We therefore expect them to be performed in the order scheduled.
            // This case is somewhat trivial.
            expected.push(i);
            scheduler
                .lock()
                .unwrap()
                .schedule(i, &format!("{:?}", task), task, &resources);
        }
        thread::sleep(Duration::from_millis(1000));

        for (i, task) in TASKS2.iter().enumerate() {
            // This example does not exercise the scheduler as such,
            // since results are harvested from the output channels
            // in the expected order (so scheduling does not come into play).
            // However, it does demonstrate and ensure use and usability of channels
            // to provide input to and receive output from tasks.
            let priority = TASKS1.len() - i - 1;
            task.set_input(i);
            expected.push(i * i);
            scheduler
                .lock()
                .unwrap()
                .schedule(priority, &format!("{:?}", task), task, &resources);
        }

        for (i, task) in TASKS2.iter().enumerate() {
            let result = task.out_chan.lock().unwrap().recv().unwrap();
            (*RESULT_STATE).lock().unwrap().push(result);
        }

        thread::sleep(Duration::from_millis(1000));

        scheduler.lock().unwrap().stop();

        assert_eq!(TASKS1.len() * 4, expected.len());

        assert_eq!(expected, *RESULT_STATE.lock().unwrap());
    }
}

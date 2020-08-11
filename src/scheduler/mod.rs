extern crate rand;

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

pub trait Resource {
    /// `dir_id` uniquely identifies the directory associated with the resource.
    fn dir_id(&self) -> String;
    /// `name` is the descriptive name of the resource and defaults to wrapping `dir_id`.
    fn name(&self) -> String {
        format!("Resource #{}", self.dir_id())
    }
}

pub struct Scheduler<R: Resource + Copy + Send + Sync + 'static> {
    scheduler_root: Arc<Mutex<SchedulerRoot<R>>>,
    resource_schedulers: HashMap<PathBuf, ResourceScheduler<R>>,
    control_chan: Option<mpsc::Sender<()>>,
    poll_interval: Duration,
}

impl<'a, R: 'a + Resource + Copy + Send + Sync> Scheduler<R> {
    pub fn new(root: PathBuf) -> Result<Self, Error> {
        Self::new_with_poll_interval(root, POLL_INTERVAL_MS)
    }

    pub fn new_with_poll_interval(root: PathBuf, poll_interval: Duration) -> Result<Self, Error> {
        let scheduler = SchedulerRoot::new(root)?;
        Ok(Self {
            scheduler_root: Arc::new(Mutex::new(scheduler)),
            resource_schedulers: Default::default(),
            control_chan: None,
            poll_interval,
        })
    }

    pub fn start(scheduler: &'static Mutex<Self>) -> Result<SchedulerHandle, Error> {
        let (control_tx, control_rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let poll_interval = scheduler.lock().unwrap().poll_interval;
            let should_stop = || !control_rx.try_recv().is_err();
            loop {
                if should_stop() {
                    scheduler.lock().unwrap().cleanup();
                    break;
                };
                for (_, s) in scheduler.lock().unwrap().resource_schedulers.iter_mut() {
                    if should_stop() {
                        break;
                    };

                    s.handle_next().expect("failed in handle_next"); // FIXME
                }
                thread::sleep(poll_interval);
            }
        });
        Ok(SchedulerHandle {
            handle,
            control_chan: control_tx,
        })
    }

    /// `cleanup` performs any (currently none) cleanup required when a scheduler terminates.
    fn cleanup(&self) {}

    pub fn schedule(
        &mut self,
        priority: usize,
        name: &str,
        task: Box<dyn Executable<R> + Sync + Send>,
        resources: &[Arc<R>],
    ) -> Result<(), Error> {
        resources.iter().for_each(|r| {
            self.ensure_resource_scheduler(r.clone());
        });
        let task_ident = self
            .scheduler_root
            .lock()
            .unwrap()
            .new_ident(priority, name);
        let task = Task::new(Arc::new(task));
        self.scheduler_root.lock().unwrap().schedule(
            task_ident,
            task,
            resources,
            &self.resource_schedulers,
            self.poll_interval,
        )
    }

    pub fn stop(&self) -> Result<(), mpsc::SendError<()>> {
        if let Some(c) = self.control_chan.as_ref() {
            c.send(())?
        };
        Ok(())
    }

    fn ensure_resource_scheduler(&mut self, resource: Arc<R>) {
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

/// Scheduler will be terminated when `SchedulerHandle` is dropped.
pub struct SchedulerHandle {
    control_chan: mpsc::Sender<()>,
    handle: thread::JoinHandle<()>,
}

impl Drop for SchedulerHandle {
    fn drop(&mut self) {
        self.control_chan.send(());
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
    fn new_ident(&mut self, priority: Priority, name: &str) -> TaskIdent {
        let id = self.ident_counter;
        self.ident_counter += 1;
        TaskIdent::new(priority, name, id)
    }
    fn schedule(
        &mut self,
        task_ident: TaskIdent,
        task: Task<R>,
        resources: &[Arc<R>],
        resource_schedulers: &HashMap<PathBuf, ResourceScheduler<R>>,
        poll_interval: Duration,
    ) -> Result<(), Error> {
        self.own_tasks.insert(task_ident.clone(), Mutex::new(task));

        // Create all resource dirs, if necessary (even if task is performed before enqueuing on all of them).
        for resource in resources.iter() {
            let dir = self.root.join(resource.dir_id());
            create_dir_all(&dir)?;
        }
        for resource in resources.iter() {
            let dir = self.root.join(resource.dir_id());
            let task_file = task_ident.enqueue_in_dir(&dir)?;

            self.task_files
                .entry(task_ident.clone())
                .or_insert(Default::default())
                .insert(task_file);

            // In order to respect resource preference order, give each resource (in order)
            // one chance to be performed if resource is free.
            // This depends on `schedule` not being called from the polling loop's thread.
            thread::sleep(poll_interval);
        }
        Ok(())
    }
}

struct ResourceScheduler<R: Resource + 'static> {
    root_scheduler: Arc<Mutex<SchedulerRoot<R>>>,
    dir: PathBuf,
    resource: Arc<R>,
}

impl<'a, R: Resource + Sync + Send> ResourceScheduler<R> {
    fn new(root_scheduler: Arc<Mutex<SchedulerRoot<R>>>, dir: PathBuf, resource: Arc<R>) -> Self {
        Self {
            root_scheduler,
            dir,
            resource,
        }
    }

    fn lock(&self) -> Result<ResourceLock, Error> {
        ResourceLock::acquire(&self.dir, &*self.resource)
    }

    fn try_lock(&self) -> Result<Option<ResourceLock>, Error> {
        ResourceLock::maybe_acquire(&self.dir, &*self.resource)
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

        // If resource is locked, there is nothing to do now.
        if self.try_lock()?.is_none() {
            return Ok(());
        }

        let (ident, locked) = match Self::next_task_ident(&self.dir) {
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
            self.handle_own(&ident, &mut self.root_scheduler.lock().unwrap(), false);
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
        root_scheduler: &mut SchedulerRoot<R>,
        already_performed: bool,
    ) -> Result<(), Error> {
        let mut performed_task = already_performed;
        {
            // Lock the task so a sibling won't remove it.
            let mut guard_result = root_scheduler
                .own_tasks
                .get(&ident)
                .expect("own task missing")
                .try_lock();

            if let Ok(guard) = guard_result {
                let task = guard;

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

                self.perform_task(&*task)?;
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
            root_scheduler.own_tasks.remove(&ident);
        }

        Ok(())
    }

    fn perform_task(&self, task: &Task<R>) -> Result<(), Error> {
        let _lock = self.lock()?;

        let preemption_checker = PreemptionChecker {
            dir: self.dir.clone(),
        };

        let resource = self.resource.clone();
        let executable = task.executable.clone();
        thread::spawn(move || {
            executable.execute(&resource, &preemption_checker);
        });
        Ok(())
        // Lock is dropped, and therefore released here, at end of scope.
    }
}

struct PreemptionChecker {
    dir: PathBuf,
}

impl<R: Resource> Preemption<R> for PreemptionChecker {
    fn should_preempt_now(&self, _task: &Task<R>) -> bool {
        todo!();
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

    struct Dummy<R> {
        _r: PhantomData<R>,
    }
    impl<R: Resource> Preemption<R> for Dummy<R> {
        fn should_preempt_now(&self, _task: &Task<R>) -> bool {
            false
        }
    }

    #[derive(Debug)]
    struct Task0 {
        id: usize,
        time: Duration,
    }

    impl<R: Resource> Executable<R> for Task0 {
        fn execute(&self, _resource: &R, _p: &dyn Preemption<R>) {
            (*RESULT_STATE).lock().unwrap().push(0);
            thread::sleep(self.time);
        }
    }

    #[derive(Debug)]
    struct Task1 {
        id: usize,
    }

    impl<R: Resource> Executable<R> for Task1 {
        fn execute(&self, _resource: &R, _p: &dyn Preemption<R>) {
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
        fn execute(&self, _resource: &R, _p: &dyn Preemption<R>) {
            let input = self.in_chan_internal.lock().unwrap().recv().unwrap_or(999);
            let result = input * input;
            self.out_chan_internal.lock().unwrap().send(result);
        }
    }

    const TEST_POLL_INTERVAL: Duration = Duration::from_millis(5);
    lazy_static! {
        static ref RESULT_STATE: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        static ref SCHEDULER: Mutex<Scheduler::<Rsrc>> = Mutex::new(
            Scheduler::<Rsrc>::new_with_poll_interval(
                tempfile::tempdir().unwrap().into_path(),
                TEST_POLL_INTERVAL
            )
            .expect("Failed to create scheduler"),
        );
    }

    #[test]
    // TODO: Fix the tests. Now that we immediately schedule when possible (to respect resource preference),
    // simultaneous deferred scheduling used by these initial tests no longer behaves as it did then.
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

        let num_resources = 3;
        let resources = (0..num_resources)
            .map(|id| Arc::new(Rsrc { id }))
            .collect::<Vec<_>>();

        let mut expected = Vec::new();

        let scheduler_handle = Scheduler::start(scheduler).expect("Failed to start scheduler.");

        let tasks0 = (0..10)
            .map(|id| Task0 {
                id,
                time: Duration::from_millis(100),
            })
            .collect::<Vec<_>>();
        for (i, task) in tasks0.into_iter().take(num_resources).enumerate() {
            /// Schedule a slow task to tie up all the resources
            /// while the next batch of `Task1`s is scheduled.
            let priority = 0;
            expected.push(0);
            scheduler.lock().unwrap().schedule(
                priority,
                &format!("{:?}", task),
                Box::new(task),
                &resources,
            );
        }

        let tasks1 = (0..5).map(|id| Task1 { id }).collect::<Vec<_>>();
        let tasks1_len = 5;
        for (i, task) in tasks1.into_iter().enumerate() {
            // When tasks are added very quickly (relative to the poll interval),
            // they should be performed in order of their priority.
            // In this group, we set priority to be the 'inverse' of task id.
            // So task 0 has a high-numbered priority and should be performed last.
            // Therefore, we push the highest id onto `expected` first.
            let priority = tasks1_len - i - 1;
            expected.push(priority);
            scheduler.lock().unwrap().schedule(
                priority,
                &format!("{:?}", task),
                Box::new(task),
                &resources,
            );
        }
        thread::sleep(Duration::from_millis(100));

        let tasks1 = (0..5).map(|id| Task1 { id }).collect::<Vec<_>>();
        for (i, task) in tasks1.into_iter().enumerate() {
            // This example is like the previous, except that we sleep for twice the length of the poll interval
            // between each call to `schedule`. TODO: set the poll interval explicitly in the test.
            // Because each task is fully processed, they are performed in the order scheduled.
            let priority = tasks1_len - i - 1;
            expected.push(i);
            thread::sleep(Duration::from_millis(200));
            scheduler.lock().unwrap().schedule(
                priority,
                &format!("{:?}", task),
                Box::new(task),
                &resources,
            );
        }
        thread::sleep(Duration::from_millis(100));

        let tasks1 = (0..5).map(|id| Task1 { id }).collect::<Vec<_>>();
        for (i, task) in tasks1.into_iter().enumerate() {
            // In this example, tasks are added quickly and with priority matching id.
            // We therefore expect them to be performed in the order scheduled.
            // This case is somewhat trivial.
            expected.push(i);
            scheduler.lock().unwrap().schedule(
                i,
                &format!("{:?}", task),
                Box::new(task),
                &resources,
            );
        }
        thread::sleep(Duration::from_millis(100));

        let tasks2 = (0..5).map(|id| Task2::new(id)).collect::<Vec<_>>();
        for (i, task) in tasks2.into_iter().enumerate() {
            // This example does not exercise the scheduler as such,
            // since results are harvested from the output channels
            // in the expected order (so scheduling does not come into play).
            // However, it does demonstrate and ensure use and usability of channels
            // to provide input to and receive output from tasks.
            let priority = tasks1_len - i - 1; // WARN: Or tasks2_len?
            task.set_input(i);
            expected.push(i * i);
            scheduler.lock().unwrap().schedule(
                priority,
                &format!("{:?}", task),
                Box::new(task),
                &resources,
            );
        }

        let tasks2 = (0..5).map(|id| Task2::new(id)).collect::<Vec<_>>();
        for (i, task) in tasks2.into_iter().enumerate() {
            let result = task.out_chan.lock().unwrap().recv().unwrap();
            (*RESULT_STATE).lock().unwrap().push(result);
        }

        thread::sleep(Duration::from_millis(100));

        scheduler.lock().unwrap().stop();

        let tasks1_len = 5;
        assert_eq!(
            num_resources + tasks1_len * 4, //num_resources + tasks1.len() * 4,
            RESULT_STATE.lock().unwrap().len()
        );

        assert_eq!(expected, *RESULT_STATE.lock().unwrap());
    }
}

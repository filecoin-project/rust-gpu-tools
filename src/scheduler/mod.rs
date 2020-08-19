extern crate futures;
extern crate rand;

use ::lazy_static::lazy_static;

use self::task_ident::TaskIdent;
use fs2::FileExt;
use futures::{future, Future};
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
    task::{Preemption, Task},
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
        Ok(Self {
            scheduler_root: Arc::new(Mutex::new(scheduler)),
            resource_schedulers: Default::default(),
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
                        scheduler.try_lock().map(|scheduler| {
                            scheduler
                                .scheduler_root
                                .try_lock()
                                .map(|mut scheduler_root| {
                                    scheduler_root.task_files.remove(&task_ident);
                                    if let Some(task_file) =
                                        scheduler_root.task_files.get(&task_ident).map(|task_set| {
                                            task_set
                                                .iter()
                                                .find(|t| (**t).path.starts_with(dir.clone()))
                                        })
                                    {
                                        task_ident.try_destroy(&dir);
                                        scheduler_root.own_tasks.remove(&task_ident);
                                        scheduler_root.task_files.remove(&task_ident);
                                    }
                                });
                        });
                    }
                    _ => (),
                }
                thread::sleep(poll_interval);
            }
        });
        Ok(SchedulerHandle {
            handle,
            control_chan: control_tx,
        })
    }

    /// Schedule `task_function` and return a `Receiver` for the result.
    pub fn schedule<F: 'static, T: 'static>(
        &mut self,
        priority: usize,
        name: &str,
        resources: &Vec<R>,
        task_function: F,
    ) -> mpsc::Receiver<T>
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
                (*tx_mutex.lock().unwrap()).send(result);
            }),
            resources,
        )
        .unwrap();
        rx
    }

    /// Schedule `task_function` and block until result is available, then return it.
    pub fn schedule_wait<F: 'static, T: 'static>(
        &mut self,
        priority: usize,
        name: &str,
        resources: &Vec<R>,
        task_function: F,
    ) -> Result<T, mpsc::RecvError>
    where
        F: FnOnce(&R) -> T + Sync + Send,
        T: Sync + Send,
    {
        let rx = self.schedule(priority, name, resources, task_function);

        rx.recv()
    }

    /// Schedule `task_function` and return a `Future` for the value.
    pub fn schedule_future<F: 'static, T: 'static>(
        &mut self,
        priority: usize,
        name: &str,
        resources: &Vec<R>,
        task_function: F,
    ) -> futures::channel::oneshot::Receiver<T>
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
                    tx.send(result);
                }
            }),
            resources,
        )
        .unwrap();
        rx
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
    ) -> Result<(), Error> {
        resources.iter().for_each(|r| {
            self.ensure_resource_scheduler(r.clone());
        });
        let task_ident = self
            .scheduler_root
            .lock()
            .unwrap()
            .new_ident(priority, name);

        self.scheduler_root
            .lock()
            .unwrap()
            .start_scheduling(task_ident.clone(), task);

        for resource in resources.iter() {
            if self
                .scheduler_root
                .lock()
                .unwrap()
                .enqueue_task_for_resource(&task_ident, resource)?
            {
                break;
            };

            // Give the just-enqueued task time to be assigned to a resource – so a less-preferred resource is not
            // assigned if a more-preferred resource is available. Note that we are not holding a lock on
            // `self.scheduler_root`, so the scheduler loop will have an opportunity to work, if possible.

            thread::sleep(self.poll_interval);
        }

        self.scheduler_root
            .lock()
            .unwrap()
            .finish_scheduling(task_ident);

        Ok(())
    }

    /// `cleanup` performs any (currently none) cleanup required when a scheduler terminates.
    fn cleanup(&self) {}

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

/// Scheduler will be terminated when `SchedulerHandle` is dropped.
pub struct SchedulerHandle {
    control_chan: Arc<Mutex<mpsc::Sender<Control>>>,
    handle: thread::JoinHandle<()>,
}

impl Drop for SchedulerHandle {
    fn drop(&mut self) {
        self.control_chan.lock().unwrap().send(Control::Stop);
    }
}

struct SchedulerRoot<R: Resource + 'static> {
    root: PathBuf,
    /// A given `Task` (identified uniquely by a `TaskIdent`) may have multiple `TaskFile`s associated,
    /// one per `Resource` for which it is currently scheduled (but only one `Resource` will eventually be assigned).
    task_files: HashMap<TaskIdent, HashSet<TaskFile>>,
    /// Each `Task` (identified uniquely by a `TaskIdent`) is protected by a `Mutex`.
    own_tasks: HashMap<TaskIdent, Task<R>>,
    /// `TaskIdent`s are added to `assigned_tasks` when they have been assigned to a resource and are being performed.
    /// Once this happens, they must not be enqueued on another resource, since that might lead to double assignment.
    assigned_tasks: HashSet<TaskIdent>,
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
            assigned_tasks: Default::default(),
            ident_counter: 0,
        })
    }

    fn new_ident(&mut self, priority: Priority, name: &str) -> TaskIdent {
        let id = self.ident_counter;
        self.ident_counter += 1;
        TaskIdent::new(priority, name, id)
    }

    fn start_scheduling(&mut self, task_ident: TaskIdent, task: Task<R>) {
        self.own_tasks.insert(task_ident.clone(), task);
    }

    // Returns true if task has already been assigned.
    fn enqueue_task_for_resource(
        &mut self,
        task_ident: &TaskIdent,
        resource: &R,
    ) -> Result<bool, Error> {
        if let Some(task) = self.own_tasks.get(&task_ident) {
            {
                if self.assigned_tasks.get(&task_ident).is_some() {
                    // If the task has been assigned to a resource already (having been scheduled here on an earlier
                    // iteration) then no further enqueueing is required.
                    return Ok(true);
                }
                let dir = self.root.join(resource.dir_id());
                create_dir_all(&dir)?;
                let task_file = task_ident.enqueue_in_dir(&dir)?;

                self.task_files
                    .entry(task_ident.clone())
                    .or_insert_with(|| Default::default())
                    .insert(task_file);
            }
        }
        Ok(false)
    }

    fn finish_scheduling(&mut self, task_ident: TaskIdent) {
        self.assigned_tasks.remove(&task_ident);
    }
}

struct ResourceScheduler<R: Resource + 'static> {
    root_scheduler: Arc<Mutex<SchedulerRoot<R>>>,
    dir: PathBuf,
    resource: R,
}

impl<'a, R: Resource + Sync + Send> ResourceScheduler<R> {
    fn new(root_scheduler: Arc<Mutex<SchedulerRoot<R>>>, dir: PathBuf, resource: R) -> Self {
        Self {
            root_scheduler,
            dir,
            resource,
        }
    }

    fn lock(&self) -> Result<ResourceLock, Error> {
        ResourceLock::acquire(&self.dir, &self.resource)
    }

    fn try_lock(&self) -> Result<Option<ResourceLock>, Error> {
        ResourceLock::maybe_acquire(&self.dir, &self.resource)
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

    fn handle_next(
        &mut self,
        control_chan: Arc<Mutex<mpsc::Sender<Control>>>,
    ) -> Result<(), Error> {
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
        let task = root_scheduler
            .own_tasks
            .remove(&ident)
            .expect("own task missing");

        if let Some(all_task_files) = root_scheduler.task_files.get(&ident) {
            all_task_files.iter().for_each(|task_file| {
                // Don't destroy this directory's task file until we are done performing the task
                if !task_file.path.starts_with(self.dir.clone()) {
                    task_file.destroy();
                };
            });
        }
        root_scheduler.assigned_tasks.insert(ident.clone());
        self.perform_task(task, control_chan, ident, self.dir.clone());
    }

    fn perform_task(
        &self,
        task: Task<R>,
        control_chan: Arc<Mutex<mpsc::Sender<Control>>>,
        ident: &TaskIdent,
        dir: PathBuf,
    ) {
        let lock = self.lock().unwrap();

        let preemption_checker = PreemptionChecker {
            dir: self.dir.clone(),
        };

        let resource = self.resource.clone();
        let ident = ident.clone();
        thread::spawn(move || {
            let _captured_lock = lock;
            (task)(&resource, &preemption_checker);
            control_chan
                .lock()
                .unwrap()
                .send(Control::Finished(dir, ident));
            // Lock is dropped, and therefore released here, at end of scope after task has been performed.
        });
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
    use crate::scheduler::futures::FutureExt;
    use futures::Future;

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
        let resources = (0..num_resources).map(|id| Rsrc { id }).collect::<Vec<_>>();

        let mut expected = Vec::new();

        let scheduler_handle = Scheduler::start(scheduler).expect("Failed to start scheduler.");

        for i in 0..num_resources {
            /// Schedule a slow task to tie up all the resources
            /// while the next batch of `Task1`s is scheduled.
            let priority = 0;
            expected.push(0);
            scheduler.lock().unwrap().schedule(
                priority,
                &format!("Task0[{}]", i),
                &resources,
                |r| {
                    (*RESULT_STATE).lock().unwrap().push(0);
                    thread::sleep(Duration::from_millis(100));
                },
            );
        }

        let tasks1_len = 5;
        for id in 0..tasks1_len {
            // When tasks are added very quickly (relative to the poll interval),
            // they should be performed in order of their priority.
            // In this group, we set priority to be the 'inverse' of task id.
            // So task 0 has a high-numbered priority and should be performed last.
            // Therefore, we push the highest id onto `expected` first.
            let priority = tasks1_len - id - 1;
            expected.push(priority);
            scheduler.lock().unwrap().schedule(
                priority,
                &format!("Task1[{}]", id),
                &resources,
                move |_| {
                    (*RESULT_STATE).lock().unwrap().push(id);
                },
            );
        }
        thread::sleep(Duration::from_millis(100));

        for id in 0..tasks1_len {
            // This example is like the previous, except that we sleep for twice the length of the poll interval
            // between each call to `schedule`. TODO: set the poll interval explicitly in the test.
            // Because each task is fully processed, they are performed in the order scheduled.
            let priority = tasks1_len - id - 1;
            expected.push(id);
            thread::sleep(Duration::from_millis(200));
            scheduler.lock().unwrap().schedule(
                priority,
                &format!("Task1[{}]", id),
                &resources,
                move |_| {
                    (*RESULT_STATE).lock().unwrap().push(id);
                },
            );
        }
        thread::sleep(Duration::from_millis(100));

        for id in 0..tasks1_len {
            // In this example, tasks are added quickly and with priority matching id.
            // We therefore expect them to be performed in the order scheduled.
            // This case is somewhat trivial.
            expected.push(id);
            scheduler.lock().unwrap().schedule(
                id,
                &format!("Task1[{}]", id),
                &resources,
                move |_| {
                    (*RESULT_STATE).lock().unwrap().push(id);
                },
            );
        }

        thread::sleep(Duration::from_millis(100));

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
            scheduler.lock().unwrap().schedule(
                priority,
                &format!("Task2[{}]", i),
                &resources,
                move |_| {
                    let input = i;
                    let result = input * input;
                    tx.lock().unwrap().send(result).unwrap();
                },
            );
        }

        for rx in out_rxs.iter() {
            let result = rx.recv().unwrap();
            (*RESULT_STATE).lock().unwrap().push(result);
        }

        thread::sleep(Duration::from_millis(100));

        scheduler.lock().unwrap().stop();

        let tasks1_len = 5;
        assert_eq!(
            num_resources + tasks1_len * 4,
            RESULT_STATE.lock().unwrap().len()
        );

        assert_eq!(expected, *RESULT_STATE.lock().unwrap());
    }

    lazy_static! {
        static ref SCHEDULER2: Mutex<Scheduler::<Rsrc>> = Mutex::new(
            Scheduler::<Rsrc>::new_with_poll_interval(
                tempfile::tempdir().unwrap().into_path(),
                TEST_POLL_INTERVAL
            )
            .expect("Failed to create scheduler"),
        );
        static ref GUARD_FAILURE: Mutex<bool> = Mutex::new(false);
        static ref RESOURCES: Vec<Rsrc> = { (0..3).map(|id| Rsrc { id }).collect::<Vec<_>>() };
        static ref RESOURCE_LOCKS: HashMap<String, Mutex<()>> = {
            let mut map = HashMap::new();
            for rsrc in RESOURCES.iter() {
                map.insert(rsrc.name(), Mutex::new(()));
            }
            map
        };
    }

    #[test]
    fn test_guards() {
        let scheduler = &*SCHEDULER2;
        let scheduler_handle = Scheduler::start(scheduler).expect("Failed to start scheduler.");

        for id in 0..10 {
            scheduler.lock().unwrap().schedule(
                0,
                &format!("MyTask[{}]", id),
                &*RESOURCES,
                move |r| {
                    let mutex = &RESOURCE_LOCKS[&r.name()];
                    // No more than one task should be able to run on a single resource at a time!
                    let lock = match mutex.try_lock() {
                        Ok(lock) => lock,
                        Err(_) => {
                            *GUARD_FAILURE.lock().unwrap() = true;
                            return;
                        }
                    };
                    thread::sleep(Duration::from_millis(100));
                },
            );
        }

        thread::sleep(Duration::from_millis(100));

        assert!(!*GUARD_FAILURE.lock().unwrap());
    }

    lazy_static! {
        static ref SCHEDULER3: Mutex<Scheduler::<Rsrc>> = Mutex::new(
            Scheduler::<Rsrc>::new_with_poll_interval(
                tempfile::tempdir().unwrap().into_path(),
                Duration::from_millis(1),
            )
            .expect("Failed to create scheduler"),
        );
    }

    #[test]

    fn test_schedule_fn() {
        let scheduler_handle = Scheduler::start(&*SCHEDULER3).expect("Failed to start scheduler.");
        thread::sleep(Duration::from_millis(300));

        let n = 10;
        let resources = (0..n).map(|id| Rsrc { id }).collect::<Vec<_>>();
        let mut futures = (0..n)
            .map(|i| {
                Some((*SCHEDULER3).lock().unwrap().schedule_future(
                    n - i,
                    &format!("task {}", i),
                    &resources,
                    move |rsrc| i,
                ))
            })
            .collect::<Vec<_>>();

        // This only tests that every function is run and returns its expected value.
        // The order of esults is determined by the order in which the task functions are scheduled,
        // not on that in which they are performed. Actual scheduling is not exercised here.
        let mut results = Vec::new();

        thread::sleep(Duration::from_millis(100));

        for future in futures.iter_mut() {
            if let Some(future) = future.take() {
                if let Some(value) = future.now_or_never() {
                    results.push(value.unwrap());
                }
            }
        }

        let mut expected = (0..n).collect::<Vec<_>>();

        assert_eq!(expected, results);
    }
}

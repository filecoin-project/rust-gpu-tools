mod error;
mod task;
mod utils;

use crate::opencl::*;
pub use error::*;
use fs2::FileExt;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::thread;
use std::time::Duration;
pub use task::*;

pub trait Resource {
    // Each resource should have a unique identifier
    fn id(&self) -> String;
}

impl Resource for Device {
    fn id(&self) -> String {
        format!("GPU-{}", self.bus_id())
    }
}

pub struct Scheduler<R: Resource> {
    root: PathBuf,
    polling: Duration,
    resources: Vec<R>,
}

impl<R: Resource> Scheduler<R> {
    pub fn new(
        root: PathBuf,
        polling: Duration,
        resources: Vec<R>,
    ) -> SchedulerResult<Scheduler<R>> {
        // Create subdirectories for all of the resources
        for resource in resources.iter() {
            fs::create_dir_all(root.join(resource.id()))?;
        }

        Ok(Scheduler {
            root,
            polling,
            resources,
        })
    }

    // Returns all tasks that are queued on the given resource by examining its
    // subdirectory.
    fn tasks_of(&self, resource: &R) -> SchedulerResult<Vec<Task>> {
        let resource_path = self.root.join(resource.id());

        Ok(fs::read_dir(&resource_path)?
            .filter_map(|res| res.ok())
            // We only consider locked files as real tasks. Those that are
            // unlocked are either dead (Because of a crash or failure) or
            // not fully initialized.
            .filter(|dir| {
                fs::File::create(dir.path())
                    .map(|f| f.try_lock_exclusive().is_err())
                    .unwrap_or(false)
            })
            // Deserialize file names to Tasks
            .filter_map(|dir| {
                dir.file_name()
                    .into_string()
                    .ok()
                    .and_then(|name| Task::from_str(&name).ok())
            })
            .collect::<Vec<_>>())
    }

    // Schedule `f` on one of the `self.resources`
    pub fn schedule<F, T>(&mut self, task: Task, f: F) -> SchedulerResult<T>
    where
        F: FnOnce(&R) -> T,
    {
        // We will keep all of our locked task files here, so that whenever `schedule` function
        // is exited (Peacefully, or because of an error/crash), all of its associated Task files
        // (And thus their locks) are dropped automatically and they no longer will be considered
        // as active tasks and won't show up when `tasks_of()` is called.
        let mut lock_files = HashMap::<PathBuf, fs::File>::new();

        loop {
            // Check if we can find a free device before enqueueing task files.
            for resource in self.resources.iter() {
                let tasks = self.tasks_of(resource)?;
                if task.has_priority_over(&tasks) {
                    // Although the given task has priority over all of the tasks currently queued
                    // on the resource, we can't use it until the corresponding lock file of that
                    // device is free. (I.e we can have the priority, but it doesn't mean that the
                    // lower-priority Task has released the resource yet)
                    // Having a separate lock-file for the resource, gives us the guarantee that
                    // no two tasks can use the resource at the same time.
                    let resource_lock_path = self.root.join(resource.id()).join("LOCK");
                    match fs::File::create(resource_lock_path)?.try_lock_exclusive() {
                        Ok(lock) => {
                            // This is not needed, and it's just for the purpose of keeping
                            // our scheduler directory clean.
                            for (path, _) in lock_files.iter() {
                                fs::remove_file(path)?;
                            }

                            let result = f(resource);
                            drop(lock);
                            return Ok(result);
                        }
                        Err(_) => {
                            continue;
                        }
                    }
                }
            }

            // No free devices found! Enqueue the task into all of the devices.
            for resource in self.resources.iter() {
                let task_path = self.root.join(resource.id()).join(&task.to_string());

                if !lock_files.contains_key(&task_path) {
                    let f = fs::File::create(&task_path)?;
                    f.lock_exclusive()?;
                    lock_files.insert(task_path, f);
                }
            }

            thread::sleep(self.polling);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::*;
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone)]
    struct DummyResource {
        id: usize,
    }

    impl Resource for DummyResource {
        fn id(&self) -> String {
            format!("Dummy-{}", self.id)
        }
    }

    #[test]
    fn test_program_pool() {
        let _ = env_logger::try_init();

        let rng = &mut thread_rng();

        const NUM_RESOURCES: usize = 3;
        const NUM_TASKS: usize = 30;

        let order = Arc::new(Mutex::new(Vec::new()));

        let resources = (0..NUM_RESOURCES)
            .map(|i| DummyResource { id: i })
            .collect::<Vec<_>>();

        // Create `NUM_TASKS` tasks all with different priorities and shuffle them.
        let mut tasks = (0..NUM_TASKS)
            .map(|i| Task::new(format!("Task{}", i), i as usize))
            .collect::<Vec<_>>();
        tasks.shuffle(rng);

        let mut threads = Vec::new();
        for t in tasks {
            let order = Arc::clone(&order);
            let resources = resources.clone();
            threads.push(thread::spawn(move || {
                let mut pool = Scheduler::<DummyResource>::new(
                    "/tmp/gpus".into(),
                    Duration::from_millis(100),
                    resources.clone(),
                )
                .unwrap();
                pool.schedule(t.clone(), |resource| {
                    info!("Task {} scheduled on {}!", t.id, resource.id());
                    if let Ok(mut order) = order.lock() {
                        order.push(t.priority);
                    }
                    thread::sleep(Duration::from_millis(500));
                })
                .unwrap();
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        let order = order.lock().unwrap()[NUM_RESOURCES..].to_vec();
        let mut sorted_order = order.clone();
        sorted_order.sort();

        assert_eq!(order, sorted_order);
    }
}

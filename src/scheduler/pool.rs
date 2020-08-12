use super::*;

#[derive(Copy, Clone, Debug)]
struct Device {
    bus_id: usize,
}

impl Resource for Device {
    fn dir_id(&self) -> String {
        self.bus_id.to_string()
    }
}

#[derive(Debug)]
struct MyTask {
    id: usize,
    time: Duration,
}

impl<R: Resource> Executable<R> for MyTask {
    fn execute(&self, resource: &R, _p: &dyn Preemption<R>) {
        println!("Start task {} on {}!", self.id, resource.name());
        thread::sleep(self.time);
        println!("Done task {}!", self.id);
    }
}

const POLL_INTERVAL: Duration = Duration::from_millis(100);
lazy_static! {
    static ref SCHEDULER: Mutex<Scheduler::<Device>> = Mutex::new(
        Scheduler::<Device>::new_with_poll_interval(
            tempfile::tempdir().unwrap().into_path(),
            POLL_INTERVAL
        )
        .expect("Failed to create scheduler"),
    );
}

#[test]
fn test_pool() {
    let scheduler = &*SCHEDULER;
    let scheduler_handle = Scheduler::start(scheduler).expect("Failed to start scheduler.");

    let num_resources = 3;
    let resources = (0..num_resources)
        .map(|bus_id| Arc::new(Device { bus_id }))
        .collect::<Vec<_>>();

    let tasks: Vec<MyTask> = (0..10)
        .map(|i| MyTask {
            id: i,
            time: Duration::from_millis(2000),
        })
        .collect();

    for t in tasks.into_iter() {
        scheduler
            .lock()
            .unwrap()
            .schedule(0, &format!("{:?}", t), Box::new(t), &resources);
    }

    thread::sleep(Duration::from_millis(3000));
}

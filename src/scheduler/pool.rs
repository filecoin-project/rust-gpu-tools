use super::*;
use rayon::prelude::*;
use std::sync::mpsc;

#[derive(Copy, Clone, Debug)]
pub struct Device {
    bus_id: usize,
}

impl Resource for Device {
    fn dir_id(&self) -> String {
        self.bus_id.to_string()
    }
}

#[derive(Debug)]
struct MyTask<F, T>
where
    F: Fn(&Device) -> T,
{
    id: usize,
    func: F,
}

impl<F, T> Executable<Device> for MyTask<F, T>
where
    F: Fn(&Device) -> T,
{
    fn execute(&self, resource: &Device, _p: &dyn Preemption<Device>) {
        (self.func)(resource);
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

pub fn schedule<F: 'static, T: 'static>(devices: &Vec<Device>, f: F) -> T
where
    F: Fn(&Device) -> T + Sync + Send,
    T: Sync + Send,
{
    let (tx, rx) = mpsc::channel();
    let tx = Mutex::new(tx);
    let t = MyTask::<_, ()> {
        id: 0,
        func: move |dev| {
            tx.lock().unwrap().send(f(dev)).unwrap();
        },
    };
    let scheduler = &*SCHEDULER;
    scheduler
        .lock()
        .unwrap()
        .schedule(0, &t.id.to_string(), Box::new(t), devices)
        .unwrap();
    rx.recv().unwrap()
}

#[test]
fn test_pool() {
    let scheduler_handle = Scheduler::start(&*SCHEDULER).expect("Failed to start scheduler.");

    let devs = (0..3).map(|bus_id| Device { bus_id }).collect::<Vec<_>>();

    (0..10)
        .into_par_iter()
        .map(|i| {
            schedule(&devs, move |dev| {
                println!("Scheduled task{} on device {}!", i, dev.name());
                thread::sleep(Duration::from_millis(2000));
                println!("Done task {}!", i);
            });
        })
        .collect::<Vec<_>>();
}

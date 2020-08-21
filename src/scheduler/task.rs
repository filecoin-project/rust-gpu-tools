use super::{Resource, ResourceScheduler};

pub trait Preemption<R: Resource> {
    // Return true if task should be preempted now.
    // `Executable`s which are preemptible, must call this method.
    fn should_preempt_now(&self) -> bool;
}

impl<'a, R: Resource> Preemption<R> for ResourceScheduler<R> {
    /// The current `Task` should be preempted if the high-priority lock has been acquired
    /// by another `Task`.
    fn should_preempt_now(&self) -> bool {
        todo!();
    }
}

pub type Task<R> = Box<dyn FnOnce(&R, &dyn Preemption<R>) -> () + Sync + Send>;

use std::marker::PhantomData;
use std::sync::Arc;

use super::{Resource, ResourceScheduler};

pub trait Preemption<R: Resource> {
    // Return true if task should be preempted now.
    // `Executable`s which are preemptible, must call this method.
    fn should_preempt_now(&self, _task: &Task<R>) -> bool;
}

impl<'a, R: Resource> Preemption<R> for ResourceScheduler<R> {
    /// The current `Task` should be preempted if the high-priority lock has been acquired
    /// by another `Task`.
    fn should_preempt_now(&self, _task: &Task<R>) -> bool {
        todo!();
    }
}

pub type Task<R: Resource + 'static> = Box<dyn FnOnce(&R, &dyn Preemption<R>) -> () + Sync + Send>;

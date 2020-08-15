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

pub struct Task<R: Resource + 'static> {
    /// These are the resources for which the `Task` has been requested to be scheduled,
    /// in order of preference. It is guaranteed that the `Task` will be scheduled on only one of these.
    pub(crate) executable: Arc<Box<dyn Fn(&R, &dyn Preemption<R>) -> () + Sync + Send>>,
}

impl<'a, R: Resource> Task<R> {
    pub fn new(executable: Arc<Box<dyn Fn(&R, &dyn Preemption<R>) -> () + Sync + Send>>) -> Self {
        Self {
            executable: executable,
        }
    }

    pub fn execute(&self, resource: &R, preemption: &dyn Preemption<R>) {
        (self.executable)(resource, preemption)
    }
}

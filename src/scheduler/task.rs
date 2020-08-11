use std::sync::Arc;

use super::{Resource, ResourceScheduler};

/// Implementers of `Executable` act as a callback which executes the job associated with a task.
pub trait Executable<R: Resource> {
    /// `execute` executes a task's job. `preempt.should_preempt_now()` should be polled as appropriate,
    /// and execution should terminate if it returns true. Tasks which are not preemptible need not
    /// ever check for preemption.
    fn execute(&self, resource: &R, preempt: &dyn Preemption<R>);

    /// Returns true if the job associated with this `Executable` can be preempted. `Executable`s
    /// which return `true` should periodically poll for preemption while executing.
    fn is_preemptible(&self) -> bool {
        false
    }
}

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
    pub(crate) executable: Arc<Box<dyn Executable<R> + Sync + Send>>,
}

impl<'a, R: Resource> Task<R> {
    pub fn new(executable: Arc<Box<dyn Executable<R> + Sync + Send>>) -> Self {
        Self {
            executable: executable,
        }
    }

    pub fn execute(&self, resource: &R, preemption: &dyn Preemption<R>) {
        self.executable.execute(resource, preemption)
    }
}

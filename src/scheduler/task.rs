use super::{Resource, ResourceScheduler};

/// Implementers of `Executable` act as a callback which executes the job associated with a task.
pub trait Executable<R: Resource> {
    /// `execute` executes a task's job. `preempt.should_preempt_now()` should be polled as appropriate,
    /// and execution should terminate if it returns true. Tasks which are not preemptible need not
    /// ever check for preemption.
    fn execute(&self, preempt: &dyn Preemption<R>);

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

impl<'a, R: Resource> Preemption<R> for ResourceScheduler<'a, R> {
    /// The current `Task` should be preempted if the high-priority lock has been acquired
    /// by another `Task`.
    fn should_preempt_now(&self, _task: &Task<R>) -> bool {
        todo!();
    }
}

pub struct Task<'a, R: Resource> {
    /// These are the resources for which the `Task` has been requested to be scheduled,
    /// in order of preference. It is guaranteed that the `Task` will be scheduled on only one of these.
    executable: Box<&'a dyn Executable<R>>,
}

impl<'a, R: Resource> Clone for Task<'a, R> {
    fn clone(&self) -> Self {
        Self {
            executable: Box::new(*self.executable),
        }
    }
}

impl<'a, R: Resource> Task<'a, R> {
    pub fn new(executable: Box<&'a dyn Executable<R>>) -> Self {
        Self { executable }
    }

    pub fn execute(&self, preempt: &dyn Preemption<R>) {
        self.executable.execute(preempt)
    }
}

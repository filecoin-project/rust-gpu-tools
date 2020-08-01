use super::*;

// When `Ord` is derived on structs, it will produce a lexicographic ordering based on
// the top-to-bottom declaration order of the struct's members. Thus first based on
// `priority`, then `timestamp`.
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct Task {
    pub(super) priority: usize,
    pub(super) timestamp: u64,
    pub(super) id: String,
}

impl Task {
    pub fn new(id: String, priority: usize) -> Task {
        Task {
            id,
            priority,
            timestamp: utils::timestamp(),
        }
    }

    // Returns true if the task has higher priority over all of the given `tasks` based
    // on `priority` and `timestamp`.
    pub fn has_priority_over(&self, tasks: &Vec<Task>) -> bool {
        let mut sorted = tasks.clone();
        sorted.push(self.clone());
        sorted.sort();
        sorted[0] == *self
    }
}

// Serialize to a file name
impl ToString for Task {
    fn to_string(&self) -> String {
        format!("{}-{}-{}", self.priority, self.timestamp, self.id)
    }
}

// Deserialize from a file name
impl std::str::FromStr for Task {
    type Err = SchedulerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = s.split("-").collect();
        if parts.len() != 3 {
            return Err(SchedulerError::ParseError);
        }
        Ok(Task {
            priority: usize::from_str(parts[0]).map_err(|_| SchedulerError::ParseError)?,
            timestamp: u64::from_str(parts[1]).map_err(|_| SchedulerError::ParseError)?,
            id: parts[2].into(),
        })
    }
}

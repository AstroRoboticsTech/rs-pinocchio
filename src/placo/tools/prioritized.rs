//! Priority handling shared by tasks and constraints (PlaCo `Prioritized`).

/// The priority of a task or constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Priority {
    /// Enforced exactly, as an equality constraint.
    Hard,
    /// Best-effort, added to the objective with a weight.
    #[default]
    Soft,
    /// Like [`Priority::Hard`], but with a decision-variable scaling factor.
    Scaled,
}

impl Priority {
    /// Parses `"hard"`, `"soft"` or `"scaled"`.
    ///
    /// # Panics
    /// On any other string.
    pub fn from_name(name: &str) -> Self {
        match name {
            "hard" => Priority::Hard,
            "soft" => Priority::Soft,
            "scaled" => Priority::Scaled,
            other => panic!("Prioritized: invalid priority: {other}"),
        }
    }

    /// The lower-case name of the priority.
    pub fn name(&self) -> &'static str {
        match self {
            Priority::Hard => "hard",
            Priority::Soft => "soft",
            Priority::Scaled => "scaled",
        }
    }
}

/// A named object carrying a [`Priority`] and a soft-task weight.
#[derive(Clone, Debug)]
pub struct Prioritized {
    /// Object name.
    pub name: String,
    /// Object priority.
    pub priority: Priority,
    /// Weight, used for soft objects only.
    pub weight: f64,
}

impl Default for Prioritized {
    fn default() -> Self {
        Self {
            name: String::new(),
            priority: Priority::Soft,
            weight: 1.0,
        }
    }
}

impl Prioritized {
    /// A new soft, unit-weight, unnamed object.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets name, priority and weight in one call.
    pub fn configure(&mut self, name: impl Into<String>, priority: Priority, weight: f64) {
        self.name = name.into();
        self.priority = priority;
        self.weight = weight;
    }

    /// The lower-case name of the current priority.
    pub fn priority_name(&self) -> &'static str {
        self.priority.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_roundtrips_through_name() {
        for p in [Priority::Hard, Priority::Soft, Priority::Scaled] {
            assert_eq!(Priority::from_name(p.name()), p);
        }
    }

    #[test]
    fn configure_sets_fields() {
        let mut o = Prioritized::new();
        o.configure("task", Priority::Hard, 2.5);
        assert_eq!(o.name, "task");
        assert_eq!(o.priority, Priority::Hard);
        assert_eq!(o.weight, 2.5);
        assert_eq!(o.priority_name(), "hard");
    }
}

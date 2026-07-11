//! QP constraints built from expressions (PlaCo `ProblemConstraint`).

use super::expression::Expression;

/// Whether a constraint is an equality or an inequality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ConstraintType {
    /// `A·x + b = 0`.
    #[default]
    Equality,
    /// `A·x + b >= 0`.
    Inequality,
}

/// Whether a constraint must hold or is merely an objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ConstraintPriority {
    /// Best-effort: folded into the objective (weighted).
    Soft,
    /// Must be enforced (default).
    #[default]
    Hard,
}

/// A constraint to be enforced (or minimized) by a [`super::Problem`].
///
/// Build these with the [`Expression`] comparison helpers
/// (`e1.equal_to(&e2)`, `e.leq_scalar(1.0)`, …), then optionally
/// [`Constraint::configure`] the priority/weight.
#[derive(Clone, Debug, PartialEq)]
pub struct Constraint {
    /// The `A·x + b` expression compared against zero.
    pub expression: Expression,
    /// Equality or inequality.
    pub type_: ConstraintType,
    /// Hard or soft.
    pub priority: ConstraintPriority,
    /// Weight, used for soft constraints only.
    pub weight: f64,
    /// Set by the solver: whether the constraint is active at the optimum.
    pub is_active: bool,
}

impl Constraint {
    /// An equality constraint `expression = 0` (hard by default).
    pub fn equality(expression: Expression) -> Self {
        Self {
            expression,
            type_: ConstraintType::Equality,
            priority: ConstraintPriority::Hard,
            weight: 1.0,
            is_active: false,
        }
    }

    /// An inequality constraint `expression >= 0` (hard by default).
    pub fn inequality(expression: Expression) -> Self {
        Self {
            expression,
            type_: ConstraintType::Inequality,
            priority: ConstraintPriority::Hard,
            weight: 1.0,
            is_active: false,
        }
    }

    /// Sets priority and weight; returns `self` for chaining.
    pub fn configure(&mut self, priority: ConstraintPriority, weight: f64) -> &mut Self {
        self.priority = priority;
        self.weight = weight;
        self
    }

    /// Sets priority from `"hard"`/`"soft"` and a weight; returns `self`.
    ///
    /// # Panics
    /// If `priority` is neither `"hard"` nor `"soft"`.
    pub fn configure_named(&mut self, priority: &str, weight: f64) -> &mut Self {
        self.priority = match priority {
            "hard" => ConstraintPriority::Hard,
            "soft" => ConstraintPriority::Soft,
            other => panic!("Constraint: invalid priority: {other}"),
        };
        self.weight = weight;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn scalar_expr() -> Expression {
        Expression {
            a: DMatrix::from_row_slice(1, 1, &[1.0]),
            b: DVector::from_element(1, 0.0),
        }
    }

    #[test]
    fn defaults_are_hard() {
        let c = Constraint::equality(scalar_expr());
        assert_eq!(c.type_, ConstraintType::Equality);
        assert_eq!(c.priority, ConstraintPriority::Hard);
        assert_eq!(c.weight, 1.0);
        assert!(!c.is_active);
    }

    #[test]
    fn configure_named_sets_soft() {
        let mut c = Constraint::inequality(scalar_expr());
        c.configure_named("soft", 3.0);
        assert_eq!(c.priority, ConstraintPriority::Soft);
        assert_eq!(c.weight, 3.0);
    }
}

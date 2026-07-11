//! The QP modeling layer (PlaCo `placo::problem`).
//!
//! Build a [`Problem`], add [`Variable`]s, form affine [`Expression`]s from them,
//! turn comparisons into [`Constraint`]s, then [`Problem::solve`]. Soft
//! constraints become weighted least-squares objectives; hard ones become QP
//! equality/inequality constraints.

mod constraint;
mod error;
mod expression;
mod integrator;
mod polygon_constraint;
#[allow(clippy::module_inception)]
mod problem;
mod problem_polynom;
mod sparsity;
mod variable;

pub use constraint::{Constraint, ConstraintPriority, ConstraintType};
pub use error::{QpError, QpResult};
pub use expression::Expression;
pub use integrator::{Integrator, Trajectory};
pub use polygon_constraint::{in_polygon, in_polygon_xy};
pub use problem::Problem;
pub use problem_polynom::ProblemPolynom;
pub use sparsity::{Interval, Sparsity};
pub use variable::Variable;

//! Task-space inverse dynamics (PlaCo `placo::dynamics`).
//!
//! A [`DynamicsSolver`] collects [`DynamicsTask`]s (PD control → accelerations,
//! or torque relations) and [`Contact`]s (force variables + friction cones), and
//! solves a QP over `[qdd, contact forces]` subject to the equation of motion for
//! the joint torques. Requires the `ffi` feature (it drives Pinocchio through
//! [`crate::placo::model::RobotWrapper`]).

mod contacts;
mod more_tasks;
mod relative_tasks;
mod solver;
mod task;
mod tasks;

pub use contacts::{Contact, Contact6D, ExternalWrenchContact, PointContact, PuppetContact};
pub use more_tasks::{CoMTask, OrientationTask, TargetTau, TorqueTask};
pub use relative_tasks::{RelativeOrientationTask, RelativePositionTask};
pub use solver::{ContactId, DynamicsResult, DynamicsSolver, FrameTaskHandle, TaskId};
pub use task::{DynamicsTask, TaskBase};
pub use tasks::{JointsTask, PositionTask};

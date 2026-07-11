//! Task-space inverse kinematics (PlaCo `placo::kinematics`).
//!
//! A [`KinematicsSolver`] collects [`KinematicsTask`]s — each producing an
//! `A·qd = b` relation from a [`crate::placo::model::RobotWrapper`] Jacobian —
//! and solves for the joint velocity `qd` via the [`crate::placo::problem`] QP
//! layer. Requires the `ffi` feature (it drives Pinocchio through RobotWrapper).

mod constraints;
mod relative_tasks;
mod solver;
mod task;
mod tasks;

pub use constraints::{
    CoMPolygonConstraint, ConeConstraint, DistanceConstraint, JointSpaceHalfSpacesConstraint,
    KinematicsConstraint, YawConstraint,
};
pub use relative_tasks::{
    AxisAlignTask, CentroidalMomentumTask, DistanceTask, RelativeOrientationTask,
    RelativePositionTask,
};
pub use solver::{ConstraintId, FrameTaskHandle, KinematicsSolver, TaskId};
pub use task::{KinematicsTask, TaskBase};
pub use tasks::{CoMTask, JointsTask, OrientationTask, PositionTask, RegularizationTask};

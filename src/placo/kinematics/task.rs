//! The kinematics task interface (PlaCo `kinematics::Task`).
//!
//! A task expresses `A·qd = b`, where `qd` is the joint velocity (delta) solved
//! by the [`super::KinematicsSolver`]. `A` (rows × nv) and `b` (rows) are rebuilt
//! from the robot state and target on each solve. Hard tasks become equality
//! constraints; soft tasks become weighted least-squares objectives.

use std::any::Any;

use nalgebra::{DMatrix, DVector};

use crate::error::Result;
use crate::placo::model::RobotWrapper;
use crate::placo::tools::{AxisesMask, Priority};

/// Shared task state: name, priority, weight, and the last-built `(A, b)`.
#[derive(Clone, Debug)]
pub struct TaskBase {
    /// Task name.
    pub name: String,
    /// Priority (hard = equality, soft = weighted objective).
    pub priority: Priority,
    /// Soft-task weight.
    pub weight: f64,
    /// Axis mask applied to the task rows.
    pub mask: AxisesMask,
    /// `A` in `A·qd = b`, rebuilt by [`KinematicsTask::update`].
    pub a: DMatrix<f64>,
    /// `b` in `A·qd = b`.
    pub b: DVector<f64>,
}

impl Default for TaskBase {
    fn default() -> Self {
        Self {
            name: String::new(),
            priority: Priority::Soft,
            weight: 1.0,
            mask: AxisesMask::new(),
            a: DMatrix::zeros(0, 0),
            b: DVector::zeros(0),
        }
    }
}

/// A kinematics task solved by the [`super::KinematicsSolver`].
pub trait KinematicsTask: Any {
    /// Shared task state.
    fn base(&self) -> &TaskBase;
    /// Mutable shared task state.
    fn base_mut(&mut self) -> &mut TaskBase;
    /// Rebuilds `A` and `b` from the current robot state and target. `dt` is the
    /// solver timestep (0 if unset); most tasks ignore it.
    fn update(&mut self, robot: &mut RobotWrapper, dt: f64) -> Result<()>;
    /// Task type name (e.g. `"position"`).
    fn type_name(&self) -> &'static str;
    /// Downcast hook for typed reconfiguration.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// The task priority.
    fn priority(&self) -> Priority {
        self.base().priority
    }
    /// The task weight.
    fn weight(&self) -> f64 {
        self.base().weight
    }
    /// The last-built `A`.
    fn a(&self) -> &DMatrix<f64> {
        &self.base().a
    }
    /// The last-built `b`.
    fn b(&self) -> &DVector<f64> {
        &self.base().b
    }
    /// The task error norm (`‖b‖`).
    fn error_norm(&self) -> f64 {
        self.base().b.norm()
    }
}

//! The dynamics task interface (PlaCo `dynamics::Task`).
//!
//! A dynamics task produces `A·qdd = b` (a desired acceleration relation) via PD
//! control on a task error, or — when `tau_task` is set — a relation on the joint
//! torques. Tasks are assembled by the [`super::DynamicsSolver`].

use std::any::Any;

use nalgebra::{DMatrix, DVector};

use crate::error::Result;
use crate::placo::model::RobotWrapper;
use crate::placo::tools::{AxisesMask, Priority};

/// Shared dynamics-task state.
#[derive(Clone, Debug)]
pub struct TaskBase {
    /// Task name.
    pub name: String,
    /// Priority (hard/soft).
    pub priority: Priority,
    /// Soft weight.
    pub weight: f64,
    /// Proportional gain.
    pub kp: f64,
    /// Derivative gain (negative → critically damped from `kp`).
    pub kd: f64,
    /// Whether the task constrains torques (`true`) or accelerations (`false`).
    pub tau_task: bool,
    /// Axis mask.
    pub mask: AxisesMask,
    /// `A` in `A·qdd = b` (or the torque selector when `tau_task`).
    pub a: DMatrix<f64>,
    /// `b` in `A·qdd = b`.
    pub b: DVector<f64>,
}

impl Default for TaskBase {
    fn default() -> Self {
        Self {
            name: String::new(),
            priority: Priority::Soft,
            weight: 1.0,
            kp: 1e3,
            kd: -1.0,
            tau_task: false,
            mask: AxisesMask::new(),
            a: DMatrix::zeros(0, 0),
            b: DVector::zeros(0),
        }
    }
}

impl TaskBase {
    /// The derivative gain actually used (critically damped `2√kp` when `kd < 0`).
    pub fn get_kd(&self) -> f64 {
        if self.kd < 0.0 {
            2.0 * self.kp.sqrt()
        } else {
            self.kd
        }
    }
}

/// A task solved by the [`super::DynamicsSolver`].
pub trait DynamicsTask: Any {
    /// Shared state.
    fn base(&self) -> &TaskBase;
    /// Mutable shared state.
    fn base_mut(&mut self) -> &mut TaskBase;
    /// Rebuilds `A`/`b` from the robot state and target.
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()>;
    /// Task type name.
    fn type_name(&self) -> &'static str;
    /// Downcast hook.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// The task priority.
    fn priority(&self) -> Priority {
        self.base().priority
    }
    /// The task weight.
    fn weight(&self) -> f64 {
        self.base().weight
    }
    /// Whether this is a torque task.
    fn is_tau_task(&self) -> bool {
        self.base().tau_task
    }
    /// The last-built `A`.
    fn a(&self) -> &DMatrix<f64> {
        &self.base().a
    }
    /// The last-built `b`.
    fn b(&self) -> &DVector<f64> {
        &self.base().b
    }
}

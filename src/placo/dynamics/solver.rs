//! The task-space inverse-dynamics solver (PlaCo `DynamicsSolver`).
//!
//! Solves a QP over `[qdd, contact_forces]` subject to the equation of motion
//! `tau = M·qdd + h − Σ Jcᵀ f`, with the floating base unactuated
//! (`tau[0..6] = 0`) and the actuated torques minimized. Acceleration tasks
//! constrain `qdd`; torque tasks constrain `tau`.

use nalgebra::{DVector, Matrix3};

use super::contacts::{
    Contact, Contact6D, ExternalWrenchContact, LineContact, PointContact, PuppetContact,
};
use super::more_tasks::{CoMTask, OrientationTask, TorqueTask};
use super::relative_tasks::{RelativeOrientationTask, RelativePositionTask};
use super::task::DynamicsTask;
use super::tasks::{JointsTask, PositionTask};
use crate::error::{Error, Result};
use crate::placo::model::RobotWrapper;
use crate::placo::problem::{Constraint, ConstraintPriority, Expression, Problem};
use crate::placo::tools::Priority;

/// Handle to a dynamics task.
pub type TaskId = usize;
/// Handle to a contact.
pub type ContactId = usize;

/// Handles to the position + orientation sub-tasks of a dynamics frame task.
#[derive(Clone, Copy, Debug)]
pub struct FrameTaskHandle {
    /// The position sub-task.
    pub position: TaskId,
    /// The orientation sub-task.
    pub orientation: TaskId,
}

/// The result of a dynamics solve.
#[derive(Clone, Debug)]
pub struct DynamicsResult {
    /// Whether the solve succeeded.
    pub success: bool,
    /// Generalized torques (length `nv`; the first 6 are the floating base).
    pub tau: DVector<f64>,
    /// Generalized accelerations (length `nv`).
    pub qdd: DVector<f64>,
    /// Generalized torques produced by the contact forces (length `nv`).
    pub tau_contacts: DVector<f64>,
}

/// A QP-based inverse-dynamics solver over a [`RobotWrapper`].
pub struct DynamicsSolver {
    tasks: Vec<Box<dyn DynamicsTask>>,
    contacts: Vec<Box<dyn Contact>>,
    n: usize,
    wrenches: Vec<Option<DVector<f64>>>,
    /// Fix the floating base (`qdd[0..6] = 0`, and give it torque authority).
    pub masked_fbase: bool,
    /// Use only gravity (instead of the full non-linear effects) in `h`.
    pub gravity_only: bool,
    /// Joint velocity damping added to `tau`.
    pub damping: f64,
    /// Weight of the actuated-torque minimization.
    pub torque_cost: f64,
    /// Integration timestep (for `solve(integrate = true)`, and the limits).
    pub dt: f64,
    /// Enforce actuated torque limits (from the URDF effort limits).
    pub torque_limits: bool,
    /// Enforce joint velocity limits (needs `dt`).
    pub velocity_limits: bool,
    /// Enforce joint position limits (needs `dt`).
    pub joint_limits: bool,
    /// Safe deceleration used by the position limits [rad/s²].
    pub qdd_safe: f64,
}

impl DynamicsSolver {
    /// Builds a solver for `robot`.
    pub fn new(robot: &RobotWrapper) -> Self {
        Self {
            tasks: Vec::new(),
            contacts: Vec::new(),
            n: robot.nv(),
            wrenches: Vec::new(),
            masked_fbase: false,
            gravity_only: false,
            damping: 0.0,
            torque_cost: 1e-3,
            dt: 0.0,
            torque_limits: false,
            velocity_limits: false,
            joint_limits: false,
            qdd_safe: 1.0,
        }
    }

    fn add_limits(
        &self,
        problem: &mut Problem,
        qdd: crate::placo::problem::Variable,
        tau: &Expression,
        robot: &RobotWrapper,
    ) -> Result<()> {
        let count = self.n - 6;
        if (self.velocity_limits || self.joint_limits) && self.dt == 0.0 {
            return Err(Error::Solver("dynamics limits enabled but dt is 0".into()));
        }

        if self.torque_limits {
            let effort = robot.effort_limits().rows(6, count).into_owned();
            problem.add_constraint(tau.slice(6, count).leq_vector(effort.clone()));
            problem.add_constraint(tau.slice(6, count).geq_vector(-effort));
        }

        if self.velocity_limits {
            let vlim = robot.velocity_limits().rows(6, count).into_owned();
            let qd_bottom = robot.state.qd.rows(6, count).into_owned();
            let e = qdd
                .expr_slice(6, count)
                .scale(self.dt)
                .add_vector(&qd_bottom);
            problem.add_constraint(e.leq_vector(vlim.clone()));
            problem.add_constraint(e.geq_vector(-vlim));
        }

        if self.joint_limits {
            let (lower, upper) = robot.position_limits();
            let qd_bottom = robot.state.qd.rows(6, count).into_owned();
            let mut qd_max_up = DVector::zeros(count);
            let mut qd_max_lo = DVector::zeros(count);
            for k in 0..count {
                // q index = v index + 1 for the actuated (single-DoF) joints.
                let q = robot.state.q[k + 7];
                let du = (upper[k + 7] - q).clamp(0.0, 1e6);
                let dl = (q - lower[k + 7]).clamp(0.0, 1e6);
                qd_max_up[k] = (2.0 * du * self.qdd_safe).sqrt();
                qd_max_lo[k] = (2.0 * dl * self.qdd_safe).sqrt();
            }
            let e = qdd
                .expr_slice(6, count)
                .scale(self.dt)
                .add_vector(&qd_bottom);
            problem.add_constraint(e.leq_vector(qd_max_up));
            problem.add_constraint(e.geq_vector(-qd_max_lo));
        }
        Ok(())
    }

    fn push_task(&mut self, task: Box<dyn DynamicsTask>) -> TaskId {
        self.tasks.push(task);
        self.tasks.len() - 1
    }

    /// Adds a position task on `frame_index`.
    pub fn add_position_task(
        &mut self,
        frame_index: usize,
        target_world: nalgebra::Vector3<f64>,
    ) -> TaskId {
        self.push_task(Box::new(PositionTask::new(frame_index, target_world)))
    }

    /// Adds an orientation task on `frame_index`.
    pub fn add_orientation_task(
        &mut self,
        frame_index: usize,
        r_world_frame: Matrix3<f64>,
    ) -> TaskId {
        self.push_task(Box::new(OrientationTask::new(frame_index, r_world_frame)))
    }

    /// Adds a frame (position + orientation) task on `frame_index`.
    pub fn add_frame_task(
        &mut self,
        frame_index: usize,
        t_world_frame: nalgebra::Isometry3<f64>,
    ) -> FrameTaskHandle {
        let position = self.add_position_task(frame_index, t_world_frame.translation.vector);
        let r = t_world_frame.rotation.to_rotation_matrix().into_inner();
        let orientation = self.add_orientation_task(frame_index, r);
        FrameTaskHandle {
            position,
            orientation,
        }
    }

    /// Adds a CoM task.
    pub fn add_com_task(&mut self, target_world: nalgebra::Vector3<f64>) -> TaskId {
        self.push_task(Box::new(CoMTask::new(target_world)))
    }

    /// Adds a relative-position task (position of `frame_b` in `frame_a`).
    pub fn add_relative_position_task(
        &mut self,
        frame_a: usize,
        frame_b: usize,
        target: nalgebra::Vector3<f64>,
    ) -> TaskId {
        self.push_task(Box::new(RelativePositionTask::new(
            frame_a, frame_b, target,
        )))
    }

    /// Adds a relative-orientation task (orientation of `frame_b` in `frame_a`).
    pub fn add_relative_orientation_task(
        &mut self,
        frame_a: usize,
        frame_b: usize,
        r_a_b: Matrix3<f64>,
    ) -> TaskId {
        self.push_task(Box::new(RelativeOrientationTask::new(
            frame_a, frame_b, r_a_b,
        )))
    }

    /// Adds a relative-frame (position + orientation) task: the pose of
    /// `frame_b` expressed in `frame_a`.
    pub fn add_relative_frame_task(
        &mut self,
        frame_a: usize,
        frame_b: usize,
        t_a_b: nalgebra::Isometry3<f64>,
    ) -> FrameTaskHandle {
        let position = self.add_relative_position_task(frame_a, frame_b, t_a_b.translation.vector);
        let r = t_a_b.rotation.to_rotation_matrix().into_inner();
        let orientation = self.add_relative_orientation_task(frame_a, frame_b, r);
        FrameTaskHandle {
            position,
            orientation,
        }
    }

    /// Adds an (empty) torque task.
    pub fn add_torque_task(&mut self) -> TaskId {
        self.push_task(Box::new(TorqueTask::new()))
    }

    /// Adds an (empty) joints task.
    pub fn add_joints_task(&mut self) -> TaskId {
        self.push_task(Box::new(JointsTask::new()))
    }

    /// Sets a task's priority and weight.
    pub fn configure_task(&mut self, id: TaskId, priority: Priority, weight: f64) {
        if let Some(task) = self.tasks.get_mut(id) {
            let base = task.base_mut();
            base.priority = priority;
            base.weight = weight;
        }
    }

    /// Downcasts a task to its concrete type (e.g. to set gains or a target).
    pub fn task_mut<T: DynamicsTask>(&mut self, id: TaskId) -> Option<&mut T> {
        self.tasks.get_mut(id)?.as_any_mut().downcast_mut::<T>()
    }

    /// Adds a bilateral (fixed) point contact on `frame_index`.
    pub fn add_point_contact(&mut self, frame_index: usize) -> ContactId {
        self.contacts
            .push(Box::new(PointContact::new(frame_index, false)));
        self.contacts.len() - 1
    }

    /// Adds a unilateral point contact (friction cone, pushes only).
    pub fn add_unilateral_point_contact(&mut self, frame_index: usize) -> ContactId {
        self.contacts
            .push(Box::new(PointContact::new(frame_index, true)));
        self.contacts.len() - 1
    }

    /// Adds a bilateral (fixed) 6-DoF contact on `frame_index`.
    pub fn add_fixed_contact(&mut self, frame_index: usize) -> ContactId {
        self.contacts
            .push(Box::new(Contact6D::new(frame_index, false)));
        self.contacts.len() - 1
    }

    /// Adds a unilateral planar 6-DoF contact (set its `length`/`width` via
    /// [`DynamicsSolver::contact_mut`]).
    pub fn add_planar_contact(&mut self, frame_index: usize) -> ContactId {
        self.contacts
            .push(Box::new(Contact6D::new(frame_index, true)));
        self.contacts.len() - 1
    }

    /// Adds a unilateral line ("knife-edge") contact along the frame's local
    /// x-axis (set its `length` via [`DynamicsSolver::contact_mut`]).
    pub fn add_line_contact(&mut self, frame_index: usize) -> ContactId {
        self.contacts
            .push(Box::new(LineContact::new(frame_index, true)));
        self.contacts.len() - 1
    }

    /// Adds a puppet contact: an unconstrained generalized force on every DoF
    /// (identity Jacobian), fully actuating the robot.
    pub fn add_puppet_contact(&mut self) -> ContactId {
        self.contacts.push(Box::new(PuppetContact::new()));
        self.contacts.len() - 1
    }

    /// Adds a known external wrench applied at `frame_index` (fixed, not
    /// optimized). Set the wrench via [`DynamicsSolver::contact_mut`].
    pub fn add_external_wrench_contact(
        &mut self,
        frame_index: usize,
        reference: crate::ReferenceFrame,
    ) -> ContactId {
        self.contacts
            .push(Box::new(ExternalWrenchContact::new(frame_index, reference)));
        self.contacts.len() - 1
    }

    /// Downcasts a contact to its concrete type.
    pub fn contact_mut<T: Contact>(&mut self, id: ContactId) -> Option<&mut T> {
        self.contacts.get_mut(id)?.as_any_mut().downcast_mut::<T>()
    }

    /// The wrench of contact `id` from the last solve (if it was active).
    pub fn contact_wrench(&self, id: ContactId) -> Option<&DVector<f64>> {
        self.wrenches.get(id).and_then(|w| w.as_ref())
    }

    /// Solves the inverse-dynamics QP. With `integrate`, applies `qdd` over `dt`.
    pub fn solve(&mut self, robot: &mut RobotWrapper, integrate: bool) -> Result<DynamicsResult> {
        let nv = self.n;
        let mut problem = Problem::new();
        let qdd = problem.add_variable(nv);

        for task in &mut self.tasks {
            task.update(robot)?;
        }

        // Contact force variables.
        let mut contact_fvars: Vec<(usize, crate::placo::problem::Variable)> = Vec::new();
        for (ci, contact) in self.contacts.iter_mut().enumerate() {
            if !contact.active() {
                continue;
            }
            contact.update(robot)?;
            let fvar = problem.add_variable(contact.size());
            contact_fvars.push((ci, fvar));
        }

        // tau = M·qdd + damping·qd + h − Σ Jcᵀ f.
        let m = robot.mass_matrix()?;
        let mut tau = qdd.expr().left_multiply(&m);
        if self.damping != 0.0 {
            tau = tau.add_vector(&(&robot.state.qd * self.damping));
        }
        let h = if self.gravity_only {
            robot.generalized_gravity()?
        } else {
            robot.non_linear_effects()?
        };
        tau = tau.add_vector(&h);
        for (ci, fvar) in &contact_fvars {
            let jt = self.contacts[*ci].jacobian().transpose();
            tau = tau.subtract(&fvar.expr().left_multiply(&jt));
        }

        if self.masked_fbase {
            problem.add_constraint(qdd.expr_slice(0, 6).equal_vector(DVector::zeros(6)));
        }

        // Tasks.
        for task in &self.tasks {
            if task.a().nrows() == 0 {
                continue;
            }
            let expr = if task.is_tau_task() {
                tau.left_multiply(task.a()).sub_vector(task.b())
            } else {
                Expression {
                    a: task.a().clone(),
                    b: -task.b(),
                }
            };
            let mut constraint = Constraint::equality(expr);
            match task.priority() {
                Priority::Soft => {
                    constraint.configure(ConstraintPriority::Soft, task.weight());
                }
                Priority::Hard => {}
                Priority::Scaled => {
                    return Err(Error::Solver(
                        "DynamicsSolver: Scaled priority is not supported".into(),
                    ));
                }
            }
            problem.add_constraint(constraint);
        }

        // Contact constraints (friction cones, etc.).
        for (ci, fvar) in &contact_fvars {
            self.contacts[*ci].add_constraints(&mut problem, *fvar);
        }

        // Torque / velocity / joint limits.
        self.add_limits(&mut problem, qdd, &tau, robot)?;

        // Floating base has no actuation (unless masked).
        if !self.masked_fbase {
            problem.add_constraint(tau.slice(0, 6).equal_vector(DVector::zeros(6)));
        }
        // Minimize actuated torques.
        problem
            .add_constraint(tau.slice(6, nv - 6).equal_vector(DVector::zeros(nv - 6)))
            .configure(ConstraintPriority::Soft, self.torque_cost);

        problem
            .solve()
            .map_err(|e| Error::Solver(format!("dynamics solve failed: {e}")))?;

        let sol = problem.solution().clone();
        let tau_val = tau.value(&sol);
        let qdd_val = qdd.value(&problem);

        // Contact wrenches + their generalized torques.
        self.wrenches = vec![None; self.contacts.len()];
        let mut tau_contacts = DVector::zeros(nv);
        for (ci, fvar) in &contact_fvars {
            let wrench = fvar.value(&problem);
            tau_contacts += self.contacts[*ci].jacobian().transpose() * &wrench;
            self.wrenches[*ci] = Some(wrench);
        }

        if integrate {
            if self.dt == 0.0 {
                return Err(Error::Solver(
                    "dynamics integrate requested but dt is 0".into(),
                ));
            }
            robot.state.qdd = qdd_val.clone();
            if self.masked_fbase {
                for i in 0..6 {
                    robot.state.qdd[i] = 0.0;
                }
            }
            robot.integrate(self.dt)?;
        }

        Ok(DynamicsResult {
            success: true,
            tau: tau_val,
            qdd: qdd_val,
            tau_contacts,
        })
    }
}

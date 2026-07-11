//! The task-space inverse-kinematics solver (PlaCo `KinematicsSolver`).

use std::collections::BTreeSet;

use nalgebra::{DVector, Isometry3, Matrix3, Vector2, Vector3};

use super::constraints::{
    AvoidSelfCollisionsConstraint, CoMPolygonConstraint, ConeConstraint, DistanceConstraint,
    JointSpaceHalfSpacesConstraint, KinematicsConstraint, YawConstraint,
};
use super::more_tasks::{
    GearTask, KineticEnergyRegularizationTask, ManipulabilityTask, ManipulabilityType,
};
use super::relative_tasks::{
    AxisAlignTask, CentroidalMomentumTask, DistanceTask, RelativeOrientationTask,
    RelativePositionTask,
};
use super::task::KinematicsTask;
use super::tasks::{CoMTask, JointsTask, OrientationTask, PositionTask, RegularizationTask};
use super::wheel_task::WheelTask;
use crate::error::Result;
use crate::placo::model::RobotWrapper;
use crate::placo::problem::{Constraint, ConstraintPriority, Expression, Problem};
use crate::placo::tools::Priority;

/// Handle to a task added to the solver.
pub type TaskId = usize;

/// Handle to a constraint added to the solver.
pub type ConstraintId = usize;

/// Handles to the position + orientation tasks of an [`KinematicsSolver::add_frame_task`].
#[derive(Clone, Copy, Debug)]
pub struct FrameTaskHandle {
    /// The position sub-task.
    pub position: TaskId,
    /// The orientation sub-task.
    pub orientation: TaskId,
}

/// A QP-based inverse-kinematics solver over a [`RobotWrapper`].
///
/// Add tasks (and optionally masks / limits), then call [`KinematicsSolver::solve`]
/// repeatedly, applying the resulting `qd` to the robot to converge.
pub struct KinematicsSolver {
    tasks: Vec<Box<dyn KinematicsTask>>,
    constraints: Vec<Box<dyn KinematicsConstraint>>,
    masked_dof: BTreeSet<usize>,
    masked_fbase: bool,
    n: usize,
    /// Enforce joint position limits (needs URDF limits). Off by default.
    pub joint_limits: bool,
    /// Enforce joint velocity limits (needs [`KinematicsSolver::dt`]). Off by default.
    pub velocity_limits: bool,
    /// Integration timestep used by the velocity limits (0 disables the check).
    pub dt: f64,
}

impl KinematicsSolver {
    /// Builds a solver for `robot`.
    pub fn new(robot: &RobotWrapper) -> Self {
        Self {
            tasks: Vec::new(),
            constraints: Vec::new(),
            masked_dof: BTreeSet::new(),
            masked_fbase: false,
            n: robot.nv(),
            joint_limits: false,
            velocity_limits: false,
            dt: 0.0,
        }
    }

    fn push(&mut self, task: Box<dyn KinematicsTask>) -> TaskId {
        self.tasks.push(task);
        self.tasks.len() - 1
    }

    fn push_constraint(&mut self, constraint: Box<dyn KinematicsConstraint>) -> ConstraintId {
        self.constraints.push(constraint);
        self.constraints.len() - 1
    }

    /// Adds a cone constraint (z-axes of two frames within `angle_max`).
    pub fn add_cone_constraint(
        &mut self,
        frame_a: usize,
        frame_b: usize,
        angle_max: f64,
    ) -> ConstraintId {
        self.push_constraint(Box::new(ConeConstraint::new(frame_a, frame_b, angle_max)))
    }

    /// Adds a yaw constraint (relative yaw of two frames within `± angle_max`).
    pub fn add_yaw_constraint(
        &mut self,
        frame_a: usize,
        frame_b: usize,
        angle_max: f64,
    ) -> ConstraintId {
        self.push_constraint(Box::new(YawConstraint::new(frame_a, frame_b, angle_max)))
    }

    /// Adds a distance constraint (frames at most `distance_max` apart).
    pub fn add_distance_constraint(
        &mut self,
        frame_a: usize,
        frame_b: usize,
        distance_max: f64,
    ) -> ConstraintId {
        self.push_constraint(Box::new(DistanceConstraint::new(
            frame_a,
            frame_b,
            distance_max,
        )))
    }

    /// Adds a CoM-polygon constraint (CoM xy inside a clockwise polygon).
    pub fn add_com_polygon_constraint(
        &mut self,
        polygon: Vec<Vector2<f64>>,
        margin: f64,
    ) -> ConstraintId {
        self.push_constraint(Box::new(CoMPolygonConstraint::new(polygon, margin)))
    }

    /// Adds a joint-space half-spaces constraint `A·q ≤ b` (`A` has `nq` cols).
    pub fn add_joint_space_half_spaces_constraint(
        &mut self,
        a: nalgebra::DMatrix<f64>,
        b: nalgebra::DVector<f64>,
    ) -> ConstraintId {
        self.push_constraint(Box::new(JointSpaceHalfSpacesConstraint::new(a, b)))
    }

    /// Adds a self-collision-avoidance constraint. Supply the nearest-point
    /// distances each solve via [`KinematicsSolver::constraint_mut`].
    pub fn add_avoid_self_collisions_constraint(&mut self) -> ConstraintId {
        self.push_constraint(Box::new(AvoidSelfCollisionsConstraint::new()))
    }

    /// Sets a constraint's priority (hard/soft) and soft weight.
    pub fn configure_constraint(
        &mut self,
        id: ConstraintId,
        priority: ConstraintPriority,
        weight: f64,
    ) {
        if let Some(c) = self.constraints.get_mut(id) {
            c.set_priority_weight(priority, weight);
        }
    }

    /// Downcasts a constraint to its concrete type (e.g. to supply collision
    /// distances to an [`AvoidSelfCollisionsConstraint`]).
    pub fn constraint_mut<T: KinematicsConstraint>(&mut self, id: ConstraintId) -> Option<&mut T> {
        self.constraints
            .get_mut(id)?
            .as_any_mut()
            .downcast_mut::<T>()
    }

    /// Adds a position task on `frame_index` targeting `target_world`.
    pub fn add_position_task(&mut self, frame_index: usize, target_world: Vector3<f64>) -> TaskId {
        self.push(Box::new(PositionTask::new(frame_index, target_world)))
    }

    /// Adds an orientation task on `frame_index` targeting `r_world_frame`.
    pub fn add_orientation_task(
        &mut self,
        frame_index: usize,
        r_world_frame: Matrix3<f64>,
    ) -> TaskId {
        self.push(Box::new(OrientationTask::new(frame_index, r_world_frame)))
    }

    /// Adds a position + orientation task on `frame_index` targeting `t_world_frame`.
    pub fn add_frame_task(
        &mut self,
        frame_index: usize,
        t_world_frame: Isometry3<f64>,
    ) -> FrameTaskHandle {
        let position = self.add_position_task(frame_index, t_world_frame.translation.vector);
        let r = t_world_frame.rotation.to_rotation_matrix().into_inner();
        let orientation = self.add_orientation_task(frame_index, r);
        FrameTaskHandle {
            position,
            orientation,
        }
    }

    /// Adds a relative-position task: position of `frame_b` in `frame_a` → `target`.
    pub fn add_relative_position_task(
        &mut self,
        frame_a: usize,
        frame_b: usize,
        target: Vector3<f64>,
    ) -> TaskId {
        self.push(Box::new(RelativePositionTask::new(
            frame_a, frame_b, target,
        )))
    }

    /// Adds a relative-orientation task: `R_a_b` → `r_a_b`.
    pub fn add_relative_orientation_task(
        &mut self,
        frame_a: usize,
        frame_b: usize,
        r_a_b: Matrix3<f64>,
    ) -> TaskId {
        self.push(Box::new(RelativeOrientationTask::new(
            frame_a, frame_b, r_a_b,
        )))
    }

    /// Adds a relative-frame task (relative position + orientation).
    pub fn add_relative_frame_task(
        &mut self,
        frame_a: usize,
        frame_b: usize,
        t_a_b: Isometry3<f64>,
    ) -> FrameTaskHandle {
        let position = self.add_relative_position_task(frame_a, frame_b, t_a_b.translation.vector);
        let r = t_a_b.rotation.to_rotation_matrix().into_inner();
        let orientation = self.add_relative_orientation_task(frame_a, frame_b, r);
        FrameTaskHandle {
            position,
            orientation,
        }
    }

    /// Adds an axis-align task: aligns `axis_frame` (in the frame) with
    /// `target_axis_world`.
    pub fn add_axis_align_task(
        &mut self,
        frame_index: usize,
        axis_frame: Vector3<f64>,
        target_axis_world: Vector3<f64>,
    ) -> TaskId {
        self.push(Box::new(AxisAlignTask::new(
            frame_index,
            axis_frame,
            target_axis_world,
        )))
    }

    /// Adds a distance task: distance between `frame_a` and `frame_b` → `distance`.
    pub fn add_distance_task(&mut self, frame_a: usize, frame_b: usize, distance: f64) -> TaskId {
        self.push(Box::new(DistanceTask::new(frame_a, frame_b, distance)))
    }

    /// Adds a centroidal-momentum task (angular). Requires [`KinematicsSolver::dt`].
    pub fn add_centroidal_momentum_task(&mut self, l_world: Vector3<f64>) -> TaskId {
        self.push(Box::new(CentroidalMomentumTask::new(l_world)))
    }

    /// Adds a CoM task targeting `target_world`.
    pub fn add_com_task(&mut self, target_world: Vector3<f64>) -> TaskId {
        self.push(Box::new(CoMTask::new(target_world)))
    }

    /// Adds an (initially empty) joints task.
    pub fn add_joints_task(&mut self) -> TaskId {
        self.push(Box::new(JointsTask::new()))
    }

    /// Adds a regularization task with the given magnitude (soft, weight 1).
    pub fn add_regularization_task(&mut self, magnitude: f64) -> TaskId {
        let id = self.push(Box::new(RegularizationTask::new(magnitude)));
        self.configure_task(id, "regularization", Priority::Soft, 1.0);
        id
    }

    /// Adds an (empty) gear task coupling joints with ratios.
    pub fn add_gear_task(&mut self) -> TaskId {
        self.push(Box::new(GearTask::new()))
    }

    /// Adds a rolling-without-slipping wheel task on `joint` (spinning about its
    /// local z-axis) with the given `radius`. Set `t_world_surface` (default
    /// identity) via [`KinematicsSolver::task_mut`].
    pub fn add_wheel_task(
        &mut self,
        joint: impl Into<String>,
        radius: f64,
        omniwheel: bool,
    ) -> TaskId {
        self.push(Box::new(WheelTask::new(joint, radius, omniwheel)))
    }

    /// Adds a manipulability task on `frame_index` (soft; gradient-ascent on
    /// `sqrt(det(J·Jᵀ))`).
    pub fn add_manipulability_task(
        &mut self,
        frame_index: usize,
        kind: ManipulabilityType,
        lambda: f64,
    ) -> TaskId {
        self.push(Box::new(ManipulabilityTask::new(frame_index, kind, lambda)))
    }

    /// Adds a kinetic-energy regularization task (requires
    /// [`KinematicsSolver::dt`]). Soft, with the given weight.
    pub fn add_kinetic_energy_regularization_task(&mut self, magnitude: f64) -> TaskId {
        let id = self.push(Box::new(KineticEnergyRegularizationTask::new()));
        self.configure_task(
            id,
            "kinetic_energy_regularization",
            Priority::Soft,
            magnitude,
        );
        id
    }

    /// Sets a task's name, priority and weight.
    pub fn configure_task(&mut self, id: TaskId, name: &str, priority: Priority, weight: f64) {
        if let Some(task) = self.tasks.get_mut(id) {
            let base = task.base_mut();
            base.name = name.to_string();
            base.priority = priority;
            base.weight = weight;
        }
    }

    /// Downcasts a task to its concrete type for reconfiguration (e.g. to update
    /// a [`PositionTask`]'s target between solves).
    pub fn task_mut<T: KinematicsTask>(&mut self, id: TaskId) -> Option<&mut T> {
        self.tasks.get_mut(id)?.as_any_mut().downcast_mut::<T>()
    }

    /// The number of tasks.
    pub fn tasks_count(&self) -> usize {
        self.tasks.len()
    }

    /// Masks all velocity DoFs of joint `name` (fixes them at zero delta).
    pub fn mask_dof(&mut self, robot: &RobotWrapper, name: &str) -> Result<()> {
        let offset = robot.joint_v_offset(name)?;
        for i in 0..robot.joint_v_size(name)? {
            self.masked_dof.insert(offset + i);
        }
        Ok(())
    }

    /// Unmasks all velocity DoFs of joint `name`.
    pub fn unmask_dof(&mut self, robot: &RobotWrapper, name: &str) -> Result<()> {
        let offset = robot.joint_v_offset(name)?;
        for i in 0..robot.joint_v_size(name)? {
            self.masked_dof.remove(&(offset + i));
        }
        Ok(())
    }

    /// Masks (or unmasks) the 6 floating-base DoFs.
    pub fn mask_fbase(&mut self, masked: bool) {
        self.masked_fbase = masked;
    }

    /// Solves for the joint velocity `qd`. With `apply`, integrates it into the
    /// robot's configuration. Refreshes the robot kinematics first.
    pub fn solve(&mut self, robot: &mut RobotWrapper, apply: bool) -> Result<DVector<f64>> {
        robot.update_kinematics()?;

        let n = self.n;
        let mut problem = Problem::new();
        let qd = problem.add_variable(n);

        for task in &mut self.tasks {
            task.update(robot, self.dt)?;
            if task.a().nrows() == 0 {
                continue;
            }
            let expr = Expression {
                a: task.a().clone(),
                b: -task.b(),
            };
            let mut constraint = Constraint::equality(expr);
            match task.priority() {
                Priority::Soft => {
                    constraint.configure(ConstraintPriority::Soft, task.weight());
                }
                // Hard and (for now) Scaled are enforced as equalities.
                Priority::Hard | Priority::Scaled => {}
            }
            problem.add_constraint(constraint);
        }

        for &joint in &self.masked_dof {
            problem.add_constraint(qd.expr_slice(joint, 1).equal_scalar(0.0));
        }
        if self.masked_fbase {
            problem.add_constraint(qd.expr_slice(0, 6).equal_vector(DVector::zeros(6)));
        }

        self.add_limits(&mut problem, qd, robot)?;

        for constraint in &self.constraints {
            constraint.add_to(&mut problem, qd, robot, self.dt)?;
        }

        problem
            .solve()
            .map_err(|e| crate::error::Error::Solver(e.to_string()))?;
        let qd_sol = qd.value(&problem);

        if apply {
            robot.integrate_configuration(&qd_sol)?;
        }
        Ok(qd_sol)
    }

    fn add_limits(
        &self,
        problem: &mut Problem,
        qd: crate::placo::problem::Variable,
        robot: &RobotWrapper,
    ) -> Result<()> {
        let n = self.n;
        let count = n - 6;

        if self.joint_limits {
            let (lower, upper) = robot.position_limits();
            let q_bottom = robot.state.q.rows(7, count).into_owned();
            let lower_bottom = lower.rows(7, count).into_owned();
            let upper_bottom = upper.rows(7, count).into_owned();
            let e = qd.expr_slice(6, count).add_vector(&q_bottom);
            problem.add_constraint(e.leq_vector(upper_bottom));
            problem.add_constraint(e.geq_vector(lower_bottom));
        }

        if self.velocity_limits {
            if self.dt == 0.0 {
                return Err(crate::error::Error::Solver(
                    "velocity limits enabled but solver.dt is 0".into(),
                ));
            }
            let vlimit = robot.velocity_limits().rows(6, count).into_owned();
            let e = qd.expr_slice(6, count);
            problem.add_constraint(e.leq_vector(self.dt * &vlimit));
            problem.add_constraint(e.geq_vector(-self.dt * &vlimit));
        }
        Ok(())
    }
}

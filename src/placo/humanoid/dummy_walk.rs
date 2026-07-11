//! A minimal open-loop walk generator (PlaCo `DummyWalk`).
//!
//! Owns a [`RobotWrapper`] + a kinematics solver with foot/trunk frame tasks,
//! and marches it through steps by interpolating the flying foot (with a lift
//! spline) and solving IK. A simple alternative to the full
//! [`super::WalkPatternGenerator`] for testing / teleoperation.

use nalgebra::{Isometry3, Vector3};

use super::footsteps::FootstepsPlanner;
use super::footsteps::FootstepsPlannerRepetitive;
use super::parameters::HumanoidParameters;
use super::side::Side;
use crate::error::{Error, Result};
use crate::placo::kinematics::{FrameTaskHandle, KinematicsSolver, OrientationTask, PositionTask};
use crate::placo::model::RobotWrapper;
use crate::placo::tools::{flatten_on_floor, interpolate_frames, CubicSpline};

fn translation(x: f64, y: f64, z: f64) -> Isometry3<f64> {
    Isometry3::translation(x, y, z)
}

/// An open-loop humanoid walk driven by frame tasks.
pub struct DummyWalk {
    /// The robot being walked.
    pub robot: RobotWrapper,
    /// Walk parameters.
    pub parameters: HumanoidParameters,
    /// The internal kinematics solver.
    pub solver: KinematicsSolver,
    /// Trunk x-offset [m].
    pub trunk_x_offset: f64,
    /// Whether the current support is the left foot.
    pub support_left: bool,
    /// Left foot at the start of the current step.
    pub t_world_left: Isometry3<f64>,
    /// Right foot at the start of the current step.
    pub t_world_right: Isometry3<f64>,
    /// Target of the current flying foot.
    pub t_world_next: Isometry3<f64>,
    /// Last requested step.
    pub dx: f64,
    /// Last requested step.
    pub dy: f64,
    /// Last requested step.
    pub dtheta: f64,
    lift_spline: CubicSpline,
    footsteps_planner: FootstepsPlannerRepetitive,
    left_foot_task: FrameTaskHandle,
    right_foot_task: FrameTaskHandle,
    trunk_task: FrameTaskHandle,
}

impl DummyWalk {
    /// Builds a dummy walk over `robot` (frame names `left_foot`/`right_foot`/
    /// `trunk`), resetting to a neutral double stance.
    pub fn new(mut robot: RobotWrapper, parameters: HumanoidParameters) -> Result<Self> {
        let find = |r: &RobotWrapper, name: &str| {
            r.frame_index(name)
                .ok_or_else(|| Error::FrameNotFound(name.to_string()))
        };
        let left_foot = find(&robot, "left_foot")?;
        let right_foot = find(&robot, "right_foot")?;
        let trunk = find(&robot, "trunk")?;

        let mut solver = KinematicsSolver::new(&robot);
        solver.velocity_limits = true;
        solver.dt = 0.1;
        robot.update_kinematics()?;

        let left_foot_task = solver.add_frame_task(left_foot, robot.t_world_frame(left_foot)?);
        let right_foot_task = solver.add_frame_task(right_foot, robot.t_world_frame(right_foot)?);
        let trunk_task = solver.add_frame_task(trunk, robot.t_world_frame(trunk)?);

        let footsteps_planner = FootstepsPlannerRepetitive::new(parameters.clone());

        let mut dw = Self {
            robot,
            parameters,
            solver,
            trunk_x_offset: 0.05,
            support_left: false,
            t_world_left: Isometry3::identity(),
            t_world_right: Isometry3::identity(),
            t_world_next: Isometry3::identity(),
            dx: 0.0,
            dy: 0.0,
            dtheta: 0.0,
            lift_spline: CubicSpline::new(false),
            footsteps_planner,
            left_foot_task,
            right_foot_task,
            trunk_task,
        };
        dw.reset(false)?;
        Ok(dw)
    }

    /// Resets the walk to a neutral double stance.
    pub fn reset(&mut self, support_left: bool) -> Result<()> {
        let h = self.parameters.walk_foot_height;
        let rise = self.parameters.walk_foot_rise_ratio;
        self.lift_spline.clear();
        self.lift_spline.add_point(0.0, 0.0, 0.0);
        self.lift_spline.add_point(0.5 - rise / 2.0, h, 0.0);
        self.lift_spline.add_point(0.5 + rise / 2.0, h, 0.0);
        self.lift_spline.add_point(1.0, 0.0, 0.0);

        self.robot.reset();
        self.robot.update_kinematics()?;

        self.support_left = support_left;
        let s = self.parameters.feet_spacing / 2.0;
        self.t_world_left = translation(0.0, s, 0.0);
        self.t_world_right = translation(0.0, -s, 0.0);

        self.compute_next_support(0.0, 0.0, 0.0);
        self.update(0.0)
    }

    /// Advances to the next step, swapping the support foot.
    pub fn next_step(&mut self, dx: f64, dy: f64, dtheta: f64) {
        if self.support_left {
            self.t_world_right = self.t_world_next;
        } else {
            self.t_world_left = self.t_world_next;
        }
        self.support_left = !self.support_left;
        self.compute_next_support(dx, dy, dtheta);
    }

    /// Updates the internal IK for the step phase `t` in `[0, 1]`.
    pub fn update(&mut self, t: f64) -> Result<()> {
        let mut t_left = self.t_world_left;
        let mut t_right = self.t_world_right;

        if self.support_left {
            t_right = interpolate_frames(&self.t_world_right, &self.t_world_next, t);
            t_right.translation.z = self.lift_spline.pos(t);
        } else {
            t_left = interpolate_frames(&self.t_world_left, &self.t_world_next, t);
            t_left.translation.z = self.lift_spline.pos(t);
        }

        let t_mid =
            interpolate_frames(&flatten_on_floor(&t_left), &flatten_on_floor(&t_right), 0.5);
        let t_trunk = t_mid
            * translation(self.trunk_x_offset, 0.0, self.parameters.walk_com_height)
            * Isometry3::rotation(Vector3::y() * self.parameters.walk_trunk_pitch);

        self.set_frame_target(self.left_foot_task, t_left);
        self.set_frame_target(self.right_foot_task, t_right);
        self.set_frame_target(self.trunk_task, t_trunk);
        self.solve()
    }

    fn set_frame_target(&mut self, handle: FrameTaskHandle, target: Isometry3<f64>) {
        if let Some(p) = self.solver.task_mut::<PositionTask>(handle.position) {
            p.target_world = target.translation.vector;
        }
        if let Some(o) = self.solver.task_mut::<OrientationTask>(handle.orientation) {
            o.r_world_frame = target.rotation.to_rotation_matrix().into_inner();
        }
    }

    fn compute_next_support(&mut self, dx: f64, dy: f64, dtheta: f64) {
        self.dx = dx;
        self.dy = dy;
        self.dtheta = dtheta;
        self.footsteps_planner.configure(dx, dy, dtheta, 2);
        let flying = if self.support_left {
            Side::Right
        } else {
            Side::Left
        };
        let footsteps = self
            .footsteps_planner
            .plan(flying, self.t_world_left, self.t_world_right);
        self.t_world_next = footsteps[2].frame;
    }

    fn solve(&mut self) -> Result<()> {
        for _ in 0..4 {
            self.robot.update_kinematics()?;
            self.solver.solve(&mut self.robot, true)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Structural coverage lives in tests/walk_pipeline.rs (needs Pinocchio).
}

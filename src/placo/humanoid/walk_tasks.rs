//! Wiring a walk trajectory into the kinematics solver (PlaCo `WalkTasks`).
//!
//! Holds the frame/CoM/trunk kinematics tasks that make a
//! [`super::WalkTrajectory`] executable on a [`super::HumanoidRobot`]: feed the
//! trajectory targets at time `t`, then solve the IK to track the walk.

use nalgebra::{Isometry3, Matrix3, Vector3};

use super::humanoid_robot::HumanoidRobot;
use super::walk_pattern_generator::WalkTrajectory;
use crate::error::Result;
use crate::placo::kinematics::{
    CoMTask, FrameTaskHandle, KinematicsSolver, OrientationTask, PositionTask, TaskId,
};
use crate::placo::tools::Priority;

/// The kinematics tasks that track a walk: both feet (frame tasks), the trunk
/// orientation, and the CoM.
pub struct WalkTasks {
    /// Left-foot frame task.
    pub left_foot: FrameTaskHandle,
    /// Right-foot frame task.
    pub right_foot: FrameTaskHandle,
    /// Trunk orientation task.
    pub trunk_orientation: TaskId,
    /// CoM task.
    pub com: TaskId,
    /// Extra CoM x offset in the trunk frame.
    pub com_x: f64,
    /// Extra CoM y offset in the trunk frame.
    pub com_y: f64,
}

impl WalkTasks {
    /// Creates the walk tracking tasks on `solver`, initialized to the robot's
    /// current feet/trunk/CoM (all soft, unit weight).
    pub fn initialize(solver: &mut KinematicsSolver, robot: &mut HumanoidRobot) -> Result<Self> {
        robot.robot.update_kinematics()?;
        let t_left = robot.t_world_left()?;
        let t_right = robot.t_world_right()?;
        let r_trunk = robot
            .t_world_trunk()?
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let com = robot.robot.com_world()?;

        let left_foot = solver.add_frame_task(robot.left_foot, t_left);
        solver.configure_task(
            left_foot.position,
            "left_foot_position",
            Priority::Soft,
            1.0,
        );
        solver.configure_task(
            left_foot.orientation,
            "left_foot_orientation",
            Priority::Soft,
            1.0,
        );

        let right_foot = solver.add_frame_task(robot.right_foot, t_right);
        solver.configure_task(
            right_foot.position,
            "right_foot_position",
            Priority::Soft,
            1.0,
        );
        solver.configure_task(
            right_foot.orientation,
            "right_foot_orientation",
            Priority::Soft,
            1.0,
        );

        let trunk_orientation = solver.add_orientation_task(robot.trunk, r_trunk);
        solver.configure_task(trunk_orientation, "trunk", Priority::Soft, 1.0);

        let com_task = solver.add_com_task(com);
        solver.configure_task(com_task, "com", Priority::Soft, 1.0);

        Ok(Self {
            left_foot,
            right_foot,
            trunk_orientation,
            com: com_task,
            com_x: 0.0,
            com_y: 0.0,
        })
    }

    /// Points the tasks at explicit targets.
    pub fn update(
        &self,
        solver: &mut KinematicsSolver,
        robot: &HumanoidRobot,
        t_world_left: Isometry3<f64>,
        t_world_right: Isometry3<f64>,
        com_world: Vector3<f64>,
        r_world_trunk: Matrix3<f64>,
    ) -> Result<()> {
        let offset = robot
            .t_world_trunk()?
            .rotation
            .to_rotation_matrix()
            .into_inner()
            * Vector3::new(self.com_x, self.com_y, 0.0);

        if let Some(p) = solver.task_mut::<PositionTask>(self.left_foot.position) {
            p.target_world = t_world_left.translation.vector;
        }
        if let Some(o) = solver.task_mut::<OrientationTask>(self.left_foot.orientation) {
            o.r_world_frame = t_world_left.rotation.to_rotation_matrix().into_inner();
        }
        if let Some(p) = solver.task_mut::<PositionTask>(self.right_foot.position) {
            p.target_world = t_world_right.translation.vector;
        }
        if let Some(o) = solver.task_mut::<OrientationTask>(self.right_foot.orientation) {
            o.r_world_frame = t_world_right.rotation.to_rotation_matrix().into_inner();
        }
        if let Some(o) = solver.task_mut::<OrientationTask>(self.trunk_orientation) {
            o.r_world_frame = r_world_trunk;
        }
        if let Some(c) = solver.task_mut::<CoMTask>(self.com) {
            c.target_world = com_world + offset;
        }
        Ok(())
    }

    /// Points the tasks at the walk trajectory sampled at time `t`.
    pub fn update_from_trajectory(
        &self,
        solver: &mut KinematicsSolver,
        robot: &HumanoidRobot,
        trajectory: &mut WalkTrajectory,
        t: f64,
    ) -> Result<()> {
        let t_left = trajectory.t_world_left(t);
        let t_right = trajectory.t_world_right(t);
        let com = trajectory.p_world_com(t);
        let r_trunk = trajectory.r_world_trunk(t);
        self.update(solver, robot, t_left, t_right, com, r_trunk)
    }
}

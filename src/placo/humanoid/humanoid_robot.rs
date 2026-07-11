//! Humanoid robot model (PlaCo `HumanoidRobot`).
//!
//! Wraps a [`RobotWrapper`] with bipedal support-state management (which foot is
//! on the floor, floor placement, DCM/ZMP). Frame names are configurable (PlaCo
//! hard-codes `left_foot`/`right_foot`/`trunk`); the defaults match.

use std::path::Path;

use nalgebra::{Isometry3, Vector2};

use super::side::Side;
use crate::error::{Error, Result};
use crate::placo::model::RobotWrapper;
use crate::placo::tools::flatten_on_floor;

/// A bipedal humanoid model with support-state tracking.
pub struct HumanoidRobot {
    /// The underlying robot model.
    pub robot: RobotWrapper,
    /// Current support side.
    pub support_side: Side,
    /// Whether both feet support the robot.
    pub support_is_both: bool,
    /// Frame index of the current support foot.
    pub support_frame: usize,
    /// World placement of the support.
    pub t_world_support: Isometry3<f64>,
    /// Left-foot frame index.
    pub left_foot: usize,
    /// Right-foot frame index.
    pub right_foot: usize,
    /// Trunk frame index.
    pub trunk: usize,
}

impl HumanoidRobot {
    /// Loads a humanoid using the default frame names `left_foot`, `right_foot`,
    /// `trunk`.
    pub fn from_urdf(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_urdf_with_frames(path, "left_foot", "right_foot", "trunk")
    }

    /// Loads a humanoid with explicit foot/trunk frame names.
    pub fn from_urdf_with_frames(
        path: impl AsRef<Path>,
        left_foot: &str,
        right_foot: &str,
        trunk: &str,
    ) -> Result<Self> {
        let robot = RobotWrapper::from_urdf(path)?;
        let find = |name: &str| {
            robot
                .frame_index(name)
                .ok_or_else(|| Error::FrameNotFound(name.to_string()))
        };
        let left_foot = find(left_foot)?;
        let right_foot = find(right_foot)?;
        let trunk = find(trunk)?;
        let mut humanoid = Self {
            robot,
            support_side: Side::Left,
            support_is_both: false,
            support_frame: left_foot,
            t_world_support: Isometry3::identity(),
            left_foot,
            right_foot,
            trunk,
        };
        humanoid.init_config()?;
        Ok(humanoid)
    }

    fn init_config(&mut self) -> Result<()> {
        self.support_side = Side::Left;
        self.support_frame = self.left_foot;
        self.support_is_both = false;
        self.t_world_support = Isometry3::identity();
        self.ensure_on_floor()
    }

    /// Places the robot so its current support foot sits at `t_world_support`.
    pub fn ensure_on_floor(&mut self) -> Result<()> {
        self.robot.update_kinematics()?;
        self.robot
            .set_t_world_frame(self.support_frame, self.t_world_support)?;
        self.robot.update_kinematics()
    }

    /// Switches the support to `side` (Left/Right), updating the support frame
    /// and its floor placement.
    pub fn update_support_side(&mut self, side: Side) -> Result<()> {
        let frame = if side == Side::Left {
            self.left_foot
        } else {
            self.right_foot
        };
        if frame != self.support_frame {
            self.support_frame = frame;
            self.support_side = side;
            self.support_is_both = false;
            self.t_world_support = flatten_on_floor(&self.robot.t_world_frame(frame)?);
        }
        Ok(())
    }

    /// World placement of the left foot.
    pub fn t_world_left(&self) -> Result<Isometry3<f64>> {
        self.robot.t_world_frame(self.left_foot)
    }

    /// World placement of the right foot.
    pub fn t_world_right(&self) -> Result<Isometry3<f64>> {
        self.robot.t_world_frame(self.right_foot)
    }

    /// World placement of the trunk.
    pub fn t_world_trunk(&self) -> Result<Isometry3<f64>> {
        self.robot.t_world_frame(self.trunk)
    }

    /// Divergent Component of Motion: `c_xy + com_velocity / omega`.
    pub fn dcm(&mut self, omega: f64, com_velocity: Vector2<f64>) -> Result<Vector2<f64>> {
        let com = self.robot.com_world()?;
        Ok(com.xy() + com_velocity / omega)
    }

    /// Zero-tilting Moment Point: `c_xy - com_acceleration / omega²`.
    pub fn zmp(&mut self, omega: f64, com_acceleration: Vector2<f64>) -> Result<Vector2<f64>> {
        let com = self.robot.com_world()?;
        Ok(com.xy() - com_acceleration / (omega * omega))
    }
}

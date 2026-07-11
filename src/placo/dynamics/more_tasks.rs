//! Orientation, CoM and torque dynamics tasks (PlaCo `dynamics::OrientationTask`,
//! `CoMTask`, `TorqueTask`).

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

use super::task::{DynamicsTask, TaskBase};
use crate::error::Result;
use crate::placo::model::RobotWrapper;
use crate::ReferenceFrame;

fn vec3_to_col(v: &Vector3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 1, v.as_slice())
}

fn col_to_vector(m: DMatrix<f64>) -> DVector<f64> {
    m.column(0).into_owned()
}

/// PD acceleration task on a frame's orientation (PlaCo dynamics
/// `OrientationTask`).
pub struct OrientationTask {
    base: TaskBase,
    /// Target frame.
    pub frame_index: usize,
    /// Target orientation `R_world_frame`.
    pub r_world_frame: Matrix3<f64>,
    /// Target angular velocity.
    pub omega_world: Vector3<f64>,
    /// Target angular acceleration.
    pub domega_world: Vector3<f64>,
}

impl OrientationTask {
    pub(crate) fn new(frame_index: usize, r_world_frame: Matrix3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            frame_index,
            r_world_frame,
            omega_world: Vector3::zeros(),
            domega_world: Vector3::zeros(),
        }
    }
}

impl DynamicsTask for OrientationTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "orientation"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        let j = robot.frame_jacobian(self.frame_index, ReferenceFrame::World)?;
        let dj = robot.frame_jacobian_time_variation(self.frame_index, ReferenceFrame::World)?;
        let j_rot = j.rows(3, 3).into_owned();
        let dj_rot = dj.rows(3, 3).into_owned();

        let t = robot.t_world_frame(self.frame_index)?;
        let r_current = t.rotation.to_rotation_matrix().into_inner();
        let error = Rotation3::from_matrix_unchecked(self.r_world_frame * r_current.transpose())
            .scaled_axis();
        let velocity_world = &j_rot * &robot.state.qd;
        let velocity_error = self.omega_world - velocity_world;
        let kd = self.base.get_kd();
        let desired_acc = self.base.kp * error + kd * velocity_error + self.domega_world;

        self.base.mask.r_local_world = self.r_world_frame.transpose();
        self.base.a = self.base.mask.apply(&j_rot);
        let b_vec = desired_acc - dj_rot * &robot.state.qd;
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&b_vec)));
        Ok(())
    }
}

/// PD acceleration task on the center of mass (PlaCo dynamics `CoMTask`).
pub struct CoMTask {
    base: TaskBase,
    /// Target CoM position in the world.
    pub target_world: Vector3<f64>,
    /// Target CoM velocity.
    pub dtarget_world: Vector3<f64>,
    /// Target CoM acceleration.
    pub ddtarget_world: Vector3<f64>,
}

impl CoMTask {
    pub(crate) fn new(target_world: Vector3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            target_world,
            dtarget_world: Vector3::zeros(),
            ddtarget_world: Vector3::zeros(),
        }
    }
}

impl DynamicsTask for CoMTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "com"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        let j = robot.com_jacobian()?;
        let dj = robot.com_jacobian_time_variation()?;
        let position_error = self.target_world - robot.com_world()?;
        let velocity_world = &j * &robot.state.qd;
        let velocity_error = self.dtarget_world - velocity_world;
        let kd = self.base.get_kd();
        let desired_acc = self.base.kp * position_error + kd * velocity_error + self.ddtarget_world;

        self.base.a = self.base.mask.apply(&j);
        let b_vec = desired_acc - dj * &robot.state.qd;
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&b_vec)));
        Ok(())
    }
}

/// A per-joint torque target with PD terms.
#[derive(Clone, Copy, Debug)]
pub struct TargetTau {
    /// Feedforward torque.
    pub torque: f64,
    /// Position gain.
    pub kp: f64,
    /// Velocity gain.
    pub kd: f64,
}

/// A torque task: constrains joint torques (`tau_task`) (PlaCo dynamics
/// `TorqueTask`).
pub struct TorqueTask {
    base: TaskBase,
    /// Per-joint `(name, target)` torque entries.
    pub torques: Vec<(String, TargetTau)>,
}

impl TorqueTask {
    pub(crate) fn new() -> Self {
        Self {
            base: TaskBase {
                tau_task: true,
                ..TaskBase::default()
            },
            torques: Vec::new(),
        }
    }

    /// Sets a joint torque target with PD gains.
    pub fn set_torque(&mut self, joint: impl Into<String>, torque: f64, kp: f64, kd: f64) {
        let joint = joint.into();
        let target = TargetTau { torque, kp, kd };
        if let Some(e) = self.torques.iter_mut().find(|(n, _)| *n == joint) {
            e.1 = target;
        } else {
            self.torques.push((joint, target));
        }
    }
}

impl DynamicsTask for TorqueTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "torques"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        let n = robot.nv();
        let mut a = DMatrix::zeros(self.torques.len(), n);
        let mut b = DVector::zeros(self.torques.len());
        for (k, (name, target)) in self.torques.iter().enumerate() {
            let vq = robot.joint_v_offset(name)?;
            a[(k, vq)] = 1.0;
            b[k] = target.torque + target.kp * robot.joint(name)?
                - target.kd * robot.joint_velocity(name)?;
        }
        self.base.a = a;
        self.base.b = b;
        Ok(())
    }
}

//! Core dynamics tasks (PlaCo `dynamics::PositionTask`, `JointsTask`).

use nalgebra::{DMatrix, DVector, Vector3};

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

/// PD acceleration task on a frame's position (PlaCo dynamics `PositionTask`).
pub struct PositionTask {
    base: TaskBase,
    /// Target frame.
    pub frame_index: usize,
    /// Target position in the world.
    pub target_world: Vector3<f64>,
    /// Target velocity in the world.
    pub dtarget_world: Vector3<f64>,
    /// Target acceleration in the world.
    pub ddtarget_world: Vector3<f64>,
}

impl PositionTask {
    pub(crate) fn new(frame_index: usize, target_world: Vector3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            frame_index,
            target_world,
            dtarget_world: Vector3::zeros(),
            ddtarget_world: Vector3::zeros(),
        }
    }
}

impl DynamicsTask for PositionTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "position"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        let j = robot.frame_jacobian(self.frame_index, ReferenceFrame::LocalWorldAligned)?;
        let dj = robot
            .frame_jacobian_time_variation(self.frame_index, ReferenceFrame::LocalWorldAligned)?;
        let j_pos = j.rows(0, 3).into_owned();
        let dj_pos = dj.rows(0, 3).into_owned();

        let t = robot.t_world_frame(self.frame_index)?;
        let position_error = self.target_world - t.translation.vector;
        let velocity_world = &j_pos * &robot.state.qd;
        let velocity_error = self.dtarget_world - velocity_world;
        let kd = self.base.get_kd();
        let desired_acc = self.base.kp * position_error + kd * velocity_error + self.ddtarget_world;

        self.base.a = self.base.mask.apply(&j_pos);
        let b_vec = desired_acc - dj_pos * &robot.state.qd;
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&b_vec)));
        Ok(())
    }
}

/// PD acceleration task driving named joints to targets (PlaCo dynamics
/// `JointsTask`).
pub struct JointsTask {
    base: TaskBase,
    /// Target `(joint name, position, velocity, acceleration)` entries.
    pub joints: Vec<(String, f64, f64, f64)>,
}

impl JointsTask {
    pub(crate) fn new() -> Self {
        Self {
            base: TaskBase::default(),
            joints: Vec::new(),
        }
    }

    /// Sets a joint position target (velocity/acceleration targets zero).
    pub fn set_joint(&mut self, name: impl Into<String>, target: f64) {
        let name = name.into();
        if let Some(e) = self.joints.iter_mut().find(|(n, ..)| *n == name) {
            e.1 = target;
        } else {
            self.joints.push((name, target, 0.0, 0.0));
        }
    }
}

impl DynamicsTask for JointsTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "joints"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        let n = robot.nv();
        let mut a = DMatrix::zeros(self.joints.len(), n);
        let mut b = DVector::zeros(self.joints.len());
        let kd = self.base.get_kd();
        for (k, (name, target, dtarget, ddtarget)) in self.joints.iter().enumerate() {
            let vq = robot.joint_v_offset(name)?;
            a[(k, vq)] = 1.0;
            let position_error = target - robot.joint(name)?;
            let velocity_error = dtarget - robot.state.qd[vq];
            b[k] = self.base.kp * position_error + kd * velocity_error + ddtarget;
        }
        self.base.a = a;
        self.base.b = b;
        Ok(())
    }
}

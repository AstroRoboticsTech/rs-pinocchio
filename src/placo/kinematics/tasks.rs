//! Core kinematics tasks (PlaCo `kinematics::*Task`).

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

use super::task::{KinematicsTask, TaskBase};
use crate::error::Result;
use crate::placo::model::RobotWrapper;
use crate::ReferenceFrame;

fn vec3_to_col(v: &Vector3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 1, v.as_slice())
}

fn col_to_vector(m: DMatrix<f64>) -> DVector<f64> {
    m.column(0).into_owned()
}

/// Drives a frame's position to a world target (PlaCo `PositionTask`).
pub struct PositionTask {
    base: TaskBase,
    /// Target frame index.
    pub frame_index: usize,
    /// Target position in the world.
    pub target_world: Vector3<f64>,
}

impl PositionTask {
    pub(crate) fn new(frame_index: usize, target_world: Vector3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            frame_index,
            target_world,
        }
    }
}

impl KinematicsTask for PositionTask {
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
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let t_world_frame = robot.t_world_frame(self.frame_index)?;
        let r = t_world_frame.rotation.to_rotation_matrix().into_inner();
        self.base.mask.r_local_world = r.transpose();
        let error = self.target_world - t_world_frame.translation.vector;
        let j = robot.frame_jacobian(self.frame_index, ReferenceFrame::LocalWorldAligned)?;
        let j_pos = j.rows(0, 3).into_owned();
        self.base.a = self.base.mask.apply(&j_pos);
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&error)));
        Ok(())
    }
}

/// Drives a frame's orientation to a world target (PlaCo `OrientationTask`).
pub struct OrientationTask {
    base: TaskBase,
    /// Target frame index.
    pub frame_index: usize,
    /// Target orientation `R_world_frame`.
    pub r_world_frame: Matrix3<f64>,
}

impl OrientationTask {
    pub(crate) fn new(frame_index: usize, r_world_frame: Matrix3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            frame_index,
            r_world_frame,
        }
    }
}

impl KinematicsTask for OrientationTask {
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
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let t_world_frame = robot.t_world_frame(self.frame_index)?;
        let r_current = t_world_frame.rotation.to_rotation_matrix().into_inner();
        // error = log3(R_target · R_current⁻¹)
        let m = self.r_world_frame * r_current.transpose();
        let error = Rotation3::from_matrix_unchecked(m).scaled_axis();
        let j = robot.frame_jacobian(self.frame_index, ReferenceFrame::World)?;
        let j_rot = j.rows(3, 3).into_owned();
        self.base.mask.r_local_world = self.r_world_frame.transpose();
        self.base.a = self.base.mask.apply(&j_rot);
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&error)));
        Ok(())
    }
}

/// Drives the center of mass to a world target (PlaCo `CoMTask`).
pub struct CoMTask {
    base: TaskBase,
    /// Target CoM position in the world.
    pub target_world: Vector3<f64>,
}

impl CoMTask {
    pub(crate) fn new(target_world: Vector3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            target_world,
        }
    }
}

impl KinematicsTask for CoMTask {
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
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let jac = robot.com_jacobian()?;
        let error = self.target_world - robot.com_world()?;
        self.base.a = self.base.mask.apply(&jac);
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&error)));
        Ok(())
    }
}

/// Drives a set of named joints to target values (PlaCo `JointsTask`).
pub struct JointsTask {
    base: TaskBase,
    /// Target `(joint name, value)` pairs.
    pub joints: Vec<(String, f64)>,
}

impl JointsTask {
    pub(crate) fn new() -> Self {
        Self {
            base: TaskBase::default(),
            joints: Vec::new(),
        }
    }

    /// Sets a joint target (adds it if absent).
    pub fn set_joint(&mut self, name: impl Into<String>, target: f64) {
        let name = name.into();
        if let Some(entry) = self.joints.iter_mut().find(|(n, _)| *n == name) {
            entry.1 = target;
        } else {
            self.joints.push((name, target));
        }
    }
}

impl KinematicsTask for JointsTask {
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
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let n = robot.nv();
        let mut a = DMatrix::zeros(self.joints.len(), n);
        let mut b = DVector::zeros(self.joints.len());
        for (k, (name, target)) in self.joints.iter().enumerate() {
            a[(k, robot.joint_v_offset(name)?)] = 1.0;
            b[k] = target - robot.joint(name)?;
        }
        self.base.a = a;
        self.base.b = b;
        Ok(())
    }
}

/// Regularizes joint velocities towards zero (PlaCo `RegularizationTask`).
///
/// The floating-base rows are excluded, matching PlaCo.
pub struct RegularizationTask {
    base: TaskBase,
    /// Default per-DoF weight (square-rooted internally).
    pub magnitude: f64,
    /// Per-joint weight overrides (joint name → weight).
    joint_weights: Vec<(String, f64)>,
}

impl RegularizationTask {
    pub(crate) fn new(magnitude: f64) -> Self {
        Self {
            base: TaskBase::default(),
            magnitude,
            joint_weights: Vec::new(),
        }
    }

    /// Overrides the regularization weight for a single joint (PlaCo
    /// `set_joint_weight`); the last value for a joint wins.
    pub fn set_joint_weight(&mut self, joint: impl Into<String>, weight: f64) {
        let joint = joint.into();
        self.joint_weights.retain(|(j, _)| *j != joint);
        self.joint_weights.push((joint, weight));
    }
}

impl KinematicsTask for RegularizationTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "regularization"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let n = robot.nv();
        // sqrt so the weight squares back out in the QP objective.
        let w = self.magnitude.sqrt();
        let mut a = DMatrix::zeros(n - 6, n);
        for i in 0..(n - 6) {
            a[(i, 6 + i)] = w;
        }
        // Per-joint overrides (rows/cols offset by the floating base's 6 DoFs).
        for (joint, weight) in &self.joint_weights {
            let off = robot.joint_v_offset(joint)?;
            let size = robot.joint_v_size(joint)?;
            for i in 0..size {
                if off + i >= 6 {
                    a[(off + i - 6, off + i)] = weight.sqrt();
                }
            }
        }
        self.base.a = a;
        self.base.b = DVector::zeros(n - 6);
        Ok(())
    }
}

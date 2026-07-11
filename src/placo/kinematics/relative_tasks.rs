//! Relative-frame, axis-align, distance and centroidal-momentum kinematics
//! tasks (PlaCo `Relative*Task`, `AxisAlignTask`, `DistanceTask`,
//! `CentroidalMomentumTask`).

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

use super::task::{KinematicsTask, TaskBase};
use crate::error::{Error, Result};
use crate::placo::model::RobotWrapper;
use crate::placo::tools::safe_acos;
use crate::ReferenceFrame;

fn vec3_to_col(v: &Vector3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 1, v.as_slice())
}

fn col_to_vector(m: DMatrix<f64>) -> DVector<f64> {
    m.column(0).into_owned()
}

fn dmat3(m: &Matrix3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 3, m.as_slice())
}

/// Drives the position of frame `b` (expressed in `a`) to a target (PlaCo
/// `RelativePositionTask`).
pub struct RelativePositionTask {
    base: TaskBase,
    /// Reference frame `a`.
    pub frame_a: usize,
    /// Target frame `b`.
    pub frame_b: usize,
    /// Target position of `b` expressed in `a`.
    pub target: Vector3<f64>,
}

impl RelativePositionTask {
    pub(crate) fn new(frame_a: usize, frame_b: usize, target: Vector3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            frame_a,
            frame_b,
            target,
        }
    }
}

impl KinematicsTask for RelativePositionTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "relative_position"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let t_a_b = robot.t_a_b(self.frame_a, self.frame_b)?;
        let error = self.target - t_a_b.translation.vector;
        let j = robot.relative_position_jacobian(self.frame_a, self.frame_b)?;
        self.base.a = self.base.mask.apply(&j);
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&error)));
        Ok(())
    }
}

/// Drives the relative orientation `R_a_b` to a target (PlaCo
/// `RelativeOrientationTask`).
pub struct RelativeOrientationTask {
    base: TaskBase,
    /// Reference frame `a`.
    pub frame_a: usize,
    /// Target frame `b`.
    pub frame_b: usize,
    /// Target relative orientation `R_a_b`.
    pub r_a_b: Matrix3<f64>,
}

impl RelativeOrientationTask {
    pub(crate) fn new(frame_a: usize, frame_b: usize, r_a_b: Matrix3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            frame_a,
            frame_b,
            r_a_b,
        }
    }
}

impl KinematicsTask for RelativeOrientationTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "relative_orientation"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let t_world_a = robot.t_world_frame(self.frame_a)?;
        let t_a_b = t_world_a.inverse() * robot.t_world_frame(self.frame_b)?;
        let r_a_b_current = t_a_b.rotation.to_rotation_matrix().into_inner();
        let error =
            Rotation3::from_matrix_unchecked(self.r_a_b * r_a_b_current.transpose()).scaled_axis();

        let r_world_a = t_world_a.rotation.to_rotation_matrix().into_inner();
        let ja = robot.frame_jacobian(self.frame_a, ReferenceFrame::World)?;
        let jb = robot.frame_jacobian(self.frame_b, ReferenceFrame::World)?;
        let diff_rot = (jb.rows(3, 3).into_owned()) - (ja.rows(3, 3).into_owned());
        let j_ab = dmat3(&r_world_a.transpose()) * diff_rot;

        self.base.a = self.base.mask.apply(&j_ab);
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&error)));
        Ok(())
    }
}

/// Aligns a frame axis with a target world axis (PlaCo `AxisAlignTask`).
///
/// Rotation about the aligned axis is left free.
pub struct AxisAlignTask {
    base: TaskBase,
    /// Frame index.
    pub frame_index: usize,
    /// The axis, expressed in the frame.
    pub axis_frame: Vector3<f64>,
    /// The target axis, expressed in the world.
    pub target_axis_world: Vector3<f64>,
}

impl AxisAlignTask {
    pub(crate) fn new(
        frame_index: usize,
        axis_frame: Vector3<f64>,
        target_axis_world: Vector3<f64>,
    ) -> Self {
        Self {
            base: TaskBase::default(),
            frame_index,
            axis_frame,
            target_axis_world,
        }
    }
}

impl KinematicsTask for AxisAlignTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "axis_align"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let t_world_frame = robot.t_world_frame(self.frame_index)?;
        let r_world = t_world_frame.rotation.to_rotation_matrix().into_inner();
        let target = self.target_axis_world.normalize();

        // Build an "axis frame": x on the current axis, z the rotation axis to
        // correct the error, y the remaining axis.
        let x = (r_world * self.axis_frame).normalize();
        let z = x.cross(&target).normalize();
        let y = z.cross(&x);
        let r_world_axisframe = Matrix3::from_columns(&[x, y, z]);

        let error_angle = safe_acos(x.dot(&target));

        let j = robot.frame_jacobian(self.frame_index, ReferenceFrame::World)?;
        let j_rot = j.rows(3, 3).into_owned();
        let j_axisframe = dmat3(&r_world_axisframe.transpose()) * j_rot;

        // Keep only y and z rows (x rotation is free).
        self.base.a = j_axisframe.rows(1, 2).into_owned();
        self.base.b = DVector::from_vec(vec![0.0, error_angle]);
        Ok(())
    }
}

/// Drives the distance between two frames to a target (PlaCo `DistanceTask`).
pub struct DistanceTask {
    base: TaskBase,
    /// Frame `a`.
    pub frame_a: usize,
    /// Frame `b`.
    pub frame_b: usize,
    /// Target distance.
    pub distance: f64,
}

impl DistanceTask {
    pub(crate) fn new(frame_a: usize, frame_b: usize, distance: f64) -> Self {
        Self {
            base: TaskBase::default(),
            frame_a,
            frame_b,
            distance,
        }
    }
}

impl KinematicsTask for DistanceTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "distance"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let ta = robot.t_world_frame(self.frame_a)?;
        let tb = robot.t_world_frame(self.frame_b)?;
        let ab = tb.translation.vector - ta.translation.vector;
        let error = self.distance - ab.norm();
        let direction = ab.normalize();

        let ja = robot.frame_jacobian(self.frame_a, ReferenceFrame::LocalWorldAligned)?;
        let jb = robot.frame_jacobian(self.frame_b, ReferenceFrame::LocalWorldAligned)?;
        let diff_pos = (jb.rows(0, 3).into_owned()) - (ja.rows(0, 3).into_owned());
        let dir_row = DMatrix::from_row_slice(1, 3, direction.as_slice());

        self.base.a = dir_row * diff_pos;
        self.base.b = DVector::from_element(1, error);
        Ok(())
    }
}

/// Drives the angular centroidal momentum to a target (PlaCo
/// `CentroidalMomentumTask`). Requires `solver.dt`.
pub struct CentroidalMomentumTask {
    base: TaskBase,
    /// Target angular momentum in the world.
    pub l_world: Vector3<f64>,
}

impl CentroidalMomentumTask {
    pub(crate) fn new(l_world: Vector3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            l_world,
        }
    }
}

impl KinematicsTask for CentroidalMomentumTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "centroidal_momentum"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, dt: f64) -> Result<()> {
        if dt == 0.0 {
            return Err(Error::Solver(
                "CentroidalMomentumTask requires solver.dt to be set".into(),
            ));
        }
        let ag = robot.centroidal_map()?;
        let ag_angular = ag.rows(3, 3).into_owned();
        self.base.a = self.base.mask.apply(&ag_angular) / dt;
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&self.l_world)));
        Ok(())
    }
}

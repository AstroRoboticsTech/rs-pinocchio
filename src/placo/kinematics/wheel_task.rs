//! Rolling-without-slipping wheel kinematics task (PlaCo `kinematics::WheelTask`).
//!
//! Constrains the contact point of a wheel (a revolute joint spinning about its
//! local z-axis) to roll on a planar surface without slipping. The instantaneous
//! velocity of the ground-contact point is driven to zero (except its height,
//! which is driven onto the surface), which is the rolling constraint. An
//! *omniwheel* additionally frees the lateral direction.

use nalgebra::{DMatrix, DVector, Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};

use super::task::{KinematicsTask, TaskBase};
use crate::error::Result;
use crate::placo::model::RobotWrapper;
use crate::ReferenceFrame;

fn dmat3(m: &Matrix3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 3, m.as_slice())
}

fn skew(v: &Vector3<f64>) -> DMatrix<f64> {
    DMatrix::from_row_slice(3, 3, &[0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0])
}

/// Keeps a wheel joint rolling without slipping on a surface (PlaCo `WheelTask`).
pub struct WheelTask {
    base: TaskBase,
    /// The wheel joint name (spins about its local z-axis).
    pub joint: String,
    /// Wheel radius [m].
    pub radius: f64,
    /// Whether the wheel is an omniwheel (free lateral sliding).
    pub omniwheel: bool,
    /// Pose of the rolling surface in the world (its z-axis is the up normal).
    pub t_world_surface: Isometry3<f64>,
}

impl WheelTask {
    pub(crate) fn new(joint: impl Into<String>, radius: f64, omniwheel: bool) -> Self {
        Self {
            base: TaskBase::default(),
            joint: joint.into(),
            radius,
            omniwheel,
            t_world_surface: Isometry3::identity(),
        }
    }
}

impl KinematicsTask for WheelTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "wheel"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let jid = robot.joint_id(&self.joint)?;
        let t_world_wheel = robot.t_world_joint(jid)?;
        let t_surface_wheel = self.t_world_surface.inverse() * t_world_wheel;

        // Build the contact frame in the surface frame: its z is the surface
        // up-normal, its x is the wheel's rolling direction (perpendicular to
        // both the wheel spin axis and up), and it sits `radius` below the wheel.
        let r_sw = t_surface_wheel.rotation.to_rotation_matrix().into_inner();
        let wheel_z: Vector3<f64> = r_sw.column(2).into();
        let up = Vector3::z();
        let x = wheel_z.cross(&up).normalize();
        let z = up;
        let y = z.cross(&x);
        let r_sc = Matrix3::from_columns(&[x, y, z]);
        // Step `radius` toward the surface along the in-plane steepest-descent
        // direction (the wheel rim's lowest point), matching PlaCo: project
        // -surface_z (expressed in the wheel frame, i.e. -R_swᵀ row) onto the
        // wheel disk plane, then map back to the surface frame. For an upright
        // wheel this reduces to straight down (-up).
        let mut down_axis_wheel = -r_sw.row(2).transpose();
        down_axis_wheel[2] = 0.0;
        let down_axis_surface = r_sw * down_axis_wheel.normalize();
        let origin = t_surface_wheel.translation.vector + self.radius * down_axis_surface;
        let t_surface_contact = Isometry3::from_parts(
            Translation3::from(origin),
            UnitQuaternion::from_matrix(&r_sc),
        );
        let t_contact_wheel = t_surface_contact.inverse() * t_surface_wheel;

        // Linear velocity of the contact point (in the contact frame) from the
        // local joint Jacobian, via the SE3 action matrix top rows:
        //   v = R·J_lin + skew(p)·R·J_ang.
        let r = dmat3(&t_contact_wheel.rotation.to_rotation_matrix().into_inner());
        let p = t_contact_wheel.translation.vector;
        let j_local = robot.joint_jacobian(jid, ReferenceFrame::Local)?;
        let j_lin = j_local.rows(0, 3).into_owned();
        let j_ang = j_local.rows(3, 3).into_owned();
        let a_full = &r * j_lin + skew(&p) * &r * j_ang; // 3 x nv

        let z_contact = t_surface_contact.translation.z;
        if self.omniwheel {
            // Free the lateral (y) direction: keep the rolling (x) and vertical
            // (z) rows only.
            let mut a = DMatrix::zeros(2, robot.nv());
            a.row_mut(0).copy_from(&a_full.row(0));
            a.row_mut(1).copy_from(&a_full.row(2));
            self.base.a = a;
            self.base.b = DVector::from_vec(vec![0.0, -z_contact]);
        } else {
            self.base.a = a_full;
            self.base.b = DVector::from_vec(vec![0.0, 0.0, -z_contact]);
        }
        Ok(())
    }
}

//! Relative-frame dynamics tasks (PlaCo `dynamics::RelativePositionTask`,
//! `RelativeOrientationTask`).
//!
//! The relative-orientation task uses the SO(3) log Jacobian `Jlog3` to map the
//! world-frame angular error to a generalized-acceleration residual.

use nalgebra::{DMatrix, DVector, Vector3};

use super::task::{DynamicsTask, TaskBase};
use crate::error::Result;
use crate::placo::model::RobotWrapper;
use crate::ReferenceFrame;

fn skew(v: &Vector3<f64>) -> DMatrix<f64> {
    DMatrix::from_row_slice(3, 3, &[0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0])
}

fn dmat3(m: &nalgebra::Matrix3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 3, m.as_slice())
}

fn vec3_to_col(v: &Vector3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 1, v.as_slice())
}

fn col_to_vector(m: DMatrix<f64>) -> DVector<f64> {
    m.column(0).into_owned()
}

/// PD acceleration task on the position of `frame_b` expressed in `frame_a`
/// (PlaCo dynamics `RelativePositionTask`).
pub struct RelativePositionTask {
    base: TaskBase,
    /// Reference frame `a`.
    pub frame_a: usize,
    /// Target frame `b`.
    pub frame_b: usize,
    /// Target position of `b` in `a`.
    pub target: Vector3<f64>,
    /// Target velocity.
    pub dtarget: Vector3<f64>,
    /// Target acceleration.
    pub ddtarget: Vector3<f64>,
}

impl RelativePositionTask {
    pub(crate) fn new(frame_a: usize, frame_b: usize, target: Vector3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            frame_a,
            frame_b,
            target,
            dtarget: Vector3::zeros(),
            ddtarget: Vector3::zeros(),
        }
    }
}

impl DynamicsTask for RelativePositionTask {
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
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        let w_t_a = robot.t_world_frame(self.frame_a)?;
        let w_t_b = robot.t_world_frame(self.frame_b)?;
        let r_world_a = w_t_a.rotation.to_rotation_matrix().into_inner();
        let a_r_w = dmat3(&r_world_a.transpose());

        let w_ab = w_t_b.translation.vector - w_t_a.translation.vector;
        let a_ab = &a_r_w * w_ab;
        let a_ab_v = Vector3::new(a_ab[0], a_ab[1], a_ab[2]);

        let ja = robot.frame_jacobian(self.frame_a, ReferenceFrame::LocalWorldAligned)?;
        let dja =
            robot.frame_jacobian_time_variation(self.frame_a, ReferenceFrame::LocalWorldAligned)?;
        let jb = robot.frame_jacobian(self.frame_b, ReferenceFrame::LocalWorldAligned)?;
        let djb =
            robot.frame_jacobian_time_variation(self.frame_b, ReferenceFrame::LocalWorldAligned)?;

        let ja_pos = ja.rows(0, 3).into_owned();
        let ja_rot = ja.rows(3, 3).into_owned();
        let jb_pos = jb.rows(0, 3).into_owned();
        let dja_rot = dja.rows(3, 3).into_owned();
        let djb_rot = djb.rows(3, 3).into_owned();
        let qd = &robot.state.qd;

        let w_omega_a = &ja_rot * qd;
        let a_omega_w = -(&a_r_w * &w_omega_a);
        let a_omega_w_v = Vector3::new(a_omega_w[0], a_omega_w[1], a_omega_w[2]);
        let w_dab = (&jb_pos - &ja_pos) * qd;
        let a_dab = a_omega_w_v.cross(&a_ab_v) + &a_r_w * &w_dab;

        let position_error = self.target - a_ab_v;
        let velocity_error = self.dtarget - Vector3::new(a_dab[0], a_dab[1], a_dab[2]);
        let kd = self.base.get_kd();
        let desired_acc = self.base.kp * position_error + kd * velocity_error + self.ddtarget;

        // J = skew(a_AB)·a_R_w·Ja_rot + a_R_w·(Jb_pos − Ja_pos)
        let j = skew(&a_ab_v) * &a_r_w * &ja_rot + &a_r_w * (&jb_pos - &ja_pos);

        // Coriolis term e.
        let mut e = 2.0 * skew(&a_omega_w_v) * &a_r_w * &w_dab;
        e += 2.0 * skew(&a_omega_w_v) * skew(&a_omega_w_v) * &a_ab;
        e += &a_r_w * (&djb_rot - &dja_rot) * qd;
        e += skew(&a_ab_v) * &a_r_w * (&dja_rot * qd);

        self.base.a = self.base.mask.apply(&j);
        let b_vec = -Vector3::new(e[0], e[1], e[2]) + desired_acc;
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&b_vec)));
        Ok(())
    }
}

/// SO(3) log Jacobian `Jlog3(v)` for `v = log3(R)` (Pinocchio's formula).
fn jlog3(log: &Vector3<f64>) -> nalgebra::Matrix3<f64> {
    let t2 = log.norm_squared();
    let t = t2.sqrt();
    let (alpha, diag) = if t < 1e-8 {
        (1.0 / 12.0 + t2 / 720.0, 1.0 - t2 / 12.0)
    } else {
        let (st, ct) = (t.sin(), t.cos());
        let st_1mct = st / (1.0 - ct);
        (1.0 / t2 - st_1mct / (2.0 * t), 0.5 * t * st_1mct)
    };
    let vvt = log * log.transpose();
    #[rustfmt::skip]
    let skew = nalgebra::Matrix3::new(
        0.0, -log.z, log.y,
        log.z, 0.0, -log.x,
        -log.y, log.x, 0.0,
    );
    alpha * vvt + diag * nalgebra::Matrix3::identity() + 0.5 * skew
}

/// PD acceleration task on the relative orientation `R_a_b` (PlaCo dynamics
/// `RelativeOrientationTask`).
pub struct RelativeOrientationTask {
    base: TaskBase,
    /// Reference frame `a`.
    pub frame_a: usize,
    /// Target frame `b`.
    pub frame_b: usize,
    /// Target relative orientation `R_a_b`.
    pub r_a_b: nalgebra::Matrix3<f64>,
    /// Target relative angular velocity (in `a`).
    pub omega_a_b: Vector3<f64>,
    /// Target relative angular acceleration (in `a`).
    pub domega_a_b: Vector3<f64>,
}

impl RelativeOrientationTask {
    pub(crate) fn new(frame_a: usize, frame_b: usize, r_a_b: nalgebra::Matrix3<f64>) -> Self {
        Self {
            base: TaskBase::default(),
            frame_a,
            frame_b,
            r_a_b,
            omega_a_b: Vector3::zeros(),
            domega_a_b: Vector3::zeros(),
        }
    }
}

impl DynamicsTask for RelativeOrientationTask {
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
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        use crate::ReferenceFrame::World;
        let ja = robot.frame_jacobian(self.frame_a, World)?;
        let dja = robot.frame_jacobian_time_variation(self.frame_a, World)?;
        let jb = robot.frame_jacobian(self.frame_b, World)?;
        let djb = robot.frame_jacobian_time_variation(self.frame_b, World)?;
        let ja_rot = ja.rows(3, 3).into_owned();
        let jb_rot = jb.rows(3, 3).into_owned();
        let dja_rot = dja.rows(3, 3).into_owned();
        let djb_rot = djb.rows(3, 3).into_owned();
        let qd = &robot.state.qd;

        let r_world_a = robot
            .t_world_frame(self.frame_a)?
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let r_world_b = robot
            .t_world_frame(self.frame_b)?
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let r_a_b_real = r_world_a.transpose() * r_world_b;
        let m = self.r_a_b * r_a_b_real.transpose();
        let log = nalgebra::Rotation3::from_matrix_unchecked(m).scaled_axis();
        let error_world = r_world_a * log;

        let omega_a = &ja_rot * qd;
        let omega_b = &jb_rot * qd;
        let omega_ab_real = &omega_b - &omega_a;
        let vel_error = r_world_a * self.omega_a_b
            - Vector3::new(omega_ab_real[0], omega_ab_real[1], omega_ab_real[2]);
        let kd = self.base.get_kd();
        let desired_acc = self.base.kp * error_world + kd * vel_error + self.domega_a_b;

        let jlog = dmat3(&jlog3(&log));
        let a = self.base.mask.apply(&(&jlog * (&jb_rot - &ja_rot)));
        let coriolis = &jlog * ((&djb_rot * qd) - (&dja_rot * qd));
        let b_vec = desired_acc - Vector3::new(coriolis[0], coriolis[1], coriolis[2]);
        self.base.a = a;
        self.base.b = col_to_vector(self.base.mask.apply(&vec3_to_col(&b_vec)));
        Ok(())
    }
}

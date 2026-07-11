//! The robot model, its state, and rigid-body queries (PlaCo `RobotWrapper`).
//!
//! Wraps the Pinocchio binding. Following PlaCo, a **free-flyer floating base is
//! always prepended**, so `q` (length [`RobotWrapper::nq`]) is longer than `qd`
//! / `qdd` (length [`RobotWrapper::nv`]): the base contributes 7 to `nq` (x y z +
//! unit quaternion) and 6 to `nv`.

use std::path::Path;

use cxx::UniquePtr;
use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix3, Quaternion, Translation3, UnitQuaternion, Vector3,
};

use crate::error::{Error, Result};
use crate::ffi::bridge as ffi;
use crate::ReferenceFrame;

/// The robot state: configuration `q`, velocity `qd`, acceleration `qdd`.
///
/// `q` has length `nq`; `qd` and `qdd` have length `nv` (see [`RobotWrapper`]).
#[derive(Clone, Debug)]
pub struct State {
    /// Configuration `q` (length `nq`).
    pub q: DVector<f64>,
    /// Velocity `qd` (length `nv`).
    pub qd: DVector<f64>,
    /// Acceleration `qdd` (length `nv`).
    pub qdd: DVector<f64>,
}

/// A robot model bundled with its working data and [`State`].
///
/// Call [`RobotWrapper::update_kinematics`] after changing the state before
/// reading frame placements or Jacobians.
pub struct RobotWrapper {
    inner: UniquePtr<ffi::PinocchioModel>,
    /// The current robot state.
    pub state: State,
    nq: usize,
    nv: usize,
}

impl RobotWrapper {
    /// Loads a robot from a URDF file, prepending a free-flyer floating base.
    pub fn from_urdf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| Error::UrdfLoad(format!("non-UTF-8 path: {}", path.display())))?;
        let inner =
            ffi::from_urdf(path_str, true).map_err(|e| Error::UrdfLoad(e.what().to_string()))?;
        let nq = inner.nq();
        let nv = inner.nv();
        let mut wrapper = Self {
            inner,
            state: State {
                q: DVector::zeros(nq),
                qd: DVector::zeros(nv),
                qdd: DVector::zeros(nv),
            },
            nq,
            nv,
        };
        wrapper.reset();
        Ok(wrapper)
    }

    /// Configuration-space dimension (`nq`, includes the 7-DoF floating base).
    pub fn nq(&self) -> usize {
        self.nq
    }

    /// Velocity / tangent-space dimension (`nv`, includes the 6-DoF base).
    pub fn nv(&self) -> usize {
        self.nv
    }

    /// The neutral state: neutral configuration, zero velocity/acceleration.
    pub fn neutral_state(&self) -> State {
        let mut q = DVector::zeros(self.nq);
        self.inner.neutral(q.as_mut_slice());
        State {
            q,
            qd: DVector::zeros(self.nv),
            qdd: DVector::zeros(self.nv),
        }
    }

    /// Resets the state to neutral.
    pub fn reset(&mut self) {
        self.state = self.neutral_state();
    }

    /// Refreshes frame placements and Jacobians from the current state. Call
    /// after modifying [`RobotWrapper::state`] and before reading placements or
    /// Jacobians.
    pub fn update_kinematics(&mut self) -> Result<()> {
        self.inner
            .pin_mut()
            .update_kinematics(self.state.q.as_slice(), self.state.qd.as_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))
    }

    /// Frame index for `name`, or `None` if absent.
    pub fn frame_index(&self, name: &str) -> Option<usize> {
        let id = self.inner.frame_id(name);
        (id >= 0).then_some(id as usize)
    }

    fn frame_index_checked(&self, name: &str) -> Result<usize> {
        self.frame_index(name)
            .ok_or_else(|| Error::FrameNotFound(name.to_string()))
    }

    /// World placement of frame `id` (needs a prior [`update_kinematics`]).
    pub fn t_world_frame(&self, id: usize) -> Result<Isometry3<f64>> {
        let p = self
            .inner
            .frame_placement(id as i64)
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(frame_placement_to_isometry(&p))
    }

    /// World placement of the named frame.
    pub fn t_world_frame_by_name(&self, name: &str) -> Result<Isometry3<f64>> {
        self.t_world_frame(self.frame_index_checked(name)?)
    }

    /// Transform from frame `b` to frame `a` (`T_a_b = T_world_a⁻¹ · T_world_b`).
    pub fn t_a_b(&self, a: usize, b: usize) -> Result<Isometry3<f64>> {
        let ta = self.t_world_frame(a)?;
        let tb = self.t_world_frame(b)?;
        Ok(ta.inverse() * tb)
    }

    /// World placement of the floating base (root) frame.
    pub fn t_world_fbase(&self) -> Isometry3<f64> {
        let q = &self.state.q;
        let trans = Translation3::new(q[0], q[1], q[2]);
        // Free-flyer stores the quaternion as (qx, qy, qz, qw).
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(q[6], q[3], q[4], q[5]));
        let se3_q = Isometry3::from_parts(trans, quat);
        frame_placement_to_isometry(&self.inner.root_joint_placement()) * se3_q
    }

    /// Sets the floating base so its world placement is `t`.
    pub fn set_t_world_fbase(&mut self, t: Isometry3<f64>) {
        let root = frame_placement_to_isometry(&self.inner.root_joint_placement());
        let tt = root.inverse() * t;
        self.state.q[0] = tt.translation.x;
        self.state.q[1] = tt.translation.y;
        self.state.q[2] = tt.translation.z;
        let q = tt.rotation.quaternion();
        self.state.q[3] = q.i;
        self.state.q[4] = q.j;
        self.state.q[5] = q.k;
        self.state.q[6] = q.w;
    }

    /// Moves the floating base so that frame `frame` has world placement
    /// `t_target`. Requires a prior [`update_kinematics`].
    pub fn set_t_world_frame(&mut self, frame: usize, t_target: Isometry3<f64>) -> Result<()> {
        let t_world_fbase = self.t_world_fbase();
        let t_world_frame = self.t_world_frame(frame)?;
        let t_frame_fbase = t_world_frame.inverse() * t_world_fbase;
        self.set_t_world_fbase(t_target * t_frame_fbase);
        Ok(())
    }

    /// The `6 × nv` frame Jacobian in `reference` (needs [`update_kinematics`]).
    pub fn frame_jacobian(&self, id: usize, reference: ReferenceFrame) -> Result<DMatrix<f64>> {
        let mut out = vec![0.0; 6 * self.nv];
        self.inner
            .frame_jacobian(id as i64, reference as u8, out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DMatrix::from_row_slice(6, self.nv, &out))
    }

    /// The `6 × nv` frame Jacobian time variation `J̇` in `reference`.
    pub fn frame_jacobian_time_variation(
        &self,
        id: usize,
        reference: ReferenceFrame,
    ) -> Result<DMatrix<f64>> {
        let mut out = vec![0.0; 6 * self.nv];
        self.inner
            .frame_jacobian_time_variation(id as i64, reference as u8, out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DMatrix::from_row_slice(6, self.nv, &out))
    }

    /// The `3 × nv` Jacobian of frame `b`'s position expressed in frame `a`.
    ///
    /// Needs a prior [`update_kinematics`].
    pub fn relative_position_jacobian(&self, a: usize, b: usize) -> Result<DMatrix<f64>> {
        let t_world_a = self.t_world_frame(a)?;
        let t_a_b = t_world_a.inverse() * self.t_world_frame(b)?;
        let r_world_a = dmat3(&t_world_a.rotation.to_rotation_matrix().into_inner());
        let ja = self.frame_jacobian(a, ReferenceFrame::LocalWorldAligned)?;
        let jb = self.frame_jacobian(b, ReferenceFrame::LocalWorldAligned)?;
        let ja_pos = ja.rows(0, 3).into_owned();
        let ja_rot = ja.rows(3, 3).into_owned();
        let jb_pos = jb.rows(0, 3).into_owned();
        let rt = r_world_a.transpose();
        Ok(&rt * (jb_pos - ja_pos) + skew(&t_a_b.translation.vector) * &rt * ja_rot)
    }

    /// The `3 × nv` CoM Jacobian (in the world frame).
    pub fn com_jacobian(&mut self) -> Result<DMatrix<f64>> {
        let mut out = vec![0.0; 3 * self.nv];
        self.inner
            .pin_mut()
            .com_jacobian(self.state.q.as_slice(), out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DMatrix::from_row_slice(3, self.nv, &out))
    }

    /// Computes the joint kinematic hessians (needed before [`Self::frame_hessian`]).
    pub fn compute_hessians(&mut self) {
        self.inner.pin_mut().compute_hessians();
    }

    /// The `6 × nv` frame hessian component for velocity DoF `joint_v_index`.
    /// Requires a prior [`Self::compute_hessians`].
    pub fn frame_hessian(&mut self, frame: usize, joint_v_index: usize) -> Result<DMatrix<f64>> {
        let mut out = vec![0.0; 6 * self.nv];
        self.inner
            .pin_mut()
            .frame_hessian(frame as i64, joint_v_index as i64, out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DMatrix::from_row_slice(6, self.nv, &out))
    }

    /// The `3 × nv` CoM Jacobian time variation.
    pub fn com_jacobian_time_variation(&mut self) -> Result<DMatrix<f64>> {
        let mut out = vec![0.0; 3 * self.nv];
        self.inner
            .pin_mut()
            .com_jacobian_time_variation(
                self.state.q.as_slice(),
                self.state.qd.as_slice(),
                out.as_mut_slice(),
            )
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DMatrix::from_row_slice(3, self.nv, &out))
    }

    /// The `6 × nv` centroidal map `Ag`.
    pub fn centroidal_map(&mut self) -> Result<DMatrix<f64>> {
        let mut out = vec![0.0; 6 * self.nv];
        self.inner
            .pin_mut()
            .centroidal_map(self.state.q.as_slice(), out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DMatrix::from_row_slice(6, self.nv, &out))
    }

    /// CoM position in the world for the current configuration.
    pub fn com_world(&mut self) -> Result<Vector3<f64>> {
        let mut out = [0.0; 3];
        self.inner
            .pin_mut()
            .center_of_mass(self.state.q.as_slice(), &mut out)
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(Vector3::new(out[0], out[1], out[2]))
    }

    /// The `nv × nv` mass matrix (CRBA, symmetrized, with rotor inertia).
    pub fn mass_matrix(&mut self) -> Result<DMatrix<f64>> {
        let mut out = vec![0.0; self.nv * self.nv];
        self.inner
            .pin_mut()
            .mass_matrix(self.state.q.as_slice(), out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DMatrix::from_row_slice(self.nv, self.nv, &out))
    }

    /// Generalized gravity vector (length `nv`).
    pub fn generalized_gravity(&mut self) -> Result<DVector<f64>> {
        let mut out = vec![0.0; self.nv];
        self.inner
            .pin_mut()
            .generalized_gravity(self.state.q.as_slice(), out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DVector::from_vec(out))
    }

    /// Non-linear effects (Coriolis + centrifugal + gravity), length `nv`.
    pub fn non_linear_effects(&mut self) -> Result<DVector<f64>> {
        let mut out = vec![0.0; self.nv];
        self.inner
            .pin_mut()
            .non_linear_effects(
                self.state.q.as_slice(),
                self.state.qd.as_slice(),
                out.as_mut_slice(),
            )
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DVector::from_vec(out))
    }

    /// Total mass of the model.
    pub fn total_mass(&self) -> f64 {
        self.inner.total_mass()
    }

    /// Applies a configuration delta on the manifold: `q ← integrate(q, dq)`
    /// (`dq` has length `nv`). Used to apply an IK/ID velocity step.
    pub fn integrate_configuration(&mut self, dq: &DVector<f64>) -> Result<()> {
        let mut out = DVector::zeros(self.nq);
        self.inner
            .integrate(self.state.q.as_slice(), dq.as_slice(), out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        self.state.q = out;
        Ok(())
    }

    /// Integrates the state over `dt`: `qd += dt·qdd`, then `q ← integrate(q, dt·qd)`.
    pub fn integrate(&mut self, dt: f64) -> Result<()> {
        self.state.qd += dt * &self.state.qdd;
        let dq = dt * &self.state.qd;
        let mut out = DVector::zeros(self.nq);
        self.inner
            .integrate(self.state.q.as_slice(), dq.as_slice(), out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        self.state.q = out;
        Ok(())
    }

    // --- joint metadata ------------------------------------------------------

    /// `q`-offset of joint `name`.
    pub fn joint_offset(&self, name: &str) -> Result<usize> {
        let o = self.inner.joint_q_offset(name);
        (o >= 0)
            .then_some(o as usize)
            .ok_or_else(|| Error::Cxx(format!("joint not found: {name}")))
    }

    /// `v`-offset of joint `name`.
    pub fn joint_v_offset(&self, name: &str) -> Result<usize> {
        let o = self.inner.joint_v_offset(name);
        (o >= 0)
            .then_some(o as usize)
            .ok_or_else(|| Error::Cxx(format!("joint not found: {name}")))
    }

    /// Reads joint `name`'s value from `state.q`.
    pub fn joint(&self, name: &str) -> Result<f64> {
        Ok(self.state.q[self.joint_offset(name)?])
    }

    /// Sets joint `name`'s value in `state.q`.
    pub fn set_joint(&mut self, name: &str, value: f64) -> Result<()> {
        let o = self.joint_offset(name)?;
        self.state.q[o] = value;
        Ok(())
    }

    /// Whether a joint named `name` exists.
    pub fn has_joint(&self, name: &str) -> bool {
        self.inner.exist_joint(name)
    }

    /// Number of configuration DoF (`nq`) of joint `name`.
    pub fn joint_size(&self, name: &str) -> Result<usize> {
        let n = self.inner.joint_nq(name);
        (n >= 0)
            .then_some(n as usize)
            .ok_or_else(|| Error::Cxx(format!("joint not found: {name}")))
    }

    /// Number of velocity DoF (`nv`) of joint `name`.
    pub fn joint_v_size(&self, name: &str) -> Result<usize> {
        let n = self.inner.joint_nv(name);
        (n >= 0)
            .then_some(n as usize)
            .ok_or_else(|| Error::Cxx(format!("joint not found: {name}")))
    }

    /// Reads joint `name`'s velocity from `state.qd`.
    pub fn joint_velocity(&self, name: &str) -> Result<f64> {
        Ok(self.state.qd[self.joint_v_offset(name)?])
    }

    /// Sets joint `name`'s velocity in `state.qd`.
    pub fn set_joint_velocity(&mut self, name: &str, value: f64) -> Result<()> {
        let o = self.joint_v_offset(name)?;
        self.state.qd[o] = value;
        Ok(())
    }

    /// Reads joint `name`'s acceleration from `state.qdd`.
    pub fn joint_acceleration(&self, name: &str) -> Result<f64> {
        Ok(self.state.qdd[self.joint_v_offset(name)?])
    }

    /// Sets joint `name`'s acceleration in `state.qdd`.
    pub fn set_joint_acceleration(&mut self, name: &str, value: f64) -> Result<()> {
        let o = self.joint_v_offset(name)?;
        self.state.qdd[o] = value;
        Ok(())
    }

    /// The `6 × nv` Jacobian of joint index `joint_id` in `reference`.
    pub fn joint_jacobian(
        &self,
        joint_id: usize,
        reference: ReferenceFrame,
    ) -> Result<DMatrix<f64>> {
        let mut out = vec![0.0; 6 * self.nv];
        self.inner
            .joint_jacobian(joint_id as i64, reference as u8, out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        Ok(DMatrix::from_row_slice(6, self.nv, &out))
    }

    /// Position limits `(lower, upper)`, each length `nq`.
    pub fn position_limits(&self) -> (DVector<f64>, DVector<f64>) {
        let mut lower = vec![0.0; self.nq];
        let mut upper = vec![0.0; self.nq];
        self.inner.lower_position_limits(lower.as_mut_slice());
        self.inner.upper_position_limits(upper.as_mut_slice());
        (DVector::from_vec(lower), DVector::from_vec(upper))
    }

    /// Velocity limits (length `nv`).
    pub fn velocity_limits(&self) -> DVector<f64> {
        let mut out = vec![0.0; self.nv];
        self.inner.velocity_limits(out.as_mut_slice());
        DVector::from_vec(out)
    }

    /// Effort (torque) limits (length `nv`).
    pub fn effort_limits(&self) -> DVector<f64> {
        let mut out = vec![0.0; self.nv];
        self.inner.effort_limits(out.as_mut_slice());
        DVector::from_vec(out)
    }

    /// The `(lower, upper)` position limits of joint `name`.
    pub fn joint_limits(&self, name: &str) -> Result<(f64, f64)> {
        let k = self.joint_offset(name)?;
        let (lower, upper) = self.position_limits();
        Ok((lower[k], upper[k]))
    }

    /// Overrides joint `name`'s position limits.
    pub fn set_joint_limits(&mut self, name: &str, lower: f64, upper: f64) -> Result<()> {
        let k = self.joint_offset(name)? as i64;
        self.inner.pin_mut().set_position_limit(k, lower, upper);
        Ok(())
    }

    /// Overrides joint `name`'s velocity limit.
    pub fn set_velocity_limit(&mut self, name: &str, limit: f64) -> Result<()> {
        let k = self.joint_v_offset(name)? as i64;
        self.inner.pin_mut().set_velocity_limit(k, limit);
        Ok(())
    }

    /// Overrides joint `name`'s effort (torque) limit.
    pub fn set_torque_limit(&mut self, name: &str, limit: f64) -> Result<()> {
        let k = self.joint_v_offset(name)? as i64;
        self.inner.pin_mut().set_effort_limit(k, limit);
        Ok(())
    }

    /// Sets joint `name`'s rotor inertia (apparent-inertia model).
    pub fn set_rotor_inertia(&mut self, name: &str, inertia: f64) -> Result<()> {
        let k = self.joint_v_offset(name)? as i64;
        self.inner.pin_mut().set_rotor_inertia(k, inertia);
        Ok(())
    }

    /// Sets joint `name`'s rotor gear ratio (apparent-inertia model).
    pub fn set_gear_ratio(&mut self, name: &str, ratio: f64) -> Result<()> {
        let k = self.joint_v_offset(name)? as i64;
        self.inner.pin_mut().set_gear_ratio(k, ratio);
        Ok(())
    }

    /// All joint names (excluding the `universe` / `root_joint` by default).
    pub fn joint_names(&self, include_floating_base: bool) -> Vec<String> {
        self.inner.joint_names(include_floating_base)
    }

    /// All frame names, indexed by frame id.
    pub fn frame_names(&self) -> Vec<String> {
        self.inner.frame_names()
    }

    /// Sets the (linear) gravity vector.
    pub fn set_gravity(&mut self, gravity: Vector3<f64>) {
        self.inner
            .pin_mut()
            .set_gravity(gravity.x, gravity.y, gravity.z);
    }
}

fn dmat3(m: &Matrix3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 3, m.as_slice())
}

/// `3 × 3` skew-symmetric matrix of `v` (so `skew(v)·w = v × w`), as a `DMatrix`.
fn skew(v: &Vector3<f64>) -> DMatrix<f64> {
    DMatrix::from_row_slice(3, 3, &[0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0])
}

fn frame_placement_to_isometry(p: &ffi::FramePlacement) -> Isometry3<f64> {
    let translation = Translation3::new(p.translation[0], p.translation[1], p.translation[2]);
    // FramePlacement stores (qx, qy, qz, qw); Quaternion::new is (w, i, j, k).
    let quat = UnitQuaternion::from_quaternion(Quaternion::new(
        p.rotation[3],
        p.rotation[0],
        p.rotation[1],
        p.rotation[2],
    ));
    Isometry3::from_parts(translation, quat)
}

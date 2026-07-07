//! Safe wrapper over the Pinocchio shim.

use std::path::Path;

use cxx::UniquePtr;
use nalgebra::{DMatrix, DVector, Isometry3, Quaternion, Translation3, UnitQuaternion};

use crate::error::{Error, Result};
use crate::ffi::bridge as ffi;

/// Reference frame in which a frame Jacobian is expressed.
///
/// The discriminants form the ABI contract with the C++ shim and are stable;
/// they intentionally differ from Pinocchio's internal `ReferenceFrame` ordering
/// (the shim maps them explicitly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ReferenceFrame {
    /// Expressed in the frame's own local coordinates.
    Local = 0,
    /// Expressed in the world (universe) frame.
    World = 1,
    /// Origin at the frame, axes aligned with the world frame.
    LocalWorldAligned = 2,
}

/// A Pinocchio kinematic model bundled with its working `Data`.
///
/// Not thread-safe: the underlying `Data` is mutated in place by
/// [`Model::forward_kinematics`] and [`Model::compute_joint_jacobians`].
pub struct Model {
    inner: UniquePtr<ffi::PinocchioModel>,
}

impl Model {
    /// Build a model from a URDF file.
    ///
    /// With `floating_base`, a free-flyer root joint is prepended so mobile /
    /// whole-body bases can be represented (adds 7 to `nq`, 6 to `nv`).
    pub fn from_urdf(path: impl AsRef<Path>, floating_base: bool) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path
            .to_str()
            .ok_or_else(|| Error::UrdfLoad(format!("non-UTF-8 path: {}", path.display())))?;

        #[cfg(feature = "trace")]
        tracing::debug!(path = %path.display(), floating_base, "building pinocchio model");

        let inner = ffi::from_urdf(path_str, floating_base)
            .map_err(|e| Error::UrdfLoad(e.what().to_string()))?;
        Ok(Self { inner })
    }

    /// Configuration-space dimension (`nq`).
    pub fn nq(&self) -> usize {
        self.inner.nq()
    }

    /// Velocity / tangent-space dimension (`nv`).
    pub fn nv(&self) -> usize {
        self.inner.nv()
    }

    /// Run forward kinematics for configuration `q` (length must equal `nq`).
    pub fn forward_kinematics(&mut self, q: &DVector<f64>) -> Result<()> {
        self.check_q(q)?;
        self.inner
            .pin_mut()
            .forward_kinematics(q.as_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))
    }

    /// Refresh frame placements (`oMf`) from the last forward-kinematics pass.
    pub fn update_frame_placements(&mut self) {
        self.inner.pin_mut().update_frame_placements();
    }

    /// Frame index for `name`, or `None` if the model has no such frame.
    pub fn frame_id(&self, name: &str) -> Option<usize> {
        let id = self.inner.frame_id(name);
        if id < 0 {
            None
        } else {
            Some(id as usize)
        }
    }

    /// World placement of frame `id` as an [`Isometry3`].
    ///
    /// Requires a prior [`Model::forward_kinematics`] +
    /// [`Model::update_frame_placements`].
    pub fn frame_placement(&self, id: usize) -> Result<Isometry3<f64>> {
        let p = self
            .inner
            .frame_placement(id as i64)
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        let translation = Translation3::new(p.translation[0], p.translation[1], p.translation[2]);
        // FramePlacement stores (qx, qy, qz, qw); Quaternion::new is (w, i, j, k).
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(
            p.rotation[3],
            p.rotation[0],
            p.rotation[1],
            p.rotation[2],
        ));
        Ok(Isometry3::from_parts(translation, quat))
    }

    /// World placement of the named frame. Errors with [`Error::FrameNotFound`]
    /// if it is absent.
    pub fn frame_placement_by_name(&self, name: &str) -> Result<Isometry3<f64>> {
        let id = self
            .frame_id(name)
            .ok_or_else(|| Error::FrameNotFound(name.to_string()))?;
        self.frame_placement(id)
    }

    /// Compute joint Jacobians for configuration `q` (length must equal `nq`).
    ///
    /// Must be called before [`Model::frame_jacobian`].
    pub fn compute_joint_jacobians(&mut self, q: &DVector<f64>) -> Result<()> {
        self.check_q(q)?;
        self.inner
            .pin_mut()
            .compute_joint_jacobians(q.as_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))
    }

    /// The `6 x nv` frame Jacobian of frame `id` in `reference_frame`.
    ///
    /// Requires a prior [`Model::compute_joint_jacobians`]. Rows are the spatial
    /// velocity `[vx, vy, vz, wx, wy, wz]`.
    pub fn frame_jacobian(
        &self,
        id: usize,
        reference_frame: ReferenceFrame,
    ) -> Result<DMatrix<f64>> {
        let nv = self.nv();
        let mut out = vec![0.0f64; 6 * nv];
        self.inner
            .frame_jacobian(id as i64, reference_frame as u8, out.as_mut_slice())
            .map_err(|e| Error::Cxx(e.what().to_string()))?;
        // Shim fills row-major; DMatrix is column-major, so read as row slice.
        Ok(DMatrix::from_row_slice(6, nv, &out))
    }

    /// The `6 x nv` frame Jacobian of the named frame. Errors with
    /// [`Error::FrameNotFound`] if it is absent.
    pub fn frame_jacobian_by_name(
        &self,
        name: &str,
        reference_frame: ReferenceFrame,
    ) -> Result<DMatrix<f64>> {
        let id = self
            .frame_id(name)
            .ok_or_else(|| Error::FrameNotFound(name.to_string()))?;
        self.frame_jacobian(id, reference_frame)
    }

    fn check_q(&self, q: &DVector<f64>) -> Result<()> {
        let nq = self.nq();
        if q.len() != nq {
            return Err(Error::DimMismatch {
                what: "configuration q",
                expected: nq,
                got: q.len(),
            });
        }
        Ok(())
    }
}

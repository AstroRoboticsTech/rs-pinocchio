//! Low-level `cxx` bridge to the C++ Pinocchio shim.
//!
//! This module is an implementation detail; use the safe [`crate::Model`]
//! wrapper. The bridge mirrors the shim declared in
//! `cpp/pinocchio_shim.h` one-to-one.

#[cxx::bridge(namespace = "pinocchio_shim")]
pub(crate) mod bridge {
    /// SE3 frame placement: translation `(x, y, z)` and unit quaternion
    /// `(qx, qy, qz, qw)`. Mirrors the `[f64; 7]` layout requested by callers.
    #[derive(Clone, Copy, Debug)]
    struct FramePlacement {
        translation: [f64; 3],
        rotation: [f64; 4],
    }

    unsafe extern "C++" {
        include!("pinocchio-rs/cpp/pinocchio_shim.h");

        /// Opaque handle bundling a Pinocchio `Model` + `Data`.
        type PinocchioModel;

        /// Build a model (+ optional free-flyer root) and data from a URDF file.
        fn from_urdf(path: &str, floating_base: bool) -> Result<UniquePtr<PinocchioModel>>;

        /// Configuration-space dimension.
        fn nq(self: &PinocchioModel) -> usize;
        /// Velocity/tangent-space dimension.
        fn nv(self: &PinocchioModel) -> usize;

        /// Forward kinematics for configuration `q` (length must equal `nq`).
        fn forward_kinematics(self: Pin<&mut PinocchioModel>, q: &[f64]) -> Result<()>;
        /// Refresh frame placements (`oMf`) from the last forward kinematics.
        fn update_frame_placements(self: Pin<&mut PinocchioModel>);

        /// Frame index for `name`, or `-1` if absent.
        fn frame_id(self: &PinocchioModel, name: &str) -> i64;
        /// World placement of frame `id`.
        fn frame_placement(self: &PinocchioModel, id: i64) -> Result<FramePlacement>;

        /// Compute joint Jacobians for `q` (length must equal `nq`).
        fn compute_joint_jacobians(self: Pin<&mut PinocchioModel>, q: &[f64]) -> Result<()>;
        /// Fill `out` (length must equal `6 * nv`) with the row-major 6xnv frame
        /// Jacobian. `reference_frame`: 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED.
        fn frame_jacobian(
            self: &PinocchioModel,
            id: i64,
            reference_frame: u8,
            out: &mut [f64],
        ) -> Result<()>;
    }
}

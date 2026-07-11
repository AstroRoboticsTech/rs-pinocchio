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
        include!("rs-pinocchio/cpp/pinocchio_shim.h");

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

        // --- RobotWrapper-supporting primitives ------------------------------

        /// Full kinematics update for state `(q, v)`: frames forward kinematics,
        /// joint Jacobians, joint-Jacobian time variation, frame placements.
        fn update_kinematics(self: Pin<&mut PinocchioModel>, q: &[f64], v: &[f64]) -> Result<()>;

        /// Fill `out` (`6 * nv`, row-major) with the frame Jacobian time
        /// variation. Requires a prior [`update_kinematics`].
        fn frame_jacobian_time_variation(
            self: &PinocchioModel,
            id: i64,
            reference_frame: u8,
            out: &mut [f64],
        ) -> Result<()>;

        /// Fill `out` (`6 * nv`, row-major) with the joint Jacobian.
        fn joint_jacobian(
            self: &PinocchioModel,
            joint_id: i64,
            reference_frame: u8,
            out: &mut [f64],
        ) -> Result<()>;

        /// Neutral configuration into `out` (length `nq`).
        fn neutral(self: &PinocchioModel, out: &mut [f64]);
        /// `out = integrate(q, v)` on the configuration manifold (`out` len `nq`).
        fn integrate(self: &PinocchioModel, q: &[f64], v: &[f64], out: &mut [f64]) -> Result<()>;

        /// Center of mass position for `q` into `out` (length 3).
        fn center_of_mass(self: Pin<&mut PinocchioModel>, q: &[f64], out: &mut [f64])
            -> Result<()>;
        /// CoM Jacobian for `q` into `out` (`3 * nv`, row-major).
        fn com_jacobian(self: Pin<&mut PinocchioModel>, q: &[f64], out: &mut [f64]) -> Result<()>;
        /// CoM Jacobian time variation for `(q, v)` into `out` (`3 * nv`, row-major).
        fn com_jacobian_time_variation(
            self: Pin<&mut PinocchioModel>,
            q: &[f64],
            v: &[f64],
            out: &mut [f64],
        ) -> Result<()>;
        /// Centroidal map `Ag` for `q` into `out` (`6 * nv`, row-major).
        fn centroidal_map(self: Pin<&mut PinocchioModel>, q: &[f64], out: &mut [f64])
            -> Result<()>;

        /// Mass matrix (CRBA, symmetrized, with rotor inertia) into `out`
        /// (`nv * nv`, row-major).
        fn mass_matrix(self: Pin<&mut PinocchioModel>, q: &[f64], out: &mut [f64]) -> Result<()>;
        /// Generalized gravity for `q` into `out` (length `nv`).
        fn generalized_gravity(
            self: Pin<&mut PinocchioModel>,
            q: &[f64],
            out: &mut [f64],
        ) -> Result<()>;
        /// Non-linear effects for `(q, v)` into `out` (length `nv`).
        fn non_linear_effects(
            self: Pin<&mut PinocchioModel>,
            q: &[f64],
            v: &[f64],
            out: &mut [f64],
        ) -> Result<()>;

        /// Total mass of the model.
        fn total_mass(self: &PinocchioModel) -> f64;

        /// Placement of the root (free-flyer) joint, `model.jointPlacements[1]`
        /// (usually identity). Used to map the free-flyer config to the base pose.
        fn root_joint_placement(self: &PinocchioModel) -> FramePlacement;

        // Joint metadata (by name). Offsets are `-1` when the joint is absent.
        /// Whether a joint with `name` exists.
        fn exist_joint(self: &PinocchioModel, name: &str) -> bool;
        /// `idx_q` (offset into `q`) of joint `name`, or `-1`.
        fn joint_q_offset(self: &PinocchioModel, name: &str) -> i64;
        /// `idx_v` (offset into `v`/`a`) of joint `name`, or `-1`.
        fn joint_v_offset(self: &PinocchioModel, name: &str) -> i64;
        /// `nq` of joint `name`, or `-1`.
        fn joint_nq(self: &PinocchioModel, name: &str) -> i64;
        /// `nv` of joint `name`, or `-1`.
        fn joint_nv(self: &PinocchioModel, name: &str) -> i64;

        /// Joint names (optionally including the `universe`/`root_joint`).
        fn joint_names(self: &PinocchioModel, include_floating_base: bool) -> Vec<String>;
        /// All frame names, indexed by frame id.
        fn frame_names(self: &PinocchioModel) -> Vec<String>;

        // Limit vectors (copied out). Lengths: position `nq`, velocity/effort `nv`.
        fn lower_position_limits(self: &PinocchioModel, out: &mut [f64]);
        fn upper_position_limits(self: &PinocchioModel, out: &mut [f64]);
        fn velocity_limits(self: &PinocchioModel, out: &mut [f64]);
        fn effort_limits(self: &PinocchioModel, out: &mut [f64]);

        /// Override a joint's position limits (by `q` offset `k`).
        fn set_position_limit(self: Pin<&mut PinocchioModel>, k: i64, lower: f64, upper: f64);
        /// Override a joint's velocity limit (by `v` offset `k`).
        fn set_velocity_limit(self: Pin<&mut PinocchioModel>, k: i64, limit: f64);
        /// Override a joint's effort (torque) limit (by `v` offset `k`).
        fn set_effort_limit(self: Pin<&mut PinocchioModel>, k: i64, limit: f64);

        /// Rotor inertia / gear ratio (apparent-inertia model), by `v` offset.
        fn set_rotor_inertia(self: Pin<&mut PinocchioModel>, k: i64, inertia: f64);
        fn set_gear_ratio(self: Pin<&mut PinocchioModel>, k: i64, ratio: f64);
        /// Set the (linear) gravity vector.
        fn set_gravity(self: Pin<&mut PinocchioModel>, x: f64, y: f64, z: f64);
    }
}

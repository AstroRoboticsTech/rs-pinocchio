// Clean-room C++ shim over Pinocchio's public API for the pinocchio-rs cxx bridge.
//
// This header is deliberately Pinocchio-free: it only forward-declares the cxx
// shared struct and declares the opaque `PinocchioModel` class + the `from_urdf`
// factory. All Pinocchio includes live in pinocchio_shim.cpp so the generated
// cxx bridge code stays cheap to compile and free of Eigen/Boost templates.
//
// Written from scratch against Pinocchio's own BSD-2 public API
// (pinocchio::urdf::buildModel, forwardKinematics, updateFramePlacements,
// getFrameId, getFrameJacobian, computeJointJacobians, ReferenceFrame).

#pragma once

#include "rust/cxx.h"

#include <cstdint>
#include <memory>

namespace pinocchio_shim {

// Defined by the cxx-generated header (pinocchio-rs/src/ffi.rs.h). Only the name
// is needed here for by-value return declarations; the full definition is pulled
// into pinocchio_shim.cpp.
struct FramePlacement;

// Owns a Pinocchio Model + its companion Data. Opaque to Rust; held via
// std::unique_ptr through the cxx bridge.
class PinocchioModel {
public:
  explicit PinocchioModel(std::unique_ptr<struct ModelData> impl);
  ~PinocchioModel();

  PinocchioModel(const PinocchioModel &) = delete;
  PinocchioModel &operator=(const PinocchioModel &) = delete;

  // Configuration / velocity dimensions.
  std::size_t nq() const noexcept;
  std::size_t nv() const noexcept;

  // Run forward kinematics for configuration `q` (length must equal nq).
  void forward_kinematics(rust::Slice<const double> q);

  // Refresh oMf (frame placements) from the last forward_kinematics call.
  void update_frame_placements();

  // Frame index for `name`, or -1 if the model has no such frame.
  std::int64_t frame_id(rust::Str name) const;

  // World placement of frame `id` (requires a prior forward_kinematics +
  // update_frame_placements). Throws if `id` is out of range.
  FramePlacement frame_placement(std::int64_t id) const;

  // Compute joint Jacobians for configuration `q` (length must equal nq).
  // Must run before frame_jacobian.
  void compute_joint_jacobians(rust::Slice<const double> q);

  // Fill `out` (length must equal 6*nv) with the 6xnv frame Jacobian of frame
  // `id`, row-major. `reference_frame`: 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED.
  void frame_jacobian(std::int64_t id, std::uint8_t reference_frame,
                      rust::Slice<double> out) const;

  // --- RobotWrapper-supporting primitives ------------------------------------

  // Full kinematics update for state (q, v): frames FK, joint Jacobians +
  // their time variation, and frame placements.
  void update_kinematics(rust::Slice<const double> q, rust::Slice<const double> v);

  void frame_jacobian_time_variation(std::int64_t id, std::uint8_t reference_frame,
                                     rust::Slice<double> out) const;
  void joint_jacobian(std::int64_t joint_id, std::uint8_t reference_frame,
                      rust::Slice<double> out) const;

  void neutral(rust::Slice<double> out) const;
  void integrate(rust::Slice<const double> q, rust::Slice<const double> v,
                 rust::Slice<double> out) const;

  void center_of_mass(rust::Slice<const double> q, rust::Slice<double> out);
  void com_jacobian(rust::Slice<const double> q, rust::Slice<double> out);
  void centroidal_map(rust::Slice<const double> q, rust::Slice<double> out);

  void mass_matrix(rust::Slice<const double> q, rust::Slice<double> out);
  void generalized_gravity(rust::Slice<const double> q, rust::Slice<double> out);
  void non_linear_effects(rust::Slice<const double> q, rust::Slice<const double> v,
                          rust::Slice<double> out);

  double total_mass() const;

  FramePlacement root_joint_placement() const;

  bool exist_joint(rust::Str name) const;
  std::int64_t joint_q_offset(rust::Str name) const;
  std::int64_t joint_v_offset(rust::Str name) const;
  std::int64_t joint_nq(rust::Str name) const;
  std::int64_t joint_nv(rust::Str name) const;

  rust::Vec<rust::String> joint_names(bool include_floating_base) const;
  rust::Vec<rust::String> frame_names() const;

  void lower_position_limits(rust::Slice<double> out) const;
  void upper_position_limits(rust::Slice<double> out) const;
  void velocity_limits(rust::Slice<double> out) const;
  void effort_limits(rust::Slice<double> out) const;

  void set_position_limit(std::int64_t k, double lower, double upper);
  void set_velocity_limit(std::int64_t k, double limit);
  void set_effort_limit(std::int64_t k, double limit);

  void set_rotor_inertia(std::int64_t k, double inertia);
  void set_gear_ratio(std::int64_t k, double ratio);
  void set_gravity(double x, double y, double z);

private:
  std::unique_ptr<struct ModelData> impl_;
};

// Build a model + data from a URDF file. With `floating_base`, a free-flyer
// root joint is prepended (for mobile / whole-body bases). Throws on parse
// failure (surfaced as a Rust Result by cxx).
std::unique_ptr<PinocchioModel> from_urdf(rust::Str path, bool floating_base);

} // namespace pinocchio_shim

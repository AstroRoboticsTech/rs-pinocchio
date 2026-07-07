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

private:
  std::unique_ptr<struct ModelData> impl_;
};

// Build a model + data from a URDF file. With `floating_base`, a free-flyer
// root joint is prepended (for mobile / whole-body bases). Throws on parse
// failure (surfaced as a Rust Result by cxx).
std::unique_ptr<PinocchioModel> from_urdf(rust::Str path, bool floating_base);

} // namespace pinocchio_shim

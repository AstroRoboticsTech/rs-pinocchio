// Clean-room implementation of the pinocchio-rs C++ shim.
//
// Only Pinocchio's public BSD-2 API is used here. No third-party binding code
// was consulted or copied.

// fwd.hpp must precede Eigen/Boost so Pinocchio can install its Eigen plugins.
#include <pinocchio/fwd.hpp>

// Umbrella headers (Model, Data, joints incl. free-flyer, SE3). Pinocchio 4.x
// consolidated the per-class headers behind these.
#include <pinocchio/multibody.hpp>
#include <pinocchio/spatial.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <Eigen/Geometry>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "pinocchio-rs/cpp/pinocchio_shim.h"
// Full definition of the cxx shared struct (FramePlacement).
#include "pinocchio-rs/src/ffi.rs.h"

namespace pinocchio_shim {

// Private payload: keeps all Pinocchio types out of the public header.
struct ModelData {
  pinocchio::Model model;
  pinocchio::Data data;

  explicit ModelData(pinocchio::Model m)
      : model(std::move(m)), data(model) {}
};

PinocchioModel::PinocchioModel(std::unique_ptr<ModelData> impl)
    : impl_(std::move(impl)) {}

PinocchioModel::~PinocchioModel() = default;

std::size_t PinocchioModel::nq() const noexcept {
  return static_cast<std::size_t>(impl_->model.nq);
}

std::size_t PinocchioModel::nv() const noexcept {
  return static_cast<std::size_t>(impl_->model.nv);
}

void PinocchioModel::forward_kinematics(rust::Slice<const double> q) {
  if (static_cast<Eigen::Index>(q.size()) != impl_->model.nq) {
    throw std::invalid_argument(
        "forward_kinematics: q length does not match model.nq");
  }
  Eigen::Map<const Eigen::VectorXd> qv(q.data(),
                                       static_cast<Eigen::Index>(q.size()));
  pinocchio::forwardKinematics(impl_->model, impl_->data, qv);
}

void PinocchioModel::update_frame_placements() {
  pinocchio::updateFramePlacements(impl_->model, impl_->data);
}

std::int64_t PinocchioModel::frame_id(rust::Str name) const {
  const std::string n(name);
  if (!impl_->model.existFrame(n)) {
    return -1;
  }
  return static_cast<std::int64_t>(impl_->model.getFrameId(n));
}

FramePlacement PinocchioModel::frame_placement(std::int64_t id) const {
  if (id < 0 ||
      static_cast<std::size_t>(id) >= impl_->data.oMf.size()) {
    throw std::out_of_range("frame_placement: frame id out of range");
  }
  const pinocchio::SE3 &M = impl_->data.oMf[static_cast<std::size_t>(id)];
  const Eigen::Quaterniond quat(M.rotation());

  FramePlacement fp{};
  fp.translation[0] = M.translation().x();
  fp.translation[1] = M.translation().y();
  fp.translation[2] = M.translation().z();
  fp.rotation[0] = quat.x();
  fp.rotation[1] = quat.y();
  fp.rotation[2] = quat.z();
  fp.rotation[3] = quat.w();
  return fp;
}

void PinocchioModel::compute_joint_jacobians(rust::Slice<const double> q) {
  if (static_cast<Eigen::Index>(q.size()) != impl_->model.nq) {
    throw std::invalid_argument(
        "compute_joint_jacobians: q length does not match model.nq");
  }
  Eigen::Map<const Eigen::VectorXd> qv(q.data(),
                                       static_cast<Eigen::Index>(q.size()));
  pinocchio::computeJointJacobians(impl_->model, impl_->data, qv);
}

void PinocchioModel::frame_jacobian(std::int64_t id, std::uint8_t reference_frame,
                                    rust::Slice<double> out) const {
  const Eigen::Index nv = impl_->model.nv;
  if (static_cast<Eigen::Index>(out.size()) != 6 * nv) {
    throw std::invalid_argument(
        "frame_jacobian: out length must equal 6 * nv");
  }
  if (id < 0 ||
      static_cast<std::size_t>(id) >= impl_->model.frames.size()) {
    throw std::out_of_range("frame_jacobian: frame id out of range");
  }

  // Map the caller's ABI-stable reference-frame code onto Pinocchio's enum.
  // (Pinocchio's own enum ordering differs, so map explicitly.)
  pinocchio::ReferenceFrame ref;
  switch (reference_frame) {
  case 0:
    ref = pinocchio::LOCAL;
    break;
  case 1:
    ref = pinocchio::WORLD;
    break;
  case 2:
    ref = pinocchio::LOCAL_WORLD_ALIGNED;
    break;
  default:
    throw std::invalid_argument("frame_jacobian: unknown reference_frame");
  }

  pinocchio::Data::Matrix6x J(6, nv);
  J.setZero();
  pinocchio::getFrameJacobian(impl_->model, impl_->data,
                              static_cast<pinocchio::FrameIndex>(id), ref, J);

  // Row-major fill: out[r*nv + c] = J(r, c).
  for (Eigen::Index r = 0; r < 6; ++r) {
    for (Eigen::Index c = 0; c < nv; ++c) {
      out[static_cast<std::size_t>(r * nv + c)] = J(r, c);
    }
  }
}

std::unique_ptr<PinocchioModel> from_urdf(rust::Str path, bool floating_base) {
  const std::string p(path);
  pinocchio::Model model;
  if (floating_base) {
    pinocchio::urdf::buildModel(p, pinocchio::JointModelFreeFlyer(), model);
  } else {
    pinocchio::urdf::buildModel(p, model);
  }
  auto impl = std::make_unique<ModelData>(std::move(model));
  return std::make_unique<PinocchioModel>(std::move(impl));
}

} // namespace pinocchio_shim

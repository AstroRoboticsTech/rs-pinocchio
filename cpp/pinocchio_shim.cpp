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
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <Eigen/Geometry>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "rs-pinocchio/cpp/pinocchio_shim.h"
// Full definition of the cxx shared struct (FramePlacement).
#include "rs-pinocchio/src/ffi.rs.h"

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

// --- RobotWrapper-supporting primitives --------------------------------------

namespace {
// Row-major copy of an Eigen matrix into a Rust slice (out.size() == rows*cols).
template <typename Derived>
void fill_row_major(rust::Slice<double> out, const Eigen::MatrixBase<Derived> &m) {
  const Eigen::Index rows = m.rows();
  const Eigen::Index cols = m.cols();
  if (static_cast<Eigen::Index>(out.size()) != rows * cols) {
    throw std::invalid_argument("output slice size does not match matrix size");
  }
  std::size_t idx = 0;
  for (Eigen::Index r = 0; r < rows; ++r) {
    for (Eigen::Index c = 0; c < cols; ++c) {
      out[idx++] = m(r, c);
    }
  }
}

void copy_vector(rust::Slice<double> out, const Eigen::VectorXd &v) {
  if (static_cast<Eigen::Index>(out.size()) != v.size()) {
    throw std::invalid_argument("output slice size does not match vector size");
  }
  for (Eigen::Index k = 0; k < v.size(); ++k) {
    out[static_cast<std::size_t>(k)] = v[k];
  }
}

Eigen::Map<const Eigen::VectorXd> map_q(const pinocchio::Model &model,
                                        rust::Slice<const double> q) {
  if (static_cast<Eigen::Index>(q.size()) != model.nq) {
    throw std::invalid_argument("q length does not match model.nq");
  }
  return Eigen::Map<const Eigen::VectorXd>(q.data(), static_cast<Eigen::Index>(q.size()));
}

Eigen::Map<const Eigen::VectorXd> map_v(const pinocchio::Model &model,
                                        rust::Slice<const double> v) {
  if (static_cast<Eigen::Index>(v.size()) != model.nv) {
    throw std::invalid_argument("v length does not match model.nv");
  }
  return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()));
}

pinocchio::ReferenceFrame to_reference(std::uint8_t code) {
  switch (code) {
  case 0:
    return pinocchio::LOCAL;
  case 1:
    return pinocchio::WORLD;
  case 2:
    return pinocchio::LOCAL_WORLD_ALIGNED;
  default:
    throw std::invalid_argument("unknown reference_frame");
  }
}
} // namespace

void PinocchioModel::update_kinematics(rust::Slice<const double> q,
                                       rust::Slice<const double> v) {
  auto qv = map_q(impl_->model, q);
  auto vv = map_v(impl_->model, v);
  pinocchio::framesForwardKinematics(impl_->model, impl_->data, qv);
  pinocchio::computeJointJacobians(impl_->model, impl_->data, qv);
  pinocchio::computeJointJacobiansTimeVariation(impl_->model, impl_->data, qv, vv);
  pinocchio::updateFramePlacements(impl_->model, impl_->data);
}

void PinocchioModel::frame_jacobian_time_variation(std::int64_t id,
                                                   std::uint8_t reference_frame,
                                                   rust::Slice<double> out) const {
  const Eigen::Index nv = impl_->model.nv;
  if (id < 0 || static_cast<std::size_t>(id) >= impl_->model.frames.size()) {
    throw std::out_of_range("frame_jacobian_time_variation: frame id out of range");
  }
  pinocchio::Data::Matrix6x J(6, nv);
  J.setZero();
  pinocchio::getFrameJacobianTimeVariation(impl_->model, impl_->data,
                                           static_cast<pinocchio::FrameIndex>(id),
                                           to_reference(reference_frame), J);
  fill_row_major(out, J);
}

void PinocchioModel::joint_jacobian(std::int64_t joint_id, std::uint8_t reference_frame,
                                    rust::Slice<double> out) const {
  const Eigen::Index nv = impl_->model.nv;
  if (joint_id < 0 || static_cast<std::size_t>(joint_id) >= impl_->model.joints.size()) {
    throw std::out_of_range("joint_jacobian: joint id out of range");
  }
  pinocchio::Data::Matrix6x J(6, nv);
  J.setZero();
  pinocchio::getJointJacobian(impl_->model, impl_->data,
                              static_cast<pinocchio::JointIndex>(joint_id),
                              to_reference(reference_frame), J);
  fill_row_major(out, J);
}

void PinocchioModel::neutral(rust::Slice<double> out) const {
  copy_vector(out, pinocchio::neutral(impl_->model));
}

void PinocchioModel::integrate(rust::Slice<const double> q, rust::Slice<const double> v,
                               rust::Slice<double> out) const {
  auto qv = map_q(impl_->model, q);
  auto vv = map_v(impl_->model, v);
  copy_vector(out, pinocchio::integrate(impl_->model, qv, vv));
}

void PinocchioModel::center_of_mass(rust::Slice<const double> q, rust::Slice<double> out) {
  auto qv = map_q(impl_->model, q);
  pinocchio::centerOfMass(impl_->model, impl_->data, qv);
  const Eigen::Vector3d &com = impl_->data.com[0];
  if (out.size() != 3) {
    throw std::invalid_argument("center_of_mass: out length must be 3");
  }
  out[0] = com.x();
  out[1] = com.y();
  out[2] = com.z();
}

void PinocchioModel::com_jacobian(rust::Slice<const double> q, rust::Slice<double> out) {
  auto qv = map_q(impl_->model, q);
  fill_row_major(out, pinocchio::jacobianCenterOfMass(impl_->model, impl_->data, qv));
}

void PinocchioModel::com_jacobian_time_variation(rust::Slice<const double> q,
                                                 rust::Slice<const double> v,
                                                 rust::Slice<double> out) {
  auto qv = map_q(impl_->model, q);
  auto vv = map_v(impl_->model, v);
  // dJ_com = (dAg / m).topRows(3); see stack-of-tasks/pinocchio#1297.
  Eigen::MatrixXd dag =
      pinocchio::computeCentroidalMapTimeVariation(impl_->model, impl_->data, qv, vv);
  double mass = total_mass();
  fill_row_major(out, (dag.topRows(3) / mass).eval());
}

void PinocchioModel::centroidal_map(rust::Slice<const double> q, rust::Slice<double> out) {
  auto qv = map_q(impl_->model, q);
  fill_row_major(out, pinocchio::computeCentroidalMap(impl_->model, impl_->data, qv));
}

void PinocchioModel::mass_matrix(rust::Slice<const double> q, rust::Slice<double> out) {
  auto qv = map_q(impl_->model, q);
  pinocchio::crba(impl_->model, impl_->data, qv);
  // crba fills only the upper triangle; mirror it.
  impl_->data.M.triangularView<Eigen::StrictlyLower>() =
      impl_->data.M.transpose().triangularView<Eigen::StrictlyLower>();
  Eigen::MatrixXd M = impl_->data.M;
  for (Eigen::Index k = 0; k < M.rows(); ++k) {
    M(k, k) += impl_->model.rotorGearRatio[k] * impl_->model.rotorGearRatio[k] *
               impl_->model.rotorInertia[k];
  }
  fill_row_major(out, M);
}

void PinocchioModel::generalized_gravity(rust::Slice<const double> q, rust::Slice<double> out) {
  auto qv = map_q(impl_->model, q);
  pinocchio::computeGeneralizedGravity(impl_->model, impl_->data, qv);
  copy_vector(out, impl_->data.g);
}

void PinocchioModel::non_linear_effects(rust::Slice<const double> q,
                                        rust::Slice<const double> v, rust::Slice<double> out) {
  auto qv = map_q(impl_->model, q);
  auto vv = map_v(impl_->model, v);
  copy_vector(out, pinocchio::nonLinearEffects(impl_->model, impl_->data, qv, vv));
}

double PinocchioModel::total_mass() const {
  double mass = 0.0;
  for (const auto &inertia : impl_->model.inertias) {
    mass += inertia.mass();
  }
  return mass;
}

FramePlacement PinocchioModel::root_joint_placement() const {
  // jointPlacements[0] is the universe; [1] is the (free-flyer) root joint.
  const pinocchio::SE3 &M = (impl_->model.jointPlacements.size() > 1)
                                ? impl_->model.jointPlacements[1]
                                : pinocchio::SE3::Identity();
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

bool PinocchioModel::exist_joint(rust::Str name) const {
  return impl_->model.existJointName(std::string(name));
}

std::int64_t PinocchioModel::joint_q_offset(rust::Str name) const {
  const std::string n(name);
  if (!impl_->model.existJointName(n)) {
    return -1;
  }
  return impl_->model.joints[impl_->model.getJointId(n)].idx_q();
}

std::int64_t PinocchioModel::joint_v_offset(rust::Str name) const {
  const std::string n(name);
  if (!impl_->model.existJointName(n)) {
    return -1;
  }
  return impl_->model.joints[impl_->model.getJointId(n)].idx_v();
}

std::int64_t PinocchioModel::joint_nq(rust::Str name) const {
  const std::string n(name);
  if (!impl_->model.existJointName(n)) {
    return -1;
  }
  return impl_->model.joints[impl_->model.getJointId(n)].nq();
}

std::int64_t PinocchioModel::joint_nv(rust::Str name) const {
  const std::string n(name);
  if (!impl_->model.existJointName(n)) {
    return -1;
  }
  return impl_->model.joints[impl_->model.getJointId(n)].nv();
}

rust::Vec<rust::String> PinocchioModel::joint_names(bool include_floating_base) const {
  rust::Vec<rust::String> out;
  for (const auto &name : impl_->model.names) {
    if (!include_floating_base && (name == "universe" || name == "root_joint")) {
      continue;
    }
    out.push_back(rust::String(name));
  }
  return out;
}

rust::Vec<rust::String> PinocchioModel::frame_names() const {
  rust::Vec<rust::String> out;
  for (const auto &frame : impl_->model.frames) {
    out.push_back(rust::String(frame.name));
  }
  return out;
}

void PinocchioModel::lower_position_limits(rust::Slice<double> out) const {
  copy_vector(out, impl_->model.lowerPositionLimit);
}

void PinocchioModel::upper_position_limits(rust::Slice<double> out) const {
  copy_vector(out, impl_->model.upperPositionLimit);
}

void PinocchioModel::velocity_limits(rust::Slice<double> out) const {
  copy_vector(out, impl_->model.velocityLimit);
}

void PinocchioModel::effort_limits(rust::Slice<double> out) const {
  copy_vector(out, impl_->model.effortLimit);
}

void PinocchioModel::set_position_limit(std::int64_t k, double lower, double upper) {
  impl_->model.lowerPositionLimit[static_cast<Eigen::Index>(k)] = lower;
  impl_->model.upperPositionLimit[static_cast<Eigen::Index>(k)] = upper;
}

void PinocchioModel::set_velocity_limit(std::int64_t k, double limit) {
  impl_->model.velocityLimit[static_cast<Eigen::Index>(k)] = limit;
}

void PinocchioModel::set_effort_limit(std::int64_t k, double limit) {
  impl_->model.effortLimit[static_cast<Eigen::Index>(k)] = limit;
}

void PinocchioModel::set_rotor_inertia(std::int64_t k, double inertia) {
  impl_->model.rotorInertia[static_cast<Eigen::Index>(k)] = inertia;
}

void PinocchioModel::set_gear_ratio(std::int64_t k, double ratio) {
  impl_->model.rotorGearRatio[static_cast<Eigen::Index>(k)] = ratio;
}

void PinocchioModel::set_gravity(double x, double y, double z) {
  impl_->model.gravity.linear() = Eigen::Vector3d(x, y, z);
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

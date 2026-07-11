//! Pure-Rust port of Rhoban [PlaCo](https://github.com/Rhoban/placo), a QP-based
//! planning and control framework.
//!
//! The port mirrors PlaCo's C++ module layout. Function signatures are adapted
//! to idiomatic Rust (methods instead of operator overloads, handles instead of
//! back-pointers) but the API usage stays close to the original.
//!
//! Layers, in dependency order:
//!
//! - [`tools`] — splines, polynomials, 2D geometry, angle helpers. No native deps.
//! - [`problem`] — the QP modeling layer (variables, affine expressions,
//!   constraints, and a [`problem::Problem`] solved via [`clarabel`]). No native deps.
//!
//! Higher layers (`RobotWrapper`, the kinematics/dynamics solvers, humanoid walk
//! planning) build on the Pinocchio binding and additionally require the `ffi`
//! feature.

#[cfg(feature = "ffi")]
pub mod model;
pub mod problem;
pub mod tools;

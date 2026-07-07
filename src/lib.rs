//! Clean-room Rust bindings for the [Pinocchio] rigid-body dynamics library,
//! via [cxx].
//!
//! The crate version tracks the bound Pinocchio version (currently `4.1.0`).
//! Scope is forward kinematics + frame Jacobians — enough for a whole-body
//! differential-IK layer to consume.
//!
//! ```no_run
//! use nalgebra::DVector;
//! use rs_pinocchio::{Model, ReferenceFrame};
//!
//! let mut model = Model::from_urdf("robot.urdf", /* floating_base = */ false)?;
//! let q = DVector::zeros(model.nq());
//!
//! model.forward_kinematics(&q)?;
//! model.update_frame_placements();
//! let tip = model.frame_id("tool").expect("frame exists");
//! let placement = model.frame_placement(tip)?;
//!
//! model.compute_joint_jacobians(&q)?;
//! let jac = model.frame_jacobian(tip, ReferenceFrame::LocalWorldAligned)?; // 6 x nv
//! # Ok::<(), rs_pinocchio::Error>(())
//! ```
//!
//! [Pinocchio]: https://github.com/stack-of-tasks/pinocchio
//! [cxx]: https://cxx.rs

mod error;
mod ffi;
mod model;

pub use error::{Error, Result};
pub use model::{Model, ReferenceFrame};

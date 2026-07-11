//! Robot model + state (PlaCo `placo::model`).
//!
//! Requires the `ffi` feature (it wraps the Pinocchio binding).

mod robot_wrapper;

pub use robot_wrapper::{RobotWrapper, State};

//! Rust rigid-body dynamics and QP-based planning/control.
//!
//! Two layers live in this crate, each behind a Cargo feature:
//!
//! - **`ffi`** (default) — clean-room [cxx] bindings to the [Pinocchio]
//!   rigid-body dynamics library: the low-level [`Model`] + [`ReferenceFrame`].
//!   Requires a linkable Pinocchio install at build time.
//! - **`placo`** — a pure-Rust port of Rhoban [PlaCo], a QP-based planning and
//!   control framework (task-space inverse kinematics / dynamics, footstep and
//!   walk planning). The [`placo::tools`] and [`placo::problem`] layers need no
//!   native dependencies and build without Pinocchio; higher layers additionally
//!   require `ffi`.
//!
//! For a pure-Rust build (no Pinocchio needed), disable defaults:
//!
//! ```sh
//! cargo test --no-default-features --features placo
//! ```
//!
//! [Pinocchio]: https://github.com/stack-of-tasks/pinocchio
//! [PlaCo]: https://github.com/Rhoban/placo
//! [cxx]: https://cxx.rs

mod error;

#[cfg(feature = "ffi")]
mod ffi;
#[cfg(feature = "ffi")]
mod model;

#[cfg(feature = "placo")]
pub mod placo;

pub use error::{Error, Result};

#[cfg(feature = "ffi")]
pub use model::{Model, ReferenceFrame};

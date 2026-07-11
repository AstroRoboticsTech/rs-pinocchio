//! Humanoid planning and control (PlaCo `placo::humanoid`).
//!
//! The footstep-planning layer ([`side`], [`parameters`], [`footsteps`]) is pure
//! Rust and needs no Pinocchio. Higher layers (the walk pattern generator and
//! `HumanoidRobot`) will additionally require the `ffi` feature.

mod footsteps;
mod lipm;
mod parameters;
mod side;
mod swing_foot;

pub use footsteps::{
    make_supports, Footstep, FootstepsPlanner, FootstepsPlannerNaive, FootstepsPlannerRepetitive,
    Support,
};
pub use lipm::{Lipm, LipmTrajectory};
pub use parameters::{FootstepClipping, HumanoidParameters};
pub use side::Side;
pub use swing_foot::{SwingFoot, SwingFootQuintic};

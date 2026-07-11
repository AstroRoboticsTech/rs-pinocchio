//! Humanoid planning and control (PlaCo `placo::humanoid`).
//!
//! The footstep-planning layer ([`side`], [`parameters`], [`footsteps`]) is pure
//! Rust and needs no Pinocchio. Higher layers (the walk pattern generator and
//! `HumanoidRobot`) will additionally require the `ffi` feature.

mod footsteps;
mod parameters;
mod side;

pub use footsteps::{
    make_supports, Footstep, FootstepsPlanner, FootstepsPlannerNaive, FootstepsPlannerRepetitive,
    Support,
};
pub use parameters::{FootstepClipping, HumanoidParameters};
pub use side::Side;

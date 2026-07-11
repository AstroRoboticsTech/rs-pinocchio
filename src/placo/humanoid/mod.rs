//! Humanoid planning and control (PlaCo `placo::humanoid`).
//!
//! The footstep-planning layer ([`side`], [`parameters`], [`footsteps`]) is pure
//! Rust and needs no Pinocchio. Higher layers (the walk pattern generator and
//! `HumanoidRobot`) will additionally require the `ffi` feature.

#[cfg(feature = "ffi")]
mod dummy_walk;
mod footsteps;
#[cfg(feature = "ffi")]
mod humanoid_robot;
mod kick;
mod lipm;
mod parameters;
mod side;
mod swing_foot;
mod walk_pattern_generator;
#[cfg(feature = "ffi")]
mod walk_tasks;

#[cfg(feature = "ffi")]
pub use dummy_walk::DummyWalk;
pub use footsteps::{
    make_supports, Footstep, FootstepsPlanner, FootstepsPlannerNaive, FootstepsPlannerRepetitive,
    Support,
};
#[cfg(feature = "ffi")]
pub use humanoid_robot::HumanoidRobot;
pub use kick::{make_trajectory as make_kick_trajectory, KickTrajectory};
pub use lipm::{Lipm, LipmTrajectory};
pub use parameters::{FootstepClipping, HumanoidParameters};
pub use side::Side;
pub use swing_foot::{SwingFoot, SwingFootCubic, SwingFootQuintic};
pub use walk_pattern_generator::{TrajectoryPart, WalkPatternGenerator, WalkTrajectory};
#[cfg(feature = "ffi")]
pub use walk_tasks::WalkTasks;

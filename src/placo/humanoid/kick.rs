//! Kick foot trajectory (PlaCo `Kick`).
//!
//! Mirrors PlaCo, where `Kick::make_trajectory`'s body is currently commented
//! out (the kick parameters — up/shot/neutral durations, amplitude, foot height
//! — are not part of `HumanoidParameters`). The trajectory is therefore an empty
//! [`CubicSpline3D`]; `pos`/`vel` return zero until the profile is populated.

use nalgebra::Vector3;

use super::parameters::HumanoidParameters;
use super::side::Side;
use crate::placo::tools::CubicSpline3D;

/// A kick foot trajectory over a 3D cubic spline.
#[derive(Clone, Debug, Default)]
pub struct KickTrajectory {
    /// Trajectory start time.
    pub t_start: f64,
    /// Trajectory end time.
    pub t_end: f64,
    /// The foot trajectory spline.
    pub foot_trajectory: CubicSpline3D,
}

impl KickTrajectory {
    /// Foot position at time `t`.
    pub fn pos(&mut self, t: f64) -> Vector3<f64> {
        self.foot_trajectory.pos(t)
    }

    /// Foot velocity at time `t`.
    pub fn vel(&mut self, t: f64) -> Vector3<f64> {
        self.foot_trajectory.vel(t)
    }
}

/// Builds a kick trajectory (currently a stub, matching upstream PlaCo).
#[allow(clippy::too_many_arguments)]
pub fn make_trajectory(
    _kicking_side: Side,
    t_start: f64,
    t_end: f64,
    _start: Vector3<f64>,
    _target: Vector3<f64>,
    _t_world_opposite: nalgebra::Isometry3<f64>,
    _parameters: &HumanoidParameters,
) -> KickTrajectory {
    // Upstream PlaCo leaves the profile construction commented out.
    KickTrajectory {
        t_start,
        t_end,
        foot_trajectory: CubicSpline3D::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_trajectory_carries_timing() {
        let k = make_trajectory(
            Side::Left,
            0.0,
            1.0,
            Vector3::zeros(),
            Vector3::new(0.1, 0.0, 0.0),
            nalgebra::Isometry3::identity(),
            &HumanoidParameters::new(),
        );
        assert_eq!(k.t_start, 0.0);
        assert_eq!(k.t_end, 1.0);
    }
}

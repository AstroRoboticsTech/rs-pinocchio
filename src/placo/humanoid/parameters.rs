//! Walk/planning parameters for a humanoid (PlaCo `HumanoidParameters`).

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

use super::side::Side;

/// Step-size clipping mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FootstepClipping {
    /// L2 (ellipsoid) clipping.
    Ellipsoid,
    /// L1 (box) clipping.
    Box,
    /// Conic clipping (default).
    #[default]
    Conic,
}

/// Capabilities and constants for humanoid planning and control.
///
/// Field defaults mirror PlaCo's.
#[derive(Clone, Debug)]
pub struct HumanoidParameters {
    /// Single-support duration [s].
    pub single_support_duration: f64,
    /// Timesteps per single support.
    pub single_support_timesteps: i32,
    /// Double/single support duration ratio.
    pub double_support_ratio: f64,
    /// Start/end double-support ratio.
    pub startend_double_support_ratio: f64,
    /// CoM planning horizon (timesteps).
    pub planned_timesteps: i32,
    /// ZMP margin inside the support polygon [m].
    pub zmp_margin: f64,
    /// Foot rise height while walking [m].
    pub walk_foot_height: f64,
    /// Fraction of the step spent at foot height.
    pub walk_foot_rise_ratio: f64,
    /// Target CoM height while walking [m].
    pub walk_com_height: f64,
    /// Trunk pitch while walking [rad].
    pub walk_trunk_pitch: f64,
    /// Maximum forward step [m].
    pub walk_max_dx_forward: f64,
    /// Maximum backward step [m].
    pub walk_max_dx_backward: f64,
    /// Maximum lateral step [m].
    pub walk_max_dy: f64,
    /// Maximum step yaw [rad].
    pub walk_max_dtheta: f64,
    /// Feet spacing per dtheta [m/rad].
    pub walk_dtheta_spacing: f64,
    /// Lateral spacing between feet [m].
    pub feet_spacing: f64,
    /// Foot width [m].
    pub foot_width: f64,
    /// Foot length [m].
    pub foot_length: f64,
    /// ZMP x target offset in the foot frame [m].
    pub foot_zmp_target_x: f64,
    /// ZMP y target offset in the foot frame [m] (positive = outward).
    pub foot_zmp_target_y: f64,
    /// ZMP reference weight in the solver.
    pub zmp_reference_weight: f64,
}

impl Default for HumanoidParameters {
    fn default() -> Self {
        Self {
            single_support_duration: 1.0,
            single_support_timesteps: 10,
            double_support_ratio: 1.0,
            startend_double_support_ratio: 1.0,
            planned_timesteps: 100,
            zmp_margin: 0.025,
            walk_foot_height: 0.05,
            walk_foot_rise_ratio: 0.2,
            walk_com_height: 0.35,
            walk_trunk_pitch: 0.0,
            walk_max_dx_forward: 0.08,
            walk_max_dx_backward: 0.03,
            walk_max_dy: 0.04,
            walk_max_dtheta: 0.35,
            walk_dtheta_spacing: 0.05,
            feet_spacing: 0.15,
            foot_width: 0.1,
            foot_length: 0.15,
            foot_zmp_target_x: 0.0,
            foot_zmp_target_y: 0.0,
            zmp_reference_weight: 1e-1,
        }
    }
}

impl HumanoidParameters {
    /// A parameter set with PlaCo defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Planning timestep [s].
    pub fn dt(&self) -> f64 {
        self.single_support_duration / self.single_support_timesteps as f64
    }

    /// Timesteps for a double support.
    pub fn double_support_timesteps(&self) -> i32 {
        (self.double_support_ratio * self.single_support_timesteps as f64).round() as i32
    }

    /// Timesteps for a start/end double support.
    pub fn startend_double_support_timesteps(&self) -> i32 {
        (self.startend_double_support_ratio * self.single_support_timesteps as f64).round() as i32
    }

    /// Duration of a double support [s].
    pub fn double_support_duration(&self) -> f64 {
        self.double_support_timesteps() as f64 * self.dt()
    }

    /// Duration of a start/end double support [s].
    pub fn startend_double_support_duration(&self) -> f64 {
        self.startend_double_support_timesteps() as f64 * self.dt()
    }

    /// Whether the walk has (non-zero) double supports.
    pub fn has_double_support(&self) -> bool {
        self.double_support_timesteps() > 0
    }

    fn step_factor(&self, step: &Vector3<f64>) -> Vector3<f64> {
        Vector3::new(
            if step.x >= 0.0 {
                self.walk_max_dx_forward
            } else {
                self.walk_max_dx_backward
            },
            self.walk_max_dy,
            self.walk_max_dtheta,
        )
    }

    fn clip_with_norm(
        &self,
        step: Vector3<f64>,
        norm: impl Fn(&Vector3<f64>) -> f64,
    ) -> Vector3<f64> {
        let factor = self.step_factor(&step);
        let mut s = Vector3::new(step.x / factor.x, step.y / factor.y, step.z / factor.z);
        let n = norm(&s);
        if n > 1.0 {
            s /= n;
        }
        Vector3::new(s.x * factor.x, s.y * factor.y, s.z * factor.z)
    }

    /// Ellipsoid (L2) clipping of a step `(dx, dy, dtheta)`.
    pub fn ellipsoid_clip(&self, step: Vector3<f64>) -> Vector3<f64> {
        self.clip_with_norm(step, |s| s.norm())
    }

    /// Box (L1) clipping of a step.
    pub fn box_clip(&self, step: Vector3<f64>) -> Vector3<f64> {
        self.clip_with_norm(step, |s| s.x.abs() + s.y.abs() + s.z.abs())
    }

    /// Conic clipping of a step.
    pub fn conic_clip(&self, step: Vector3<f64>) -> Vector3<f64> {
        self.clip_with_norm(step, |s| (s.x * s.x + s.y * s.y).sqrt() + s.z.abs())
    }

    /// Frame of the opposite foot at neutral spacing (`feet_spacing`) from
    /// `t_world_foot`, offset by `(d_x, d_y, d_theta)`.
    pub fn opposite_frame(
        &self,
        side: Side,
        t_world_foot: Isometry3<f64>,
        d_x: f64,
        d_y: f64,
        d_theta: f64,
    ) -> Isometry3<f64> {
        self.offset_frame(side, t_world_foot, self.feet_spacing, d_x, d_y, d_theta)
    }

    /// Frame at half spacing (`feet_spacing / 2`) from `t_world_foot`.
    pub fn neutral_frame(
        &self,
        side: Side,
        t_world_foot: Isometry3<f64>,
        d_x: f64,
        d_y: f64,
        d_theta: f64,
    ) -> Isometry3<f64> {
        self.offset_frame(
            side,
            t_world_foot,
            self.feet_spacing / 2.0,
            d_x,
            d_y,
            d_theta,
        )
    }

    fn offset_frame(
        &self,
        side: Side,
        t_world_foot: Isometry3<f64>,
        spacing: f64,
        d_x: f64,
        d_y: f64,
        d_theta: f64,
    ) -> Isometry3<f64> {
        let sign = if side == Side::Left { -1.0 } else { 1.0 };
        let mut frame = t_world_foot * Translation3::new(0.0, sign * spacing, 0.0);
        frame *= Translation3::new(d_x, d_y, 0.0);
        frame *= UnitQuaternion::from_axis_angle(&Vector3::z_axis(), d_theta);
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dt_and_double_support() {
        let p = HumanoidParameters::new();
        assert!((p.dt() - 0.1).abs() < 1e-12);
        assert_eq!(p.double_support_timesteps(), 10);
        assert!(p.has_double_support());
    }

    #[test]
    fn ellipsoid_clip_bounds_step() {
        let p = HumanoidParameters::new();
        // A huge forward step clips onto the ellipsoid boundary.
        let clipped = p.ellipsoid_clip(Vector3::new(10.0, 0.0, 0.0));
        assert!((clipped.x - p.walk_max_dx_forward).abs() < 1e-9);
        // A small step is unchanged.
        let small = Vector3::new(0.01, 0.0, 0.0);
        let c = p.ellipsoid_clip(small);
        assert!((c - small).norm() < 1e-9);
    }

    #[test]
    fn opposite_frame_offsets_laterally() {
        let p = HumanoidParameters::new();
        let f = p.opposite_frame(Side::Left, Isometry3::identity(), 0.0, 0.0, 0.0);
        // Left support -> opposite (right) foot at -feet_spacing in y.
        assert!((f.translation.y + p.feet_spacing).abs() < 1e-9);
    }
}

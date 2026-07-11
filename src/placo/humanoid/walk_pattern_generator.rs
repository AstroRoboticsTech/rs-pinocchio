//! Walk pattern generation (PlaCo `WalkPatternGenerator`).
//!
//! Plans a CoM trajectory (chained LIPMs with the ZMP kept inside each support
//! polygon) together with per-step swing-foot and yaw trajectories, and exposes
//! a time-indexed query API for the resulting walk. Pure Rust — `plan` takes the
//! initial CoM directly, so no robot model is required.
//!
//! This ports the core `plan` + trajectory queries. Adaptive `replan` /
//! `update_supports` (online DCM tracking) are not yet ported.

use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector2, Vector3};

use super::footsteps::Support;
use super::parameters::HumanoidParameters;
use super::side::Side;
use super::swing_foot::SwingFoot;
use crate::error::{Error, Result};
use crate::placo::problem::{in_polygon_xy, ConstraintPriority, Problem};
use crate::placo::tools::{frame_yaw, CubicSpline};

use super::lipm::{Lipm, LipmTrajectory};

/// A single support phase and its associated CoM / swing trajectories.
pub struct TrajectoryPart {
    /// Part start time.
    pub t_start: f64,
    /// Part end time.
    pub t_end: f64,
    /// The support for this part.
    pub support: Support,
    /// The CoM trajectory over this part.
    pub com_trajectory: LipmTrajectory,
    /// The swing-foot trajectory (only for single supports).
    pub swing_trajectory: Option<SwingFoot>,
}

/// A planned walk trajectory: a sequence of [`TrajectoryPart`]s plus foot/trunk
/// yaw splines, queryable by time.
pub struct WalkTrajectory {
    /// Trajectory start time.
    pub t_start: f64,
    /// Trajectory end time.
    pub t_end: f64,
    /// Target CoM height.
    pub com_target_z: f64,
    /// Trunk pitch.
    pub trunk_pitch: f64,
    /// Trunk roll.
    pub trunk_roll: f64,
    /// The support parts.
    pub parts: Vec<TrajectoryPart>,
    left_foot_yaw: CubicSpline,
    right_foot_yaw: CubicSpline,
    trunk_yaw: CubicSpline,
}

fn build_frame(position: Vector3<f64>, yaw: f64) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(position),
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
    )
}

impl WalkTrajectory {
    fn new(com_target_z: f64, t_start: f64, trunk_pitch: f64, trunk_roll: f64) -> Self {
        Self {
            t_start,
            t_end: t_start,
            com_target_z,
            trunk_pitch,
            trunk_roll,
            parts: Vec::new(),
            left_foot_yaw: CubicSpline::new(true),
            right_foot_yaw: CubicSpline::new(true),
            trunk_yaw: CubicSpline::new(true),
        }
    }

    /// Trajectory duration.
    pub fn duration(&self) -> f64 {
        self.t_end - self.t_start
    }

    fn find_part(&self, t: f64) -> &TrajectoryPart {
        let mut low = 0usize;
        let mut high = self.parts.len() - 1;
        while low != high {
            let mid = (low + high) / 2;
            let part = &self.parts[mid];
            if t < part.t_start {
                high = mid;
            } else if t > part.t_end {
                low = mid + 1;
            } else {
                return part;
            }
        }
        &self.parts[low]
    }

    fn foot_yaw_mut(&mut self, side: Side) -> &mut CubicSpline {
        if side == Side::Left {
            &mut self.left_foot_yaw
        } else {
            &mut self.right_foot_yaw
        }
    }

    fn add_supports(&mut self, t: f64, support: &Support) {
        for footstep in &support.footsteps {
            let yaw = frame_yaw(&footstep.frame.rotation.to_rotation_matrix().into_inner());
            self.foot_yaw_mut(footstep.side).add_point(t, yaw, 0.0);
        }
    }

    /// CoM position in the world at time `t`.
    pub fn p_world_com(&self, t: f64) -> Vector3<f64> {
        let p = self.find_part(t).com_trajectory.pos(t);
        Vector3::new(p.x, p.y, self.com_target_z)
    }

    /// CoM velocity in the world at time `t`.
    pub fn v_world_com(&self, t: f64) -> Vector3<f64> {
        let v = self.find_part(t).com_trajectory.vel(t);
        Vector3::new(v.x, v.y, 0.0)
    }

    /// CoM acceleration in the world at time `t`.
    pub fn a_world_com(&self, t: f64) -> Vector3<f64> {
        let a = self.find_part(t).com_trajectory.acc(t);
        Vector3::new(a.x, a.y, 0.0)
    }

    /// CoM jerk in the world at time `t`.
    pub fn j_world_com(&self, t: f64) -> Vector3<f64> {
        let j = self.find_part(t).com_trajectory.jerk(t);
        Vector3::new(j.x, j.y, 0.0)
    }

    /// DCM in the world at time `t`.
    pub fn p_world_dcm(&self, t: f64, omega: f64) -> Vector2<f64> {
        self.p_world_com(t).xy() + self.v_world_com(t).xy() / omega
    }

    /// ZMP in the world at time `t`.
    pub fn p_world_zmp(&self, t: f64, omega: f64) -> Vector2<f64> {
        self.p_world_com(t).xy() - self.a_world_com(t).xy() / (omega * omega)
    }

    /// Whether `side`'s foot is flying (swinging) at time `t`.
    pub fn is_flying(&self, side: Side, t: f64) -> bool {
        let support = &self.find_part(t).support;
        !support.is_both() && support.side() == side.other()
    }

    /// The support side at time `t` (the first footstep's side for double support).
    pub fn support_side(&self, t: f64) -> Side {
        self.find_part(t).support.footsteps[0].side
    }

    /// Whether the support at time `t` is a double support.
    pub fn support_is_both(&self, t: f64) -> bool {
        self.find_part(t).support.is_both()
    }

    /// World placement of `side`'s foot at time `t`.
    pub fn t_world_foot(&mut self, side: Side, t: f64) -> Isometry3<f64> {
        let flying = self.is_flying(side, t);
        // Snapshot the part data we need before borrowing the yaw spline mutably.
        let pos = {
            let part = self.find_part(t);
            if flying {
                part.swing_trajectory.as_ref().map(|s| s.pos(t))
            } else {
                Some(part.support.footstep_frame(side).translation.vector)
            }
        };
        let yaw = self.foot_yaw_mut(side).pos(t);
        build_frame(pos.unwrap_or_else(Vector3::zeros), yaw)
    }

    /// World placement of the left foot at time `t`.
    pub fn t_world_left(&mut self, t: f64) -> Isometry3<f64> {
        self.t_world_foot(Side::Left, t)
    }

    /// World placement of the right foot at time `t`.
    pub fn t_world_right(&mut self, t: f64) -> Isometry3<f64> {
        self.t_world_foot(Side::Right, t)
    }

    /// Trunk yaw in the world at time `t`.
    pub fn yaw_world_trunk(&mut self, t: f64) -> f64 {
        self.trunk_yaw.pos(t)
    }

    /// Trunk orientation in the world at time `t`
    /// (`Rz(yaw) · Ry(pitch) · Rx(roll)`).
    pub fn r_world_trunk(&mut self, t: f64) -> nalgebra::Matrix3<f64> {
        let yaw = self.trunk_yaw.pos(t);
        let rz = nalgebra::Rotation3::from_axis_angle(&Vector3::z_axis(), yaw);
        let ry = nalgebra::Rotation3::from_axis_angle(&Vector3::y_axis(), self.trunk_pitch);
        let rx = nalgebra::Rotation3::from_axis_angle(&Vector3::x_axis(), self.trunk_roll);
        (rz * ry * rx).into_inner()
    }
}

/// Generates walk trajectories from a support sequence.
pub struct WalkPatternGenerator {
    /// Planning parameters.
    pub parameters: HumanoidParameters,
    /// Whether to treat the ZMP / terminal constraints as soft.
    pub soft: bool,
    /// Weight for the soft ZMP-in-support constraint.
    pub zmp_in_support_weight: f64,
    /// Weight for the soft terminal-stop constraint.
    pub stop_end_support_weight: f64,
    omega: f64,
    omega_2: f64,
}

impl WalkPatternGenerator {
    /// Builds a generator for the given parameters.
    pub fn new(parameters: HumanoidParameters) -> Self {
        let omega = Lipm::compute_omega(parameters.walk_com_height);
        Self {
            parameters,
            soft: false,
            zmp_in_support_weight: 1e3,
            stop_end_support_weight: 1e3,
            omega,
            omega_2: omega * omega,
        }
    }

    /// The LIPM natural frequency used by the generator.
    pub fn omega(&self) -> f64 {
        self.omega
    }

    fn support_default_duration(&self, support: &Support) -> f64 {
        if support.is_both() {
            if support.start || support.end {
                self.parameters.startend_double_support_duration()
            } else {
                self.parameters.double_support_duration()
            }
        } else {
            self.parameters.single_support_duration
        }
    }

    fn support_default_timesteps(&self, support: &Support) -> i32 {
        if support.is_both() {
            if support.start || support.end {
                self.parameters.startend_double_support_timesteps()
            } else {
                self.parameters.double_support_timesteps()
            }
        } else {
            self.parameters.single_support_timesteps
        }
    }

    /// Plans a walk trajectory for `supports`, starting from `initial_com_world`.
    pub fn plan(
        &self,
        supports: &mut [Support],
        initial_com_world: Vector3<f64>,
        t_start: f64,
    ) -> Result<WalkTrajectory> {
        if supports.is_empty() {
            return Err(Error::Solver("plan() called with 0 supports".into()));
        }
        let mut trajectory = WalkTrajectory::new(
            self.parameters.walk_com_height,
            t_start,
            self.parameters.walk_trunk_pitch,
            0.0,
        );
        self.plan_com(&mut trajectory, supports, initial_com_world.xy())?;
        self.plan_feet_trajectories(&mut trajectory);
        Ok(trajectory)
    }

    fn plan_com(
        &self,
        trajectory: &mut WalkTrajectory,
        supports: &mut [Support],
        initial_pos: Vector2<f64>,
    ) -> Result<()> {
        let mut problem = Problem::new();
        let mut lipms: Vec<Lipm> = Vec::new();
        let mut part_meta: Vec<(f64, f64, Support)> = Vec::new();
        let mut t = trajectory.t_start;

        for support in supports.iter_mut() {
            let support_duration = (1.0 - support.elapsed_ratio)
                * self.support_default_duration(support)
                * support.time_ratio;
            let lipm_timesteps = ((1.0 - support.elapsed_ratio)
                * self.support_default_timesteps(support) as f64)
                as i32;
            let lipm_timesteps = lipm_timesteps.max(1) as usize;

            if support.t_start < 0.0 {
                support.t_start = t;
            }
            let part_t_start = t;
            t += support_duration;
            let lipm_dt = support_duration / lipm_timesteps as f64;

            let lipm = if (part_t_start - trajectory.t_start).abs() < 1e-12 {
                Lipm::new(
                    &mut problem,
                    lipm_dt,
                    lipm_timesteps,
                    part_t_start,
                    initial_pos,
                    Vector2::zeros(),
                    Vector2::zeros(),
                )
            } else {
                Lipm::from_previous(
                    &mut problem,
                    lipm_dt,
                    lipm_timesteps,
                    part_t_start,
                    lipms.last().unwrap(),
                )
            };
            self.constrain_lipm(&mut problem, &lipm, support);
            lipms.push(lipm);
            part_meta.push((part_t_start, t, support.clone()));
        }
        trajectory.t_end = t;

        problem
            .solve()
            .map_err(|e| Error::Solver(format!("walk CoM plan failed: {e}")))?;

        for (i, (part_t_start, part_t_end, mut support)) in part_meta.into_iter().enumerate() {
            let com_trajectory = lipms[i].trajectory(&problem);
            support.target_world_dcm = com_trajectory.dcm(part_t_end, self.omega);
            trajectory.parts.push(TrajectoryPart {
                t_start: part_t_start,
                t_end: part_t_end,
                support,
                com_trajectory,
                swing_trajectory: None,
            });
        }
        Ok(())
    }

    fn constrain_lipm(&self, problem: &mut Problem, lipm: &Lipm, support: &Support) {
        let polygon = support.support_polygon();
        for timestep in 1..=lipm.timesteps {
            let mut zmp_constraint = in_polygon_xy(
                &lipm.zmp(timestep, self.omega_2),
                &polygon,
                self.parameters.zmp_margin,
            );
            if self.soft {
                zmp_constraint.configure(ConstraintPriority::Soft, self.zmp_in_support_weight);
            }
            problem.add_constraint(zmp_constraint);

            let (x_offset, y_offset) = if support.is_both() {
                (0.0, 0.0)
            } else {
                let y = if support.side() == Side::Left {
                    self.parameters.foot_zmp_target_y
                } else {
                    -self.parameters.foot_zmp_target_y
                };
                (self.parameters.foot_zmp_target_x, y)
            };
            let target_pt = support.frame() * nalgebra::Point3::new(x_offset, y_offset, 0.0);
            let zmp_target = nalgebra::DVector::from_vec(vec![target_pt.x, target_pt.y]);
            problem
                .add_constraint(lipm.zmp(timestep, self.omega_2).equal_vector(zmp_target))
                .configure(
                    ConstraintPriority::Soft,
                    self.parameters.zmp_reference_weight,
                );

            if support.end && timestep == lipm.timesteps {
                let frame = support.frame();
                let pos =
                    nalgebra::DVector::from_vec(vec![frame.translation.x, frame.translation.y]);
                let pri = if self.soft {
                    ConstraintPriority::Soft
                } else {
                    ConstraintPriority::Hard
                };
                problem
                    .add_constraint(lipm.pos(timestep).equal_vector(pos))
                    .configure(pri, self.stop_end_support_weight);
                problem
                    .add_constraint(lipm.vel(timestep).equal_vector(nalgebra::DVector::zeros(2)))
                    .configure(pri, self.stop_end_support_weight);
                problem
                    .add_constraint(lipm.acc(timestep).equal_vector(nalgebra::DVector::zeros(2)))
                    .configure(pri, self.stop_end_support_weight);
            }
        }
    }

    fn plan_feet_trajectories(&self, trajectory: &mut WalkTrajectory) {
        let first_support = trajectory.parts[0].support.clone();
        trajectory.add_supports(trajectory.t_start, &first_support);
        let trunk_yaw0 = frame_yaw(
            &first_support
                .frame()
                .rotation
                .to_rotation_matrix()
                .into_inner(),
        );
        trajectory
            .trunk_yaw
            .add_point(trajectory.t_start, trunk_yaw0, 0.0);

        for i in 0..trajectory.parts.len() {
            if trajectory.parts[i].support.footsteps.len() == 1 {
                self.plan_sgl_support(trajectory, i);
            } else {
                self.plan_dbl_support(trajectory, i);
            }
        }
    }

    fn plan_dbl_support(&self, trajectory: &mut WalkTrajectory, i: usize) {
        let (t_end, support) = {
            let part = &trajectory.parts[i];
            (part.t_end, part.support.clone())
        };
        trajectory.add_supports(t_end, &support);
        let yaw = frame_yaw(&support.frame().rotation.to_rotation_matrix().into_inner());
        trajectory.trunk_yaw.add_point(t_end, yaw, 0.0);
    }

    fn plan_sgl_support(&self, trajectory: &mut WalkTrajectory, i: usize) {
        let part = &trajectory.parts[i];
        let (t_start, t_end) = (part.t_start, part.t_end);
        let support = part.support.clone();
        let flying_side = support.footsteps[0].side.other();
        // Next part's footstep for the flying side is the landing target.
        let t_world_end = trajectory.parts[i + 1].support.footstep_frame(flying_side);

        let virt_duration = self.support_default_duration(&support) * support.time_ratio;
        let start = if i > 0 {
            trajectory.parts[i - 1]
                .support
                .footstep_frame(flying_side)
                .translation
                .vector
        } else {
            // No previous part (first part is single): degenerate start = target.
            t_world_end.translation.vector
        };

        let swing = SwingFoot::make_trajectory(
            t_start,
            t_start + virt_duration,
            self.parameters.walk_foot_height,
            start,
            t_world_end.translation.vector,
        );
        trajectory.parts[i].swing_trajectory = Some(swing);

        let end_yaw = frame_yaw(&t_world_end.rotation.to_rotation_matrix().into_inner());
        trajectory
            .foot_yaw_mut(flying_side)
            .add_point(t_end, end_yaw, 0.0);
        if !self.parameters.has_double_support() {
            trajectory.trunk_yaw.add_point(t_end, end_yaw, 0.0);
        }
        trajectory.add_supports(t_end, &support);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placo::humanoid::{make_supports, FootstepsPlanner, FootstepsPlannerRepetitive};

    #[test]
    fn plans_a_forward_walk() {
        let params = HumanoidParameters::new();

        // Plan a few forward steps and wrap them into supports.
        let mut planner = FootstepsPlannerRepetitive::new(params.clone());
        planner.configure(0.05, 0.0, 0.0, 4);
        let footsteps = planner.plan(
            Side::Left,
            Isometry3::translation(0.0, params.feet_spacing / 2.0, 0.0),
            Isometry3::translation(0.0, -params.feet_spacing / 2.0, 0.0),
        );
        let mut supports = make_supports(&footsteps, 0.0, true, false, true);

        let wpg = WalkPatternGenerator::new(params);
        let initial_com = Vector3::new(0.0, 0.0, wpg.parameters.walk_com_height);
        let mut traj = wpg
            .plan(&mut supports, initial_com, 0.0)
            .expect("plan walk");

        assert!(traj.t_end > traj.t_start);
        assert!(!traj.parts.is_empty());

        // The CoM trajectory is finite over the whole horizon.
        let mut t = traj.t_start;
        while t <= traj.t_end {
            let com = traj.p_world_com(t);
            assert!(com.iter().all(|v| v.is_finite()));
            assert!((com.z - wpg.parameters.walk_com_height).abs() < 1e-9);
            let zmp = traj.p_world_zmp(t, wpg.omega());
            assert!(zmp.iter().all(|v| v.is_finite()));
            t += 0.05;
        }

        // The CoM advances forward over the walk.
        assert!(traj.p_world_com(traj.t_end).x > traj.p_world_com(traj.t_start).x);

        // Feet frames are finite.
        let lf = traj.t_world_left(traj.t_start);
        let rf = traj.t_world_right(traj.t_start);
        assert!(lf.translation.vector.iter().all(|v| v.is_finite()));
        assert!(rf.translation.vector.iter().all(|v| v.is_finite()));
    }
}

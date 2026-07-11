//! Footstep and support planning (PlaCo `FootstepsPlanner`, `Footstep`,
//! `Support`, and the naive / repetitive planners).

use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector2, Vector3};

use super::parameters::{FootstepClipping, HumanoidParameters};
use super::side::Side;
use crate::placo::tools::{frame_yaw, interpolate_frames};

/// A footstep: a foot placed on the ground.
#[derive(Clone, Debug)]
pub struct Footstep {
    /// Foot width [m].
    pub foot_width: f64,
    /// Foot length [m].
    pub foot_length: f64,
    /// Which foot.
    pub side: Side,
    /// World placement of the foot.
    pub frame: Isometry3<f64>,
}

impl Footstep {
    /// A footstep with the given foot dimensions (side/frame set by the planner).
    pub fn new(foot_width: f64, foot_length: f64) -> Self {
        Self {
            foot_width,
            foot_length,
            side: Side::Left,
            frame: Isometry3::identity(),
        }
    }

    /// The four corners of the foot rectangle (clockwise), inflated by `margin`.
    pub fn compute_polygon(&self, margin: f64) -> Vec<Vector2<f64>> {
        let hx = margin + self.foot_length / 2.0;
        let hy = margin + self.foot_width / 2.0;
        [(-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
            .iter()
            .map(|(sx, sy)| {
                let c = self.frame * Point3::new(sx * hx, sy * hy, 0.0);
                Vector2::new(c.x, c.y)
            })
            .collect()
    }

    /// The (zero-margin) support polygon.
    pub fn support_polygon(&self) -> Vec<Vector2<f64>> {
        self.compute_polygon(0.0)
    }

    /// Whether `point` is inside the clockwise `polygon`.
    pub fn polygon_contains(polygon: &[Vector2<f64>], point: Vector2<f64>) -> bool {
        let mut last = polygon[polygon.len() - 1];
        for &current in polygon {
            let v = current - last;
            let n = Vector2::new(v.y, -v.x);
            if n.dot(&(point - last)) < 0.0 {
                return false;
            }
            last = current;
        }
        true
    }

    /// Whether this footstep's (margin-inflated) polygon overlaps `other`'s.
    pub fn overlap(&self, other: &Footstep, margin: f64) -> bool {
        let a = self.compute_polygon(margin);
        let b = other.compute_polygon(margin);
        a.iter().any(|&p| Footstep::polygon_contains(&b, p))
            || b.iter().any(|&p| Footstep::polygon_contains(&a, p))
    }
}

/// A support: one foot (single support) or two (double support) on the ground.
#[derive(Clone, Debug)]
pub struct Support {
    /// The footsteps making up this support.
    pub footsteps: Vec<Footstep>,
    /// Time the support starts (`-1` = uninitialized).
    pub t_start: f64,
    /// Elapsed ratio of the support phase (0..1).
    pub elapsed_ratio: f64,
    /// Time ratio of the remaining part of the phase.
    pub time_ratio: f64,
    /// Target world DCM at the end of the phase.
    pub target_world_dcm: Vector2<f64>,
    /// Whether this is the initial support.
    pub start: bool,
    /// Whether this is the final support.
    pub end: bool,
}

impl Support {
    /// A support from a set of footsteps.
    pub fn new(footsteps: Vec<Footstep>) -> Self {
        Self {
            footsteps,
            t_start: -1.0,
            elapsed_ratio: 0.0,
            time_ratio: 1.0,
            target_world_dcm: Vector2::zeros(),
            start: false,
            end: false,
        }
    }

    /// The convex-hull support polygon (clockwise) of all footsteps.
    pub fn support_polygon(&self) -> Vec<Vector2<f64>> {
        let mut points = Vec::new();
        for f in &self.footsteps {
            points.extend(f.support_polygon());
        }
        convex_hull_clockwise(&points)
    }

    /// The (interpolated average) support frame.
    pub fn frame(&self) -> Isometry3<f64> {
        let mut f = self.footsteps[0].frame;
        for (i, footstep) in self.footsteps.iter().enumerate().skip(1) {
            let n = (i + 1) as f64;
            f = interpolate_frames(&f, &footstep.frame, 1.0 / n);
        }
        f
    }

    /// The frame of the footstep on `side`.
    ///
    /// # Panics
    /// If no footstep on that side is present.
    pub fn footstep_frame(&self, side: Side) -> Isometry3<f64> {
        self.footsteps
            .iter()
            .find(|f| f.side == side)
            .map(|f| f.frame)
            .expect("Support: asked for a frame that doesn't exist")
    }

    /// Translates all footsteps by a 2D offset.
    pub fn apply_offset(&mut self, offset: Vector2<f64>) {
        for f in &mut self.footsteps {
            f.frame = Translation3::new(offset.x, offset.y, 0.0) * f.frame;
        }
    }

    /// The single-support side.
    ///
    /// # Panics
    /// If this is a double support (check [`Support::is_both`] first).
    pub fn side(&self) -> Side {
        if self.footsteps.len() > 1 {
            panic!("Support: side() called on a double support");
        }
        self.footsteps[0].side
    }

    /// Whether this is a double support.
    pub fn is_both(&self) -> bool {
        self.footsteps.len() == 2
    }
}

/// Groups footsteps into a support sequence (initial/final double supports, and
/// optional double supports between steps). Mirrors PlaCo's `make_supports`.
pub fn make_supports(
    footsteps: &[Footstep],
    t_start: f64,
    start: bool,
    middle: bool,
    end: bool,
) -> Vec<Support> {
    let mut supports = Vec::new();

    if footsteps.len() > 2 {
        if start {
            let mut s = Support::new(vec![footsteps[0].clone(), footsteps[1].clone()]);
            s.start = true;
            s.t_start = t_start;
            supports.push(s);
        } else {
            let mut s = Support::new(vec![footsteps[0].clone()]);
            s.t_start = t_start;
            supports.push(s);
            if middle {
                let mut ds = Support::new(vec![footsteps[0].clone(), footsteps[1].clone()]);
                ds.t_start = t_start;
                supports.push(ds);
            }
        }

        for step in 1..footsteps.len() - 1 {
            supports.push(Support::new(vec![footsteps[step].clone()]));
            let is_end = step == footsteps.len() - 2;
            if !is_end && middle {
                supports.push(Support::new(vec![
                    footsteps[step].clone(),
                    footsteps[step + 1].clone(),
                ]));
            }
        }
    }

    if end {
        let n = footsteps.len();
        let mut s = Support::new(vec![footsteps[n - 2].clone(), footsteps[n - 1].clone()]);
        s.end = true;
        supports.push(s);
    }

    supports
}

/// Common footstep-planner behavior (PlaCo `FootstepsPlanner`).
pub trait FootstepsPlanner {
    /// The planning parameters.
    fn parameters(&self) -> &HumanoidParameters;
    /// The clipping mode used by [`FootstepsPlanner::clipped_opposite_footstep`].
    fn footstep_clipping(&self) -> FootstepClipping;
    /// The planner name.
    fn name(&self) -> &'static str;
    /// Planner-specific footstep generation (appends to `footsteps`).
    fn plan_impl(
        &self,
        footsteps: &mut Vec<Footstep>,
        flying_side: Side,
        t_world_left: Isometry3<f64>,
        t_world_right: Isometry3<f64>,
    );

    /// Plans the full footstep sequence (starting with the two initial feet).
    fn plan(
        &self,
        flying_side: Side,
        t_world_left: Isometry3<f64>,
        t_world_right: Isometry3<f64>,
    ) -> Vec<Footstep> {
        let mut footsteps = Vec::new();
        let frame_of = |s: Side| {
            if s == Side::Left {
                t_world_left
            } else {
                t_world_right
            }
        };

        footsteps.push(self.create_footstep(flying_side, frame_of(flying_side)));
        let other = flying_side.other();
        footsteps.push(self.create_footstep(other, frame_of(other)));

        self.plan_impl(&mut footsteps, flying_side, t_world_left, t_world_right);
        footsteps
    }

    /// Builds a footstep for `side` at `t_world_foot` (with the model foot dims).
    fn create_footstep(&self, side: Side, t_world_foot: Isometry3<f64>) -> Footstep {
        let p = self.parameters();
        let mut f = Footstep::new(p.foot_width, p.foot_length);
        f.side = side;
        f.frame = t_world_foot;
        f
    }

    /// The opposite footstep at neutral spacing, offset by `(d_x, d_y, d_theta)`.
    fn opposite_footstep(&self, footstep: &Footstep, d_x: f64, d_y: f64, d_theta: f64) -> Footstep {
        let mut f = footstep.clone();
        f.frame =
            self.parameters()
                .opposite_frame(footstep.side, footstep.frame, d_x, d_y, d_theta);
        f.side = footstep.side.other();
        f
    }

    /// Like [`FootstepsPlanner::opposite_footstep`] but clipped and made
    /// non-overlapping.
    fn clipped_opposite_footstep(
        &self,
        footstep: &Footstep,
        d_x: f64,
        d_y: f64,
        d_theta: f64,
    ) -> Footstep {
        let p = self.parameters();
        let mut step = Vector3::new(d_x, d_y, d_theta);
        // dtheta spacing bias.
        if footstep.side == Side::Left {
            step.y -= p.walk_dtheta_spacing * step.z.abs();
        } else {
            step.y += p.walk_dtheta_spacing * step.z.abs();
        }
        step = match self.footstep_clipping() {
            FootstepClipping::Conic => p.conic_clip(step),
            FootstepClipping::Ellipsoid => p.ellipsoid_clip(step),
            FootstepClipping::Box => p.box_clip(step),
        };

        for _ in 0..32 {
            let new_footstep = self.opposite_footstep(footstep, step.x, step.y, step.z);
            if new_footstep.overlap(footstep, 1e-2) {
                step *= 0.9;
            } else {
                return new_footstep;
            }
        }
        self.opposite_footstep(footstep, step.x, step.y, step.z)
    }
}

/// Plans footsteps towards target left/right foot placements (PlaCo
/// `FootstepsPlannerNaive`).
pub struct FootstepsPlannerNaive {
    /// Planning parameters.
    pub parameters: HumanoidParameters,
    /// Clipping mode.
    pub footstep_clipping: FootstepClipping,
    /// Target left-foot placement.
    pub target_left: Isometry3<f64>,
    /// Target right-foot placement.
    pub target_right: Isometry3<f64>,
    /// Accessibility window length [m].
    pub accessibility_length: f64,
    /// Accessibility window width [m].
    pub accessibility_width: f64,
    /// Accessibility window yaw [rad].
    pub accessibility_yaw: f64,
    /// Distance threshold below which yaw tracks the target orientation [m].
    pub place_threshold: f64,
    /// Maximum number of planned steps.
    pub max_steps: i32,
}

impl FootstepsPlannerNaive {
    /// Builds a naive planner from `parameters` (accessibility from the walk
    /// limits).
    pub fn new(parameters: HumanoidParameters) -> Self {
        let (l, w, y) = (
            parameters.walk_max_dx_forward,
            parameters.walk_max_dy,
            parameters.walk_max_dtheta,
        );
        Self {
            parameters,
            footstep_clipping: FootstepClipping::Conic,
            target_left: Isometry3::identity(),
            target_right: Isometry3::identity(),
            accessibility_length: l,
            accessibility_width: w,
            accessibility_yaw: y,
            place_threshold: 0.5,
            max_steps: 100,
        }
    }

    /// Sets the target foot placements.
    pub fn configure(&mut self, target_left: Isometry3<f64>, target_right: Isometry3<f64>) {
        self.target_left = target_left;
        self.target_right = target_right;
    }
}

impl FootstepsPlanner for FootstepsPlannerNaive {
    fn parameters(&self) -> &HumanoidParameters {
        &self.parameters
    }
    fn footstep_clipping(&self) -> FootstepClipping {
        self.footstep_clipping
    }
    fn name(&self) -> &'static str {
        "naive"
    }
    fn plan_impl(
        &self,
        footsteps: &mut Vec<Footstep>,
        flying_side: Side,
        t_world_left: Isometry3<f64>,
        t_world_right: Isometry3<f64>,
    ) {
        let p = &self.parameters;
        let t_world_target = interpolate_frames(&self.target_left, &self.target_right, 0.5);

        let mut cur_left = t_world_left;
        let mut cur_right = t_world_right;
        let mut support_side = flying_side.other();
        let (mut left_arrived, mut right_arrived) = (false, false);
        let mut steps = 0;

        while (!left_arrived || !right_arrived) && steps < self.max_steps {
            steps += 1;
            let mut arrived = true;

            let t_world_support = if support_side == Side::Left {
                cur_left
            } else {
                cur_right
            };
            let idle_y = if support_side == Side::Left {
                -p.feet_spacing
            } else {
                p.feet_spacing
            };
            let center_y = idle_y / 2.0;

            let target_foot = if support_side == Side::Left {
                self.target_right
            } else {
                self.target_left
            };
            let mut t_support_target = t_world_support.inverse() * target_foot;
            t_support_target.translation.z = 0.0;

            let error0 = t_support_target.translation.vector - Vector3::new(0.0, idle_y, 0.0);
            let mut rescale = 1.0_f64;
            if error0.x < -self.accessibility_length {
                rescale = rescale.min(-self.accessibility_length / error0.x);
                arrived = false;
            }
            if error0.x > self.accessibility_length {
                rescale = rescale.min(self.accessibility_length / error0.x);
                arrived = false;
            }
            if error0.y < -self.accessibility_width {
                rescale = rescale.min(-self.accessibility_width / error0.y);
                arrived = false;
            }
            if error0.y > self.accessibility_width {
                rescale = rescale.min(self.accessibility_width / error0.y);
                arrived = false;
            }
            let dist = error0.norm();
            let error = error0 * rescale;

            let mut error_yaw = if dist > self.place_threshold {
                let target_to_center = (t_world_support.inverse() * t_world_target)
                    .translation
                    .vector
                    - Vector3::new(0.0, center_y, 0.0);
                target_to_center.y.atan2(target_to_center.x)
            } else {
                frame_yaw(&t_support_target.rotation.to_rotation_matrix().into_inner())
            };
            if error_yaw < -self.accessibility_yaw {
                arrived = false;
                error_yaw = -self.accessibility_yaw;
            }
            if error_yaw > self.accessibility_yaw {
                arrived = false;
                error_yaw = self.accessibility_yaw;
            }

            let step = p.ellipsoid_clip(Vector3::new(error.x, error.y, error_yaw));
            let new_step = Isometry3::from_parts(
                Translation3::new(step.x, idle_y + step.y, 0.0),
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), step.z),
            );

            let footstep = self.create_footstep(support_side.other(), t_world_support * new_step);
            let frame = footstep.frame;
            footsteps.push(footstep);

            if support_side == Side::Left {
                right_arrived = arrived;
                cur_right = frame;
                support_side = Side::Right;
            } else {
                left_arrived = arrived;
                cur_left = frame;
                support_side = Side::Left;
            }
        }
    }
}

/// Plans a fixed number of repeated steps of a given size (PlaCo
/// `FootstepsPlannerRepetitive`).
pub struct FootstepsPlannerRepetitive {
    /// Planning parameters.
    pub parameters: HumanoidParameters,
    /// Clipping mode.
    pub footstep_clipping: FootstepClipping,
    d_x: f64,
    d_y: f64,
    d_theta: f64,
    nb_steps: i32,
}

impl FootstepsPlannerRepetitive {
    /// Builds a repetitive planner.
    pub fn new(parameters: HumanoidParameters) -> Self {
        Self {
            parameters,
            footstep_clipping: FootstepClipping::Conic,
            d_x: 0.0,
            d_y: 0.0,
            d_theta: 0.0,
            nb_steps: 0,
        }
    }

    /// Configures the per-step motion `(x, y, theta)` and step count.
    pub fn configure(&mut self, x: f64, y: f64, theta: f64, steps: i32) {
        self.d_x = x;
        self.d_y = y;
        self.d_theta = theta;
        self.nb_steps = steps;
    }
}

impl FootstepsPlanner for FootstepsPlannerRepetitive {
    fn parameters(&self) -> &HumanoidParameters {
        &self.parameters
    }
    fn footstep_clipping(&self) -> FootstepClipping {
        self.footstep_clipping
    }
    fn name(&self) -> &'static str {
        "repetitive"
    }
    fn plan_impl(
        &self,
        footsteps: &mut Vec<Footstep>,
        _flying_side: Side,
        _t_world_left: Isometry3<f64>,
        _t_world_right: Isometry3<f64>,
    ) {
        let mut footstep = footsteps[1].clone();
        if self.nb_steps > 0 {
            for _ in 0..self.nb_steps - 1 {
                footstep =
                    self.clipped_opposite_footstep(&footstep, self.d_x, self.d_y, self.d_theta);
                footsteps.push(footstep.clone());
            }
            // Final footstep to return to double support.
            footsteps.push(self.clipped_opposite_footstep(&footstep, 0.0, 0.0, 0.0));
        }
    }
}

/// Clockwise convex hull (Andrew's monotone chain, then reversed to clockwise).
fn convex_hull_clockwise(points: &[Vector2<f64>]) -> Vec<Vector2<f64>> {
    let mut pts: Vec<Vector2<f64>> = points.to_vec();
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap()
            .then(a.y.partial_cmp(&b.y).unwrap())
    });
    pts.dedup_by(|a, b| (a.x - b.x).abs() < 1e-12 && (a.y - b.y).abs() < 1e-12);
    if pts.len() < 3 {
        return pts;
    }

    let cross = |o: &Vector2<f64>, a: &Vector2<f64>, b: &Vector2<f64>| {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    };

    let mut lower: Vec<Vector2<f64>> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], &p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Vector2<f64>> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], &p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper); // counter-clockwise hull
    lower.reverse(); // clockwise
    lower
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> HumanoidParameters {
        HumanoidParameters::new()
    }

    #[test]
    fn footstep_polygon_and_containment() {
        let mut f = Footstep::new(0.1, 0.15);
        f.frame = Isometry3::identity();
        let poly = f.support_polygon();
        assert_eq!(poly.len(), 4);
        // Center is inside; a far point is outside.
        assert!(Footstep::polygon_contains(&poly, Vector2::new(0.0, 0.0)));
        assert!(!Footstep::polygon_contains(&poly, Vector2::new(1.0, 0.0)));
    }

    #[test]
    fn overlapping_footsteps_detected() {
        let a = {
            let mut f = Footstep::new(0.1, 0.15);
            f.frame = Isometry3::identity();
            f
        };
        let mut b = a.clone();
        b.frame = Isometry3::translation(0.02, 0.0, 0.0); // heavily overlapping
        assert!(a.overlap(&b, 0.0));
        let mut c = a.clone();
        c.frame = Isometry3::translation(1.0, 0.0, 0.0); // far apart
        assert!(!a.overlap(&c, 0.0));
    }

    #[test]
    fn repetitive_planner_generates_forward_steps() {
        let mut planner = FootstepsPlannerRepetitive::new(params());
        planner.configure(0.05, 0.0, 0.0, 4);
        let left = Isometry3::translation(0.0, 0.075, 0.0);
        let right = Isometry3::translation(0.0, -0.075, 0.0);
        let footsteps = planner.plan(Side::Left, left, right);
        // 2 initial + (nb_steps-1) + 1 final = 2 + 3 + 1 = 6.
        assert_eq!(footsteps.len(), 6);
        // Steps alternate sides and advance forward.
        assert!(footsteps.last().unwrap().frame.translation.x > footsteps[1].frame.translation.x);
        for w in footsteps.windows(2) {
            assert_ne!(w[0].side, w[1].side);
        }
    }

    #[test]
    fn make_supports_wraps_with_double_supports() {
        let mut planner = FootstepsPlannerRepetitive::new(params());
        planner.configure(0.05, 0.0, 0.0, 4);
        let footsteps = planner.plan(
            Side::Left,
            Isometry3::translation(0.0, 0.075, 0.0),
            Isometry3::translation(0.0, -0.075, 0.0),
        );
        let supports = make_supports(&footsteps, 0.0, true, false, true);
        // First and last are double supports.
        assert!(supports[0].is_both());
        assert!(supports[0].start);
        assert!(supports.last().unwrap().is_both());
        assert!(supports.last().unwrap().end);
    }

    #[test]
    fn support_polygon_is_convex_hull() {
        let mut planner = FootstepsPlannerRepetitive::new(params());
        planner.configure(0.05, 0.0, 0.0, 4);
        let footsteps = planner.plan(
            Side::Left,
            Isometry3::translation(0.0, 0.075, 0.0),
            Isometry3::translation(0.0, -0.075, 0.0),
        );
        let double = Support::new(vec![footsteps[0].clone(), footsteps[1].clone()]);
        let hull = double.support_polygon();
        assert!(hull.len() >= 4);
        // Every foot corner lies inside (or on) the hull.
        for f in &double.footsteps {
            for corner in f.support_polygon() {
                assert!(Footstep::polygon_contains(&hull, corner));
            }
        }
    }
}

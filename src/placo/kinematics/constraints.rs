//! Kinematics constraints (PlaCo `kinematics::*Constraint`).
//!
//! Unlike tasks, a constraint injects its own inequalities directly into the QP
//! over the joint-velocity variable `qd`.

use std::any::Any;

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector2, Vector3};

use crate::error::Result;
use crate::placo::model::RobotWrapper;
use crate::placo::problem::{in_polygon_xy, ConstraintPriority, Expression, Problem, Variable};
use crate::ReferenceFrame;

fn dmat3(m: &Matrix3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 3, m.as_slice())
}

fn skew(v: &Vector3<f64>) -> DMatrix<f64> {
    DMatrix::from_row_slice(3, 3, &[0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0])
}

/// A constraint added to the [`super::KinematicsSolver`] QP.
pub trait KinematicsConstraint: Any {
    /// Adds the constraint's inequalities to `problem` over the `qd` variable.
    fn add_to(
        &self,
        problem: &mut Problem,
        qd: Variable,
        robot: &mut RobotWrapper,
        dt: f64,
    ) -> Result<()>;
    /// Sets the constraint priority (hard/soft) and soft weight.
    fn set_priority_weight(&mut self, priority: ConstraintPriority, weight: f64);
    /// Downcast hook for typed reconfiguration.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// Shared hard/soft config with a helper to add a built expression.
#[derive(Clone, Copy, Debug)]
struct Config {
    priority: ConstraintPriority,
    weight: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            priority: ConstraintPriority::Hard,
            weight: 1.0,
        }
    }
}

impl Config {
    fn add(&self, problem: &mut Problem, mut constraint: crate::placo::problem::Constraint) {
        constraint.configure(self.priority, self.weight);
        problem.add_constraint(constraint);
    }
}

/// Keeps the z-axes of two frames within a cone of half-angle `angle_max`
/// (PlaCo `ConeConstraint`). The cone is discretized into `n` slices.
pub struct ConeConstraint {
    config: Config,
    /// Frame `a`.
    pub frame_a: usize,
    /// Frame `b`.
    pub frame_b: usize,
    /// Maximum half-angle between the z-axes.
    pub angle_max: f64,
    /// Number of discretization slices.
    pub n: usize,
    /// Discretization range around the current orientation (radians).
    pub range: f64,
}

impl ConeConstraint {
    pub(crate) fn new(frame_a: usize, frame_b: usize, angle_max: f64) -> Self {
        Self {
            config: Config::default(),
            frame_a,
            frame_b,
            angle_max,
            n: 8,
            range: 0.25,
        }
    }
}

impl KinematicsConstraint for ConeConstraint {
    fn set_priority_weight(&mut self, priority: ConstraintPriority, weight: f64) {
        self.config = Config { priority, weight };
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn add_to(
        &self,
        problem: &mut Problem,
        _qd: Variable,
        robot: &mut RobotWrapper,
        _dt: f64,
    ) -> Result<()> {
        let t_a_b = robot.t_a_b(self.frame_a, self.frame_b)?;
        let r_ab = t_a_b.rotation.to_rotation_matrix().into_inner();
        let axis_cone = r_ab * Vector3::z();

        let jb = robot.frame_jacobian(self.frame_b, ReferenceFrame::Local)?;
        let ja = robot.frame_jacobian(self.frame_a, ReferenceFrame::Local)?;
        let j_cone = dmat3(&r_ab) * jb.rows(3, 3).into_owned() - ja.rows(3, 3).into_owned();

        let n = self.n;
        let nv = robot.nv();
        let mut a = DMatrix::zeros(n, nv);
        let mut b = DVector::zeros(n);
        let slice_alpha_offset = axis_cone.y.atan2(axis_cone.x);

        for k in 0..n {
            let slice_alpha =
                slice_alpha_offset + (k as f64 * 2.0 * self.range / n as f64) - self.range;
            let r_cone_slice =
                Rotation3::from_axis_angle(&Vector3::z_axis(), slice_alpha).into_inner();
            let axis_slice = r_cone_slice.transpose() * axis_cone;
            let alpha = axis_slice.x.atan2(axis_slice.z);
            let rotation_axis = r_cone_slice.column(1).into_owned();
            let j_slice = DMatrix::from_row_slice(1, 3, rotation_axis.as_slice()) * &j_cone;
            a.row_mut(k).copy_from(&j_slice.row(0));
            b[k] = alpha;
        }

        self.config
            .add(problem, Expression { a, b }.leq_scalar(self.angle_max));
        Ok(())
    }
}

/// Bounds the relative yaw between two frames to `± angle_max` (PlaCo
/// `YawConstraint`).
pub struct YawConstraint {
    config: Config,
    /// Frame `a`.
    pub frame_a: usize,
    /// Frame `b`.
    pub frame_b: usize,
    /// Maximum absolute yaw angle.
    pub angle_max: f64,
}

impl YawConstraint {
    pub(crate) fn new(frame_a: usize, frame_b: usize, angle_max: f64) -> Self {
        Self {
            config: Config::default(),
            frame_a,
            frame_b,
            angle_max,
        }
    }
}

impl KinematicsConstraint for YawConstraint {
    fn set_priority_weight(&mut self, priority: ConstraintPriority, weight: f64) {
        self.config = Config { priority, weight };
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn add_to(
        &self,
        problem: &mut Problem,
        _qd: Variable,
        robot: &mut RobotWrapper,
        _dt: f64,
    ) -> Result<()> {
        let t_a_b = robot.t_a_b(self.frame_a, self.frame_b)?;
        let r_ab = t_a_b.rotation.to_rotation_matrix().into_inner();
        let jb = robot.frame_jacobian(self.frame_b, ReferenceFrame::Local)?;
        let ja = robot.frame_jacobian(self.frame_a, ReferenceFrame::Local)?;
        let j_relative = dmat3(&r_ab) * jb.rows(3, 3).into_owned() - ja.rows(3, 3).into_owned();

        let x_axis = r_ab.column(0).into_owned();
        let alpha = x_axis.y.atan2(x_axis.x);
        let perp_axis = Vector3::z().cross(&x_axis).normalize();
        let yaw_axis = x_axis.cross(&perp_axis).normalize();
        let j_angle = DMatrix::from_row_slice(1, 3, yaw_axis.as_slice()) * &j_relative; // 1 x nv

        // |alpha + J·qd| <= angle_max  ->  two rows.
        let nv = robot.nv();
        let mut a = DMatrix::zeros(2, nv);
        a.row_mut(0).copy_from(&j_angle.row(0));
        a.row_mut(1).copy_from(&(-&j_angle).row(0));
        let b = DVector::from_vec(vec![alpha, -alpha]);

        self.config
            .add(problem, Expression { a, b }.leq_scalar(self.angle_max));
        Ok(())
    }
}

/// Bounds the distance between two frames to at most `distance_max` (PlaCo
/// `DistanceConstraint`).
pub struct DistanceConstraint {
    config: Config,
    /// Frame `a`.
    pub frame_a: usize,
    /// Frame `b`.
    pub frame_b: usize,
    /// Maximum distance.
    pub distance_max: f64,
}

impl DistanceConstraint {
    pub(crate) fn new(frame_a: usize, frame_b: usize, distance_max: f64) -> Self {
        Self {
            config: Config::default(),
            frame_a,
            frame_b,
            distance_max,
        }
    }
}

impl KinematicsConstraint for DistanceConstraint {
    fn set_priority_weight(&mut self, priority: ConstraintPriority, weight: f64) {
        self.config = Config { priority, weight };
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn add_to(
        &self,
        problem: &mut Problem,
        _qd: Variable,
        robot: &mut RobotWrapper,
        _dt: f64,
    ) -> Result<()> {
        let ta = robot.t_world_frame(self.frame_a)?;
        let tb = robot.t_world_frame(self.frame_b)?;
        let ab = tb.translation.vector - ta.translation.vector;
        let distance = ab.norm();
        let direction = ab.normalize();

        let ja = robot.frame_jacobian(self.frame_a, ReferenceFrame::LocalWorldAligned)?;
        let jb = robot.frame_jacobian(self.frame_b, ReferenceFrame::LocalWorldAligned)?;
        let diff = jb.rows(0, 3).into_owned() - ja.rows(0, 3).into_owned();
        let a = DMatrix::from_row_slice(1, 3, direction.as_slice()) * diff;
        let b = DVector::from_element(1, distance);

        self.config
            .add(problem, Expression { a, b }.leq_scalar(self.distance_max));
        Ok(())
    }
}

/// Keeps the CoM (xy) inside a clockwise polygon with a margin (PlaCo
/// `CoMPolygonConstraint`).
pub struct CoMPolygonConstraint {
    config: Config,
    /// Clockwise polygon vertices.
    pub polygon: Vec<Vector2<f64>>,
    /// Inward margin.
    pub margin: f64,
    /// Constrain the DCM instead of the CoM (needs [`Self::omega`] and `solver.dt`).
    pub dcm: bool,
    /// LIPM natural frequency `sqrt(g/h)`, used when [`Self::dcm`] is set.
    pub omega: f64,
}

impl CoMPolygonConstraint {
    pub(crate) fn new(polygon: Vec<Vector2<f64>>, margin: f64) -> Self {
        Self {
            config: Config::default(),
            polygon,
            margin,
            dcm: false,
            omega: 0.0,
        }
    }
}

impl KinematicsConstraint for CoMPolygonConstraint {
    fn set_priority_weight(&mut self, priority: ConstraintPriority, weight: f64) {
        self.config = Config { priority, weight };
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn add_to(
        &self,
        problem: &mut Problem,
        _qd: Variable,
        robot: &mut RobotWrapper,
        dt: f64,
    ) -> Result<()> {
        let com = robot.com_world()?;
        let jac = robot.com_jacobian()?;
        let mut a = jac.rows(0, 2).into_owned();
        if self.dcm {
            // Future DCM = c + J·dq·(1 + 1/(dt·omega)).
            if dt == 0.0 || self.omega == 0.0 {
                return Err(crate::error::Error::Solver(
                    "CoMPolygonConstraint DCM mode needs a non-zero solver.dt and omega".into(),
                ));
            }
            a *= 1.0 + 1.0 / (dt * self.omega);
        }
        let com_xy = Expression {
            a,
            b: DVector::from_vec(vec![com.x, com.y]),
        };
        let mut constraint = in_polygon_xy(&com_xy, &self.polygon, self.margin);
        constraint.configure(self.config.priority, self.config.weight);
        problem.add_constraint(constraint);
        Ok(())
    }
}

/// A pairwise nearest-point distance between two robot bodies, used by
/// [`AvoidSelfCollisionsConstraint`]. Mirrors one entry of PlaCo's
/// `RobotWrapper::distances()` (which is produced by the coal collision backend).
#[derive(Clone, Debug)]
pub struct CollisionDistance {
    /// Frame index of body A.
    pub frame_a: usize,
    /// Frame index of body B.
    pub frame_b: usize,
    /// Nearest point on body A, in world coordinates.
    pub point_a: Vector3<f64>,
    /// Nearest point on body B, in world coordinates.
    pub point_b: Vector3<f64>,
    /// Signed minimum distance (negative on interpenetration).
    pub min_distance: f64,
}

/// Keeps robot bodies from colliding with each other (PlaCo
/// `AvoidSelfCollisionsConstraint`).
///
/// For each supplied [`CollisionDistance`] closer than `self_collisions_trigger`,
/// it adds `n·(J_B − J_A)·qd + d ≥ self_collisions_margin`, where `n` is the unit
/// vector between the nearest points (from A to B, flipped on interpenetration)
/// and `J_A`/`J_B` are the world-aligned point-velocity Jacobians at those
/// nearest points — i.e. the relative separation velocity must keep the bodies
/// at least the margin apart.
///
/// The nearest-point [`distances`](Self::distances) are supplied by the caller
/// (from their collision backend); the raw coal geometry query is a Pinocchio
/// concern that is not exposed by the current binding.
pub struct AvoidSelfCollisionsConstraint {
    config: Config,
    /// Margin kept between bodies [m].
    pub self_collisions_margin: f64,
    /// Distance below which a pair is constrained [m].
    pub self_collisions_trigger: f64,
    /// The pairwise nearest-point distances to enforce.
    pub distances: Vec<CollisionDistance>,
}

impl AvoidSelfCollisionsConstraint {
    pub(crate) fn new() -> Self {
        Self {
            config: Config::default(),
            self_collisions_margin: 0.005,
            self_collisions_trigger: 0.01,
            distances: Vec::new(),
        }
    }

    // World-aligned point-velocity Jacobian at `point` on `frame`
    // (`J_lin − skew(point − origin)·J_ang`).
    fn point_jacobian(
        robot: &RobotWrapper,
        frame: usize,
        point: &Vector3<f64>,
    ) -> Result<DMatrix<f64>> {
        let j = robot.frame_jacobian(frame, ReferenceFrame::LocalWorldAligned)?;
        let origin = robot.t_world_frame(frame)?.translation.vector;
        let r = point - origin;
        Ok(j.rows(0, 3).into_owned() - skew(&r) * j.rows(3, 3).into_owned())
    }
}

impl KinematicsConstraint for AvoidSelfCollisionsConstraint {
    fn set_priority_weight(&mut self, priority: ConstraintPriority, weight: f64) {
        self.config = Config { priority, weight };
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn add_to(
        &self,
        problem: &mut Problem,
        _qd: Variable,
        robot: &mut RobotWrapper,
        _dt: f64,
    ) -> Result<()> {
        let active: Vec<&CollisionDistance> = self
            .distances
            .iter()
            .filter(|d| d.min_distance < self.self_collisions_trigger)
            .collect();
        if active.is_empty() {
            return Ok(());
        }

        let nv = robot.nv();
        let mut a = DMatrix::zeros(active.len(), nv);
        let mut b = DVector::zeros(active.len());
        for (k, d) in active.iter().enumerate() {
            let jpa = Self::point_jacobian(robot, d.frame_a, &d.point_a)?;
            let jpb = Self::point_jacobian(robot, d.frame_b, &d.point_b)?;
            let mut n = (d.point_b - d.point_a).normalize();
            if d.min_distance < 0.0 {
                n = -n;
            }
            let row = DMatrix::from_row_slice(1, 3, n.as_slice()) * (jpb - jpa);
            a.row_mut(k).copy_from(&row.row(0));
            b[k] = d.min_distance;
        }

        self.config.add(
            problem,
            Expression { a, b }.geq_vector(DVector::from_element(
                active.len(),
                self.self_collisions_margin,
            )),
        );
        Ok(())
    }
}

/// Joint-space half-space constraint `A·q ≤ b` (PlaCo
/// `JointSpaceHalfSpacesConstraint`). `A` has `nq` columns; the floating base is
/// excluded from the delta.
pub struct JointSpaceHalfSpacesConstraint {
    config: Config,
    /// The half-space matrix `A` (`rows × nq`).
    pub a: DMatrix<f64>,
    /// The half-space bound `b` (`rows`).
    pub b: DVector<f64>,
}

impl JointSpaceHalfSpacesConstraint {
    pub(crate) fn new(a: DMatrix<f64>, b: DVector<f64>) -> Self {
        Self {
            config: Config::default(),
            a,
            b,
        }
    }
}

impl KinematicsConstraint for JointSpaceHalfSpacesConstraint {
    fn set_priority_weight(&mut self, priority: ConstraintPriority, weight: f64) {
        self.config = Config { priority, weight };
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn add_to(
        &self,
        problem: &mut Problem,
        _qd: Variable,
        robot: &mut RobotWrapper,
        _dt: f64,
    ) -> Result<()> {
        let nv = robot.nv();
        let ndof = nv - 6;
        let a_no_fbase = self.a.columns(7, self.a.ncols() - 7).into_owned();

        let mut expr_a = DMatrix::zeros(self.a.nrows(), nv);
        expr_a
            .view_mut((0, 6), (self.a.nrows(), ndof))
            .copy_from(&a_no_fbase);
        let q_bottom = robot.state.q.rows(7, ndof).into_owned();
        let expr_b = &a_no_fbase * q_bottom;

        self.config.add(
            problem,
            Expression {
                a: expr_a,
                b: expr_b,
            }
            .leq_vector(self.b.clone()),
        );
        Ok(())
    }
}

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
}

impl CoMPolygonConstraint {
    pub(crate) fn new(polygon: Vec<Vector2<f64>>, margin: f64) -> Self {
        Self {
            config: Config::default(),
            polygon,
            margin,
        }
    }
}

impl KinematicsConstraint for CoMPolygonConstraint {
    fn set_priority_weight(&mut self, priority: ConstraintPriority, weight: f64) {
        self.config = Config { priority, weight };
    }
    fn add_to(
        &self,
        problem: &mut Problem,
        _qd: Variable,
        robot: &mut RobotWrapper,
        _dt: f64,
    ) -> Result<()> {
        let com = robot.com_world()?;
        let jac = robot.com_jacobian()?;
        let com_xy = Expression {
            a: jac.rows(0, 2).into_owned(),
            b: DVector::from_vec(vec![com.x, com.y]),
        };
        let mut constraint = in_polygon_xy(&com_xy, &self.polygon, self.margin);
        constraint.configure(self.config.priority, self.config.weight);
        problem.add_constraint(constraint);
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

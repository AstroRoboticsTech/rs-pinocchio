//! Contact models for the dynamics solver (PlaCo `dynamics::Contact`).

use std::any::Any;

use nalgebra::{DMatrix, Matrix3};

use crate::error::Result;
use crate::placo::model::RobotWrapper;
use crate::placo::problem::{ConstraintPriority, Problem, Variable};
use crate::ReferenceFrame;

fn dmat3(m: &Matrix3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 3, m.as_slice())
}

/// A contact contributing forces to the dynamics QP.
pub trait Contact: Any {
    /// Whether the contact is active.
    fn active(&self) -> bool;
    /// Number of force components.
    fn size(&self) -> usize;
    /// The contact constraint Jacobian (`size × nv`), set by [`Contact::update`].
    fn jacobian(&self) -> &DMatrix<f64>;
    /// Rebuilds the contact Jacobian.
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()>;
    /// Adds the contact's constraints (friction cone, unilaterality) on `f`.
    fn add_constraints(&self, problem: &mut Problem, f: Variable);
    /// Downcast hook.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A 3-DoF point contact, optionally unilateral with a friction cone (PlaCo
/// `PointContact`).
pub struct PointContact {
    /// Contact frame index.
    pub frame_index: usize,
    /// Whether the contact is unilateral (pushes only, with friction).
    pub unilateral: bool,
    /// Friction coefficient.
    pub mu: f64,
    /// Rotation from the surface frame to the world.
    pub r_world_surface: Matrix3<f64>,
    /// Soft weight to minimize the total force.
    pub weight_forces: f64,
    /// Whether this contact is active.
    pub active: bool,
    j: DMatrix<f64>,
}

impl PointContact {
    pub(crate) fn new(frame_index: usize, unilateral: bool) -> Self {
        Self {
            frame_index,
            unilateral,
            mu: 1.0,
            r_world_surface: Matrix3::identity(),
            weight_forces: 0.0,
            active: true,
            j: DMatrix::zeros(0, 0),
        }
    }
}

impl Contact for PointContact {
    fn active(&self) -> bool {
        self.active
    }
    fn size(&self) -> usize {
        3
    }
    fn jacobian(&self) -> &DMatrix<f64> {
        &self.j
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        let jac = robot.frame_jacobian(self.frame_index, ReferenceFrame::LocalWorldAligned)?;
        self.j = jac.rows(0, 3).into_owned();
        Ok(())
    }
    fn add_constraints(&self, problem: &mut Problem, f: Variable) {
        let f_expr = f.expr();
        // Force expressed in the surface frame.
        let f_surface = f_expr.left_multiply(&dmat3(&self.r_world_surface.transpose()));

        if self.unilateral {
            let fx = f_surface.slice(0, 1);
            let fy = f_surface.slice(1, 1);
            let fz = f_surface.slice(2, 1);
            // Normal force is non-negative.
            problem.add_constraint(fz.geq_scalar(0.0));
            // Coulomb friction (linearized box): |Fx|, |Fy| <= mu Fz.
            problem.add_constraint(fx.leq(&fz.scale(self.mu)));
            problem.add_constraint(fx.geq(&fz.scale(-self.mu)));
            problem.add_constraint(fy.leq(&fz.scale(self.mu)));
            problem.add_constraint(fy.geq(&fz.scale(-self.mu)));
        }

        if self.weight_forces > 0.0 {
            problem
                .add_constraint(f_expr.slice(0, 3).equal_vector(nalgebra::DVector::zeros(3)))
                .configure(ConstraintPriority::Soft, self.weight_forces);
        }
    }
}

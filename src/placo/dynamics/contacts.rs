//! Contact models for the dynamics solver (PlaCo `dynamics::Contact`).

use std::any::Any;

use nalgebra::{DMatrix, DVector, Matrix3};

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

/// A 6-DoF planar / fixed contact (force + moment), expressed in the frame's
/// local frame (PlaCo `Contact6D`). Wrench layout: `[Fx, Fy, Fz, Mx, My, Mz]`.
pub struct Contact6D {
    /// Contact frame index.
    pub frame_index: usize,
    /// Whether the contact is unilateral (planar ZMP + friction limits).
    pub unilateral: bool,
    /// Friction coefficient.
    pub mu: f64,
    /// Contact length (x) for the ZMP box [m].
    pub length: f64,
    /// Contact width (y) for the ZMP box [m].
    pub width: f64,
    /// Soft weight to minimize forces.
    pub weight_forces: f64,
    /// Soft weight to minimize moments.
    pub weight_moments: f64,
    /// Whether this contact is active.
    pub active: bool,
    j: DMatrix<f64>,
}

impl Contact6D {
    pub(crate) fn new(frame_index: usize, unilateral: bool) -> Self {
        Self {
            frame_index,
            unilateral,
            mu: 1.0,
            length: 0.0,
            width: 0.0,
            weight_forces: 0.0,
            weight_moments: 0.0,
            active: true,
            j: DMatrix::zeros(0, 0),
        }
    }
}

impl Contact for Contact6D {
    fn active(&self) -> bool {
        self.active
    }
    fn size(&self) -> usize {
        6
    }
    fn jacobian(&self) -> &DMatrix<f64> {
        &self.j
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        // Position rows + orientation rows of the LOCAL frame Jacobian.
        self.j = robot.frame_jacobian(self.frame_index, ReferenceFrame::Local)?;
        Ok(())
    }
    fn add_constraints(&self, problem: &mut Problem, f: Variable) {
        // f = [Fx, Fy, Fz, Mx, My, Mz] in the local frame.
        let e = f.expr();
        let (fx, fy, fz) = (e.slice(0, 1), e.slice(1, 1), e.slice(2, 1));
        let (mx, my) = (e.slice(3, 1), e.slice(4, 1));

        if self.unilateral {
            problem.add_constraint(fz.geq_scalar(0.0));
            // ZMP in the contact box: |My| <= l/2 Fz, |Mx| <= w/2 Fz.
            let l = self.length / 2.0;
            let w = self.width / 2.0;
            problem.add_constraint(my.leq(&fz.scale(l)));
            problem.add_constraint(my.geq(&fz.scale(-l)));
            problem.add_constraint(mx.leq(&fz.scale(w)));
            problem.add_constraint(mx.geq(&fz.scale(-w)));
            // Friction: |Fx|, |Fy| <= mu Fz.
            problem.add_constraint(fx.leq(&fz.scale(self.mu)));
            problem.add_constraint(fx.geq(&fz.scale(-self.mu)));
            problem.add_constraint(fy.leq(&fz.scale(self.mu)));
            problem.add_constraint(fy.geq(&fz.scale(-self.mu)));
        }

        if self.weight_forces > 0.0 {
            problem
                .add_constraint(e.slice(0, 3).equal_vector(nalgebra::DVector::zeros(3)))
                .configure(ConstraintPriority::Soft, self.weight_forces);
        }
        if self.weight_moments > 0.0 {
            problem
                .add_constraint(e.slice(3, 3).equal_vector(nalgebra::DVector::zeros(3)))
                .configure(ConstraintPriority::Soft, self.weight_moments);
        }
    }
}

/// A known external wrench applied at a frame (PlaCo `ExternalWrenchContact`).
///
/// The 6-DoF "force" is fixed to `w_ext` rather than optimized.
pub struct ExternalWrenchContact {
    /// Frame index.
    pub frame_index: usize,
    /// Reference frame of the wrench Jacobian.
    pub reference: ReferenceFrame,
    /// The applied wrench `[Fx, Fy, Fz, Mx, My, Mz]`.
    pub w_ext: DVector<f64>,
    /// Whether this contact is active.
    pub active: bool,
    j: DMatrix<f64>,
}

impl ExternalWrenchContact {
    pub(crate) fn new(frame_index: usize, reference: ReferenceFrame) -> Self {
        Self {
            frame_index,
            reference,
            w_ext: DVector::zeros(6),
            active: true,
            j: DMatrix::zeros(0, 0),
        }
    }
}

impl Contact for ExternalWrenchContact {
    fn active(&self) -> bool {
        self.active
    }
    fn size(&self) -> usize {
        6
    }
    fn jacobian(&self) -> &DMatrix<f64> {
        &self.j
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        self.j = robot.frame_jacobian(self.frame_index, self.reference)?;
        Ok(())
    }
    fn add_constraints(&self, problem: &mut Problem, f: Variable) {
        // The wrench is fixed (not a free variable).
        problem.add_constraint(f.expr().equal_vector(self.w_ext.clone()));
    }
}

/// A "puppet" contact applying an unconstrained generalized force on every DoF
/// (identity Jacobian) (PlaCo `PuppetContact`). Useful to fully actuate a robot
/// (e.g. for testing or a floating puppet).
pub struct PuppetContact {
    /// Whether this contact is active.
    pub active: bool,
    j: DMatrix<f64>,
}

impl PuppetContact {
    pub(crate) fn new() -> Self {
        Self {
            active: true,
            j: DMatrix::zeros(0, 0),
        }
    }
}

impl Contact for PuppetContact {
    fn active(&self) -> bool {
        self.active
    }
    fn size(&self) -> usize {
        self.j.nrows()
    }
    fn jacobian(&self) -> &DMatrix<f64> {
        &self.j
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper) -> Result<()> {
        let n = robot.nv();
        self.j = DMatrix::identity(n, n);
        Ok(())
    }
    fn add_constraints(&self, _problem: &mut Problem, _f: Variable) {}
}

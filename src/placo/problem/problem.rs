//! The QP problem: variables, constraints and a [`clarabel`]-backed solve
//! (PlaCo `Problem`).
//!
//! PlaCo assembles, over decision variables `x` (plus slacks `s` for soft
//! inequalities):
//!
//! - a cost `½ xᵀP x + qᵀx`, where each *soft equality* task `A·x + b` adds
//!   `w·AᵀA` to `P` and `w·Aᵀb` to `q` (i.e. minimizes `w·‖A·x + b‖²`),
//! - *hard equalities* `A·x + b = 0`,
//! - *hard inequalities* `A·x + b ≥ 0`,
//! - *soft inequalities*, handled with a non-negative slack per row.
//!
//! Upstream solves this with eiquadprog (Goldfarb–Idnani). This port maps the
//! same QP onto clarabel's cone form (`min ½xᵀPx + qᵀx` s.t. `Ax + s = b`,
//! `s ∈ K`), using a zero cone for equalities and a non-negative cone for
//! inequalities. Results match to solver tolerance rather than bit-for-bit.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, SolverStatus,
    SupportedConeT::{NonnegativeConeT, ZeroConeT},
};
use nalgebra::{DMatrix, DVector};

use super::constraint::{Constraint, ConstraintPriority, ConstraintType};
use super::error::{QpError, QpResult};
use super::expression::Expression;
use super::variable::Variable;

/// A quadratic program: a set of variables and constraints solved together.
pub struct Problem {
    n_variables: usize,
    constraints: Vec<Constraint>,
    /// Ridge regularization added to the free-variable Hessian diagonal.
    pub regularization: f64,
    /// Number of hard equality rows in the last solve.
    pub n_equalities: usize,
    /// Number of inequality rows in the last solve.
    pub n_inequalities: usize,
    /// Number of slack variables in the last solve.
    pub slack_variables: usize,
    x: DVector<f64>,
    slacks: DVector<f64>,
    version: usize,
}

impl Default for Problem {
    fn default() -> Self {
        Self::new()
    }
}

impl Problem {
    /// An empty problem.
    pub fn new() -> Self {
        Self {
            n_variables: 0,
            constraints: Vec::new(),
            regularization: 1e-8,
            n_equalities: 0,
            n_inequalities: 0,
            slack_variables: 0,
            x: DVector::zeros(0),
            slacks: DVector::zeros(0),
            version: 0,
        }
    }

    /// Adds a `size`-dimensional variable and returns its handle.
    pub fn add_variable(&mut self, size: usize) -> Variable {
        let v = Variable {
            k_start: self.n_variables,
            k_end: self.n_variables + size,
        };
        self.n_variables += size;
        v
    }

    /// Number of scalar decision variables added so far.
    pub fn n_variables(&self) -> usize {
        self.n_variables
    }

    /// Adds a constraint; returns a mutable reference so it can be configured.
    pub fn add_constraint(&mut self, constraint: Constraint) -> &mut Constraint {
        self.constraints.push(constraint);
        self.constraints.last_mut().unwrap()
    }

    /// Adds an absolute-value limit `|A·x + b| ≤ target` (elementwise).
    pub fn add_limit(&mut self, expression: Expression, target: DVector<f64>) -> &mut Constraint {
        // Stack [A; -A] x + [b; -b] ≤ [target; target].
        let rows = expression.rows();
        let cols = expression.cols();
        let mut a = DMatrix::zeros(rows * 2, cols);
        a.view_mut((0, 0), (rows, cols)).copy_from(&expression.a);
        a.view_mut((rows, 0), (rows, cols))
            .copy_from(&(-&expression.a));
        let mut b = DVector::zeros(rows * 2);
        b.rows_mut(0, rows).copy_from(&expression.b);
        b.rows_mut(rows, rows).copy_from(&(-&expression.b));
        let mut targets = DVector::zeros(rows * 2);
        targets.rows_mut(0, rows).copy_from(&target);
        targets.rows_mut(rows, rows).copy_from(&target);

        let stacked = Expression { a, b };
        self.add_constraint(stacked.leq_vector(targets))
    }

    /// Removes all constraints (keeps the variables).
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }

    /// Removes all variables and constraints.
    pub fn clear_variables(&mut self) {
        self.constraints.clear();
        self.n_variables = 0;
    }

    /// The full solved decision-variable vector (after [`Problem::solve`]).
    pub fn solution(&self) -> &DVector<f64> {
        &self.x
    }

    /// The solved slack variables (after [`Problem::solve`]).
    pub fn slacks(&self) -> &DVector<f64> {
        &self.slacks
    }

    /// Read-only view of the constraints (their `is_active` flags are updated by
    /// [`Problem::solve`]).
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Solves the QP, filling [`Problem::solution`] and constraint activity.
    pub fn solve(&mut self) -> QpResult<()> {
        let n = self.n_variables;

        // --- pass 1: count slacks / equalities -------------------------------
        let mut slack_variables = 0usize;
        let mut n_equalities = 0usize;
        for c in &self.constraints {
            match c.type_ {
                ConstraintType::Inequality => {
                    if c.priority == ConstraintPriority::Soft {
                        slack_variables += c.expression.rows();
                    }
                }
                ConstraintType::Equality => {
                    if c.priority == ConstraintPriority::Hard {
                        n_equalities += c.expression.rows();
                    }
                }
            }
        }

        let nvar = n + slack_variables;

        // --- objective P, q --------------------------------------------------
        let mut p = DMatrix::zeros(nvar, nvar);
        let mut q = DVector::zeros(nvar);
        for i in 0..n {
            p[(i, i)] = self.regularization;
        }

        // --- equality stack --------------------------------------------------
        let mut a_eq = DMatrix::zeros(n_equalities, nvar);
        let mut b_eq = DVector::zeros(n_equalities);
        let mut k_eq = 0usize;

        // --- inequality stack (Gx + h >= 0): first slack>=0, then hard --------
        let mut n_inequalities = 0usize;
        for c in &self.constraints {
            self.validate_constraint(c)?;
            if c.type_ == ConstraintType::Inequality {
                n_inequalities += c.expression.rows();
            }
        }
        let mut g = DMatrix::zeros(n_inequalities, nvar);
        let mut h = DVector::zeros(n_inequalities);
        let mut k_ineq = 0usize;
        let mut k_slack = 0usize;

        // slack_i >= 0
        for slack in 0..slack_variables {
            g[(k_ineq, n + slack)] = 1.0;
            k_ineq += 1;
        }

        // Map QP inequality rows back to constraint indices for activity report.
        let mut hard_row_owner: Vec<Option<usize>> = vec![None; n_inequalities];
        let mut soft_slack_owner: Vec<Option<usize>> = vec![None; slack_variables];

        for (ci, c) in self.constraints.iter().enumerate() {
            let rows = c.expression.rows();
            let cols = c.expression.cols();
            match c.type_ {
                ConstraintType::Equality if c.priority == ConstraintPriority::Hard => {
                    a_eq.view_mut((k_eq, 0), (rows, cols))
                        .copy_from(&c.expression.a);
                    b_eq.rows_mut(k_eq, rows).copy_from(&c.expression.b);
                    k_eq += rows;
                }
                ConstraintType::Equality => {
                    // Soft equality -> objective: w * ||A x + b||^2.
                    let a = &c.expression.a;
                    let b = &c.expression.b;
                    let ata = a.transpose() * a; // cols x cols
                    let atb = a.transpose() * b; // cols
                    {
                        let mut block = p.view_mut((0, 0), (cols, cols));
                        block += c.weight * ata;
                    }
                    q.rows_mut(0, cols).axpy(c.weight, &atb, 1.0);
                }
                ConstraintType::Inequality if c.priority == ConstraintPriority::Hard => {
                    g.view_mut((k_ineq, 0), (rows, cols))
                        .copy_from(&c.expression.a);
                    h.rows_mut(k_ineq, rows).copy_from(&c.expression.b);
                    for r in 0..rows {
                        hard_row_owner[k_ineq + r] = Some(ci);
                    }
                    k_ineq += rows;
                }
                ConstraintType::Inequality => {
                    // Soft inequality -> min w * ||A x + b - s||^2, s >= 0.
                    let mut as_ = DMatrix::zeros(rows, nvar);
                    as_.view_mut((0, 0), (rows, cols))
                        .copy_from(&c.expression.a);
                    for r in 0..rows {
                        soft_slack_owner[k_slack] = Some(ci);
                        as_[(r, n + k_slack)] = -1.0;
                        k_slack += 1;
                    }
                    let asa = as_.transpose() * &as_;
                    let asb = as_.transpose() * &c.expression.b;
                    p += c.weight * asa;
                    q.axpy(c.weight, &asb, 1.0);
                }
            }
        }

        // --- solve via clarabel ---------------------------------------------
        let x = self.solve_clarabel(&p, &q, &a_eq, &b_eq, &g, &h, n_equalities, n_inequalities)?;

        if x.iter().any(|v| v.is_nan()) {
            return Err(QpError::Nan);
        }

        // Verify hard equalities held (clarabel returns feasibility only softly).
        if n_equalities > 0 {
            let residual = &a_eq * &x + &b_eq;
            if residual.amax() > 1e-6 {
                return Err(QpError::Infeasible(
                    "equality constraints were not enforced".into(),
                ));
            }
        }

        // --- record results (compute activity against local `x` first) -------
        let slacks = x.rows(n, slack_variables).into_owned();

        for c in &mut self.constraints {
            c.is_active = c.type_ == ConstraintType::Equality;
        }
        for (row, owner) in hard_row_owner.iter().enumerate() {
            if let Some(ci) = owner {
                let val = (g.row(row) * &x)[0] + h[row];
                if val <= 1e-6 {
                    self.constraints[*ci].is_active = true;
                }
            }
        }
        for (k, owner) in soft_slack_owner.iter().enumerate() {
            if let Some(ci) = owner {
                if slacks[k] <= 1e-6 {
                    self.constraints[*ci].is_active = true;
                }
            }
        }

        self.n_equalities = n_equalities;
        self.n_inequalities = n_inequalities;
        self.slack_variables = slack_variables;
        self.slacks = slacks;
        self.x = x;
        self.version += 1;

        Ok(())
    }

    fn validate_constraint(&self, c: &Constraint) -> QpResult<()> {
        if c.expression.cols() > self.n_variables {
            return Err(QpError::Malformed("inconsistent problem size".into()));
        }
        if c.expression.a.nrows() == 0 || c.expression.b.nrows() == 0 {
            return Err(QpError::Malformed("A or b is empty".into()));
        }
        if c.expression.a.nrows() != c.expression.b.nrows() {
            return Err(QpError::Malformed("A.rows() != b.rows()".into()));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_clarabel(
        &self,
        p: &DMatrix<f64>,
        q: &DVector<f64>,
        a_eq: &DMatrix<f64>,
        b_eq: &DVector<f64>,
        g: &DMatrix<f64>,
        h: &DVector<f64>,
        n_equalities: usize,
        n_inequalities: usize,
    ) -> QpResult<DVector<f64>> {
        let nvar = p.nrows();

        // Constraint matrix: [A_eq; -G] z + s = [-b_eq; h], zero then nonneg cone.
        let m = n_equalities + n_inequalities;
        let mut a_full = DMatrix::zeros(m, nvar);
        let mut b_full = DVector::zeros(m);
        if n_equalities > 0 {
            a_full
                .view_mut((0, 0), (n_equalities, nvar))
                .copy_from(a_eq);
            b_full.rows_mut(0, n_equalities).copy_from(&(-b_eq));
        }
        if n_inequalities > 0 {
            a_full
                .view_mut((n_equalities, 0), (n_inequalities, nvar))
                .copy_from(&(-g));
            b_full.rows_mut(n_equalities, n_inequalities).copy_from(h);
        }

        let p_csc = dense_to_csc_upper(p);
        let a_csc = dense_to_csc(&a_full);
        let q_vec: Vec<f64> = q.iter().copied().collect();
        let b_vec: Vec<f64> = b_full.iter().copied().collect();

        let mut cones = Vec::new();
        if n_equalities > 0 {
            cones.push(ZeroConeT(n_equalities));
        }
        if n_inequalities > 0 {
            cones.push(NonnegativeConeT(n_inequalities));
        }

        let settings = DefaultSettings::<f64> {
            verbose: false,
            ..DefaultSettings::default()
        };

        let mut solver = DefaultSolver::new(&p_csc, &q_vec, &a_csc, &b_vec, &cones, settings);
        solver.solve();

        match solver.solution.status {
            SolverStatus::Solved | SolverStatus::AlmostSolved => {
                Ok(DVector::from_vec(solver.solution.x))
            }
            other => Err(QpError::Infeasible(format!("solver status: {other:?}"))),
        }
    }
}

/// Dense → CSC (all structural non-zeros).
fn dense_to_csc(m: &DMatrix<f64>) -> CscMatrix<f64> {
    let (nrows, ncols) = (m.nrows(), m.ncols());
    let mut colptr = Vec::with_capacity(ncols + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..ncols {
        for i in 0..nrows {
            let v = m[(i, j)];
            if v != 0.0 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(nzval.len());
    }
    CscMatrix::new(nrows, ncols, colptr, rowval, nzval)
}

/// Dense symmetric → CSC upper triangle (clarabel expects `P` as upper-tri).
fn dense_to_csc_upper(m: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = m.nrows();
    let mut colptr = Vec::with_capacity(n + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for j in 0..n {
        for i in 0..=j {
            let v = m[(i, j)];
            if v != 0.0 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(nzval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_squares_hits_soft_target() {
        // Minimize ||x - 3||^2  ->  x = 3.
        let mut problem = Problem::new();
        let x = problem.add_variable(1);
        let c = x.expr().equal_scalar(3.0);
        problem
            .add_constraint(c)
            .configure(ConstraintPriority::Soft, 1.0);
        problem.solve().unwrap();
        assert!((x.value(&problem)[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn hard_equality_is_enforced() {
        // min ||x||^2 s.t. x = 5  -> x = 5.
        let mut problem = Problem::new();
        let x = problem.add_variable(1);
        problem
            .add_constraint(x.expr().equal_scalar(0.0))
            .configure(ConstraintPriority::Soft, 1.0);
        problem.add_constraint(x.expr().equal_scalar(5.0)); // hard by default
        problem.solve().unwrap();
        assert!((x.value(&problem)[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn inequality_bounds_the_solution() {
        // min ||x - 10||^2 s.t. x <= 4  -> x = 4.
        let mut problem = Problem::new();
        let x = problem.add_variable(1);
        problem
            .add_constraint(x.expr().equal_scalar(10.0))
            .configure(ConstraintPriority::Soft, 1.0);
        problem.add_constraint(x.expr().leq_scalar(4.0));
        problem.solve().unwrap();
        let v = x.value(&problem)[0];
        assert!((v - 4.0).abs() < 1e-5, "x = {v}");
    }

    #[test]
    fn two_variables_weighted_least_squares() {
        // min ||x0 - 1||^2 + ||x1 - 2||^2  -> (1, 2).
        let mut problem = Problem::new();
        let v = problem.add_variable(2);
        problem
            .add_constraint(v.expr_slice(0, 1).equal_scalar(1.0))
            .configure(ConstraintPriority::Soft, 1.0);
        problem
            .add_constraint(v.expr_slice(1, 1).equal_scalar(2.0))
            .configure(ConstraintPriority::Soft, 1.0);
        problem.solve().unwrap();
        let val = v.value(&problem);
        assert!((val[0] - 1.0).abs() < 1e-6);
        assert!((val[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn add_limit_constrains_absolute_value() {
        // min ||x - 10||^2 s.t. |x| <= 3 -> x = 3.
        let mut problem = Problem::new();
        let x = problem.add_variable(1);
        problem
            .add_constraint(x.expr().equal_scalar(10.0))
            .configure(ConstraintPriority::Soft, 1.0);
        problem.add_limit(x.expr(), DVector::from_element(1, 3.0));
        problem.solve().unwrap();
        assert!((x.value(&problem)[0] - 3.0).abs() < 1e-5);
    }
}

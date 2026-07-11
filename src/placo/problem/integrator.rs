//! Linear-system integration over a decision variable (PlaCo `Integrator`).
//!
//! Builds expressions and post-solve trajectories for a state that evolves as
//! `dX = M·X`, where the command is the decision variable. Used by the walk
//! pattern generator to formulate jerk/acceleration-driven CoM trajectories.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};

use super::expression::Expression;
use super::problem::Problem;
use super::variable::Variable;

/// A continuous trajectory recovered from a solved [`Integrator`].
#[derive(Clone, Debug)]
pub struct Trajectory {
    /// The solved command values (one per step).
    pub variable_value: DVector<f64>,
    /// The continuous system matrix.
    pub m: DMatrix<f64>,
    /// State keyframes at each step boundary.
    pub keyframes: HashMap<usize, DVector<f64>>,
    /// System order.
    pub order: usize,
    /// Step duration.
    pub dt: f64,
    /// Time offset.
    pub t_start: f64,
}

impl Trajectory {
    /// Total trajectory duration.
    pub fn duration(&self) -> f64 {
        self.keyframes.len() as f64 * self.dt
    }

    /// Value of the trajectory at time `t`, differentiated `diff` times.
    pub fn value(&self, mut t: f64, diff: usize) -> f64 {
        t -= self.t_start;
        assert!(
            diff <= self.order,
            "Trajectory: diff {diff} > order {}",
            self.order
        );

        let mut k = (t / self.dt).floor() as isize;
        if k < 0 {
            k = 0;
        }
        let max_k = self.variable_value.len() as isize - 1;
        if k > max_k {
            k = max_k;
        }
        let k = k as usize;

        let remaining_dt = (t - k as f64 * self.dt).clamp(0.0, self.dt);

        if diff == self.order {
            self.variable_value[k]
        } else {
            let (ar, br) = ab_matrices(&self.m, self.order, remaining_dt);
            let result = &ar * &self.keyframes[&k] + &br * self.variable_value[k];
            result[diff]
        }
    }
}

/// Integrates a decision variable through a linear system `dX = M·X`.
pub struct Integrator {
    /// The decision variable (command sequence).
    pub variable: Variable,
    /// Number of steps.
    pub n: usize,
    /// Continuous system matrix.
    pub m: DMatrix<f64>,
    /// Discrete transition matrix `A` (`X_{k+1} = A·X_k + B·u_k`).
    pub a: DMatrix<f64>,
    /// Discrete input matrix `B`.
    pub b: DVector<f64>,
    /// Initial state expression.
    pub x0: Expression,
    /// Final transition matrix caching the command contributions.
    pub final_transition_matrix: DMatrix<f64>,
    /// Cached powers of `A`.
    pub a_powers: HashMap<usize, DMatrix<f64>>,
    /// System order.
    pub order: usize,
    /// Step duration.
    pub dt: f64,
    /// Time offset.
    pub t_start: f64,
}

impl Integrator {
    /// The upper-shift system matrix for a chain of integrators of `order`.
    pub fn upper_shift_matrix(order: usize) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(order + 1, order + 1);
        for k in 0..order {
            m[(k, k + 1)] = 1.0;
        }
        m
    }

    /// Builds an integrator over `variable` with an order-`order` integrator
    /// chain, initial state `x0` and step `dt`.
    ///
    /// # Panics
    /// If `x0` does not have `order` rows.
    pub fn new(variable: Variable, x0: Expression, order: usize, dt: f64) -> Self {
        assert_eq!(x0.rows(), order, "Integrator: X0 should have {order} rows");
        Self::with_system(variable, x0, Self::upper_shift_matrix(order), dt)
    }

    /// Builds an integrator with a custom continuous system matrix `dX = M·X`.
    pub fn with_system(
        variable: Variable,
        x0: Expression,
        system_matrix: DMatrix<f64>,
        dt: f64,
    ) -> Self {
        let order = system_matrix.nrows() - 1;
        let n = variable.size();
        let (a, b) = ab_matrices(&system_matrix, order, dt);

        let mut final_transition_matrix = DMatrix::zeros(order, n);
        let mut a_powers: HashMap<usize, DMatrix<f64>> = HashMap::new();
        let mut ak = DMatrix::identity(order, order);
        a_powers.insert(0, ak.clone());

        for step in 0..n {
            let col = &ak * &b; // order x 1
            final_transition_matrix
                .view_mut((0, n - step - 1), (order, 1))
                .copy_from(&col);
            ak = &a * &ak;
            a_powers.insert(step + 1, ak.clone());
        }

        Self {
            variable,
            n,
            m: system_matrix,
            a,
            b,
            x0,
            final_transition_matrix,
            a_powers,
            order,
            dt,
            t_start: 0.0,
        }
    }

    /// Expression for the state at `step`, differentiated `diff` times.
    ///
    /// `step == usize::MAX` selects the last step. `diff == usize::MAX` returns
    /// the full order-length state vector.
    pub fn expr(&self, step: usize, diff: usize) -> Expression {
        let all = diff == usize::MAX;
        let step = if step == usize::MAX { self.n } else { step };
        assert!(step <= self.n, "Integrator: step {step} out of range");

        if !all && diff == self.order {
            return self.variable.expr_slice(step, 1);
        }

        let rows = if all { self.order } else { 1 };
        let mut a = DMatrix::zeros(rows, self.variable.k_end);
        let b = DVector::zeros(rows);

        if all {
            a.view_mut((0, self.variable.k_start), (rows, step))
                .copy_from(
                    &self
                        .final_transition_matrix
                        .view((0, self.n - step), (rows, step)),
                );
            let e = Expression { a, b };
            let ax0 = self.x0.left_multiply(&self.a_powers[&step]);
            e.add(&ax0)
        } else {
            a.view_mut((0, self.variable.k_start), (1, step)).copy_from(
                &self
                    .final_transition_matrix
                    .view((diff, self.n - step), (1, step)),
            );
            let e = Expression { a, b };
            let ax0 = self.x0.left_multiply(&self.a_powers[&step]).slice(diff, 1);
            e.add(&ax0)
        }
    }

    /// Expression for the state at time `t`, differentiated `diff` times.
    pub fn expr_t(&self, mut t: f64, diff: usize) -> Expression {
        t -= self.t_start;
        let step = ((t / self.dt) as isize).clamp(0, self.n as isize - 1) as usize;

        if diff != usize::MAX && diff == self.order {
            return self.variable.expr_slice(step, 1);
        }

        let remaining_dt = t - step as f64 * self.dt;
        let (ar, br) = ab_matrices(&self.m, self.order, remaining_dt);
        let br_mat = DMatrix::from_column_slice(self.order, 1, br.as_slice());
        let mut e = self.expr(step, usize::MAX).left_multiply(&ar);
        let command = self.variable.expr_slice(step, 1).left_multiply(&br_mat);
        e = e.add(&command);

        if diff != usize::MAX {
            e = e.slice(diff, 1);
        }
        e
    }

    /// Recovers the continuous trajectory from a solved [`Problem`].
    pub fn trajectory(&self, problem: &Problem) -> Trajectory {
        let variable_value = self.variable.value(problem);
        let mut keyframes: HashMap<usize, DVector<f64>> = HashMap::new();
        let mut x = self.x0.value(problem.solution());
        keyframes.insert(0, x.clone());
        for k in 1..=self.variable.size() {
            x = &self.a * &x + &self.b * variable_value[k - 1];
            keyframes.insert(k, x.clone());
        }
        Trajectory {
            variable_value,
            m: self.m.clone(),
            keyframes,
            order: self.order,
            dt: self.dt,
            t_start: self.t_start,
        }
    }
}

/// Discrete `(A, B)` from the continuous system `M` via `exp(M·dt)`.
pub fn ab_matrices(m: &DMatrix<f64>, order: usize, dt: f64) -> (DMatrix<f64>, DVector<f64>) {
    let me = matrix_exp(&(m * dt));
    let a = me.view((0, 0), (order, order)).into_owned();
    let b = me.view((0, order), (order, 1)).column(0).into_owned();
    (a, b)
}

/// Matrix exponential via scaling-and-squaring with a Taylor series.
fn matrix_exp(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    // Scale so that ‖A/2^s‖ is small, then square s times.
    let norm = a.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())) * n as f64;
    let s = if norm > 0.5 {
        (norm.log2().ceil() as i32 + 1).max(0) as u32
    } else {
        0
    };
    let scale = 2f64.powi(s as i32);
    let a_scaled = a / scale;

    // Taylor series exp(A) ≈ Σ A^k / k!
    let mut term = DMatrix::identity(n, n);
    let mut result = DMatrix::identity(n, n);
    for k in 1..=18 {
        term = (&term * &a_scaled) / k as f64;
        result += &term;
    }

    for _ in 0..s {
        result = &result * &result;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_exp_of_zero_is_identity() {
        let z = DMatrix::zeros(3, 3);
        let e = matrix_exp(&z);
        assert!((e - DMatrix::<f64>::identity(3, 3)).amax() < 1e-12);
    }

    #[test]
    fn matrix_exp_diagonal() {
        let mut d = DMatrix::zeros(2, 2);
        d[(0, 0)] = 1.0;
        d[(1, 1)] = 2.0;
        let e = matrix_exp(&d);
        assert!((e[(0, 0)] - 1.0_f64.exp()).abs() < 1e-9);
        assert!((e[(1, 1)] - 2.0_f64.exp()).abs() < 1e-9);
        assert!(e[(0, 1)].abs() < 1e-12);
    }

    #[test]
    fn upper_shift_double_integrator_ab() {
        // Order 2 (double integrator): x'' = u. Discrete A/B for dt.
        let m = Integrator::upper_shift_matrix(2);
        let dt = 0.1;
        let (a, b) = ab_matrices(&m, 2, dt);
        // A = [[1, dt],[0,1]], B = [dt^2/2, dt].
        assert!((a[(0, 0)] - 1.0).abs() < 1e-9);
        assert!((a[(0, 1)] - dt).abs() < 1e-9);
        assert!((a[(1, 1)] - 1.0).abs() < 1e-9);
        assert!((a[(1, 0)]).abs() < 1e-12);
        assert!((b[0] - dt * dt / 2.0).abs() < 1e-9);
        assert!((b[1] - dt).abs() < 1e-9);
    }

    #[test]
    fn integrator_trajectory_matches_forward_sim() {
        // Double integrator, constant command u = 1 over 3 steps.
        let mut problem = Problem::new();
        let var = problem.add_variable(3);
        let x0 = Expression::from_vector(DVector::from_vec(vec![0.0, 0.0]));
        let integ = Integrator::new(var, x0, 2, 1.0);
        // Pin the command to 1 for each step.
        for k in 0..3 {
            problem.add_constraint(var.expr_slice(k, 1).equal_scalar(1.0));
        }
        problem.solve().unwrap();
        let traj = integ.trajectory(&problem);
        // Position after 3 unit steps of a double integrator with u=1:
        // keyframe positions: 0, 0.5, 2.0, 4.5.
        assert!((traj.keyframes[&0][0] - 0.0).abs() < 1e-6);
        assert!((traj.keyframes[&1][0] - 0.5).abs() < 1e-6);
        assert!((traj.keyframes[&3][0] - 4.5).abs() < 1e-6);
    }
}

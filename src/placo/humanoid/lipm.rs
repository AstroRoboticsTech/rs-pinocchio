//! Linear Inverted Pendulum Model helper (PlaCo `LIPM`).
//!
//! Builds a jerk-driven CoM trajectory (two order-3 integrators, x and y) whose
//! decision variables are the piecewise-constant jerks. Provides QP expressions
//! for the CoM position/velocity/acceleration/jerk and the DCM/ZMP, plus a
//! post-solve continuous trajectory.

use nalgebra::{DVector, Vector2};

use crate::placo::problem::{Expression, Integrator, Problem, Trajectory as IntegratorTrajectory};

/// Standard gravity [m/s²].
const GRAVITY: f64 = 9.80665;

/// A LIPM CoM trajectory recovered after solving.
#[derive(Clone, Debug)]
pub struct LipmTrajectory {
    /// x-axis integrator trajectory.
    pub x: IntegratorTrajectory,
    /// y-axis integrator trajectory.
    pub y: IntegratorTrajectory,
}

impl LipmTrajectory {
    /// CoM position at time `t`.
    pub fn pos(&self, t: f64) -> Vector2<f64> {
        Vector2::new(self.x.value(t, 0), self.y.value(t, 0))
    }
    /// CoM velocity at time `t`.
    pub fn vel(&self, t: f64) -> Vector2<f64> {
        Vector2::new(self.x.value(t, 1), self.y.value(t, 1))
    }
    /// CoM acceleration at time `t`.
    pub fn acc(&self, t: f64) -> Vector2<f64> {
        Vector2::new(self.x.value(t, 2), self.y.value(t, 2))
    }
    /// CoM jerk at time `t`.
    pub fn jerk(&self, t: f64) -> Vector2<f64> {
        Vector2::new(self.x.value(t, 3), self.y.value(t, 3))
    }
    /// Divergent component of motion at time `t`.
    pub fn dcm(&self, t: f64, omega: f64) -> Vector2<f64> {
        self.pos(t) + self.vel(t) / omega
    }
    /// Zero-moment point at time `t` (`omega_2 = omega²`).
    pub fn zmp(&self, t: f64, omega_2: f64) -> Vector2<f64> {
        self.pos(t) - self.acc(t) / omega_2
    }
}

/// A LIPM problem helper over two order-3 integrators.
pub struct Lipm {
    /// Timestep duration.
    pub dt: f64,
    /// Number of timesteps.
    pub timesteps: usize,
    /// Trajectory start time.
    pub t_start: f64,
    x: Integrator,
    y: Integrator,
}

impl Lipm {
    /// Builds a LIPM from an initial CoM state.
    pub fn new(
        problem: &mut Problem,
        dt: f64,
        timesteps: usize,
        t_start: f64,
        initial_pos: Vector2<f64>,
        initial_vel: Vector2<f64>,
        initial_acc: Vector2<f64>,
    ) -> Self {
        let x_var = problem.add_variable(timesteps);
        let y_var = problem.add_variable(timesteps);
        let x0_x = Expression::from_vector(DVector::from_vec(vec![
            initial_pos.x,
            initial_vel.x,
            initial_acc.x,
        ]));
        let x0_y = Expression::from_vector(DVector::from_vec(vec![
            initial_pos.y,
            initial_vel.y,
            initial_acc.y,
        ]));
        let mut x = Integrator::new(x_var, x0_x, 3, dt);
        let mut y = Integrator::new(y_var, x0_y, 3, dt);
        x.t_start = t_start;
        y.t_start = t_start;
        Self {
            dt,
            timesteps,
            t_start,
            x,
            y,
        }
    }

    /// Builds a LIPM continuing from a previous one's final state.
    pub fn from_previous(
        problem: &mut Problem,
        dt: f64,
        timesteps: usize,
        t_start: f64,
        previous: &Lipm,
    ) -> Self {
        let x_var = problem.add_variable(timesteps);
        let y_var = problem.add_variable(timesteps);
        let x0_x = previous.x.expr(usize::MAX, usize::MAX);
        let x0_y = previous.y.expr(usize::MAX, usize::MAX);
        let mut x = Integrator::new(x_var, x0_x, 3, dt);
        let mut y = Integrator::new(y_var, x0_y, 3, dt);
        x.t_start = t_start;
        y.t_start = t_start;
        Self {
            dt,
            timesteps,
            t_start,
            x,
            y,
        }
    }

    /// Trajectory end time.
    pub fn t_end(&self) -> f64 {
        self.t_start + self.timesteps as f64 * self.dt
    }

    fn stacked(&self, x: Expression, y: Expression) -> Expression {
        x.vstack(&y)
    }

    /// CoM position expression at `timestep` (2 rows: x, y).
    pub fn pos(&self, timestep: usize) -> Expression {
        self.stacked(self.x.expr(timestep, 0), self.y.expr(timestep, 0))
    }
    /// CoM velocity expression at `timestep`.
    pub fn vel(&self, timestep: usize) -> Expression {
        self.stacked(self.x.expr(timestep, 1), self.y.expr(timestep, 1))
    }
    /// CoM acceleration expression at `timestep`.
    pub fn acc(&self, timestep: usize) -> Expression {
        self.stacked(self.x.expr(timestep, 2), self.y.expr(timestep, 2))
    }
    /// CoM jerk expression at `timestep`.
    pub fn jerk(&self, timestep: usize) -> Expression {
        self.stacked(self.x.expr(timestep, 3), self.y.expr(timestep, 3))
    }

    /// DCM expression at `timestep`.
    pub fn dcm(&self, timestep: usize, omega: f64) -> Expression {
        let ex = self
            .x
            .expr(timestep, 0)
            .add(&self.x.expr(timestep, 1).scale(1.0 / omega));
        let ey = self
            .y
            .expr(timestep, 0)
            .add(&self.y.expr(timestep, 1).scale(1.0 / omega));
        self.stacked(ex, ey)
    }

    /// ZMP expression at `timestep` (`omega_2 = omega²`).
    pub fn zmp(&self, timestep: usize, omega_2: f64) -> Expression {
        let ex = self
            .x
            .expr(timestep, 0)
            .subtract(&self.x.expr(timestep, 2).scale(1.0 / omega_2));
        let ey = self
            .y
            .expr(timestep, 0)
            .subtract(&self.y.expr(timestep, 2).scale(1.0 / omega_2));
        self.stacked(ex, ey)
    }

    /// The LIPM natural frequency `omega = sqrt(g / h)`.
    pub fn compute_omega(com_height: f64) -> f64 {
        (GRAVITY / com_height).sqrt()
    }

    /// The continuous CoM trajectory (after solving).
    pub fn trajectory(&self, problem: &Problem) -> LipmTrajectory {
        LipmTrajectory {
            x: self.x.trajectory(problem),
            y: self.y.trajectory(problem),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placo::problem::ConstraintPriority;

    #[test]
    fn compute_omega_matches_formula() {
        let omega = Lipm::compute_omega(0.35);
        assert!((omega - (GRAVITY / 0.35_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn min_jerk_lipm_reaches_terminal_target() {
        // Minimum-jerk CoM move from (0,0) to (0.1, 0.05), rest-to-rest.
        let mut problem = Problem::new();
        let dt = 0.1;
        let timesteps = 20;
        let lipm = Lipm::new(
            &mut problem,
            dt,
            timesteps,
            0.0,
            Vector2::zeros(),
            Vector2::zeros(),
            Vector2::zeros(),
        );

        // Regularize the jerk (soft) for a well-posed minimum-jerk problem.
        for k in 0..timesteps {
            problem
                .add_constraint(lipm.jerk(k).equal_vector(DVector::zeros(2)))
                .configure(ConstraintPriority::Soft, 1e-6);
        }
        // Terminal position, velocity and acceleration (hard).
        let target = Vector2::new(0.1, 0.05);
        problem.add_constraint(
            lipm.pos(timesteps)
                .equal_vector(DVector::from_vec(vec![target.x, target.y])),
        );
        problem.add_constraint(lipm.vel(timesteps).equal_vector(DVector::zeros(2)));
        problem.add_constraint(lipm.acc(timesteps).equal_vector(DVector::zeros(2)));

        problem.solve().unwrap();

        let traj = lipm.trajectory(&problem);
        let reached = traj.pos(lipm.t_end());
        assert!(
            (reached - target).norm() < 1e-4,
            "LIPM reached {reached:?}, want {target:?}"
        );
        // Starts at the origin.
        assert!(traj.pos(0.0).norm() < 1e-6);
    }
}

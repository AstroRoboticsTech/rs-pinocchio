//! A polynomial whose coefficients are QP decision variables (PlaCo
//! `ProblemPolynom`).

use nalgebra::DMatrix;

use super::expression::Expression;
use super::problem::Problem;
use super::variable::Variable;
use crate::placo::tools::Polynom;

/// Wraps a [`Variable`] as the coefficient vector (highest degree first) of a
/// polynomial, so its values at given abscissae are affine QP expressions.
#[derive(Clone, Copy, Debug)]
pub struct ProblemPolynom {
    variable: Variable,
}

impl ProblemPolynom {
    /// Wraps `variable` as polynomial coefficients.
    pub fn new(variable: Variable) -> Self {
        Self { variable }
    }

    /// Expression for the polynomial's `derivative`-th derivative at `x`.
    pub fn expr(&self, x: f64, derivative: i32) -> Expression {
        let size = self.variable.size();
        let mut coefficients = DMatrix::zeros(1, size);
        let mut x_pow = 1.0;
        let mut order = derivative;
        while (order as usize) < size {
            let idx = size - order as usize - 1;
            coefficients[(0, idx)] =
                Polynom::derivative_coefficient(order, derivative) as f64 * x_pow;
            x_pow *= x;
            order += 1;
        }
        self.variable.expr().left_multiply(&coefficients)
    }

    /// The concrete polynomial after the problem is solved.
    pub fn polynom(&self, problem: &Problem) -> Polynom {
        Polynom::new(self.variable.value(problem))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placo::problem::ConstraintPriority;

    #[test]
    fn fits_a_line_through_two_points() {
        // Degree-1 polynom p(x) = a x + b, fit p(0)=1, p(1)=3 -> a=2, b=1.
        let mut problem = Problem::new();
        let coeffs = problem.add_variable(2); // [a, b]
        let poly = ProblemPolynom::new(coeffs);
        problem.add_constraint(poly.expr(0.0, 0).equal_scalar(1.0));
        problem.add_constraint(poly.expr(1.0, 0).equal_scalar(3.0));
        // Regularize so the QP is well-posed.
        problem
            .add_constraint(coeffs.expr().equal_vector(nalgebra::DVector::zeros(2)))
            .configure(ConstraintPriority::Soft, 1e-6);
        problem.solve().unwrap();

        let p = poly.polynom(&problem);
        assert!((p.value(0.0, 0) - 1.0).abs() < 1e-5);
        assert!((p.value(1.0, 0) - 3.0).abs() < 1e-5);
        assert!((p.value(2.0, 0) - 5.0).abs() < 1e-5); // a*2+b = 5
    }
}

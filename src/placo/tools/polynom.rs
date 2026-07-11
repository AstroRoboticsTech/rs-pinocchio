//! Dense polynomial with derivative evaluation (PlaCo `Polynom`).

use nalgebra::DVector;

/// A polynomial stored by its coefficients, highest degree first.
#[derive(Clone, Debug, PartialEq)]
pub struct Polynom {
    /// Coefficients, from highest degree to lowest.
    pub coefficients: DVector<f64>,
}

impl Polynom {
    /// Builds a polynomial from its coefficients (highest degree first).
    pub fn new(coefficients: DVector<f64>) -> Self {
        Self { coefficients }
    }

    /// The coefficient in front of the degree-`degree` term after `derivative`
    /// differentiations (the falling factorial `degree · (degree-1) · …`).
    ///
    /// Returns `0` when `derivative > degree`.
    pub fn derivative_coefficient(mut degree: i32, mut derivative: i32) -> i64 {
        if derivative > degree {
            return 0;
        }
        let mut coefficient: i64 = 1;
        while derivative > 0 {
            coefficient *= degree as i64;
            degree -= 1;
            derivative -= 1;
        }
        coefficient
    }

    /// Evaluates the polynomial (or its `derivative`-th derivative) at `x`.
    pub fn value(&self, x: f64, derivative: i32) -> f64 {
        let n = self.coefficients.len() as i32;
        let mut p = 0.0;
        let mut x_pow = 1.0;
        let mut order = derivative;
        while order < n {
            let idx = (n - order - 1) as usize;
            p += Self::derivative_coefficient(order, derivative) as f64
                * self.coefficients[idx]
                * x_pow;
            x_pow *= x;
            order += 1;
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivative_coefficient_falling_factorial() {
        assert_eq!(Polynom::derivative_coefficient(3, 0), 1);
        assert_eq!(Polynom::derivative_coefficient(3, 1), 3);
        assert_eq!(Polynom::derivative_coefficient(3, 2), 6); // 3*2
        assert_eq!(Polynom::derivative_coefficient(3, 3), 6); // 3*2*1
        assert_eq!(Polynom::derivative_coefficient(2, 3), 0); // derivative > degree
    }

    #[test]
    fn evaluates_polynomial_and_derivatives() {
        // p(x) = 2x^2 + 3x + 1  (coeffs highest-first)
        let p = Polynom::new(DVector::from_vec(vec![2.0, 3.0, 1.0]));
        assert_eq!(p.value(0.0, 0), 1.0);
        assert_eq!(p.value(2.0, 0), 2.0 * 4.0 + 3.0 * 2.0 + 1.0); // 15
                                                                  // p'(x) = 4x + 3
        assert_eq!(p.value(2.0, 1), 4.0 * 2.0 + 3.0); // 11
                                                      // p''(x) = 4
        assert_eq!(p.value(5.0, 2), 4.0);
        // p'''(x) = 0
        assert_eq!(p.value(5.0, 3), 0.0);
    }
}

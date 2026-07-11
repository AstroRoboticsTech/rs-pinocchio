//! Decision-variable handles (PlaCo `Variable`).

use nalgebra::{DMatrix, DVector};

use super::expression::Expression;
use super::problem::Problem;

/// A handle to a block of decision variables in a [`Problem`].
///
/// Unlike PlaCo's C++ `Variable` (which stores its own value and a back-pointer
/// to the problem), this is a light `Copy` handle: after [`Problem::solve`] read
/// its value with [`Variable::value`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variable {
    /// Start offset (inclusive) in the problem's variable vector.
    pub k_start: usize,
    /// End offset (exclusive) in the problem's variable vector.
    pub k_end: usize,
}

impl Variable {
    /// Number of scalar decision variables in this block.
    pub fn size(&self) -> usize {
        self.k_end - self.k_start
    }

    /// An [`Expression`] selecting the whole variable.
    pub fn expr(&self) -> Expression {
        self.expr_slice(0, self.size())
    }

    /// An [`Expression`] selecting `rows` entries starting at `start` within the
    /// variable.
    pub fn expr_slice(&self, start: usize, rows: usize) -> Expression {
        let mut a = DMatrix::zeros(rows, self.k_end);
        for k in 0..rows {
            a[(k, self.k_start + start + k)] = 1.0;
        }
        Expression {
            a,
            b: DVector::zeros(rows),
        }
    }

    /// The solved value of this variable (call after [`Problem::solve`]).
    pub fn value(&self, problem: &Problem) -> DVector<f64> {
        problem
            .solution()
            .rows(self.k_start, self.size())
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_is_identity_selector() {
        let v = Variable {
            k_start: 2,
            k_end: 5,
        };
        assert_eq!(v.size(), 3);
        let e = v.expr();
        assert_eq!(e.rows(), 3);
        assert_eq!(e.cols(), 5);
        assert_eq!(e.a[(0, 2)], 1.0);
        assert_eq!(e.a[(1, 3)], 1.0);
        assert_eq!(e.a[(2, 4)], 1.0);
    }

    #[test]
    fn expr_slice_selects_subrange() {
        let v = Variable {
            k_start: 0,
            k_end: 4,
        };
        let e = v.expr_slice(1, 2);
        assert_eq!(e.rows(), 2);
        assert_eq!(e.a[(0, 1)], 1.0);
        assert_eq!(e.a[(1, 2)], 1.0);
    }
}

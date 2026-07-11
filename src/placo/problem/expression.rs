//! Affine expressions `A·x + b` over the decision variables (PlaCo `Expression`).

use nalgebra::{DMatrix, DVector};

use super::constraint::Constraint;

/// A linear combination of decision variables, `A·x + b`, that constraints and
/// objectives are built from.
///
/// Arithmetic is available both as methods and through the standard operators on
/// owned values (`e1 + e2`, `-e`, `e * 2.0`). Comparisons build [`Constraint`]s
/// via [`Expression::geq`], [`Expression::leq`] and [`Expression::equal_to`]
/// (Rust cannot overload `>=`/`==` to return a constraint).
#[derive(Clone, Debug, PartialEq)]
pub struct Expression {
    /// The `A` matrix in `A·x + b`.
    pub a: DMatrix<f64>,
    /// The `b` vector in `A·x + b`.
    pub b: DVector<f64>,
}

impl Expression {
    /// An empty expression (0 rows, 0 cols).
    pub fn empty() -> Self {
        Self {
            a: DMatrix::zeros(0, 0),
            b: DVector::zeros(0),
        }
    }

    /// A constant expression from a vector (`A` has zero columns).
    pub fn from_vector(v: DVector<f64>) -> Self {
        let rows = v.len();
        Self {
            a: DMatrix::zeros(rows, 0),
            b: v,
        }
    }

    /// A one-row constant expression from a scalar.
    pub fn from_double(value: f64) -> Self {
        Self {
            a: DMatrix::zeros(1, 0),
            b: DVector::from_element(1, value),
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.a.nrows()
    }

    /// Number of columns (decision variables referenced).
    pub fn cols(&self) -> usize {
        self.a.ncols()
    }

    /// Whether the expression has a single row.
    pub fn is_scalar(&self) -> bool {
        self.rows() == 1
    }

    /// Whether the expression is constant (does not depend on any variable).
    pub fn is_constant(&self) -> bool {
        self.cols() == 0
    }

    /// Rows `[start, start + rows)` of the expression.
    pub fn slice(&self, start: usize, rows: usize) -> Expression {
        Expression {
            a: self.a.rows(start, rows).into_owned(),
            b: self.b.rows(start, rows).into_owned(),
        }
    }

    /// Adds `f` to every row (broadcast scalar add).
    pub fn piecewise_add(&self, f: f64) -> Expression {
        let mut e = self.clone();
        e.b.add_scalar_mut(f);
        e
    }

    /// Left-multiplies by matrix `m`: `M·(A·x + b)`.
    pub fn left_multiply(&self, m: &DMatrix<f64>) -> Expression {
        Expression {
            a: m * &self.a,
            b: m * &self.b,
        }
    }

    /// Multiplies two expressions, where one is scalar (a single row) and the
    /// other constant (no variables): broadcasts the scalar affine expression by
    /// the constant vector (PlaCo's `Expression::operator*(Expression)`).
    ///
    /// # Panics
    /// If neither operand is a scalar-times-constant pairing.
    pub fn mul_expr(&self, other: &Expression) -> Expression {
        if self.is_scalar() && other.is_constant() {
            let mut a = DMatrix::zeros(other.rows(), self.cols());
            let mut b = DVector::zeros(other.rows());
            for k in 0..other.rows() {
                a.row_mut(k).copy_from(&(self.a.row(0) * other.b[k]));
                b[k] = self.b[0] * other.b[k];
            }
            Expression { a, b }
        } else if other.is_scalar() && self.is_constant() {
            other.mul_expr(self)
        } else {
            panic!("mul_expr: one expression must be scalar and the other constant");
        }
    }

    /// Reduces a multi-row expression to the sum of its rows (one row out).
    pub fn sum(&self) -> Expression {
        let mut a = DMatrix::zeros(1, self.cols());
        for k in 0..self.rows() {
            for j in 0..self.cols() {
                a[(0, j)] += self.a[(k, j)];
            }
        }
        Expression {
            a,
            b: DVector::from_element(1, self.b.sum()),
        }
    }

    /// Reduces a multi-row expression to the mean of its rows.
    ///
    /// Mirrors PlaCo, which divides the summed row by the number of columns
    /// (decision variables), not the number of rows.
    pub fn mean(&self) -> Expression {
        let cols = self.cols() as f64;
        let s = self.sum();
        Expression {
            a: s.a / cols,
            b: s.b / cols,
        }
    }

    /// Stacks `self` on top of `other` (PlaCo's `/` operator).
    pub fn vstack(&self, other: &Expression) -> Expression {
        let cols = self.cols().max(other.cols());
        let rows = self.rows() + other.rows();
        let mut a = DMatrix::zeros(rows, cols);
        let mut b = DVector::zeros(rows);
        a.view_mut((0, 0), (self.rows(), self.cols()))
            .copy_from(&self.a);
        a.view_mut((self.rows(), 0), (other.rows(), other.cols()))
            .copy_from(&other.a);
        b.rows_mut(0, self.rows()).copy_from(&self.b);
        b.rows_mut(self.rows(), other.rows()).copy_from(&other.b);
        Expression { a, b }
    }

    /// The value of the expression given a decision-variable vector `x`.
    pub fn value(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.a * x.rows(0, self.cols()) + &self.b
    }

    /// Adds two expressions, broadcasting a scalar constant over the rows.
    pub fn add(&self, other: &Expression) -> Expression {
        if self.is_scalar() && self.is_constant() {
            return other.piecewise_add(self.b[0]);
        }
        if other.is_scalar() && other.is_constant() {
            return self.piecewise_add(other.b[0]);
        }
        assert_eq!(
            self.rows(),
            other.rows(),
            "Expression::add: mismatched rows ({} vs {})",
            self.rows(),
            other.rows()
        );
        let cols = self.cols().max(other.cols());
        let mut a = DMatrix::zeros(self.rows(), cols);
        a.view_mut((0, 0), (self.rows(), self.cols()))
            .copy_from(&self.a);
        {
            let mut left = a.view_mut((0, 0), (other.rows(), other.cols()));
            left += &other.a;
        }
        Expression {
            a,
            b: &self.b + &other.b,
        }
    }

    /// Subtracts `other` from `self`.
    pub fn subtract(&self, other: &Expression) -> Expression {
        self.add(&other.scale(-1.0))
    }

    /// Scales the whole expression by `f`.
    pub fn scale(&self, f: f64) -> Expression {
        Expression {
            a: &self.a * f,
            b: &self.b * f,
        }
    }

    /// Adds a constant vector to `b`.
    pub fn add_vector(&self, v: &DVector<f64>) -> Expression {
        let mut e = self.clone();
        e.b += v;
        e
    }

    /// Subtracts a constant vector from `b`.
    pub fn sub_vector(&self, v: &DVector<f64>) -> Expression {
        let mut e = self.clone();
        e.b -= v;
        e
    }

    // --- constraint builders -------------------------------------------------

    /// `self >= other` (inequality `self - other >= 0`).
    pub fn geq(&self, other: &Expression) -> Constraint {
        Constraint::inequality(self.subtract(other))
    }

    /// `self <= other` (inequality `other - self >= 0`).
    pub fn leq(&self, other: &Expression) -> Constraint {
        Constraint::inequality(self.subtract(other).scale(-1.0))
    }

    /// `self == other` (equality `self - other = 0`).
    pub fn equal_to(&self, other: &Expression) -> Constraint {
        Constraint::equality(self.subtract(other))
    }

    /// `self >= f` for a scalar bound.
    pub fn geq_scalar(&self, f: f64) -> Constraint {
        self.geq(&Expression::from_double(f))
    }

    /// `self <= f` for a scalar bound.
    pub fn leq_scalar(&self, f: f64) -> Constraint {
        self.leq(&Expression::from_double(f))
    }

    /// `self == f` for a scalar bound.
    pub fn equal_scalar(&self, f: f64) -> Constraint {
        self.equal_to(&Expression::from_double(f))
    }

    /// `self >= v` for a vector bound.
    pub fn geq_vector(&self, v: DVector<f64>) -> Constraint {
        self.geq(&Expression::from_vector(v))
    }

    /// `self <= v` for a vector bound.
    pub fn leq_vector(&self, v: DVector<f64>) -> Constraint {
        self.leq(&Expression::from_vector(v))
    }

    /// `self == v` for a vector bound.
    pub fn equal_vector(&self, v: DVector<f64>) -> Constraint {
        self.equal_to(&Expression::from_vector(v))
    }
}

// --- operator sugar on owned values -----------------------------------------

impl std::ops::Add for Expression {
    type Output = Expression;
    fn add(self, other: Expression) -> Expression {
        Expression::add(&self, &other)
    }
}

impl std::ops::Sub for Expression {
    type Output = Expression;
    fn sub(self, other: Expression) -> Expression {
        self.subtract(&other)
    }
}

impl std::ops::Neg for Expression {
    type Output = Expression;
    fn neg(self) -> Expression {
        self.scale(-1.0)
    }
}

impl std::ops::Mul<f64> for Expression {
    type Output = Expression;
    fn mul(self, f: f64) -> Expression {
        self.scale(f)
    }
}

impl std::ops::Mul<Expression> for f64 {
    type Output = Expression;
    fn mul(self, e: Expression) -> Expression {
        e.scale(self)
    }
}

/// `M · Expression`.
impl std::ops::Mul<&Expression> for &DMatrix<f64> {
    type Output = Expression;
    fn mul(self, e: &Expression) -> Expression {
        e.left_multiply(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placo::problem::constraint::ConstraintType;

    fn var_expr(k_start: usize, k_end: usize) -> Expression {
        // A single variable's identity expression (mirrors Variable::expr).
        let size = k_end - k_start;
        let mut a = DMatrix::zeros(size, k_end);
        for k in 0..size {
            a[(k, k_start + k)] = 1.0;
        }
        Expression {
            a,
            b: DVector::zeros(size),
        }
    }

    #[test]
    fn mean_divides_by_cols() {
        // Two rows over three variables: mean divides the summed row by cols (3).
        let e = Expression {
            a: DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 3.0, 4.0, 5.0]),
            b: DVector::from_vec(vec![6.0, 12.0]),
        };
        let m = e.mean();
        assert_eq!(m.rows(), 1);
        assert!((m.a[(0, 0)] - 4.0 / 3.0).abs() < 1e-12);
        assert!((m.b[0] - 18.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn mul_expr_broadcasts_scalar_by_constant() {
        // scalar (1 row, 2 vars) times constant vector [10, 20].
        let scalar = Expression {
            a: DMatrix::from_row_slice(1, 2, &[1.0, 2.0]),
            b: DVector::from_element(1, 3.0),
        };
        let constant = Expression::from_vector(DVector::from_vec(vec![10.0, 20.0]));
        let p = scalar.mul_expr(&constant);
        assert_eq!(p.rows(), 2);
        assert_eq!(p.cols(), 2);
        assert!((p.a[(0, 0)] - 10.0).abs() < 1e-12 && (p.a[(0, 1)] - 20.0).abs() < 1e-12);
        assert!((p.a[(1, 0)] - 20.0).abs() < 1e-12 && (p.a[(1, 1)] - 40.0).abs() < 1e-12);
        assert!((p.b[0] - 30.0).abs() < 1e-12 && (p.b[1] - 60.0).abs() < 1e-12);
        // Commuted form (constant * scalar) yields the same result.
        let q = constant.mul_expr(&scalar);
        assert_eq!(p, q);
    }

    #[test]
    fn add_pads_to_max_cols() {
        // Same rows, different column counts: add pads to max cols.
        let x = var_expr(0, 2); // 2 rows x 2 cols
        let y = Expression {
            a: DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            b: DVector::zeros(2),
        }; // 2 rows x 3 cols
        let s = x.add(&y);
        assert_eq!(s.rows(), 2);
        assert_eq!(s.cols(), 3);
        // Row 0: x0 + y0 -> coefficient 2 at col 0
        assert_eq!(s.a[(0, 0)], 2.0);
        assert_eq!(s.a[(1, 1)], 2.0);
    }

    #[test]
    fn scalar_constant_broadcast_add() {
        let x = var_expr(0, 3);
        let c = Expression::from_double(5.0);
        let s = x.add(&c);
        assert_eq!(s.rows(), 3);
        for k in 0..3 {
            assert_eq!(s.b[k], 5.0);
        }
    }

    #[test]
    fn value_evaluates_affine() {
        let x = var_expr(0, 2);
        let e = x.scale(2.0).add_vector(&DVector::from_vec(vec![1.0, -1.0]));
        let val = e.value(&DVector::from_vec(vec![3.0, 4.0]));
        assert_eq!(val, DVector::from_vec(vec![7.0, 7.0])); // 2*3+1, 2*4-1
    }

    #[test]
    fn geq_builds_inequality() {
        let x = var_expr(0, 1);
        let c = x.geq_scalar(2.0);
        assert_eq!(c.type_, ConstraintType::Inequality);
        // expression is x - 2 >= 0
        assert_eq!(c.expression.b[0], -2.0);
        assert_eq!(c.expression.a[(0, 0)], 1.0);
    }

    #[test]
    fn vstack_stacks_rows() {
        let x = var_expr(0, 2);
        let y = var_expr(2, 4);
        let s = x.vstack(&y);
        assert_eq!(s.rows(), 4);
        assert_eq!(s.cols(), 4);
    }
}

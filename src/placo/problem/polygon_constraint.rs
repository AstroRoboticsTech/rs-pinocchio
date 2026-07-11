//! Helpers building 2D "point inside polygon" QP constraints (PlaCo
//! `PolygonConstraint`).

use nalgebra::{DMatrix, DVector, Vector2};

use super::constraint::Constraint;
use super::expression::Expression;

/// Constrains a 2-row expression `(x, y)` to lie inside a **clockwise** polygon,
/// with an optional inward `margin`.
///
/// # Panics
/// If `expression_xy` does not have exactly 2 rows.
pub fn in_polygon_xy(
    expression_xy: &Expression,
    polygon: &[Vector2<f64>],
    margin: f64,
) -> Constraint {
    assert_eq!(
        expression_xy.rows(),
        2,
        "in_polygon_xy: expected a 2-row expression"
    );

    let n = polygon.len();
    let cols = expression_xy.cols();
    let mut a = DMatrix::zeros(n, cols);
    let mut b = DVector::zeros(n);

    for i in 0..n {
        let j = (i + 1) % n;
        let (pa, pb) = (polygon[i], polygon[j]);

        // Inward normal for a clockwise polygon.
        let mut normal = Vector2::new((pb - pa).y, (pa - pb).x);
        normal.normalize_mut();

        // n^T (P - A) - margin >= 0.
        let normal_mat = DMatrix::from_row_slice(1, 2, normal.as_slice());
        let shifted = expression_xy.sub_vector(&DVector::from_column_slice(pa.as_slice()));
        let value = shifted.left_multiply(&normal_mat).piecewise_add(-margin);

        a.view_mut((i, 0), (1, cols)).copy_from(&value.a);
        b[i] = value.b[0];
    }

    Expression { a, b }.geq(&Expression::from_vector(DVector::zeros(n)))
}

/// Same as [`in_polygon_xy`] but from separate `x` and `y` expressions.
pub fn in_polygon(
    expression_x: &Expression,
    expression_y: &Expression,
    polygon: &[Vector2<f64>],
    margin: f64,
) -> Constraint {
    let e = expression_x.vstack(expression_y);
    in_polygon_xy(&e, polygon, margin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placo::problem::{ConstraintPriority, Problem};

    fn unit_square_cw() -> Vec<Vector2<f64>> {
        // Clockwise square [-1,1]^2 (exterior on the trigonometric normal).
        vec![
            Vector2::new(-1.0, -1.0),
            Vector2::new(-1.0, 1.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(1.0, -1.0),
        ]
    }

    #[test]
    fn keeps_point_inside_square() {
        // min ||p - (5, 0)||^2 s.t. p in [-1,1]^2 -> p = (1, 0).
        let mut problem = Problem::new();
        let p = problem.add_variable(2);
        problem
            .add_constraint(p.expr_slice(0, 1).equal_scalar(5.0))
            .configure(ConstraintPriority::Soft, 1.0);
        problem
            .add_constraint(p.expr_slice(1, 1).equal_scalar(0.0))
            .configure(ConstraintPriority::Soft, 1.0);
        problem.add_constraint(in_polygon_xy(&p.expr(), &unit_square_cw(), 0.0));
        problem.solve().unwrap();
        let v = p.value(&problem);
        assert!((v[0] - 1.0).abs() < 1e-4, "x = {}", v[0]);
        assert!(v[1].abs() < 1e-4, "y = {}", v[1]);
    }
}

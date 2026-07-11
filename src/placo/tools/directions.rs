//! Uniformly spread direction sets in 2D and 3D (PlaCo `directions`).

use std::f64::consts::PI;

use nalgebra::DMatrix;

/// `n` evenly-spaced unit directions on the circle, as an `n × 2` matrix.
pub fn directions_2d(n: usize) -> DMatrix<f64> {
    let mut directions = DMatrix::zeros(n, 2);
    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / n as f64;
        directions[(i, 0)] = angle.cos();
        directions[(i, 1)] = angle.sin();
    }
    directions
}

/// `n` roughly-uniform unit directions on the sphere via a Fibonacci lattice,
/// as an `n × 3` matrix. `epsilon` tunes the pole distribution.
pub fn directions_3d(n: usize, epsilon: f64) -> DMatrix<f64> {
    let mut directions = DMatrix::zeros(n, 3);
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    for i in 0..n {
        let x = (i as f64 / phi).rem_euclid(1.0);
        let y = (i as f64 + epsilon) / (n as f64 - 1.0 + 2.0 * epsilon);
        let alpha = 2.0 * PI * x;
        let beta = (1.0 - 2.0 * y).acos();
        directions[(i, 0)] = beta.sin() * alpha.cos();
        directions[(i, 1)] = beta.sin() * alpha.sin();
        directions[(i, 2)] = beta.cos();
    }
    directions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directions_2d_are_unit_and_evenly_spaced() {
        let d = directions_2d(4);
        assert_eq!(d.shape(), (4, 2));
        for i in 0..4 {
            let n = (d[(i, 0)].powi(2) + d[(i, 1)].powi(2)).sqrt();
            assert!((n - 1.0).abs() < 1e-12);
        }
        assert!((d[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((d[(1, 1)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn directions_3d_are_unit() {
        let d = directions_3d(50, 0.5);
        assert_eq!(d.shape(), (50, 3));
        for i in 0..50 {
            let n = (d[(i, 0)].powi(2) + d[(i, 1)].powi(2) + d[(i, 2)].powi(2)).sqrt();
            assert!((n - 1.0).abs() < 1e-9);
        }
    }
}

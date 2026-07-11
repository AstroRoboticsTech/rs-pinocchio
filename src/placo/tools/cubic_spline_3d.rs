//! Three independent cubic splines driven together (PlaCo `CubicSpline3D`).

use nalgebra::Vector3;

use super::cubic_spline::CubicSpline;

/// A 3D cubic spline: one [`CubicSpline`] per axis, sharing the same knot times.
#[derive(Clone, Debug, Default)]
pub struct CubicSpline3D {
    x: CubicSpline,
    y: CubicSpline,
    z: CubicSpline,
}

impl CubicSpline3D {
    /// Builds an empty 3D spline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a knot at time `t` with value `x` and velocity `dx`.
    pub fn add_point(&mut self, t: f64, x: Vector3<f64>, dx: Vector3<f64>) {
        self.x.add_point(t, x[0], dx[0]);
        self.y.add_point(t, x[1], dx[1]);
        self.z.add_point(t, x[2], dx[2]);
    }

    /// Clears all knots.
    pub fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
        self.z.clear();
    }

    /// Total time spanned by the spline.
    pub fn duration(&self) -> f64 {
        self.x.duration()
    }

    /// Position at time `t`.
    pub fn pos(&mut self, t: f64) -> Vector3<f64> {
        Vector3::new(self.x.pos(t), self.y.pos(t), self.z.pos(t))
    }

    /// Velocity at time `t`.
    pub fn vel(&mut self, t: f64) -> Vector3<f64> {
        Vector3::new(self.x.vel(t), self.y.vel(t), self.z.vel(t))
    }

    /// Acceleration at time `t`.
    pub fn acc(&mut self, t: f64) -> Vector3<f64> {
        Vector3::new(self.x.acc(t), self.y.acc(t), self.z.acc(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_each_axis() {
        let mut s = CubicSpline3D::new();
        s.add_point(0.0, Vector3::new(0.0, 0.0, 0.0), Vector3::zeros());
        s.add_point(1.0, Vector3::new(1.0, 2.0, 3.0), Vector3::zeros());

        let p = s.pos(1.0);
        assert!((p - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-9);
        let mid = s.pos(0.5);
        assert!((mid - Vector3::new(0.5, 1.0, 1.5)).norm() < 1e-9);
    }
}

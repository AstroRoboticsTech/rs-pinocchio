//! Piecewise-cubic spline with position + velocity knots (PlaCo `CubicSpline`).

use nalgebra::{Matrix4, Vector4};

use super::utils::wrap_angle;

/// A knot: time, value and velocity.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    /// Time.
    pub t: f64,
    /// Value.
    pub x: f64,
    /// Velocity.
    pub dx: f64,
}

/// Which quantity to read from the spline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueType {
    Value,
    Speed,
    Acceleration,
}

// Cubic segment a·t³ + b·t² + c·t + d, valid over [t_start, t_end).
#[derive(Clone, Copy, Debug)]
struct Poly {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

#[derive(Clone, Copy, Debug)]
struct Spline {
    poly: Poly,
    t_start: f64,
    t_end: f64,
}

/// A cubic spline interpolating a sequence of (time, value, velocity) knots.
///
/// Set `angular = true` to wrap successive values into a continuous angle before
/// fitting (so a spline through 179° → -179° goes the short way).
#[derive(Clone, Debug, Default)]
pub struct CubicSpline {
    angular: bool,
    dirty: bool,
    points: Vec<Point>,
    splines: Vec<Spline>,
}

impl CubicSpline {
    /// Builds an empty spline. `angular` enables angle-continuous fitting.
    pub fn new(angular: bool) -> Self {
        Self {
            angular,
            dirty: true,
            points: Vec::new(),
            splines: Vec::new(),
        }
    }

    /// Adds a knot at time `t` with value `x` and velocity `dx`.
    ///
    /// # Panics
    /// If `t` is not strictly greater than the previous knot's time.
    pub fn add_point(&mut self, t: f64, mut x: f64, dx: f64) {
        if self.angular {
            if let Some(last) = self.points.last() {
                x = last.x + wrap_angle(x - last.x);
            }
        }
        if let Some(last) = self.points.last() {
            if t <= last.t {
                panic!("CubicSpline: point added before a previous one (t={t})");
            }
        }
        self.points.push(Point { t, x, dx });
        self.dirty = true;
    }

    /// Clears all knots and cached segments.
    pub fn clear(&mut self) {
        self.points.clear();
        self.splines.clear();
        self.dirty = true;
    }

    /// Total time spanned by the spline.
    pub fn duration(&self) -> f64 {
        let n = self.splines.len();
        self.splines[n - 1].t_end - self.splines[0].t_start
    }

    /// Position at time `t` (clamped to the spline's time range).
    pub fn pos(&mut self, t: f64) -> f64 {
        self.build();
        self.eval(t, ValueType::Value)
    }

    /// Velocity at time `t`.
    pub fn vel(&mut self, t: f64) -> f64 {
        self.build();
        self.eval(t, ValueType::Speed)
    }

    /// Acceleration at time `t`.
    pub fn acc(&mut self, t: f64) -> f64 {
        self.build();
        self.eval(t, ValueType::Acceleration)
    }

    /// Forces the internal cubic segments to be (re)computed, so the immutable
    /// `*_at` evaluators can be called afterwards. Idempotent.
    pub fn build(&mut self) {
        if self.dirty {
            self.compute_splines();
            self.dirty = false;
        }
    }

    /// Position at `t` without rebuilding. Call [`build`](Self::build) first.
    pub fn value_at(&self, t: f64) -> f64 {
        self.eval(t, ValueType::Value)
    }

    /// Velocity at `t` without rebuilding. Call [`build`](Self::build) first.
    pub fn speed_at(&self, t: f64) -> f64 {
        self.eval(t, ValueType::Speed)
    }

    /// The knots currently in the spline.
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    fn eval(&self, mut t: f64, type_: ValueType) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        if self.points.len() == 1 {
            let p = self.points[0];
            return if type_ == ValueType::Value { p.x } else { p.dx };
        }

        let first = self.splines[0];
        let last = self.splines[self.splines.len() - 1];
        if t < first.t_start {
            t = first.t_start;
        }
        if t > last.t_end {
            t = last.t_end;
        }

        for s in &self.splines {
            if t >= s.t_start && t <= s.t_end {
                let dt = t - s.t_start;
                return match type_ {
                    ValueType::Value => polynom_value(dt, &s.poly),
                    ValueType::Speed => polynom_diff(dt, &s.poly),
                    ValueType::Acceleration => polynom_diff2(dt, &s.poly),
                };
            }
        }
        0.0
    }

    fn compute_splines(&mut self) {
        self.splines.clear();
        if self.points.len() < 2 {
            return;
        }
        for i in 1..self.points.len() {
            let (p0, p1) = (self.points[i - 1], self.points[i]);
            let t_start = p0.t;
            let poly = fit(p0.t - t_start, p0.x, p0.dx, p1.t - t_start, p1.x, p1.dx);
            self.splines.push(Spline {
                poly,
                t_start: p0.t,
                t_end: p1.t,
            });
        }
    }
}

fn polynom_value(t: f64, p: &Poly) -> f64 {
    p.d + t * (t * (p.a * t + p.b) + p.c)
}

fn polynom_diff(t: f64, p: &Poly) -> f64 {
    t * (3.0 * p.a * t + 2.0 * p.b) + p.c
}

fn polynom_diff2(t: f64, p: &Poly) -> f64 {
    6.0 * p.a * t + 2.0 * p.b
}

fn fit(t1: f64, x1: f64, dx1: f64, t2: f64, x2: f64, dx2: f64) -> Poly {
    let (t1_2, t1_3) = (t1 * t1, t1 * t1 * t1);
    let (t2_2, t2_3) = (t2 * t2, t2 * t2 * t2);

    #[rustfmt::skip]
    let m = Matrix4::new(
        t1_3,        t1_2,     t1,  1.0,
        3.0 * t1_2,  2.0 * t1, 1.0, 0.0,
        t2_3,        t2_2,     t2,  1.0,
        3.0 * t2_2,  2.0 * t2, 1.0, 0.0,
    );
    let v = Vector4::new(x1, dx1, x2, dx2);
    let abcd = m
        .try_inverse()
        .expect("CubicSpline::fit: singular fit matrix")
        * v;
    Poly {
        a: abcd[0],
        b: abcd[1],
        c: abcd[2],
        d: abcd[3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_endpoints_and_velocities() {
        let mut s = CubicSpline::new(false);
        s.add_point(0.0, 0.0, 0.0);
        s.add_point(1.0, 1.0, 0.0);

        assert!((s.pos(0.0) - 0.0).abs() < 1e-9);
        assert!((s.pos(1.0) - 1.0).abs() < 1e-9);
        // Zero velocity imposed at both ends.
        assert!((s.vel(0.0)).abs() < 1e-9);
        assert!((s.vel(1.0)).abs() < 1e-9);
        // Symmetric hermite spline: midpoint at 0.5.
        assert!((s.pos(0.5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn clamps_outside_range() {
        let mut s = CubicSpline::new(false);
        s.add_point(0.0, 2.0, 0.0);
        s.add_point(1.0, 5.0, 0.0);
        assert!((s.pos(-1.0) - 2.0).abs() < 1e-9);
        assert!((s.pos(2.0) - 5.0).abs() < 1e-9);
        assert!((s.duration() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn angular_takes_short_way() {
        let mut s = CubicSpline::new(true);
        let deg = std::f64::consts::PI / 180.0;
        s.add_point(0.0, 179.0 * deg, 0.0);
        s.add_point(1.0, -179.0 * deg, 0.0);
        // Continuous unwrap: second knot stored as 181°, so midpoint ~180°.
        let mid = s.pos(0.5);
        assert!((mid - 180.0 * deg).abs() < 1e-6, "mid = {mid}");
    }

    #[test]
    #[should_panic]
    fn rejects_non_increasing_time() {
        let mut s = CubicSpline::new(false);
        s.add_point(1.0, 0.0, 0.0);
        s.add_point(1.0, 1.0, 0.0);
    }
}

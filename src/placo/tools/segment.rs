//! 2D segment geometry: parallelism, alignment, intersection (PlaCo `Segment`).

use nalgebra::Vector2;

/// A 2D line segment from `start` to `end`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// Segment start.
    pub start: Vector2<f64>,
    /// Segment end.
    pub end: Vector2<f64>,
}

impl Segment {
    /// Builds a segment from its endpoints.
    pub fn new(start: Vector2<f64>, end: Vector2<f64>) -> Self {
        Self { start, end }
    }

    /// Segment length.
    pub fn norm(&self) -> f64 {
        (self.end - self.start).norm()
    }

    /// Whether this segment is parallel to `s` within `epsilon` (sine of the
    /// error angle).
    pub fn is_parallel(&self, s: &Segment, epsilon: f64) -> bool {
        let v1 = self.end - self.start;
        let v2 = s.end - s.start;
        (v1.x * v2.y - v1.y * v2.x).abs() / (v1.norm() * v2.norm()) < epsilon
    }

    /// Whether `point` lies on the infinite line through this segment.
    pub fn is_point_aligned(&self, point: &Vector2<f64>, epsilon: f64) -> bool {
        if (self.start - point).norm().abs() < epsilon {
            return true;
        }
        self.is_parallel(&Segment::new(self.start, *point), epsilon)
    }

    /// Whether `s` is collinear with this segment.
    pub fn is_segment_aligned(&self, s: &Segment, epsilon: f64) -> bool {
        self.is_point_aligned(&s.start, epsilon) && self.is_point_aligned(&s.end, epsilon)
    }

    /// Whether `point` lies on this segment (between the endpoints).
    pub fn is_point_in_segment(&self, point: &Vector2<f64>, epsilon: f64) -> bool {
        let v1 = self.end - self.start;
        let v2 = point - self.start;
        self.is_segment_aligned(&Segment::new(self.start, *point), epsilon)
            && v1.dot(&v2) >= 0.0
            && v1.dot(&v2) <= v1.dot(&v1)
    }

    /// The `(λ₁, λ₂)` parameters of the intersection of the two supporting lines.
    ///
    /// # Panics
    /// If the segments are parallel.
    pub fn get_lambdas(&self, s: &Segment) -> (f64, f64) {
        if self.is_parallel(s, 1e-5) {
            panic!("Segment: can't compute intersection of parallels");
        }
        let v1 = self.end - self.start;
        let v2 = s.end - s.start;
        let p1 = self.start;
        let p2 = s.start;
        let det = v1.x * v2.y - v1.y * v2.x;
        let l1 = (v2.y * (p2.x - p1.x) + v2.x * (p1.y - p2.y)) / det;
        let l2 = (v1.y * (p2.x - p1.x) + v1.x * (p1.y - p2.y)) / det;
        (l1, l2)
    }

    /// Whether the two segments intersect (the lines cross within both spans).
    pub fn intersects(&self, s: &Segment) -> bool {
        self.line_pass_through(s) && s.line_pass_through(self)
    }

    /// Whether the supporting line of `s` crosses within this segment's span.
    pub fn line_pass_through(&self, s: &Segment) -> bool {
        let (l1, _) = self.get_lambdas(s);
        (0.0..=1.0).contains(&l1)
    }

    /// Whether the half-line from `start` through `end` crosses `s`.
    pub fn half_line_pass_through(&self, s: &Segment) -> bool {
        let (l1, l2) = self.get_lambdas(s);
        (0.0..=1.0).contains(&l1) && l2 >= 0.0
    }

    /// The intersection point of the two supporting lines.
    pub fn lines_intersection(&self, s: &Segment) -> Vector2<f64> {
        let (l1, _) = self.get_lambdas(s);
        let v = self.end - self.start;
        self.start + l1 * v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64) -> Vector2<f64> {
        Vector2::new(x, y)
    }

    #[test]
    fn norm_and_parallel() {
        let a = Segment::new(v(0.0, 0.0), v(3.0, 4.0));
        assert!((a.norm() - 5.0).abs() < 1e-12);
        let b = Segment::new(v(1.0, 1.0), v(4.0, 5.0));
        assert!(a.is_parallel(&b, 1e-6));
        let c = Segment::new(v(0.0, 0.0), v(1.0, 0.0));
        assert!(!a.is_parallel(&c, 1e-6));
    }

    #[test]
    fn crossing_segments_intersect_at_center() {
        let a = Segment::new(v(-1.0, 0.0), v(1.0, 0.0));
        let b = Segment::new(v(0.0, -1.0), v(0.0, 1.0));
        assert!(a.intersects(&b));
        let p = a.lines_intersection(&b);
        assert!((p - v(0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn point_in_segment() {
        let a = Segment::new(v(0.0, 0.0), v(2.0, 0.0));
        assert!(a.is_point_in_segment(&v(1.0, 0.0), 1e-6));
        assert!(!a.is_point_in_segment(&v(3.0, 0.0), 1e-6));
        assert!(!a.is_point_in_segment(&v(1.0, 1.0), 1e-6));
    }
}

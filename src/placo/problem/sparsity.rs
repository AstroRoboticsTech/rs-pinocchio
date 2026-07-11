//! Column-sparsity detection for QP Hessian assembly (PlaCo `Sparsity`).

use nalgebra::DMatrix;

/// A closed range of non-sparse columns `[start, end]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    /// First column in the interval.
    pub start: usize,
    /// Last column in the interval (inclusive).
    pub end: usize,
}

impl Interval {
    /// Builds an interval.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Whether column `i` is inside the interval.
    pub fn contains(&self, i: usize) -> bool {
        self.start <= i && i <= self.end
    }
}

/// The set of column intervals that are not all-zero, kept sorted and merged.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sparsity {
    /// Non-sparse column intervals.
    pub intervals: Vec<Interval>,
}

impl Sparsity {
    /// An empty sparsity pattern.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `[start, end]`, merging with any overlapping/adjacent intervals.
    pub fn add_interval(&mut self, mut start: usize, end: usize) {
        let old = std::mem::take(&mut self.intervals);
        let mut inserted = false;

        for interval in old {
            if inserted {
                self.intervals.push(interval);
            } else if end < interval.start {
                // Our interval ends strictly before the next starts.
                self.intervals.push(Interval::new(start, end));
                self.intervals.push(interval);
                inserted = true;
            } else if interval.contains(start) && interval.contains(end) {
                self.intervals.push(interval);
                inserted = true;
            } else if interval.contains(start) {
                start = interval.start;
            } else if interval.contains(end) {
                self.intervals.push(Interval::new(start, interval.end));
                inserted = true;
            } else if start > interval.start {
                self.intervals.push(interval);
            }
        }

        if !inserted {
            self.intervals.push(Interval::new(start, end));
        }
    }

    /// Detects contiguous runs of non-zero columns in `m`.
    pub fn detect_columns(m: &DMatrix<f64>) -> Sparsity {
        let mut sparsity = Sparsity::new();
        let mut last_nonzero: Option<usize> = None;

        for column in 0..m.ncols() {
            let is_zero = m.column(column).iter().all(|v| v.abs() <= 1e-12);
            if is_zero {
                if let Some(start) = last_nonzero.take() {
                    sparsity.add_interval(start, column - 1);
                }
            } else if last_nonzero.is_none() {
                last_nonzero = Some(column);
            }
        }
        if let Some(start) = last_nonzero {
            sparsity.add_interval(start, m.ncols() - 1);
        }
        sparsity
    }
}

impl std::ops::Add for &Sparsity {
    type Output = Sparsity;

    fn add(self, other: &Sparsity) -> Sparsity {
        let mut s = Sparsity::new();
        for interval in self.intervals.iter().chain(other.intervals.iter()) {
            s.add_interval(interval.start, interval.end);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_single_block() {
        let m = DMatrix::from_row_slice(2, 5, &[0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
        let s = Sparsity::detect_columns(&m);
        assert_eq!(s.intervals, vec![Interval::new(1, 2)]);
    }

    #[test]
    fn detects_two_blocks() {
        let m = DMatrix::from_row_slice(1, 6, &[1.0, 0.0, 0.0, 2.0, 3.0, 0.0]);
        let s = Sparsity::detect_columns(&m);
        assert_eq!(s.intervals, vec![Interval::new(0, 0), Interval::new(3, 4)]);
    }

    #[test]
    fn merges_added_intervals() {
        let mut s = Sparsity::new();
        s.add_interval(0, 2);
        s.add_interval(5, 6);
        s.add_interval(2, 5);
        assert_eq!(s.intervals, vec![Interval::new(0, 6)]);
    }
}

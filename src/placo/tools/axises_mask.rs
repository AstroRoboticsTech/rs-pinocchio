//! Selecting / re-framing task axes (PlaCo `AxisesMask`).

use nalgebra::{DMatrix, Matrix3};

fn matrix3_to_dmatrix(m: &Matrix3<f64>) -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 3, m.as_slice())
}

/// Reference frame in which the axis masking is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MaskFrame {
    /// Use the (usually world) frame provided by the task.
    #[default]
    Task,
    /// Use the task's local frame (via [`AxisesMask::r_local_world`]).
    Local,
    /// Use a user-provided custom frame (via [`AxisesMask::r_custom_world`]).
    Custom,
}

/// Keeps a subset of a task's axes, optionally after rotating into another frame.
///
/// ```
/// # use rs_pinocchio::placo::tools::{AxisesMask, MaskFrame};
/// let mut mask = AxisesMask::new();
/// mask.set_axises("xy", MaskFrame::Task); // keep only x and y rows
/// ```
#[derive(Clone, Debug)]
pub struct AxisesMask {
    /// Rotation from world to the task's local frame.
    pub r_local_world: Matrix3<f64>,
    /// Rotation from world to a user-defined custom frame.
    pub r_custom_world: Matrix3<f64>,
    /// Row indices kept by the mask (`0=x, 1=y, 2=z`).
    pub indices: Vec<usize>,
    /// Frame the masking is applied in.
    pub frame: MaskFrame,
}

impl Default for AxisesMask {
    fn default() -> Self {
        Self::new()
    }
}

impl AxisesMask {
    /// A mask that keeps all three axes in the task frame.
    pub fn new() -> Self {
        Self {
            r_local_world: Matrix3::identity(),
            r_custom_world: Matrix3::identity(),
            indices: vec![0, 1, 2],
            frame: MaskFrame::Task,
        }
    }

    /// Sets which axes to keep, e.g. `"xy"`, in the given frame.
    ///
    /// # Panics
    /// If `axises` contains a character other than `x`, `y`, `z` (any case).
    pub fn set_axises(&mut self, axises: &str, frame: MaskFrame) {
        self.indices.clear();
        self.frame = frame;
        for c in axises.chars() {
            match c.to_ascii_lowercase() {
                'x' => self.indices.push(0),
                'y' => self.indices.push(1),
                'z' => self.indices.push(2),
                other => panic!("AxisesMask: invalid axis: {other}"),
            }
        }
    }

    /// Sets which axes to keep from a frame name (`"task"`/`"world"`, `"local"`,
    /// `"custom"`).
    ///
    /// # Panics
    /// If `frame` is not a recognized name.
    pub fn set_axises_named(&mut self, axises: &str, frame: &str) {
        let f = match frame {
            "task" | "world" => MaskFrame::Task,
            "local" => MaskFrame::Local,
            "custom" => MaskFrame::Custom,
            other => panic!("AxisesMask: invalid frame: {other}"),
        };
        self.set_axises(axises, f);
    }

    /// Applies the mask to a `3 × n` matrix: rotates into the mask frame (if any)
    /// then keeps only the selected rows, giving a `|indices| × n` matrix.
    pub fn apply(&self, m: &DMatrix<f64>) -> DMatrix<f64> {
        let rotated = match self.frame {
            MaskFrame::Custom => matrix3_to_dmatrix(&self.r_custom_world) * m,
            MaskFrame::Local => matrix3_to_dmatrix(&self.r_local_world) * m,
            MaskFrame::Task => m.clone(),
        };
        rotated.select_rows(self.indices.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_selected_rows_in_task_frame() {
        let mut mask = AxisesMask::new();
        mask.set_axises("xz", MaskFrame::Task);
        let m = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = mask.apply(&m);
        assert_eq!(out.shape(), (2, 2));
        assert_eq!(out[(0, 0)], 1.0); // x row
        assert_eq!(out[(1, 0)], 5.0); // z row
    }

    #[test]
    fn default_keeps_all_three() {
        let mask = AxisesMask::new();
        assert_eq!(mask.indices, vec![0, 1, 2]);
        let m = DMatrix::identity(3, 3);
        assert_eq!(mask.apply(&m).shape(), (3, 3));
    }

    #[test]
    #[should_panic]
    fn rejects_bad_axis() {
        AxisesMask::new().set_axises("xw", MaskFrame::Task);
    }
}

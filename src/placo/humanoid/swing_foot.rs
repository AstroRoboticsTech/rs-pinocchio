//! Swing-foot trajectories (PlaCo `SwingFoot`, `SwingFootQuintic`).

use nalgebra::{DMatrix, DVector, Matrix4, Vector3, Vector4};

fn pos_coeffs(t: f64) -> Vector4<f64> {
    Vector4::new(t * t * t, t * t, t, 1.0)
}

fn vel_coeffs(t: f64) -> Vector4<f64> {
    Vector4::new(3.0 * t * t, 2.0 * t, 1.0, 0.0)
}

/// A cubic swing-foot trajectory: `a·τ³ + b·τ² + c·τ + d` with `τ = t - t_start`.
#[derive(Clone, Copy, Debug)]
pub struct SwingFoot {
    /// Trajectory start time.
    pub t_start: f64,
    /// Trajectory end time.
    pub t_end: f64,
    /// Cubic coefficients (per axis).
    pub a: Vector3<f64>,
    /// Cubic coefficients (per axis).
    pub b: Vector3<f64>,
    /// Cubic coefficients (per axis).
    pub c: Vector3<f64>,
    /// Cubic coefficients (per axis).
    pub d: Vector3<f64>,
}

impl SwingFoot {
    /// Builds a swing trajectory from `start` to `target` over `[t_start, t_end]`,
    /// rising to `height` around the middle.
    ///
    /// x/y have zero endpoint velocities; z passes through `height` at 1/4 and 3/4.
    pub fn make_trajectory(
        t_start: f64,
        t_end: f64,
        height: f64,
        start: Vector3<f64>,
        target: Vector3<f64>,
    ) -> SwingFoot {
        let dur = t_end - t_start;
        // Rows: pos(0), pos(dur), vel(0), vel(dur).
        let a_mat = mat4_rows(&[
            pos_coeffs(0.0),
            pos_coeffs(dur),
            vel_coeffs(0.0),
            vel_coeffs(dur),
        ]);
        let a_inv = a_mat.try_inverse().expect("SwingFoot: singular xy fit");

        let abcd_x = a_inv * Vector4::new(start.x, target.x, 0.0, 0.0);
        let abcd_y = a_inv * Vector4::new(start.y, target.y, 0.0, 0.0);

        // z: pos(0), pos(dur/4), pos(3dur/4), pos(dur).
        let b_mat = mat4_rows(&[
            pos_coeffs(0.0),
            pos_coeffs(0.25 * dur),
            pos_coeffs(0.75 * dur),
            pos_coeffs(dur),
        ]);
        let b_inv = b_mat.try_inverse().expect("SwingFoot: singular z fit");
        let abcd_z = b_inv * Vector4::new(start.z, height, height, target.z);

        SwingFoot {
            t_start,
            t_end,
            a: Vector3::new(abcd_x[0], abcd_y[0], abcd_z[0]),
            b: Vector3::new(abcd_x[1], abcd_y[1], abcd_z[1]),
            c: Vector3::new(abcd_x[2], abcd_y[2], abcd_z[2]),
            d: Vector3::new(abcd_x[3], abcd_y[3], abcd_z[3]),
        }
    }

    /// Replans the landing from the current point `t`, keeping the z profile.
    pub fn remake_trajectory(&self, t: f64, target: Vector3<f64>) -> SwingFoot {
        let dur = self.t_end - self.t_start;
        let tau = t - self.t_start;
        let a_mat = mat4_rows(&[
            pos_coeffs(tau),
            pos_coeffs(dur),
            vel_coeffs(tau),
            vel_coeffs(dur),
        ]);
        let a_inv = a_mat.try_inverse().expect("SwingFoot: singular replan fit");
        let (p, v) = (self.pos(t), self.vel(t));
        let abcd_x = a_inv * Vector4::new(p.x, target.x, v.x, 0.0);
        let abcd_y = a_inv * Vector4::new(p.y, target.y, v.y, 0.0);

        SwingFoot {
            t_start: self.t_start,
            t_end: self.t_end,
            a: Vector3::new(abcd_x[0], abcd_y[0], self.a.z),
            b: Vector3::new(abcd_x[1], abcd_y[1], self.b.z),
            c: Vector3::new(abcd_x[2], abcd_y[2], self.c.z),
            d: Vector3::new(abcd_x[3], abcd_y[3], self.d.z),
        }
    }

    /// Position at time `t`.
    pub fn pos(&self, t: f64) -> Vector3<f64> {
        let tau = t - self.t_start;
        self.a * tau.powi(3) + self.b * tau.powi(2) + self.c * tau + self.d
    }

    /// Velocity at time `t`.
    pub fn vel(&self, t: f64) -> Vector3<f64> {
        let tau = t - self.t_start;
        3.0 * self.a * tau.powi(2) + 2.0 * self.b * tau + self.c
    }
}

fn mat4_rows(rows: &[Vector4<f64>; 4]) -> Matrix4<f64> {
    Matrix4::from_rows(&[
        rows[0].transpose(),
        rows[1].transpose(),
        rows[2].transpose(),
        rows[3].transpose(),
    ])
}

/// A quintic swing-foot trajectory in absolute time: `a·t⁵ + … + f` per axis
/// (PlaCo `SwingFootQuintic`). Zero endpoint velocity, and zero endpoint
/// acceleration in x/y.
#[derive(Clone, Copy, Debug)]
pub struct SwingFootQuintic {
    /// Trajectory start time.
    pub t_start: f64,
    /// Trajectory end time.
    pub t_end: f64,
    a: Vector3<f64>,
    b: Vector3<f64>,
    c: Vector3<f64>,
    d: Vector3<f64>,
    e: Vector3<f64>,
    f: Vector3<f64>,
}

impl SwingFootQuintic {
    /// Builds a quintic swing trajectory from `start` to `target`.
    pub fn make_trajectory(
        t_start: f64,
        t_end: f64,
        height: f64,
        start: Vector3<f64>,
        target: Vector3<f64>,
    ) -> SwingFootQuintic {
        let t_a = t_start + (t_end - t_start) / 4.0;
        let t_b = t_start + 3.0 * (t_end - t_start) / 4.0;

        let pos = |t: f64| [t.powi(5), t.powi(4), t.powi(3), t.powi(2), t, 1.0];
        let vel = |t: f64| {
            [
                5.0 * t.powi(4),
                4.0 * t.powi(3),
                3.0 * t.powi(2),
                2.0 * t,
                1.0,
                0.0,
            ]
        };
        let acc = |t: f64| [20.0 * t.powi(3), 12.0 * t.powi(2), 6.0 * t, 2.0, 0.0, 0.0];

        let mut a = DMatrix::zeros(18, 18);
        let mut b = DVector::zeros(18);
        // Place a 6-coefficient row for `axis` (0=x,1=y,2=z) at matrix row `r`.
        let set = |m: &mut DMatrix<f64>, r: usize, axis: usize, coeffs: [f64; 6]| {
            for (j, v) in coeffs.iter().enumerate() {
                m[(r, axis * 6 + j)] = *v;
            }
        };

        // Initial / final positions.
        for axis in 0..3 {
            set(&mut a, axis, axis, pos(t_start));
            set(&mut a, 3 + axis, axis, pos(t_end));
        }
        for i in 0..3 {
            b[i] = start[i];
            b[3 + i] = target[i];
        }
        // Zero endpoint velocities (all axes).
        for axis in 0..3 {
            set(&mut a, 6 + axis, axis, vel(t_start));
            set(&mut a, 9 + axis, axis, vel(t_end));
        }
        // Zero endpoint accelerations (x, y only).
        for axis in 0..2 {
            set(&mut a, 12 + axis, axis, acc(t_start));
            set(&mut a, 14 + axis, axis, acc(t_end));
        }
        // z passes through `height` at t_a and t_b.
        set(&mut a, 16, 2, pos(t_a));
        b[16] = height;
        set(&mut a, 17, 2, pos(t_b));
        b[17] = height;

        let coeffs = a.try_inverse().expect("SwingFootQuintic: singular fit") * b;
        let axis_vec = |k: usize| Vector3::new(coeffs[k], coeffs[6 + k], coeffs[12 + k]);

        SwingFootQuintic {
            t_start,
            t_end,
            a: axis_vec(0),
            b: axis_vec(1),
            c: axis_vec(2),
            d: axis_vec(3),
            e: axis_vec(4),
            f: axis_vec(5),
        }
    }

    /// Position at (absolute) time `t`.
    pub fn pos(&self, t: f64) -> Vector3<f64> {
        self.a * t.powi(5)
            + self.b * t.powi(4)
            + self.c * t.powi(3)
            + self.d * t.powi(2)
            + self.e * t
            + self.f
    }

    /// Velocity at (absolute) time `t`.
    pub fn vel(&self, t: f64) -> Vector3<f64> {
        5.0 * self.a * t.powi(4)
            + 4.0 * self.b * t.powi(3)
            + 3.0 * self.c * t.powi(2)
            + 2.0 * self.d * t
            + self.e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_hits_endpoints_and_apex() {
        let start = Vector3::new(0.0, 0.0, 0.0);
        let target = Vector3::new(0.2, 0.1, 0.0);
        let traj = SwingFoot::make_trajectory(0.0, 1.0, 0.05, start, target);
        assert!((traj.pos(0.0) - start).norm() < 1e-9);
        assert!((traj.pos(1.0) - target).norm() < 1e-9);
        // Zero endpoint velocity in x/y.
        assert!(traj.vel(0.0).xy().norm() < 1e-9);
        assert!(traj.vel(1.0).xy().norm() < 1e-9);
        // Foot rises above ground mid-swing.
        assert!(traj.pos(0.25).z > 0.04);
    }

    #[test]
    fn quintic_hits_endpoints_with_zero_velocity() {
        let start = Vector3::new(0.0, 0.0, 0.0);
        let target = Vector3::new(0.15, -0.05, 0.0);
        let traj = SwingFootQuintic::make_trajectory(0.0, 1.0, 0.06, start, target);
        assert!((traj.pos(0.0) - start).norm() < 1e-6);
        assert!((traj.pos(1.0) - target).norm() < 1e-6);
        assert!(traj.vel(0.0).norm() < 1e-6);
        assert!(traj.vel(1.0).norm() < 1e-6);
        assert!(traj.pos(0.25).z > 0.05);
    }
}

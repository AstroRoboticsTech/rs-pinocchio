//! Angle, frame and point-cloud helpers (PlaCo `placo::tools` free functions).

use nalgebra::{DMatrix, Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};

/// SO(3) exponential map: rotation matrix of the rotation vector `v` (axis·angle).
pub fn exp3(v: Vector3<f64>) -> Matrix3<f64> {
    Rotation3::from_scaled_axis(v).into_inner()
}

/// Interpolates between two frames: slerp on rotation, lerp on translation.
///
/// `a_to_b` in `[0, 1]` selects `frame_a` at 0 and `frame_b` at 1.
pub fn interpolate_frames(
    frame_a: &Isometry3<f64>,
    frame_b: &Isometry3<f64>,
    a_to_b: f64,
) -> Isometry3<f64> {
    let rot = frame_a.rotation.slerp(&frame_b.rotation, a_to_b);
    let trans = frame_a.translation.vector * (1.0 - a_to_b) + frame_b.translation.vector * a_to_b;
    Isometry3::from_parts(Translation3::from(trans), rot)
}

/// Wraps an angle into `[-π, π]`.
pub fn wrap_angle(angle: f64) -> f64 {
    angle.sin().atan2(angle.cos())
}

/// The yaw of an orientation: the heading of its rotated x-axis in the xy-plane.
pub fn frame_yaw(rotation: &Matrix3<f64>) -> f64 {
    let x_in_new = rotation * Vector3::x();
    x_in_new.y.atan2(x_in_new.x)
}

/// Builds a rotation that maps the given canonical `axis` (`"x"`, `"y"` or `"z"`)
/// onto the (normalized) target `vector`.
///
/// # Panics
/// If `axis` is not one of `"x"`, `"y"`, `"z"`.
pub fn rotation_from_axis(axis: &str, mut vector: Vector3<f64>) -> Matrix3<f64> {
    if vector.norm() > 0.0 {
        vector.normalize_mut();
    }
    let vector_id = match axis {
        "x" => Vector3::x(),
        "y" => Vector3::y(),
        "z" => Vector3::z(),
        other => panic!("rotation_from_axis: unknown axis: {other}"),
    };

    let mut w = vector_id.cross(&vector);
    let theta = safe_acos(vector_id.dot(&vector));
    if w.norm() == 0.0 {
        w = match axis {
            "x" => Vector3::y(),
            "y" => Vector3::z(),
            _ => Vector3::x(),
        };
    } else {
        w.normalize_mut();
    }
    exp3(w * theta)
}

/// Flattens a transform onto the floor: zeroes z and keeps only the yaw.
pub fn flatten_on_floor(transformation: &Isometry3<f64>) -> Isometry3<f64> {
    let yaw = frame_yaw(&transformation.rotation.to_rotation_matrix().into_inner());
    let rot = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
    let mut trans = transformation.translation.vector;
    trans.z = 0.0;
    Isometry3::from_parts(Translation3::from(trans), rot)
}

/// `acos` clamped to `[-1, 1]` first (guards against floating-point overshoot).
pub fn safe_acos(v: f64) -> f64 {
    v.clamp(-1.0, 1.0).acos()
}

/// The transform `T_a_b` minimizing the sum of squared distances between the
/// same points expressed in frames A and B (Kabsch/Umeyama, rotation only).
///
/// Points are stacked in rows (columns x, y, z). Requires at least 3 rows and
/// matching row counts.
///
/// # Panics
/// If the shapes are invalid or fewer than 3 points are given.
pub fn optimal_transformation(
    points_in_a: &DMatrix<f64>,
    points_in_b: &DMatrix<f64>,
) -> Isometry3<f64> {
    assert!(
        points_in_a.ncols() == 3 && points_in_b.ncols() == 3,
        "optimal_transformation: points should have 3 columns (x, y, z)"
    );
    assert!(
        points_in_a.nrows() == points_in_b.nrows(),
        "optimal_transformation: A and B should have the same number of rows"
    );
    assert!(
        points_in_a.nrows() >= 3,
        "optimal_transformation: at least 3 points are required"
    );

    let barycenter_a = points_in_a.row_mean().transpose();
    let barycenter_b = points_in_b.row_mean().transpose();

    let centered_a = DMatrix::from_fn(points_in_a.nrows(), 3, |i, j| {
        points_in_a[(i, j)] - barycenter_a[j]
    });
    let centered_b = DMatrix::from_fn(points_in_b.nrows(), 3, |i, j| {
        points_in_b[(i, j)] - barycenter_b[j]
    });

    let h = centered_b.transpose() * &centered_a; // 3x3
    let h3 = Matrix3::from_iterator(h.iter().copied());
    let svd = h3.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let v = v_t.transpose();

    let mut r = v * u.transpose();
    if r.determinant() < 0.0 {
        let mut z = Matrix3::identity();
        z[(2, 2)] = -1.0;
        r = v * z * u.transpose();
    }

    let ba = Vector3::new(barycenter_a[0], barycenter_a[1], barycenter_a[2]);
    let bb = Vector3::new(barycenter_b[0], barycenter_b[1], barycenter_b[2]);
    let t = ba - r * bb;

    let rot = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    Isometry3::from_parts(Translation3::from(t), rot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn wrap_angle_folds_into_pi() {
        assert!(
            (wrap_angle(3.0 * PI) - PI).abs() < 1e-9 || (wrap_angle(3.0 * PI) + PI).abs() < 1e-9
        );
        assert!((wrap_angle(0.5) - 0.5).abs() < 1e-12);
        assert!((wrap_angle(-2.5 * PI) - (-0.5 * PI)).abs() < 1e-9);
    }

    #[test]
    fn safe_acos_clamps() {
        assert!((safe_acos(2.0) - 0.0).abs() < 1e-12);
        assert!((safe_acos(-2.0) - PI).abs() < 1e-12);
    }

    #[test]
    fn frame_yaw_of_z_rotation() {
        let r = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.7).into_inner();
        assert!((frame_yaw(&r) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn rotation_from_axis_maps_axis_onto_vector() {
        let target = Vector3::new(0.0, 0.0, 1.0);
        let r = rotation_from_axis("x", target);
        let mapped = r * Vector3::x();
        assert!((mapped - target).norm() < 1e-9);
    }

    #[test]
    fn exp3_of_z_axis() {
        let r = exp3(Vector3::new(0.0, 0.0, PI / 2.0));
        let mapped = r * Vector3::x();
        assert!((mapped - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-9);
    }

    #[test]
    fn interpolate_frames_endpoints() {
        let a = Isometry3::translation(0.0, 0.0, 0.0);
        let b = Isometry3::translation(2.0, 4.0, 6.0);
        let mid = interpolate_frames(&a, &b, 0.5);
        assert!((mid.translation.vector - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-9);
    }

    #[test]
    fn optimal_transformation_recovers_known_transform() {
        // A known rotation + translation applied to B gives A; recover it.
        let r = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.3);
        let t = Vector3::new(1.0, -2.0, 0.5);
        let pts_b = DMatrix::from_row_slice(
            4,
            3,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        let mut pts_a = DMatrix::zeros(4, 3);
        for i in 0..4 {
            let p = Vector3::new(pts_b[(i, 0)], pts_b[(i, 1)], pts_b[(i, 2)]);
            let q = r * p + t;
            pts_a[(i, 0)] = q.x;
            pts_a[(i, 1)] = q.y;
            pts_a[(i, 2)] = q.z;
        }
        let recovered = optimal_transformation(&pts_a, &pts_b);
        assert!((recovered.translation.vector - t).norm() < 1e-6);
        let rr = recovered.rotation.to_rotation_matrix().into_inner();
        assert!((rr - r.into_inner()).norm() < 1e-6);
    }
}

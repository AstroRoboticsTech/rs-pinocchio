//! Integration tests exercising the full cxx bridge against a live Pinocchio.
//!
//! These are gated with `#[ignore]` because they require Pinocchio to be
//! installed and linkable. Run them explicitly once it is available:
//!
//! ```sh
//! cargo test -- --ignored
//! ```

use std::io::Write;
use std::path::PathBuf;

use nalgebra::{DMatrix, DVector, Isometry3};
use rs_pinocchio::{Error, Model, ReferenceFrame};

/// A non-degenerate two-revolute arm:
/// `base -(j1: z @ 0,0,0.1)-> link1 -(j2: y @ 0,0,0.2)-> link2 -(fixed @ 0.1,0,0)-> tool`.
/// The fixed x-offset after `j2` makes both joints move the `tool` frame, so the
/// Jacobian is fully exercised. nq = nv = 2.
const ARM_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="link1"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="link2"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="tool"><inertial><mass value="0.5"/>
    <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial></link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <joint name="tool_fixed" type="fixed">
    <parent link="link2"/><child link="tool"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"#;

fn write_fixture() -> PathBuf {
    // Unique per call: tests run in parallel and each removes its own fixture.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut path = std::env::temp_dir();
    path.push(format!(
        "rs_pinocchio_test_arm_{}_{}.urdf",
        std::process::id(),
        n
    ));
    let mut f = std::fs::File::create(&path).expect("create temp urdf");
    f.write_all(ARM_URDF.as_bytes()).expect("write urdf");
    path
}

fn tool_placement(model: &mut Model, q: &DVector<f64>, tool: usize) -> Isometry3<f64> {
    model.forward_kinematics(q).expect("fk");
    model.update_frame_placements();
    model.frame_placement(tool).expect("placement")
}

fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    assert!((a - b).abs() <= tol, "{msg}: {a} vs {b} (tol {tol})");
}

// --- basic shape / smoke -----------------------------------------------------

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn fixed_base_fk_and_jacobian() {
    let path = write_fixture();
    let mut model = Model::from_urdf(&path, false).expect("load model");

    assert_eq!(model.nq(), 2);
    assert_eq!(model.nv(), 2);

    let q = DVector::from_vec(vec![0.3, -0.5]);
    let tip = model.frame_id("tool").expect("tool frame exists");
    assert!(model.frame_id("does_not_exist").is_none());

    let placement = tool_placement(&mut model, &q, tip);
    assert!(placement.translation.vector.iter().all(|v| v.is_finite()));

    model.compute_joint_jacobians(&q).expect("joint jacobians");
    for rf in [
        ReferenceFrame::Local,
        ReferenceFrame::World,
        ReferenceFrame::LocalWorldAligned,
    ] {
        let jac = model.frame_jacobian(tip, rf).expect("frame jacobian");
        assert_eq!((jac.nrows(), jac.ncols()), (6, 2));
        assert!(jac.iter().all(|v| v.is_finite()));
    }
    let _ = std::fs::remove_file(&path);
}

// --- forward-kinematics correctness (analytic) -------------------------------

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn fk_matches_analytic_at_zero() {
    let path = write_fixture();
    let mut model = Model::from_urdf(&path, false).expect("load");
    let tip = model.frame_id("tool").unwrap();

    // q = 0 → tool at (0,0,0.1)+(0,0,0.2)+(0.1,0,0) = (0.1, 0, 0.3), identity rot.
    let m = tool_placement(&mut model, &DVector::from_vec(vec![0.0, 0.0]), tip);
    assert_close(m.translation.x, 0.1, 1e-9, "x");
    assert_close(m.translation.y, 0.0, 1e-9, "y");
    assert_close(m.translation.z, 0.3, 1e-9, "z");
    assert_close(m.rotation.angle(), 0.0, 1e-9, "identity rotation");
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn fk_joint2_moves_tip_in_xz() {
    let path = write_fixture();
    let mut model = Model::from_urdf(&path, false).expect("load");
    let tip = model.frame_id("tool").unwrap();

    // Rotate joint2 (y-axis) by +90°: the (0.1,0,0) tool offset maps x→-z.
    // tool = (0,0,0.3) + Ry(90)*(0.1,0,0) = (0,0,0.3) + (0,0,-0.1) = (0,0,0.2).
    let m = tool_placement(
        &mut model,
        &DVector::from_vec(vec![0.0, std::f64::consts::FRAC_PI_2]),
        tip,
    );
    assert_close(m.translation.x, 0.0, 1e-6, "x");
    assert_close(m.translation.z, 0.2, 1e-6, "z");
    let _ = std::fs::remove_file(&path);
}

// --- Jacobian correctness vs central finite difference (the real proof) ------

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn local_jacobian_matches_finite_difference() {
    let path = write_fixture();
    let mut model = Model::from_urdf(&path, false).expect("load");
    let tip = model.frame_id("tool").unwrap();
    let eps = 1e-6;

    for q in [
        DVector::from_vec(vec![0.3, -0.5]),
        DVector::from_vec(vec![0.9, 1.2]),
    ] {
        model.compute_joint_jacobians(&q).expect("jac");
        let j = model
            .frame_jacobian(tip, ReferenceFrame::Local)
            .expect("local jac");

        for i in 0..model.nv() {
            let mut qp = q.clone();
            qp[i] += eps;
            let mut qm = q.clone();
            qm[i] -= eps;
            let mp = tool_placement(&mut model, &qp, tip);
            let mm = tool_placement(&mut model, &qm, tip);

            // Central difference of the local relative motion → [v; w].
            let rel = mm.inverse() * mp;
            let v = rel.translation.vector / (2.0 * eps);
            let w = rel.rotation.scaled_axis() / (2.0 * eps);
            let fd = [v.x, v.y, v.z, w.x, w.y, w.z];

            for (row, fdv) in fd.iter().enumerate() {
                assert_close(j[(row, i)], *fdv, 1e-4, &format!("J[{row},{i}] vs FD"));
            }
        }
    }
    let _ = std::fs::remove_file(&path);
}

// --- reference-frame algebra cross-check -------------------------------------

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn local_world_aligned_equals_rotated_local() {
    let path = write_fixture();
    let mut model = Model::from_urdf(&path, false).expect("load");
    let tip = model.frame_id("tool").unwrap();
    let q = DVector::from_vec(vec![0.4, -0.7]);

    let m = tool_placement(&mut model, &q, tip);
    model.compute_joint_jacobians(&q).expect("jac");
    let j_local = model.frame_jacobian(tip, ReferenceFrame::Local).unwrap();
    let j_lwa = model
        .frame_jacobian(tip, ReferenceFrame::LocalWorldAligned)
        .unwrap();

    // LOCAL_WORLD_ALIGNED = local origin, world axes ⇒ J_lwa = diag(R,R)·J_local.
    let r = m.rotation.to_rotation_matrix();
    let rdm = DMatrix::from_column_slice(3, 3, r.matrix().as_slice());
    let expect_lin = &rdm * j_local.rows(0, 3);
    let expect_ang = &rdm * j_local.rows(3, 3);

    for i in 0..model.nv() {
        for row in 0..3 {
            assert_close(j_lwa[(row, i)], expect_lin[(row, i)], 1e-9, "lwa linear");
            assert_close(
                j_lwa[(row + 3, i)],
                expect_ang[(row, i)],
                1e-9,
                "lwa angular",
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

// --- by-name == by-id, and missing-frame errors ------------------------------

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn by_name_matches_by_id_and_errors() {
    let path = write_fixture();
    let mut model = Model::from_urdf(&path, false).expect("load");
    let q = DVector::from_vec(vec![0.2, 0.6]);
    let tip = model.frame_id("tool").unwrap();

    let _ = tool_placement(&mut model, &q, tip);
    let by_id = model.frame_placement(tip).unwrap();
    let by_name = model.frame_placement_by_name("tool").unwrap();
    assert_close(
        (by_id.translation.vector - by_name.translation.vector).norm(),
        0.0,
        1e-12,
        "placement",
    );

    model.compute_joint_jacobians(&q).unwrap();
    let jid = model.frame_jacobian(tip, ReferenceFrame::World).unwrap();
    let jname = model
        .frame_jacobian_by_name("tool", ReferenceFrame::World)
        .unwrap();
    assert_close((jid - jname).amax(), 0.0, 1e-12, "jacobian");

    assert!(matches!(
        model.frame_placement_by_name("nope"),
        Err(Error::FrameNotFound(_))
    ));
    assert!(matches!(
        model.frame_jacobian_by_name("nope", ReferenceFrame::Local),
        Err(Error::FrameNotFound(_))
    ));
    let _ = std::fs::remove_file(&path);
}

// --- dimension validation ----------------------------------------------------

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn wrong_length_q_errors_not_panics() {
    let path = write_fixture();
    let mut model = Model::from_urdf(&path, false).expect("load");
    let bad = DVector::from_vec(vec![0.0, 0.0, 0.0]); // nq is 2

    assert!(matches!(
        model.forward_kinematics(&bad),
        Err(Error::DimMismatch {
            expected: 2,
            got: 3,
            ..
        })
    ));
    assert!(matches!(
        model.compute_joint_jacobians(&bad),
        Err(Error::DimMismatch {
            expected: 2,
            got: 3,
            ..
        })
    ));
    let _ = std::fs::remove_file(&path);
}

// --- floating base -----------------------------------------------------------

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn floating_base_adds_free_flyer() {
    let path = write_fixture();
    let model = Model::from_urdf(&path, true).expect("load floating-base");
    assert_eq!(model.nq(), 2 + 7); // +[x y z qx qy qz qw]
    assert_eq!(model.nv(), 2 + 6);
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn floating_base_translates_frame() {
    let path = write_fixture();
    let mut model = Model::from_urdf(&path, true).expect("load floating-base");
    let tip = model.frame_id("tool").unwrap();

    // q = [x y z | qx qy qz qw | j1 j2]; identity orientation, base at (1,2,3).
    let q = DVector::from_vec(vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let m = tool_placement(&mut model, &q, tip);
    // Fixed-base tool at q=0 is (0.1,0,0.3); base offset shifts it by (1,2,3).
    assert_close(m.translation.x, 1.1, 1e-9, "x");
    assert_close(m.translation.y, 2.0, 1e-9, "y");
    assert_close(m.translation.z, 3.3, 1e-9, "z");
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn bad_urdf_path_errors() {
    assert!(Model::from_urdf("/nonexistent/robot.urdf", false).is_err());
}

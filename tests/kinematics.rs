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

use nalgebra::DVector;
use pinocchio_rs::{Model, ReferenceFrame};

/// A minimal two-revolute-joint arm: base_link -> link1 -> tool.
/// nq = nv = 2. Frame `tool` is the end-effector.
const ARM_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="tool">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tool"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
</robot>
"#;

fn write_fixture() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("pinocchio_rs_test_arm_{}.urdf", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp urdf");
    f.write_all(ARM_URDF.as_bytes()).expect("write urdf");
    path
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn fixed_base_fk_and_jacobian() {
    let path = write_fixture();
    let mut model = Model::from_urdf(&path, false).expect("load model");

    assert_eq!(model.nq(), 2, "two revolute joints => nq = 2");
    assert_eq!(model.nv(), 2, "two revolute joints => nv = 2");

    let q = DVector::from_vec(vec![0.3, -0.5]);
    model.forward_kinematics(&q).expect("fk");
    model.update_frame_placements();

    let tip = model.frame_id("tool").expect("tool frame exists");
    assert!(model.frame_id("does_not_exist").is_none());

    let placement = model.frame_placement(tip).expect("placement");
    assert!(
        placement.translation.vector.iter().all(|v| v.is_finite()),
        "placement must be finite"
    );

    model.compute_joint_jacobians(&q).expect("joint jacobians");
    for rf in [
        ReferenceFrame::Local,
        ReferenceFrame::World,
        ReferenceFrame::LocalWorldAligned,
    ] {
        let jac = model.frame_jacobian(tip, rf).expect("frame jacobian");
        assert_eq!(jac.nrows(), 6, "spatial velocity has 6 rows");
        assert_eq!(jac.ncols(), model.nv(), "one column per DoF");
        assert!(jac.iter().all(|v| v.is_finite()), "jacobian must be finite");
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn floating_base_adds_free_flyer() {
    let path = write_fixture();
    let model = Model::from_urdf(&path, true).expect("load floating-base model");

    // Free-flyer adds 7 to nq (position + quaternion) and 6 to nv.
    assert_eq!(model.nq(), 2 + 7);
    assert_eq!(model.nv(), 2 + 6);

    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn bad_urdf_path_errors() {
    let err = Model::from_urdf("/nonexistent/robot.urdf", false);
    assert!(err.is_err(), "loading a missing URDF must error, not panic");
}

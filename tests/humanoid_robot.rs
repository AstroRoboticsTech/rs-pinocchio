//! Integration tests for `placo::humanoid::HumanoidRobot` against live Pinocchio.
//!
//! Run with `cargo test --features placo --test humanoid_robot -- --ignored`.
#![cfg(all(feature = "ffi", feature = "placo"))]

use std::io::Write;
use std::path::PathBuf;

use nalgebra::Vector2;
use rs_pinocchio::placo::humanoid::{HumanoidRobot, Side};

// Minimal biped: a trunk with two legs ending in left_foot / right_foot frames.
const BIPED_URDF: &str = r#"<?xml version="1.0"?>
<robot name="biped">
  <link name="trunk"><inertial><mass value="2.0"/>
    <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
  <link name="left_leg"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="left_foot"><inertial><mass value="0.5"/>
    <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial></link>
  <link name="right_leg"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="right_foot"><inertial><mass value="0.5"/>
    <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial></link>
  <joint name="left_hip" type="revolute">
    <parent link="trunk"/><child link="left_leg"/>
    <origin xyz="0 0.1 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="10"/>
  </joint>
  <joint name="left_ankle" type="fixed">
    <parent link="left_leg"/><child link="left_foot"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  </joint>
  <joint name="right_hip" type="revolute">
    <parent link="trunk"/><child link="right_leg"/>
    <origin xyz="0 -0.1 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="10"/>
  </joint>
  <joint name="right_ankle" type="fixed">
    <parent link="right_leg"/><child link="right_foot"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  </joint>
</robot>
"#;

fn write_fixture() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "rs_pinocchio_biped_{}_{}.urdf",
        std::process::id(),
        n
    ));
    let mut f = std::fs::File::create(&path).expect("create temp urdf");
    f.write_all(BIPED_URDF.as_bytes()).expect("write urdf");
    path
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn loads_and_places_support_on_floor() {
    let path = write_fixture();
    let robot = HumanoidRobot::from_urdf(&path).expect("load biped");

    assert_eq!(robot.support_side, Side::Left);
    assert!(!robot.support_is_both);

    // init_config places the left foot at the identity support frame.
    let t_left = robot.t_world_left().expect("left foot");
    assert!(
        t_left.translation.vector.norm() < 1e-9,
        "left foot not on floor: {t_left:?}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn switching_support_updates_frame_and_placement() {
    let path = write_fixture();
    let mut robot = HumanoidRobot::from_urdf(&path).expect("load biped");
    let left = robot.left_foot;
    let right = robot.right_foot;

    assert_eq!(robot.support_frame, left);
    robot
        .update_support_side(Side::Right)
        .expect("switch support");
    assert_eq!(robot.support_frame, right);
    assert_eq!(robot.support_side, Side::Right);
    // The support placement is the right foot flattened on the floor (z = 0).
    assert!(robot.t_world_support.translation.z.abs() < 1e-9);
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn dcm_zmp_are_finite() {
    let path = write_fixture();
    let mut robot = HumanoidRobot::from_urdf(&path).expect("load biped");
    let omega = 5.0;
    let dcm = robot.dcm(omega, Vector2::new(0.1, 0.0)).expect("dcm");
    let zmp = robot.zmp(omega, Vector2::new(0.0, 0.2)).expect("zmp");
    assert!(dcm.iter().all(|v| v.is_finite()));
    assert!(zmp.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_file(&path);
}

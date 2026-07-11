//! Integration tests for `placo::model::RobotWrapper` against a live Pinocchio.
//!
//! Gated behind `ffi` + `placo` and `#[ignore]` (needs a linkable Pinocchio).
//! Run with `cargo test --features placo --test robot_wrapper -- --ignored`.
#![cfg(all(feature = "ffi", feature = "placo"))]

use std::io::Write;
use std::path::PathBuf;

use rs_pinocchio::placo::model::RobotWrapper;
use rs_pinocchio::ReferenceFrame;

// Two-revolute arm; RobotWrapper always prepends a free-flyer base.
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
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("rs_pinocchio_rw_{}_{}.urdf", std::process::id(), n));
    let mut f = std::fs::File::create(&path).expect("create temp urdf");
    f.write_all(ARM_URDF.as_bytes()).expect("write urdf");
    path
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn dimensions_include_floating_base() {
    let path = write_fixture();
    let robot = RobotWrapper::from_urdf(&path).expect("load");
    // 2 actuated DoF + 7/6 for the free-flyer base.
    assert_eq!(robot.nq(), 2 + 7);
    assert_eq!(robot.nv(), 2 + 6);
    assert_eq!(robot.state.q.len(), robot.nq());
    assert_eq!(robot.state.qd.len(), robot.nv());
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn kinematics_and_dynamics_are_finite_and_well_shaped() {
    let path = write_fixture();
    let mut robot = RobotWrapper::from_urdf(&path).expect("load");
    let nv = robot.nv();

    robot.update_kinematics().expect("update_kinematics");

    let tool = robot.frame_index("tool").expect("tool frame");
    let placement = robot.t_world_frame(tool).expect("placement");
    assert!(placement.translation.vector.iter().all(|v| v.is_finite()));

    let j = robot
        .frame_jacobian(tool, ReferenceFrame::LocalWorldAligned)
        .expect("frame jacobian");
    assert_eq!((j.nrows(), j.ncols()), (6, nv));
    assert!(j.iter().all(|v| v.is_finite()));

    let m = robot.mass_matrix().expect("mass matrix");
    assert_eq!((m.nrows(), m.ncols()), (nv, nv));
    // Symmetric.
    assert!((&m - m.transpose()).amax() < 1e-9);

    let g = robot.generalized_gravity().expect("gravity");
    assert_eq!(g.len(), nv);
    let h = robot.non_linear_effects().expect("nle");
    assert_eq!(h.len(), nv);

    let com = robot.com_world().expect("com");
    assert!(com.iter().all(|v| v.is_finite()));

    let cj = robot.com_jacobian().expect("com jacobian");
    assert_eq!((cj.nrows(), cj.ncols()), (3, nv));

    // Total mass = 1 + 1 + 1 + 0.5.
    assert!((robot.total_mass() - 3.5).abs() < 1e-9);
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn joints_and_integration() {
    let path = write_fixture();
    let mut robot = RobotWrapper::from_urdf(&path).expect("load");

    let names = robot.joint_names(false);
    assert!(names.iter().any(|n| n == "joint1"));
    assert!(names.iter().any(|n| n == "joint2"));

    // Set joint2, read it back through the state.
    robot.set_joint("joint2", 0.5).expect("set joint2");
    assert!((robot.joint("joint2").expect("get joint2") - 0.5).abs() < 1e-12);

    // Integrate a constant velocity for one step; q must stay finite.
    let o = robot.joint_v_offset("joint1").expect("v offset");
    robot.state.qd[o] = 1.0;
    robot.integrate(0.1).expect("integrate");
    assert!(robot.state.q.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_file(&path);
}

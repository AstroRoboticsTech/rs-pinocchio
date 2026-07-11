//! Integration test: the kinematics IK solver converges to a reachable target.
//!
//! Gated behind `ffi` + `placo` and `#[ignore]` (needs a linkable Pinocchio).
//! Run with `cargo test --features placo --test kinematics_solver -- --ignored`.
#![cfg(all(feature = "ffi", feature = "placo"))]

use std::io::Write;
use std::path::PathBuf;

use rs_pinocchio::placo::kinematics::KinematicsSolver;
use rs_pinocchio::placo::model::RobotWrapper;

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
    path.push(format!("rs_pinocchio_ik_{}_{}.urdf", std::process::id(), n));
    let mut f = std::fs::File::create(&path).expect("create temp urdf");
    f.write_all(ARM_URDF.as_bytes()).expect("write urdf");
    path
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn ik_reaches_a_reachable_position_target() {
    let path = write_fixture();
    let mut robot = RobotWrapper::from_urdf(&path).expect("load");
    let tool = robot.frame_index("tool").expect("tool frame");

    // Target = FK of a known joint configuration (so it is exactly reachable).
    robot.set_joint("joint1", 0.4).unwrap();
    robot.set_joint("joint2", -0.6).unwrap();
    robot.update_kinematics().unwrap();
    let target = robot.t_world_frame(tool).unwrap().translation.vector;

    // Reset to neutral and solve IK back to the target.
    robot.reset();

    let mut solver = KinematicsSolver::new(&robot);
    solver.mask_fbase(true); // only the two revolute joints move
    solver.add_position_task(tool, target);
    solver.add_regularization_task(1e-6);

    for _ in 0..200 {
        solver.solve(&mut robot, true).expect("solve");
    }

    robot.update_kinematics().unwrap();
    let reached = robot.t_world_frame(tool).unwrap().translation.vector;
    let err = (reached - target).norm();
    assert!(
        err < 1e-4,
        "IK did not converge: error {err}, reached {reached:?}"
    );
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn ik_relative_position_reaches_target() {
    let path = write_fixture();
    let mut robot = RobotWrapper::from_urdf(&path).expect("load");
    let base = robot.frame_index("base_link").expect("base frame");
    let tool = robot.frame_index("tool").expect("tool frame");

    // Reachable relative target from FK of a known configuration.
    robot.set_joint("joint1", -0.3).unwrap();
    robot.set_joint("joint2", 0.5).unwrap();
    robot.update_kinematics().unwrap();
    let target = robot.t_a_b(base, tool).unwrap().translation.vector;

    robot.reset();
    let mut solver = KinematicsSolver::new(&robot);
    solver.mask_fbase(true);
    solver.add_relative_position_task(base, tool, target);
    solver.add_regularization_task(1e-6);

    for _ in 0..200 {
        solver.solve(&mut robot, true).expect("solve");
    }

    robot.update_kinematics().unwrap();
    let reached = robot.t_a_b(base, tool).unwrap().translation.vector;
    let err = (reached - target).norm();
    assert!(err < 1e-4, "relative IK did not converge: error {err}");
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn masked_fbase_keeps_base_fixed() {
    let path = write_fixture();
    let mut robot = RobotWrapper::from_urdf(&path).expect("load");
    let tool = robot.frame_index("tool").expect("tool frame");

    robot.update_kinematics().unwrap();
    let base_before = robot.state.q.rows(0, 7).into_owned();

    let mut solver = KinematicsSolver::new(&robot);
    solver.mask_fbase(true);
    // Pull the tool far away; without a fixed base the base would translate.
    solver.add_position_task(tool, nalgebra::Vector3::new(5.0, 5.0, 5.0));

    for _ in 0..20 {
        solver.solve(&mut robot, true).expect("solve");
    }

    let base_after = robot.state.q.rows(0, 7).into_owned();
    assert!(
        (base_before - base_after).norm() < 1e-9,
        "floating base moved despite being masked"
    );
    let _ = std::fs::remove_file(&path);
}

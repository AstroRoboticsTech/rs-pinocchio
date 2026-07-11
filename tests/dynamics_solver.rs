//! Integration test: the inverse-dynamics solver balances gravity through
//! foot contacts.
//!
//! Run with `cargo test --features placo --test dynamics_solver -- --ignored`.
#![cfg(all(feature = "ffi", feature = "placo"))]

use std::io::Write;
use std::path::PathBuf;

use rs_pinocchio::placo::dynamics::DynamicsSolver;
use rs_pinocchio::placo::model::RobotWrapper;
use rs_pinocchio::placo::tools::Priority;

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
        "rs_pinocchio_dyn_{}_{}.urdf",
        std::process::id(),
        n
    ));
    let mut f = std::fs::File::create(&path).expect("create temp urdf");
    f.write_all(BIPED_URDF.as_bytes()).expect("write urdf");
    path
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn foot_contacts_balance_gravity() {
    let path = write_fixture();
    let mut robot = RobotWrapper::from_urdf(&path).expect("load biped");
    robot.update_kinematics().expect("update kinematics");

    let left = robot.frame_index("left_foot").expect("left foot");
    let right = robot.frame_index("right_foot").expect("right foot");
    let total_mass = robot.total_mass();

    let lt = robot.t_world_frame(left).unwrap().translation.vector;
    let rt = robot.t_world_frame(right).unwrap().translation.vector;

    let mut solver = DynamicsSolver::new(&robot);
    solver.gravity_only = true;

    // Hold both feet in place, and put a contact under each.
    let lid = solver.add_position_task(left, lt);
    solver.configure_task(lid, Priority::Hard, 1.0);
    let rid = solver.add_position_task(right, rt);
    solver.configure_task(rid, Priority::Hard, 1.0);
    let lc = solver.add_unilateral_point_contact(left);
    let rc = solver.add_unilateral_point_contact(right);

    let result = solver.solve(&mut robot, false).expect("dynamics solve");
    assert!(result.success);
    assert!(result.tau.iter().all(|v| v.is_finite()));
    // The floating base carries no torque.
    assert!(
        result.tau.rows(0, 6).norm() < 1e-6,
        "fbase torque: {}",
        result.tau.rows(0, 6).norm()
    );

    // The vertical contact forces support the full weight.
    let fz_left = solver.contact_wrench(lc).unwrap()[2];
    let fz_right = solver.contact_wrench(rc).unwrap()[2];
    let total_fz = fz_left + fz_right;
    // Pinocchio's default gravity magnitude is 9.81.
    let weight = total_mass * 9.81;
    assert!(
        (total_fz - weight).abs() < 1e-2,
        "vertical contact force {total_fz} != weight {weight}"
    );
    // Unilateral contacts push (non-negative normal force).
    assert!(fz_left >= -1e-9 && fz_right >= -1e-9);
    let _ = std::fs::remove_file(&path);
}

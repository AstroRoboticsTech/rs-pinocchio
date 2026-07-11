//! Integration test: the inverse-dynamics solver balances gravity through
//! foot contacts.
//!
//! Run with `cargo test --features placo --test dynamics_solver -- --ignored`.
#![cfg(all(feature = "ffi", feature = "placo"))]

use std::io::Write;
use std::path::PathBuf;

use nalgebra::Matrix3;

use rs_pinocchio::placo::dynamics::{DynamicsSolver, LineContact};
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

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn relative_orientation_task_drives_legs() {
    let path = write_fixture();
    let mut robot = RobotWrapper::from_urdf(&path).expect("load biped");
    robot.update_kinematics().expect("update kinematics");

    let left = robot.frame_index("left_leg").expect("left leg");
    let right = robot.frame_index("right_leg").expect("right leg");

    let mut solver = DynamicsSolver::new(&robot);
    solver.gravity_only = true;
    // Fully actuate the robot so the orientation target is reachable.
    solver.add_puppet_contact();

    // Target a 0.3 rad relative rotation of the right leg about y in the left
    // leg's frame; both legs start aligned, so the task must produce a non-zero
    // generalized acceleration.
    let (s, c) = (0.3f64.sin(), 0.3f64.cos());
    #[rustfmt::skip]
    let r_a_b = Matrix3::new(
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c,
    );
    let tid = solver.add_relative_orientation_task(left, right, r_a_b);
    solver.configure_task(tid, Priority::Soft, 1.0);

    let result = solver.solve(&mut robot, false).expect("dynamics solve");
    assert!(result.success);
    assert!(result.tau.iter().all(|v| v.is_finite()));
    assert!(result.qdd.iter().all(|v| v.is_finite()));
    // The task pushes the legs apart, so the solve is not trivially zero.
    assert!(result.qdd.norm() > 1e-6, "qdd norm {}", result.qdd.norm());
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn line_contacts_balance_gravity() {
    let path = write_fixture();
    let mut robot = RobotWrapper::from_urdf(&path).expect("load biped");
    robot.update_kinematics().expect("update kinematics");

    let left = robot.frame_index("left_foot").expect("left foot");
    let right = robot.frame_index("right_foot").expect("right foot");
    let total_mass = robot.total_mass();

    let lt = robot.t_world_frame(left).unwrap();
    let rt = robot.t_world_frame(right).unwrap();

    let mut solver = DynamicsSolver::new(&robot);
    solver.gravity_only = true;

    // Hold each foot's full pose (position + orientation).
    let lh = solver.add_frame_task(left, lt);
    solver.configure_task(lh.position, Priority::Hard, 1.0);
    solver.configure_task(lh.orientation, Priority::Hard, 1.0);
    let rh = solver.add_frame_task(right, rt);
    solver.configure_task(rh.position, Priority::Hard, 1.0);
    solver.configure_task(rh.orientation, Priority::Hard, 1.0);

    let lc = solver.add_line_contact(left);
    let rc = solver.add_line_contact(right);
    solver.contact_mut::<LineContact>(lc).unwrap().length = 0.1;
    solver.contact_mut::<LineContact>(rc).unwrap().length = 0.1;

    let result = solver.solve(&mut robot, false).expect("dynamics solve");
    assert!(result.success);
    assert!(result.tau.iter().all(|v| v.is_finite()));
    assert!(
        result.tau.rows(0, 6).norm() < 1e-6,
        "fbase torque: {}",
        result.tau.rows(0, 6).norm()
    );

    let wl = solver.contact_wrench(lc).unwrap();
    let wr = solver.contact_wrench(rc).unwrap();
    // Vertical forces carry the full weight.
    let total_fz = wl[2] + wr[2];
    assert!(
        (total_fz - total_mass * 9.81).abs() < 1e-2,
        "vertical force {total_fz} != weight {}",
        total_mass * 9.81
    );
    // A line contact resists no roll moment about its own axis.
    assert!(wl[3].abs() < 1e-9 && wr[3].abs() < 1e-9, "Mx not zero");
    let _ = std::fs::remove_file(&path);
}

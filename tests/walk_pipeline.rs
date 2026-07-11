//! Integration test: the full walk pipeline wires together — HumanoidRobot →
//! footsteps → WalkPatternGenerator → WalkTasks → kinematics solve.
//!
//! Run with `cargo test --features placo --test walk_pipeline -- --ignored`.
#![cfg(all(feature = "ffi", feature = "placo"))]

use std::io::Write;
use std::path::PathBuf;

use nalgebra::{Isometry3, Vector3};
use rs_pinocchio::placo::humanoid::{
    make_supports, DummyWalk, FootstepsPlanner, FootstepsPlannerRepetitive, HumanoidParameters,
    HumanoidRobot, Side, WalkPatternGenerator, WalkTasks,
};
use rs_pinocchio::placo::kinematics::KinematicsSolver;
use rs_pinocchio::placo::model::RobotWrapper;

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
        "rs_pinocchio_walk_{}_{}.urdf",
        std::process::id(),
        n
    ));
    let mut f = std::fs::File::create(&path).expect("create temp urdf");
    f.write_all(BIPED_URDF.as_bytes()).expect("write urdf");
    path
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn full_walk_pipeline_runs() {
    let path = write_fixture();
    let mut robot = HumanoidRobot::from_urdf(&path).expect("load biped");

    let params = HumanoidParameters::new();

    // Plan a few forward steps from the standard feet spacing.
    let mut planner = FootstepsPlannerRepetitive::new(params.clone());
    planner.configure(0.05, 0.0, 0.0, 4);
    let footsteps = planner.plan(
        Side::Left,
        Isometry3::translation(0.0, params.feet_spacing / 2.0, 0.0),
        Isometry3::translation(0.0, -params.feet_spacing / 2.0, 0.0),
    );
    let mut supports = make_supports(&footsteps, 0.0, true, false, true);

    let wpg = WalkPatternGenerator::new(params);
    let initial_com = Vector3::new(0.0, 0.0, wpg.parameters.walk_com_height);
    let mut trajectory = wpg
        .plan(&mut supports, initial_com, 0.0)
        .expect("plan walk");

    // Wire the trajectory into an IK solver via walk tasks.
    let mut solver = KinematicsSolver::new(&robot.robot);
    let tasks = WalkTasks::initialize(&mut solver, &mut robot).expect("init walk tasks");

    // Drive the whole horizon: update targets + solve each step.
    let com_start = trajectory.p_world_com(trajectory.t_start);
    let mut t = trajectory.t_start;
    while t <= trajectory.t_end {
        tasks
            .update_from_trajectory(&mut solver, &robot, &mut trajectory, t)
            .expect("update tasks");
        solver.solve(&mut robot.robot, true).expect("ik solve");
        t += 0.1;
    }

    // The commanded CoM advances forward over the walk.
    let com_end = trajectory.p_world_com(trajectory.t_end);
    assert!(com_end.x > com_start.x, "walk CoM did not advance");
    // The robot state stayed finite through the tracking loop.
    assert!(robot.robot.state.q.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_file(&path);
}

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn dummy_walk_steps_forward() {
    let path = write_fixture();
    let robot = RobotWrapper::from_urdf(&path).expect("load biped");
    let params = HumanoidParameters::new();

    let mut dw = DummyWalk::new(robot, params).expect("dummy walk");
    let x0 = dw.t_world_next.translation.x;

    // Walk a few forward steps, marching the phase each time.
    for _ in 0..4 {
        dw.next_step(0.05, 0.0, 0.0);
        for i in 0..=10 {
            dw.update(i as f64 / 10.0).expect("update");
        }
    }

    // The flying-foot target advanced forward and the state stayed finite.
    assert!(
        dw.t_world_next.translation.x > x0 + 0.05,
        "dummy walk did not advance"
    );
    assert!(dw.robot.state.q.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_file(&path);
}

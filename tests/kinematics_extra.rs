//! Integration tests for the wheel task and the self-collision constraint.
//!
//! Gated behind `ffi` + `placo` and `#[ignore]` (needs a linkable Pinocchio).
//! Run with `cargo test --features placo --test kinematics_extra -- --ignored`.
#![cfg(all(feature = "ffi", feature = "placo"))]

use std::io::Write;
use std::path::PathBuf;

use nalgebra::DMatrix;

use rs_pinocchio::placo::kinematics::{
    AvoidSelfCollisionsConstraint, CollisionDistance, KinematicsSolver,
};
use rs_pinocchio::placo::model::RobotWrapper;
use rs_pinocchio::placo::problem::ConstraintPriority;
use rs_pinocchio::placo::tools::Priority;
use rs_pinocchio::ReferenceFrame;

fn write_fixture(name: &str, urdf: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "rs_pinocchio_{}_{}_{}.urdf",
        name,
        std::process::id(),
        n
    ));
    let mut f = std::fs::File::create(&path).expect("create temp urdf");
    f.write_all(urdf.as_bytes()).expect("write urdf");
    path
}

// A base with a single wheel spinning about its local z-axis. The joint's
// rpy rotates local z onto world y, so the wheel rolls along world x.
const WHEEL_URDF: &str = r#"<?xml version="1.0"?>
<robot name="wheeled">
  <link name="base"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="wheel"><inertial><mass value="0.5"/>
    <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial></link>
  <joint name="wheel_joint" type="continuous">
    <parent link="base"/><child link="wheel"/>
    <origin xyz="0 0 0" rpy="-1.5707963 0 0"/><axis xyz="0 0 1"/>
  </joint>
</robot>
"#;

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn wheel_task_rolls_when_base_moves() {
    let path = write_fixture("wheel", WHEEL_URDF);
    let mut robot = RobotWrapper::from_urdf(&path).expect("load wheeled");

    // Place the base so the wheel (radius 0.1) touches the ground (z = 0).
    let radius = 0.1;
    robot.update_kinematics().unwrap();
    let mut t = robot.t_world_fbase();
    t.translation.z = radius;
    robot.set_t_world_fbase(t);
    robot.update_kinematics().unwrap();

    let base = robot.frame_index("base").expect("base frame");
    let t_base = robot.t_world_frame(base).unwrap();

    let mut solver = KinematicsSolver::new(&robot);
    // Roll the base forward by 5 cm along x, keeping the rest of the pose.
    let mut target = t_base;
    target.translation.x += 0.05;
    let fh = solver.add_frame_task(base, target);
    solver.configure_task(fh.position, "base_pos", Priority::Hard, 1.0);
    solver.configure_task(fh.orientation, "base_ori", Priority::Hard, 1.0);
    let wid = solver.add_wheel_task("wheel_joint", radius, false);
    solver.configure_task(wid, "wheel", Priority::Hard, 1.0);

    let qd = solver.solve(&mut robot, false).expect("solve");
    assert!(qd.iter().all(|v| v.is_finite()));

    let v_wheel = qd[robot.joint_v_offset("wheel_joint").unwrap()];
    // Moving the base forward forces the wheel to spin (rolling, no slip).
    assert!(
        v_wheel.abs() > 1e-4,
        "wheel did not spin: v_wheel = {v_wheel}, qd = {qd:?}"
    );
    // The base velocity is forward along x.
    assert!(qd[0] > 1e-4, "base did not advance: qd[0] = {}", qd[0]);
    let _ = std::fs::remove_file(&path);
}

// Two independent arms whose end frames can approach each other.
const COLLIDER_URDF: &str = r#"<?xml version="1.0"?>
<robot name="collider">
  <link name="base"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="left"><inertial><mass value="0.5"/>
    <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial></link>
  <link name="right"><inertial><mass value="0.5"/>
    <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial></link>
  <joint name="left_joint" type="prismatic">
    <parent link="base"/><child link="left"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/><axis xyz="0 -1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="10"/>
  </joint>
  <joint name="right_joint" type="prismatic">
    <parent link="base"/><child link="right"/>
    <origin xyz="0 -0.2 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="10"/>
  </joint>
</robot>
"#;

#[test]
#[ignore = "requires a linkable Pinocchio install"]
fn self_collision_constraint_blocks_approach() {
    let path = write_fixture("collider", COLLIDER_URDF);
    let mut robot = RobotWrapper::from_urdf(&path).expect("load collider");
    robot.update_kinematics().unwrap();

    let left = robot.frame_index("left").expect("left");
    let right = robot.frame_index("right").expect("right");
    let oa = robot.t_world_frame(left).unwrap().translation.vector;
    let ob = robot.t_world_frame(right).unwrap().translation.vector;

    let mut solver = KinematicsSolver::new(&robot);
    solver.mask_fbase(true);

    // Pull each end toward the other (soft), which would close the gap.
    let lt = solver.add_position_task(left, ob);
    solver.configure_task(lt, "left_pull", Priority::Soft, 1.0);
    let rt = solver.add_position_task(right, oa);
    solver.configure_task(rt, "right_pull", Priority::Soft, 1.0);

    // Report the pair as just at the trigger/margin (nearest points at the
    // frame origins), so the constraint must stop them from approaching.
    let margin = 0.005;
    let cid = solver.add_avoid_self_collisions_constraint();
    solver.configure_constraint(cid, ConstraintPriority::Hard, 1.0);
    solver
        .constraint_mut::<AvoidSelfCollisionsConstraint>(cid)
        .unwrap()
        .distances = vec![CollisionDistance {
        frame_a: left,
        frame_b: right,
        point_a: oa,
        point_b: ob,
        min_distance: margin,
    }];

    let qd = solver.solve(&mut robot, false).expect("solve");
    assert!(qd.iter().all(|v| v.is_finite()));

    // Recompute the separation velocity along the collision normal. With the
    // witness points at the frame origins, the point Jacobian is the frame's
    // linear Jacobian, so this matches the constraint exactly.
    let n = (ob - oa).normalize();
    let ja = robot
        .frame_jacobian(left, ReferenceFrame::LocalWorldAligned)
        .unwrap();
    let jb = robot
        .frame_jacobian(right, ReferenceFrame::LocalWorldAligned)
        .unwrap();
    let diff = jb.rows(0, 3).into_owned() - ja.rows(0, 3).into_owned();
    let row = DMatrix::from_row_slice(1, 3, n.as_slice()) * diff; // 1 x nv
    let sep_vel = (row * &qd)[0];
    // Constraint: sep_vel + min_distance >= margin  =>  sep_vel >= 0.
    assert!(
        sep_vel >= -1e-6,
        "self-collision constraint violated: sep_vel = {sep_vel}"
    );
    // And the pull tasks genuinely wanted to close the gap (non-trivial solve).
    assert!(qd.norm() > 1e-9);

    // Sanity: without the constraint, the ends WOULD approach (sep_vel < 0).
    let mut open = KinematicsSolver::new(&robot);
    open.mask_fbase(true);
    let lt2 = open.add_position_task(left, ob);
    open.configure_task(lt2, "left_pull", Priority::Soft, 1.0);
    let rt2 = open.add_position_task(right, oa);
    open.configure_task(rt2, "right_pull", Priority::Soft, 1.0);
    let qd_open = open.solve(&mut robot, false).expect("solve open");
    let ja2 = robot
        .frame_jacobian(left, ReferenceFrame::LocalWorldAligned)
        .unwrap();
    let jb2 = robot
        .frame_jacobian(right, ReferenceFrame::LocalWorldAligned)
        .unwrap();
    let diff2 = jb2.rows(0, 3).into_owned() - ja2.rows(0, 3).into_owned();
    let sep_open = (DMatrix::from_row_slice(1, 3, n.as_slice()) * diff2 * &qd_open)[0];
    assert!(
        sep_open < -1e-6,
        "expected the unconstrained solve to approach: sep_open = {sep_open}"
    );

    let _ = std::fs::remove_file(&path);
}

# pinocchio-rs

Clean-room Rust bindings for the [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
rigid-body dynamics library, via [cxx](https://cxx.rs). Built for whole-body IK
(forward kinematics + frame Jacobians), reusable across projects.

The bindings are written from scratch against Pinocchio's own public BSD-2 C++
API (`pinocchio::urdf::buildModel`, `forwardKinematics`, `updateFramePlacements`,
`getFrameId`, `getFrameJacobian`, `computeJointJacobians`, `ReferenceFrame`). No
code from any other Rust binding was consulted or copied.

## Versioning

The crate version **tracks the bound Pinocchio version** for convenience — e.g.
`pinocchio-rs = "4.1.x"` binds Pinocchio `4.1.0`. Bump the crate minor/patch to
follow Pinocchio releases. (The FK / frame-Jacobian API is stable across the
4.0/4.1 series, so the crate also links and runs against Pinocchio 4.0.x.)

## Scope (v4.1)

- Load a `Model` from URDF (optional free-flyer root for mobile / whole-body bases)
- Forward kinematics + `updateFramePlacements`
- Frame lookup (`getFrameId`) + frame placements (SE3)
- Frame Jacobians (`getFrameJacobian`; `LOCAL` / `WORLD` / `LOCAL_WORLD_ALIGNED`)
- `nq` / `nv`

Enough for a differential-IK / whole-body-IK layer to consume. Dynamics (mass
matrix, RNEA/ABA, derivatives) are intentionally out of scope for now.

## Public API

Everything returns [`nalgebra`](https://nalgebra.org) types.

```rust
use nalgebra::DVector;
use pinocchio_rs::{Model, ReferenceFrame};

// Build (floating_base = true prepends a free-flyer root: +7 nq, +6 nv).
let mut model = Model::from_urdf("robot.urdf", false)?;

let (nq, nv) = (model.nq(), model.nv());
let q = DVector::<f64>::zeros(nq);

// Forward kinematics + frame placement (Isometry3<f64>).
model.forward_kinematics(&q)?;
model.update_frame_placements();
let tip = model.frame_id("tool").expect("frame exists");   // -> Option<usize>
let pose = model.frame_placement(tip)?;                     // -> Isometry3<f64>

// 6 x nv frame Jacobian (DMatrix<f64>), rows = [vx vy vz wx wy wz].
model.compute_joint_jacobians(&q)?;
let j = model.frame_jacobian(tip, ReferenceFrame::LocalWorldAligned)?;
assert_eq!((j.nrows(), j.ncols()), (6, nv));
# Ok::<(), pinocchio_rs::Error>(())
```

Convenience by-name variants (`frame_placement_by_name`,
`frame_jacobian_by_name`) return [`Error::FrameNotFound`] if the frame is absent.
Errors are a `thiserror` enum: `UrdfLoad`, `FrameNotFound`, `DimMismatch`, `Cxx`.

`ReferenceFrame` discriminants are the stable ABI with the C++ shim:
`Local = 0`, `World = 1`, `LocalWorldAligned = 2` (mapped explicitly onto
Pinocchio's own enum).

### Cargo features

- `trace` — emit [`tracing`](https://docs.rs/tracing) spans/events from the wrapper.

## Build requirements

Pinocchio 4.1.0 (headers + libs) must be installed. The build script locates it
in this order:

1. `pkg-config` (conda-forge / robotpkg ship a `pinocchio.pc`).
2. `PINOCCHIO_PREFIX` — an install prefix containing `include/` and `lib/`.
3. `/opt/ros/$ROS_DISTRO` — ROS 2 debian packaging (`ros-<distro>-pinocchio`).

Install options:

```sh
# conda-forge (easiest, gives you exactly 4.1.0)
conda install -c conda-forge pinocchio=4.1.0

# robotpkg (Ubuntu; add the robotpkg apt source first)
sudo apt install robotpkg-pinocchio

# ROS 2 (4.0.x on Jazzy; API-compatible for this crate's scope)
sudo apt install ros-<distro>-pinocchio
source /opt/ros/<distro>/setup.bash   # exports ROS_DISTRO
```

If Pinocchio lives in a custom prefix:

```sh
PINOCCHIO_PREFIX=/path/to/prefix cargo build
```

Also required at build time: a C++17 compiler, Eigen 3 (via `pkg-config eigen3`
or `/usr/include/eigen3`), and Boost headers.

## Tests

The integration tests link against a live Pinocchio and are gated with
`#[ignore]` so `cargo test` stays green when Pinocchio is absent (note: the crate
itself still needs Pinocchio to *link*, like any `-sys`-style binding). Run them
explicitly once it is installed:

```sh
cargo test -- --ignored
```

Licensed BSD-2-Clause. Not affiliated with any other `pinocchio-rs` crate.

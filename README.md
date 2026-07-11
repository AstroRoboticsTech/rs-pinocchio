# rs-pinocchio

Rust rigid-body dynamics **and** QP-based planning/control: clean-room
[cxx](https://cxx.rs) bindings to the
[Pinocchio](https://github.com/stack-of-tasks/pinocchio) C++ library, plus a
pure-Rust port of [Rhoban PlaCo](https://github.com/Rhoban/placo) — task-space
inverse kinematics & dynamics, footstep and walk planning.

The Pinocchio bindings are written from scratch against Pinocchio's own public
BSD-2 C++ API (`pinocchio::urdf::buildModel`, `forwardKinematics`,
`getFrameJacobian`, CRBA / RNEA / centroidal, `ReferenceFrame`, …). No code from
any other Rust binding was consulted or copied.

## Versioning

The crate follows SemVer (`MAJOR.MINOR.PATCH`), with the components repurposed to
track the bound Pinocchio version:

- **`MAJOR.MINOR`** mirror the bound Pinocchio version's major.minor — `4.1.x`
  binds Pinocchio **4.1**.
- **`PATCH`** is *this crate's* own release counter within that Pinocchio line
  (`4.1.0`, `4.1.1`, …), independent of Pinocchio's own patch number.

The exact bound Pinocchio version is **4.1.0** (the crate also links and runs
against Pinocchio 4.0.x for the FK / Jacobian subset). SemVer allows only three
numeric components, so a four-part "custom bump" like `4.1.0.2` is **not** a valid
crate version — the custom bump lives in the patch instead.

## Scope

**Pinocchio binding (`ffi` feature):**

- `Model` / `RobotWrapper` from URDF (optional free-flyer root for floating bases)
- Forward kinematics, frame placements (SE3), frame lookup
- Frame & joint Jacobians (+ time variation): `LOCAL` / `WORLD` / `LOCAL_WORLD_ALIGNED`
- Dynamics: mass matrix (CRBA), generalized gravity, non-linear effects (RNEA),
  CoM & CoM Jacobian, centroidal map, kinematic hessians
- Manifold `integrate` / `difference` / `neutral`, joint offsets & limits

**PlaCo port (`placo` feature), pure Rust:**

- `tools` (splines, polynomials, …) and `problem` — a QP modeling layer over
  [clarabel](https://docs.rs/clarabel); both need no native deps
- `kinematics` — task-space IK solver, ~15 tasks and 5 constraints
- `dynamics` — task-space ID solver, tasks, contacts (point / 6D / line / wrench /
  puppet / task) and joint/velocity/torque limits
- `humanoid` — footstep planning, LIPM, swing-foot, walk pattern generator,
  `HumanoidRobot`, `WalkTasks`

Collision / self-collision geometry queries (coal) are not yet bound.

## Public API

Everything returns [`nalgebra`](https://nalgebra.org) types.

```rust
use nalgebra::DVector;
use rs_pinocchio::{Model, ReferenceFrame};

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
# Ok::<(), rs_pinocchio::Error>(())
```

Convenience by-name variants (`frame_placement_by_name`,
`frame_jacobian_by_name`) return [`Error::FrameNotFound`] if the frame is absent.
Errors are a `thiserror` enum: `UrdfLoad`, `FrameNotFound`, `DimMismatch`, `Cxx`.

`ReferenceFrame` discriminants are the stable ABI with the C++ shim:
`Local = 0`, `World = 1`, `LocalWorldAligned = 2` (mapped explicitly onto
Pinocchio's own enum).

### Cargo features

| Feature | Default | Enables |
|---------|:-------:|---------|
| `ffi`   |   ✅    | The Pinocchio cxx binding (`Model`, `RobotWrapper`). Needs Pinocchio at build time (see [Build requirements](#build-requirements)). |
| `placo` |         | The pure-Rust PlaCo port. Its `tools` + `problem` layers need no native deps; the kinematics / dynamics solvers additionally require `ffi`. |
| `trace` |         | Emit [`tracing`](https://docs.rs/tracing) spans/events from the safe wrappers. |

```toml
# Default — the Pinocchio binding only:
rs-pinocchio = "4.1"

# Pure-Rust PlaCo planning layers, no Pinocchio needed:
rs-pinocchio = { version = "4.1", default-features = false, features = ["placo"] }

# Full framework — Pinocchio binding + PlaCo IK/ID solvers and walk planning:
rs-pinocchio = { version = "4.1", features = ["placo"] }
```

## Build requirements

By default the build is **self-contained**: Pinocchio 4.1.0 and its URDF-parser
dependencies are vendored as git submodules under `third_party/` and compiled
from source (collision support off) into `third_party/install` on the first
`ffi` build, then cached. No system Pinocchio, conda, or pip is required.

```sh
git clone --recurse-submodules https://github.com/AstroRoboticsTech/rs-pinocchio
cargo build --features ffi        # first build compiles Pinocchio (~10-15 min), then caches
```

If you cloned without `--recurse-submodules`, the build script initializes the
needed submodules itself.

Build-time tools + system libraries (not vendored): a **C++17 compiler**,
**CMake ≥ 3.10** + a generator (**Ninja**), **Eigen 3** (via `pkg-config eigen3`,
or brew/`/usr/include/eigen3`), and **Boost** headers (via `BOOST_ROOT`, brew, or
a standard prefix).

### Using a preinstalled Pinocchio instead

To skip the source build, point the crate at an existing Pinocchio 4.1.0 install
— the build script prefers it over the vendored build, in this order:

1. `pkg-config` (conda-forge / robotpkg ship a `pinocchio.pc`).
2. `PINOCCHIO_PREFIX` — an install prefix containing `include/` and `lib/`.
3. `/opt/ros/$ROS_DISTRO` — ROS 2 debian packaging (`ros-<distro>-pinocchio`).

```sh
PINOCCHIO_PREFIX=/path/to/prefix cargo build --features ffi
```

## Tests

The integration tests link against a live Pinocchio and are gated with
`#[ignore]` so `cargo test` stays green when Pinocchio is absent (note: the crate
itself still needs Pinocchio to *link*, like any `-sys`-style binding). Run them
explicitly once it is installed:

```sh
cargo test -- --ignored
```

Licensed BSD-2-Clause. Not affiliated with the `pinocchio` or `pinocchio_rs` crates.

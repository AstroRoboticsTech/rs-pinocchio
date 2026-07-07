# pinocchio-rs

Clean-room Rust bindings for the [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
rigid-body dynamics library, via [cxx](https://cxx.rs). Built for whole-body IK
(forward kinematics + frame Jacobians), reusable across projects.

## Versioning

The crate version **tracks the bound Pinocchio version** for convenience — e.g.
`pinocchio-rs = "4.1.x"` binds Pinocchio `4.1.0`. Bump the crate minor/patch to
follow Pinocchio releases.

## Scope (v4.1)

- Load a `Model` from URDF (optional free-flyer root for mobile bases)
- Forward kinematics + `updateFramePlacements`
- Frame lookup (`getFrameId`) + frame placements (SE3)
- Frame Jacobians (`getFrameJacobian`, LOCAL / WORLD / LOCAL_WORLD_ALIGNED)
- `nq` / `nv`

Enough for a differential-IK / whole-body-IK layer to consume.

## Build requirements

Pinocchio 4.1.0 must be installed (headers + libs discoverable via `pkg-config`):

```sh
# conda-forge (easiest)
conda install -c conda-forge pinocchio=4.1.0
# or robotpkg (Ubuntu)
```

Licensed BSD-2-Clause. Not affiliated with any other `pinocchio-rs` crate.

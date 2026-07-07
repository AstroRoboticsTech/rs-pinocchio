//! Build script: locate Pinocchio (+ Eigen) and compile the cxx bridge + shim.
//!
//! Resolution order for Pinocchio:
//!
//! 1. `pkg-config` (conda-forge / robotpkg ship a `pinocchio.pc`).
//! 2. `PINOCCHIO_PREFIX` env var (an install prefix with `include/` + `lib/`).
//! 3. `/opt/ros/$ROS_DISTRO` (ROS 2 debian packaging).
//!
//! If none resolve, panic with an install hint.
//!
//! Prefers pkg-config + cc over any external build tool (e.g. xmake) so the
//! crate cross-compiles cleanly.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=cpp/pinocchio_shim.cpp");
    println!("cargo:rerun-if-changed=cpp/pinocchio_shim.h");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-env-changed=PINOCCHIO_PREFIX");
    println!("cargo:rerun-if-env-changed=ROS_DISTRO");

    let mut include_paths: Vec<PathBuf> = Vec::new();

    // --- Eigen (ubiquitous; needed to compile the Pinocchio headers) ----------
    match pkg_config::Config::new().probe("eigen3") {
        Ok(eigen) => include_paths.extend(eigen.include_paths),
        Err(_) => {
            let fallback = PathBuf::from("/usr/include/eigen3");
            if fallback.exists() {
                include_paths.push(fallback);
            }
        }
    }

    // --- Pinocchio ------------------------------------------------------------
    let mut located = false;

    // 1. pkg-config (emits link directives automatically).
    if let Ok(pino) = pkg_config::Config::new().probe("pinocchio") {
        include_paths.extend(pino.include_paths);
        located = true;
    }

    // 2. / 3. PINOCCHIO_PREFIX or ROS install layout (manual link directives).
    if !located {
        for prefix in candidate_prefixes() {
            let inc = prefix.join("include");
            // `fwd.hpp` is a stable top-level sentinel across Pinocchio 2.x-4.x
            // (the per-class header layout changed between minor versions).
            if !inc.join("pinocchio/fwd.hpp").exists() {
                continue;
            }
            include_paths.push(inc);

            for sub in ["lib", "lib/x86_64-linux-gnu", "lib64"] {
                let libdir = prefix.join(sub);
                if libdir.exists() {
                    println!("cargo:rustc-link-search=native={}", libdir.display());
                    // Embed an rpath so test/bin targets find the .so at runtime
                    // without needing LD_LIBRARY_PATH.
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir.display());
                }
            }
            // Modern Pinocchio (3.x/4.x) splits the library; we need the default
            // scalar instantiations plus the URDF parser.
            println!("cargo:rustc-link-lib=dylib=pinocchio_default");
            println!("cargo:rustc-link-lib=dylib=pinocchio_parsers");

            located = true;
            break;
        }
    }

    if !located {
        panic!(
            "\nPinocchio 4.1.0 not found.\n\
             Install it, then set PINOCCHIO_PREFIX if it is in a non-standard prefix:\n\
             \n  conda-forge:  conda install -c conda-forge pinocchio=4.1.0\n\
             \n  robotpkg:     sudo apt install robotpkg-pinocchio   (after adding the robotpkg apt source)\n\
             \n  ROS 2:        sudo apt install ros-<distro>-pinocchio  &&  source /opt/ros/<distro>/setup.bash\n\
             \nThen re-run `cargo build`, or set PINOCCHIO_PREFIX=/path/to/prefix.\n"
        );
    }

    // --- Compile the cxx bridge + shim ---------------------------------------
    let mut build = cxx_build::bridge("src/ffi.rs");
    build.file("cpp/pinocchio_shim.cpp").std("c++17");
    for p in dedup(&include_paths) {
        build.include(p);
    }
    // Pinocchio's default joint variant has >20 alternatives, exceeding Boost
    // MPL's preprocessed list/vector limit. Pinocchio itself is compiled with
    // these raised — match them so the headers instantiate (and stay ABI-compatible
    // with the prebuilt libpinocchio_default).
    build
        .define("BOOST_MPL_LIMIT_LIST_SIZE", "30")
        .define("BOOST_MPL_LIMIT_VECTOR_SIZE", "30")
        .define("BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS", None);
    // Pinocchio 3/4 headers trip a lot of these on GCC; they are not our bugs.
    build
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-deprecated-copy")
        .flag_if_supported("-Wno-unused-parameter");
    build.compile("pinocchio_shim");
}

/// Candidate install prefixes to probe (in priority order).
fn candidate_prefixes() -> Vec<PathBuf> {
    let mut prefixes = Vec::new();
    if let Ok(prefix) = std::env::var("PINOCCHIO_PREFIX") {
        prefixes.push(PathBuf::from(prefix));
    }
    if let Ok(distro) = std::env::var("ROS_DISTRO") {
        prefixes.push(PathBuf::from(format!("/opt/ros/{distro}")));
    }
    prefixes
}

fn dedup(paths: &[PathBuf]) -> Vec<&Path> {
    let mut seen = std::collections::HashSet::new();
    paths
        .iter()
        .filter(|p| seen.insert(p.as_path()))
        .map(|p| p.as_path())
        .collect()
}

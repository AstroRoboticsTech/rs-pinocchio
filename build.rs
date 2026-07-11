//! Build script: locate (or build) Pinocchio (+ Eigen/Boost) and compile the
//! cxx bridge + shim.
//!
//! Resolution order for Pinocchio:
//!
//! 1. `pkg-config` (conda-forge / robotpkg ship a `pinocchio.pc`).
//! 2. `PINOCCHIO_PREFIX` env var (an install prefix with `include/` + `lib/`).
//! 3. `/opt/ros/$ROS_DISTRO` (ROS 2 debian packaging).
//! 4. A self-contained source build of the vendored `third_party/` submodules
//!    into `third_party/install` (built once, then cached).
//!
//! Eigen and Boost are resolved from the system (pkg-config / brew / standard
//! prefixes); only Pinocchio and its URDF-parser deps are vendored + built.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // The Pinocchio binding is gated behind the `ffi` feature. When it is off
    // (e.g. a pure-Rust `placo` build/test), there is nothing native to compile
    // or link, so skip the probe entirely.
    if std::env::var_os("CARGO_FEATURE_FFI").is_none() {
        return;
    }

    // docs.rs has no Pinocchio install: skip the native probe + C++ compile so
    // `cargo doc` (which never links) still builds the Rust API docs.
    if std::env::var_os("DOCS_RS").is_some() {
        return;
    }

    println!("cargo:rerun-if-changed=cpp/pinocchio_shim.cpp");
    println!("cargo:rerun-if-changed=cpp/pinocchio_shim.h");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PINOCCHIO_PREFIX");
    println!("cargo:rerun-if-env-changed=BOOST_ROOT");
    println!("cargo:rerun-if-env-changed=ROS_DISTRO");

    let mut include_paths: Vec<PathBuf> = Vec::new();

    // --- Eigen + Boost (needed to compile the Pinocchio headers) --------------
    add_eigen(&mut include_paths);
    add_boost(&mut include_paths);

    // --- Pinocchio ------------------------------------------------------------
    // 1. pkg-config (emits link directives automatically).
    if let Ok(pino) = pkg_config::Config::new().probe("pinocchio") {
        include_paths.extend(pino.include_paths);
        compile_shim(&include_paths);
        return;
    }

    // 2. / 3. PINOCCHIO_PREFIX or ROS install layout.
    for prefix in candidate_prefixes() {
        if link_prefix(&prefix, &mut include_paths) {
            compile_shim(&include_paths);
            return;
        }
    }

    // 4. Vendored source build (built once into third_party/install, then cached).
    let prefix = build_vendored();
    if !link_prefix(&prefix, &mut include_paths) {
        panic!(
            "vendored Pinocchio build did not produce a usable prefix at {}",
            prefix.display()
        );
    }
    compile_shim(&include_paths);
}

/// Adds Eigen's include dir (pkg-config `eigen3`, else the common fallback).
fn add_eigen(include_paths: &mut Vec<PathBuf>) {
    match pkg_config::Config::new().probe("eigen3") {
        Ok(eigen) => include_paths.extend(eigen.include_paths),
        Err(_) => {
            for c in ["/opt/homebrew/include/eigen3", "/usr/include/eigen3"] {
                let p = PathBuf::from(c);
                if p.exists() {
                    include_paths.push(p);
                    break;
                }
            }
        }
    }
}

/// Adds Boost's include dir. A vendored Pinocchio build links the system Boost,
/// so the shim needs its headers on the include path (BOOST_ROOT, then brew /
/// standard prefixes). A prebuilt prefix that bundles Boost is covered by its
/// own `include/`, so a miss here is not fatal.
fn add_boost(include_paths: &mut Vec<PathBuf>) {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(root) = std::env::var("BOOST_ROOT") {
        candidates.push(PathBuf::from(root).join("include"));
    }
    for c in [
        "/opt/homebrew/include",
        "/opt/homebrew/opt/boost/include",
        "/usr/local/include",
        "/usr/include",
    ] {
        candidates.push(PathBuf::from(c));
    }
    for c in candidates {
        if c.join("boost/version.hpp").exists() {
            include_paths.push(c);
            return;
        }
    }
}

/// Adds `prefix`'s include + link directives if it holds a Pinocchio install.
/// Returns whether it looked like a valid prefix.
fn link_prefix(prefix: &Path, include_paths: &mut Vec<PathBuf>) -> bool {
    let inc = prefix.join("include");
    // `fwd.hpp` is a stable top-level sentinel across Pinocchio 2.x-4.x.
    if !inc.join("pinocchio/fwd.hpp").exists() {
        return false;
    }
    include_paths.push(inc.clone());
    // urdfdom_headers 1.1.x installs `urdf_model/` under a namespaced
    // `include/urdfdom_headers/` dir; add it so `<urdf_model/model.h>` resolves.
    let urdf_ns = inc.join("urdfdom_headers");
    if urdf_ns.join("urdf_model/model.h").exists() {
        include_paths.push(urdf_ns);
    }

    for sub in ["lib", "lib/x86_64-linux-gnu", "lib64"] {
        let libdir = prefix.join(sub);
        if libdir.exists() {
            println!("cargo:rustc-link-search=native={}", libdir.display());
            // Embed an rpath so test/bin targets find the .dylib/.so at runtime.
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir.display());
        }
    }
    // Modern Pinocchio (3.x/4.x) splits the library; the default scalar
    // instantiations plus the URDF parser are what the shim needs.
    println!("cargo:rustc-link-lib=dylib=pinocchio_default");
    println!("cargo:rustc-link-lib=dylib=pinocchio_parsers");
    true
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

/// Builds the vendored Pinocchio + URDF deps (collision off) into
/// `third_party/install`, once. Returns the install prefix.
fn build_vendored() -> PathBuf {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let third_party = manifest.join("third_party");
    let prefix = third_party.join("install");

    // Already built (sentinel present)? Reuse it.
    if prefix.join("include/pinocchio/config.hpp").exists() {
        return prefix;
    }

    ensure_submodules(&manifest, &third_party);

    let build_root = third_party.join("build");
    let cmake_common: Vec<String> = vec![
        "-GNinja".into(),
        "-DCMAKE_BUILD_TYPE=Release".into(),
        format!("-DCMAKE_INSTALL_PREFIX={}", prefix.display()),
        "-DBUILD_SHARED_LIBS=ON".into(),
        "-DBUILD_TESTING=OFF".into(),
        // CMake 4 dropped compatibility with the pre-3.5 minimums some of these
        // projects declare; restore it.
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5".into(),
        format!("-DCMAKE_INSTALL_RPATH={}", prefix.join("lib").display()),
        "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON".into(),
        // Let each project find the ones installed before it (and system Boost/Eigen).
        format!(
            "-DCMAKE_PREFIX_PATH={};/opt/homebrew;/opt/homebrew/opt/boost;/opt/homebrew/opt/eigen",
            prefix.display()
        ),
    ];

    let cmake_build = |name: &str, extra: &[&str]| {
        let src = third_party.join(name);
        let build = build_root.join(name);
        std::fs::create_dir_all(&build).ok();
        let mut cfg = vec!["-S".to_string(), src.display().to_string()];
        cfg.push("-B".into());
        cfg.push(build.display().to_string());
        cfg.extend(cmake_common.iter().cloned());
        cfg.extend(extra.iter().map(|s| s.to_string()));
        run("cmake", &cfg, &third_party);
        run(
            "cmake",
            &[
                "--build".into(),
                build.display().to_string(),
                "--target".into(),
                "install".into(),
            ],
            &third_party,
        );
    };

    // URDF-parser dependency chain, then Pinocchio itself.
    cmake_build("tinyxml2", &[]);
    cmake_build("console_bridge", &[]);
    cmake_build("urdfdom_headers", &[]);
    cmake_build("urdfdom", &[]);
    cmake_build(
        "pinocchio",
        &[
            "-DBUILD_PYTHON_INTERFACE=OFF",
            "-DBUILD_WITH_COLLISION_SUPPORT=OFF",
            "-DBUILD_WITH_URDF_SUPPORT=ON",
            "-DBUILD_WITH_AUTODIFF_SUPPORT=OFF",
            "-DBUILD_WITH_CASADI_SUPPORT=OFF",
            "-DBUILD_WITH_CODEGEN_SUPPORT=OFF",
            "-DBUILD_WITH_OPENMP_SUPPORT=OFF",
            "-DBUILD_UTILS=OFF",
            // Examples/benchmark pull in the large example-robot-data submodule.
            "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_BENCHMARK=OFF",
            "-DGENERATE_PYTHON_STUBS=OFF",
        ],
    );

    prefix
}

/// Checks out the vendored source submodules if they are not present.
fn ensure_submodules(manifest: &Path, third_party: &Path) {
    if third_party.join("pinocchio/CMakeLists.txt").exists() {
        return;
    }
    println!("cargo:warning=initializing vendored Pinocchio source submodules");
    run(
        "git",
        &[
            "submodule".into(),
            "update".into(),
            "--init".into(),
            "third_party/tinyxml2".into(),
            "third_party/console_bridge".into(),
            "third_party/urdfdom_headers".into(),
            "third_party/urdfdom".into(),
            "third_party/pinocchio".into(),
        ],
        manifest,
    );
    // Pinocchio needs its jrl-cmakemodules submodule, but not the large
    // example-robot-data models submodule.
    run(
        "git",
        &[
            "-C".into(),
            "third_party/pinocchio".into(),
            "submodule".into(),
            "update".into(),
            "--init".into(),
            "cmake".into(),
        ],
        manifest,
    );
}

/// Runs a command in `cwd`, panicking with its output on failure.
fn run(program: &str, args: &[String], cwd: &Path) {
    let status = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn `{program}`: {e}"));
    if !status.success() {
        panic!("`{program} {}` failed with {status}", args.join(" "));
    }
}

/// Compiles the cxx bridge + shim against the resolved include paths.
fn compile_shim(include_paths: &[PathBuf]) {
    let mut build = cxx_build::bridge("src/ffi.rs");
    build.file("cpp/pinocchio_shim.cpp").std("c++17");
    for p in dedup(include_paths) {
        build.include(p);
    }
    // Pinocchio's default joint variant has >20 alternatives, exceeding Boost
    // MPL's preprocessed list/vector limit. Match Pinocchio's own raised limits
    // so the headers instantiate (and stay ABI-compatible with libpinocchio).
    build
        .define("BOOST_MPL_LIMIT_LIST_SIZE", "30")
        .define("BOOST_MPL_LIMIT_VECTOR_SIZE", "30")
        .define("BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS", None);
    // Pinocchio 3/4 headers trip a lot of these; they are not our bugs.
    build
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-deprecated-copy")
        .flag_if_supported("-Wno-unused-parameter");
    build.compile("pinocchio_shim");
}

fn dedup(paths: &[PathBuf]) -> Vec<&Path> {
    let mut seen = std::collections::HashSet::new();
    paths
        .iter()
        .filter(|p| seen.insert(p.as_path()))
        .map(|p| p.as_path())
        .collect()
}

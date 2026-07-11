//! Gear and kinetic-energy regularization kinematics tasks (PlaCo
//! `kinematics::GearTask`, `KineticEnergyRegularizationTask`).

use nalgebra::{DMatrix, DVector};

use super::task::{KinematicsTask, TaskBase};
use crate::error::{Error, Result};
use crate::placo::model::RobotWrapper;

/// Couples joints with gear ratios: keeps `q_target − Σ ratio·q_source` constant
/// (PlaCo `GearTask`).
pub struct GearTask {
    base: TaskBase,
    /// `(target joint, [(source joint, ratio)])` entries.
    pub gears: Vec<(String, Vec<(String, f64)>)>,
}

impl GearTask {
    pub(crate) fn new() -> Self {
        Self {
            base: TaskBase::default(),
            gears: Vec::new(),
        }
    }

    /// Sets a single gear relation `target = ratio · source` (replaces existing
    /// sources for `target`).
    pub fn set_gear(&mut self, target: impl Into<String>, source: impl Into<String>, ratio: f64) {
        let target = target.into();
        self.gears.retain(|(t, _)| *t != target);
        self.gears.push((target, vec![(source.into(), ratio)]));
    }

    /// Adds a source term `+= ratio · source` to `target`'s gear relation.
    pub fn add_gear(&mut self, target: impl Into<String>, source: impl Into<String>, ratio: f64) {
        let target = target.into();
        let source = source.into();
        if let Some((_, sources)) = self.gears.iter_mut().find(|(t, _)| *t == target) {
            sources.push((source, ratio));
        } else {
            self.gears.push((target, vec![(source, ratio)]));
        }
    }
}

impl KinematicsTask for GearTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "gear"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let n = robot.nv();
        let mut a = DMatrix::zeros(self.gears.len(), n);
        let mut b = DVector::zeros(self.gears.len());
        for (k, (target, sources)) in self.gears.iter().enumerate() {
            let t_off = robot.joint_v_offset(target)?;
            a[(k, t_off)] = -1.0;
            // For single-DoF actuated joints, q index = v index + 1 (floating base).
            b[k] = robot.state.q[t_off + 1];
            for (source, ratio) in sources {
                let s_off = robot.joint_v_offset(source)?;
                a[(k, s_off)] = *ratio;
                b[k] -= robot.state.q[s_off + 1] * ratio;
            }
        }
        self.base.a = a;
        self.base.b = b;
        Ok(())
    }
}

/// Regularizes towards minimum kinetic energy `½·qdᵀ·M·qd` (PlaCo
/// `KineticEnergyRegularizationTask`). Requires `solver.dt`.
///
/// Excludes the floating base, matching PlaCo.
pub struct KineticEnergyRegularizationTask {
    base: TaskBase,
}

impl KineticEnergyRegularizationTask {
    pub(crate) fn new() -> Self {
        Self {
            base: TaskBase::default(),
        }
    }
}

impl KinematicsTask for KineticEnergyRegularizationTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "kinetic_energy_regularization"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, dt: f64) -> Result<()> {
        if dt == 0.0 {
            return Err(Error::Solver(
                "KineticEnergyRegularizationTask requires solver.dt".into(),
            ));
        }
        let m = robot.mass_matrix()?;
        let m_sqrt = symmetric_sqrt(&m);
        self.base.a = m_sqrt / (2.0_f64.sqrt() * dt);
        self.base.b = DVector::zeros(robot.nv());
        Ok(())
    }
}

/// Symmetric square root of an SPD matrix via its eigendecomposition.
fn symmetric_sqrt(m: &DMatrix<f64>) -> DMatrix<f64> {
    let eig = m.clone().symmetric_eigen();
    let sqrt_vals = DMatrix::from_diagonal(&eig.eigenvalues.map(|v| v.max(0.0).sqrt()));
    &eig.eigenvectors * sqrt_vals * eig.eigenvectors.transpose()
}

/// Which part of the Jacobian the manipulability is measured over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManipulabilityType {
    /// Translational manipulability (Jacobian rows 0..3).
    Position,
    /// Rotational manipulability (Jacobian rows 3..6).
    Orientation,
    /// Both (Jacobian rows 0..6).
    Both,
}

/// Gradient-ascent task on the frame manipulability `sqrt(det(J·Jᵀ))` (PlaCo
/// `ManipulabilityTask`). Excludes the floating base.
pub struct ManipulabilityTask {
    base: TaskBase,
    /// Frame index.
    pub frame_index: usize,
    /// Regularization magnitude.
    pub lambda: f64,
    /// Which manipulability to optimize.
    pub kind: ManipulabilityType,
    /// If true, minimize instead of maximize manipulability.
    pub minimize: bool,
    /// The manipulability value from the last update.
    pub manipulability: f64,
}

impl ManipulabilityTask {
    pub(crate) fn new(frame_index: usize, kind: ManipulabilityType, lambda: f64) -> Self {
        Self {
            base: TaskBase::default(),
            frame_index,
            lambda,
            kind,
            minimize: false,
            manipulability: 0.0,
        }
    }

    fn mask_matrix(&self, m: &DMatrix<f64>) -> DMatrix<f64> {
        let cols = m.ncols() - 6;
        match self.kind {
            ManipulabilityType::Position => m.view((0, 6), (3, cols)).into_owned(),
            ManipulabilityType::Orientation => m.view((3, 6), (3, cols)).into_owned(),
            ManipulabilityType::Both => m.view((0, 6), (6, cols)).into_owned(),
        }
    }
}

impl KinematicsTask for ManipulabilityTask {
    fn base(&self) -> &TaskBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut TaskBase {
        &mut self.base
    }
    fn type_name(&self) -> &'static str {
        "manipulability"
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn update(&mut self, robot: &mut RobotWrapper, _dt: f64) -> Result<()> {
        let n = robot.nv();
        let j_unmasked = robot.frame_jacobian(self.frame_index, crate::ReferenceFrame::Local)?;
        let j = self.mask_matrix(&j_unmasked);
        let jjt = &j * j.transpose();
        self.manipulability = jjt.determinant().max(0.0).sqrt();

        let jjt_inv = match jjt.clone().try_inverse() {
            Some(inv) => inv,
            None => {
                // Singular: skip the task this step.
                self.base.a = DMatrix::zeros(0, n);
                self.base.b = DVector::zeros(0);
                return Ok(());
            }
        };

        robot.compute_hessians();
        let mut grad = DVector::zeros(n - 6);
        for dof in 6..n {
            let h_dof = self.mask_matrix(&robot.frame_hessian(self.frame_index, dof)?);
            let jh = &j * h_dof.transpose();
            grad[dof - 6] = self.manipulability * jh.component_mul(&jjt_inv).sum();
        }

        let mut a = DMatrix::zeros(n - 6, n);
        for i in 0..(n - 6) {
            a[(i, 6 + i)] = self.lambda;
        }
        let mut b = grad / (2.0 * self.lambda);
        if self.minimize {
            b = -b;
        }
        self.base.a = a;
        self.base.b = b;
        Ok(())
    }
}

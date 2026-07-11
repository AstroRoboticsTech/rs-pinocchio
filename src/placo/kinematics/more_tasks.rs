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

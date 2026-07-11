//! QP failure type (PlaCo `QPError`).

use thiserror::Error;

/// Raised by [`super::Problem::solve`] when the QP cannot be solved.
#[derive(Debug, Error)]
pub enum QpError {
    /// The QP is infeasible (conflicting hard constraints) or the solver failed.
    #[error("QP solve failed: {0}")]
    Infeasible(String),

    /// The problem was malformed (empty / inconsistent constraint expressions).
    #[error("QP problem is malformed: {0}")]
    Malformed(String),

    /// The solution contained NaNs.
    #[error("QP solution contained NaN")]
    Nan,
}

/// Result alias for QP operations.
pub type QpResult<T> = std::result::Result<T, QpError>;

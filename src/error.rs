//! Error types for the safe wrapper.

use thiserror::Error;

/// Errors returned by [`crate::Model`] operations.
#[derive(Debug, Error)]
pub enum Error {
    /// The URDF could not be parsed / the model could not be built.
    #[error("failed to load URDF model: {0}")]
    UrdfLoad(String),

    /// No frame with the requested name exists in the model.
    #[error("frame not found: {0}")]
    FrameNotFound(String),

    /// A supplied slice/vector had the wrong length for the operation.
    #[error("dimension mismatch: expected {expected}, got {got} ({what})")]
    DimMismatch {
        /// What was being sized (e.g. `"configuration q"`).
        what: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length supplied.
        got: usize,
    },

    /// A C++ exception surfaced from the Pinocchio shim.
    #[error("pinocchio error: {0}")]
    Cxx(String),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, Error>;

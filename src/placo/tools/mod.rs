//! Geometry, spline and math helpers (PlaCo `placo::tools`).

mod axises_mask;
mod cubic_spline;
mod cubic_spline_3d;
mod directions;
mod polynom;
mod prioritized;
mod segment;
mod utils;

pub use axises_mask::{AxisesMask, MaskFrame};
pub use cubic_spline::CubicSpline;
pub use cubic_spline_3d::CubicSpline3D;
pub use directions::{directions_2d, directions_3d};
pub use polynom::Polynom;
pub use prioritized::{Prioritized, Priority};
pub use segment::Segment;
pub use utils::{
    exp3, flatten_on_floor, frame_yaw, interpolate_frames, optimal_transformation,
    rotation_from_axis, safe_acos, wrap_angle,
};

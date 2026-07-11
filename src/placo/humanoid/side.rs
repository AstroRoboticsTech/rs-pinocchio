//! Support side of a humanoid (PlaCo `HumanoidRobot::Side`).

/// Which foot / support side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Left foot.
    Left,
    /// Right foot.
    Right,
    /// Double support (both feet).
    Both,
}

impl Side {
    /// The opposite of `Left`/`Right` (`Both` maps to itself).
    pub fn other(self) -> Side {
        match self {
            Side::Left => Side::Right,
            Side::Right => Side::Left,
            Side::Both => Side::Both,
        }
    }

    /// Parses `"left"`/`"right"`/`"both"`.
    ///
    /// # Panics
    /// On any other string.
    pub fn from_name(name: &str) -> Side {
        match name {
            "left" => Side::Left,
            "right" => Side::Right,
            "both" => Side::Both,
            other => panic!("Side: invalid side: {other}"),
        }
    }

    /// The lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
            Side::Both => "both",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other_and_roundtrip() {
        assert_eq!(Side::Left.other(), Side::Right);
        assert_eq!(Side::Right.other(), Side::Left);
        for s in [Side::Left, Side::Right, Side::Both] {
            assert_eq!(Side::from_name(s.name()), s);
        }
    }
}

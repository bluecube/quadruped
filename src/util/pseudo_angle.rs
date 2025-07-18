use std::{
    f64::consts::{PI, SQRT_2},
    ops::Neg,
};

use more_asserts::{assert_ge, assert_le, debug_assert_gt};
use nalgebra::{Point2, Vector2};

/// Represents signed angle (-PI to PI) between two vectors.
/// This basically stores cosine of the angle, but wrapped so that the range covers both left and right.
/// Can be cheaply compared to another PseudoAngle, or (slightly less cheaply) converted to a regular angle in radians.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct PseudoAngle(f64);

impl PseudoAngle {
    pub const ZERO: PseudoAngle = PseudoAngle(0.0);
    pub const PI: PseudoAngle = PseudoAngle(2.0);
    pub const FRAC_PI_2: PseudoAngle = PseudoAngle(1.0);
    pub const FRAC_PI_4: PseudoAngle = PseudoAngle(1.0 - 1.0 / SQRT_2);

    /// Creates a new pseudo angle representing the given angle in radians
    pub fn from_radians(angle: f64) -> Self {
        PseudoAngle((1.0 - angle.cos()).copysign(angle))
    }

    /// Creates a new pseudo angle representing angle between vectors a and b.
    /// To avoid square roots, vector lengths are passed as arguments.
    pub fn with_vectors_and_lengths(
        a: Vector2<f64>,
        length_a: f64,
        b: Vector2<f64>,
        length_b: f64,
    ) -> Self {
        debug_assert_gt!(length_a, 0.0);
        debug_assert_gt!(length_b, 0.0);

        let dot = a.dot(&b) / (length_a * length_b);
        let side = a.perp(&b);
        PseudoAngle((1.0 - dot).copysign(side))
    }

    /// Creates a new pseudo angle representing angle between line segments AB and BC.
    /// To avoid square roots, lengths are passed as arguments.
    pub fn with_points_and_lengths(
        a: &Point2<f64>,
        b: &Point2<f64>,
        c: &Point2<f64>,
        length_ab: f64,
        length_bc: f64,
    ) -> Self {
        Self::with_vectors_and_lengths(b - a, length_ab, c - b, length_bc)
    }

    /// Creates a new pseudo angle representing angle between line segments AB and BC.
    /// This will be slower than `with_points_and_lengths`
    pub fn with_points(a: &Point2<f64>, b: &Point2<f64>, c: &Point2<f64>) -> Self {
        let ab = b - a;
        let bc = c - b;
        Self::with_vectors_and_lengths(ab, ab.norm(), bc, bc.norm())
    }

    /// Converts the pseudo angle back to radians.
    pub fn to_radians(&self) -> f64 {
        (1.0 - self.0.abs())
            .clamp(-1.0, 1.0)
            .acos()
            .copysign(self.0)
    }

    pub fn to_degrees(&self) -> f64 {
        self.to_radians().to_degrees()
    }

    pub fn to_raw(&self) -> f64 {
        self.0
    }

    pub fn from_raw(raw: f64) -> Self {
        PseudoAngle(raw)
    }

    pub fn abs(&self) -> Self {
        PseudoAngle(self.0.abs())
    }
}

impl Neg for PseudoAngle {
    type Output = PseudoAngle;

    fn neg(self) -> Self::Output {
        PseudoAngle(-self.0)
    }
}

/// Allowed range of an angle.
/// Both values are inclusive.
#[derive(Clone, Debug)]
pub struct AngleRange {
    pub min: PseudoAngle,
    pub max: PseudoAngle,
}

impl Default for AngleRange {
    fn default() -> Self {
        AngleRange {
            min: -PseudoAngle::from_raw(-f64::INFINITY),
            max: PseudoAngle::from_raw(f64::INFINITY),
        }
    }
}

impl AngleRange {
    pub fn contains(&self, value: PseudoAngle) -> bool {
        value >= self.min && value <= self.max
    }

    pub fn from_radians(min: f64, max: f64) -> Self {
        assert_le!(min, max);
        assert_ge!(min, -PI);
        assert_le!(max, PI);

        AngleRange {
            min: PseudoAngle::from_radians(min),
            max: PseudoAngle::from_radians(max),
        }
    }

    pub fn from_degrees(min: f64, max: f64) -> Self {
        Self::from_radians(min.to_radians(), max.to_radians())
    }

    pub fn half_range(positive: bool) -> Self {
        if positive {
            AngleRange {
                min: PseudoAngle::ZERO,
                max: PseudoAngle::PI,
            }
        } else {
            AngleRange {
                min: -PseudoAngle::PI,
                max: PseudoAngle::ZERO,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, relative_eq};
    use proptest::prelude::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    use test_case::test_case;
    use test_strategy::proptest;

    #[proptest]
    fn roundtrip(#[strategy(-PI..PI)] angle: f64) {
        let pseudo = PseudoAngle::from_radians(angle);
        let roundtripped = pseudo.to_radians();

        prop_assert!(relative_eq!(roundtripped, angle, max_relative = 1e-6));
    }

    #[test_case(Vector2::x(), Vector2::x(), 0.0; "zero")]
    #[test_case(Vector2::x(), Vector2::y(), FRAC_PI_2; "pos_45")]
    #[test_case(Vector2::y(), Vector2::x(), -FRAC_PI_2; "neg_90")]
    #[test_case(Vector2::x(), -Vector2::x(), PI; "pos_180")]
    fn with_vector_examples(a: Vector2<f64>, b: Vector2<f64>, expected: f64) {
        let pa = PseudoAngle::with_vectors_and_lengths(a, a.norm(), b, b.norm());
        let angle = pa.to_radians();
        assert_relative_eq!(angle, expected, max_relative = 1e-6);
    }

    #[test_case(Point2::new(1.0, 0.0), Point2::new(2.0, 0.0), 0.0; "zero")]
    #[test_case(Point2::new(1.0, 0.0), Point2::new(2.0, 1.0), FRAC_PI_4; "pos_45")]
    #[test_case(Point2::new(1.0, 0.0), Point2::new(1.0, -1.0), -FRAC_PI_2; "neg_90")]
    fn with_points_examples(b: Point2<f64>, c: Point2<f64>, expected: f64) {
        let pa = PseudoAngle::with_points(&Point2::origin(), &b, &c);
        let angle = pa.to_radians();
        assert_relative_eq!(angle, expected, max_relative = 1e-6);
    }

    #[proptest]
    fn ordering(#[strategy(-PI..PI)] angle1: f64, #[strategy(-PI..PI)] angle2: f64) {
        let (angle1, angle2) = (angle1.min(angle2), angle1.max(angle2));

        let pa1 = PseudoAngle::from_radians(angle1);
        let pa2 = PseudoAngle::from_radians(angle2);

        prop_assert!(pa1 <= pa2);
    }

    #[test]
    fn pseudo_angle_edge_cases() {
        let pa_pos_pi = PseudoAngle::from_radians(PI);
        let pa_neg_pi = PseudoAngle::from_radians(-PI);
        assert_ne!(pa_pos_pi, pa_neg_pi); // Distinct

        let angle1 = pa_pos_pi.to_radians();
        let angle2 = pa_neg_pi.to_radians();
        assert!(relative_eq!(
            angle1.abs(),
            angle2.abs(),
            max_relative = 1e-10
        ));
    }

    #[proptest]
    fn symmetry(#[strategy(-PI..PI)] angle: f64) {
        let pa = PseudoAngle::from_radians(angle);
        let neg = -pa;
        let expected = PseudoAngle::from_radians(-angle);

        prop_assert!(relative_eq!(neg.0, expected.0, max_relative = 1e-10));
    }

    #[test_case(PseudoAngle::ZERO, 0.0; "ZERO")]
    #[test_case(PseudoAngle::PI, PI; "PI")]
    #[test_case(PseudoAngle::FRAC_PI_2, FRAC_PI_2; "FRAC_PI_2")]
    #[test_case(PseudoAngle::FRAC_PI_4, FRAC_PI_4; "FRAC_PI_4")]
    fn consts(pa: PseudoAngle, expected: f64) {
        assert!(
            relative_eq!(pa.to_radians(), expected, max_relative = 1e-6),
            "{} == {}",
            pa.to_radians(),
            expected
        );
    }
}

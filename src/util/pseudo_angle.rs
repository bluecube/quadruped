use std::{f64::consts::PI, ops::Neg};

use more_asserts::{assert_ge, assert_le};
use nalgebra::{Point2, Vector2, one, zero};
use num::FromPrimitive;
use simba::simd::{SimdBool, SimdRealField, SimdValue};

/// Represents signed angle (-PI to PI) between two vectors.
/// This basically stores cosine of the angle, but wrapped so that the range covers both left and right.
/// Can be cheaply compared to another PseudoAngle, or (slightly less cheaply) converted to a regular angle in radians.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct PseudoAngle<T>(T);

impl<T: SimdRealField> PseudoAngle<T> {
    pub fn zero() -> PseudoAngle<T> {
        PseudoAngle(zero())
    }

    pub fn pi() -> PseudoAngle<T> {
        PseudoAngle(one::<T>() + one())
    }

    pub fn frac_pi_2() -> PseudoAngle<T> {
        PseudoAngle(one())
    }

    pub fn frac_pi_4() -> PseudoAngle<T> {
        let one = one::<T>();
        let two = one.clone() + one.clone();
        PseudoAngle(one.clone() - one / two.simd_sqrt())
    }
}

impl<T: SimdRealField> PseudoAngle<T> {
    /// Creates a new pseudo angle representing the given angle in radians
    pub fn from_radians(angle: T) -> Self {
        PseudoAngle((one::<T>() - angle.clone().simd_cos()).simd_copysign(angle))
    }

    /// Creates a new pseudo angle representing angle between vectors a and b.
    /// To avoid square roots, vector lengths are passed as arguments.
    pub fn with_vectors_and_lengths(
        a: Vector2<T>,
        length_a: T,
        b: Vector2<T>,
        length_b: T,
    ) -> Self {
        debug_assert!(length_a.clone().is_simd_positive().all());
        debug_assert!(length_b.clone().is_simd_positive().all());

        let dot = a.dot(&b) / (length_a * length_b);
        let side = a.perp(&b);
        PseudoAngle((one::<T>() - dot).simd_copysign(side))
    }

    /// Creates a new pseudo angle representing angle between line segments AB and BC.
    /// To avoid square roots, lengths are passed as arguments.
    pub fn with_points_and_lengths(
        a: &Point2<T>,
        b: &Point2<T>,
        c: &Point2<T>,
        length_ab: T,
        length_bc: T,
    ) -> Self {
        Self::with_vectors_and_lengths(b - a, length_ab, c - b, length_bc)
    }

    /// Creates a new pseudo angle representing angle between line segments AB and BC.
    /// This will be slower than `with_points_and_lengths`
    pub fn with_points(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>) -> Self {
        let ab = b - a;
        let bc = c - b;
        let ab_norm = ab.norm();
        let bc_norm = bc.norm();
        Self::with_vectors_and_lengths(ab, ab_norm, bc, bc_norm)
    }

    /// Converts the pseudo angle back to radians.
    pub fn to_radians(&self) -> T {
        (one::<T>() - self.0.clone().simd_abs())
            .simd_clamp(-one::<T>(), one())
            .simd_acos()
            .simd_copysign(self.0.clone())
    }

    pub fn to_raw(&self) -> T {
        self.0.clone()
    }

    pub fn from_raw(raw: T) -> Self {
        PseudoAngle(raw)
    }

    pub fn abs(&self) -> Self {
        PseudoAngle(self.0.clone().simd_abs())
    }
}

impl<T: SimdRealField + FromPrimitive> PseudoAngle<T> {
    pub fn to_degrees(&self) -> T {
        self.to_radians() * T::from_f32(180.0).unwrap() / T::simd_pi()
    }
}

impl<T: SimdValue> SimdValue for PseudoAngle<T> {
    const LANES: usize = T::LANES;
    type Element = PseudoAngle<T::Element>;
    type SimdBool = T::SimdBool;

    fn splat(val: Self::Element) -> Self {
        PseudoAngle(T::splat(val.0))
    }

    fn extract(&self, i: usize) -> Self::Element {
        PseudoAngle(self.0.extract(i))
    }

    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        PseudoAngle(unsafe { self.0.extract_unchecked(i) })
    }

    fn replace(&mut self, i: usize, val: Self::Element) {
        self.0.replace(i, val.0);
    }

    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        unsafe { self.0.replace_unchecked(i, val.0) };
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        PseudoAngle(self.0.select(cond, other.0))
    }
}

impl<T: Neg> Neg for PseudoAngle<T> {
    type Output = PseudoAngle<T::Output>;

    fn neg(self) -> Self::Output {
        PseudoAngle(self.0.neg())
    }
}

/// Allowed range of an angle.
/// Both values are inclusive.
#[derive(Clone, Debug)]
pub struct AngleRange<T: SimdRealField> {
    pub min: PseudoAngle<T>,
    pub max: PseudoAngle<T>,
}

// impl<T: SimdRealField> Default for AngleRange<T> {
//     fn default() -> Self {
//         AngleRange {
//             min: -PseudoAngle::from_raw(-f64::INFINITY),
//             max: PseudoAngle::from_raw(f64::INFINITY),
//         }
//     }
// }

impl<T: SimdRealField> AngleRange<T> {
    pub fn contains(&self, value: PseudoAngle<T>) -> T::SimdBool {
        value.0.clone().simd_ge(self.min.0.clone()) & value.0.clone().simd_le(self.max.0.clone())
    }

    pub fn half_range(positive: bool) -> Self {
        if positive {
            AngleRange {
                min: PseudoAngle::zero(),
                max: PseudoAngle::pi(),
            }
        } else {
            AngleRange {
                min: -PseudoAngle::<T>::pi(),
                max: PseudoAngle::zero(),
            }
        }
    }
}

impl<T: SimdValue + SimdRealField> AngleRange<T>
where
    T::Element: SimdRealField + Copy,
{
    pub fn splat(value: &AngleRange<T::Element>) -> AngleRange<T> {
        AngleRange {
            min: PseudoAngle::splat(value.min),
            max: PseudoAngle::splat(value.max),
        }
    }
}

impl AngleRange<f64> {
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

    #[test_case(PseudoAngle::zero(), 0.0; "ZERO")]
    #[test_case(PseudoAngle::pi(), PI; "PI")]
    #[test_case(PseudoAngle::frac_pi_2(), FRAC_PI_2; "FRAC_PI_2")]
    #[test_case(PseudoAngle::frac_pi_4(), FRAC_PI_4; "FRAC_PI_4")]
    fn consts(pa: PseudoAngle<f64>, expected: f64) {
        assert!(
            relative_eq!(pa.to_radians(), expected, max_relative = 1e-6),
            "{} == {}",
            pa.to_radians(),
            expected
        );
    }
}

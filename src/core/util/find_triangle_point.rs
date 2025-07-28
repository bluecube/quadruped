use nalgebra::{Point2, Scalar, one, zero};
use simba::simd::{SimdBool as _, SimdRealField};

use super::{
    planar::perpendicular,
    pseudo_angle::{AngleRange, PseudoAngle},
    simd::MaskedValue,
};

/// Given two points `a` and `b` and two trigangle side lengths
/// `length_a` (= |AC|) and `length_b` (= |CB|), calculates point C, making sure that angle
/// between AC and CB is within `c_angle_range`.
/// If no such triangle exists, returns None.
/// If the angle range is not enough to disambiguate, then the point which results in positive C angle will be returned.
pub fn find_triangle_point<T: SimdRealField + Copy>(
    a: &Point2<T>,
    length_a: T,
    b: &Point2<T>,
    length_b: T,
    c_angle_range: &AngleRange<T>,
) -> MaskedValue<Point2<T>, T::SimdBool>
where
    T::Element: Scalar,
{
    assert!(length_a.is_simd_positive().all(), "length_a = {length_a:?}");
    assert!(length_b.is_simd_positive().all(), "length_b = {length_b:?}");

    // D = A + t * (B - A) ; Point D is the foot of the height DC
    // C = D + s * perpendicular(B - A).
    // Pythagorean theorem for two right triangles ADC and BDC gives us two equations
    // to solve for t and s.

    let ab = b - a;
    let ab_norm_squared = ab.norm_squared();

    let p = (length_a * length_a) / ab_norm_squared;
    let q = (length_b * length_b) / ab_norm_squared;

    // There's a numerical instability hiding here if ab.norm() is too small.
    // For now let's not deal with this as on the real robot the condition should be
    // filtered out by joint angle limits

    let t = (p - q + one::<T>()) / (one::<T>() + one::<T>());
    let s_squared = p - t * t;
    let valid_mask = s_squared.simd_ge(-T::simd_default_epsilon().simd_sqrt());

    let s = s_squared.simd_max(zero()).simd_sqrt();
    let ad = ab * t;
    let d = a + ad;

    // The following are vectors for the positive angle solution
    let dc_positive = perpendicular(&ab) * s;
    let ac_positive = ad + dc_positive;
    let cb_positive = ab - ac_positive;

    let angle_positive =
        PseudoAngle::with_vectors_and_lengths(ac_positive, length_a, cb_positive, length_b);
    let positive_valid = c_angle_range.contains(angle_positive);
    let negative_valid = c_angle_range.contains(-angle_positive);

    let c = d + dc_positive.map(|x| x.select(positive_valid, -x));
    let valid_mask = valid_mask & (positive_valid | negative_valid);

    MaskedValue {
        value: c,
        mask: valid_mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::proptest_util::*;
    use approx::{abs_diff_eq, assert_abs_diff_eq, relative_eq};
    use nalgebra::distance;
    use proptest::prelude::*;
    use test_case::test_case;
    use test_strategy::proptest;

    #[test_case(Point2::new(0.0, 0.0), 5.0, Point2::new(3.0, 0.0), 4.0, true, Some(Point2::new(3.0, -4.0)); "pythagorean_triangle - left")]
    #[test_case(Point2::new(0.0, 0.0), 5.0, Point2::new(3.0, 0.0), 4.0, false, Some(Point2::new(3.0, 4.0)); "pythagorean_triangle - right")]
    #[test_case(Point2::new(0.0, 0.0), 1.0, Point2::new(3.0, 0.0), 1.0, true, None; "too_short")]
    #[test_case(Point2::new(0.0, 0.0), 10.0, Point2::new(3.0, 0.0), 1.0, true, None; "too_long")]
    fn examples(
        point_a: Point2<f64>,
        length_a: f64,
        point_b: Point2<f64>,
        length_b: f64,
        left: bool,
        expected: Option<Point2<f64>>,
    ) {
        let result = find_triangle_point(
            &point_a,
            length_a,
            &point_b,
            length_b,
            &AngleRange::half_range(left),
        )
        .into_option();

        match (result, expected) {
            (Some(result), Some(expected)) => {
                assert_abs_diff_eq!(&result, &expected, epsilon = 1e-6);
            }
            _ => assert_eq!(result, expected),
        }
    }

    /// Tests that a non-degenerate triangle can be solved by find_triangle_point.
    #[proptest]
    fn non_degenerate(
        #[strategy(point2_strategy(-1000.0, 1000.0))] point_a: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))] point_b: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))] point_c: Point2<f64>,
    ) {
        prop_assume!(distance(&point_a, &point_b) > 1e-3);
        let length_a = distance(&point_a, &point_c);
        let length_b = distance(&point_b, &point_c);
        let angle =
            PseudoAngle::with_points_and_lengths(&point_a, &point_c, &point_b, length_a, length_b);

        // Assume that the angle is not 0 or 180 degrees
        prop_assume!((angle.abs().to_degrees() - 90.0).abs() < 85.0);

        let result = find_triangle_point(
            &point_a,
            length_a,
            &point_b,
            length_b,
            &AngleRange::half_range(angle > PseudoAngle::zero()),
        )
        .into_option();

        prop_assert!(result.is_some());
        let result = result.unwrap();
        prop_assert!(relative_eq!(
            distance(&point_a, &result),
            length_a,
            max_relative = 1e-6
        ));
        prop_assert!(relative_eq!(
            distance(&point_b, &result),
            length_b,
            max_relative = 1e-6
        ));
        prop_assert!(abs_diff_eq!(&point_c, &result, epsilon = 1e-6));
    }

    /// Tests that find_triangle_point works for output points that lie exactly on the AB line.
    /// For this test we don't limit the angles, as the +-PI situation is difficult to handle because
    /// of floating point inaccuracies.
    #[proptest]
    fn on_the_line(
        #[strategy(point2_strategy(-100.0, 100.0))] point_a: Point2<f64>,
        #[strategy(point2_strategy(-100.0, 100.0))] point_b: Point2<f64>,
        #[strategy(-10.5..9.5)] point_c_relative_position: f64,
    ) {
        prop_assume!(distance(&point_a, &point_b) > 1e-3);
        prop_assume!(point_c_relative_position.abs() > 1e-3);

        let base = distance(&point_a, &point_b);

        let length_a = (base * point_c_relative_position).abs();
        let length_b = (base * (point_c_relative_position - 1.0)).abs();
        let expected = point_a + (point_b - point_a) * point_c_relative_position;

        let result = find_triangle_point(
            &point_a,
            length_a,
            &point_b,
            length_b,
            &AngleRange {
                min: PseudoAngle::from_raw(f64::MIN),
                max: PseudoAngle::from_raw(f64::MAX),
            },
        )
        .into_option();
        prop_assert!(result.is_some());
        let result = result.unwrap();
        prop_assert!(abs_diff_eq!(&result, &expected, epsilon = base * 1e-6));
    }

    #[proptest]
    fn too_short(
        #[strategy(point2_strategy(-1000.0, 1000.0))] point_a: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))] point_b: Point2<f64>,
        #[strategy(0.0..1.0)] ab_fraction: f64,
        #[strategy(0.0..1.0)] a_fraction: f64,
        left: bool,
    ) {
        let length_ab = distance(&point_a, &point_b) * ab_fraction;
        let length_a = length_ab * a_fraction;
        let length_b = length_ab - length_a;

        assert_eq!(
            find_triangle_point(
                &point_a,
                length_a,
                &point_b,
                length_b,
                &AngleRange::half_range(left),
            )
            .into_option(),
            None
        );
    }

    #[proptest]
    fn too_long(
        #[strategy(point2_strategy(-1000.0, 1000.0))] point_a: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))] point_b: Point2<f64>,
        #[strategy(1.0..100.0)] a_fraction: f64,
        #[strategy(0.0..1.0)] b_fraction: f64,
        left: bool,
    ) {
        let d = distance(&point_a, &point_b);
        let length_a = d * a_fraction;
        let length_b = (length_a - d) * b_fraction;

        assert_eq!(
            find_triangle_point(
                &point_a,
                length_a,
                &point_b,
                length_b,
                &AngleRange::half_range(left),
            )
            .into_option(),
            None
        );
    }
}

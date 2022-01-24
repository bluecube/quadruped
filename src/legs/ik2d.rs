
use nalgebra::{Point2, Vector2};
//use dimensioned::si;


/// Given two points `point_a` and `point_b` and two trigangle side lengths
/// `length_a` (= |AC|) and `length_b` (= |BC|), calculates (one possible) point C.
/// If no such triangle exists, returns None.
fn find_triangle_point(point_a: Point2<f64>, length_a: f64, point_b: Point2<f64>, length_b: f64) -> Option<Point2<f64>> {
    // TODO: There are two solutions mirrored around AB, for now we just take one to the left from AB
    // TODO: Maybe also return an angle between X axis and AC.

    assert!(length_a >= 0.0);
    assert!(length_b >= 0.0);

    // D = A + t * (B - A)
    // C = D + s * perpendicular(B - A).
    // Pythagorean theorem for two right triangles ADC and BDC gives us two equations
    // to solve for t and s.

    let ba = point_b - point_a;
    let ba_norm_squared = ba.norm_squared();

    let t = (length_a * length_a - length_b * length_b) / (2.0 * ba_norm_squared) + 0.5;
    let s_squared = (length_a * length_a) / ba_norm_squared - t * t;
    if s_squared < 0.0 {
        return None
    }
    let s = s_squared.sqrt();

    let ba_perpendicular = Vector2::new(ba.y, -ba.x);

    let point_c = point_a + ba * t + ba_perpendicular * s;

    Some(point_c)
}


#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use test_strategy::proptest;
    use nalgebra::proptest::vector;
    use nalgebra::{distance, Const};
    use approx::assert_relative_eq;

    fn point2_strategy() -> impl Strategy<Value=Point2<f64>> {
        //vector(proptest::num::f64::NORMAL, Const::<2>).prop_map(|x| Point2::from(x))
        vector(-1000.0..1000.0, Const::<2>).prop_map(|x| Point2::from(x))
    }

    #[proptest]
    fn find_triangle_point_distances(
        #[strategy(point2_strategy())]
        point_a: Point2<f64>,
        //#[strategy(proptest::num::f64::POSITIVE)]
        #[strategy(0.0..1000.0)]
        length_a: f64,
        #[strategy(point2_strategy())]
        point_b: Point2<f64>,
        //#[strategy(proptest::num::f64::POSITIVE)]
        #[strategy(0.0..1000.0)]
        length_b: f64
    ) {
        match find_triangle_point(point_a, length_a, point_b, length_b) {
            None => {
                let length_c = distance(&point_a, &point_b);

                assert!(
                    (length_a + length_b < length_c) ||
                    (length_a + length_c < length_b) ||
                    (length_b + length_c < length_a),
                    "length_a = {}, length_b = {}, length_c = {}",
                    length_a, length_b, length_c
                );
            }
            Some(point_c) => {
                assert_relative_eq!(distance(&point_a, &point_c), length_a, max_relative=1e-6);
                assert_relative_eq!(distance(&point_b, &point_c), length_b, max_relative=1e-6);
            }
        }
    }
}

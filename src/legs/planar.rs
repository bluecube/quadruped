use nalgebra::{Point2, Vector2, distance};
//use dimensioned::si;

/// Definition of leg geometry, see leg-schematic.svg for description of the values
#[derive(Clone, Debug)]
pub struct Params {
    pub point_a: Point2<f64>,
    pub point_b: Point2<f64>,
    pub len_ac: f64,
    pub len_bd: f64,
    pub len_ce: f64,
    pub len_df: f64,

    /// vector DE = FD * fd_to_de_scale.x + perpendicular(FD) * fd_to_de_scale.y
    pub fd_to_de_scale: Vector2<f64>,
}

impl Params {
    fn is_valid(&self, min_distance: f64) -> bool {
        let len_ab = distance(&self.point_a, &self.point_b);
        let len_de = self.fd_to_de_scale.norm() * self.len_df;
        let has_nonzero_lengths =
            len_ab >= min_distance &&
            self.len_ac >= min_distance &&
            self.len_bd >= min_distance &&
            self.len_ce >= min_distance &&
            self.len_df >= min_distance &&
            len_de >= min_distance;
        let a_b_close_enough = len_ab < self.len_ac + self.len_ce + len_de + self.len_bd;

        has_nonzero_lengths & a_b_close_enough
    }
}

/// Positions of points in the 2D leg plane view
#[derive(Clone, Debug)]
pub struct KinematicState {
    // Points A and B are repeated here from parameters to allow using an example
    // kinematic state to construct KinematicParams
    pub point_c: Point2<f64>,
    pub point_d: Point2<f64>,
    pub point_e: Point2<f64>,
    pub point_f: Point2<f64>
}

/// Create a perpendicular 2D vector to a given vector.
/// Magnitude is identical to the argument.
fn perpendicular(v: Vector2<f64>) -> Vector2<f64> {
    Vector2::new(v.y, -v.x)
}

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

    let point_c = point_a + ba * t + perpendicular(ba) * s;

    Some(point_c)
}

fn calculate_state(point_f: Point2<f64>, params: &Params) -> Option<KinematicState> {
    assert!(params.is_valid(1e-6));


    println!("point_f: {point_f:?}");
    let point_d = find_triangle_point(params.point_b, params.len_bd, point_f, params.len_df)?;
    println!("point_d: {point_d:?}");

    let fd_direction = point_d - point_f;
    let point_e = point_d +
        fd_direction * params.fd_to_de_scale.x +
        perpendicular(fd_direction) * params.fd_to_de_scale.y;
    println!("point_e: {point_e:?}");

    let point_c = find_triangle_point(params.point_a, params.len_ac, point_e, params.len_ce)?;
    println!("point_c: {point_c:?}");

    Some(KinematicState {
        point_c,
        point_d,
        point_e,
        point_f
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use test_strategy::proptest;
    use nalgebra::proptest::vector;
    use nalgebra::Const;
    use approx::assert_relative_eq;

    fn angle_and_length_to_vector(angle: f64, length: f64) -> Vector2<f64> {
        Vector2::new(angle.cos(), angle.sin()) * length
    }

    fn point2_strategy(min_value: f64, max_value: f64) -> impl Strategy<Value=Point2<f64>> {
        //vector(proptest::num::f64::NORMAL, Const::<2>).prop_map(|x| Point2::from(x))
        vector(min_value..max_value, Const::<2>).prop_map(|x| Point2::from(x))
    }

    fn params_strategy(min_value: f64, max_value: f64) -> impl Strategy<Value=Params> {
        (
            point2_strategy(-max_value, max_value),
            0.0..1.0,
            0.0..std::f64::consts::PI,
            min_value..max_value,
            min_value..max_value,
            min_value..max_value,
            min_value..max_value,
            min_value..max_value,
            0.0..std::f64::consts::PI,
        ).prop_filter_map(
            "Invalid params",
            move |(
                point_a,
                point_b_relative_distance, point_b_angle,
                len_ac, len_bd, len_ce, len_df,
                len_de,
                de_relative_angle,
            )| {
                let point_b_max_distance = (len_ac + len_ce + len_bd + len_de) / 2.0;
                let point_b_distance = min_value + (point_b_max_distance - min_value) * point_b_relative_distance;
                let point_b = point_a + angle_and_length_to_vector(point_b_angle, point_b_distance);
                let fd_to_de_scale = angle_and_length_to_vector(de_relative_angle, len_de / len_df);
                let ret = Params {
                    point_a,
                    point_b,
                    len_ac,
                    len_bd,
                    len_ce,
                    len_df,

                    fd_to_de_scale
                };

                if ret.is_valid(min_value) {
                    Some(ret)
                } else {
                    None
                }
            }
        )
    }

    fn simple_test_params() -> Params {
        Params {
            point_a: Point2::new(0.0, 0.0),
            point_b: Point2::new(-1.0, -1.0),
            len_ac: 1.0,
            len_bd: 1.0,
            len_ce: 2.2,
            len_df: 1.0,
            fd_to_de_scale: Vector2::new(1.0, 0.2)
        }
    }

    #[proptest]
    fn perpendicular_has_zero_dot_product(
        #[strategy(vector(-1000.0..1000.0, Const::<2>))]
        v: Vector2<f64>
    ) {
        assert_eq!(v.dot(&perpendicular(v)), 0.0);
    }

    #[proptest]
    fn find_triangle_point_distances(
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_a: Point2<f64>,
        //#[strategy(proptest::num::f64::POSITIVE)]
        #[strategy(0.0..1000.0)]
        length_a: f64,
        #[strategy(point2_strategy(-1000.0, 1000.0))]
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

    //#[test]
    //fn calculate_state_preserves_segment_lengths_x() {
    //    calculate_state_preserves_segment_lengths(Point2::new(174.2208523518127, -55.847698359675405));
    //}

    #[proptest]
    fn calculate_state_preserves_segment_lengths_known_good_params(
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_f: Point2<f64>,
    ) {
        let params = simple_test_params();
        println!("point_f: {point_f}, params: {params:?}");
        match calculate_state(point_f, &params) {
            None => {
                println!("No solution");
                prop_assume!(false); // We're not concerned with non-solutions now
            }
            Some(state) => {
                assert_relative_eq!(distance(&params.point_a, &state.point_c), params.len_ac, max_relative=1e-6);
                assert_relative_eq!(distance(&params.point_b, &state.point_d), params.len_bd, max_relative=1e-6);
                assert_relative_eq!(distance(&state.point_c, &state.point_e), params.len_ce, max_relative=1e-6);
                assert_relative_eq!(distance(&state.point_d, &state.point_f), params.len_df, max_relative=1e-6);
            }
        }
    }

    #[proptest]
    fn calculate_state_preserves_segment_lengths(
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_f: Point2<f64>,
        #[strategy(params_strategy(1e-6, 1000.0))]
        params: Params
    ) {
        //let params = simple_test_params();
        println!("point_f: {point_f}, params: {params:?}");
        match calculate_state(point_f, &params) {
            None => {
                println!("No solution");
                //prop_assume!(false); // We're not concerned with non-solutions now
            }
            Some(state) => {
                assert_relative_eq!(distance(&params.point_a, &state.point_c), params.len_ac, max_relative=1e-6);
                assert_relative_eq!(distance(&params.point_b, &state.point_d), params.len_bd, max_relative=1e-6);
                assert_relative_eq!(distance(&state.point_c, &state.point_e), params.len_ce, max_relative=1e-6);
                assert_relative_eq!(distance(&state.point_d, &state.point_f), params.len_df, max_relative=1e-6);
            }
        }
    }

}

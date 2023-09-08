use nalgebra::{Point2, Vector2, distance};

/// Definition of leg geometry, see leg-schematic.svg for description of the values
/// Note that some of the parameters are redundant for simpler calculation
#[derive(Clone, Debug)]
pub struct Params {
    pub point_a: Point2<f64>,
    pub point_b: Point2<f64>,
    pub len_ac: f64,
    pub len_bd: f64,
    pub len_ce: f64,
    pub len_df: f64,
    pub len_de: f64,

    pub f_from_ed: ThirdPointRelativePosition,
    pub e_from_fd: ThirdPointRelativePosition,
}

/// Positions of points in the 2D leg plane view
/// This is a necessary half way form when doing both forward and inverse kinematics.
#[derive(Clone, Debug)]
pub struct KinematicState {
    pub point_a: Point2<f64>,
    pub point_b: Point2<f64>,
    pub point_c: Point2<f64>,
    pub point_d: Point2<f64>,
    pub point_e: Point2<f64>,
    pub point_f: Point2<f64>
}

#[derive(Clone, Debug)]
pub struct JointAngles {
    pub alpha: f64,
    pub beta: f64
}

impl KinematicState {
    /// Construct new KinematicState from joint angles
    /// Not all combinations of joint angles might be valid, returns None if no
    /// solution is possible.
    pub fn with_joint_angles(joint_angles: &JointAngles, params: &Params) -> Option<KinematicState> {
        let point_c = params.point_a + angle_and_length_to_vector(joint_angles.alpha, params.len_ac);
        let point_d = params.point_b + angle_and_length_to_vector(joint_angles.beta, params.len_bd);
        let point_e = find_triangle_point(&point_c, params.len_ce, &point_d, params.len_de)?;
        let point_f = params.f_from_ed.apply(&point_e, &point_d);

        Some(KinematicState { point_a: params.point_a, point_b: params.point_b, point_c, point_d, point_e, point_f })
    }

    /// Construct new KinematicState from a foot point and parameters.
    /// Returns None if there is no solution possible.
    pub fn with_foot_position(foot_position: Point2<f64>, params: &Params) -> Option<KinematicState> {
        let point_f = foot_position;
        let point_d = find_triangle_point(&params.point_b, params.len_bd, &point_f, params.len_df)?;
        let point_e = params.e_from_fd.apply(&point_f, &point_d);
        let point_c = find_triangle_point(&params.point_a, params.len_ac, &point_e, params.len_ce)?;

        Some(KinematicState { point_a: params.point_a, point_b: params.point_b, point_c, point_d, point_e, point_f })
    }

    #[cfg(test)]
    /// Extract parameters that correspond to this kinematic state.
    pub fn get_params(&self) -> Params {
        Params {
            point_a: self.point_a,
            point_b: self.point_b,
            len_ac: distance(&self.point_a, &self.point_c),
            len_bd: distance(&self.point_b, &self.point_d),
            len_ce: distance(&self.point_c, &self.point_e),
            len_df: distance(&self.point_d, &self.point_f),
            len_de: distance(&self.point_d, &self.point_e),
            e_from_fd: ThirdPointRelativePosition::new(&self.point_f, &self.point_d, &self.point_e),
            f_from_ed: ThirdPointRelativePosition::new(&self.point_e, &self.point_d, &self.point_f),
        }
    }

    pub fn get_foot_position(&self) -> &Point2<f64> {
        &self.point_f
    }

    pub fn get_joint_angles(&self) -> JointAngles {
        JointAngles { alpha: vector_to_angle(self.point_c - self.point_a), beta: vector_to_angle(self.point_d - self.point_b) }
    }
}

/// Represents a position of a point relative to two other points, forming a simillar triangle.
#[derive(Debug, Copy, Clone)]
pub struct ThirdPointRelativePosition {
    longitudal: f64,
    lateral: f64,
}

impl ThirdPointRelativePosition {
    /// Construct new ThirdPointRelativePosition from an example,
    /// first two points are the pattern, third point is the expected output
    pub fn new(a: &Point2<f64>, b: &Point2<f64>, c: &Point2<f64>) -> Self {
        let ab = b - a;
        let bc = c - b;
        ThirdPointRelativePosition {
            longitudal: ab.dot(&bc) / ab.norm_squared(),
            lateral: perpendicular(&ab).dot(&bc) / ab.norm_squared()
        }
    }

    /// Obtain a position of the third point from the two pattern points
    pub fn apply(&self, a: &Point2<f64>, b: &Point2<f64>) -> Point2<f64> {
        let ab = b - a;
        b + ab * self.longitudal + perpendicular(&ab) * self.lateral
    }
}

/// Create a perpendicular 2D vector to a given vector.
/// Magnitude is identical to the argument.
fn perpendicular(v: &Vector2<f64>) -> Vector2<f64> {
    Vector2::new(v.y, -v.x)
}

fn angle_and_length_to_vector(angle: f64, length: f64) -> Vector2<f64> {
    let (s, c) = angle.sin_cos();
    Vector2::new(c, s) * length
}

fn vector_to_angle(v: Vector2<f64>) -> f64 {
    v.y.atan2(v.x)
}

/// Given two points `point_a` and `point_b` and two trigangle side lengths
/// `length_a` (= |AC|) and `length_b` (= |BC|), calculates point C.
/// This point is always to the left of the AB line.
/// If no such triangle exists, returns None.
fn find_triangle_point(point_a: &Point2<f64>, length_a: f64, point_b: &Point2<f64>, length_b: f64) -> Option<Point2<f64>> {
    // TODO: Maybe also return an angle between X axis and AC.

    assert!(length_a >= 0.0);
    assert!(length_b >= 0.0);

    // D = A + t * (B - A)
    // C = D + s * perpendicular(B - A).
    // Pythagorean theorem for two right triangles ADC and BDC gives us two equations
    // to solve for t and s.

    let ba = point_b - point_a;
    let ba_norm_squared = ba.norm_squared();

    let p = (length_a * length_a) / ba_norm_squared;
    let q = (length_b * length_b) / ba_norm_squared;

    let t = (p - q + 1.0) / 2.0;
    let s_squared = p - t * t;
    if s_squared < -1e-6f64 {
        None
    } else {
        let s = s_squared.max(0.0).sqrt();
        let point_c = point_a + ba * t + perpendicular(&ba) * s;
        Some(point_c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use test_strategy::proptest;
    use test_case::test_case;
    use nalgebra::proptest::vector;
    use nalgebra::{Const, distance_squared};
    use approx::assert_relative_eq;
    use more_asserts::*;

    macro_rules! assert_points_approx_eq {
        ($p1:expr, $p2:expr, $max_dist:expr) => {
            assert!(distance($p1, $p2) <= $max_dist,
            "assert_points_approx_eq!({}, {}, {})

    left  = {:?}
    right = {:?}
    distance(left, right) = {} > {}

", stringify!($p1), stringify!($p2), stringify!($dist), $p1, $p2, distance($p2, $p1), $max_dist)
        }
    }

    /// Strategy for 64bit floating point numbers that minimize to nicely readable integer values
    fn f64_strategy(range: std::ops::Range<f64>) -> impl Strategy<Value=f64> + Clone {
        (0f64..1f64, (range.start.floor() as i64)..(range.end.floor() as i64))
            .prop_filter_map("Value does not fit in the range",
                move |(fractional, integral)| {
                    let v = (integral as f64) + fractional;
                    if range.contains(&v) {
                        Some(v)
                    } else {
                        None
                    }
                })
    }

    fn point2_strategy(min_value: f64, max_value: f64) -> impl Strategy<Value=Point2<f64>> {
        //vector(proptest::num::f64::NORMAL, Const::<2>).prop_map(|x| Point2::from(x))
        vector(f64_strategy(min_value..max_value), Const::<2>).prop_map(|x| Point2::from(x))
    }

    fn kinematic_state_strategy() -> impl Strategy<Value=KinematicState> {
        proptest::collection::vec(point2_strategy(-1000.0, 1000.0), 6)
            .prop_filter_map(
                "Coincident points",
                |points| {
                    for i in 0..points.len() {
                        for j in i+1..points.len() {
                            if distance(&points[i], &points[j]) < 1e-3 {
                                return None;
                            }
                        }
                    }
                    Some(KinematicState {
                        point_a: points[0], point_b: points[1], point_c: points[2], point_d: points[3], point_e: points[4], point_f: points[5]
                    })
                }
            )
    }

    /// Kinematic state taken from the sketch in leg-schematic.svg
    fn known_good_kinematic_state() -> KinematicState {
        KinematicState {
            point_a: Point2::new(0.0, 0.0),
            point_b: Point2::new(-6.0, -7.0),
            point_c: Point2::new(-9.0, 4.0),
            point_d: Point2::new(-16.0, -11.0),
            point_e: Point2::new(-26.0, -5.0),
            point_f: Point2::new(-4.0, -23.0),
        }
    }

    /// Calculate distance between two kinematic states.
    fn kinematic_state_mean_square_deviation(state1: &KinematicState, state2: &KinematicState) -> f64 {
        (
            distance_squared(&state2.point_a, &state1.point_a) +
            distance_squared(&state2.point_b, &state1.point_b) +
            distance_squared(&state2.point_c, &state1.point_c) +
            distance_squared(&state2.point_d, &state1.point_d) +
            distance_squared(&state2.point_e, &state1.point_e) +
            distance_squared(&state2.point_f, &state1.point_f)
        ) / 6.0
    }

    #[proptest]
    fn third_point_relative_position(
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        a1: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        b1: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        c1: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        a2: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        b2: Point2<f64>,
    ) {
        prop_assume!(distance_squared(&a1, &b1) > 1e-3);
        prop_assume!(distance_squared(&a2, &b2) > 1e-3);
        let t = ThirdPointRelativePosition::new(&a1, &b1, &c1);

        let ab1 = distance(&a1, &b1);
        let ac1 = distance(&a1, &c1);
        let bc1 = distance(&b1, &c1);
        let ac1_norm = ac1 / ab1;
        let bc1_norm = bc1 / ab1;

        let c2 = t.apply(&a2, &b2);

        let ac2 = distance(&a2, &c2);
        let ab2 = distance(&a2, &b2);
        let bc2 = distance(&b2, &c2);
        let ac2_norm = ac2 / ab2;
        let bc2_norm = bc2 / ab2;

        assert_relative_eq!(ac2_norm, ac1_norm, max_relative = 1e-3);
        assert_relative_eq!(bc2_norm, bc1_norm, max_relative = 1e-3);
    }

    #[proptest]
    fn perpendicular_has_zero_dot_product(
        #[strategy(vector(-1000.0..1000.0, Const::<2>))]
        v: Vector2<f64>
    ) {
        assert_eq!(v.dot(&perpendicular(&v)), 0.0);
    }

    #[proptest]
    fn vector_to_angle_roundtrip(
        #[strategy(-std::f64::consts::PI..std::f64::consts::PI)]
        angle: f64,
        #[strategy(1.0..1000.0)]
        length: f64,
    ) {
        let v = angle_and_length_to_vector(angle, length);
        let angle2 = vector_to_angle(v);

        assert_relative_eq!(angle2, angle, max_relative=1e-6);
    }

    #[test_case(Point2::new(0.0, 0.0), 5.0, Point2::new(3.0, 0.0), 4.0, Some(Point2::new(3.0, -4.0)); "pythagorean_triangle")]
    #[test_case(Point2::new(0.0, 0.0), 1.0, Point2::new(3.0, 0.0), 1.0, None; "too_short")]
    #[test_case(Point2::new(0.0, 0.0), 10.0, Point2::new(3.0, 0.0), 1.0, None; "too_long")]
    fn find_triangle_point_examples(point_a: Point2<f64>, length_a: f64, point_b: Point2<f64>, length_b: f64, expected: Option<Point2<f64>>) {
        let result = find_triangle_point(&point_a, length_a, &point_b, length_b);

        match (result, expected) {
            (Some(result), Some(expected)) => assert_points_approx_eq!(&result, &expected, 1e-6),
            _ => assert_eq!(result, expected),
        }
    }

    #[proptest]
    fn find_triangle_point_valid(
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_a: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_b: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_c: Point2<f64>,
    ) {
        let length_a = distance(&point_a, &point_c);
        let length_b = distance(&point_b, &point_c);

        let result = find_triangle_point(&point_a, length_a, &point_b, length_b).unwrap();

        assert_relative_eq!(distance(&point_a, &result), length_a, max_relative=1e-6);
        assert_relative_eq!(distance(&point_b, &result), length_b, max_relative=1e-6);

        // assert_points_approx_eq!(point_c, result, 1e-6);
            // Does not work 50% of time because of the "arbitrary" choice of one of the solutions
    }

    /// Verify that find_triangle_point works for output points that lie exactly on the AB line.
    #[proptest]
    fn find_triangle_point_on_the_line(
        #[strategy(point2_strategy(-100.0, 100.0))]
        point_a: Point2<f64>,
        #[strategy(point2_strategy(-100.0, 100.0))]
        point_b: Point2<f64>,
        #[strategy(-10.5..9.5)]
        point_c_relative_position: f64
    ) {
        let base = distance(&point_a, &point_b);

        let length_a = (base * point_c_relative_position).abs();
        let length_b = (base * (point_c_relative_position - 1.0)).abs();
        let expected = point_a + (point_b - point_a) * point_c_relative_position;

        let result = find_triangle_point(&point_a, length_a, &point_b, length_b).unwrap();
        assert_points_approx_eq!(&result, &expected, base * 1e-6);
    }

    #[proptest]
    fn find_triangle_point_too_short(
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_a: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_b: Point2<f64>,
        #[strategy(0.0..1.0)]
        ab_fraction: f64,
        #[strategy(0.0..1.0)]
        a_fraction: f64
    ) {
        let length_ab = distance(&point_a, &point_b) * ab_fraction;
        let length_a = length_ab * a_fraction;
        let length_b = length_ab - length_a;

        assert_eq!(find_triangle_point(&point_a, length_a, &point_b, length_b), None);
    }

    #[proptest]
    fn find_triangle_point_too_long(
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_a: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))]
        point_b: Point2<f64>,
        #[strategy(1.0..100.0)]
        a_fraction: f64,
        #[strategy(0.0..1.0)]
        b_fraction: f64
    ) {
        let d = distance(&point_a, &point_b);
        let length_a = d * a_fraction;
        let length_b = (length_a - d) * b_fraction;

        assert_eq!(find_triangle_point(&point_a, length_a, &point_b, length_b), None);
    }

/*
    #[test]
    fn forward_kinematics_joint_angles_roundtrip_example() {
        let ks = known_good_kinematic_state();
        let params = ks.get_params();
        let joint_angles = ks.get_joint_angles();
        let foot_position = ks.get_foot_position();

        let ks2 = KinematicState::with_joint_angles(&joint_angles, &params).unwrap();

        dbg!(&ks);
        dbg!(&ks2);

        assert_points_approx_eq!(&ks2.get_foot_position(), &foot_position, 1e-6);
        assert_lt!(kinematic_state_mean_square_deviation(&ks, &ks2), 1e-3);
    }

    #[test]
    fn forward_kinematics_foot_position_roundtrip_example() {
        let ks = known_good_kinematic_state();
        let params = ks.get_params();
        let foot_position = ks.get_foot_position();

        dbg!(&ks);
        dbg!(&foot_position);

        let ks2 = KinematicState::with_foot_position(foot_position.clone(), &params).unwrap();

        dbg!(&ks2);

        assert_points_approx_eq!(&ks2.get_foot_position(), &foot_position, 1e-6);
        assert_lt!(kinematic_state_mean_square_deviation(&ks, &ks2), 1e-3);
    }

    #[proptest]
    fn forward_kinematics_joint_angles_roundtrip(
        #[strategy(kinematic_state_strategy())]
        ks: KinematicState,
    ) {
        let params = ks.get_params();
        let joint_angles = ks.get_joint_angles();
        let foot_position = ks.get_foot_position();

        dbg!(&ks);
        dbg!(&joint_angles);

        let ks2 = KinematicState::with_joint_angles(&joint_angles, &params).unwrap();

        dbg!(&ks2);

        assert_points_approx_eq!(&ks2.get_foot_position(), &foot_position, 1e-3);
        assert_lt!(kinematic_state_mean_square_deviation(&ks, &ks2), 1e-3);
    }

    #[proptest]
    fn forward_kinematics_foot_position_roundtrip(
        #[strategy(kinematic_state_strategy())]
        ks: KinematicState,
    ) {
        let params = ks.get_params();
        let joint_angles = ks.get_joint_angles();
        let foot_position = ks.get_foot_position();

        let ks2 = KinematicState::with_foot_position(foot_position.clone(), &params).unwrap();

        assert_points_approx_eq!(&ks2.get_foot_position(), foot_position, 1e-6);
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
        match KinematicState::new_inverse_kinematics(point_f, &params) {
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
*/
}

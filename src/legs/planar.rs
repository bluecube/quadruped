use std::f64::consts::PI;

use display_json::DebugAsJson;
use nalgebra::{Point2, distance};
use serde::{Deserialize, Serialize};

use crate::util::{
    find_triangle_point,
    planar::{angle_and_length_to_vector, perpendicular, vector_to_angle},
    pseudo_angle::{AngleRange, PseudoAngle},
};

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

    // pub a_range: AngleRange,
    // pub b_range: AngleRange,
    pub ace_range: AngleRange,
    pub bdf_range: AngleRange,
    pub ced_range: AngleRange,
}

/// Positions of points in the 2D leg plane view
/// This is a necessary half way form when doing both forward and inverse kinematics.
#[derive(Clone, DebugAsJson, Serialize, Deserialize)]
pub struct KinematicState {
    pub point_a: Point2<f64>,
    pub point_b: Point2<f64>,
    pub point_c: Point2<f64>,
    pub point_d: Point2<f64>,
    pub point_e: Point2<f64>,
    pub point_f: Point2<f64>,
}

#[derive(Clone, Debug)]
pub struct JointAngles {
    pub alpha: f64,
    pub beta: f64,
}

impl Params {
    /// Verifies that kinematic state corresponds to this parameter set.
    /// Testing only, this is going to be slow.
    pub fn verify(&self, ks: &KinematicState) -> Result<(), Vec<String>> {
        let mut problems = Vec::new();

        let mut check_position = |name, point: &Point2<f64>, expected: &Point2<f64>| {
            if distance(point, expected) > 1e-3 {
                problems.push(format!(
                    "Point {name} expected {expected:?}, have {point:?}"
                ));
            }
        };
        check_position("a", &ks.point_a, &self.point_a);
        check_position("b", &ks.point_b, &self.point_b);

        let mut check_length = |name, point1: &Point2<_>, point2: &Point2<_>, expected: f64| {
            let length: f64 = distance(point1, point2);
            if (length - expected).abs() > 1e-3 {
                problems.push(format!("Length {name} expected {expected}, have {length}"));
            }
        };
        check_length("ac", &ks.point_a, &ks.point_c, self.len_ac);
        check_length("bd", &ks.point_b, &ks.point_d, self.len_bd);
        check_length("ce", &ks.point_c, &ks.point_e, self.len_ce);
        check_length("df", &ks.point_d, &ks.point_f, self.len_df);
        check_length("de", &ks.point_d, &ks.point_e, self.len_de);

        let mut check_angle =
            |name, a: &Point2<_>, b: &Point2<_>, c: &Point2<_>, range: &AngleRange| {
                if !range.contains(PseudoAngle::with_points(a, b, c)) {
                    problems.push(format!("Angle range {name} not satisfied"));
                }
            };
        check_angle(
            "ace",
            &ks.point_a,
            &ks.point_c,
            &ks.point_e,
            &self.ace_range,
        );
        check_angle(
            "bdf",
            &ks.point_b,
            &ks.point_d,
            &ks.point_f,
            &self.bdf_range,
        );
        check_angle(
            "ced",
            &ks.point_c,
            &ks.point_e,
            &ks.point_d,
            &self.ced_range,
        );

        if problems.is_empty() {
            Ok(())
        } else {
            Err(problems)
        }
    }

    /// Extract parameters that correspond to this kinematic state.
    /// This is mostly for testing only.
    /// `angle_tolerance` is angle difference in radians from the kinematic state allowed by
    /// the new params (but the angle will always be cropped at  0 and +-180)
    pub fn with_kinematic_state(ks: &KinematicState, angle_tolerance_radians: f64) -> Self {
        let range_from_points = |a, b, c| {
            let angle = PseudoAngle::with_points(a, b, c).to_radians();
            let min = angle - angle_tolerance_radians;
            let max = angle + angle_tolerance_radians;

            let (min, max) = if angle > 0.0 {
                (min.max(0.0), max.min(PI))
            } else {
                (min.max(-PI), max.min(0.0))
            };

            AngleRange {
                min: PseudoAngle::from_radians(min),
                max: PseudoAngle::from_radians(max),
            }
        };

        Params {
            point_a: ks.point_a,
            point_b: ks.point_b,
            len_ac: distance(&ks.point_a, &ks.point_c),
            len_bd: distance(&ks.point_b, &ks.point_d),
            len_ce: distance(&ks.point_c, &ks.point_e),
            len_df: distance(&ks.point_d, &ks.point_f),
            len_de: distance(&ks.point_d, &ks.point_e),
            e_from_fd: ThirdPointRelativePosition::new(&ks.point_f, &ks.point_d, &ks.point_e),
            f_from_ed: ThirdPointRelativePosition::new(&ks.point_e, &ks.point_d, &ks.point_f),
            ace_range: range_from_points(&ks.point_a, &ks.point_c, &ks.point_e),
            bdf_range: range_from_points(&ks.point_b, &ks.point_d, &ks.point_f),
            ced_range: range_from_points(&ks.point_c, &ks.point_e, &ks.point_d),
        }
    }
}

impl KinematicState {
    /// Construct new KinematicState from joint angles
    /// Not all combinations of joint angles might be valid, returns None if no
    /// solution is possible.
    pub fn with_joint_angles(
        joint_angles: &JointAngles,
        params: &Params,
    ) -> Option<KinematicState> {
        // TODO: Check the driving joint angles
        // if !params.a_range.contains(joint_angles.alpha) {
        //     return None;
        // }
        // if !params.b_range.contains(joint_angles.beta) {
        //     return None;
        // }
        let point_c =
            params.point_a + angle_and_length_to_vector(joint_angles.alpha, params.len_ac);
        let point_d = params.point_b + angle_and_length_to_vector(joint_angles.beta, params.len_bd);
        let point_e = find_triangle_point(
            &point_c,
            params.len_ce,
            &point_d,
            params.len_de,
            &params.ced_range,
        )?;

        if !params
            .ace_range
            .contains(PseudoAngle::with_points_and_lengths(
                &params.point_a,
                &point_c,
                &point_e,
                params.len_ac,
                params.len_ce,
            ))
        {
            return None;
        }
        let point_f = params.f_from_ed.apply(&point_e, &point_d);

        if !params
            .bdf_range
            .contains(PseudoAngle::with_points_and_lengths(
                &params.point_b,
                &point_d,
                &point_f,
                params.len_bd,
                params.len_df,
            ))
        {
            return None;
        }

        Some(KinematicState {
            point_a: params.point_a,
            point_b: params.point_b,
            point_c,
            point_d,
            point_e,
            point_f,
        })
    }

    /// Construct new KinematicState from a foot point and parameters.
    /// Returns None if there is no solution possible.
    pub fn with_foot_position(
        foot_position: Point2<f64>,
        params: &Params,
    ) -> Option<KinematicState> {
        let point_f = foot_position;
        let point_d = find_triangle_point(
            &params.point_b,
            params.len_bd,
            &point_f,
            params.len_df,
            &params.bdf_range,
        )?;
        let point_e = params.e_from_fd.apply(&point_f, &point_d);
        let point_c = find_triangle_point(
            &params.point_a,
            params.len_ac,
            &point_e,
            params.len_ce,
            &params.ace_range,
        )?;

        if !params
            .bdf_range
            .contains(PseudoAngle::with_points_and_lengths(
                &params.point_b,
                &point_d,
                &point_f,
                params.len_bd,
                params.len_df,
            ))
        {
            return None;
        }

        // TODO: Check A and B angle ranges?

        Some(KinematicState {
            point_a: params.point_a,
            point_b: params.point_b,
            point_c,
            point_d,
            point_e,
            point_f,
        })
    }

    pub fn get_foot_position(&self) -> Point2<f64> {
        self.point_f
    }

    pub fn get_joint_angles(&self) -> JointAngles {
        JointAngles {
            alpha: vector_to_angle(self.point_c - self.point_a),
            beta: vector_to_angle(self.point_d - self.point_b),
        }
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
            lateral: -ab.perp(&bc) / ab.norm_squared(),
        }
    }

    /// Obtain a position of the third point from the two pattern points
    pub fn apply(&self, a: &Point2<f64>, b: &Point2<f64>) -> Point2<f64> {
        let ab = b - a;
        b + ab * self.longitudal + perpendicular(&ab) * self.lateral
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::proptest_util::point2_strategy;
    use approx::{assert_abs_diff_eq, relative_eq};
    use more_asserts::{assert_gt, assert_lt};
    use nalgebra::distance_squared;
    use proptest::prelude::*;
    use std::{f64::consts::FRAC_PI_4, panic};
    use test_strategy::proptest;

    fn kinematic_state_strategy() -> impl Strategy<Value = KinematicState> {
        proptest::collection::vec(point2_strategy(-1000.0, 1000.0), 6).prop_filter_map(
            "Coincident points",
            |points| {
                for i in 0..points.len() {
                    for j in i + 1..points.len() {
                        if distance(&points[i], &points[j]) < 1e-3 {
                            return None;
                        }
                    }
                }
                Some(KinematicState {
                    point_a: points[0],
                    point_b: points[1],
                    point_c: points[2],
                    point_d: points[3],
                    point_e: points[4],
                    point_f: points[5],
                })
            },
        )
    }

    /// Kinematic state and parameters taken from the sketch in leg-schematic.svg
    fn example_leg() -> (Params, KinematicState) {
        let ks = KinematicState {
            point_a: Point2::new(0.0, 0.0),
            point_b: Point2::new(-6.0, -7.0),
            point_c: Point2::new(-9.0, 4.0),
            point_d: Point2::new(-16.0, -11.0),
            point_e: Point2::new(-26.0, -5.0),
            point_f: Point2::new(-4.0, -23.0),
        };

        let mut params = Params::with_kinematic_state(&ks, 0.0);

        assert_gt!(params.ace_range.min.to_radians(), 0.0);
        params.ace_range = AngleRange::from_degrees(0.0, 170.0);

        assert_gt!(params.bdf_range.min.to_radians(), 0.0);
        params.bdf_range = AngleRange::from_degrees(0.0, 170.0);

        assert_gt!(params.ced_range.min.to_radians(), 0.0);
        params.ced_range = AngleRange::from_degrees(0.0, 170.0);

        (params, ks)
    }

    /// Calculate distance between two kinematic states.
    fn kinematic_state_mean_square_deviation(
        state1: &KinematicState,
        state2: &KinematicState,
    ) -> f64 {
        (distance_squared(&state2.point_a, &state1.point_a)
            + distance_squared(&state2.point_b, &state1.point_b)
            + distance_squared(&state2.point_c, &state1.point_c)
            + distance_squared(&state2.point_d, &state1.point_d)
            + distance_squared(&state2.point_e, &state1.point_e)
            + distance_squared(&state2.point_f, &state1.point_f))
            / 6.0
    }

    #[proptest]
    fn third_point_relative_position(
        #[strategy(point2_strategy(-1000.0, 1000.0))] a1: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))] b1: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))] c1: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))] a2: Point2<f64>,
        #[strategy(point2_strategy(-1000.0, 1000.0))] b2: Point2<f64>,
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

        prop_assert!(relative_eq!(ac2_norm, ac1_norm, max_relative = 1e-3));
        prop_assert!(relative_eq!(bc2_norm, bc1_norm, max_relative = 1e-3));
    }

    #[test]
    fn forward_kinematics_example() {
        let (params, ks) = example_leg();
        let joint_angles = ks.get_joint_angles();
        let foot_position = ks.get_foot_position();

        let ks2 = KinematicState::with_joint_angles(&joint_angles, &params).unwrap();
        params.verify(&ks2).unwrap();

        dbg!(&ks);
        dbg!(&ks2);

        assert_abs_diff_eq!(&ks2.get_foot_position(), &foot_position, epsilon = 1e-6);
        assert_lt!(kinematic_state_mean_square_deviation(&ks, &ks2), 1e-3);
    }

    #[test]
    fn inverse_kinematics_example() {
        let (params, ks) = example_leg();
        let joint_angles = ks.get_joint_angles();
        let foot_position = ks.get_foot_position();

        let ks2 = KinematicState::with_foot_position(foot_position, &params).unwrap();
        params.verify(&ks2).unwrap();

        dbg!(&ks);
        dbg!(&ks2);

        assert_abs_diff_eq!(
            ks2.get_joint_angles().alpha,
            joint_angles.alpha,
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            ks2.get_joint_angles().beta,
            joint_angles.beta,
            epsilon = 1e-3
        );
        assert_lt!(kinematic_state_mean_square_deviation(&ks, &ks2), 1e-3);
    }

    #[proptest]
    fn forward_kinematics_fuzzing(#[strategy(kinematic_state_strategy())] ks: KinematicState) {
        let params = Params::with_kinematic_state(&ks, FRAC_PI_4);
        let joint_angles = ks.get_joint_angles();
        let foot_position = ks.get_foot_position();

        let ks2 = KinematicState::with_joint_angles(&joint_angles, &params).unwrap();
        params.verify(&ks2).unwrap();

        dbg!(&ks);
        dbg!(&ks2);

        assert_abs_diff_eq!(&ks2.get_foot_position(), &foot_position, epsilon = 1e-6);
        assert_lt!(kinematic_state_mean_square_deviation(&ks, &ks2), 1e-3);
    }

    #[proptest]
    fn inverse_kinematics_fuzzing(#[strategy(kinematic_state_strategy())] ks: KinematicState) {
        let params = Params::with_kinematic_state(&ks, FRAC_PI_4);
        let joint_angles = ks.get_joint_angles();
        let foot_position = ks.get_foot_position();

        let ks2 = KinematicState::with_foot_position(foot_position, &params).unwrap();
        params.verify(&ks2).unwrap();

        dbg!(&ks);
        dbg!(&ks2);

        assert_abs_diff_eq!(
            ks2.get_joint_angles().alpha,
            joint_angles.alpha,
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            ks2.get_joint_angles().beta,
            joint_angles.beta,
            epsilon = 1e-3
        );
        assert_lt!(kinematic_state_mean_square_deviation(&ks, &ks2), 1e-3);
    }
}

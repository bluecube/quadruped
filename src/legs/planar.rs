use std::f64::consts::PI;

use nalgebra::{Matrix2, Point2, SimdBool, SimdValue, Vector2, distance};
use serde::{Deserialize, Serialize};
use simba::simd::SimdRealField;

use crate::util::{
    find_triangle_point,
    planar::{angle_and_length_to_vector, perpendicular, vector_to_angle},
    pseudo_angle::{AngleRange, PseudoAngle},
    simd::MaskedValue,
};

/// Definition of leg geometry, see leg-schematic.svg for description of the values
/// Note that some of the parameters are redundant for simpler calculation
#[derive(Clone, Debug, PartialEq)]
pub struct Leg2D<T: SimdRealField> {
    pub point_a: Point2<T>,
    pub point_b: Point2<T>,
    pub len_ac: T,
    pub len_bd: T,
    pub len_ce: T,
    pub len_df: T,
    pub len_de: T,

    pub f_from_ed: ThirdPointRelativePosition<T>,
    pub e_from_fd: ThirdPointRelativePosition<T>,

    // pub a_range: AngleRange<T>,
    // pub b_range: AngleRange<T>,
    pub ace_range: AngleRange<T>,
    pub bdf_range: AngleRange<T>,
    pub ced_range: AngleRange<T>,
}

/// Positions of points in the 2D leg plane view
/// This is a necessary half way form when doing both forward and inverse kinematics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leg2DState<T: SimdRealField> {
    pub point_a: Point2<T>,
    pub point_b: Point2<T>,
    pub point_c: Point2<T>,
    pub point_d: Point2<T>,
    pub point_e: Point2<T>,
    pub point_f: Point2<T>,
}

#[derive(Clone, Debug)]
pub struct JointAngles<T> {
    pub alpha: T,
    pub beta: T,
}

#[derive(Clone, Debug)]
pub struct JointTorques<T> {
    pub torque_a: T,
    pub torque_b: T,
}

impl<U: SimdRealField + Copy> Leg2D<U> {
    /// Construct new Leg2DState from joint angles.
    pub fn forward_kinematics<T>(
        &self,
        joint_angles: &JointAngles<T>,
    ) -> MaskedValue<Leg2DState<T>, T::SimdBool>
    where
        T: SimdRealField + SimdValue<Element = U> + Copy,
    {
        let self_simd = Splatter(self);

        let point_a = self_simd.point_a();
        let point_b = self_simd.point_b();

        // TODO: Check the driving joint angles
        // if !self_simd.a_range.contains(joint_angles.alpha) {
        //     return None;
        // }
        // if !self_simd.b_range.contains(joint_angles.beta) {
        //     return None;
        // }

        let point_c = point_a + angle_and_length_to_vector(joint_angles.alpha, self_simd.len_ac());
        let point_d = point_b + angle_and_length_to_vector(joint_angles.beta, self_simd.len_bd());
        let (point_e, mut valid_mask) = find_triangle_point(
            &point_c,
            self_simd.len_ce(),
            &point_d,
            self_simd.len_de(),
            &self_simd.ced_range(),
        )
        .into_parts();

        valid_mask = valid_mask
            & self_simd
                .ace_range()
                .contains(PseudoAngle::with_points_and_lengths(
                    &point_a,
                    &point_c,
                    &point_e,
                    self_simd.len_ac(),
                    self_simd.len_ce(),
                ));
        let point_f = self_simd.f_from_ed().apply(&point_e, &point_d);

        valid_mask = valid_mask
            & self_simd
                .bdf_range()
                .contains(PseudoAngle::with_points_and_lengths(
                    &point_b,
                    &point_d,
                    &point_f,
                    self_simd.len_bd(),
                    self_simd.len_df(),
                ));

        MaskedValue::new(
            Leg2DState {
                point_a,
                point_b,
                point_c,
                point_d,
                point_e,
                point_f,
            },
            valid_mask,
        )
    }

    /// Construct new Leg2DState from a foot point.
    pub fn inverse_kinematics<T>(
        &self,
        foot_position: Point2<T>,
    ) -> MaskedValue<Leg2DState<T>, T::SimdBool>
    where
        T: SimdRealField + SimdValue<Element = U> + Copy,
    {
        let self_simd = Splatter::<T>(self);

        let point_a = self_simd.point_a();
        let point_b = self_simd.point_b();
        let point_f = foot_position;

        let (point_d, mut valid_mask) = find_triangle_point(
            &point_b,
            self_simd.len_bd(),
            &point_f,
            self_simd.len_df(),
            &self_simd.bdf_range(),
        )
        .into_parts();
        let point_e = self_simd.e_from_fd().apply(&point_f, &point_d);
        let point_c = find_triangle_point(
            &point_a,
            self_simd.len_ac(),
            &point_e,
            self_simd.len_ce(),
            &self_simd.ace_range(),
        )
        .unwrap_and_update_mask(&mut valid_mask);

        valid_mask = valid_mask
            & self_simd
                .bdf_range()
                .contains(PseudoAngle::with_points_and_lengths(
                    &point_b,
                    &point_d,
                    &point_f,
                    self_simd.len_bd(),
                    self_simd.len_df(),
                ));

        // TODO: Check A and B angle ranges?

        MaskedValue::new(
            Leg2DState {
                point_a,
                point_b,
                point_c,
                point_d,
                point_e,
                point_f,
            },
            valid_mask,
        )
    }
}

impl<T: SimdRealField + Copy> Leg2D<T> {
    pub fn map<T2: SimdRealField, F: FnMut(T) -> T2>(&self, mut f: F) -> Leg2D<T2> {
        Leg2D {
            point_a: self.point_a.map(&mut f),
            point_b: self.point_b.map(&mut f),
            len_ac: f(self.len_ac),
            len_bd: f(self.len_bd),
            len_ce: f(self.len_ce),
            len_df: f(self.len_df),
            len_de: f(self.len_de),
            f_from_ed: self.f_from_ed.map(&mut f),
            e_from_fd: self.e_from_fd.map(&mut f),
            ace_range: self.ace_range.map(&mut f),
            bdf_range: self.bdf_range.map(&mut f),
            ced_range: self.ced_range.map(&mut f),
        }
    }
}

/// Debugging functions, slow.
impl Leg2D<f64> {
    /// Verifies that leg state corresponds to this parameter set.
    /// In case the state mismatches, returns vector of strings with problems with the state.
    pub fn verify_state(&self, state: &Leg2DState<f64>) -> Result<(), Vec<String>> {
        let mut problems = Vec::new();

        let mut check_position = |name, point: &Point2<f64>, expected: &Point2<f64>| {
            if distance(point, expected) > 1e-3 {
                problems.push(format!(
                    "Point {name} expected {expected:?}, have {point:?}"
                ));
            }
        };
        check_position("a", &state.point_a, &self.point_a);
        check_position("b", &state.point_b, &self.point_b);

        let mut check_length = |name, point1: &Point2<f64>, point2: &Point2<f64>, expected: f64| {
            let length: f64 = distance(point1, point2);
            if (length - expected).abs() > 1e-3 {
                problems.push(format!("Length {name} expected {expected}, have {length}"));
            }
        };
        check_length("ac", &state.point_a, &state.point_c, self.len_ac);
        check_length("bd", &state.point_b, &state.point_d, self.len_bd);
        check_length("ce", &state.point_c, &state.point_e, self.len_ce);
        check_length("df", &state.point_d, &state.point_f, self.len_df);
        check_length("de", &state.point_d, &state.point_e, self.len_de);

        let mut check_angle =
            |name, a: &Point2<_>, b: &Point2<_>, c: &Point2<_>, range: &AngleRange<f64>| {
                if !range.contains(PseudoAngle::with_points(a, b, c)) {
                    problems.push(format!("Angle range {name} not satisfied"));
                }
            };
        check_angle(
            "ace",
            &state.point_a,
            &state.point_c,
            &state.point_e,
            &self.ace_range,
        );
        check_angle(
            "bdf",
            &state.point_b,
            &state.point_d,
            &state.point_f,
            &self.bdf_range,
        );
        check_angle(
            "ced",
            &state.point_c,
            &state.point_e,
            &state.point_d,
            &self.ced_range,
        );

        if problems.is_empty() {
            Ok(())
        } else {
            Err(problems)
        }
    }

    /// Extract __some__ parameters that correspond to an existing state.
    ///
    /// `angle_tolerance` is angle difference in radians from the input state allowed by
    /// the new leg parameters (but the angle will always be cropped at  0 and +-180)
    pub fn with_state(state: &Leg2DState<f64>, angle_tolerance_radians: f64) -> Self {
        let range_from_points = |a, b, c| {
            let angle = PseudoAngle::<f64>::with_points(a, b, c).to_radians();
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

        Leg2D {
            point_a: state.point_a,
            point_b: state.point_b,
            len_ac: distance(&state.point_a, &state.point_c),
            len_bd: distance(&state.point_b, &state.point_d),
            len_ce: distance(&state.point_c, &state.point_e),
            len_df: distance(&state.point_d, &state.point_f),
            len_de: distance(&state.point_d, &state.point_e),
            e_from_fd: ThirdPointRelativePosition::new(
                &state.point_f,
                &state.point_d,
                &state.point_e,
            ),
            f_from_ed: ThirdPointRelativePosition::new(
                &state.point_e,
                &state.point_d,
                &state.point_f,
            ),
            ace_range: range_from_points(&state.point_a, &state.point_c, &state.point_e),
            bdf_range: range_from_points(&state.point_b, &state.point_d, &state.point_f),
            ced_range: range_from_points(&state.point_c, &state.point_e, &state.point_d),
        }
    }

    pub fn jacobian(&self, joint_angles: &JointAngles<f64>) -> Option<Matrix2<f64>> {
        type AF = autofloat::AutoFloat<f64, 2>;
        let foot_pos_pds = self
            .map(|x| AF::constant(x))
            .forward_kinematics::<AF>(&JointAngles {
                alpha: AF::variable(joint_angles.alpha, 0),
                beta: AF::variable(joint_angles.beta, 1),
            })
            .into_option()?
            .get_foot_position();

        let jacobian = Matrix2::from_fn(|i, j| foot_pos_pds[i].dx[j]);
        Some(jacobian)
    }

    pub fn forward_force_transfer_jacobian(
        &self,
        joint_angles: &JointAngles<f64>,
        joint_torques: &JointTorques<f64>,
    ) -> Option<Vector2<f64>> {
        let jacobian = self.jacobian(joint_angles)?;
        let torque = Vector2::new(joint_torques.torque_a, joint_torques.torque_b);
        //torque = jacobian.transpose() * force
        let foot_force = jacobian.transpose().try_inverse().unwrap() * torque;

        Some(foot_force)
    }

    pub fn inverse_force_transfer_jacobian(
        &self,
        joint_angles: &JointAngles<f64>,
        foot_force: &Vector2<f64>,
    ) -> Option<JointTorques<f64>> {
        let jacobian = self.jacobian(joint_angles)?;
        let torque = jacobian.transpose() * foot_force;

        let torque = JointTorques {
            torque_a: torque[0],
            torque_b: torque[1],
        };

        Some(torque)
    }
}

/// Accessing results of FK an IK.
impl<T: SimdRealField + Copy> Leg2DState<T>
where
    T::Element: SimdRealField + Copy,
    T::SimdBool: SimdBool,
{
    pub fn get_foot_position(&self) -> Point2<T> {
        self.point_f
    }

    pub fn get_joint_angles(&self) -> JointAngles<T> {
        JointAngles {
            alpha: vector_to_angle(&(self.point_c - self.point_a)),
            beta: vector_to_angle(&(self.point_d - self.point_b)),
        }
    }
}

/// Force transfer
impl<T: SimdRealField + Copy> Leg2DState<T> {
    pub fn forward_force_transfer(&self, torques: &JointTorques<T>, leg: &Leg2D<T>) -> Vector2<T> {
        let ac = self.point_c - self.point_a;
        let ce = self.point_e - self.point_c;
        let bd = self.point_d - self.point_b;
        let fd = self.point_d - self.point_f;
        let fe = self.point_e - self.point_f;

        let force_ce = ce * torques.torque_a / ac.perp(&ce);

        // Force applied at point d by torque_b
        let force_d_from_torque_b =
            perpendicular(&bd) * (-torques.torque_b / (leg.len_bd * leg.len_bd));
        let force_bd =
            bd * (-(fd.perp(&force_d_from_torque_b) + fe.perp(&force_ce)) / fd.perp(&bd));

        force_ce + force_d_from_torque_b + force_bd
    }
}

/// Represents a position of a point relative to two other points, forming a simillar triangle.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ThirdPointRelativePosition<T: SimdRealField> {
    longitudal: T,
    lateral: T,
}

impl<T: SimdRealField + Copy> ThirdPointRelativePosition<T> {
    /// Construct new ThirdPointRelativePosition from an example,
    /// first two points are the pattern, third point is the expected output
    pub fn new(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>) -> Self {
        let ab = b - a;
        let bc = c - b;
        ThirdPointRelativePosition {
            longitudal: bc.dot(&ab) / ab.norm_squared(),
            lateral: bc.perp(&ab) / ab.norm_squared(),
        }
    }

    /// Obtain a position of the third point from the two pattern points
    pub fn apply(&self, a: &Point2<T>, b: &Point2<T>) -> Point2<T> {
        let ab = b - a;
        b + ab * self.longitudal + perpendicular(&ab) * self.lateral
    }
}

impl<T: SimdRealField + Copy> ThirdPointRelativePosition<T> {
    pub fn map<T2: SimdRealField, F: FnMut(T) -> T2>(
        &self,
        mut f: F,
    ) -> ThirdPointRelativePosition<T2> {
        ThirdPointRelativePosition {
            longitudal: f(self.longitudal),
            lateral: f(self.lateral),
        }
    }
}

struct Splatter<'a, T: SimdRealField>(&'a Leg2D<T::Element>)
where
    T::Element: SimdRealField;

impl<'a, T: SimdRealField> Splatter<'a, T>
where
    T::Element: SimdRealField + Copy,
{
    fn point_a(&self) -> Point2<T> {
        self.0.point_a.map(T::splat)
    }

    fn point_b(&self) -> Point2<T> {
        self.0.point_b.map(T::splat)
    }

    fn len_ac(&self) -> T {
        T::splat(self.0.len_ac)
    }

    fn len_bd(&self) -> T {
        T::splat(self.0.len_bd)
    }

    fn len_ce(&self) -> T {
        T::splat(self.0.len_ce)
    }

    fn len_df(&self) -> T {
        T::splat(self.0.len_df)
    }

    fn len_de(&self) -> T {
        T::splat(self.0.len_de)
    }

    fn f_from_ed(&self) -> ThirdPointRelativePosition<T> {
        self.0.f_from_ed.map(T::splat)
    }

    fn e_from_fd(&self) -> ThirdPointRelativePosition<T> {
        self.0.e_from_fd.map(T::splat)
    }

    fn ace_range(&self) -> AngleRange<T> {
        self.0.ace_range.map(T::splat)
    }

    fn bdf_range(&self) -> AngleRange<T> {
        self.0.bdf_range.map(T::splat)
    }

    fn ced_range(&self) -> AngleRange<T> {
        self.0.ced_range.map(T::splat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::proptest_util::{point2_strategy, vector2_strategy};
    use approx::{abs_diff_eq, relative_eq};
    use clap::error::Result;
    use more_asserts::assert_gt;
    use nalgebra::distance_squared;
    use proptest::prelude::*;
    use std::{f64::consts::FRAC_PI_4, panic};
    use test_strategy::proptest;

    fn leg_state_strategy() -> impl Strategy<Value = Leg2DState<f64>> {
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
                Some(Leg2DState {
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

    /// Leg definition and state taken from the sketch in leg-schematic.svg
    fn example_leg() -> (Leg2D<f64>, Leg2DState<f64>) {
        let state = Leg2DState {
            point_a: Point2::new(0.0, 0.0),
            point_b: Point2::new(-30.0, -25.0),
            point_c: Point2::new(-45.0, 35.0),
            point_d: Point2::new(-75.0, -10.0),
            point_e: Point2::new(-120.0, 20.0),
            point_f: Point2::new(-15.0, -120.0),
        };

        let mut params = Leg2D::with_state(&state, 0.0);

        assert_gt!(params.ace_range.min.to_radians(), 0.0);
        params.ace_range = AngleRange::from_degrees(0.0, 170.0);

        assert_gt!(params.bdf_range.min.to_radians(), 0.0);
        params.bdf_range = AngleRange::from_degrees(0.0, 170.0);

        assert_gt!(params.ced_range.min.to_radians(), 0.0);
        params.ced_range = AngleRange::from_degrees(0.0, 170.0);

        (params, state)
    }

    /// Calculate distance between two leg states.
    fn leg2d_state_mean_square_deviation(
        state1: &Leg2DState<f64>,
        state2: &Leg2DState<f64>,
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

    fn forward_kinematics_test(
        leg: Leg2D<f64>,
        leg_state: Leg2DState<f64>,
    ) -> Result<(), TestCaseError> {
        let joint_angles = leg_state.get_joint_angles();
        let foot_position = leg_state.get_foot_position();

        let leg_state2 = leg.forward_kinematics(&joint_angles).into_option();

        dbg!(&leg_state);
        dbg!(&leg_state2);

        prop_assert!(leg_state2.is_some());
        let leg_state2 = leg_state2.unwrap();
        if let Err(problems) = leg.verify_state(&leg_state2) {
            prop_assert!(false, "leg state verification failed: {:?}", problems);
        }

        prop_assert!(abs_diff_eq!(
            &leg_state2.get_foot_position(),
            &foot_position,
            epsilon = 1e-6
        ));
        prop_assert!(leg2d_state_mean_square_deviation(&leg_state, &leg_state2) < 1e-3);

        Ok(())
    }

    fn inverse_kinematics_test(
        leg: Leg2D<f64>,
        leg_state: Leg2DState<f64>,
    ) -> Result<(), TestCaseError> {
        let joint_angles = leg_state.get_joint_angles();
        let foot_position = leg_state.get_foot_position();

        let leg_state2 = leg.inverse_kinematics(foot_position).into_option();

        dbg!(&leg_state);
        dbg!(&leg_state2);

        prop_assert!(leg_state2.is_some());
        let leg_state2 = leg_state2.unwrap();
        if let Err(problems) = leg.verify_state(&leg_state2) {
            prop_assert!(false, "leg state verification failed: {:?}", problems);
        }

        prop_assert!(abs_diff_eq!(
            leg_state2.get_joint_angles().alpha,
            joint_angles.alpha,
            epsilon = 1e-3
        ));
        prop_assert!(abs_diff_eq!(
            leg_state2.get_joint_angles().beta,
            joint_angles.beta,
            epsilon = 1e-3
        ));
        prop_assert!(leg2d_state_mean_square_deviation(&leg_state, &leg_state2) < 1e-3);

        Ok(())
    }

    #[test]
    fn forward_kinematics_example() -> Result<(), TestCaseError> {
        let (leg, leg_state) = example_leg();
        forward_kinematics_test(leg, leg_state)
    }

    #[test]
    fn inverse_kinematics_example() -> Result<(), TestCaseError> {
        let (leg, leg_state) = example_leg();
        inverse_kinematics_test(leg, leg_state)
    }

    #[proptest]
    fn forward_kinematics_fuzzing(
        #[strategy(leg_state_strategy())] leg_state: Leg2DState<f64>,
    ) -> Result<(), TestCaseError> {
        let leg = Leg2D::with_state(&leg_state, FRAC_PI_4);
        forward_kinematics_test(leg, leg_state)
    }

    #[proptest]
    fn inverse_kinematics_fuzzing(
        #[strategy(leg_state_strategy())] leg_state: Leg2DState<f64>,
    ) -> Result<(), TestCaseError> {
        let leg = Leg2D::with_state(&leg_state, FRAC_PI_4);
        forward_kinematics_test(leg, leg_state)
    }

    #[proptest]
    fn leg_map_identity(#[strategy(leg_state_strategy())] leg_state: Leg2DState<f64>) {
        let leg = Leg2D::with_state(&leg_state, FRAC_PI_4);

        let mapped = leg.map(|x| x);

        prop_assert_eq!(mapped, leg);
    }

    #[proptest]
    fn forward_force_direct_vs_jacobian(
        #[strategy(leg_state_strategy())] leg_state: Leg2DState<f64>,
        #[strategy(((-1.0f64..1.0f64), (-1.0f64..1.0f64)).prop_map(|(torque_a, torque_b)| JointTorques{torque_a, torque_b}))]
        joint_torques: JointTorques<f64>,
    ) {
        // Don't allow ACE angles close to 0, because that means forces are indeterminate
        prop_assume!(
            PseudoAngle::with_points(&leg_state.point_a, &leg_state.point_c, &leg_state.point_e)
                .to_degrees()
                .abs()
                > 5.0
        );

        let leg = Leg2D::with_state(&leg_state, FRAC_PI_4);
        let joint_angles = leg_state.get_joint_angles();

        let direct_force = leg_state.forward_force_transfer(&joint_torques, &leg);
        let jacobian_force = leg
            .forward_force_transfer_jacobian(&joint_angles, &joint_torques)
            .unwrap();

        prop_assert!(
            relative_eq!(direct_force, jacobian_force, max_relative = 1e-3),
            "{:?} != {:?}",
            direct_force,
            jacobian_force
        );
    }

    #[proptest]
    fn force_round_trip(
        #[strategy(leg_state_strategy())] leg_state: Leg2DState<f64>,
        #[strategy(vector2_strategy(-100.0, 100.0))] foot_force: Vector2<f64>,
    ) {
        // Don't allow ACE angles close to 0, because that means forces are indeterminate
        prop_assume!(
            PseudoAngle::with_points(&leg_state.point_a, &leg_state.point_c, &leg_state.point_e)
                .to_degrees()
                .abs()
                > 5.0
        );

        let leg = Leg2D::with_state(&leg_state, FRAC_PI_4);
        let joint_angles = leg_state.get_joint_angles();

        let torque = leg.inverse_force_transfer_jacobian(&joint_angles, &foot_force);
        prop_assert!(torque.is_some());
        let torque = torque.unwrap();

        let reconstructed_foot_force = leg_state.forward_force_transfer(&torque, &leg);

        prop_assert!(
            relative_eq!(reconstructed_foot_force, foot_force, max_relative = 1e-3),
            "{:?} != {:?}",
            reconstructed_foot_force,
            foot_force
        );
    }
}

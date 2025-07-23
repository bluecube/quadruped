//! Random collection of utilities for working with 2D vectors
use nalgebra::Vector2;
use simba::simd::SimdRealField;

/// Create a perpendicular 2D vector to a given vector.
/// Magnitude is identical to the argument.
pub fn perpendicular<T: SimdRealField + Copy>(v: &Vector2<T>) -> Vector2<T> {
    Vector2::new(v.y, -v.x)
}

pub fn vector_to_angle<T: SimdRealField + Copy>(v: &Vector2<T>) -> T {
    v.y.simd_atan2(v.x)
}

pub fn angle_and_length_to_vector<T: SimdRealField>(angle: T, length: T) -> Vector2<T> {
    let (s, c) = angle.simd_sin_cos();
    Vector2::new(c, s) * length
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::relative_eq;
    use nalgebra::Const;
    use proptest::prelude::*;
    use test_strategy::proptest;

    #[proptest]
    fn perpendicular_has_zero_dot_product(
        #[strategy(nalgebra::proptest::vector(-1000.0..1000.0, Const::<2>))] v: Vector2<f64>,
    ) {
        prop_assert_eq!(v.dot(&perpendicular(&v)), 0.0);
    }

    #[proptest]
    fn vector_to_angle_roundtrip(
        #[strategy(-std::f64::consts::PI..std::f64::consts::PI)] angle: f64,
        #[strategy(1.0..1000.0)] length: f64,
    ) {
        let v = angle_and_length_to_vector(angle, length);
        let angle2 = vector_to_angle(&v);

        prop_assert!(relative_eq!(angle2, angle, max_relative = 1e-6));
    }
}

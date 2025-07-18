use nalgebra::{Const, Point2};
use proptest::prelude::*;

/// Strategy for 64bit floating point numbers that minimize to nicely readable integer values
pub fn f64_strategy(range: std::ops::Range<f64>) -> impl Strategy<Value = f64> + Clone {
    (
        0f64..1f64,
        (range.start.floor() as i64)..(range.end.floor() as i64),
    )
        .prop_filter_map(
            "Value does not fit in the range",
            move |(fractional, integral)| {
                let v = (integral as f64) + fractional;
                if range.contains(&v) {
                    Some(v)
                } else {
                    None
                }
            },
        )
}

pub fn point2_strategy(min_value: f64, max_value: f64) -> impl Strategy<Value = Point2<f64>> {
    //vector(proptest::num::f64::NORMAL, Const::<2>).prop_map(|x| Point2::from(x))
    nalgebra::proptest::vector(f64_strategy(min_value..max_value), Const::<2>).prop_map_into()
}

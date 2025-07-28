use nalgebra::SimdBool;

pub struct MaskedValue<V, M> {
    pub value: V,
    pub mask: M,
}

impl<V, M> MaskedValue<V, M> {
    pub fn new(value: V, mask: M) -> MaskedValue<V, M> {
        MaskedValue { value, mask }
    }

    pub fn into_parts(self) -> (V, M) {
        (self.value, self.mask)
    }
}

impl<V, M: SimdBool> MaskedValue<V, M> {
    pub fn unwrap_and_update_mask(self, mask: &mut M) -> V {
        *mask = *mask & self.mask;
        self.value
    }
}

impl<T: std::fmt::Debug> MaskedValue<T, bool> {
    pub fn into_option(self) -> Option<T> {
        if self.mask { Some(self.value) } else { None }
    }

    pub fn unwrap(self) -> T {
        if !self.mask {
            panic!("Called `MaskedValue::unwrap()` on a value with `false` mask");
        }

        self.value
    }
}

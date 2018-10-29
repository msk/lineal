use std::default::Default;
use std::ops::{Add, AddAssign, Mul};

pub fn dot<T>(x: &[T], y: &[T]) -> T
where
    T: Add + AddAssign + Mul<Output = T> + Copy + Default,
{
    let mut sum: T = Default::default();
    for (&a, &b) in x.iter().zip(y) {
        sum += a * b;
    }
    sum
}

#[cfg(test)]
mod tests {
    #[test]
    fn sdot() {
        let x: Vec<f32> = vec![10., 15., -6., 3., 14., 7.];
        let y: Vec<f32> = vec![8., -2., 4., 7., 6., -3.];
        assert_eq!(super::dot(&x, &y), 110f32);
    }

    #[test]
    fn ddot() {
        let x: Vec<f64> = vec![10., 15., -6., 3., 14., 7.];
        let y: Vec<f64> = vec![8., -2., 4., 7., 6., -3.];
        assert_eq!(super::dot(&x, &y), 110f64);
    }
}

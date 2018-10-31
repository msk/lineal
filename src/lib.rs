use std::cmp;
use std::default::Default;
use std::ops::{Add, AddAssign, Mul};

pub fn dot<T>(x: &[T], y: &[T]) -> T
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + Default,
{
    let len = cmp::min(x.len(), y.len());
    let mut x = &x[..len];
    let mut y = &y[..len];

    let mut sum0: T = Default::default();
    let mut sum1: T = Default::default();
    let mut sum2: T = Default::default();
    let mut sum3: T = Default::default();
    let mut sum4: T = Default::default();
    let mut sum5: T = Default::default();
    let mut sum6: T = Default::default();
    let mut sum7: T = Default::default();
    while x.len() >= 8 {
        sum0 += x[0] * y[0];
        sum1 += x[1] * y[1];
        sum2 += x[2] * y[2];
        sum3 += x[3] * y[3];
        sum4 += x[4] * y[4];
        sum5 += x[5] * y[5];
        sum6 += x[6] * y[6];
        sum7 += x[7] * y[7];

        x = &x[8..];
        y = &y[8..];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
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

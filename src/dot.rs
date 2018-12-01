#[cfg(target_arch = "x86")]
use std::arch::x86;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as x86;
use std::cmp;
use std::default::Default;
use std::mem;
use std::ops::{Add, AddAssign, Mul};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let len = cmp::min(x.len(), y.len());

    if len >= 8 && is_aligned(x.as_ptr(), 8) && is_aligned(y.as_ptr(), 8) {
        if is_x86_feature_detected!("avx") {
            let remainder = len % 8;
            let xptr = x.as_ptr();
            let yptr = y.as_ptr();
            if is_aligned(xptr, 32) && is_aligned(yptr, 32) {
                let mut remaining = (len - remainder) as isize;
                let unpacked: (f64, f64, f64, f64) = unsafe {
                    let mut sum0 = x86::_mm256_setzero_pd();
                    let mut sum1 = x86::_mm256_setzero_pd();
                    while remaining != 0 {
                        let x0 = x86::_mm256_load_pd(xptr.offset(remaining - 4));
                        let y0 = x86::_mm256_load_pd(yptr.offset(remaining - 4));
                        let p0 = x86::_mm256_mul_pd(x0, y0);
                        sum0 = x86::_mm256_add_pd(sum0, p0);
                        let x1 = x86::_mm256_load_pd(xptr.offset(remaining - 8));
                        let y1 = x86::_mm256_load_pd(yptr.offset(remaining - 8));
                        let p1 = x86::_mm256_mul_pd(x1, y1);
                        sum1 = x86::_mm256_add_pd(sum1, p1);
                        remaining -= 8;
                    }
                    let sum = x86::_mm256_add_pd(sum0, sum1);
                    mem::transmute(sum)
                };
                let mut sum = unpacked.0 + unpacked.1 + unpacked.2 + unpacked.3;
                for (a, b) in x[len - remainder..len]
                    .iter()
                    .zip(y[len - remainder..len].iter())
                {
                    sum += a * b;
                }
                sum
            } else {
                dot_unaligned(x, y, len)
            }
        } else if is_x86_feature_detected!("sse2") {
            let mut remaining = len as isize;
            let mut xptr = x.as_ptr();
            let mut yptr = y.as_ptr();
            let mut sum = 0f64;
            if !is_aligned(yptr, 16) {
                if remaining == 8 {
                    let mut sum = 0f64;
                    for (&a, &b) in x.iter().zip(y) {
                        sum += a * b;
                    }
                    return sum;
                }
                unsafe {
                    sum = *xptr * *yptr;
                    xptr = xptr.offset(1);
                    yptr = yptr.offset(1);
                }
                remaining -= 1;
            };
            if is_aligned(xptr, 16) {
                let remainder = remaining as usize % 8;
                remaining -= remainder as isize;
                let unpacked: (f64, f64) = unsafe {
                    let mut sum0 = x86::_mm_setzero_pd();
                    let mut sum1 = x86::_mm_setzero_pd();
                    let mut sum2 = x86::_mm_setzero_pd();
                    let mut sum3 = x86::_mm_setzero_pd();
                    while remaining != 0 {
                        let x0 = x86::_mm_load_pd(xptr.offset(remaining - 2));
                        let y0 = x86::_mm_load_pd(yptr.offset(remaining - 2));
                        let p0 = x86::_mm_mul_pd(x0, y0);
                        sum0 = x86::_mm_add_pd(sum0, p0);
                        let x1 = x86::_mm_load_pd(xptr.offset(remaining - 4));
                        let y1 = x86::_mm_load_pd(yptr.offset(remaining - 4));
                        let p1 = x86::_mm_mul_pd(x1, y1);
                        sum1 = x86::_mm_add_pd(sum1, p1);
                        let x2 = x86::_mm_load_pd(xptr.offset(remaining - 6));
                        let y2 = x86::_mm_load_pd(yptr.offset(remaining - 6));
                        let p2 = x86::_mm_mul_pd(x2, y2);
                        sum2 = x86::_mm_add_pd(sum2, p2);
                        let x3 = x86::_mm_load_pd(xptr.offset(remaining - 8));
                        let y3 = x86::_mm_load_pd(yptr.offset(remaining - 8));
                        let p3 = x86::_mm_mul_pd(x3, y3);
                        sum3 = x86::_mm_add_pd(sum3, p3);
                        remaining -= 8;
                    }
                    let sum01 = x86::_mm_add_pd(sum0, sum1);
                    let sum23 = x86::_mm_add_pd(sum2, sum3);
                    let sum = x86::_mm_add_pd(sum01, sum23);
                    mem::transmute(sum)
                };
                sum += unpacked.0 + unpacked.1;
                for (a, b) in x[len - remainder..len]
                    .iter()
                    .zip(y[len - remainder..len].iter())
                {
                    sum += a * b;
                }
                sum
            } else {
                dot_unaligned(x, y, len)
            }
        } else {
            dot(x, y)
        }
    } else {
        let mut sum = 0f64;
        for (&a, &b) in x.iter().zip(y) {
            sum += a * b;
        }
        sum
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    dot(x, y)
}

pub fn dot<T>(x: &[T], y: &[T]) -> T
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + Default,
{
    let len = cmp::min(x.len(), y.len());
    dot_unaligned(x, y, len)
}

fn dot_unaligned<T>(x: &[T], y: &[T], len: usize) -> T
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + Default,
{
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

fn is_aligned<T>(ptr: *const T, size: usize) -> bool {
    ptr as usize % size == 0
}

#[cfg(test)]
mod tests {
    extern crate permutator;

    use self::permutator::cartesian_product;

    #[test]
    fn sdot() {
        let x: Vec<f32> = vec![10., 15., -6., 3., 14., 7.];
        let y: Vec<f32> = vec![8., -2., 4., 7., 6., -3.];
        assert_eq!(super::dot(&x, &y), 110f32);
    }

    #[test]
    fn ddot() {
        let x = [10f64, 15f64, -6f64, 3f64, 14f64, 7f64];
        let y = [8f64, -2f64, 4f64, 7f64, 6f64, -3f64];
        shift_and_ddot(&x, &y, 110f64);
    }

    #[test]
    fn ddot_long() {
        let x = [1f64, 1f64, 1f64, 1f64, 1f64, 1f64, 1f64, 1f64, 1f64, 1f64];
        let y = [2f64, 1f64, 1f64, 1f64, 1f64, 1f64, 1f64, 1f64, 3f64, 4f64];
        shift_and_ddot(&x, &y, 16f64);
    }

    fn copy(dst: &mut Vec<f64>, src: &[f64], offset: usize) {
        assert!(offset < 4);
        assert!(dst.len() >= src.len() + offset);
        for mut d in dst[0..offset].iter_mut() {
            *d = 0f64;
        }
        for (mut d, s) in dst[offset..].iter_mut().zip(src) {
            *d = *s;
        }
    }

    fn shift_and_ddot(x: &[f64], y: &[f64], expected: f64) {
        let mut x_shifted = vec![0f64; x.len() + 3];
        let mut y_shifted = vec![0f64; y.len() + 3];
        cartesian_product(&[&[0, 1, 2, 3], &[0, 1, 2, 3]], |offsets| {
            copy(&mut x_shifted, &x, *offsets[0]);
            copy(&mut y_shifted, &y, *offsets[1]);
            assert_eq!(
                super::ddot(
                    &x_shifted[*offsets[0]..offsets[0] + x.len()],
                    &y_shifted[*offsets[1]..offsets[1] + y.len()]
                ),
                expected
            );
        });
    }
}

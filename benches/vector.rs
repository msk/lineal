#[macro_use]
extern crate criterion;
extern crate lineal;
extern crate rand;

use criterion::Criterion;
use rand::distributions::Standard;
use rand::{IsaacRng, Rng, SeedableRng};

fn dot_product_rblas(x: &[f64], y: &[f64]) -> f64 {
    rblas::Dot::dot(x, y)
}

fn ddot(c: &mut Criterion) {
    let data_length = 1000;
    let mut rng: IsaacRng = SeedableRng::from_seed([0u8; 32]);
    let x: Vec<f64> = rng.sample_iter(&Standard).take(data_length + 7).collect();
    let y: Vec<f64> = rng.sample_iter(&Standard).take(data_length + 7).collect();

    for offset_x in 0..4 {
        for offset_y in 0..4 {
            let mut xtmp = vec![0f64; data_length + 7];
            let align_offset = 32 - (xtmp.as_ptr() as usize) % 32;
            let x_start = if align_offset / 8 < offset_x {
                offset_x - align_offset / 8
            } else if align_offset / 8 > offset_x {
                4 + offset_x - align_offset / 8
            } else {
                0
            };
            xtmp[x_start..x_start + data_length].copy_from_slice(&x[0..data_length]);
            let mut ytmp = vec![0f64; data_length + 7];
            let align_offset = 32 - (ytmp.as_ptr() as usize) % 32;
            let y_start = if align_offset / 8 < offset_y {
                offset_y - align_offset / 8
            } else if align_offset / 8 > offset_y {
                4 + offset_y - align_offset / 8
            } else {
                0
            };
            ytmp[y_start..y_start + data_length].copy_from_slice(&y[0..data_length]);
            c.bench_function(
                &format!("ddot lineal ({}, {})", offset_x, offset_y),
                move |b| {
                    b.iter(|| {
                        lineal::ddot(
                            &xtmp[x_start..x_start + data_length],
                            &ytmp[y_start..y_start + data_length],
                        )
                    })
                },
            );

            let mut xtmp = vec![0f64; data_length + 7];
            let align_offset = 32 - (xtmp.as_ptr() as usize) % 32;
            let x_start = if align_offset / 8 < offset_x {
                offset_x - align_offset / 8
            } else if align_offset / 8 > offset_x {
                4 + offset_x - align_offset / 8
            } else {
                0
            };
            xtmp[x_start..x_start + data_length].copy_from_slice(&x[0..data_length]);
            let mut ytmp = vec![0f64; data_length + 7];
            let align_offset = 32 - (ytmp.as_ptr() as usize) % 32;
            let y_start = if align_offset / 8 < offset_y {
                offset_y - align_offset / 8
            } else if align_offset / 8 > offset_y {
                4 + offset_y - align_offset / 8
            } else {
                0
            };
            ytmp[y_start..y_start + data_length].copy_from_slice(&y[0..data_length]);
            c.bench_function(
                &format!("ddot rblas ({}, {})", offset_x, offset_y),
                move |b| {
                    b.iter(|| {
                        dot_product_rblas(
                            &xtmp[x_start..x_start + data_length],
                            &ytmp[y_start..y_start + data_length],
                        )
                    })
                },
            );
        }
    }
}

criterion_group!(vector, ddot);
criterion_main!(vector);

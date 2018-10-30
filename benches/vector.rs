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
    let mut rng: IsaacRng = SeedableRng::from_seed([0u8; 32]);
    let x: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();
    let y: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();

    let xtmp = x.clone();
    let ytmp = y.clone();
    c.bench_function("ddot_lineal", move |b| {
        b.iter(|| lineal::ddot(&xtmp, &ytmp))
    });

    let xtmp = x.clone();
    let ytmp = y.clone();
    c.bench_function("ddot_rblas", move |b| {
        b.iter(|| dot_product_rblas(&xtmp, &ytmp))
    });
}

criterion_group!(vector, ddot);
criterion_main!(vector);

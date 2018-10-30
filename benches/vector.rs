#[macro_use]
extern crate criterion;
extern crate lineal;
extern crate rand;

use criterion::Criterion;
use rand::distributions::Standard;
use rand::{IsaacRng, Rng, SeedableRng};

fn ddot(c: &mut Criterion) {
    let mut rng: IsaacRng = SeedableRng::from_seed([0u8; 32]);
    let x: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();
    let y: Vec<f64> = rng.sample_iter(&Standard).take(1000).collect();
    c.bench_function("ddot_lineal", move |b| b.iter(|| lineal::ddot(&x, &y)));
}

criterion_group!(vector, ddot);
criterion_main!(vector);

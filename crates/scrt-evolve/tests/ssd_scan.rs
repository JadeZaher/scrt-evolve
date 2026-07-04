//! PB-1 correctness tests for the SSD selective-scan CPU reference.
//! Validated against hand-computed analytic outputs on constructed inputs.
//! Also verifies shape mismatches surface as `Err`, not panics.

#![cfg(feature = "train")]

use candle_core::{Device, Tensor};
use scrt_evolve::serve::ops::{ssd_scan, ScanError};

const TOL: f32 = 1e-6;

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn approx(actual: f32, expected: f32, ctx: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= TOL,
        "{ctx}: got {actual}, expected {expected}, |Δ|={diff} > {TOL}"
    );
}

/// Case 1: A=0, all ones, D=None ⇒ pure accumulator.
/// state_t = (t+1), y_t = (t+1) at every (h,p).
#[test]
fn case_pure_accumulator_a_zero() {
    let dev = Device::Cpu;
    let (bs, l, h, p, n) = (1, 5, 2, 3, 4);

    let x = Tensor::ones((bs, l, h, p), candle_core::DType::F32, &dev).unwrap();
    let dt = Tensor::ones((bs, l, h), candle_core::DType::F32, &dev).unwrap();
    let a = Tensor::zeros((h,), candle_core::DType::F32, &dev).unwrap();
    let b = Tensor::ones((bs, l, h, n), candle_core::DType::F32, &dev).unwrap();
    let c = Tensor::ones((bs, l, h, n), candle_core::DType::F32, &dev).unwrap();

    let y = ssd_scan(&x, &dt, &a, &b, &c, None).unwrap();
    let yv = flat(&y);
    let idx = |bi: usize, ti: usize, hi: usize, pi: usize| ((bi * l + ti) * h + hi) * p + pi;

    for ti in 0..l {
        // With N states each contributing (t+1), sum over n gives n*(t+1).
        let expected = (n as f32) * ((ti + 1) as f32);
        for hi in 0..h {
            for pi in 0..p {
                approx(yv[idx(0, ti, hi, pi)], expected, &format!("t={ti},h={hi},p={pi}"));
            }
        }
    }
}

/// Case 2: initial-step correctness with A=-1.
/// state_0 = 0*exp(-1) + 1·1·1 = 1; y_0 = Σ_n 1 · 1 = N.
#[test]
fn case_initial_step_a_negative() {
    let dev = Device::Cpu;
    let (bs, l, h, p, n) = (1, 1, 1, 1, 3);

    let x = Tensor::ones((bs, l, h, p), candle_core::DType::F32, &dev).unwrap();
    let dt = Tensor::ones((bs, l, h), candle_core::DType::F32, &dev).unwrap();
    let a = Tensor::from_vec(vec![-1.0f32], (h,), &dev).unwrap();
    let b = Tensor::ones((bs, l, h, n), candle_core::DType::F32, &dev).unwrap();
    let c = Tensor::ones((bs, l, h, n), candle_core::DType::F32, &dev).unwrap();

    let y = ssd_scan(&x, &dt, &a, &b, &c, None).unwrap();
    let v = flat(&y);
    approx(v[0], n as f32, "initial-step");
}

/// Case 3: two-step decay compounding with A=-1, unit params.
/// state_0 = 1; state_1 = e^{-1} + 1. y_1 = Σ_n state_1 = N * (1 + e^{-1}).
#[test]
fn case_two_step_decay() {
    let dev = Device::Cpu;
    let (bs, l, h, p, n) = (1, 2, 1, 1, 2);

    let x = Tensor::ones((bs, l, h, p), candle_core::DType::F32, &dev).unwrap();
    let dt = Tensor::ones((bs, l, h), candle_core::DType::F32, &dev).unwrap();
    let a = Tensor::from_vec(vec![-1.0f32], (h,), &dev).unwrap();
    let b = Tensor::ones((bs, l, h, n), candle_core::DType::F32, &dev).unwrap();
    let c = Tensor::ones((bs, l, h, n), candle_core::DType::F32, &dev).unwrap();

    let y = ssd_scan(&x, &dt, &a, &b, &c, None).unwrap();
    let v = flat(&y);
    approx(v[0], n as f32, "t=0");
    let expected_t1 = (n as f32) * (1.0 + (-1.0f32).exp());
    approx(v[1], expected_t1, "t=1");
}

/// Case 4: D-skip decoupled from recurrence.
/// B=0 ⇒ state stays 0 ⇒ y = D·x purely.
#[test]
fn case_d_skip_only() {
    let dev = Device::Cpu;
    let (bs, l, h, p, n) = (2, 4, 3, 2, 5);

    // x = 2.0 everywhere; D = [1, 2, 3]; expect y[b,t,h,p] = D[h] * 2.
    let x = Tensor::from_vec(vec![2.0f32; bs * l * h * p], (bs, l, h, p), &dev).unwrap();
    let dt = Tensor::ones((bs, l, h), candle_core::DType::F32, &dev).unwrap();
    let a = Tensor::from_vec(vec![-2.71f32, 0.5, -0.1], (h,), &dev).unwrap();
    let b = Tensor::zeros((bs, l, h, n), candle_core::DType::F32, &dev).unwrap();
    let c = Tensor::ones((bs, l, h, n), candle_core::DType::F32, &dev).unwrap();
    let d = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (h,), &dev).unwrap();

    let y = ssd_scan(&x, &dt, &a, &b, &c, Some(&d)).unwrap();
    let v = flat(&y);
    let idx = |bi: usize, ti: usize, hi: usize, pi: usize| ((bi * l + ti) * h + hi) * p + pi;
    let d_vals = [1.0f32, 2.0, 3.0];
    for bi in 0..bs {
        for ti in 0..l {
            for hi in 0..h {
                for pi in 0..p {
                    approx(v[idx(bi, ti, hi, pi)], d_vals[hi] * 2.0, "d-skip");
                }
            }
        }
    }
}

/// Case 5: hand-computed cross-check with a distinct scalar reference
/// implementation, on fixed-seed pseudo-random inputs.
#[test]
fn case_random_vs_scalar_reference() {
    let dev = Device::Cpu;
    let (bs, l, h, p, n) = (2, 6, 3, 4, 5);

    // Deterministic pseudo-random fill (SplitMix64) — reproducible w/o RNG dep.
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let gen_n = |s: &mut u64, count: usize| -> Vec<f32> {
        (0..count)
            .map(|_| {
                *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = *s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    };

    let xv = gen_n(&mut s, bs * l * h * p);
    let dtv: Vec<f32> = gen_n(&mut s, bs * l * h).into_iter().map(|v| v.abs() + 1e-3).collect();
    // A negative for stability — common in practice.
    let av: Vec<f32> = gen_n(&mut s, h).into_iter().map(|v| -v.abs()).collect();
    let bv = gen_n(&mut s, bs * l * h * n);
    let cv = gen_n(&mut s, bs * l * h * n);
    let dv = gen_n(&mut s, h);

    let x = Tensor::from_vec(xv, (bs, l, h, p), &dev).unwrap();
    let dt = Tensor::from_vec(dtv, (bs, l, h), &dev).unwrap();
    let a = Tensor::from_vec(av, (h,), &dev).unwrap();
    let b = Tensor::from_vec(bv, (bs, l, h, n), &dev).unwrap();
    let c = Tensor::from_vec(cv, (bs, l, h, n), &dev).unwrap();
    let d = Tensor::from_vec(dv, (h,), &dev).unwrap();

    let y = ssd_scan(&x, &dt, &a, &b, &c, Some(&d)).unwrap();
    let got = flat(&y);

    let xh = flat(&x);
    let dth = flat(&dt);
    let ah = a.to_vec1::<f32>().unwrap();
    let bh = flat(&b);
    let ch = flat(&c);
    let dh = d.to_vec1::<f32>().unwrap();

    let x_idx = |bi: usize, ti: usize, hi: usize, pi: usize| ((bi * l + ti) * h + hi) * p + pi;
    let dt_idx = |bi: usize, ti: usize, hi: usize| (bi * l + ti) * h + hi;
    let bc_idx = |bi: usize, ti: usize, hi: usize, ni: usize| ((bi * l + ti) * h + hi) * n + ni;
    let s_idx = |hi: usize, ni: usize, pi: usize| (hi * n + ni) * p + pi;

    for bi in 0..bs {
        let mut st = vec![0f32; h * n * p];
        for ti in 0..l {
            for hi in 0..h {
                let dtv = dth[dt_idx(bi, ti, hi)];
                let da = (dtv * ah[hi]).exp();
                for ni in 0..n {
                    let bv = bh[bc_idx(bi, ti, hi, ni)];
                    for pi in 0..p {
                        let s = &mut st[s_idx(hi, ni, pi)];
                        *s = da * *s + dtv * bv * xh[x_idx(bi, ti, hi, pi)];
                    }
                }
                for pi in 0..p {
                    let mut acc = dh[hi] * xh[x_idx(bi, ti, hi, pi)];
                    for ni in 0..n {
                        acc += ch[bc_idx(bi, ti, hi, ni)] * st[s_idx(hi, ni, pi)];
                    }
                    let g = got[x_idx(bi, ti, hi, pi)];
                    let diff = (g - acc).abs();
                    assert!(diff <= 1e-4, "b={bi} t={ti} h={hi} p={pi}: got {g}, ref {acc}, Δ={diff}");
                }
            }
        }
    }
}

// ── shape-error surface (never panic) ────────────────────────────────────

#[test]
fn shape_error_x_rank() {
    let dev = Device::Cpu;
    let x = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &dev).unwrap();
    let dt = Tensor::zeros((2, 3, 4), candle_core::DType::F32, &dev).unwrap();
    let a = Tensor::zeros((4,), candle_core::DType::F32, &dev).unwrap();
    let b = Tensor::zeros((2, 3, 4, 5), candle_core::DType::F32, &dev).unwrap();
    let c = Tensor::zeros((2, 3, 4, 5), candle_core::DType::F32, &dev).unwrap();
    let e = ssd_scan(&x, &dt, &a, &b, &c, None).unwrap_err();
    assert!(matches!(e, ScanError::Shape(_)), "expected Shape err, got {e:?}");
}

#[test]
fn shape_error_b_c_mismatch() {
    let dev = Device::Cpu;
    let x = Tensor::zeros((1, 2, 2, 3), candle_core::DType::F32, &dev).unwrap();
    let dt = Tensor::zeros((1, 2, 2), candle_core::DType::F32, &dev).unwrap();
    let a = Tensor::zeros((2,), candle_core::DType::F32, &dev).unwrap();
    let b = Tensor::zeros((1, 2, 2, 4), candle_core::DType::F32, &dev).unwrap();
    let c = Tensor::zeros((1, 2, 2, 5), candle_core::DType::F32, &dev).unwrap();
    let e = ssd_scan(&x, &dt, &a, &b, &c, None).unwrap_err();
    assert!(matches!(e, ScanError::Shape(_)));
}

#[test]
fn shape_error_a_wrong() {
    let dev = Device::Cpu;
    let x = Tensor::zeros((1, 2, 2, 3), candle_core::DType::F32, &dev).unwrap();
    let dt = Tensor::zeros((1, 2, 2), candle_core::DType::F32, &dev).unwrap();
    let a = Tensor::zeros((3,), candle_core::DType::F32, &dev).unwrap();
    let b = Tensor::zeros((1, 2, 2, 4), candle_core::DType::F32, &dev).unwrap();
    let c = Tensor::zeros((1, 2, 2, 4), candle_core::DType::F32, &dev).unwrap();
    let e = ssd_scan(&x, &dt, &a, &b, &c, None).unwrap_err();
    assert!(matches!(e, ScanError::Shape(_)));
}

//! Mamba2 / SSD selective-scan CPU reference op (track 39, PB-1).
//!
//! **Correctness-critical.** See `AGENTS.md` in this directory for the exact
//! math this file implements, the public contract, and the validation
//! strategy. Do not "optimize" this file — PB-4 owns speed on CUDA; this
//! file is the oracle every other implementation is checked against.

use candle_core::{DType, Tensor};

/// Errors returned by [`ssd_scan`]. Shape checks are `Err`, never panic.
#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    /// Rank / dim mismatch across the tuple `(x, dt, A, B, C, D)`.
    #[error("ssd_scan shape mismatch: {0}")]
    Shape(String),
    /// A candle op (dtype cast, tensor build, host copy) failed.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

/// Selective SSD scan.
///
/// See `serve/ops/AGENTS.md` for the recurrence and the validated cases.
/// Returns `y: (B, L, H, P)` on `x.device()`, always fp32.
pub fn ssd_scan(
    x: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: Option<&Tensor>,
) -> Result<Tensor, ScanError> {
    // ── shape guards ────────────────────────────────────────────────────
    let xd = x.dims();
    if xd.len() != 4 {
        return Err(ScanError::Shape(format!("x must be rank-4 (B,L,H,P), got {:?}", xd)));
    }
    let (bs, l, h, p) = (xd[0], xd[1], xd[2], xd[3]);

    let bd = b.dims();
    if bd.len() != 4 || bd[0] != bs || bd[1] != l || bd[2] != h {
        return Err(ScanError::Shape(format!("B must be (B,L,H,N) matching x; got {:?} vs x {:?}", bd, xd)));
    }
    let n = bd[3];

    let cd = c.dims();
    if cd != bd {
        return Err(ScanError::Shape(format!("C shape {:?} must equal B shape {:?}", cd, bd)));
    }

    let dtd = dt.dims();
    if dtd.len() != 3 || dtd[0] != bs || dtd[1] != l || dtd[2] != h {
        return Err(ScanError::Shape(format!("dt must be (B,L,H); got {:?}", dtd)));
    }

    let ad = a.dims();
    if ad.len() != 1 || ad[0] != h {
        return Err(ScanError::Shape(format!("A must be (H,); got {:?}", ad)));
    }

    if let Some(dv) = d {
        let ddims = dv.dims();
        if ddims.len() != 1 || ddims[0] != h {
            return Err(ScanError::Shape(format!("D must be (H,); got {:?}", ddims)));
        }
    }

    // ── pull to host fp32 (flat, index manually — candle has no to_vec4) ─
    let dev = x.device().clone();
    let x_v: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let dt_v: Vec<f32> = dt.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let a_v: Vec<f32> = a.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let b_v: Vec<f32> = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let c_v: Vec<f32> = c.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let d_v: Option<Vec<f32>> = match d {
        Some(t) => Some(t.to_dtype(DType::F32)?.to_vec1::<f32>()?),
        None => None,
    };

    // Row-major indexers matching the tensor shapes above.
    let x_idx = |bi: usize, ti: usize, hi: usize, pi: usize| ((bi * l + ti) * h + hi) * p + pi;
    let dt_idx = |bi: usize, ti: usize, hi: usize| (bi * l + ti) * h + hi;
    let bc_idx = |bi: usize, ti: usize, hi: usize, ni: usize| ((bi * l + ti) * h + hi) * n + ni;

    // state[b,h,n,p]; init 0. Flat Vec — this is the CPU reference; state
    // lives outside the tensor graph on purpose.
    let mut state = vec![0f32; bs * h * n * p];
    let s_idx = |bi: usize, hi: usize, ni: usize, pi: usize| ((bi * h + hi) * n + ni) * p + pi;
    let mut y = vec![0f32; bs * l * h * p];

    // ── sequential scan over time ──────────────────────────────────────
    for bi in 0..bs {
        for ti in 0..l {
            for hi in 0..h {
                let dt_bth = dt_v[dt_idx(bi, ti, hi)];
                let da = (dt_bth * a_v[hi]).exp();
                // state <- dA * state + dt * B ⊗ x   (elementwise in (n,p))
                for ni in 0..n {
                    let dbx_factor = dt_bth * b_v[bc_idx(bi, ti, hi, ni)];
                    for pi in 0..p {
                        let s = &mut state[s_idx(bi, hi, ni, pi)];
                        *s = da * *s + dbx_factor * x_v[x_idx(bi, ti, hi, pi)];
                    }
                }
                // y = C · state (+ D·x)
                for pi in 0..p {
                    let mut acc = 0f32;
                    for ni in 0..n {
                        acc += c_v[bc_idx(bi, ti, hi, ni)] * state[s_idx(bi, hi, ni, pi)];
                    }
                    if let Some(dv) = &d_v {
                        acc += dv[hi] * x_v[x_idx(bi, ti, hi, pi)];
                    }
                    y[x_idx(bi, ti, hi, pi)] = acc;
                }
            }
        }
    }

    let out = Tensor::from_vec(y, (bs, l, h, p), &dev)?;
    Ok(out)
}

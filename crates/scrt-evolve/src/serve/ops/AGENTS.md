# serve/ops — native custom ops for track 39 Phase B

## PB-1: Mamba2 / SSD selective-scan CPU reference (`ssd.rs`)

The load-bearing kernel of the native Granite path. **Correctness is the bar;
speed is PB-4.** Implemented as a straight sequential scan over the time axis
in fp32, entirely on host, then repacked into a candle `Tensor`. It has no
matmul-fusion, no chunking, no CUDA — deliberately.

### Math implemented

For each batch `b`, timestep `t`, head `h`:

```
dA_{b,t,h}          = exp( dt_{b,t,h} * A_h )                    scalar
state_{b,t,h,n,p}   = dA * state_{b,t-1,h,n,p}
                    + dt_{b,t,h} * B_{b,t,h,n} * x_{b,t,h,p}
y_{b,t,h,p}         = Σ_n C_{b,t,h,n} * state_{b,t,h,n,p}
                    + D_h * x_{b,t,h,p}                          (if D given)
state_{b,-1,·}      = 0                                          (init)
```

That is: `y = C · state + D * x` where the internal `state ∈ R^{N×P}` per
`(b,h)` is a discrete-time linear system with per-step **input-dependent**
decay `dA` and input-dependent input matrix `dt * B * xᵀ`. This is the
"discretized diagonal-A" form used by Mamba2/SSD (see the SSD paper: Dao &
Gu, "Transformers are SSMs", Sec. 3 & 5) — one shared scalar `A` per head,
diagonal-in-`n`, so the recurrence is elementwise in `n` and `p`. `dt` is
expected to already be **positive** (caller applies `softplus(dt_raw +
dt_bias)`); we treat it as-is.

### Public signature

```rust
pub fn ssd_scan(
    x:  &Tensor,          // (B, L, H, P)  input, fp32-castable
    dt: &Tensor,          // (B, L, H)     positive step size (post-softplus)
    a:  &Tensor,          // (H,)          per-head A (typically negative)
    b:  &Tensor,          // (B, L, H, N)  input projection
    c:  &Tensor,          // (B, L, H, N)  output projection
    d:  Option<&Tensor>,  // (H,)          optional skip
) -> Result<Tensor, ScanError>;                // (B, L, H, P)
```

Returns a fresh f32 tensor on `x.device()`. Shape mismatches produce
`ScanError::Shape(..)` — **never panic**. Under-the-hood candle errors are
wrapped in `ScanError::Candle`.

### Not modeled here (deliberate scope)

- **Grouped B/C** (fewer B/C heads than `H`). Callers broadcast to `H` before
  calling; this keeps the reference indexing trivial. PB-2 wires the
  grouping.
- **Chunked / associative-scan / parallel-scan** forms. PB-4's CUDA path.
- **dt-bias + softplus.** Caller pre-applies; the scan sees positive `dt`.
- **Residuals / gated activations / conv1d** around the SSM. That's block
  wiring (PB-2, `arch/granite.rs`), not the kernel.

### Reference the CPU op is validated against

Analytic closed forms on constructed inputs (see `tests/ssd_scan.rs`):

1. `A=0, dt=B=C=x=1, D=0` ⇒ `state_t = t+1`, `y_t = t+1` (pure accumulator).
2. `A=-1, L=1, all-ones, D=0` ⇒ `y_0 = 1` (initial-step correctness).
3. `dt=1, B=0`, arbitrary `A` ⇒ state stays 0, `y = D·x` (D-skip
   correctness, decoupled from recurrence).
4. Small hand-computed two-step case with `A=-1`, verifying the exponential
   decay compounding across `t`.

Tolerance: `1e-6` on fp32 with hand-computed analytic values. That's the
"known-good reference" for PB-1. PB-3 (Granite parity vs Python
transformers) is the second, downstream check on the full forward.

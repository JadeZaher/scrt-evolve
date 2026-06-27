# Handoff — config-driven ambient branch evolution (2026-06-26)

A pass-off for the next session. Context is pruned to what the next pass needs.

## UPDATE — Goal 1/2/3 SHIPPED (stable-probe gate), sweep green, committed

Goals 1–3 below are **DONE** and the full sweep is green (`cargo test` +
`clippy -D warnings` + `fmt`; `test_shard.py` on `C:\scrt-cuda-venv`). What landed:

- **Goal 1 — stable probe (real cross-round gate):** `[eval].stable_probe`
  (serde-default false, additive). When set, `branch::create` REUSES the on-disk
  `probe.jsonl` instead of re-carving (carves once on round 1; later rounds load it
  and `ProbeSet::exclude_overlap` filters the fresh dataset so the probe is never
  trained on). `ModelVersion.probe_version` now records the exam each version was
  scored on; `commit()` takes it. `cmd_branch_evolve` rebuilds the baseline via
  `baseline_from_version(v, stable_pv)` — the SOLE hand-built `ScoreReport` — so the
  candidate (scored on the same fixed probe) and the stored baseline share
  `probe_version` ⇒ `classify` does a genuine Accept/Regress/Catastrophic. A
  pre-feature version (v1, no stored `probe_version`) is anchored to the on-disk
  probe's version, so **even round 2 is a real gate** (no manual migration).
  Tests: `verdict.rs` (6 covered/uncovered cases incl. probe-mismatch hard error),
  `model_store.rs` (`probe_version` round-trip), `eval.rs` (`exclude_overlap`).
  Live config `bench/branch-scrt-cli.toml` now sets `stable_probe = true`.
- **Goal 2 — cleanup:** removed `merge_adapter.py`, `run_ambient_round.sh`,
  `probe_layercall.py`, and the temp `*.log`s. KEPT the distill repro (`DISTILL.md`,
  `RESULTS.md`, `run_distill_branch.sh`, `convert_teacher_safetensors.py`,
  `seam_distill_tinyllama.py`, `probe_env.py` — the latter two are referenced by
  `RESULTS.md`). `.gitignore` already ignores `bench/**/*.log`; added `python/build/`
  + `*.egg-info/`.
- **Goal 3 — clean-code:** house patterns hold; `baseline_from_version` centralizes
  the baseline `ScoreReport`.

**Only remaining item = live GPU round-2 validation** (hardware-bound; not run here
because the GPU/teacher/venv round is interactive + minutes-long): run
`bench/scrt-cli-ambient.cmd` (or `branch evolve --name scrt-cli --config
bench/branch-scrt-cli.toml --python C:/scrt-cuda-venv/Scripts/python.exe`) twice and
confirm: round 2 commits v2 only on improvement; a forced regress rolls back (live
GGUF unchanged); the ring prunes to 2. The code path is unit-tested; this is the
end-to-end confirmation.

---
_Original handoff (pre-Goal-1) below for reference._

## STATUS UPDATE (2026-06-26, later pass)

- **GOAL 1 (stable probe) — DONE + green.** `[eval].stable_probe` (carve-once /
  reuse in `branch::create`), `ScoreReport.probe_version`, `ModelVersion.probe_version`
  round-trip, `verdict::classify` same-probe Accept/Regress/Catastrophic with a
  probe-mismatch hard error, `baseline_from_version` in `main.rs` (the sole
  hand-built baseline — Goal 3 centralization), and the `eval.rs` overlap test all
  shipped. `cargo test` exits 0. **Still owed: the live round-2 / forced-regress
  run on the CUDA box** (no GPU here) to confirm v2-commits-only-on-improvement +
  prune-to-2 end-to-end.
- **GOAL 2 (cleanup) — DONE.** Removed `merge_adapter.py`, `run_ambient_round.sh`,
  `probe_layercall.py`, `probe_env.py`, and the temp `*.log`s from
  `bench/seam_distill/`; added `bench/**/*.log` to `.gitignore`. Kept the distill
  repro (`DISTILL.md`, `RESULTS.md`, `run_distill_branch.sh`,
  `convert_teacher_safetensors.py`, `seam_distill_tinyllama.py`).
- **GOAL 3 (clean-code) — confirmed.** House patterns hold; baseline `ScoreReport`
  is centralized in `baseline_from_version`. Final sweep run at end of pass.
- **SDK consumption contract — documented.** `AGENTS.md` now codifies the
  SDK-owns-orchestration / hooks-injected / CLI-renders / `--json`-is-IPC pattern
  for the two consumers (in-process desktop client + shell-out), and names the one
  outlier to migrate (`branch evolve`, below).

## NEXT STEP (the SDK pattern resolution, deferred for GPU validation)

`branch evolve` is the one command whose orchestration still lives in
`cmd_branch_evolve` (`main.rs`) instead of the SDK. **Migrate** it to
`src/branch/evolve.rs` as `branch::evolve(cfg, name, &mut store, created, &hooks)
-> EvolveReport`, mirroring `branch::create`: the SDK owns resume-from-current-
adapter, `baseline_from_version`, the `create()` transaction, and commit+deploy;
the CLI keeps only hook wiring + render. Add an ML-free mock-hook test (KEEP commits
a version carrying `probe_version`; Regress doesn't commit). **Do this on the box
where the live GPU round can be re-run** — the path it reorganizes (trained-adapter
location vs store commit) can't be exercised ML-free, so land it with a real
`branch evolve` round, not just the unit test.

## Where we are (validated, green)

**Shipped + validated end-to-end this session:** a config-driven, self-describing
branch that **further-trains itself** on a cadence, eval-gated, with bounded
versioned weight storage and in-place reversible deploy. The full pipeline ran as
**one native command** (no WSL, no ad-hoc scripts):

```
scrt-evolve branch evolve --name scrt-cli --config bench/branch-scrt-cli.toml \
  --python C:/scrt-cuda-venv/Scripts/python.exe --steps 120
# → discover → generate(LM Studio teacher) → free-gpu → train(GPU, resumed 88
#   tensors = continue scrt-cli) → eval-gate(KEEP) → export Q4_K_M(668MB) →
#   commit v1 → deploy to LM Studio.  exit 0.
```

Full `cargo` sweep green (test + clippy `-D warnings` + fmt). The store ring holds
`v1` (current); `branch versions` / `branch rollback` work.

**Key pieces (all additive, ML-free-testable where it matters):**
- `[train.lora].init_adapter` / `trainer._resume_adapter_weights` — continue an adapter.
- `trainer.train()` now honors `--device` (was silently CPU-only — the GPU bug).
- `src/model_store.rs` `ModelStore` — bounded version ring (`[store]`: dir/keep_versions/deploy_to; commit/resolve/rollback/prune, atomic, schema-guarded; 5 tests).
- `branch evolve` / `branch versions` / `branch rollback` in `main.rs`; `persist_branch_config` + `EvolveConfig::to_toml` (self-describing branch.toml).
- `[hardware].free_gpu_command` (single-GPU teacher↔trainer VRAM handoff).
- `eval/verdict.rs::classify` — **uncovered baseline (`n==0`) ⇒ accept-unless-NaN** (the round-1 fix).
- `bench/scrt-cli-ambient.cmd` — one-line loop; `bench/branch-scrt-cli.toml` is the live config.

**Env (native, one process):** `C:\scrt-cuda-venv` (torch 2.5.1+cu121, transformers,
`scrt-evolve-ml` editable), `C:\llama.cpp` (source converter + `build\bin\llama-quantize.exe`),
teacher = LM Studio `meta-llama-3-8b-instruct`. `scrt-evolve doctor` green (the one
`python_pkg_dir` FAIL is a false negative — the package is installed in the venv).

## GOAL 1 — Stable probe (real cross-round keep|rollback)

**Problem.** `branch::create` re-carves the probe from each round's *fresh* generated
dataset, so round N's candidate and round N-1's stored baseline are scored on
*different* probes → `classify` would mismatch (or fall back to accept-unless-NaN).
Round 1 works only because the baseline is uncovered. **Round 2+ has no real gate.**

**Design (fixed probe per branch, carved once, reused):**
1. `[eval].stable_probe = true` (serde-default false; additive). When set, the eval
   stage REUSES an existing `eval::probe_path(cfg)` instead of re-carving; carve only
   if absent (first round). Keep the carve-each-round default for plain `branch create`.
2. Add `probe_version: Option<String>` to `ModelVersion` (model_store) — store the
   probe a version was scored on alongside its `correctness`.
3. In `cmd_branch_evolve`, build the baseline from the current version's stored
   `correctness` **and** `probe_version` (n>0) — not the hardcoded `"probe-branch"`.
   Then candidate (scored on the same stable probe) and baseline share `probe_version`
   → `classify` does real Accept/Regress/Catastrophic.
4. Honesty note: the fixed probe is held-out from round 1; later rounds generate
   different data so leakage is low — document it, don't over-engineer probe exclusion.

**Acceptance:** run round 2 → v2 commits only if it improves correctness on the fixed
probe; a forced regress rolls back (live GGUF unchanged); ring prunes to keep 2.
Add a `verdict.rs` unit test for the covered-baseline same-probe Accept/Regress paths
and a `model_store` test for `probe_version` round-trip.

## GOAL 2 — Cleanup / pruning (the ad-hoc artifacts)

These were validation one-offs, now SUPERSEDED by the config-driven path — remove:
- `bench/seam_distill/merge_adapter.py` — superseded by `--resume-adapter`.
- `bench/seam_distill/run_ambient_round.sh` — superseded by `branch evolve`.
- `bench/seam_distill/probe_layercall.py`, `probe_models.sh` — throwaway probes.
- Temp logs: `bench/seam_distill/*.log` (distill_run, ambient_round, native_evolve,
  convert, native_torch_install) — delete + add `bench/**/*.log` to `.gitignore`.

KEEP (real docs/repro): `DISTILL.md`, `RESULTS.md`, `run_distill_branch.sh`,
`convert_teacher_safetensors.py` (the distill feature's documented validation).

## GOAL 3 — Code design / clean-code enforcement (lite)

The new code already follows the house patterns; a light pass to confirm/tidy:
- **Additive config** (serde defaults, `Option`, `skip_serializing_if`) — keep absent ⇒ today's behavior.
- **Injected-hook cores** (`BranchHooks`/`DaemonHooks`) stay ML-free + deterministically testable; real subprocess wiring lives in `main.rs`.
- **Pure logic fns** unit-tested (`classify`, `decide_step`, `block_lr_scale`, `ModelStore::prune`).
- **Atomic writes** (tmp+rename) + **schema-versioned, guarded** manifests for on-disk state.
- Doc-comments say WHY. End every change with the sweep: `cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt --check` (+ `PYTHONPATH=python python3 python/tests/test_shard.py` for ML helpers).
- Tidy targets: drop any now-unused imports/helpers left by the evolve wiring; ensure `cmd_branch_evolve`'s baseline construction is the only place that hand-builds a `ScoreReport` (centralize once Goal 1 lands).

## GOAL 4 — Context pruning (what to carry vs drop)

- **Carry:** this file; `branch-factory-direction` memory (authoritative status);
  `bench/branch-scrt-cli.toml` (the live config); the env paths above; the Goal-1 design.
- **Drop:** the install-troubleshooting blow-by-blow, monitor-event noise, the WSL
  distill run play-by-play (it's in `DISTILL.md`), the failed-run debugging.
- **Authoritative sources** (read these, don't reconstruct from chat): `conductor/tracks.md`
  §Build status, `AGENTS.md`, `bench/seam_distill/DISTILL.md`, this handoff.

## First actions next pass
1. `cargo test && cargo clippy --all-targets -- -D warnings && cargo fmt --check` (confirm green).
2. **On the CUDA box:** run a real round 2 of `branch evolve scrt-cli` → confirm
   v2 commits only on improvement, a forced regress rolls back (live GGUF
   unchanged), ring prunes to 2. This is the only unvalidated part of Goal 1.
3. Migrate `branch evolve` → `branch::evolve` SDK fn (see NEXT STEP) and re-validate
   that same live round — land them together.
4. Commit (Goals 1+2 are done + green but uncommitted; the SDK migration is separate).

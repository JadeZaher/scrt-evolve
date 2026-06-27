# scrt-evolve — Critical DevUX + AIUX Review

Scope: the `scrt-evolve` CLI (`crates/scrt-evolve-cli/src/main.rs`), its config
schema (`crates/scrt-evolve/src/config.rs`), the printed reference
(`crates/scrt-evolve-cli/src/config_reference.rs`), the cross-language /
cross-repo contracts (`crates/scrt-evolve/src/dataset.rs`,
`crates/scrt-evolve/src/branch/manifest.rs`), the README, and the paired
steering skill (`skills/scrt-evolve/SKILL.md`).

Everything below is grounded in the actual code as of this review. Line numbers
refer to `main.rs` unless otherwise noted.

---

## Verdict — DevUX (human operator)

**Good bones, thin on the floor.** The `--help` text is genuinely excellent: long
help describes the pipeline, doc-comments on every subcommand/flag are dense and
honest, and `config-reference [--toml]` is a real asset — a queryable, annotated
schema most CLIs never ship. Error messages for *config-resolution* failures are
the best-in-class part of the tool: many `bail!`s name the exact fix (`set
[evolve].model_path`, `set [runtime].model_path or [export].out_path`). Defaults
are sane and partial configs are a deliberate, well-documented feature.

The friction is concentrated in three places: (1) the happy path is wide but
**undiscoverable as a sequence** — `train` defaults to the `candle` *fixture*
backend that "does not load real checkpoints," so the obvious first `train` run
silently does the wrong thing; (2) **subprocess failures leak raw Python
tracebacks** with no scrt-evolve-level "here's what to install / check" wrapper,
which is the single biggest real-world pain since the whole real-model path is
subprocess-driven; (3) there is **no `doctor` / preflight** to tell an operator
whether their Python venv, llama.cpp checkout, and model path are actually usable
*before* a long run.

## Verdict — AIUX (LLM agent driving the CLI)

**Drivable, but the agent must parse prose and will guess wrong on two real
forks.** The schema is fully introspectable (`config-reference`) and the
contracts (`dataset.jsonl`, `manifest.json`, `registry.json`) are genuinely
self-describing with versioning and doc-comments — strong AIUX foundations.
Exit codes are clean and uniform (`ExitCode::SUCCESS`/`FAILURE`, `main` at
L497-505). But **almost no command emits machine-readable output**: every status
line is `println!`-formatted prose (`"generate: 42 rows → ..."`,
`"eval: correctness=0.831 ..."`). An agent must regex human strings to learn what
happened, and there is no `--json` flag anywhere. The two places an agent will
actively misjudge: the **`train --backend candle` default** (an agent reading
`--help` sees `candle` is default and "real model" is `transformers`, but nothing
in the *output* warns it picked the fixture), and the **`Q4_K_M` quant sentinel**
in `export-gguf` (L1879: passing `--quant Q4_K_M` explicitly is indistinguishable
from the default, so the config value silently wins — surprising and unobservable).

---

## DevUX findings (ranked)

| # | Finding | Lens | Severity | Value/Effort | Concrete fix |
|---|---------|------|----------|--------------|--------------|
| D1 | `train` default backend is the **candle fixture** that "does not load real checkpoints" (L119 `default_value = "candle"`). A new operator's first `scrt-evolve train` runs a toy arch and prints a `final_loss` that looks real (`cmd_train`, L1396-1416). Nothing at runtime says "this was a fixture." | DevUX | **High** | High/Low | In `cmd_train` (L1403) prepend a one-line warning: `eprintln!("train(candle): FIXTURE backend — a tiny hand-built arch, NOT your model. For real training use --backend transformers.")`. Optionally flip the default to `transformers` and let candle be opt-in (bigger change; the warning is the cheap win). |
| D2 | **Subprocess failures surface raw Python tracebacks**, then a terse `bail!` (`"trainer exited with {status}"`, L1536; same for infer/score/export/dequant/run-model). An operator missing `torch`/`transformers` gets a `ModuleNotFoundError` wall with no scrt-evolve guidance. | DevUX | **High** | High/Med | In `run_subprocess_logged` (L1546) and each `if !status.success()` site, on failure append an actionable hint: `"the `{module}` subprocess failed — ensure the interpreter passed via --python (or `python` on PATH) has torch+transformers+safetensors; re-run with --python /path/to/venv/python. Full output above" + log path if set`. |
| D3 | **No preflight / `doctor` command.** Python interpreter, `python/` package dir (`find_python_pkg_dir`, L1953), llama.cpp checkout (`find_llama_gguf_py`, L1829), and `model_path` existence are only validated at the moment they're used, often deep into a run. | DevUX | **High** | High/Med | Add a `Doctor` subcommand that checks, and prints PASS/FAIL with fixes for: config parse, `[evolve].model_path` exists, `python/` dir located, `--python` interpreter imports torch/transformers, llama.cpp auto-detect resolves, work_dir writable. Reuse the existing finder helpers. |
| D4 | **The end-to-end happy path is not discoverable as one sequence.** `run` only does discover→generate(→export) (L742-750); there is no single doc/command that walks discover→generate→train→eval→export→infer. README shows fragments; `--help` long_about lists stages but not a runnable recipe. | DevUX | Med | High/Low | Add a `## Quickstart` block to README and a `long_about` line giving the literal 6-command sequence. Optionally extend `run` with `--train`/`--eval` flags (mirrors existing `--export`). |
| D5 | **`export-gguf` quant sentinel is surprising** (L1879): `if quant == "Q4_K_M"` treats the *explicit* flag value as "unset," so `--quant Q4_K_M` is silently overridden by `[export].quant`. An operator who passes the default-looking value to force it cannot. | DevUX | Med | Med/Low | Make `--quant` an `Option<String>` (clap), so "was it passed" is unambiguous; `None` ⇒ use config, `Some(q)` ⇒ use `q`. Same pattern fixes the silent-precedence surprise for an agent (see A2). |
| D6 | **`evolve` is overloaded into three different commands** by flags (`--goals`, `--schedule`, positional project; L751-779). The positional is required in one mode, optional in another, ignored in a third. Easy to invoke the wrong shape. | DevUX | Med | Med/Med | Either split into `evolve project <dir>`, `evolve goals`, `evolve schedule` subcommands, or at minimum add an example block to the `Evolve` doc-comment showing each of the three invocations explicitly. |
| D7 | **`init` warns about a missing `model_path` but the scaffold template ships a placeholder path** (`scaffold.rs` L89-95 parses the template back; the template's `model_path` is a `/path/to/...` placeholder that never exists, so the warning *always* fires). The warning is correct but un-actionable on first run — it fires even when the user has done nothing wrong yet. | DevUX | Low | Med/Low | Soften the wording to instruct rather than warn ("next: edit [evolve].model_path in {path} to point at your HF model dir"), or only warn when `model_path` is set to a non-placeholder, non-existent path. |
| D8 | **Config field `[eval].scorer_backend` defaults disagree between code and reference.** `config.rs` `default_scorer_backend()` returns `"api"` (L223-225); the reference doc shows `scorer_backend = "transformers"` (config_reference.rs L67). An operator copying the reference gets different behavior than the documented default implies. | DevUX | Med | High/Low | Make them agree: change the reference comment to `# api (no ML, default) | transformers (real forward pass)`, or change the default. Pick one source of truth. |
| D9 | **No way to see resolved config / dry-run.** Precedence is intricate (CLI flag > `[export]`/`[runtime]` > `[evolve]` > python default, scattered across cmd_* fns). An operator cannot ask "what will actually run?" | DevUX | Low | Med/Med | Add `config show` (prints the loaded `EvolveConfig` as TOML/JSON via the existing serde impls) and/or a `--dry-run` that prints the resolved subprocess command line instead of spawning it. |

## AIUX findings (ranked)

| # | Finding | Lens | Severity | Value/Effort | Concrete fix |
|---|---------|------|----------|--------------|--------------|
| A1 | **No structured output anywhere.** Every result is prose `println!` (`generate`: L828; `eval`: L1260-1273; `plan`: L924-943; `branch list`: L2076-2080; `route`: L2208-2211). An agent must regex human strings (incl. emoji-free but inconsistent formats) to extract counts, paths, correctness. Exit codes are clean but say nothing about *what* happened. | AIUX | **High** | High/Med | Add a global `--json` flag (or `--format json`). At minimum emit, for the artifact-producing commands, a single final JSON line: `{"command":"generate","rows":42,"out":"...","status":"ok"}`. `eval`/`plan`/`checkpoints`/`branch *` are the highest value. `serde_json` is already a dep and several reports already have serde impls. |
| A2 | **Silent flag-vs-config precedence the agent can't observe.** `export-gguf` quant sentinel (L1879) and the layered defaults in `cmd_export_gguf`/`cmd_run_model` mean the *effective* value differs from what the agent passed, with no echo of the resolved value in machine-readable form. The agent will assume its flag won. | AIUX | **High** | High/Low | Same fix as D5 (make overridable flags `Option`), plus echo resolved effective values in the A1 JSON summary (`"quant":"Q4_K_M","quant_source":"config"`). |
| A3 | **`train` candle-fixture default is a silent trap for agents.** An agent told "train the model" runs `train` (no backend flag), gets the fixture, sees a plausible `final_loss`, and reports success — having trained nothing real. The `--help` distinction (L104-110) is only visible if the agent reads long help, and the *output* never flags it. | AIUX | **High** | High/Low | The D1 warning to stderr is the floor; better, include `"backend":"candle","is_fixture":true` in the A1 JSON so an agent can branch on it programmatically. |
| A4 | **`interview` with no `--answer` prints questions and exits 0 without writing a directive** (L879-899). An agent that runs `interview` expecting a directive file gets success + no artifact, and the "(no --answer given; not writing directive.json)" signal is prose. | AIUX | Med | Med/Low | Emit the question set as JSON when `--json` (id/text/options/multi), and make the "nothing written" outcome an explicit machine-readable status (`{"status":"questions_only","wrote_directive":false}`) so the agent knows to supply `--answer`. |
| A5 | **Missing-artifact errors are inconsistent and only sometimes actionable.** `train(transformers)` says "run `generate` first" (L1435-1438, great); but `load_discovered` (L815) and `cmd_export` (L2316) just say "reading dataset {path}" with the underlying io error — no "run `generate` first." An agent can't reliably learn the remediation. | AIUX | Med | Med/Low | Standardize a "missing prerequisite artifact" error shape across `load_discovered`, `cmd_export`, `cmd_probe_build`, `cmd_train`: `"<cmd>: <artifact> not found at <path> — run `scrt-evolve <prereq>` first"`. Mirror the existing good message at L1435. |
| A6 | **No `--help`-equivalent machine manifest of commands.** An agent must scrape `--help` prose to learn subcommands/flags. The schema is introspectable (`config-reference`) but the *command surface* is not. | AIUX | Med | Med/Med | Add `scrt-evolve commands --json` (or document that clap's generated help is stable) listing subcommands, flags, defaults, and which artifact each consumes/produces. This is the command-surface analogue of `config-reference`. |
| A7 | **`branch list` / `route` correctness uses `f64::NAN`/prose** (L2074 `unwrap_or(f64::NAN)`, printed `{corr:.3}` ⇒ `"NaN"`). `NaN` in human output is ambiguous to an agent (vs `null`). | AIUX | Low | Med/Low | In the JSON path (A1) emit `null` for absent metrics, not `NaN`; keep `-`/`n/a` for the human path as elsewhere (cf. L1335 which already uses `"-"`). |
| A8 | **Contracts are self-describing but not *discoverable* from the CLI.** `dataset.jsonl` (dataset.rs: `#[serde(tag="kind")]`, 6 variants) and `manifest.json`/`registry.json` (manifest.rs: versioned, `MANIFEST_VERSION`, `REGISTRY_SCHEMA_VERSION`) are excellent, but an agent only learns the row schema by reading Rust source. | AIUX | Med | Med/Low | Extend `config-reference` (or add `dataset-reference`) to print the `GenExample` variants + required fields and the manifest schema — the same "queryable schema" treatment the config already gets. The doc-comments already exist; surface them. |
| A9 | **Skill omits the operational commands.** `SKILL.md` teaches an agent *what to stash* (goal-tagged curriculum) very well, but never shows how to actually run discover/generate/eval or read their output — so a driving agent knows the philosophy but not the verbs. It also doesn't mention `--json` would-be output or how to detect train-vs-fixture. | AIUX | Med | Med/Low | Add a short "Driving the CLI" section to `SKILL.md`: the canonical command sequence, that `train` needs `--backend transformers` for real models, and (post-A1) how to parse the JSON summary lines. |

---

## Top fixes to apply now

The six highest value/effort wins across both lenses. Each is specific enough to
implement directly.

1. **Warn loudly on the candle fixture (D1/A3).** In `cmd_train` (main.rs L1403),
   before printing the report, `eprintln!` that `--backend candle` is a FIXTURE
   that does not load real checkpoints and real training needs
   `--backend transformers`. Cheapest fix with the highest correctness payoff —
   it stops both humans and agents from silently "training" nothing.

2. **Wrap every subprocess failure with an actionable hint (D2/A5).** At each
   `if !status.success() { bail!(...) }` (L1536, L1648, L1718, L1758, L1819,
   L1945) and in `run_subprocess_logged` (L1546), append: the failed module, the
   `--python` remediation (interpreter needs torch+transformers+safetensors), and
   the log-file path when `SCRT_EVOLVE_LOG_FILE` is set. This is where real users
   actually get stuck.

3. **Add a global `--json` summary line (A1/A2/A7).** Emit one final JSON object
   for the artifact-producing commands (`generate`, `eval`, `plan`, `train`,
   `export-gguf`, `branch list/route/create`) with counts, resolved paths,
   effective resolved values (quant, backend, `is_fixture`), and `status`. Use
   `null` not `NaN` for absent metrics. `serde_json` is already a dependency;
   several report types already derive `Serialize`. This is the single biggest
   AIUX unlock.

4. **Make overridable flags `Option` to kill silent precedence (D5/A2).** Change
   `export-gguf --quant` (and audit `--out`, `--llama-cpp`) from
   `default_value`/sentinel (L1879 `if quant == "Q4_K_M"`) to `Option<String>`,
   so "passed vs not" is unambiguous: `None` ⇒ config default, `Some` ⇒ honor it.
   Removes a genuine "I passed Q4_K_M and it ignored me" footgun.

5. **Add a `doctor` preflight command (D3).** New subcommand that validates, with
   PASS/FAIL + fix-text: config parse, `[evolve].model_path` exists, `python/`
   dir located (`find_python_pkg_dir`), `--python` interpreter imports
   torch/transformers, llama.cpp auto-detect (`find_llama_gguf_py`) resolves, and
   work_dir writable. Reuses helpers that already exist; turns "long run dies at
   minute 9" into "told you in 2 seconds."

6. **Reconcile the docs and add a Quickstart (D4/D8/A8/A9).** (a) Fix the
   `scorer_backend` default mismatch between `config.rs` (`"api"`, L223) and the
   reference (`"transformers"`, config_reference.rs L67). (b) Add a literal
   6-command discover→generate→train→eval→export→infer Quickstart to the README
   and `long_about`. (c) Surface the `dataset.jsonl` + manifest schemas via
   `config-reference` (or a new `dataset-reference`). (d) Add a "Driving the CLI"
   section to `SKILL.md` with the command sequence and the candle-vs-transformers
   caveat. Low effort, high discoverability payoff for both humans and agents.

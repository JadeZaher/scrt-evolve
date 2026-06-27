# `crates/scrt-evolve/src` — module design notes

Directory-level design rationale for the SDK modules. **Prefer this file over
long inline comment blocks**: code carries terse one-line doc-comments (the
"what"); the "why" and the cross-module reasoning live here. Add a section when a
module's intent isn't obvious from its signatures.

## `ingest.rs` — interaction-log → training rows

Feeds the ambient daemon's living queue from real agent activity, **generically**
(no domain hardcoding — the same path serves CLI training, tool training, prose,
docs). Two cleanly split layers:

- **Parsing** (`interaction_log_rows`, `doc_completion_rows`) is pure, ML-free,
  deterministic — the testable surface. A Claude Code transcript distills into
  mixed rows: a `Bash` tool call → `Cli`; any other tool call → `ToolCall`
  (arguments minus the harness-only `description`); a prose-only assistant turn →
  `Qa`; a doc chunks into `Completion`. A tool-using turn emits only its tool
  row(s) — the surrounding prose there is reasoning, not an answer. Over-long
  payloads (heredocs, pasted files) are dropped; rows dedupe within a log.
- **Relevance** (`RelevanceJudge` / `LlmRelevanceJudge`) is an injected LLM step
  over any `ChatTransport`, so the SDK stays ML-free and the judge is unit-tested
  with a mock; the CLI wires the real chat endpoint. Relevance is a *model*
  decision against a free-text criterion, not a keyword rule — so ingestion works
  for any project. It batches, parses a JSON array of relevant item numbers, and
  **errs toward inclusion** (a failed/garbled batch keeps its rows) so a flaky
  endpoint degrades to "ingest more", never silent data loss — the eval gate is
  the real safety net.

Rows are stamped `gen = "ingest"` (`INGEST_GEN_STAMP`) so a bad ingest round can
be quarantined by that key, like a `trace:<goal>` harvest round.

The CLI layer (`daemon ingest` in `scrt-evolve-cli`) adds a cheap, generic
`--match` substring pre-filter (bounds the candidate set / LLM cost before any
call) and `--relevance` (the judge criterion). The intent prompt for a tool row
is the call's own `description` when present, else the recent user text.

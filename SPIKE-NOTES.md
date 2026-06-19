# scrt-evolve — Spike Notes (historical)

> **What this is.** These are the original research + spike notes that seeded
> **scrt-evolve**. They were written while the work still lived inside the
> `scrt-cli` repo, and they cover two threads that have since split apart:
>
> 1. **The embedding / self-evolution spike** — the corpus-distillation
>    thesis, the InfoNCE training seam, the model-path ergonomics. *This* is
>    what scrt-evolve grew out of; see [DESIGN.md](./DESIGN.md) for the
>    current, locked architecture (this doc is the prior art, not the plan).
> 2. **Lexical similarity / link discovery** (`--mp-similar`, the three
>    SimHash signals, link-as-you-stash) — this **shipped in `scrt-cli`** as a
>    first-class feature and is documented there. It's retained below only
>    because the research that produced it also informed the embedding design;
>    treat those sections as scrt-cli history.
>
> Kept verbatim for provenance. The authoritative current docs are
> scrt-evolve's [DESIGN.md](./DESIGN.md) / [README.md](./README.md) and
> scrt-cli's own README.

---

## Original spike status (as written in scrt-cli)

> **corpus export + training-loop seam complete behind a feature
> flag (ML opt-in); similarity retrieval BUILT and verified (`--mp-similar`,
> §B).** The embedding-adapter path is still a spike (training seam wired,
> not validated); the SimHash similarity path is a working, tested feature.

## Thesis

A directory of stashes built up during an agent's daily work is a
**curated retrieval signal**. Each stash is a search the agent chose to
keep: its **note** is a natural-language query, its captured **nodes** are
positives (the agent judged them relevant), and nodes from *other* stashes
are negatives. That's a ready-made contrastive dataset — no labeling, no
LLM. Distilling it into a per-agent embedding adapter would let the agent
retrieve from its own memory better over time: a self-directed,
post-deployment shaping loop, scoped per directory of work, on
unstructured data.

## What's built (and verified)

### 1. Ergonomics: feature-flagged, config-driven setup

The hard ergonomic requirement — *don't make every build carry an ML
stack* — is met with a feature flag:

- **Default build is ML-free.** `scrt-evolve`'s default features compile
  only `config` + `corpus` (no candle). Verified: a default
  `cargo test -p scrt-evolve` does **not** compile `candle-core`.
- **`--features train` opts into candle.** Verified to compile (candle
  0.8). The CLI exposes it as `cargo build -p scrt-cli --features
  evolve-train`.

The user provides **one thing** — a path to a raw local model — via
`.scrt/evolve.toml`, scaffolded by `scrt evolve init`:

```toml
[evolve]
model_path = "/models/nomic-embed-text"   # the one thing you provide
backend = "candle"                          # load raw weights in-process
epochs = 1
learning_rate = 2e-5
negatives_per_row = 4
# adapter_out defaults to .mpg/embeddings/<palace-id>.safetensors
```

`init` warns if the path doesn't exist yet, so config can be scaffolded
before the model is in place.

### 2. Corpus export (no ML) — verified end-to-end

`scrt evolve corpus` turns the palace into contrastive JSONL with zero ML
deps:

```json
{"query":"authentication rate limiting work",
 "positive_chunk":"function login() {}\n// TODO add auth rate limiting\n...",
 "negative_chunks":["...chunks from OTHER stashes..."],
 "stash":"auth"}
```

- `query` = the stash note. `positive` = one captured node's chunk.
  `negatives` = up to N chunks sampled deterministically from *other*
  stashes (index-strided, reproducible, no RNG dep).
- Stashes with an empty note are skipped (no query signal).
- Tested: row shape, note-as-query, negatives-exclude-own-stash,
  empty-note skipping, JSONL one-object-per-line.

### 3. Training loop — seam in place (feature-gated)

`scrt evolve train` (built with `evolve-train`) loads the corpus + the
user's model and runs a contrastive (InfoNCE) loop, saving the adapter to
`.mpg/embeddings/<palace-id>.safetensors`. The candle wiring lives behind
the flag; the **model-architecture-specific seam** (`load_model` /
`embed`) is where a concrete backbone (e.g. nomic-embed-text's BERT) plugs
in. Without the feature, `train` prints how to enable it and exits cleanly.

## What's pending

### A. `--retriever hybrid` (search-time blend)

The search flag that blends substring (rg) scores with a similarity score.
**Research-resolved (see §B): the answer is "all available signals", not
a single choice.** The hybrid retriever blends three layers, each lighting
up only when its dependency is present:

1. **substring/rg score** — always on, zero deps.
2. **MinHash-Jaccard score** — on once stash signatures are indexed
   (no model; the §B content-hashing path).
3. **Embedding cosine** — on once `scrt evolve train` has produced an
   adapter (this spike's §1–3 path).

So content hashing is **both its own feature** (`--mp-similar`) **and** a
hybrid input. Build order: §B's MinHash index + `--mp-similar` first
(unblocks signals 1+2 with no ML), then graft signal 3 when an adapter
exists.

### B. Similarity retrieval over stashes — **BUILT** (`--mp-similar`)

> Status: **implemented and verified.** `scrt --mp-similar <stash>` /
> `--term <text>` ranks stashes by SimHash similarity. 66 scrt-core tests
> pass (10 new for the similarity engine); end-to-end smoke test confirms
> correct semantic ranking. Crate: `scrt-core/src/palace/simhash.rs`.

#### What shipped

```
scrt --mp-similar <stash>   [--match note|full] [--score 1-10] [--top N]
scrt --term "<text>"        [--match note|full] [--score 1-10] [--top N]
```

- **Composite fingerprint** computed at stash time (the user's design): a
  stash's `<simhash(note)>-<simhash(note+body)>` id-form, surfaced in output.
- **`--score 1-10` reshapes the falloff** (rank-weight, never a cutoff): the
  full ranked list always returns; score tunes how steeply relevance decays
  with Hamming distance. 1 = wide net, 10 = near-identical only.
- **Byte-parity preserved**: fingerprints live in a scrt-only sidecar
  `.mpg/fingerprints.json`, NOT in the palace JSON — so the palace stays 100%
  Node-mpg-identical (COMPAT.md §4). Computed at stash time, lazily backfilled
  + reconciled on query, recomputable from scratch (it's a cache).

#### Random-projection vector — `--match vector` (the cheap "embedding")

A third similarity signal, **without a model**: the hashing-trick / random
projection. Each token sign-hashes into a fixed `RANDPROJ_DIM=128` float
vector; vectors are summed and L2-normalized so cosine = dot product. It's
**SimHash's cousin that keeps magnitude instead of collapsing to bits** — a
*weighted* lexical signal, smoother than Hamming. Zero deps, microseconds,
deterministic; stored as f32 per axis in the sidecar (`StashVector`,
back-compat — an older sidecar without it is upgraded on reconcile).

Use it with `scrt --mp-similar <stash> --match vector`. Same axis-selection as
`full` (typed sub-axis when dtypes match, else prose). Verified: lexically-
related stashes ~65% cosine, unrelated 0%.

**The honest limit, now asserted by a test** (`cosine_does_not_bridge_semantic_
gap`): this is still **lexical**. Two snippets sharing no tokens are near-
orthogonal regardless of meaning — "dog Rex" vs "my pet's name" stays ~0. It is
*not* a small free version of real embeddings; it's a *better lexical* signal.
Only a trained model (the `scrt-evolve` path) crosses the semantic gap. So the
three cheap signals — SimHash scalar, chunked best-pair/Jaccard, and now
random-projection cosine — are all lexical/structural; the embedding adapter
remains the one semantic tier, still gated on the model.

#### Link-as-you-stash — suggestions (the intuitive-linking feature)

The point of similarity here isn't ad-hoc querying — it's **building palaces
that link themselves intuitively**. So `--mp-stash` now runs the freshly-saved
stash against the rest of the palace and **suggests links**:

```
scrt: created stash "auth-test" (1 nodes, 15 tokens) at …
scrt: ~ related stashes (link suggestions):
   79%  auth-test2  [chunked]   scrt --mp-link auth-test auth-test2 see-also
   74%  auth-impl2  [chunked]   scrt --mp-link auth-test auth-impl2 see-also
   70%  auth-impl   [chunked]   scrt --mp-link auth-test auth-impl  see-also
```

- **Suggest, never auto-link.** The signal is lexical/structural (SimHash), so
  it emits advice + a ready `--mp-link` command — the human/agent decides. No
  `--auto-link` (a lexical signal must not create links confidently).
- **Already-linked stashes are excluded** (no nagging to re-link).
- **Opt out** with `--no-suggest-links`; **tune** with `--link-threshold
  <0-100>` (default 55, on the displayed relevance).
- This is the natural consumer for chunking: a fresh, substantial stash vs the
  whole palace is exactly the large-body partial-overlap case where best-pair
  beats the single hash.

**Score display was fixed to make the threshold meaningful.** Two changes:
1. **Gamma is display-decoupled.** `--score` now shapes only `rank_weight`
   (the sort key); the shown `relevance` is the honest method-normalized
   closeness, so a threshold compares against a stable number.
2. **Blend is additive, not averaged.** `best_pair + 0.25·jaccard` (capped),
   not `0.7·bp + 0.3·jac`. A weighted average punished the common small-palace
   case where Jaccard is structurally 0 — it dragged an 0.86 best-pair down to
   0.60 (→ 46% displayed, below threshold, so real matches were silently
   dropped). Additive keeps best-pair as the floor; Jaccard only adds dup
   confidence. After the fix: related stashes read 70–79%, unrelated 9%.

#### Chunked fingerprints — local similarity (the upgrade)

The first cut collapsed each stash into **one** whole-stash SimHash, which
destroys all similarity *within* a stash: a 200-line stash and the 5 lines
that match your query produce one hash dominated by the 195 irrelevant ones.
Fixed by **chunking**: the body is windowed (`CHUNK_WIDTH=12` features,
`CHUNK_STRIDE=6`, overlapping) and each window gets its own SimHash, so a
stash's fingerprint is now an **array** of per-window hashes (per axis).

Two stashes are then compared by **two metrics**, blended:

- **best-pair** (`0.7` weight) — for each query chunk, its closest candidate
  chunk; score = mean of the top-⅓. Rewards sharing *any* section (one
  matched function/paragraph pulls the score up even if the rest differs).
  This is the locality the single hash threw away.
- **MinHash set-Jaccard** (`0.3` weight) — a `MINHASH_K=16` signature over the
  chunk-hash set (chunks ARE the elements — no extra shingling). Measures
  overall overlap / near-duplication. On small palaces it's often ~0 (correct:
  distinct stashes), earning its keep on larger, genuinely-overlapping bodies.

Storage: chunk arrays live in the sidecar alongside the scalar fingerprint
(`StashSignature { scalar, chunks }`). An older sidecar with no `chunks` key
is detected and upgraded on read — back-compatible. The scalar `<note>-<full>`
id-form and the cheap single-XOR path are kept (note axis, term fallback).
Output shows the method per hit: `chunked via typed: best-pair 73% · jaccard
0%` vs the scalar `hamming 12/64 via note`.

Verified end-to-end: two code stashes sharing one function rank each other
above an unrelated stash, and the output names the shared-section score that
drove it. Known rough edge: the `--score` gamma falloff was tuned for scalar
Hamming closeness (which clusters high); blended chunk scores sit lower, so the
headline % reads pessimistically even when ordering is right. Re-tuning gamma
per-method is a noted follow-up.

#### Two design decisions forced by the implementation

Building it surfaced two things the abstract design missed; both are now
fixed and tested:

1. **Dual fingerprint (prose + typed), not one.** Per-type projection
   ("process on their axis") and cross-type comparison are mutually
   exclusive — you cannot Hamming-compare a code-projected SimHash to a
   prose-projected one; the distance is noise (the research's "distinct LSH
   families per metric" rule, hit in practice). So each stash stores BOTH a
   `full_prose` hash (universal — comparable to any stash and to raw terms)
   and a `full_typed` hash (same-`dtype` stashes only). The ranker picks the
   typed axis only when query and candidate share a type, else falls back to
   prose. A raw `--term` has no type, so it always uses prose — which is
   exactly what keeps it comparable to everything. Output tags each hit with
   the axis used (`via note|prose|typed`).

2. **Unigrams + n-grams, not n-grams alone.** Pure contiguous 3-shingles make
   short reordered notes look unrelated ("auth rate limiting" vs "rate
   limiting auth" share zero trigrams). Prose now blends word **unigrams**
   (shared vocabulary, order-independent) with 3-shingles (phrase signal);
   code blends token unigrams with 2-grams. This was caught by a failing
   ranking test, then fixed.

#### Per-type feature projection (as built)

| Type | Projection | Detect |
| :--- | :--- | :--- |
| prose | `w:`-tagged unigrams + word 3-shingles, stopword-stripped | default |
| code | `t:`-tagged token unigrams + 2-grams, idents normalized (digits→`#`) | ext / syntax density |
| JSON | sorted set of dotted key-paths (shape, not values) | `.json` / brace density |
| logs | timestamp/ip/number/hex → `<*>` templates, then 3-shingled | `.log` / level-prefix lines |

#### The limit, now measured (not just predicted)

SimHash on **very short note strings** (≤ ~5 words after stopwords) is
unreliable — too few features for the ±1 column votes to converge. The
unigram blend mitigates it; the body axis (`--match full`) helps more. But
the **semantic gap remains absolute**: "dog Rex" vs "my pet's name" share no
surface form and no SimHash variant bridges them. That boundary is still
where the trained-adapter embedding path (§1–3) earns its keep — confirming
the two are complementary, exactly as §A now blends them.

---

#### Original research synthesis (design rationale, retained)

A separate, related idea: make **similar stashes "sit next to each
other"** by hashing their content — different hash strategies per data
type (prose / code / JSON / logs), processed on their own axis. The goal:
given one stash, retrieve the ones "about the same thing" with **no
model, no LLM call** — cheap, deterministic, local.

#### What the research settles

The design space splits cleanly along the **distance metric** you want,
and that choice forces the LSH family:

| You want to measure | Metric | Hash family | What it is |
| :--- | :--- | :--- | :--- |
| Set overlap / near-duplicate | **Jaccard** | **MinHash + LSH banding** | k-shingle the content, keep min of each of N permutations; band the signature so similar sets collide. |
| Direction of a weighted feature vector | **Cosine** | **SimHash (Charikar)** | sum of ±randomly-signed feature hashes, take sign bits → compare by Hamming distance. |
| "Roughly the same document" (fuzzy) | tunable digest distance | **TLSH** | sliding-window tri-grams → Pearson-hashed into 128 buckets → 2-bit quartile encoding. |

Verified findings that pin the choice:

- **MinHash measures Jaccard; SimHash measures cosine.** That is *the*
  axis. They are not interchangeable — pick by what "similar" means for
  the data type (set-of-tokens overlap vs. weighted-vector direction).
- **MinHash beats SimHash for binary / set data** (presence-of-token),
  which is what short prose and code shingles reduce to.
- **TLSH is the strongest single fuzzy-hash**: AUC **0.9775**, vs
  ssdeep 0.6555 and sdhash 0.6855 in the cited comparison. Its distance
  threshold is tunable, so "about the same thing" becomes one knob.
- **LSH gives sub-linear query time with accuracy guarantees** — banding
  is what makes "find the neighbors of this stash" not an O(N²) scan.
- **Distinct LSH families per metric**: Hamming→bit-sampling,
  Angular/cosine→SimHash, Jaccard→MinHash. You cannot mix signatures
  across families and compare them directly.

#### The recommendation (Rust spike, scoped to small mixed-type snippets)

**1. Per-type feature projection, then ONE common signature family.**
The user's instinct — "different hash strategies per data type, process
on their axis" — is right at the *feature-extraction* stage, but the
signatures must land in a **single comparable space** or cross-type
similarity is undefined. So: vary the *shingling*, fix the *signature*.

| Type | Feature projection (the "axis") | Then |
| :--- | :--- | :--- |
| **prose / markdown** | lowercased word 3-shingles, stopword-stripped | → MinHash |
| **code** | token n-grams (lexer-split, identifiers normalized); AST-shape later if it earns its cost | → MinHash |
| **JSON** | sorted set of `key-path` strings (shape, not values) | → MinHash |
| **logs** | template extraction (strip timestamps / IPs / numeric ids to `<*>`), shingle the templates | → MinHash |

All four project to a **set of strings**, so a **single MinHash + LSH
band index over Jaccard** is the spine. This is the cheapest thing that
works across all four types and keeps one index, not four. Data type is
recorded as a tag; **same-type comparisons are weighted higher** at query
time (cross-type Jaccard is real but noisier).

**2. TLSH as a cheap secondary "is this basically a dup?" signal.** One
128-bit TLSH digest per stash, stored alongside the MinHash signature.
Use it only for the high-confidence near-dup verdict (one tunable
threshold) — it complements, doesn't replace, the MinHash neighbor query.

**3. On-disk ordering: defer the space-filling curve.** Hilbert/Z-order
ordering of signatures so neighbors get adjacent offsets is real prior
art, but for **tens-to-hundreds of stashes** it's premature — the whole
palace fits in memory and an LSH band lookup is already sub-linear. Keep
it as a noted post-v1 option for when a palace grows past ~10k stashes;
until then, ordering buys nothing measurable.

**4. The composite stash id is the fingerprint (user's design).** Instead
of bolting signatures on as a side field, **bake them into the stash id**
at write time, so identity *is* the similarity key. The id is two
SimHash segments joined by `-`:

```
<simhash(note)>-<simhash(note + body)>
       ▲                  ▲
   "what I was        "what I was looking for
    looking for"       AND what I found"
```

- **Left segment** = 64-bit SimHash of the **note/query string** alone.
  Matches on *intent* — "stashes I was looking for the same kind of thing
  with", independent of what they captured.
- **Right segment** = 64-bit SimHash of **note + body** together. Matches
  on *intent + content* — the fuller "about the same thing" signal.

Why SimHash *in the id* (not MinHash): SimHash is **fixed-width and
locality-preserving** — near input → near output in **Hamming distance**.
A 64-bit value reads like a real id, and similarity is one
`(a ^ b).count_ones()` (XOR + popcount). MinHash signatures are wide and
don't compress to an id cleanly. So:

> **SimHash lives in the id (cheap distance read); MinHash lives in the
> `sig` sidecar (quality ranking).** Two tiers, each playing to its
> strength.

**5. The 1–10 score reshapes the ranking falloff, never a hard cutoff.**
`--mp-similar` **always returns top-N ranked by distance** — it never
comes back empty. The `--score 1..10` flag tunes *how steeply* Hamming
distance penalizes a candidate, i.e. how fast relevance decays with
surface-form drift:

```
scrt --mp-similar <stash|--term "..."> [--top N] [--score 7]

score 1   shallow falloff → wide retrieval shape (loosely-related
          stashes still rank well)
score 10  steep falloff   → tight shape (only near-identical
          fingerprints stay near the top)
```

Because the query can be **either an existing stash OR a raw `--term`**,
the same SimHash is computed on the fly for a search string and compared
against every stash's left segment (intent match) or right segment
(intent+content match) — `--match note|full` selects which segment.

**6. Concrete crates / shape.** SimHash over `fxhash`/`ahash` feature
hashes (no crypto — speed matters, collision-attack resistance doesn't),
64-bit, compared by XOR+popcount. MinHash sidecar over the same hash
family with a hand-rolled band index (`HashMap<BandKey, Vec<StashId>>`)
for the quality re-rank of the top SimHash candidates. TLSH (`tlsh` crate
or ~150-line port) only for the near-dup verdict. Signatures computed at
**stash time**: the SimHash pair *becomes the id*, the MinHash goes in an
optional `sig` field (back-compatible — absent = recompute lazily, and
old plain-id stashes still load and just get lazily re-fingerprinted).

> **Open compatibility note:** stash ids are currently user-chosen names
> (`auth-flow-findings`). A SimHash-pair id is content-derived, so this
> is either (a) a *second* `fingerprint` field next to the human name —
> safer, keeps `--mp-get <name>` working — or (b) a real id change that
> needs a name→fingerprint lookup. **Recommend (a): keep the human name
> as the id, add `fingerprint: "<simhash>-<simhash>"` as a field.** Same
> two-segment design, but `--mp-get`/`--mp-link`/all existing
> name-addressed ops keep working unchanged. Flagged for your call before
> building.

#### Where it fails (state this plainly)

**The semantic gap is out of reach for every non-embedding hash.**
"dog Rex" and "my pet's name is Rex" share the token *Rex* and will
collide; **"dog Rex" and "my pet's name" share nothing lexical and NO
MinHash / SimHash / TLSH will ever bridge them.** These hashes measure
*surface form overlap*, not meaning. That is precisely the boundary
where the **trained-adapter embedding path (this spike's §1–3)** earns
its keep. So the two ideas are **complementary, not competing**:

- **content hashing** → free, deterministic, catches lexical/structural
  similarity and near-dups. Ship it standalone as `--mp-similar`.
- **embedding adapter** → paid (model + training), catches semantic
  similarity. The thing hashing structurally cannot do.

#### How this feeds `--retriever hybrid` (§A)

`--retriever hybrid` should blend **three** signals, each cheap to
compute and each covering the others' blind spot:

1. substring/rg score (exact, what scrt already does),
2. MinHash-Jaccard score against indexed stashes (lexical/structural),
3. *(when a trained adapter exists)* embedding cosine (semantic).

Signals 1–2 ship with zero model dependency; signal 3 lights up only
when `scrt evolve train` has produced an adapter. This is the answer to
the §A open question — the hybrid scores against **all available**
signals, and content hashing is **its own feature** (`--mp-similar`)
*and* a hybrid input, not one or the other.

### C. The measured result

The original spike goal: compare **palace + mimo + hybrid retriever** vs
**palace + mimo alone** on LongMemEval, looking for lift on
single-session-preference and multi-session (where pure substring fails).
This needs: the hybrid retriever (A), the LongMemEval dataset, and paid API
inference — a gated run deferred with the rest of the perf work. **Not yet run**; the
honest claim until then is that the corpus-distillation idea is *wired and
plausible*, not *validated*.

## Honest assessment so far

- **The ergonomics goal is met** — ML is opt-in, setup is one config file +
  one model path, corpus export works with no model at all.
- **The corpus-distillation idea is sound and cheap** — turning notes into
  queries and cross-stash nodes into negatives is a real contrastive
  signal with zero extra labeling.
- **The semantic lift is unproven.** Whether a 1-epoch adapter over a small
  per-palace corpus actually beats substring retrieval is exactly what the
  gated LongMemEval run would measure. If it's noise, that's a finding too
  — the corpus export and feature-flag plumbing remain useful regardless.

## Crate layout

```
crates/scrt-evolve/
  src/config.rs   .scrt/evolve.toml load/scaffold/validate (no ML)
  src/corpus.rs   palace -> {query, positive, negatives[]} JSONL (no ML)
  src/train.rs    candle InfoNCE loop  [cfg(feature = "train")]
  src/lib.rs      surface + default_adapter_path
```

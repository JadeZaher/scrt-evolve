#!/usr/bin/env python3
"""Benchmark two scrt-evolve datasets on CLI/tool-calling training value.

Usage: python benchmark.py <baseline.jsonl> <candidate.jsonl>
"""
import json, sys, re
from collections import Counter

# The 6 real scrt tools + their required params (ground truth from scrt-core).
SCHEMA = {
    "scrt_search": {"required": {"pattern"}, "props": {"pattern","in","cmd","url","effort","max_tokens","max_nodes","clip_chars","sort","window_curve","retriever","mp_from","mp_stash","mp_tag","mp_ttl","page","page_size"}},
    "scrt_stash": {"required": {"name","note"}, "props": {"name","note","tags","replace","ttl","palace_path"}},
    "scrt_list_stashes": {"required": set(), "props": {"tag_filter","palace_path"}},
    "scrt_get_stash": {"required": {"name"}, "props": {"name","with_nodes","palace_path"}},
    "scrt_drop_stash": {"required": {"name"}, "props": {"name","palace_path"}},
    "scrt_similar": {"required": set(), "props": {"name","term","match","score","top","palace_path"}},
}
# Real scrt CLI flags (from args.rs surface).
REAL_FLAGS = {"--in","--cmd","--url","--effort","--max-tokens","--max-nodes","--clip",
    "--sort","--window-curve","--format","--mp-stash","--mp-ttl","--mp-tag","--mp-from",
    "--mp-compose","--mp-intersect","--mp-except","--mp-graph","--mp-link","--mp-similar",
    "--mp-prune","--mp-prune-keep","--mp-prune-tag","--mp-prune-expired","--mp-list",
    "--mp-get","--mp-drop","--term","--match","--score","--top","--page","--page-size",
    "--all","--fuzzy","--json","--help","--version","--no-ignore","--hidden","--ignore-case"}

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def valid_tool_call(r):
    t = r.get("tool"); a = r.get("arguments")
    if t not in SCHEMA or not isinstance(a, dict): return False
    s = SCHEMA[t]
    if not s["required"] <= set(a.keys()): return False
    if not set(a.keys()) <= s["props"]: return False
    return True

def cli_flags(cmd):
    return {tok for tok in re.findall(r"--[a-z][a-z0-9-]*", cmd)}

def valid_cli(cmd):
    if not cmd.strip().startswith("scrt"): return False
    fl = cli_flags(cmd)
    if not fl: return True  # bare command like `scrt --help` handled below
    return fl <= REAL_FLAGS  # every flag must be real

def analyze(rows, label):
    kinds = Counter(r["kind"] for r in rows)
    total = len(rows)
    drives = kinds.get("tool_call",0)+kinds.get("cli",0)+kinds.get("instruction",0)
    # tool_call validity
    tcs = [r for r in rows if r["kind"]=="tool_call"]
    tc_valid = sum(1 for r in tcs if valid_tool_call(r))
    tools_covered = {r["tool"] for r in tcs if r.get("tool") in SCHEMA}
    # cli validity
    clis = [r for r in rows if r["kind"]=="cli"]
    cli_valid = sum(1 for r in clis if valid_cli(r.get("command","")))
    # distinctness
    sigs = set()
    for r in rows:
        sigs.add(json.dumps({k:v for k,v in r.items() if k not in ("source","gen")}, sort_keys=True))
    dup_rate = 1 - len(sigs)/total if total else 0

    print(f"\n=== {label} ===")
    print(f"  total rows         : {total}")
    print(f"  kinds              : {dict(kinds)}")
    print(f"  drives-scrt (tc+cli+instr): {drives} ({100*drives//total if total else 0}%)")
    print(f"  tool_call valid    : {tc_valid}/{len(tcs)} ({100*tc_valid//len(tcs) if tcs else 0}%)")
    print(f"  tools covered      : {len(tools_covered)}/6  {sorted(tools_covered)}")
    print(f"  cli valid (real flags): {cli_valid}/{len(clis)} ({100*cli_valid//len(clis) if clis else 0}%)")
    print(f"  duplicate rate     : {dup_rate:.1%}")
    return dict(total=total, drives=drives, tc=len(tcs), tc_valid=tc_valid,
               tools=len(tools_covered), clis=len(clis), cli_valid=cli_valid, dup=dup_rate)

if __name__ == "__main__":
    base = analyze(load(sys.argv[1]), f"BASELINE: {sys.argv[1]}")
    cand = analyze(load(sys.argv[2]), f"CANDIDATE: {sys.argv[2]}")
    print("\n=== DELTA (candidate - baseline) ===")
    def pct(n,d): return f"{100*n//d}%" if d else "n/a"
    print(f"  drives-scrt ratio  : {pct(base['drives'],base['total'])} -> {pct(cand['drives'],cand['total'])}")
    print(f"  tool_call validity : {pct(base['tc_valid'],base['tc'])} -> {pct(cand['tc_valid'],cand['tc'])}")
    print(f"  tools covered      : {base['tools']}/6 -> {cand['tools']}/6")
    print(f"  cli validity       : {pct(base['cli_valid'],base['clis'])} -> {pct(cand['cli_valid'],cand['clis'])}")
    print(f"  duplicate rate     : {base['dup']:.1%} -> {cand['dup']:.1%}")

#!/usr/bin/env python3
"""
harvest_claude_projects.py — adapt Claude Code transcripts → the generic
scrt-evolve TranscriptEntry JSONL the SDK harvester consumes (track 24 bench).

Claude Code stores sessions under ~/.claude/projects/<slug>/<uuid>.jsonl in its
NATIVE format (type/parentUuid/message{role,content:[blocks]}/attachments/
queue-operations/...). The SDK harvester (track 20 slice 4) expects the GENERIC
shape {role, text, tool?, command?}. This script is the bench-specific adapter:
it flattens CC transcripts into that generic shape so nothing CC-specific leaks
into the SDK.

Capture-then-filter discipline: we stream the (large) transcripts line by line
and emit only flattened message rows — we never load a whole 376MB tree into
memory. Per-session output goes to <out_dir>/<session>.jsonl, ready for
`scrt-evolve` harvesting (or direct distillation).

Usage:
  python harvest_claude_projects.py --projects <dir> --out <dir> [--max-chars N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _blocks_text(content) -> tuple[str, str | None]:
    """Flatten a CC message `content` (str or list of blocks) into (text, command?).

    - `text` blocks → joined prose.
    - `tool_use` blocks for Bash → surface the command (so CLI-shaped traces
      distill into Cli rows downstream).
    - `thinking`/`tool_result`/images → ignored for training signal.
    Returns (text, command_or_None).
    """
    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        return "", None
    texts: list[str] = []
    command: str | None = None
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            t = b.get("text", "")
            if t:
                texts.append(t)
        elif bt == "tool_use":
            name = b.get("name", "")
            inp = b.get("input", {})
            if name == "Bash" and isinstance(inp, dict) and inp.get("command"):
                command = inp["command"]
            # Other tool_use blocks: record the tool name as prose context.
            elif name:
                texts.append(f"[tool:{name}]")
    return "\n".join(texts).strip(), command


def adapt_session(path: Path, max_chars: int) -> list[dict]:
    """Convert one CC session .jsonl into generic TranscriptEntry dicts."""
    out: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") not in ("user", "assistant"):
                continue
            msg = o.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            text, command = _blocks_text(msg.get("content"))
            if not text and not command:
                continue
            if max_chars and len(text) > max_chars:
                text = text[:max_chars]
            entry = {"role": role, "text": text}
            if command:
                entry["command"] = command
            out.append(entry)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harvest_claude_projects")
    p.add_argument("--projects", required=True, help="~/.claude/projects dir.")
    p.add_argument("--out", required=True, help="Output dir for adapted JSONL.")
    p.add_argument("--max-chars", type=int, default=4000,
                   help="Truncate any single message to N chars (default 4000).")
    p.add_argument("--limit-sessions", type=int, default=0,
                   help="Process at most N sessions (0 = all). For smoke runs.")
    args = p.parse_args(argv)

    proj = Path(args.projects)
    if not proj.is_dir():
        print(f"ERROR: projects dir not found: {proj}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sessions = sorted(proj.rglob("*.jsonl"))
    if args.limit_sessions:
        sessions = sessions[: args.limit_sessions]

    total_entries = 0
    written = 0
    for s in sessions:
        entries = adapt_session(s, args.max_chars)
        if not entries:
            continue
        # Name the output by the session uuid (file stem), namespaced by parent
        # project dir to avoid collisions.
        name = f"{s.parent.name}__{s.stem}.jsonl"
        dest = out / name
        with dest.open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        total_entries += len(entries)
        written += 1

    summary = {
        "sessions_in": len(sessions),
        "sessions_written": written,
        "entries": total_entries,
        "out": str(out.resolve()),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

//! QA / instruction synthesis prompt templates.
//!
//! The backends are dumb executors; the prompt engineering lives here so it is
//! shared across the `local` and `api` backends. A template turns one context
//! passage into a system+user message pair instructing a teacher model to emit
//! a strict-JSON array of supervised examples.

use crate::generate::GenContext;

/// The system prompt: pins the teacher to emit ONLY a JSON array of examples in
/// the requested kinds, grounded in the passage.
pub fn system_prompt(kinds: &[String]) -> String {
    let kinds_list = kinds.join(", ");
    format!(
        "You are a meticulous dataset-generation assistant. Given a code or \
documentation passage, you produce supervised fine-tuning examples that teach a \
model to use the software described.\n\n\
Rules:\n\
- Output ONLY a JSON array. No prose, no markdown fences, no commentary.\n\
- Each array element is an object with a \"kind\" field, one of: {kinds_list}.\n\
- For kind \"qa\": fields are \"prompt\" (a natural question a user would ask) \
and \"completion\" (a correct, concise answer grounded in the passage).\n\
- For kind \"instruction\": fields are \"instruction\" (a task), optional \
\"input\" (context, may be empty string), and \"output\" (the correct result).\n\
- Ground every answer in the passage. Do NOT invent flags, commands, or behavior \
that are not supported by the passage. If the passage is too thin to ground an \
example, return fewer examples (an empty array is acceptable).\n\
- Prefer questions about HOW to accomplish a task with the tool, including exact \
command/flag usage when the passage shows it."
    )
}

/// The user prompt: the passage + how many examples to produce.
pub fn user_prompt(ctx: &GenContext) -> String {
    let kinds_list = ctx.kinds.join(", ");
    format!(
        "Produce up to {n} supervised examples (kinds: {kinds_list}) from the \
passage below. Return a JSON array only.\n\n\
Source: {source}\n\
Passage:\n\
```\n{passage}\n```",
        n = ctx.per_passage,
        source = ctx.passage.source,
        passage = ctx.passage.text,
    )
}

/// A refine/critique follow-up used when `turns > 1`: ask the teacher to fix
/// hallucinations and tighten answers, returning the same JSON-array shape.
pub fn refine_prompt() -> &'static str {
    "Review the JSON array you just produced. Remove any example whose answer is \
not directly supported by the passage, fix any inaccurate command/flag usage, and \
make answers concise. Return the corrected JSON array only — same schema, no prose."
}

/// System prompt for **tool-call** synthesis: the teacher must emit examples
/// whose answer is a structured call to one of the provided tools, grounded in
/// the real tool schemas (no invented tools, no invented parameters).
pub fn tool_call_system_prompt(tools_block: &str) -> String {
    format!(
        "You generate TOOL-CALLING fine-tuning examples. The assistant being \
trained must learn to answer a user request by calling one of these tools with \
correct arguments.\n\n\
Available tools (name — description, with parameter schema):\n{tools_block}\n\
Rules:\n\
- Output ONLY a JSON array. No prose, no markdown fences.\n\
- Each element is an object: {{\"kind\":\"tool_call\", \"prompt\": <a natural \
user request that should trigger a tool call>, \"tool\": <one of the tool names \
above, EXACTLY>, \"arguments\": <a JSON object of arguments>}}.\n\
- `tool` MUST be one of the listed names. `arguments` keys MUST be valid \
parameters from that tool's schema, and all REQUIRED parameters must be present.\n\
- Make the request realistic and specific (real stash names, real patterns, real \
flags), grounded in the passage where possible.\n\
- Do NOT invent tools or parameters that are not in the schema."
    )
}

/// User prompt for tool-call synthesis from a passage.
pub fn tool_call_user_prompt(ctx: &GenContext) -> String {
    format!(
        "From the passage below, produce up to {n} tool-call examples (JSON array \
only) that exercise the tools above. Prefer scenarios the passage describes.\n\n\
Source: {source}\n\
Passage:\n```\n{passage}\n```",
        n = ctx.per_passage,
        source = ctx.passage.source,
        passage = ctx.passage.text,
    )
}

/// System prompt for **CLI-command** synthesis: answer is a runnable `scrt …`
/// command line.
pub fn cli_system_prompt() -> String {
    "You generate CLI fine-tuning examples for the `scrt` command-line tool. The \
assistant being trained must learn to answer a user request with the exact \
runnable command.\n\n\
Rules:\n\
- Output ONLY a JSON array. No prose, no markdown fences.\n\
- Each element: {\"kind\":\"cli\", \"prompt\": <natural user request>, \
\"command\": <a single runnable `scrt …` command line>}.\n\
- The command MUST start with `scrt ` and use ONLY flags/subcommands that appear \
in the passage. Do not invent flags.\n\
- Keep commands realistic (real patterns, stash names, flag values)."
        .to_string()
}

/// User prompt for CLI-command synthesis from a passage.
pub fn cli_user_prompt(ctx: &GenContext) -> String {
    format!(
        "From the passage below, produce up to {n} CLI-command examples (JSON \
array only) using the flags/subcommands it documents.\n\n\
Source: {source}\n\
Passage:\n```\n{passage}\n```",
        n = ctx.per_passage,
        source = ctx.passage.source,
        passage = ctx.passage.text,
    )
}

/// System prompt for **skill-ingestion** synthesis (track 09): teach the model
/// to absorb a skill/capability and turn it into callable behavior.
pub fn skill_system_prompt() -> String {
    "You generate SKILL-INGESTION fine-tuning examples. The assistant being \
trained must learn to absorb a skill or capability described in the passage and \
invoke it correctly — knowing WHEN to use it, WITH WHAT inputs, and WHAT it \
produces.\n\n\
Rules:\n\
- Output ONLY a JSON array. No prose, no markdown fences.\n\
- Each element: {\"kind\":\"skill\", \"skill_name\": <the real skill/command/tool \
name from the passage>, \"prompt\": <a natural user request that should trigger \
the skill>, \"invocation\": <the structured call or runnable command that uses \
the skill>, \"expected_outcome\": <what success looks like>}.\n\
- `skill_name` and `invocation` MUST be grounded in the passage — do NOT invent \
skills, commands, flags, or parameters that are not described there.\n\
- Prefer realistic scenarios the passage actually supports. If the passage does \
not describe a usable skill, return fewer examples (an empty array is fine)."
        .to_string()
}

/// User prompt for skill-ingestion synthesis from a passage.
pub fn skill_user_prompt(ctx: &GenContext) -> String {
    format!(
        "From the passage below, produce up to {n} skill-ingestion examples (JSON \
array only) for the skill(s)/capability it documents.\n\n\
Source: {source}\n\
Passage:\n```\n{passage}\n```",
        n = ctx.per_passage,
        source = ctx.passage.source,
        passage = ctx.passage.text,
    )
}

/// System prompt for **reasoning-edit** synthesis (track 09): teach the model to
/// evolve a reasoning trace — correct a flawed chain-of-thought toward a better
/// final action, so the model learns to reason internally at inference.
pub fn reasoning_edit_system_prompt() -> String {
    "You generate REASONING-EDIT fine-tuning examples. The assistant being trained \
must learn to IMPROVE a chain of reasoning: given a task and a flawed reasoning \
trace, produce a corrected trace that leads to a better final action.\n\n\
Rules:\n\
- Output ONLY a JSON array. No prose, no markdown fences.\n\
- Each element: {\"kind\":\"reasoning_edit\", \"prompt\": <the task or question>, \
\"original_steps\": [<the flawed reasoning steps, as an array of strings>], \
\"edit_op\": <one of \"insert\",\"correct\",\"prune\",\"reorder\">, \
\"edited_steps\": [<the corrected reasoning steps>], \"final_action\": <the \
action/answer the corrected reasoning leads to>}.\n\
- The original_steps must contain a plausible but FLAWED reasoning path (a wrong \
assumption, a skipped check, a redundant or mis-ordered step), and edited_steps \
must FIX exactly that flaw via the named edit_op.\n\
- Ground the task and the final_action in the passage — do NOT invent commands, \
APIs, or facts the passage does not support. If the passage is not multi-step \
enough to support a reasoning trace, return fewer examples (empty array is fine)."
        .to_string()
}

/// User prompt for reasoning-edit synthesis from a passage.
pub fn reasoning_edit_user_prompt(ctx: &GenContext) -> String {
    format!(
        "From the passage below, produce up to {n} reasoning-edit examples (JSON \
array only) about multi-step workflows or chained tool use it describes.\n\n\
Source: {source}\n\
Passage:\n```\n{passage}\n```",
        n = ctx.per_passage,
        source = ctx.passage.source,
        passage = ctx.passage.text,
    )
}

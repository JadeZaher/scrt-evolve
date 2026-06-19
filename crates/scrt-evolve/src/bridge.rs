//! PyO3 bridge — the training-tooling interop seam (gated by `pyo3`).
//!
//! Exposes the dataset + a training-step seam to Python so conventional
//! tooling (`transformers`, `peft`, `trl`, `torch`) can consume scrt-evolve
//! datasets and drive presets. This is the merge point with the hivemind-models
//! Python training stack.
//!
//! ## The seam (track 04)
//!
//! - [`read_dataset`] loads a JSONL dataset and returns one dict per row
//!   (`kind` + the variant's fields) — this also closes track 02's
//!   carried-forward `read_dataset`.
//! - [`dataset_rows_for_training`] yields the *same* `(text, kind)` training
//!   pairs the Rust LoRA loop consumes (qa/instruction rows rendered to a
//!   single string with the prompt/completion boundary), so a Python
//!   `peft`/`trl` script trains on byte-identical rows. Parity by construction.
//!
//! The rendering here MUST match `train::lora::BatchIter` so the Rust loop and
//! the Python-driven loop consume the same rows (spec §Constraints).

#![cfg(feature = "pyo3")]

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dataset::{Dataset, GenExample};

/// Render a `qa`/`instruction` row to the `(prompt, completion)` pair the
/// trainer masks loss on. Mirrors `train::lora::BatchIter::new` exactly so the
/// Rust and Python loops see identical text. Non-trainable kinds yield `None`.
fn training_pair(row: &GenExample) -> Option<(String, String)> {
    match row {
        GenExample::Qa { prompt, completion, .. } => Some((prompt.clone(), completion.clone())),
        GenExample::Instruction { instruction, input, output, .. } => {
            let prompt = if input.is_empty() {
                format!("{instruction}\n")
            } else {
                format!("{instruction}\n{input}\n")
            };
            Some((prompt, output.clone()))
        }
        _ => None,
    }
}

/// Convert one dataset row to a Python dict (`kind` + the variant's fields).
fn row_to_dict<'py>(py: Python<'py>, row: &GenExample) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    match row {
        GenExample::Qa { prompt, completion, source, gen } => {
            d.set_item("kind", "qa")?;
            d.set_item("prompt", prompt)?;
            d.set_item("completion", completion)?;
            d.set_item("source", source.clone())?;
            d.set_item("gen", gen.clone())?;
        }
        GenExample::Instruction { instruction, input, output, source, gen } => {
            d.set_item("kind", "instruction")?;
            d.set_item("instruction", instruction)?;
            d.set_item("input", input)?;
            d.set_item("output", output)?;
            d.set_item("source", source.clone())?;
            d.set_item("gen", gen.clone())?;
        }
        GenExample::Completion { text, source } => {
            d.set_item("kind", "completion")?;
            d.set_item("text", text)?;
            d.set_item("source", source.clone())?;
        }
        GenExample::Contrastive { query, positive, negatives, stash } => {
            d.set_item("kind", "contrastive")?;
            d.set_item("query", query)?;
            d.set_item("positive", positive)?;
            d.set_item("negatives", negatives.clone())?;
            d.set_item("stash", stash.clone())?;
        }
        GenExample::ToolCall { prompt, tool, arguments, source, gen } => {
            d.set_item("kind", "tool_call")?;
            d.set_item("prompt", prompt)?;
            d.set_item("tool", tool)?;
            d.set_item("arguments", arguments.to_string())?;
            d.set_item("source", source.clone())?;
            d.set_item("gen", gen.clone())?;
        }
        GenExample::Cli { prompt, command, source, gen } => {
            d.set_item("kind", "cli")?;
            d.set_item("prompt", prompt)?;
            d.set_item("command", command)?;
            d.set_item("source", source.clone())?;
            d.set_item("gen", gen.clone())?;
        }
    }
    Ok(d)
}

/// Load a JSONL dataset and return one Python dict per row.
///
/// The dict always carries `kind`; remaining keys are the variant's fields.
/// This is the cross-language read side of the dataset contract (track 02's
/// carried-forward `read_dataset`).
#[pyfunction]
fn read_dataset(py: Python<'_>, path: String) -> PyResult<Vec<PyObject>> {
    let ds = Dataset::read_jsonl(&path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("read_dataset: {e}")))?;
    let mut out = Vec::with_capacity(ds.rows.len());
    for row in &ds.rows {
        out.push(row_to_dict(py, row)?.into());
    }
    Ok(out)
}

/// Return the `(text, kind)` training rows the Rust LoRA loop consumes.
///
/// Each qa/instruction row is rendered to a single training string
/// (`prompt + completion`) plus its `kind`, matching `train::lora::BatchIter`
/// byte-for-byte. A Python `peft`/`trl` script trains on the same rows.
#[pyfunction]
fn dataset_rows_for_training(path: String) -> PyResult<Vec<(String, String)>> {
    let ds = Dataset::read_jsonl(&path).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("dataset_rows_for_training: {e}"))
    })?;
    let mut out = Vec::new();
    for row in &ds.rows {
        let kind = match row {
            GenExample::Qa { .. } => "qa",
            GenExample::Instruction { .. } => "instruction",
            _ => continue,
        };
        if let Some((prompt, completion)) = training_pair(row) {
            out.push((format!("{prompt}{completion}"), kind.to_string()));
        }
    }
    Ok(out)
}

/// Return the `(prompt, completion)` pairs (prompt-masked boundary preserved)
/// for qa/instruction rows. Lets Python tooling mask loss on the completion
/// exactly as the Rust loop does.
#[pyfunction]
fn dataset_prompt_completion_pairs(path: String) -> PyResult<Vec<(String, String)>> {
    let ds = Dataset::read_jsonl(&path).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("dataset_prompt_completion_pairs: {e}"))
    })?;
    let mut out = Vec::new();
    for row in &ds.rows {
        if matches!(row, GenExample::Qa { .. } | GenExample::Instruction { .. }) {
            if let Some(pair) = training_pair(row) {
                out.push(pair);
            }
        }
    }
    Ok(out)
}

/// Count rows by `kind` — a cheap parity probe for Python-side tests.
#[pyfunction]
fn dataset_kind_counts(path: String) -> PyResult<HashMap<String, usize>> {
    let ds = Dataset::read_jsonl(&path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("dataset_kind_counts: {e}")))?;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for row in &ds.rows {
        let kind = match row {
            GenExample::Qa { .. } => "qa",
            GenExample::Instruction { .. } => "instruction",
            GenExample::Completion { .. } => "completion",
            GenExample::Contrastive { .. } => "contrastive",
            GenExample::ToolCall { .. } => "tool_call",
            GenExample::Cli { .. } => "cli",
        };
        *counts.entry(kind.to_string()).or_insert(0) += 1;
    }
    Ok(counts)
}

/// Placeholder returning the crate version — proves the module + the
/// `#[pymodule]` macro compile against Python headers under `--features pyo3`.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The `scrt_evolve` Python module — dataset read + training-step data seam.
#[pymodule]
fn scrt_evolve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(read_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_rows_for_training, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_prompt_completion_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_kind_counts, m)?)?;
    Ok(())
}

//! scrt tool schemas — the ground truth for tool-call generation.
//!
//! scrt-core defines its tools (`scrt_search`, `scrt_stash`, `scrt_similar`,
//! …) as function-calling descriptors via [`scrt_core::tool_spec`]. We pull
//! them in-process so generated `tool_call` rows are grounded in the *real*
//! tool names + parameter schemas, not invented ones.

use std::collections::BTreeMap;

/// One tool's name + parameter schema, distilled from the provider spec.
#[derive(Debug, Clone)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// The JSON-Schema `parameters` object (type/properties/required).
    pub parameters: serde_json::Value,
    /// Required parameter names, for convenience.
    pub required: Vec<String>,
    /// All parameter names, for convenience.
    pub properties: Vec<String>,
}

/// Load scrt's tool schemas (OpenAI shape, distilled). Returns them in the spec
/// order. Errors only if scrt-core's spec builder fails (shouldn't happen).
pub fn scrt_tools() -> anyhow::Result<Vec<ToolSchema>> {
    let spec = scrt_core::tool_spec::build_tool_spec("openai")
        .map_err(|e| anyhow::anyhow!("tool_spec: {e}"))?;
    let arr = spec
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("tool_spec: expected an array"))?;

    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let f = &entry["function"];
        let name = f["name"].as_str().unwrap_or_default().to_string();
        if name.is_empty() {
            continue;
        }
        let description = f["description"].as_str().unwrap_or_default().to_string();
        let parameters = f["parameters"].clone();
        let required = parameters["required"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let properties = parameters["properties"]
            .as_object()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        out.push(ToolSchema {
            name,
            description,
            parameters,
            required,
            properties,
        });
    }
    Ok(out)
}

/// A compact, model-readable description of all tools for a generation prompt:
/// name, description, required params, and the full parameter properties.
pub fn tools_prompt_block(tools: &[ToolSchema]) -> String {
    let mut s = String::new();
    for t in tools {
        s.push_str(&format!("- {} — {}\n", t.name, t.description));
        s.push_str(&format!(
            "  required: [{}]\n  parameters: {}\n",
            t.required.join(", "),
            serde_json::to_string(&t.parameters).unwrap_or_default()
        ));
    }
    s
}

/// A compact, low-token tool listing for the PLANNER/CRITIC (which only need to
/// know the tools exist + their key params, not the full JSON schemas). Keeps
/// the planning prompt within modest context windows.
pub fn tools_compact_block(tools: &[ToolSchema]) -> String {
    let mut s = String::new();
    for t in tools {
        let desc: String = t.description.chars().take(80).collect();
        s.push_str(&format!(
            "- {} (required: [{}]; params: [{}]) — {}\n",
            t.name,
            t.required.join(", "),
            t.properties.join(", "),
            desc
        ));
    }
    s
}

/// Index tools by name for validation lookups.
pub fn by_name(tools: &[ToolSchema]) -> BTreeMap<&str, &ToolSchema> {
    tools.iter().map(|t| (t.name.as_str(), t)).collect()
}

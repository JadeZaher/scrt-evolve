//! `ApiEndpoint` — an OpenAI-compatible chat-completions `GenBackend`.
//!
//! Configurable `base_url`, `model`, `api_key_env` (a var NAME — the key is
//! read from that env var, never inlined), and `turns` (multi-turn refine when
//! greater than 1). The HTTP transport is behind the [`ChatTransport`] trait so
//! tests can mock the model without a live endpoint.

use serde::{Deserialize, Serialize};

use crate::config::GenerateConfig;
use crate::dataset::GenExample;
use crate::generate::prompts;
use crate::generate::{GenBackend, GenContext, GenMode};
use crate::toolspec;

/// One chat message (OpenAI shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// The transport seam: send a chat request, get back the assistant's text.
/// Implemented by [`HttpTransport`] for real calls and by mocks in tests.
pub trait ChatTransport {
    fn complete(&self, messages: &[ChatMessage]) -> anyhow::Result<String>;
}

/// The configurable API generation backend.
pub struct ApiEndpoint<T: ChatTransport = HttpTransport> {
    transport: T,
    turns: usize,
}

impl ApiEndpoint<HttpTransport> {
    /// Build from `[generate]` config: resolves `base_url`, `model`, and reads
    /// the API key from the env var named by `api_key_env` (missing var is a
    /// clear error, not a panic). An empty/placeholder key is allowed for
    /// local endpoints (LM Studio, vLLM) that ignore auth.
    pub fn from_config(gcfg: &GenerateConfig) -> anyhow::Result<Self> {
        let api = gcfg.api.clone().ok_or_else(|| {
            anyhow::anyhow!("generate: backend=\"api\" needs a [generate.api] block")
        })?;
        let base_url = api
            .base_url
            .clone()
            .ok_or_else(|| anyhow::anyhow!("generate.api: `base_url` is required"))?;
        let model = api
            .model
            .clone()
            .ok_or_else(|| anyhow::anyhow!("generate.api: `model` is required"))?;

        // Read the key from the named env var. If `api_key_env` is set the var
        // MUST exist; if it's unset entirely, send no auth (local endpoints).
        let api_key = match &api.api_key_env {
            Some(var) => Some(std::env::var(var).map_err(|_| {
                anyhow::anyhow!(
                    "generate.api: env var `{var}` (api_key_env) is not set — \
                     export it or remove api_key_env for an unauthenticated \
                     local endpoint"
                )
            })?),
            None => None,
        };

        let turns = api.turns.max(1);
        Ok(Self {
            transport: HttpTransport {
                base_url,
                model,
                api_key,
            },
            turns,
        })
    }
}

impl<T: ChatTransport> ApiEndpoint<T> {
    /// Construct with an explicit transport (used by tests and by callers that
    /// want a custom transport).
    pub fn with_transport(transport: T, turns: usize) -> Self {
        Self {
            transport,
            turns: turns.max(1),
        }
    }

    /// Consume the endpoint, yielding its transport (so the planner/critic can
    /// reuse the same configured HTTP transport for non-generation calls).
    pub fn into_transport(self) -> T {
        self.transport
    }
}

impl<T: ChatTransport> GenBackend for ApiEndpoint<T> {
    fn generate(&self, ctx: &GenContext) -> anyhow::Result<Vec<GenExample>> {
        // Pick the system + user prompts for this mode. The framework ALWAYS
        // owns the output-format contract (the strict JSON-array schema +
        // validation rules); a planner-authored `custom_prompt` is layered on
        // top as extra content/strategy guidance — it cannot override the
        // format, only steer *what* to generate within it. This is what keeps
        // self-routing from breaking the parser.
        let (base_system, user) = match ctx.mode {
            GenMode::Prose => (prompts::system_prompt(ctx.kinds), prompts::user_prompt(ctx)),
            GenMode::ToolCall => (
                prompts::tool_call_system_prompt(&toolspec::tools_prompt_block(ctx.tools)),
                prompts::tool_call_user_prompt(ctx),
            ),
            GenMode::Cli => (prompts::cli_system_prompt(), prompts::cli_user_prompt(ctx)),
        };
        let system = match ctx.custom_prompt {
            Some(guidance) => format!(
                "{base_system}\n\n## Additional guidance for this batch (steers \
content, NOT format — the JSON-array schema above is mandatory):\n{guidance}"
            ),
            None => base_system,
        };

        let mut messages = vec![ChatMessage::system(system), ChatMessage::user(user)];

        let mut last = self.transport.complete(&messages)?;

        // Multi-turn refine: feed back the prior answer and ask to correct it.
        for _ in 1..self.turns {
            messages.push(ChatMessage::assistant(last.clone()));
            messages.push(ChatMessage::user(prompts::refine_prompt()));
            last = self.transport.complete(&messages)?;
        }

        parse_examples(&last, ctx)
    }
}

/// Parse the teacher's JSON-array response into [`GenExample`] rows, stamping
/// `source` (from the passage) and `gen` (provenance) where the variant carries
/// them. Tolerates a markdown code-fence wrapper around the array.
pub fn parse_examples(raw: &str, ctx: &GenContext) -> anyhow::Result<Vec<GenExample>> {
    let json = extract_json_array(raw);
    let values: Vec<serde_json::Value> = serde_json::from_str(json).map_err(|e| {
        anyhow::anyhow!("generate.api: response was not a JSON array of examples: {e}\nraw: {raw}")
    })?;

    let source = ctx.passage.source.clone();
    let provenance = "api".to_string();

    // For tool_call validation, index the real schemas by name.
    let tool_index = toolspec::by_name(ctx.tools);

    let mut out = Vec::new();
    for v in values {
        // Inject source/gen so the teacher doesn't have to (and can't get wrong).
        let mut obj = match v {
            serde_json::Value::Object(m) => m,
            _ => continue, // skip non-object array entries
        };
        // For tool_call/cli the teacher may omit `kind` (it knows the mode);
        // stamp it from the mode so the row deserializes.
        let kind = obj
            .get("kind")
            .and_then(|k| k.as_str())
            .map(String::from)
            .unwrap_or_else(|| default_kind_for(ctx.mode));
        obj.insert("kind".into(), serde_json::Value::String(kind.clone()));

        match kind.as_str() {
            "qa" | "instruction" | "tool_call" | "cli" => {
                obj.entry("source")
                    .or_insert_with(|| serde_json::Value::String(source.clone()));
                obj.entry("gen")
                    .or_insert_with(|| serde_json::Value::String(provenance.clone()));
            }
            "completion" => {
                obj.entry("source")
                    .or_insert_with(|| serde_json::Value::String(source.clone()));
            }
            _ => {}
        }

        // Validate tool_call rows against the real schema: known tool, all
        // required params present, no unknown params. Drop violators.
        if kind == "tool_call" && !valid_tool_call(&obj, &tool_index) {
            continue;
        }
        // CLI rows must actually be a `scrt ` command.
        if kind == "cli"
            && !obj
                .get("command")
                .and_then(|c| c.as_str())
                .map(|c| c.trim_start().starts_with("scrt"))
                .unwrap_or(false)
        {
            continue;
        }

        match serde_json::from_value::<GenExample>(serde_json::Value::Object(obj)) {
            Ok(ex) => out.push(ex),
            Err(_) => continue, // skip malformed individual rows, keep the rest
        }
    }
    Ok(out)
}

fn default_kind_for(mode: GenMode) -> String {
    match mode {
        GenMode::Prose => "qa".into(),
        GenMode::ToolCall => "tool_call".into(),
        GenMode::Cli => "cli".into(),
    }
}

/// Validate a candidate tool_call object against the real tool schemas: the
/// `tool` must be a known name, `arguments` must be an object, all required
/// params present, and no parameter outside the schema's property set.
fn valid_tool_call(
    obj: &serde_json::Map<String, serde_json::Value>,
    index: &std::collections::BTreeMap<&str, &toolspec::ToolSchema>,
) -> bool {
    let tool = match obj.get("tool").and_then(|t| t.as_str()) {
        Some(t) => t,
        None => return false,
    };
    let schema = match index.get(tool) {
        Some(s) => *s,
        None => return false, // hallucinated tool
    };
    let args = match obj.get("arguments").and_then(|a| a.as_object()) {
        Some(a) => a,
        None => return false,
    };
    // All required present.
    for req in &schema.required {
        if !args.contains_key(req) {
            return false;
        }
    }
    // No unknown params.
    for key in args.keys() {
        if !schema.properties.iter().any(|p| p == key) {
            return false;
        }
    }
    true
}

/// Pull the JSON array out of a model response that may be wrapped in a
/// ```json … ``` fence or have leading/trailing prose.
fn extract_json_array(raw: &str) -> &str {
    let trimmed = raw.trim();
    // Find the first '[' and the last ']' — the array body.
    if let (Some(start), Some(end)) = (trimmed.find('['), trimmed.rfind(']')) {
        if start <= end {
            return &trimmed[start..=end];
        }
    }
    trimmed
}

/// The real HTTP transport: blocking POST to `{base_url}/chat/completions`.
pub struct HttpTransport {
    base_url: String,
    model: String,
    api_key: Option<String>,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    temperature: f32,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

impl ChatTransport for HttpTransport {
    fn complete(&self, messages: &[ChatMessage]) -> anyhow::Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = ChatRequest {
            model: &self.model,
            messages,
            temperature: 0.3,
        };
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
        let mut req = client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                req = req.bearer_auth(key);
            }
        }
        let resp = req.send()?;
        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            anyhow::bail!("generate.api: {status} from {url}: {text}");
        }
        let parsed: ChatResponse = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("generate.api: bad response JSON: {e}\nbody: {text}"))?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("generate.api: response had no choices"))?;
        Ok(content)
    }
}

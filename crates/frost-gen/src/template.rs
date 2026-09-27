//! Mistral v13 chat template (from the pinned `chat_template.jinja`), rendered directly to
//! token ids so every control token is a deliberate id, never text the tokenizer matched.
//!
//!   <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]([AVAILABLE_TOOLS]{json}[/AVAILABLE_TOOLS])?
//!   ([INST]{user}[/INST]{assistant}</s> | [TOOL_RESULTS]{result}[/TOOL_RESULTS])*
//!
//! FROST always supplies a system prompt, so the template's built-in default ("Le Chat")
//! system message is never inserted. Consecutive same-role messages are merged with a
//! blank line (the upstream template raises on non-alternation; merging is our policy).

use crate::tokenizer::{ctl, Tokenizer, TokenizerError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role { User, Assistant, Tool }

/// A tool call the model emitted as `[TOOL_CALLS]name[ARGS]{json}`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall { pub name: String, pub arguments: String }

#[derive(Debug, Clone)]
pub struct Message { pub role: Role, pub content: String, pub tool_calls: Vec<ToolCall> }

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Message { Message { role, content: content.into(), tool_calls: Vec::new() } }
    pub fn user(content: impl Into<String>) -> Message { Message::new(Role::User, content) }
    pub fn assistant(content: impl Into<String>) -> Message { Message::new(Role::Assistant, content) }
    pub fn tool(content: impl Into<String>) -> Message { Message::new(Role::Tool, content) }
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Message { self.tool_calls = calls; self }
}

/// Render `system` + `messages` to prompt ids ending right after the last user/tool turn,
/// i.e. positioned for the assistant to generate. `tools_json` is emitted verbatim inside
/// [AVAILABLE_TOOLS] when present.
pub fn render(tok: &Tokenizer, system: &str, tools_json: Option<&str>, messages: &[Message]) -> Result<Vec<u32>, TokenizerError> {
    let mut ids = vec![ctl::BOS, ctl::SYSTEM_PROMPT];
    ids.extend(tok.encode(system)?);
    ids.push(ctl::SYSTEM_PROMPT_END);
    if let Some(t) = tools_json {
        ids.push(ctl::AVAILABLE_TOOLS);
        ids.extend(tok.encode(t)?);
        ids.push(ctl::AVAILABLE_TOOLS_END);
    }
    for m in merge_consecutive(messages) {
        match m.role {
            Role::User => { ids.push(ctl::INST); ids.extend(tok.encode(&m.content)?); ids.push(ctl::INST_END); }
            Role::Assistant => {
                ids.extend(tok.encode(&m.content)?);
                for c in &m.tool_calls {
                    ids.push(ctl::TOOL_CALLS); ids.extend(tok.encode(&c.name)?); ids.push(ctl::ARGS); ids.extend(tok.encode(&c.arguments)?);
                }
                ids.push(ctl::EOS);
            }
            Role::Tool => { ids.push(ctl::TOOL_RESULTS); ids.extend(tok.encode(&m.content)?); ids.push(ctl::TOOL_RESULTS_END); }
        }
    }
    Ok(ids)
}

fn merge_consecutive(messages: &[Message]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    for m in messages {
        match out.last_mut() {
            Some(prev) if prev.role == m.role && m.role != Role::Tool && prev.tool_calls.is_empty() && m.tool_calls.is_empty() => { prev.content.push_str("\n\n"); prev.content.push_str(&m.content); }
            _ => out.push(m.clone()),
        }
    }
    out
}

/// Split a generated id sequence into prose text and structured tool calls by control id.
/// Text never becomes a tool call: only the real [TOOL_CALLS]/[ARGS] ids delimit one.
pub fn parse_output(tok: &Tokenizer, ids: &[u32]) -> (String, Vec<ToolCall>) {
    let mut text_ids = Vec::new();
    let mut calls = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if ids[i] == ctl::TOOL_CALLS {
            let mut j = i + 1;
            let mut name_ids = Vec::new();
            while j < ids.len() && ids[j] != ctl::ARGS && ids[j] != ctl::EOS && ids[j] != ctl::TOOL_CALLS { name_ids.push(ids[j]); j += 1; }
            let mut arg_ids = Vec::new();
            if j < ids.len() && ids[j] == ctl::ARGS {
                j += 1;
                while j < ids.len() && ids[j] != ctl::EOS && ids[j] != ctl::TOOL_CALLS { arg_ids.push(ids[j]); j += 1; }
            }
            calls.push(ToolCall { name: tok.decode(&name_ids).trim().to_string(), arguments: tok.decode(&arg_ids).trim().to_string() });
            i = j;
            continue;
        }
        if ids[i] != ctl::EOS { text_ids.push(ids[i]); }
        i += 1;
    }
    (tok.decode(&text_ids), calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_consecutive_same_role_but_not_tools() {
        let m = vec![
            Message::user("a"), Message::user("b"), Message::tool("r1"), Message::tool("r2"),
        ];
        let out = merge_consecutive(&m);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].content, "a\n\nb");
    }
}

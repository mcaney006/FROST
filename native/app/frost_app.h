// FROST engine C ABI — implemented in Rust (crates/frost-chat/src/ffi.rs), consumed by the
// Objective-C AppKit shell. Every returned char* is UTF-8 JSON (or a bare id) owned by Rust:
// release it with frost_string_free. int32_t returns: 0 = ok, negative = failure, see
// frost_last_error. The event callback runs on a BACKGROUND thread; dispatch to main yourself.
#ifndef FROST_APP_H
#define FROST_APP_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct FrostEngine FrostEngine;
typedef void (*frost_event_cb)(void *ctx, const char *json_utf8);

// Lifecycle. data_dir may be NULL (defaults to ~/Library/Application Support/FROST).
// The model loads asynchronously; poll frost_engine_status_json or watch "status" events.
FrostEngine *frost_engine_new(const char *data_dir_utf8);
void         frost_engine_free(FrostEngine *e);            // cancels, stops the worker, frees the model
void         frost_engine_set_event_callback(FrostEngine *e, frost_event_cb cb, void *ctx);

// Status: {"state":"loading"|"ready"|"error","detail"?,"generating":bool,"mode":"quiet"|"balanced"|"performance",
//          "thermal":"Nominal"|"Fair"|"Serious"|"Critical","memory_pressure":"Normal"|"Warn"|"Critical",
//          "available_memory_bytes":int|null,"mlx_active_bytes","mlx_peak_bytes","kv_cache_bytes",|"ready"|"error","detail"?,"generating":bool,"mode":"quiet"|"balanced"|"performance",
//          "thermal":"Nominal"|"Fair"|"Serious"|"Critical","mlx_active_bytes","mlx_peak_bytes","kv_cache_bytes",
//          "context_budget":4096,"reserved_output_tokens":1024,"model_loaded":bool,"idle_unload_seconds":600}
// model_loaded=false with state "ready" means the weights were released (idle or memory pressure) and reload on the next send.
char *frost_engine_status_json(FrostEngine *e);
// {"generator":{repo,revision,architecture,quantization,layers,...,backend,weight_bytes}|null,
//  "retrieval":{model,revision,role},"sampler_backend":"mojo-dylib"|"unavailable"}
char *frost_model_info_json(FrostEngine *e);

// Conversations: [{"id","title","created_at","updated_at","mode","system_prompt","repo_path"|null}]
char   *frost_conversations_json(FrostEngine *e);
char   *frost_conversation_create(FrostEngine *e);        // returns the new id
int32_t frost_conversation_delete(FrostEngine *e, const char *id);
int32_t frost_conversation_rename(FrostEngine *e, const char *id, const char *title);
// Messages: [{"id","conversation_id","seq","role":"system"|"user"|"assistant"|"tool","content","created_at","meta":{...}}]
// role "system" rows are "context cleared" markers (meta.marker == "context_cleared").
// assistant meta: {"status":"generating"|"done"|"error","finish":"eos"|"max_tokens"|"cancelled"|"thermal_deferred:<State>"|"memory_pressure_deferred",
//   "generator":{...},"sampler","prompt_tokens","new_tokens","tokens_per_second","context_truncated_messages",
//   "tools_available":bool,"tool_calls":[{name,arguments}],"citations":[{n,path,start_line,end_line,digest,status}],...}
char *frost_messages_json(FrostEngine *e, const char *conv_id);

// Actions (0 ok; negative: loading / busy / empty / unavailable — read frost_last_error).
int32_t frost_send(FrostEngine *e, const char *conv_id, const char *text_utf8);
int32_t frost_regenerate(FrostEngine *e, const char *conv_id);   // drops the last assistant turn, generates again
int32_t frost_clear_context(FrostEngine *e, const char *conv_id); // model forgets earlier turns; transcript stays
int32_t frost_cancel(FrostEngine *e);
int32_t frost_set_mode(FrostEngine *e, const char *mode);         // "quiet" | "balanced" | "performance"
int32_t frost_set_repo(FrostEngine *e, const char *conv_id, const char *path_or_null);

// Coding-agent attempts (patches / commands the model proposed). Each waits for the user:
//   [{"id","conversation_id","message_id","kind":"propose_diff"|"run_command","payload":{...},
//     "status":"proposed"|"approved"|"denied"|"applied"|"passed"|"failed"|"timeout"|"cancelled",
//     "exit_code","stdout","stderr","duration_ms","created_at","updated_at"}]
//   payload (propose_diff): {"summary","diff","files":[...],"base_hashes":{path:sha256},"validated":{...},"requested_permission"}
//   payload (run_command): {"argv":[...],"cwd","purpose","timeout_s","max_output_bytes","requested_permission"}
char   *frost_attempts_json(FrostEngine *e, const char *conv_id);
int32_t frost_attempt_decide(FrostEngine *e, const char *conv_id, const char *attempt_id, int32_t approve); // 1 approve, 0 deny

char *frost_last_error(FrostEngine *e);
void  frost_string_free(char *s);

// Events (JSON objects, field "type"):
//   {"type":"status","status":{"state":...,"detail"?}}
//   {"type":"conversations_changed"}
//   {"type":"messages_changed","conv_id"}
//   {"type":"message_started","conv_id","message_id"}
//   {"type":"delta","conv_id","message_id","text"}          // coalesced (~25 Hz max)
//   {"type":"message_done","conv_id","message_id","content","meta"}
//   {"type":"note","conv_id","text"}                         // context truncated, thermal pause, ...
//   {"type":"error","conv_id"?,"detail"}
//   {"type":"attempt_proposed","conv_id","attempt":{...}}  // show a review card with Approve / Deny
//   {"type":"attempt_updated","conv_id","attempt":{...}}   // status changed (approved/denied/applied/passed/failed/...)

#ifdef __cplusplus
}
#endif
#endif

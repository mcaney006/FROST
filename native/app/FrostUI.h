// Internal declarations shared by the AppKit shell sources. The public ABI is frost_app.h.
#import <Cocoa/Cocoa.h>
#import "frost_app.h"

NS_ASSUME_NONNULL_BEGIN

/// Attribute set on every character of a rendered code block; the value is the block's raw
/// code (NSString), so Copy needs no side table.
extern NSString *const FrostCodeBlockAttribute;

/// Read-only transcript view: adds "Copy Code Block" to the context menu over code.
@interface FrostTranscriptTextView : NSTextView
@end

/// Owns the transcript text storage: per-message ranges, Markdown-subset rendering, streaming
/// updates that re-render only the message that changed.
@interface FrostTranscript : NSObject
- (instancetype)initWithTextView:(NSTextView *)textView;
@property (nonatomic) BOOL wrapCode;
/// Shown above the messages (model error detail). nil hides it.
@property (nonatomic, copy, nullable) NSString *bannerText;
/// Message dicts as returned by frost_messages_json: id, role, content, meta.
- (void)setMessages:(NSArray<NSDictionary *> *)messages;
- (void)appendMessage:(NSDictionary *)message;
/// Re-renders one message in place. Returns NO when the id is not in the transcript.
- (BOOL)updateMessageId:(NSString *)messageId content:(NSString *)content meta:(nullable NSDictionary *)meta;
/// Re-renders an item by id, or appends it. Items are messages, or attempts (frost_attempts_json
/// rows with "role":"attempt") which render as a review card with Approve / Deny.
- (void)upsertItem:(NSDictionary *)item;
/// Invoked by the Approve / Deny buttons on an attempt card.
@property (nonatomic, copy, nullable) void (^decideAttempt)(NSString *convId, NSString *attemptId, BOOL approve);
@end

/// Entry point called from Rust. Returns when the app terminates.
int frost_ui_run(FrostEngine *engine, int argc, const char *_Nullable *_Nullable argv);

NS_ASSUME_NONNULL_END

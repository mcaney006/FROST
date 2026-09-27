// Transcript rendering: native attributed text (TextKit 2), per-message ranges, a Markdown
// subset (fenced code with a real Copy button, inline code, **bold**, bullets, headings, diff
// colouring), and streaming updates that touch only the changed message's range.
#import "FrostUI.h"
#pragma clang diagnostic ignored "-Wunused-parameter"

NSString *const FrostCodeBlockAttribute = @"FrostCodeBlock";

static NSFont *BodyFont(void) { return [NSFont systemFontOfSize:13]; }
static NSFont *MonoFont(void) { return [NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular]; }
static NSColor *CodeBackground(void) { return NSColor.quaternarySystemFillColor; }
static NSAttributedString *Str(NSString *s, NSDictionary *attrs) { return [[NSAttributedString alloc] initWithString:s attributes:attrs]; }
static NSCharacterSet *WS(void) { return NSCharacterSet.whitespaceCharacterSet; }
static NSString *S(id v) { return [v isKindOfClass:NSString.class] ? v : nil; } // JSON string or nil
static NSString *Tail(NSString *s, NSUInteger lines) {
    NSArray *all = [s componentsSeparatedByString:@"\n"];
    if (all.count <= lines) return s;
    return [@"…\n" stringByAppendingString:[[all subarrayWithRange:NSMakeRange(all.count - lines, lines)] componentsJoinedByString:@"\n"]];
}

#pragma mark - Real NSButtons hosted inside the text (Copy, Approve, Deny)

@interface FrostButtonAttachment : NSTextAttachment
@property (nonatomic, copy) NSString *buttonTitle, *buttonIdentifier;
@property (nonatomic) NSControlSize buttonSize;
@property (nonatomic, copy) void (^onPress)(void);
@end

@interface FrostButtonProvider : NSTextAttachmentViewProvider
@end

@implementation FrostButtonProvider
- (void)loadView {
    FrostButtonAttachment *a = (FrostButtonAttachment *)self.textAttachment;
    NSButton *b = [NSButton buttonWithTitle:a.buttonTitle ?: @"" target:self action:@selector(press:)];
    b.bezelStyle = NSBezelStyleRounded;
    b.controlSize = a.buttonSize;
    b.font = [NSFont systemFontOfSize:[NSFont systemFontSizeForControlSize:a.buttonSize]];
    b.accessibilityLabel = a.buttonTitle;
    b.accessibilityIdentifier = a.buttonIdentifier;
    [b sizeToFit];
    self.view = b;
}
- (CGRect)attachmentBoundsForAttributes:(NSDictionary<NSAttributedStringKey, id> *)attributes location:(id<NSTextLocation>)location textContainer:(NSTextContainer *)textContainer proposedLineFragment:(CGRect)proposedLineFragment position:(CGPoint)position {
    NSSize s = self.view.frame.size;
    return CGRectMake(0, -s.height * 0.3, s.width, s.height);
}
- (void)press:(id)sender {
    void (^f)(void) = ((FrostButtonAttachment *)self.textAttachment).onPress;
    if (f) f();
}
@end

@implementation FrostButtonAttachment
- (NSTextAttachmentViewProvider *)viewProviderForParentView:(NSView *)parentView location:(id<NSTextLocation>)location textContainer:(NSTextContainer *)textContainer {
    return [[FrostButtonProvider alloc] initWithTextAttachment:self parentView:parentView textLayoutManager:textContainer.textLayoutManager location:location];
}
@end

/// One attachment character hosting a button; `attrs` (paragraph style etc.) apply to that character.
static NSAttributedString *Button(NSString *title, NSString *identifier, NSControlSize size, NSDictionary *attrs, void (^onPress)(void)) {
    FrostButtonAttachment *att = [FrostButtonAttachment new];
    att.buttonTitle = title;
    att.buttonIdentifier = identifier;
    att.buttonSize = size;
    att.onPress = onPress;
    NSMutableAttributedString *s = [[NSAttributedString attributedStringWithAttachment:att] mutableCopy];
    [s addAttributes:attrs range:NSMakeRange(0, s.length)];
    return s;
}

#pragma mark - Full-width background behind code paragraphs

@interface FrostCodeFragment : NSTextLayoutFragment
@end

@implementation FrostCodeFragment
- (void)drawAtPoint:(CGPoint)point inContext:(CGContextRef)ctx {
    CGRect f = self.layoutFragmentFrame;
    CGFloat w = self.textLayoutManager.textContainer.size.width;
    if (w <= 0 || w > 100000) w = f.size.width;
    CGContextSaveGState(ctx);
    CGContextSetFillColorWithColor(ctx, CodeBackground().CGColor);
    CGContextFillRect(ctx, CGRectMake(point.x, point.y, w, f.size.height));
    CGContextRestoreGState(ctx);
    [super drawAtPoint:point inContext:ctx];
}
@end

#pragma mark - Text view with "Copy Code Block" in the context menu

@implementation FrostTranscriptTextView
- (NSMenu *)menuForEvent:(NSEvent *)event {
    NSMenu *menu = [[super menuForEvent:event] copy] ?: [[NSMenu alloc] initWithTitle:@""];
    NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    NSUInteger idx = [self characterIndexForInsertionAtPoint:p];
    if (idx < self.textStorage.length) {
        NSString *code = [self.textStorage attribute:FrostCodeBlockAttribute atIndex:idx effectiveRange:NULL];
        if (code) {
            NSMenuItem *it = [[NSMenuItem alloc] initWithTitle:@"Copy Code Block" action:@selector(frostCopyCodeBlock:) keyEquivalent:@""];
            it.target = self;
            it.representedObject = code;
            [menu insertItem:it atIndex:0];
            [menu insertItem:NSMenuItem.separatorItem atIndex:1];
        }
    }
    return menu;
}
- (void)frostCopyCodeBlock:(NSMenuItem *)item {
    NSPasteboard *pb = NSPasteboard.generalPasteboard;
    [pb clearContents];
    [pb setString:item.representedObject ?: @"" forType:NSPasteboardTypeString];
}
@end

#pragma mark - Markdown subset

/// Inline `code` and **bold** over a base attribute set.
static NSAttributedString *Inline(NSString *s, NSDictionary *base) {
    NSMutableAttributedString *out = [NSMutableAttributedString new];
    NSMutableDictionary *bold = [base mutableCopy];
    bold[NSFontAttributeName] = [NSFontManager.sharedFontManager convertFont:base[NSFontAttributeName] toHaveTrait:NSBoldFontMask];
    NSMutableDictionary *code = [base mutableCopy];
    code[NSFontAttributeName] = MonoFont();
    code[NSBackgroundColorAttributeName] = CodeBackground();
    NSUInteger i = 0, n = s.length;
    while (i < n) {
        unichar c = [s characterAtIndex:i];
        if (c == '`') {
            NSRange end = [s rangeOfString:@"`" options:0 range:NSMakeRange(i + 1, n - i - 1)];
            if (end.location != NSNotFound) {
                [out appendAttributedString:Str([s substringWithRange:NSMakeRange(i + 1, end.location - i - 1)], code)];
                i = end.location + 1;
                continue;
            }
        } else if (c == '*' && i + 1 < n && [s characterAtIndex:i + 1] == '*') {
            NSRange end = [s rangeOfString:@"**" options:0 range:NSMakeRange(i + 2, n - i - 2)];
            if (end.location != NSNotFound && end.location > i + 2) {
                [out appendAttributedString:Str([s substringWithRange:NSMakeRange(i + 2, end.location - i - 2)], bold)];
                i = end.location + 2;
                continue;
            }
        }
        NSUInteger j = i + 1;
        while (j < n) {
            unichar d = [s characterAtIndex:j];
            if (d == '`' || d == '*') break;
            j++;
        }
        [out appendAttributedString:Str([s substringWithRange:NSMakeRange(i, j - i)], base)];
        i = j;
    }
    return out;
}

@interface FrostTranscript () <NSTextLayoutManagerDelegate>
@property (nonatomic, strong) NSTextView *textView;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *items;
@property (nonatomic, strong) NSMutableArray<NSValue *> *ranges; // parallel to messages
@property (nonatomic) NSUInteger bannerLength;
@end

@implementation FrostTranscript

- (instancetype)initWithTextView:(NSTextView *)textView {
    if ((self = [super init])) {
        _textView = textView;
        _items = [NSMutableArray new];
        _ranges = [NSMutableArray new];
        _wrapCode = YES;
        textView.textLayoutManager.delegate = self;
    }
    return self;
}

- (NSTextLayoutFragment *)textLayoutManager:(NSTextLayoutManager *)tlm textLayoutFragmentForLocation:(id<NSTextLocation>)location inTextElement:(NSTextElement *)element {
    if ([element isKindOfClass:NSTextParagraph.class]) {
        NSAttributedString *s = ((NSTextParagraph *)element).attributedString;
        if (s.length && [s attribute:FrostCodeBlockAttribute atIndex:0 effectiveRange:NULL]) {
            return [[FrostCodeFragment alloc] initWithTextElement:element range:element.elementRange];
        }
    }
    return [[NSTextLayoutFragment alloc] initWithTextElement:element range:element.elementRange];
}

#pragma mark Scrolling

- (BOOL)isAtBottom {
    NSScrollView *sv = self.textView.enclosingScrollView;
    return NSMaxY(sv.documentVisibleRect) >= NSMaxY(self.textView.bounds) - 40;
}
- (void)scrollToBottom { [self.textView scrollToEndOfDocument:nil]; }

#pragma mark Public updates

- (void)setBannerText:(NSString *)bannerText {
    if (bannerText == _bannerText || [bannerText isEqualToString:_bannerText]) return;
    _bannerText = [bannerText copy];
    [self setMessages:self.items];
}
- (void)setWrapCode:(BOOL)wrapCode {
    _wrapCode = wrapCode;
    [self setMessages:self.items];
}

- (void)setMessages:(NSArray<NSDictionary *> *)messages {
    self.items = [messages mutableCopy];
    [self.ranges removeAllObjects];
    NSMutableAttributedString *all = [NSMutableAttributedString new];
    if (self.bannerText.length) {
        NSMutableParagraphStyle *ps = [NSMutableParagraphStyle new];
        ps.paragraphSpacing = 10;
        [all appendAttributedString:Str([NSString stringWithFormat:@"Model unavailable: %@\n", self.bannerText],
                                        @{NSFontAttributeName: BodyFont(), NSForegroundColorAttributeName: NSColor.systemRedColor, NSParagraphStyleAttributeName: ps})];
    }
    self.bannerLength = all.length;
    for (NSDictionary *m in self.items) {
        NSAttributedString *a = [self render:m];
        [self.ranges addObject:[NSValue valueWithRange:NSMakeRange(all.length, a.length)]];
        [all appendAttributedString:a];
    }
    [self.textView.textStorage setAttributedString:all];
    [self scrollToBottom];
}

- (void)appendMessage:(NSDictionary *)message {
    BOOL bottom = [self isAtBottom];
    NSAttributedString *a = [self render:message];
    NSTextStorage *ts = self.textView.textStorage;
    [self.items addObject:message];
    [self.ranges addObject:[NSValue valueWithRange:NSMakeRange(ts.length, a.length)]];
    [ts appendAttributedString:a];
    if (bottom) [self scrollToBottom];
}

- (NSUInteger)indexOfItemId:(NSString *)itemId {
    return [self.items indexOfObjectPassingTest:^BOOL(NSDictionary *m, NSUInteger idx, BOOL *stop) { return [m[@"id"] isEqual:itemId]; }];
}

- (BOOL)updateMessageId:(NSString *)messageId content:(NSString *)content meta:(NSDictionary *)meta {
    NSUInteger i = [self indexOfItemId:messageId];
    if (i == NSNotFound) return NO;
    NSMutableDictionary *m = [self.items[i] mutableCopy];
    m[@"content"] = content ?: @"";
    if (meta) m[@"meta"] = meta;
    [self replaceItemAtIndex:i with:m];
    return YES;
}

- (void)upsertItem:(NSDictionary *)item {
    NSUInteger i = [self indexOfItemId:item[@"id"] ?: @""];
    if (i == NSNotFound) [self appendMessage:item];
    else [self replaceItemAtIndex:i with:item];
}

/// Re-renders one item and shifts the ranges after it; TextKit only re-lays-out the edited range.
- (void)replaceItemAtIndex:(NSUInteger)i with:(NSDictionary *)m {
    self.items[i] = m;
    BOOL bottom = [self isAtBottom];
    NSAttributedString *a = [self render:m];
    NSRange old = self.ranges[i].rangeValue;
    NSTextStorage *ts = self.textView.textStorage;
    [ts beginEditing];
    [ts replaceCharactersInRange:old withAttributedString:a];
    [ts endEditing];
    NSInteger delta = (NSInteger)a.length - (NSInteger)old.length;
    self.ranges[i] = [NSValue valueWithRange:NSMakeRange(old.location, a.length)];
    for (NSUInteger j = i + 1; j < self.ranges.count; j++) {
        NSRange r = self.ranges[j].rangeValue;
        self.ranges[j] = [NSValue valueWithRange:NSMakeRange((NSUInteger)((NSInteger)r.location + delta), r.length)];
    }
    if (bottom) [self scrollToBottom];
}

#pragma mark Rendering

- (NSDictionary *)gapAttrs {
    return @{NSFontAttributeName: [NSFont systemFontOfSize:6], NSParagraphStyleAttributeName: [NSMutableParagraphStyle new]};
}

- (NSAttributedString *)render:(NSDictionary *)m {
    NSString *role = [m[@"role"] isKindOfClass:NSString.class] ? m[@"role"] : @"assistant";
    NSString *content = [m[@"content"] isKindOfClass:NSString.class] ? m[@"content"] : @"";
    NSDictionary *meta = [m[@"meta"] isKindOfClass:NSDictionary.class] ? m[@"meta"] : @{};
    NSMutableAttributedString *out = [NSMutableAttributedString new];

    if ([role isEqualToString:@"attempt"]) return [self renderAttempt:m];
    if ([role isEqualToString:@"system"]) {
        NSMutableParagraphStyle *ps = [NSMutableParagraphStyle new];
        ps.alignment = NSTextAlignmentCenter;
        ps.paragraphSpacingBefore = 14;
        ps.paragraphSpacing = 6;
        [out appendAttributedString:Str(@"— context cleared —\n", @{NSFontAttributeName: [NSFont systemFontOfSize:11], NSForegroundColorAttributeName: NSColor.secondaryLabelColor, NSParagraphStyleAttributeName: ps})];
        return out;
    }

    NSString *name = [role isEqualToString:@"user"] ? @"You" : [role isEqualToString:@"tool"] ? @"Tool" : @"FROST";
    NSMutableParagraphStyle *hps = [NSMutableParagraphStyle new];
    hps.paragraphSpacingBefore = 16;
    hps.paragraphSpacing = 3;
    [out appendAttributedString:Str([name.localizedUppercaseString stringByAppendingString:@"\n"],
                                    @{NSFontAttributeName: [NSFont systemFontOfSize:10.5 weight:NSFontWeightBold], NSForegroundColorAttributeName: NSColor.secondaryLabelColor,
                                      NSKernAttributeName: @1.0, NSParagraphStyleAttributeName: hps})];

    NSString *status = [meta[@"status"] isKindOfClass:NSString.class] ? meta[@"status"] : @"";
    NSMutableAttributedString *body = [[self renderMarkdown:content] mutableCopy];
    if ([status isEqualToString:@"generating"]) {
        NSAttributedString *caret = Str(@"▍", @{NSFontAttributeName: BodyFont(), NSForegroundColorAttributeName: NSColor.secondaryLabelColor});
        if (body.length && [body.string hasSuffix:@"\n"]) [body insertAttributedString:caret atIndex:body.length - 1];
        else { [body appendAttributedString:caret]; [body appendAttributedString:Str(@"\n", @{NSFontAttributeName: BodyFont()})]; }
    }
    [out appendAttributedString:body];

    NSMutableParagraphStyle *fps = [NSMutableParagraphStyle new];
    fps.paragraphSpacingBefore = 2;
    NSDictionary *small = @{NSFontAttributeName: [NSFont systemFontOfSize:11], NSForegroundColorAttributeName: NSColor.secondaryLabelColor, NSParagraphStyleAttributeName: fps};
    if ([status isEqualToString:@"error"]) {
        NSString *detail = [meta[@"detail"] isKindOfClass:NSString.class] ? meta[@"detail"] : @"generation failed";
        NSMutableDictionary *red = [small mutableCopy];
        red[NSForegroundColorAttributeName] = NSColor.systemRedColor;
        [out appendAttributedString:Str([detail stringByAppendingString:@"\n"], red)];
    } else if ([status isEqualToString:@"done"]) {
        NSMutableArray *parts = [NSMutableArray new];
        if ([meta[@"finish"] isKindOfClass:NSString.class]) [parts addObject:meta[@"finish"]];
        if ([meta[@"tokens_per_second"] isKindOfClass:NSNumber.class]) [parts addObject:[NSString stringWithFormat:@"%.1f tok/s", [meta[@"tokens_per_second"] doubleValue]]];
        if ([meta[@"new_tokens"] isKindOfClass:NSNumber.class]) [parts addObject:[NSString stringWithFormat:@"%@ tokens", meta[@"new_tokens"]]];
        if ([meta[@"context_truncated_messages"] isKindOfClass:NSNumber.class] && [meta[@"context_truncated_messages"] integerValue] > 0)
            [parts addObject:[NSString stringWithFormat:@"%@ message(s) truncated from context", meta[@"context_truncated_messages"]]];
        if (parts.count) [out appendAttributedString:Str([[parts componentsJoinedByString:@" · "] stringByAppendingString:@"\n"], small)];
    }
    if (!out.length || ![out.string hasSuffix:@"\n"]) [out appendAttributedString:Str(@"\n", @{NSFontAttributeName: BodyFont()})];
    return out;
}

/// Review card for a coding-agent attempt (a frost_attempts_json row): what the model proposed,
/// Approve / Deny while it waits, and the outcome (status, exit code, output tails) afterwards.
- (NSAttributedString *)renderAttempt:(NSDictionary *)a {
    NSString *kind = S(a[@"kind"]) ?: @"attempt";
    NSString *status = S(a[@"status"]) ?: @"proposed";
    NSDictionary *p = [a[@"payload"] isKindOfClass:NSDictionary.class] ? a[@"payload"] : @{};
    NSMutableAttributedString *out = [NSMutableAttributedString new];
    NSColor *statusColor = [@[@"passed", @"applied", @"approved"] containsObject:status] ? NSColor.systemGreenColor
                         : [@[@"failed", @"timeout", @"denied", @"cancelled"] containsObject:status] ? NSColor.systemRedColor
                         : NSColor.systemOrangeColor;
    NSMutableParagraphStyle *hps = [NSMutableParagraphStyle new];
    hps.paragraphSpacingBefore = 16;
    hps.paragraphSpacing = 3;
    NSDictionary *h = @{NSFontAttributeName: [NSFont systemFontOfSize:10.5 weight:NSFontWeightBold], NSForegroundColorAttributeName: NSColor.secondaryLabelColor,
                        NSKernAttributeName: @1.0, NSParagraphStyleAttributeName: hps};
    NSMutableDictionary *hs = [h mutableCopy];
    hs[NSForegroundColorAttributeName] = statusColor;
    [out appendAttributedString:Str([NSString stringWithFormat:@"ATTEMPT · %@ · ", [kind stringByReplacingOccurrencesOfString:@"_" withString:@" "].uppercaseString], h)];
    [out appendAttributedString:Str([status.uppercaseString stringByAppendingString:@"\n"], hs)];

    NSMutableParagraphStyle *ps = [NSMutableParagraphStyle new];
    ps.paragraphSpacing = 3;
    NSDictionary *body = @{NSFontAttributeName: BodyFont(), NSForegroundColorAttributeName: NSColor.textColor, NSParagraphStyleAttributeName: ps};
    NSDictionary *small = @{NSFontAttributeName: [NSFont systemFontOfSize:11], NSForegroundColorAttributeName: NSColor.secondaryLabelColor, NSParagraphStyleAttributeName: ps};
    if ([kind isEqualToString:@"propose_diff"]) {
        if (S(p[@"summary"]).length) [out appendAttributedString:Str([S(p[@"summary"]) stringByAppendingString:@"\n"], body)];
        NSArray *files = [p[@"files"] isKindOfClass:NSArray.class] ? p[@"files"] : @[];
        if (files.count) [out appendAttributedString:Str([NSString stringWithFormat:@"Files: %@\n", [files componentsJoinedByString:@", "]], small)];
        if (S(p[@"diff"]).length) [out appendAttributedString:[self renderCode:[S(p[@"diff"]) componentsSeparatedByString:@"\n"] lang:@"diff"]];
    } else if ([kind isEqualToString:@"run_command"]) {
        NSArray *argv = [p[@"argv"] isKindOfClass:NSArray.class] ? p[@"argv"] : @[];
        [out appendAttributedString:[self renderCode:@[[@"$ " stringByAppendingString:[argv componentsJoinedByString:@" "]]] lang:@"command"]];
        if (S(p[@"purpose"]).length) [out appendAttributedString:Str([S(p[@"purpose"]) stringByAppendingString:@"\n"], body)];
        if (S(p[@"cwd"]).length) [out appendAttributedString:Str([NSString stringWithFormat:@"cwd: %@\n", S(p[@"cwd"])], small)];
    }
    if (S(p[@"requested_permission"]).length) [out appendAttributedString:Str([NSString stringWithFormat:@"Permission requested: %@\n", S(p[@"requested_permission"])], body)];

    if ([status isEqualToString:@"proposed"]) {
        NSString *convId = S(a[@"conversation_id"]) ?: @"", *attemptId = S(a[@"id"]) ?: @"";
        __weak typeof(self) weakSelf = self;
        void (^decide)(BOOL) = ^(BOOL approve) {
            void (^f)(NSString *, NSString *, BOOL) = weakSelf.decideAttempt;
            if (f) f(convId, attemptId, approve);
        };
        NSMutableParagraphStyle *bps = [ps mutableCopy];
        bps.paragraphSpacingBefore = 4;
        NSDictionary *battrs = @{NSParagraphStyleAttributeName: bps, NSFontAttributeName: BodyFont()};
        [out appendAttributedString:Button(@"Approve", @"frost.attempt.approve", NSControlSizeSmall, battrs, ^{ decide(YES); })];
        [out appendAttributedString:Str(@"  ", battrs)];
        [out appendAttributedString:Button(@"Deny", @"frost.attempt.deny", NSControlSizeSmall, battrs, ^{ decide(NO); })];
        [out appendAttributedString:Str(@"\n", battrs)];
    } else {
        NSMutableString *line = [status mutableCopy];
        if ([a[@"exit_code"] isKindOfClass:NSNumber.class]) [line appendFormat:@" (exit %@)", a[@"exit_code"]];
        if ([a[@"duration_ms"] isKindOfClass:NSNumber.class]) [line appendFormat:@" · %.1f s", [a[@"duration_ms"] doubleValue] / 1000.0];
        NSMutableDictionary *st = [small mutableCopy];
        st[NSForegroundColorAttributeName] = statusColor;
        [out appendAttributedString:Str([line stringByAppendingString:@"\n"], st)];
        for (NSString *stream in @[@"stdout", @"stderr"]) {
            if (S(a[stream]).length) [out appendAttributedString:[self renderCode:[Tail(S(a[stream]), 30) componentsSeparatedByString:@"\n"] lang:stream]];
        }
    }
    return out;
}

- (NSAttributedString *)renderMarkdown:(NSString *)text {
    NSMutableAttributedString *out = [NSMutableAttributedString new];
    NSArray<NSString *> *lines = [text componentsSeparatedByString:@"\n"];
    if (lines.count && lines.lastObject.length == 0) lines = [lines subarrayWithRange:NSMakeRange(0, lines.count - 1)];
    NSMutableParagraphStyle *bodyPS = [NSMutableParagraphStyle new];
    bodyPS.paragraphSpacing = 3;
    NSDictionary *base = @{NSFontAttributeName: BodyFont(), NSForegroundColorAttributeName: NSColor.textColor, NSParagraphStyleAttributeName: bodyPS};
    NSUInteger i = 0, n = lines.count;
    while (i < n) {
        NSString *line = lines[i];
        NSString *trim = [line stringByTrimmingCharactersInSet:WS()];
        if ([trim hasPrefix:@"```"]) {
            NSString *lang = [[trim substringFromIndex:3] stringByTrimmingCharactersInSet:WS()];
            NSMutableArray<NSString *> *code = [NSMutableArray new];
            i++;
            while (i < n && ![[lines[i] stringByTrimmingCharactersInSet:WS()] hasPrefix:@"```"]) { [code addObject:lines[i]]; i++; }
            if (i < n) i++; // closing fence (absent while streaming: the rest is code)
            [out appendAttributedString:[self renderCode:code lang:lang]];
            continue;
        }
        i++;
        if (trim.length == 0) {
            if (out.length) [out appendAttributedString:Str(@"\n", self.gapAttrs)];
            continue;
        }
        NSUInteger hashes = 0;
        while (hashes < trim.length && hashes < 4 && [trim characterAtIndex:hashes] == '#') hashes++;
        if (hashes > 0 && hashes < 4 && hashes < trim.length && [trim characterAtIndex:hashes] == ' ') {
            CGFloat size = hashes == 1 ? 18 : hashes == 2 ? 16 : 14;
            NSMutableParagraphStyle *ps = [NSMutableParagraphStyle new];
            ps.paragraphSpacingBefore = 8;
            ps.paragraphSpacing = 4;
            NSDictionary *h = @{NSFontAttributeName: [NSFont boldSystemFontOfSize:size], NSForegroundColorAttributeName: NSColor.textColor, NSParagraphStyleAttributeName: ps};
            [out appendAttributedString:Inline([[trim substringFromIndex:hashes + 1] stringByTrimmingCharactersInSet:WS()], h)];
            [out appendAttributedString:Str(@"\n", h)];
            continue;
        }
        if ([trim hasPrefix:@"- "] || [trim hasPrefix:@"* "]) {
            NSUInteger lead = line.length - [line stringByTrimmingCharactersInSet:WS()].length; // indent level
            NSMutableParagraphStyle *ps = [bodyPS mutableCopy];
            ps.firstLineHeadIndent = 8 + (CGFloat)MIN(lead, 8u) * 6;
            ps.headIndent = ps.firstLineHeadIndent + 14;
            ps.tabStops = @[[[NSTextTab alloc] initWithTextAlignment:NSTextAlignmentLeft location:ps.headIndent options:@{}]];
            NSMutableDictionary *b = [base mutableCopy];
            b[NSParagraphStyleAttributeName] = ps;
            [out appendAttributedString:Str(@"•\t", b)];
            [out appendAttributedString:Inline([trim substringFromIndex:2], b)];
            [out appendAttributedString:Str(@"\n", b)];
            continue;
        }
        [out appendAttributedString:Inline(line, base)];
        [out appendAttributedString:Str(@"\n", base)];
    }
    return out;
}

- (NSAttributedString *)renderCode:(NSArray<NSString *> *)lines lang:(NSString *)lang {
    NSString *code = [lines componentsJoinedByString:@"\n"];
    NSString *l = lang.lowercaseString;
    BOOL diff = [l isEqualToString:@"diff"] || [l isEqualToString:@"patch"];
    if (!diff) {
        BOOL minus = NO, plus = NO;
        for (NSString *ln in lines) { if ([ln hasPrefix:@"--- "]) minus = YES; else if ([ln hasPrefix:@"+++ "]) plus = YES; }
        diff = minus && plus;
    }
    NSMutableParagraphStyle *ps = [NSMutableParagraphStyle new];
    ps.lineBreakMode = self.wrapCode ? NSLineBreakByWordWrapping : NSLineBreakByClipping;
    ps.firstLineHeadIndent = 10;
    ps.headIndent = 10;
    ps.tailIndent = -10;
    NSDictionary *base = @{NSFontAttributeName: MonoFont(), NSForegroundColorAttributeName: NSColor.textColor, NSParagraphStyleAttributeName: ps, FrostCodeBlockAttribute: code};

    NSMutableAttributedString *out = [NSMutableAttributedString new];
    // Header line: language label + Copy button, on the same background as the code.
    NSMutableParagraphStyle *hps = [ps mutableCopy];
    hps.paragraphSpacingBefore = 6;
    hps.paragraphSpacing = 4;
    hps.lineBreakMode = NSLineBreakByTruncatingTail;
    NSDictionary *h = @{NSFontAttributeName: [NSFont systemFontOfSize:10 weight:NSFontWeightSemibold], NSForegroundColorAttributeName: NSColor.secondaryLabelColor,
                        NSParagraphStyleAttributeName: hps, NSKernAttributeName: @0.6, FrostCodeBlockAttribute: code};
    [out appendAttributedString:Str([(lang.length ? lang.uppercaseString : @"CODE") stringByAppendingString:@"   "], h)];
    [out appendAttributedString:Button(@"Copy", @"frost.copyCode", NSControlSizeMini, @{NSParagraphStyleAttributeName: hps, FrostCodeBlockAttribute: code}, ^{
        [NSPasteboard.generalPasteboard clearContents];
        [NSPasteboard.generalPasteboard setString:code forType:NSPasteboardTypeString];
    })];
    [out appendAttributedString:Str(@"\n", h)];

    for (NSString *ln in lines) {
        NSDictionary *a = base;
        if (diff) {
            NSColor *c = nil;
            if ([ln hasPrefix:@"+"] && ![ln hasPrefix:@"+++ "]) c = NSColor.systemGreenColor;
            else if ([ln hasPrefix:@"-"] && ![ln hasPrefix:@"--- "]) c = NSColor.systemRedColor;
            else if ([ln hasPrefix:@"@@"]) c = NSColor.secondaryLabelColor;
            if (c) { NSMutableDictionary *m = [base mutableCopy]; m[NSForegroundColorAttributeName] = c; a = m; }
        }
        [out appendAttributedString:Str([ln stringByAppendingString:@"\n"], a)];
    }
    NSMutableDictionary *pad = [base mutableCopy]; // bottom padding inside the block
    pad[NSFontAttributeName] = [NSFont systemFontOfSize:4];
    [out appendAttributedString:Str(@"\n", pad)];
    [out appendAttributedString:Str(@"\n", self.gapAttrs)];
    return out;
}

@end

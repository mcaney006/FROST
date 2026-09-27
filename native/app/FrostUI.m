// FROST AppKit shell: window, sidebar, composer, inspector, menus, and the bridge from the
// engine's background-thread event callback to the main thread. Rust owns the engine; this
// file only borrows the handle passed to frost_ui_run.
#import "FrostUI.h"
#pragma clang diagnostic ignored "-Wunused-parameter"

#ifndef FROST_VERSION
#define FROST_VERSION "dev"
#endif

static NSString *const kWrapCodeDefault = @"FrostWrapCodeBlocks";

#pragma mark - ABI helpers (every returned char* is freed here)

static NSString *FrostTake(char *s) {
    if (!s) return nil;
    NSString *r = [NSString stringWithUTF8String:s] ?: @"";
    frost_string_free(s);
    return r;
}
static id FrostJSON(char *s) {
    NSString *str = FrostTake(s);
    NSData *d = [str dataUsingEncoding:NSUTF8StringEncoding];
    return d ? [NSJSONSerialization JSONObjectWithData:d options:0 error:NULL] : nil;
}
static NSString *FrostBytes(NSNumber *n) {
    NSByteCountFormatter *f = [NSByteCountFormatter new];
    f.countStyle = NSByteCountFormatterCountStyleMemory;
    f.allowsNonnumericFormatting = NO; // "0 bytes", not "Zero KB"
    return [f stringFromByteCount:n.longLongValue];
}
static NSString *Short(id v) { // short revision hashes
    NSString *s = [v isKindOfClass:NSString.class] ? v : [v description] ?: @"";
    return s.length > 12 && [s rangeOfCharacterFromSet:[NSCharacterSet.alphanumericCharacterSet invertedSet]].location == NSNotFound ? [s substringToIndex:12] : s;
}

#pragma mark - Small view subclasses

/// Composer: TextKit 1 text view with a placeholder. Return sends, Shift+Return inserts a newline
/// (handled by the controller in textView:doCommandBySelector:).
@interface FrostComposerView : NSTextView
@property (nonatomic, copy) NSString *placeholder;
@end
@implementation FrostComposerView
- (void)setPlaceholder:(NSString *)placeholder { _placeholder = [placeholder copy]; self.needsDisplay = YES; }
- (void)drawRect:(NSRect)rect {
    [super drawRect:rect];
    if (self.string.length == 0 && self.placeholder.length) {
        NSPoint p = NSMakePoint(self.textContainerInset.width + self.textContainer.lineFragmentPadding, self.textContainerInset.height);
        [self.placeholder drawAtPoint:p withAttributes:@{NSFontAttributeName: self.font ?: [NSFont systemFontOfSize:13], NSForegroundColorAttributeName: NSColor.placeholderTextColor}];
    }
}
- (void)didChangeText { [super didChangeText]; self.needsDisplay = YES; }
@end

@class FrostController;
static FrostController *gController; // NSApplication.delegate is weak; keep the controller alive

/// Sidebar table: ⌫ deletes, Return renames.
@interface FrostTableView : NSTableView
@end

#pragma mark - Controller

@interface FrostController : NSObject <NSApplicationDelegate, NSWindowDelegate, NSTableViewDataSource, NSTableViewDelegate, NSTextViewDelegate, NSTextFieldDelegate, NSMenuItemValidation>
@property (nonatomic, assign) FrostEngine *engine;
@property (nonatomic, strong) NSWindow *window;
@property (nonatomic, strong) NSSplitViewController *split;
@property (nonatomic, strong) FrostTableView *table;
@property (nonatomic, strong) FrostTranscript *transcript;
@property (nonatomic, strong) FrostComposerView *composer;
@property (nonatomic, strong) NSLayoutConstraint *composerHeight;
@property (nonatomic, strong) NSButton *sendButton, *cancelButton, *detachButton;
@property (nonatomic, strong) NSTextField *statusLabel, *modelLabel, *contextLabel, *repoLabel, *notesLabel;
@property (nonatomic, strong) NSMenu *modeMenu;
@property (nonatomic, strong) NSMenuItem *wrapItem;
// engine mirror
@property (nonatomic, copy) NSArray<NSDictionary *> *conversations;
@property (nonatomic, copy) NSString *selectedId;
@property (nonatomic, copy) NSArray<NSDictionary *> *messages;
@property (nonatomic, copy) NSDictionary *status, *lastMeta;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSMutableArray<NSString *> *> *notes;
@property (nonatomic) BOOL generating, suppressSelection;
@property (nonatomic, copy) NSString *streamingId;
@property (nonatomic, strong) NSMutableString *streamingText;
@property (nonatomic, copy) NSString *errorText;
@property (nonatomic, strong) NSDate *errorTime;
- (void)deleteConversation:(id)sender;
- (void)renameConversation:(id)sender;
- (void)handleEvent:(NSDictionary *)ev;
@end

@implementation FrostTableView
- (void)keyDown:(NSEvent *)event {
    NSString *chars = event.charactersIgnoringModifiers;
    unichar ch = chars.length ? [chars characterAtIndex:0] : 0;
    if (self.selectedRow >= 0 && (ch == NSDeleteCharacter || ch == NSBackspaceCharacter)) { [gController deleteConversation:nil]; return; }
    if (self.selectedRow >= 0 && (ch == NSCarriageReturnCharacter || ch == NSEnterCharacter)) { [gController renameConversation:nil]; return; }
    [super keyDown:event];
}
@end

/// Engine callback: background thread → main queue. The JSON is copied before returning.
static void FrostEventTrampoline(void *ctx, const char *json) {
    static BOOL trace = NO;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ trace = getenv("FROST_UI_TRACE") != NULL; });
    if (trace && json) fprintf(stderr, "[frost-ui] %s\n", json); // opt-in debugging aid; stderr only
    NSString *s = json ? [NSString stringWithUTF8String:json] : nil;
    if (!s) return;
    FrostController *c = (__bridge FrostController *)ctx;
    dispatch_async(dispatch_get_main_queue(), ^{
        NSDictionary *ev = [NSJSONSerialization JSONObjectWithData:[s dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
        if ([ev isKindOfClass:NSDictionary.class]) [c handleEvent:ev];
    });
}

@implementation FrostController

- (instancetype)initWithEngine:(FrostEngine *)engine {
    if ((self = [super init])) {
        _engine = engine;
        _notes = [NSMutableDictionary new];
        _conversations = @[];
        _messages = @[];
    }
    return self;
}

#pragma mark NSApplicationDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)note {
    [self buildMenus];
    [self buildWindow];
    frost_engine_set_event_callback(self.engine, FrostEventTrampoline, (__bridge void *)self);
    [self reloadConversations];
    if (!self.conversations.count) [self newChat:nil];
    [self refreshStatus];
    [self refreshModelInfo];
    [NSTimer scheduledTimerWithTimeInterval:2 target:self selector:@selector(tick:) userInfo:nil repeats:YES];
    [self.window makeKeyAndOrderFront:nil];
    [self.window makeFirstResponder:self.composer];
    [NSApp activate];
}
- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app { return YES; }
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)app { return YES; }
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)app {
    // Do not exit here: make -[NSApp run] return so Rust frees the engine and exits.
    if (self.generating) frost_cancel(self.engine);
    [NSApp stop:nil];
    [NSApp postEvent:[NSEvent otherEventWithType:NSEventTypeApplicationDefined location:NSZeroPoint modifierFlags:0 timestamp:0 windowNumber:0 context:nil subtype:0 data1:0 data2:0] atStart:NO];
    return NSTerminateCancel;
}

#pragma mark Menus

- (NSMenuItem *)item:(NSMenu *)menu title:(NSString *)title action:(SEL)action key:(NSString *)key mods:(NSEventModifierFlags)mods target:(id)target {
    NSMenuItem *it = [[NSMenuItem alloc] initWithTitle:title action:action keyEquivalent:key];
    it.keyEquivalentModifierMask = mods;
    it.target = target;
    [menu addItem:it];
    return it;
}
- (NSMenu *)submenu:(NSMenu *)bar title:(NSString *)title {
    NSMenuItem *holder = [[NSMenuItem alloc] initWithTitle:title action:NULL keyEquivalent:@""];
    NSMenu *m = [[NSMenu alloc] initWithTitle:title];
    holder.submenu = m;
    [bar addItem:holder];
    return m;
}

- (void)buildMenus {
    NSEventModifierFlags cmd = NSEventModifierFlagCommand, cmdShift = cmd | NSEventModifierFlagShift, cmdOpt = cmd | NSEventModifierFlagOption;
    NSMenu *bar = [[NSMenu alloc] initWithTitle:@""];

    NSMenu *app = [self submenu:bar title:@"FROST"];
    [self item:app title:@"About FROST" action:@selector(about:) key:@"" mods:0 target:self];
    [app addItem:NSMenuItem.separatorItem];
    [self item:app title:@"Quit FROST" action:@selector(terminate:) key:@"q" mods:cmd target:NSApp];

    NSMenu *file = [self submenu:bar title:@"File"];
    [self item:file title:@"New Chat" action:@selector(newChat:) key:@"n" mods:cmd target:self];
    [self item:file title:@"Open Repository…" action:@selector(openRepository:) key:@"o" mods:cmd target:self];
    [file addItem:NSMenuItem.separatorItem];
    [self item:file title:@"Delete Conversation" action:@selector(deleteConversation:) key:[NSString stringWithFormat:@"%C", (unichar)NSBackspaceCharacter] mods:cmd target:self];
    [self item:file title:@"Rename Conversation" action:@selector(renameConversation:) key:@"" mods:0 target:self];

    NSMenu *edit = [self submenu:bar title:@"Edit"];
    [self item:edit title:@"Undo" action:@selector(undo:) key:@"z" mods:cmd target:nil];
    [self item:edit title:@"Redo" action:@selector(redo:) key:@"z" mods:cmdShift target:nil];
    [edit addItem:NSMenuItem.separatorItem];
    [self item:edit title:@"Cut" action:@selector(cut:) key:@"x" mods:cmd target:nil];
    [self item:edit title:@"Copy" action:@selector(copy:) key:@"c" mods:cmd target:nil];
    [self item:edit title:@"Paste" action:@selector(paste:) key:@"v" mods:cmd target:nil];
    [self item:edit title:@"Select All" action:@selector(selectAll:) key:@"a" mods:cmd target:nil];

    NSMenu *chat = [self submenu:bar title:@"Chat"];
    [self item:chat title:@"Send" action:@selector(send:) key:@"\r" mods:cmd target:self];
    [self item:chat title:@"Cancel" action:@selector(cancel:) key:@"." mods:cmd target:self];
    [self item:chat title:@"Regenerate" action:@selector(regenerate:) key:@"r" mods:cmd target:self];
    [self item:chat title:@"Clear Context" action:@selector(clearContext:) key:@"k" mods:cmd target:self];
    [chat addItem:NSMenuItem.separatorItem];
    [self item:chat title:@"Copy Last Reply" action:@selector(copyLastReply:) key:@"" mods:0 target:self];

    self.modeMenu = [self submenu:bar title:@"Mode"];
    for (NSString *mode in @[@"quiet", @"balanced", @"performance"]) {
        NSMenuItem *it = [self item:self.modeMenu title:mode.capitalizedString action:@selector(setMode:) key:@"" mods:0 target:self];
        it.representedObject = mode;
        it.accessibilityIdentifier = [@"frost.mode." stringByAppendingString:mode];
        it.accessibilityLabel = [NSString stringWithFormat:@"%@ mode", mode.capitalizedString];
    }

    NSMenu *view = [self submenu:bar title:@"View"];
    [self item:view title:@"Toggle Sidebar" action:@selector(toggleSidebar:) key:@"s" mods:cmdOpt target:nil];
    [self item:view title:@"Toggle Inspector" action:@selector(toggleInspector:) key:@"i" mods:cmdOpt target:nil];
    [view addItem:NSMenuItem.separatorItem];
    self.wrapItem = [self item:view title:@"Wrap Code Blocks" action:@selector(toggleWrapCode:) key:@"" mods:0 target:self];

    NSMenu *model = [self submenu:bar title:@"Model"];
    [self item:model title:@"Model Info…" action:@selector(modelInfo:) key:@"" mods:0 target:self];

    NSMenu *window = [self submenu:bar title:@"Window"];
    [self item:window title:@"Minimize" action:@selector(performMiniaturize:) key:@"m" mods:cmd target:nil];
    [self item:window title:@"Zoom" action:@selector(performZoom:) key:@"" mods:0 target:nil];
    [self item:window title:@"Close" action:@selector(performClose:) key:@"w" mods:cmd target:nil];

    NSApp.mainMenu = bar;
    NSApp.windowsMenu = window;
}

- (BOOL)validateMenuItem:(NSMenuItem *)item {
    SEL a = item.action;
    BOOL ready = [self.status[@"state"] isEqual:@"ready"];
    if (a == @selector(send:)) return ready && !self.generating && self.selectedId != nil;
    if (a == @selector(cancel:)) return self.generating;
    if (a == @selector(regenerate:)) return ready && !self.generating && [self lastMessageWithRole:@"user"] != nil;
    if (a == @selector(clearContext:)) return self.selectedId != nil && self.messages.count > 0;
    if (a == @selector(copyLastReply:)) return [self lastMessageWithRole:@"assistant"][@"content"] != nil;
    if (a == @selector(deleteConversation:) || a == @selector(renameConversation:) || a == @selector(openRepository:)) return self.selectedId != nil;
    if (a == @selector(setMode:)) { item.state = [item.representedObject isEqual:self.status[@"mode"]] ? NSControlStateValueOn : NSControlStateValueOff; return YES; }
    if (a == @selector(toggleWrapCode:)) { item.state = self.transcript.wrapCode ? NSControlStateValueOn : NSControlStateValueOff; return YES; }
    return YES;
}

#pragma mark Window and panes

- (void)buildWindow {
    NSWindow *w = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 1200, 800)
                                              styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                                                backing:NSBackingStoreBuffered defer:NO];
    w.title = @"FROST";
    w.contentMinSize = NSMakeSize(1100, 720);
    w.tabbingMode = NSWindowTabbingModeDisallowed;
    w.releasedWhenClosed = NO; // ARC owns it via self.window; the default extra release on close dangles it
    w.delegate = self;

    NSSplitViewController *split = [NSSplitViewController new];
    NSSplitViewItem *sidebar = [NSSplitViewItem sidebarWithViewController:[self viewController:[self buildSidebar]]];
    sidebar.minimumThickness = 200;
    sidebar.maximumThickness = 360;
    NSSplitViewItem *center = [NSSplitViewItem splitViewItemWithViewController:[self viewController:[self buildCenter]]];
    center.minimumThickness = 480;
    NSSplitViewItem *inspector = [NSSplitViewItem inspectorWithViewController:[self viewController:[self buildInspector]]];
    inspector.minimumThickness = 260;
    inspector.maximumThickness = 440;
    [split addSplitViewItem:sidebar];
    [split addSplitViewItem:center];
    [split addSplitViewItem:inspector];
    w.contentViewController = split;
    if (![w setFrameUsingName:@"FrostMain"]) { // no saved frame yet: the split view shrank us to the minimum
        [w setContentSize:NSMakeSize(1200, 800)];
        [w center];
    }
    [w setFrameAutosaveName:@"FrostMain"];
    self.split = split;
    self.window = w;
    w.initialFirstResponder = self.composer;
}

- (NSViewController *)viewController:(NSView *)view {
    NSViewController *vc = [NSViewController new];
    vc.view = view;
    return vc;
}

- (NSScrollView *)scrollViewFor:(NSTextView *)tv {
    NSScrollView *sv = [NSScrollView new];
    sv.translatesAutoresizingMaskIntoConstraints = NO;
    sv.hasVerticalScroller = YES;
    sv.borderType = NSNoBorder;
    sv.drawsBackground = NO;
    tv.frame = NSMakeRect(0, 0, 600, 100);
    tv.minSize = NSMakeSize(0, 0);
    tv.maxSize = NSMakeSize(FLT_MAX, FLT_MAX);
    tv.verticallyResizable = YES;
    tv.horizontallyResizable = NO;
    tv.autoresizingMask = NSViewWidthSizable;
    tv.textContainer.containerSize = NSMakeSize(600, FLT_MAX);
    tv.textContainer.widthTracksTextView = YES;
    sv.documentView = tv;
    return sv;
}

- (NSView *)buildSidebar {
    NSView *root = [NSView new];
    FrostTableView *table = [FrostTableView new];
    NSTableColumn *col = [[NSTableColumn alloc] initWithIdentifier:@"conv"];
    [table addTableColumn:col];
    table.headerView = nil;
    table.style = NSTableViewStyleSourceList;
    table.rowHeight = 46;
    table.dataSource = self;
    table.delegate = self;
    table.target = self;
    table.doubleAction = @selector(renameConversation:);
    table.accessibilityIdentifier = @"frost.sidebar";
    table.accessibilityLabel = @"Conversations";
    NSMenu *ctx = [[NSMenu alloc] initWithTitle:@""];
    [self item:ctx title:@"Rename" action:@selector(renameConversation:) key:@"" mods:0 target:self];
    [self item:ctx title:@"Delete…" action:@selector(deleteConversation:) key:@"" mods:0 target:self];
    table.menu = ctx;
    self.table = table;

    NSScrollView *sv = [NSScrollView new];
    sv.translatesAutoresizingMaskIntoConstraints = NO;
    sv.hasVerticalScroller = YES;
    sv.borderType = NSNoBorder;
    sv.drawsBackground = NO;
    sv.documentView = table;

    NSButton *newChat = [NSButton buttonWithTitle:@"New Chat" target:self action:@selector(newChat:)];
    newChat.translatesAutoresizingMaskIntoConstraints = NO;
    newChat.bezelStyle = NSBezelStyleRounded;
    newChat.accessibilityIdentifier = @"frost.newChat";
    newChat.accessibilityLabel = @"New Chat";

    [root addSubview:sv];
    [root addSubview:newChat];
    [NSLayoutConstraint activateConstraints:@[
        [sv.topAnchor constraintEqualToAnchor:root.topAnchor],
        [sv.leadingAnchor constraintEqualToAnchor:root.leadingAnchor],
        [sv.trailingAnchor constraintEqualToAnchor:root.trailingAnchor],
        [newChat.topAnchor constraintEqualToAnchor:sv.bottomAnchor constant:8],
        [newChat.leadingAnchor constraintEqualToAnchor:root.leadingAnchor constant:12],
        [newChat.trailingAnchor constraintEqualToAnchor:root.trailingAnchor constant:-12],
        [newChat.bottomAnchor constraintEqualToAnchor:root.bottomAnchor constant:-12],
    ]];
    return root;
}

- (NSView *)buildCenter {
    NSView *root = [NSView new];

    FrostTranscriptTextView *tv = [FrostTranscriptTextView textViewUsingTextLayoutManager:YES];
    tv.editable = NO;
    tv.selectable = YES;
    tv.richText = YES;
    tv.drawsBackground = YES;
    tv.backgroundColor = NSColor.textBackgroundColor;
    tv.textContainerInset = NSMakeSize(18, 14);
    tv.accessibilityIdentifier = @"frost.transcript";
    tv.accessibilityLabel = @"Transcript";
    NSScrollView *transcriptSV = [self scrollViewFor:tv];
    transcriptSV.drawsBackground = YES;
    transcriptSV.backgroundColor = NSColor.textBackgroundColor;
    self.transcript = [[FrostTranscript alloc] initWithTextView:tv];
    __weak typeof(self) weakSelf = self;
    self.transcript.decideAttempt = ^(NSString *convId, NSString *attemptId, BOOL approve) {
        typeof(self) me = weakSelf;
        if (!me) return;
        if (frost_attempt_decide(me.engine, convId.UTF8String, attemptId.UTF8String, approve ? 1 : 0) < 0) [me showError:approve ? @"Approve failed" : @"Deny failed"];
    };
    self.transcript.wrapCode =[NSUserDefaults.standardUserDefaults objectForKey:kWrapCodeDefault] ? [NSUserDefaults.standardUserDefaults boolForKey:kWrapCodeDefault] : YES;

    FrostComposerView *composer = [FrostComposerView textViewUsingTextLayoutManager:NO];
    composer.font = [NSFont systemFontOfSize:13];
    composer.richText = NO;
    composer.allowsUndo = YES;
    composer.delegate = self;
    composer.textContainerInset = NSMakeSize(6, 6);
    composer.placeholder = @"Loading model…";
    composer.accessibilityIdentifier = @"frost.composer";
    composer.accessibilityLabel = @"Message";
    NSScrollView *composerSV = [self scrollViewFor:composer];
    composerSV.borderType = NSBezelBorder;
    composerSV.drawsBackground = YES;
    composerSV.backgroundColor = NSColor.textBackgroundColor;
    composerSV.hasVerticalScroller = NO;
    self.composer = composer;

    NSButton *send = [NSButton buttonWithTitle:@"Send" target:self action:@selector(send:)];
    send.translatesAutoresizingMaskIntoConstraints = NO;
    send.bezelStyle = NSBezelStyleRounded;
    send.accessibilityIdentifier = @"frost.send";
    send.accessibilityLabel = @"Send";
    send.enabled = NO;
    NSButton *cancel = [NSButton buttonWithTitle:@"Cancel" target:self action:@selector(cancel:)];
    cancel.translatesAutoresizingMaskIntoConstraints = NO;
    cancel.bezelStyle = NSBezelStyleRounded;
    cancel.accessibilityIdentifier = @"frost.cancel";
    cancel.accessibilityLabel = @"Cancel generation";
    cancel.hidden = YES;
    self.sendButton = send;
    self.cancelButton = cancel;

    NSTextField *status = [NSTextField labelWithString:@""];
    status.translatesAutoresizingMaskIntoConstraints = NO;
    status.font = [NSFont systemFontOfSize:11];
    status.textColor = NSColor.secondaryLabelColor;
    status.lineBreakMode = NSLineBreakByTruncatingTail;
    status.accessibilityIdentifier = @"frost.status";
    status.accessibilityLabel = @"Status";
    self.statusLabel = status;

    [root addSubview:transcriptSV];
    [root addSubview:composerSV];
    [root addSubview:send];
    [root addSubview:cancel];
    [root addSubview:status];
    self.composerHeight = [composerSV.heightAnchor constraintEqualToConstant:34];
    [NSLayoutConstraint activateConstraints:@[
        [transcriptSV.topAnchor constraintEqualToAnchor:root.topAnchor],
        [transcriptSV.leadingAnchor constraintEqualToAnchor:root.leadingAnchor],
        [transcriptSV.trailingAnchor constraintEqualToAnchor:root.trailingAnchor],
        [composerSV.topAnchor constraintEqualToAnchor:transcriptSV.bottomAnchor constant:8],
        [composerSV.leadingAnchor constraintEqualToAnchor:root.leadingAnchor constant:12],
        self.composerHeight,
        [send.leadingAnchor constraintEqualToAnchor:composerSV.trailingAnchor constant:8],
        [send.trailingAnchor constraintEqualToAnchor:root.trailingAnchor constant:-12],
        [send.bottomAnchor constraintEqualToAnchor:composerSV.bottomAnchor],
        [send.widthAnchor constraintEqualToConstant:76],
        [cancel.leadingAnchor constraintEqualToAnchor:send.leadingAnchor],
        [cancel.trailingAnchor constraintEqualToAnchor:send.trailingAnchor],
        [cancel.bottomAnchor constraintEqualToAnchor:send.topAnchor constant:-4],
        [status.topAnchor constraintEqualToAnchor:composerSV.bottomAnchor constant:6],
        [status.leadingAnchor constraintEqualToAnchor:root.leadingAnchor constant:14],
        [status.trailingAnchor constraintEqualToAnchor:root.trailingAnchor constant:-14],
        [status.bottomAnchor constraintEqualToAnchor:root.bottomAnchor constant:-8],
    ]];
    return root;
}

- (NSTextField *)section:(NSString *)title identifier:(NSString *)ident in:(NSStackView *)stack extras:(NSArray<NSView *> *)extras {
    NSTextField *header = [NSTextField labelWithString:title.localizedUppercaseString];
    header.font = [NSFont systemFontOfSize:11 weight:NSFontWeightBold];
    header.textColor = NSColor.secondaryLabelColor;
    NSTextField *body = [NSTextField wrappingLabelWithString:@"—"];
    body.font = [NSFont systemFontOfSize:12];
    body.selectable = YES;
    body.accessibilityIdentifier = ident;
    body.accessibilityLabel = title;
    [body setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow forOrientation:NSLayoutConstraintOrientationHorizontal];
    [stack addView:header inGravity:NSStackViewGravityTop];
    [stack addView:body inGravity:NSStackViewGravityTop];
    [stack setCustomSpacing:2 afterView:header];
    if (extras.count) {
        NSStackView *row = [NSStackView stackViewWithViews:extras];
        row.orientation = NSUserInterfaceLayoutOrientationHorizontal;
        row.spacing = 8;
        [stack addView:row inGravity:NSStackViewGravityTop];
    }
    [stack setCustomSpacing:18 afterView:stack.views.lastObject];
    return body;
}

- (NSView *)buildInspector {
    NSStackView *stack = [NSStackView new];
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    stack.alignment = NSLayoutAttributeLeading;
    stack.spacing = 6;
    stack.edgeInsets = NSEdgeInsetsMake(14, 14, 14, 14);

    NSButton *open = [NSButton buttonWithTitle:@"Open Repository…" target:self action:@selector(openRepository:)];
    open.bezelStyle = NSBezelStyleRounded;
    open.controlSize = NSControlSizeSmall;
    open.accessibilityIdentifier = @"frost.openRepository";
    NSButton *detach = [NSButton buttonWithTitle:@"Detach" target:self action:@selector(detachRepository:)];
    detach.bezelStyle = NSBezelStyleRounded;
    detach.controlSize = NSControlSizeSmall;
    detach.accessibilityIdentifier = @"frost.detachRepository";
    self.detachButton = detach;

    self.modelLabel = [self section:@"Model" identifier:@"frost.inspector.model" in:stack extras:@[]];
    self.contextLabel = [self section:@"Context" identifier:@"frost.inspector.context" in:stack extras:@[]];
    self.repoLabel = [self section:@"Repository" identifier:@"frost.inspector.repository" in:stack extras:@[open, detach]];
    self.notesLabel = [self section:@"Notes" identifier:@"frost.inspector.notes" in:stack extras:@[]];

    NSScrollView *sv = [NSScrollView new];
    sv.hasVerticalScroller = YES;
    sv.borderType = NSNoBorder;
    sv.drawsBackground = NO;
    sv.documentView = stack;
    [NSLayoutConstraint activateConstraints:@[
        [stack.topAnchor constraintEqualToAnchor:sv.contentView.topAnchor],
        [stack.leadingAnchor constraintEqualToAnchor:sv.contentView.leadingAnchor],
        [stack.widthAnchor constraintEqualToAnchor:sv.contentView.widthAnchor],
        [stack.heightAnchor constraintGreaterThanOrEqualToAnchor:sv.contentView.heightAnchor], // keeps content top-aligned
    ]];
    sv.accessibilityIdentifier = @"frost.inspector";
    sv.accessibilityLabel = @"Inspector";
    return sv;
}

#pragma mark Engine mirror

- (NSDictionary *)selectedConversation {
    for (NSDictionary *c in self.conversations) if ([c[@"id"] isEqual:self.selectedId]) return c;
    return nil;
}
- (NSDictionary *)lastMessageWithRole:(NSString *)role {
    for (NSDictionary *m in self.messages.reverseObjectEnumerator) if ([m[@"role"] isEqual:role]) return m;
    return nil;
}

- (void)reloadConversations {
    NSArray *convs = FrostJSON(frost_conversations_json(self.engine));
    self.conversations = [convs isKindOfClass:NSArray.class] ? convs : @[];
    if (self.selectedId && ![self selectedConversation]) self.selectedId = nil;
    if (!self.selectedId) self.selectedId = self.conversations.firstObject[@"id"];
    self.suppressSelection = YES;
    [self.table reloadData];
    NSUInteger row = [self.conversations indexOfObjectPassingTest:^BOOL(NSDictionary *c, NSUInteger i, BOOL *stop) { return [c[@"id"] isEqual:self.selectedId]; }];
    if (row != NSNotFound) [self.table selectRowIndexes:[NSIndexSet indexSetWithIndex:row] byExtendingSelection:NO];
    else [self.table deselectAll:nil];
    self.suppressSelection = NO;
    if (![self.messages.firstObject[@"conversation_id"] isEqual:self.selectedId] || !self.selectedId) [self reloadMessages];
    [self updateRepoSection];
}

- (void)reloadMessages {
    NSArray *msgs = self.selectedId ? FrostJSON(frost_messages_json(self.engine, self.selectedId.UTF8String)) : @[];
    if (![msgs isKindOfClass:NSArray.class]) msgs = @[];
    if (self.streamingId) { // a reload mid-stream keeps the partial text
        NSMutableArray *patched = [msgs mutableCopy];
        for (NSUInteger i = 0; i < patched.count; i++) {
            if ([patched[i][@"id"] isEqual:self.streamingId] && [patched[i][@"content"] length] == 0) {
                NSMutableDictionary *m = [patched[i] mutableCopy];
                m[@"content"] = [self.streamingText copy] ?: @"";
                patched[i] = m;
            }
        }
        msgs = patched;
    }
    self.messages = msgs;
    self.lastMeta = nil;
    for (NSDictionary *m in msgs.reverseObjectEnumerator) {
        if ([m[@"role"] isEqual:@"assistant"] && [m[@"meta"][@"status"] isEqual:@"done"]) { self.lastMeta = m[@"meta"]; break; }
    }
    // Attempts (proposed patches / commands) interleave with messages by creation time.
    NSArray *attempts = self.selectedId ? FrostJSON(frost_attempts_json(self.engine, self.selectedId.UTF8String)) : nil;
    NSMutableArray *items = [msgs mutableCopy];
    if ([attempts isKindOfClass:NSArray.class]) for (NSDictionary *a in attempts) if ([a isKindOfClass:NSDictionary.class]) [items addObject:[self attemptItem:a]];
    [items sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        NSComparisonResult r = [a[@"created_at"] ?: @0 compare:b[@"created_at"] ?: @0];
        return r != NSOrderedSame ? r : [@([a[@"role"] isEqual:@"attempt"]) compare:@([b[@"role"] isEqual:@"attempt"])];
    }];
    [self.transcript setMessages:items];
    [self updateContextSection];
    [self updateNotesSection];
    [self updateStatusLabel];
}

- (NSDictionary *)attemptItem:(NSDictionary *)attempt {
    NSMutableDictionary *item = [attempt mutableCopy];
    item[@"role"] = @"attempt";
    return item;
}

- (void)refreshStatus {
    NSDictionary *s = FrostJSON(frost_engine_status_json(self.engine));
    if (![s isKindOfClass:NSDictionary.class]) return;
    self.status = s;
    if (self.generating && [s[@"generating"] isKindOfClass:NSNumber.class] && ![s[@"generating"] boolValue]) {
        // The engine is idle but we never saw message_done (event lost or handler failed): resync from the store.
        self.generating = NO;
        self.streamingId = nil;
        self.streamingText = nil;
        [self reloadMessages];
    }
    NSString *state = s[@"state"] ?: @"loading";
    BOOL loading = [state isEqualToString:@"loading"];
    self.composer.editable = !loading;
    self.composer.placeholder = loading ? @"Loading model…" : [state isEqualToString:@"error"] ? @"Model unavailable — see transcript" : @"Message FROST…  (Return sends, Shift+Return for a new line)";
    self.sendButton.enabled = [state isEqualToString:@"ready"] && !self.generating && self.selectedId != nil;
    self.cancelButton.hidden = !self.generating;
    self.cancelButton.enabled = self.generating;
    self.transcript.bannerText = [state isEqualToString:@"error"] ? (s[@"detail"] ?: @"unknown error") : nil;
    [self updateContextSection];
    [self updateStatusLabel];
}

- (void)refreshModelInfo {
    NSDictionary *info = FrostJSON(frost_model_info_json(self.engine));
    if (![info isKindOfClass:NSDictionary.class]) return;
    NSMutableString *t = [NSMutableString new];
    NSDictionary *g = [info[@"generator"] isKindOfClass:NSDictionary.class] ? info[@"generator"] : nil;
    if (g) {
        [t appendFormat:@"Generator\n%@\nrevision %@\n%@ · %@\n%@ · %@ layers · %@ weights\n", g[@"repo"] ?: @"?", Short(g[@"revision"] ?: @"?"),
                        g[@"quantization"] ?: @"?", g[@"backend"] ?: @"?", g[@"architecture"] ?: @"?", g[@"layers"] ?: @"?",
                        [g[@"weight_bytes"] isKindOfClass:NSNumber.class] ? FrostBytes(g[@"weight_bytes"]) : @"?"];
    } else {
        [t appendString:@"Generator: not loaded yet\n"];
    }
    NSDictionary *r = [info[@"retrieval"] isKindOfClass:NSDictionary.class] ? info[@"retrieval"] : nil;
    if (r) [t appendFormat:@"\nRetrieval model (retrieval only, never produces chat text)\n%@\nrevision %@\n", r[@"model"] ?: @"?", Short(r[@"revision"] ?: @"?")];
    id sampler = info[@"sampler"] ?: info[@"sampler_backend"]; // header says "sampler"; the engine currently emits "sampler_backend"
    if (sampler) [t appendFormat:@"\nSampler: %@", sampler];
    self.modelLabel.stringValue = t;
}

- (void)updateContextSection {
    NSDictionary *s = self.status ?: @{};
    NSMutableString *t = [NSMutableString stringWithFormat:@"Context budget: %@ tokens\nReserved for output: %@ tokens\n", s[@"context_budget"] ?: @"?", s[@"reserved_output_tokens"] ?: @"?"];
    if ([s[@"kv_cache_bytes"] isKindOfClass:NSNumber.class]) [t appendFormat:@"KV cache: %@\n", FrostBytes(s[@"kv_cache_bytes"])];
    NSDictionary *m = self.lastMeta;
    if (m) {
        [t appendFormat:@"\nLast reply\nprompt tokens: %@\nreused prefix tokens: %@\nnew tokens: %@\nspeed: %.1f tok/s\nfinish: %@\nmessages truncated from context: %@",
                        m[@"prompt_tokens"] ?: @"?", m[@"reused_prefix_tokens"] ?: @"?", m[@"new_tokens"] ?: @"?",
                        [m[@"tokens_per_second"] doubleValue], m[@"finish"] ?: @"?", m[@"context_truncated_messages"] ?: @"0"];
    } else {
        [t appendString:@"\nLast reply: none yet"];
    }
    self.contextLabel.stringValue = t;
}

- (void)updateRepoSection {
    id path = [self selectedConversation][@"repo_path"];
    BOOL has = [path isKindOfClass:NSString.class] && [path length] > 0;
    self.repoLabel.stringValue = has ? path : @"none";
    self.detachButton.enabled = has;
}

- (void)updateNotesSection {
    NSArray *notes = self.selectedId ? self.notes[self.selectedId] : nil;
    self.notesLabel.stringValue = notes.count ? [notes componentsJoinedByString:@"\n\n"] : @"No notes for this conversation.";
}

- (void)updateStatusLabel {
    NSDictionary *s = self.status ?: @{};
    NSString *state = s[@"state"] ?: @"loading";
    NSMutableArray *parts = [NSMutableArray new];
    if ([state isEqualToString:@"ready"]) [parts addObject:self.generating ? @"Generating…" : @"Ready"];
    else if ([state isEqualToString:@"loading"]) [parts addObject:@"Loading model…"];
    else [parts addObject:[NSString stringWithFormat:@"Error: %@", s[@"detail"] ?: @""]];
    [parts addObject:s[@"mode"] ?: @"—"];
    [parts addObject:[NSString stringWithFormat:@"thermal %@", s[@"thermal"] ?: @"—"]];
    if ([self.lastMeta[@"tokens_per_second"] isKindOfClass:NSNumber.class]) [parts addObject:[NSString stringWithFormat:@"%.1f tok/s", [self.lastMeta[@"tokens_per_second"] doubleValue]]];
    if ([s[@"mlx_active_bytes"] isKindOfClass:NSNumber.class]) [parts addObject:[NSString stringWithFormat:@"MLX %@", FrostBytes(s[@"mlx_active_bytes"])]];
    NSString *text = [parts componentsJoinedByString:@"  ·  "];
    if (self.errorText && -[self.errorTime timeIntervalSinceNow] < 12) text = [text stringByAppendingFormat:@"  —  %@", self.errorText];
    self.statusLabel.stringValue = text;
}

- (void)tick:(NSTimer *)t { [self refreshStatus]; }

/// Surfaces frost_last_error for a failed call in the status line and the Notes section.
- (void)showError:(NSString *)context {
    NSString *detail = FrostTake(frost_last_error(self.engine)) ?: @"";
    self.errorText = detail.length ? [NSString stringWithFormat:@"%@: %@", context, detail] : context;
    self.errorTime = NSDate.date;
    [self addNote:self.errorText forConversation:self.selectedId];
    [self updateStatusLabel];
}

- (void)addNote:(NSString *)text forConversation:(NSString *)convId {
    NSString *key = convId ?: self.selectedId ?: @"";
    NSMutableArray *list = self.notes[key] ?: (self.notes[key] = [NSMutableArray new]);
    NSDateFormatter *f = [NSDateFormatter new];
    f.timeStyle = NSDateFormatterShortStyle;
    f.dateStyle = NSDateFormatterNoStyle;
    [list insertObject:[NSString stringWithFormat:@"%@  %@", [f stringFromDate:NSDate.date], text] atIndex:0];
    if (list.count > 50) [list removeLastObject];
    if ([key isEqual:self.selectedId]) [self updateNotesSection];
}

#pragma mark Events (already on the main thread)

- (void)handleEvent:(NSDictionary *)ev {
    NSString *type = ev[@"type"];
    NSString *conv = [ev[@"conv_id"] isKindOfClass:NSString.class] ? ev[@"conv_id"] : nil;
    NSString *mid = [ev[@"message_id"] isKindOfClass:NSString.class] ? ev[@"message_id"] : nil;
    BOOL selected = conv && [conv isEqual:self.selectedId];
    if ([type isEqualToString:@"status"]) {
        [self refreshStatus];
        [self refreshModelInfo];
    } else if ([type isEqualToString:@"conversations_changed"]) {
        [self reloadConversations];
    } else if ([type isEqualToString:@"messages_changed"]) {
        if (selected) [self reloadMessages];
    } else if ([type isEqualToString:@"message_started"]) {
        self.streamingId = mid;
        self.streamingText = [NSMutableString new];
        self.generating = YES;
        if (selected) {
            NSDictionary *m = @{@"id": mid ?: @"", @"conversation_id": conv, @"role": @"assistant", @"content": @"", @"meta": @{@"status": @"generating"}};
            self.messages = [self.messages arrayByAddingObject:m];
            [self.transcript appendMessage:m];
        }
        [self refreshStatus];
    } else if ([type isEqualToString:@"delta"]) {
        if (mid && [mid isEqual:self.streamingId]) {
            [self.streamingText appendString:[ev[@"text"] isKindOfClass:NSString.class] ? ev[@"text"] : @""];
            if (selected && ![self.transcript updateMessageId:mid content:self.streamingText meta:nil]) [self reloadMessages];
        }
    } else if ([type isEqualToString:@"message_done"]) {
        NSString *content = [ev[@"content"] isKindOfClass:NSString.class] ? ev[@"content"] : @"";
        NSDictionary *meta = [ev[@"meta"] isKindOfClass:NSDictionary.class] ? ev[@"meta"] : @{@"status": @"done"};
        if ([mid isEqual:self.streamingId]) { self.streamingId = nil; self.streamingText = nil; }
        self.generating = NO;
        if (selected) {
            NSMutableArray *msgs = [self.messages mutableCopy];
            for (NSUInteger i = 0; i < msgs.count; i++) {
                if ([msgs[i][@"id"] isEqual:mid]) { NSMutableDictionary *m = [msgs[i] mutableCopy]; m[@"content"] = content; m[@"meta"] = meta; msgs[i] = m; }
            }
            self.messages = msgs;
            self.lastMeta = meta;
            if (![self.transcript updateMessageId:mid content:content meta:meta]) [self reloadMessages];
        }
        [self refreshStatus];
    } else if ([type isEqualToString:@"note"]) {
        NSString *text = [ev[@"text"] isKindOfClass:NSString.class] ? ev[@"text"] : @"";
        [self addNote:text forConversation:conv];
        self.errorText = text;
        self.errorTime = NSDate.date;
        [self updateStatusLabel];
    } else if ([type isEqualToString:@"attempt_proposed"] || [type isEqualToString:@"attempt_updated"]) {
        NSDictionary *att = [ev[@"attempt"] isKindOfClass:NSDictionary.class] ? ev[@"attempt"] : nil;
        if (!att) return;
        if (selected) [self.transcript upsertItem:[self attemptItem:att]];
        [self addNote:[NSString stringWithFormat:@"attempt %@: %@", att[@"kind"] ?: @"?", att[@"status"] ?: @"?"] forConversation:conv];
    } else if ([type isEqualToString:@"error"]) {
        NSString *detail = [ev[@"detail"] isKindOfClass:NSString.class] ? ev[@"detail"] : @"unknown error";
        self.streamingId = nil;
        self.streamingText = nil;
        self.generating = NO;
        [self addNote:[@"error: " stringByAppendingString:detail] forConversation:conv];
        self.errorText = detail;
        self.errorTime = NSDate.date;
        [self refreshStatus];
    }
}

#pragma mark Actions

- (void)newChat:(id)sender {
    NSString *newId = FrostTake(frost_conversation_create(self.engine));
    if (!newId.length) { [self showError:@"New chat failed"]; return; }
    self.selectedId = newId;
    [self reloadConversations];
    [self reloadMessages];
    [self.window makeFirstResponder:self.composer];
}

- (void)deleteConversation:(id)sender {
    NSDictionary *conv = [self selectedConversation];
    if (!conv) return;
    NSAlert *a = [NSAlert new];
    if (self.generating) {
        a.messageText = @"FROST is still generating";
        a.informativeText = @"Cancel or wait for the current reply before deleting a conversation.";
        [a beginSheetModalForWindow:self.window completionHandler:nil];
        return;
    }
    a.messageText = [NSString stringWithFormat:@"Delete “%@”?", conv[@"title"] ?: @"this conversation"];
    a.informativeText = @"The conversation and its messages are removed permanently.";
    [a addButtonWithTitle:@"Delete"].hasDestructiveAction = YES;
    [a addButtonWithTitle:@"Cancel"];
    NSString *cid = conv[@"id"];
    [a beginSheetModalForWindow:self.window completionHandler:^(NSModalResponse r) {
        if (r != NSAlertFirstButtonReturn) return;
        if (frost_conversation_delete(self.engine, cid.UTF8String) < 0) { [self showError:@"Delete failed"]; return; }
        if ([cid isEqual:self.selectedId]) self.selectedId = nil;
        [self reloadConversations];
        if (!self.conversations.count) [self newChat:nil];
    }];
}

- (void)renameConversation:(id)sender {
    NSInteger row = self.table.clickedRow >= 0 && sender == self.table.menu.itemArray.firstObject ? self.table.clickedRow : self.table.selectedRow;
    if (row < 0 || (NSUInteger)row >= self.conversations.count) return;
    [self.table selectRowIndexes:[NSIndexSet indexSetWithIndex:(NSUInteger)row] byExtendingSelection:NO];
    [self.table editColumn:0 row:row withEvent:nil select:YES];
}

- (void)send:(id)sender {
    if (!self.selectedId) return;
    NSString *text = [self.composer.string stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
    if (!text.length) return;
    if (frost_send(self.engine, self.selectedId.UTF8String, text.UTF8String) < 0) { [self showError:@"Send failed"]; return; }
    self.generating = YES;
    self.composer.string = @"";
    [self composerDidChange];
    [self refreshStatus];
}

- (void)cancel:(id)sender {
    if (frost_cancel(self.engine) < 0) [self showError:@"Cancel failed"];
}

- (void)regenerate:(id)sender {
    if (!self.selectedId) return;
    if (frost_regenerate(self.engine, self.selectedId.UTF8String) < 0) { [self showError:@"Regenerate failed"]; return; }
    self.generating = YES;
    [self refreshStatus];
}

- (void)clearContext:(id)sender {
    if (!self.selectedId) return;
    if (frost_clear_context(self.engine, self.selectedId.UTF8String) < 0) [self showError:@"Clear context failed"];
}

- (void)copyLastReply:(id)sender {
    NSString *content = [self lastMessageWithRole:@"assistant"][@"content"];
    if (![content isKindOfClass:NSString.class]) return;
    [NSPasteboard.generalPasteboard clearContents];
    [NSPasteboard.generalPasteboard setString:content forType:NSPasteboardTypeString];
}

- (void)setMode:(NSMenuItem *)item {
    if (frost_set_mode(self.engine, [item.representedObject UTF8String]) < 0) [self showError:@"Set mode failed"];
    [self refreshStatus];
}

- (void)toggleWrapCode:(id)sender {
    self.transcript.wrapCode = !self.transcript.wrapCode;
    [NSUserDefaults.standardUserDefaults setBool:self.transcript.wrapCode forKey:kWrapCodeDefault];
}

- (void)openRepository:(id)sender {
    if (!self.selectedId) return;
    NSOpenPanel *p = [NSOpenPanel openPanel];
    p.canChooseDirectories = YES;
    p.canChooseFiles = NO;
    p.allowsMultipleSelection = NO;
    p.prompt = @"Attach";
    p.message = @"Choose the repository FROST may search for this conversation.";
    NSString *cid = self.selectedId;
    [p beginSheetModalForWindow:self.window completionHandler:^(NSModalResponse r) {
        NSString *path = p.URL.path;
        if (r != NSModalResponseOK || !path) return;
        if (frost_set_repo(self.engine, cid.UTF8String, path.UTF8String) < 0) [self showError:@"Attach repository failed"];
        [self reloadConversations];
    }];
}

- (void)detachRepository:(id)sender {
    if (!self.selectedId) return;
    if (frost_set_repo(self.engine, self.selectedId.UTF8String, NULL) < 0) [self showError:@"Detach repository failed"];
    [self reloadConversations];
}

- (void)about:(id)sender {
    NSDictionary *g = [FrostJSON(frost_model_info_json(self.engine))[@"generator"] isKindOfClass:NSDictionary.class] ? FrostJSON(frost_model_info_json(self.engine))[@"generator"] : nil;
    NSAlert *a = [NSAlert new];
    a.messageText = @"FROST";
    a.informativeText = [NSString stringWithFormat:@"Version %s\n\nModel: %@\nRevision: %@\n%@ · %@", FROST_VERSION, g[@"repo"] ?: @"not loaded", Short(g[@"revision"] ?: @"—"), g[@"quantization"] ?: @"—", g[@"backend"] ?: @"—"];
    [a runModal];
}

- (void)modelInfo:(id)sender {
    NSDictionary *info = FrostJSON(frost_model_info_json(self.engine));
    NSMutableString *t = [NSMutableString new];
    [self flatten:info prefix:@"" into:t];
    NSAlert *a = [NSAlert new];
    a.messageText = @"Model Info";
    a.informativeText = t.length ? t : @"unavailable";
    [a beginSheetModalForWindow:self.window completionHandler:nil];
}
- (void)flatten:(id)v prefix:(NSString *)prefix into:(NSMutableString *)out {
    if ([v isKindOfClass:NSDictionary.class]) {
        for (NSString *k in [[v allKeys] sortedArrayUsingSelector:@selector(compare:)]) [self flatten:v[k] prefix:prefix.length ? [NSString stringWithFormat:@"%@.%@", prefix, k] : k into:out];
    } else if ([v isKindOfClass:NSArray.class]) {
        [out appendFormat:@"%@: %@\n", prefix, [v componentsJoinedByString:@", "]];
    } else {
        [out appendFormat:@"%@: %@\n", prefix, v == NSNull.null || !v ? @"null" : v];
    }
}

#pragma mark Composer

- (void)composerDidChange {
    NSLayoutManager *lm = self.composer.layoutManager;
    NSTextContainer *tc = self.composer.textContainer;
    [lm ensureLayoutForTextContainer:tc];
    CGFloat line = ceil([lm defaultLineHeightForFont:self.composer.font]);
    CGFloat used = [lm usedRectForTextContainer:tc].size.height;
    CGFloat inset = self.composer.textContainerInset.height * 2 + 2;
    self.composerHeight.constant = MIN(MAX(used, line), line * 8) + inset;
    self.composer.needsDisplay = YES;
}
- (void)textDidChange:(NSNotification *)n { [self composerDidChange]; }
- (BOOL)textView:(NSTextView *)tv doCommandBySelector:(SEL)sel {
    if (tv != self.composer) return NO;
    if (sel == @selector(insertNewline:)) {
        if (NSApp.currentEvent.modifierFlags & NSEventModifierFlagShift) { [tv insertNewlineIgnoringFieldEditor:nil]; return YES; }
        [self send:nil];
        return YES;
    }
    if (sel == @selector(insertLineBreak:)) { [tv insertNewlineIgnoringFieldEditor:nil]; return YES; }
    return NO;
}

#pragma mark Sidebar table

- (NSInteger)numberOfRowsInTableView:(NSTableView *)tv { return (NSInteger)self.conversations.count; }

- (NSView *)tableView:(NSTableView *)tv viewForTableColumn:(NSTableColumn *)col row:(NSInteger)row {
    NSTableCellView *cell = [tv makeViewWithIdentifier:@"conv" owner:self];
    if (!cell) {
        cell = [NSTableCellView new];
        cell.identifier = @"conv";
        NSTextField *title = [NSTextField labelWithString:@""];
        title.translatesAutoresizingMaskIntoConstraints = NO;
        title.editable = YES;
        title.font = [NSFont systemFontOfSize:13];
        title.lineBreakMode = NSLineBreakByTruncatingTail;
        title.delegate = self;
        NSTextField *date = [NSTextField labelWithString:@""];
        date.translatesAutoresizingMaskIntoConstraints = NO;
        date.font = [NSFont systemFontOfSize:11];
        date.textColor = NSColor.secondaryLabelColor;
        date.tag = 1;
        [cell addSubview:title];
        [cell addSubview:date];
        cell.textField = title;
        [NSLayoutConstraint activateConstraints:@[
            [title.leadingAnchor constraintEqualToAnchor:cell.leadingAnchor constant:6],
            [title.trailingAnchor constraintEqualToAnchor:cell.trailingAnchor constant:-6],
            [title.topAnchor constraintEqualToAnchor:cell.topAnchor constant:6],
            [date.leadingAnchor constraintEqualToAnchor:title.leadingAnchor],
            [date.trailingAnchor constraintEqualToAnchor:title.trailingAnchor],
            [date.topAnchor constraintEqualToAnchor:title.bottomAnchor constant:1],
        ]];
    }
    NSDictionary *c = self.conversations[(NSUInteger)row];
    cell.textField.stringValue = [c[@"title"] isKindOfClass:NSString.class] ? c[@"title"] : @"Untitled";
    static NSRelativeDateTimeFormatter *fmt;
    if (!fmt) { fmt = [NSRelativeDateTimeFormatter new]; fmt.dateTimeStyle = NSRelativeDateTimeFormatterStyleNamed; }
    NSDate *d = [NSDate dateWithTimeIntervalSince1970:[c[@"updated_at"] doubleValue] / 1000.0];
    ((NSTextField *)[cell viewWithTag:1]).stringValue = [fmt localizedStringForDate:d relativeToDate:NSDate.date];
    cell.accessibilityLabel = cell.textField.stringValue;
    return cell;
}

- (void)tableViewSelectionDidChange:(NSNotification *)n {
    if (self.suppressSelection) return;
    NSInteger row = self.table.selectedRow;
    NSString *cid = row >= 0 && (NSUInteger)row < self.conversations.count ? self.conversations[(NSUInteger)row][@"id"] : nil;
    if (!cid || [cid isEqual:self.selectedId]) return;
    self.selectedId = cid;
    [self reloadMessages];
    [self updateRepoSection];
}

- (void)controlTextDidEndEditing:(NSNotification *)n {
    NSTextField *field = n.object;
    NSInteger row = [self.table rowForView:field];
    if (row < 0 || (NSUInteger)row >= self.conversations.count) return;
    NSDictionary *c = self.conversations[(NSUInteger)row];
    NSString *title = [field.stringValue stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
    if (title.length && ![title isEqual:c[@"title"]]) {
        if (frost_conversation_rename(self.engine, [c[@"id"] UTF8String], title.UTF8String) < 0) [self showError:@"Rename failed"];
    }
    [self reloadConversations];
}

@end

#pragma mark - Entry point

int frost_ui_run(FrostEngine *engine, int argc, const char **argv) {
    @autoreleasepool {
        NSApplication *app = NSApplication.sharedApplication;
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        gController = [[FrostController alloc] initWithEngine:engine];
        app.delegate = gController;
        [app run]; // returns via -[NSApp stop:] from applicationShouldTerminate:
        [gController.window orderOut:nil];
    }
    return 0;
}

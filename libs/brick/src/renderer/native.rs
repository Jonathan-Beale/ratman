use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(feature = "brick_native")]
use super::font::{FontFamily, FontWeight, font_cache, font_registry};
use super::{BrickEvent, BrickRenderer};

// Optional dep — only available when brick_native is active.  Explicit extern
// crate is needed because optional deps aren't auto-imported in all editions.
#[cfg(feature = "brick_native")]
extern crate image;

// ── Window commands ──────────────────────────────────────────────────────────
//
// Click handlers that need to affect the host window (minimize, close, etc.)
// store a command here; the event loop drains it synchronously after each click.
// Thread-local because handlers run on the event-loop thread.

/// Commands a click handler can schedule for the enclosing window.
#[derive(Clone, Copy, Debug)]
pub enum WindowCmd {
    Close,
    Minimize,
    ToggleMaximize,
}

thread_local! {
    static PENDING_WINDOW_CMD: Cell<Option<WindowCmd>> = Cell::new(None);
}

// ── Hover tracking ───────────────────────────────────────────────────────────
//
// Stores the Rc pointer of the node currently under the cursor so paint_node
// can apply data-hover-fill without threading extra state through every call.
// Thread-local because the event loop and paint pass run on the same thread.
thread_local! {
    static HOVER_PTR: Cell<usize> = Cell::new(0);
}

// ── Scroll state ─────────────────────────────────────────────────────────────
//
// SCROLL_OFFSETS: stable-id → scroll-y offset in pixels for data-scroll-y nodes.
// Key is `data-scroll-id` attribute if set (survives tree rebuilds), otherwise
// the Rc pointer formatted as a decimal string (only stable within one frame).
// PAINT_CLIP: active scissor rect (x, y, w, h) pushed by scroll containers so
// children that extend outside the viewport don't bleed into adjacent panels.
// PENDING_CHARS: characters typed since the last take_pending_chars() call.
thread_local! {
    static SCROLL_OFFSETS: RefCell<HashMap<String, i32>> = RefCell::new(HashMap::new());
    static PAINT_CLIP: Cell<Option<(i32, i32, i32, i32)>> = Cell::new(None);
    static PENDING_CHARS: RefCell<String> = RefCell::new(String::new());
}

// ── Scene tree ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default, Debug)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

/// Called after the scene tree is painted. Receives the full pixel buffer,
/// buffer stride, and the bounding rect of the hooked node.
pub type PaintHook = Arc<dyn Fn(&mut [u32], u32, Rect) + Send + Sync>;

// ── Layout drag types ─────────────────────────────────────────────────────────
//
// These types are used by the IDE's area tree layout engine. They live here so
// the event loop (run_window_with_drag) can maintain DragState without importing
// from a higher-level crate.

/// Path into an AreaNode tree identifying a Split node (each element: 0=a, 1=b).
#[derive(Clone, Debug, Default)]
pub struct DividerRef(pub Vec<usize>);

/// Corner of an Area leaf used for AZone split/join gestures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AZoneCorner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Path into an AreaNode tree identifying a Leaf + which corner was grabbed.
#[derive(Clone, Debug)]
pub struct AZoneRef {
    pub path: Vec<usize>,
    pub corner: AZoneCorner,
}

/// Current drag gesture in the IDE layout.
pub enum DragState {
    None,
    DraggingDivider {
        divider: DividerRef,
        start_ratio: f32,
        start_x: i32,
        start_y: i32,
    },
    DraggingAZone {
        azone: AZoneRef,
        ghost_rect: Option<Rect>,
    },
    DraggingTab {
        pane_id: String,
        origin: (i32, i32),
        ghost_pos: (i32, i32),
    },
}

impl Default for DragState {
    fn default() -> Self {
        DragState::None
    }
}

/// Semi-transparent rectangle drawn above the scene during drag gestures.
pub struct DragOverlay {
    pub rect: Rect,
    /// xRGB color (top byte unused, same format as softbuffer pixel buffer).
    pub color: u32,
    /// Alpha: 0 = fully transparent, 255 = fully opaque.
    pub alpha: u8,
}

/// Called on mouse press; returns the drag state to enter (or `DragState::None`).
pub type OnPressFn = Box<dyn FnMut(i32, i32) -> DragState>;

/// Called on cursor move; returns updated drag state + overlays to paint.
pub type OnMoveFn = Box<dyn FnMut(&DragState, i32, i32) -> (DragState, Vec<DragOverlay>)>;

/// Called on mouse release to commit the gesture (e.g. split_leaf, join_leaves).
pub type OnReleaseFn = Box<dyn FnMut(&DragState, i32, i32)>;

/// Called on keyboard input; returns true if the event was consumed.
/// `key` is the `{:?}` debug name of the winit `KeyCode` (e.g. "Space", "KeyV", "ArrowLeft").
/// `pressed` is true on press, false on release.
pub type OnKeyFn = Box<dyn FnMut(&str, bool) -> bool>;

/// Called after the scene tree + overlays are painted. Can draw additional content
/// (e.g. tab headers) directly into the pixel buffer.
pub type OnPaintFn = Box<dyn FnMut(&mut [u32], u32, u32, u32)>;

/// Called on right mouse press. The first argument is the value of the nearest
/// `data-context` attribute walking up from the hit node (e.g. `"session-message"`,
/// `"fleet-row"`, `"file-entry"`), or `None` if no ancestor carries one. The remaining
/// args are the cursor's window-relative pixel position. Implementations typically
/// call `ContextMenu::show(items, x, y)` based on the context tag.
pub type OnRightClickFn = Box<dyn FnMut(Option<String>, i32, i32)>;

// ── Focus management ──────────────────────────────────────────────────────────

/// Modifier key state at the time of a keyboard event.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyModifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    /// Cmd (macOS) / Win (Windows) / Super (Linux).
    pub logo: bool,
}

/// A keyboard event delivered to [`OnKeyExtFn`].
///
/// Produced for both physical key presses and IME commit events.
#[derive(Clone, Debug)]
pub struct KeyInput {
    /// Physical key code debug name, e.g. `"KeyA"`, `"Enter"`, `"ArrowLeft"`.
    /// Empty for IME commit events (use `text` instead).
    pub code: String,
    /// Inserted text, if any. Respects keyboard layout, dead keys, and Shift.
    /// Always set for IME commit events.
    pub text: Option<String>,
    /// `true` on key press, `false` on release.
    pub pressed: bool,
    /// `true` when the OS is auto-repeating a held key.
    pub repeat: bool,
    /// Modifier keys active at event time.
    pub mods: KeyModifiers,
}

/// Tracks which widget currently holds keyboard focus.
///
/// Backed by `Rc<RefCell<…>>` — cheaply cloneable within a single thread.
/// All clones share the same focus cell, so any clone can update focus and
/// every other clone sees the change immediately.
///
/// Widget authors set `data-focus-id="my-widget"` on their container node;
/// [`mount_root_with_focus`][NativeRenderer::mount_root_with_focus] updates
/// focus automatically on left-click.
#[derive(Clone, Default)]
pub struct FocusManager(Rc<RefCell<Option<String>>>);

impl FocusManager {
    pub fn new() -> Self {
        FocusManager(Rc::new(RefCell::new(None)))
    }

    /// Return the `data-focus-id` of the focused widget, or `None` if no
    /// Brick widget holds focus (default: nvim grid receives input).
    pub fn focused(&self) -> Option<String> {
        self.0.borrow().clone()
    }

    /// Give focus to the widget identified by `data-focus-id = id`.
    pub fn request_focus(&self, id: impl Into<String>) {
        *self.0.borrow_mut() = Some(id.into());
    }

    /// Clear focus; reverts to platform default (nvim grid input).
    pub fn clear_focus(&self) {
        *self.0.borrow_mut() = None;
    }

    /// `true` if the given id is currently focused.
    pub fn is_focused(&self, id: &str) -> bool {
        self.0.borrow().as_deref() == Some(id)
    }
}

/// Called on every keyboard event with the full key descriptor and current
/// focus state. Return `true` to consume the event and request an immediate
/// redraw.
pub type OnKeyExtFn = Box<dyn FnMut(&KeyInput, &FocusManager) -> bool>;

enum NodeKind {
    Container,
    Text(String),
}

struct NativeNodeInner {
    kind: NodeKind,
    attrs: HashMap<String, String>,
    classes: Vec<String>,
    children: Vec<NativeNode>,
    event_handlers: Vec<(String, Rc<dyn Fn(BrickEvent)>)>,
    bounds: Rect,
}

/// A node in the native retained-mode scene tree.
///
/// Cheap to clone — backed by `Rc<RefCell<...>>`.
#[derive(Clone)]
pub struct NativeNode(Rc<RefCell<NativeNodeInner>>);

impl NativeNode {
    fn new(kind: NodeKind) -> Self {
        NativeNode(Rc::new(RefCell::new(NativeNodeInner {
            kind,
            attrs: HashMap::new(),
            classes: Vec::new(),
            children: Vec::new(),
            event_handlers: Vec::new(),
            bounds: Rect::default(),
        })))
    }
}

// ── Window options ───────────────────────────────────────────────────────────

/// Native window creation options — transparency, decorations, stacking, and
/// click-through.
///
/// Built for overlay use cases (an always-on-top HUD painted over another
/// application's window) but deliberately generic: nothing here is
/// product-specific. The defaults reproduce the historical opaque, decorated,
/// normally-stacked window so existing callers are unaffected.
///
/// **softbuffer alpha note:** softbuffer presents 32-bit pixels but does not
/// itself define the alpha byte. On platforms whose compositor honors an ARGB
/// visual (modern Windows/macOS and most Linux compositors via XWayland), a
/// winit window created with `transparent: true` *does* use the high byte as
/// alpha. To make this robust regardless of what the paint pipeline writes into
/// the high byte, the loop post-processes the framebuffer in transparent mode:
/// pixels still equal to `background` become fully transparent, every painted
/// pixel becomes fully opaque (see [`apply_overlay_alpha`]). Net effect: the
/// window shows through everywhere except where the scene tree painted.
#[derive(Clone)]
pub struct WindowOptions {
    /// Window title (also the X11 `WM_NAME`).
    pub title: String,
    /// Initial inner width in logical pixels.
    pub width: u32,
    /// Initial inner height in logical pixels.
    pub height: u32,
    /// Request a per-pixel-alpha framebuffer so `background`'s alpha shows
    /// through to whatever is behind the window.
    pub transparent: bool,
    /// Draw OS window decorations (title bar, borders). Overlays set `false`.
    pub decorations: bool,
    /// Keep the window above all normal windows (`WindowLevel::AlwaysOnTop`).
    pub always_on_top: bool,
    /// Make the whole window ignore pointer input so clicks fall through to the
    /// application beneath it. This is a *window-global* flag — winit's
    /// `set_cursor_hittest` has no per-region granularity. Per-panel
    /// interactivity (interactive only where panels are painted) requires the
    /// X11 shape/input-region extension and is deferred to a later phase.
    pub click_through: bool,
    /// Background fill applied before painting the scene each frame, as
    /// `0xAARRGGBB`. For a transparent overlay use `0x0000_0000` so only painted
    /// panels are visible.
    pub background: u32,
    /// Optional post-paint tap: called after every completed frame with the
    /// raw ARGB pixel buffer, width, and height. Used by [`HookedOverlayBackend`]
    /// to mirror the framebuffer into shared memory so an injected DLL can
    /// composite it into the game's own backbuffer.
    ///
    /// `None` (the default) is a no-op. When set, the closure runs on the
    /// event-loop thread after each [`RedrawRequested`] and before the
    /// softbuffer surface present.
    pub framebuffer_sink: Option<Arc<dyn Fn(&[u32], u32, u32) + Send + Sync>>,
    /// Application icon shown in the OS title bar and taskbar.
    /// Stored as raw RGBA bytes (`width × height × 4`) plus dimensions.
    /// `None` leaves the OS-default icon.
    pub icon_rgba: Option<(Vec<u8>, u32, u32)>,
    /// Initial outer-window position in logical pixels.  `None` lets the OS choose.
    pub position: Option<(i32, i32)>,
    /// Activate (bring to foreground / steal focus) on creation.
    /// Set to `false` for background / hot-reload windows.
    pub focus_on_open: bool,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self {
            title: "brick".to_string(),
            width: 1200,
            height: 800,
            transparent: false,
            decorations: true,
            always_on_top: false,
            click_through: false,
            background: 0x001E_1E2E, // Catppuccin Mocha base
            framebuffer_sink: None,
            icon_rgba: None,
            position: None,
            focus_on_open: true,
        }
    }
}

impl std::fmt::Debug for WindowOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowOptions")
            .field("title", &self.title)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("transparent", &self.transparent)
            .field("decorations", &self.decorations)
            .field("always_on_top", &self.always_on_top)
            .field("click_through", &self.click_through)
            .field("background", &self.background)
            .field("framebuffer_sink", &self.framebuffer_sink.as_ref().map(|_| "<sink>"))
            .field("icon_rgba", &self.icon_rgba.as_ref().map(|_| "<icon>"))
            .field("position", &self.position)
            .field("focus_on_open", &self.focus_on_open)
            .finish()
    }
}

impl WindowOptions {
    /// Preset for an in-game / always-on-top HUD overlay: borderless,
    /// always-on-top, transparent, click-through, with a fully-transparent
    /// background. Set [`width`](Self::width)/[`height`](Self::height) (and
    /// later the outer position) to the tracked window's rect before mounting.
    pub fn overlay() -> Self {
        Self {
            title: "brick-overlay".to_string(),
            transparent: true,
            decorations: false,
            always_on_top: true,
            click_through: true,
            background: 0x0000_0000,
            framebuffer_sink: None,
            ..Self::default()
        }
    }
}

/// Rewrite the framebuffer's alpha byte for transparent-window presentation.
///
/// Pixels whose RGB still equals `background`'s RGB are made fully transparent
/// (`0x0000_0000`); every other (painted) pixel is forced fully opaque
/// (`| 0xFF00_0000`). This makes per-pixel transparency independent of whatever
/// the paint routines left in the high byte.
fn apply_overlay_alpha(buf: &mut [u32], background: u32) {
    let bg_rgb = background & 0x00FF_FFFF;
    for px in buf.iter_mut() {
        if (*px & 0x00FF_FFFF) == bg_rgb {
            *px = 0x0000_0000;
        } else {
            *px |= 0xFF00_0000;
        }
    }
}

// ── BrickRenderer implementation ─────────────────────────────────────────────

/// Native desktop renderer backed by winit + softbuffer.
///
/// ZST — all methods are static dispatch with no per-instance state.
/// `mount_root` drives the winit event loop; it returns when the window closes.
pub struct NativeRenderer;

impl BrickRenderer for NativeRenderer {
    type Node = NativeNode;

    fn element(_tag: &str) -> NativeNode {
        NativeNode::new(NodeKind::Container)
    }

    fn text(content: &str) -> NativeNode {
        NativeNode::new(NodeKind::Text(content.to_string()))
    }

    fn fragment(children: Vec<NativeNode>) -> NativeNode {
        let node = NativeNode::new(NodeKind::Container);
        for child in children {
            node.0.borrow_mut().children.push(child);
        }
        node
    }

    fn set_attr(node: &NativeNode, key: &str, value: &str) {
        node.0
            .borrow_mut()
            .attrs
            .insert(key.to_string(), value.to_string());
    }

    fn remove_attr(node: &NativeNode, key: &str) {
        node.0.borrow_mut().attrs.remove(key);
    }

    fn set_text(node: &NativeNode, content: &str) {
        node.0.borrow_mut().kind = NodeKind::Text(content.to_string());
    }

    fn append(parent: &NativeNode, child: &NativeNode) {
        parent.0.borrow_mut().children.push(child.clone());
    }

    fn remove_child(parent: &NativeNode, child: &NativeNode) {
        parent
            .0
            .borrow_mut()
            .children
            .retain(|c| !Rc::ptr_eq(&c.0, &child.0));
    }

    fn insert_before(parent: &NativeNode, new_node: &NativeNode, ref_node: &NativeNode) {
        let mut inner = parent.0.borrow_mut();
        let pos = inner
            .children
            .iter()
            .position(|c| Rc::ptr_eq(&c.0, &ref_node.0))
            .unwrap_or(inner.children.len());
        inner.children.insert(pos, new_node.clone());
    }

    fn child_count(node: &NativeNode) -> usize {
        node.0.borrow().children.len()
    }

    fn nth_child(node: &NativeNode, n: usize) -> Option<NativeNode> {
        node.0.borrow().children.get(n).cloned()
    }

    fn add_class(node: &NativeNode, class: &str) {
        node.0.borrow_mut().classes.push(class.to_string());
    }

    fn on_event(node: &NativeNode, event: &str, handler: Box<dyn Fn(BrickEvent) + 'static>) {
        node.0
            .borrow_mut()
            .event_handlers
            .push((event.to_string(), Rc::from(handler)));
    }

    fn remove_node(node: &NativeNode) {
        // Detach from the scene by clearing children and handlers; the parent
        // still holds the Rc so the node isn't freed until the parent drops it.
        let mut inner = node.0.borrow_mut();
        inner.children.clear();
        inner.event_handlers.clear();
    }

    fn mount_root(node: NativeNode) {
        run_window(node, None, vec![], None, None, default_window(), None);
    }
}

/// The default window configuration used by the `mount_root*` family:
/// an opaque, decorated 1200×800 window titled `"Brick"`.
fn default_window() -> WindowOptions {
    WindowOptions {
        title: "Brick".to_string(),
        ..WindowOptions::default()
    }
}

impl NativeRenderer {
    /// Schedule a window command (minimize / maximize / close) from a click handler.
    ///
    /// Must be called from within a `brick` click handler running on the event-loop
    /// thread. The event loop drains the command immediately after the handler returns.
    pub fn request_window_cmd(cmd: WindowCmd) {
        PENDING_WINDOW_CMD.with(|p| p.set(Some(cmd)));
    }

    /// Drain all characters typed since the last call.
    ///
    /// Includes printable characters plus sentinels:
    /// - `'\x08'` — Backspace
    /// - `'\x1b'` — Escape
    ///
    /// Call this each frame (e.g., in `on_pre_paint`) to process text input
    /// without needing to pass an `on_key` callback to `mount`.
    pub fn take_pending_chars() -> String {
        PENDING_CHARS.with(|b| std::mem::take(&mut *b.borrow_mut()))
    }

    /// Like `mount_root` but accepts an optional repaint sender.
    ///
    /// When `repaint_tx` is `Some`, async tasks can wake the event loop by
    /// sending `()` through it. Used by pane watchers to trigger redraws when
    /// app state changes without polling.
    /// The caller creates a `std::sync::mpsc::sync_channel(1)`, passes the
    /// `SyncSender` to async tasks, and hands the `Receiver` here.
    pub fn mount_root_with_repaint(node: NativeNode, repaint_rx: std::sync::mpsc::Receiver<()>) {
        run_window(
            node,
            Some(repaint_rx),
            vec![],
            None,
            None,
            default_window(),
            None,
        );
    }

    /// Mount with explicit [`WindowOptions`] — the entry point for transparent,
    /// borderless, always-on-top, and/or click-through windows (e.g. an in-game
    /// overlay HUD). Equivalent to [`mount_root_with_hooks`] plus window-flag
    /// control. Pass [`WindowOptions::overlay`] for the HUD preset.
    pub fn mount_root_with_options(
        node: NativeNode,
        repaint_rx: std::sync::mpsc::Receiver<()>,
        hooks: Vec<(NativeNode, PaintHook)>,
        options: WindowOptions,
    ) {
        run_window(node, Some(repaint_rx), hooks, None, None, options, None);
    }

    /// Mount an animated overlay HUD: explicit [`WindowOptions`], a per-frame
    /// `on_pre_paint` callback for rebuilding the scene from fresh state, and a
    /// `tick` cadence that wakes the event loop on a fixed interval so countdowns
    /// and other time-based UI advance without external input.
    ///
    /// `tick = Some(d)` arms a `ControlFlow::WaitUntil` timer that fires every
    /// `d`; each fire requests a redraw, so `on_pre_paint` runs and the scene
    /// repaints at roughly `1/d` Hz (plus any input- or `repaint_rx`-driven
    /// frames). `tick = None` behaves like [`mount_root_with_options`] (redraws
    /// only on input or repaint wakeups).
    pub fn mount_root_animated(
        node: NativeNode,
        repaint_rx: std::sync::mpsc::Receiver<()>,
        options: WindowOptions,
        tick: Option<std::time::Duration>,
        on_pre_paint: Box<dyn FnMut(u32, u32)>,
    ) {
        run_window(
            node,
            Some(repaint_rx),
            vec![],
            None,
            Some(on_pre_paint),
            options,
            tick,
        );
    }

    /// Like `mount_root_with_repaint` but also registers paint hooks.
    ///
    /// Each hook is a `(NativeNode, PaintHook)` pair. After `paint_node` runs
    /// on every `RedrawRequested` event, hooks whose node has non-zero bounds
    /// are called with the pixel buffer, its stride, and the node's bounding
    /// rect. Node identity is matched by pointer equality (`Rc::ptr_eq`).
    pub fn mount_root_with_hooks(
        node: NativeNode,
        repaint_rx: std::sync::mpsc::Receiver<()>,
        hooks: Vec<(NativeNode, PaintHook)>,
    ) {
        run_window(
            node,
            Some(repaint_rx),
            hooks,
            None,
            None,
            default_window(),
            None,
        );
    }

    /// Full-featured mount: repaint wakeup, paint hooks, keyboard handler, and
    /// a pre-paint callback for dynamic layout (e.g. dock flex-fraction updates).
    ///
    /// - `on_key(ch)` — called on every printable key press; return `true` to
    ///   request an immediate redraw (e.g. after a dock toggle).
    /// - `on_pre_paint(win_w, win_h)` — called before each `layout_node` +
    ///   `paint_node` pass so the caller can update attributes in-place (e.g.
    ///   `DockLayout::update_layout`). The root node is re-laid-out afterwards.
    pub fn mount_root_with_keyboard(
        node: NativeNode,
        repaint_rx: std::sync::mpsc::Receiver<()>,
        hooks: Vec<(NativeNode, PaintHook)>,
        on_key: Box<dyn FnMut(char) -> bool>,
        on_pre_paint: Box<dyn FnMut(u32, u32)>,
    ) {
        run_window(
            node,
            Some(repaint_rx),
            hooks,
            Some(on_key),
            Some(on_pre_paint),
            default_window(),
            None,
        );
    }

    /// Like `mount_root_with_hooks` but adds a layout-aware drag state machine.
    ///
    /// Callers provide five callbacks:
    /// - `on_press`: hit-test cursor position → new DragState (or None)
    /// - `on_move`: update drag state + return overlays to paint
    /// - `on_release`: commit the gesture (e.g. call split_leaf / join_leaves)
    /// - `on_key`: handle keyboard shortcuts; return true if consumed
    /// - `on_paint`: extra painting pass (e.g. tab header strips)
    ///
    /// The event loop maintains `DragState` internally and dispatches to these
    /// callbacks. Ghost overlays (AZone preview, divider highlight) are painted
    /// by blending `DragOverlay` rects into the pixel buffer after `paint_node`.
    /// Lay out and paint a free-floating overlay node tree into `buf`.
    ///
    /// Intended for `OnPaintFn` closures that want to paint a Brick subtree
    /// (e.g. a context menu) on top of the main scene each frame. The overlay
    /// is positioned via `data-overlay` + `data-x` / `data-y` / `data-w` /
    /// `data-h` attrs and bypasses normal flow.
    ///
    /// Reactivity: rebuild `overlay` from `Signal` reads inside the closure —
    /// fresh tree per frame keeps state and rendering in sync without observers.
    pub fn paint_overlay(buf: &mut [u32], buf_w: u32, overlay: &NativeNode) {
        layout_node(overlay, 0, 0, buf_w);
        paint_node(overlay, buf, buf_w);
    }

    pub fn mount_root_with_drag_handler(
        node: NativeNode,
        repaint_rx: std::sync::mpsc::Receiver<()>,
        hooks: Vec<(NativeNode, PaintHook)>,
        on_press: OnPressFn,
        on_move: OnMoveFn,
        on_release: OnReleaseFn,
        on_key: OnKeyFn,
        on_paint: OnPaintFn,
        on_right_click: OnRightClickFn,
    ) {
        run_window_with_drag(
            node,
            Some(repaint_rx),
            hooks,
            on_press,
            on_move,
            on_release,
            on_key,
            on_paint,
            on_right_click,
        );
    }

    /// Like `mount_root_with_hooks` but adds keyboard routing through a
    /// [`FocusManager`].
    ///
    /// - Left-clicks on nodes with `data-focus-id` automatically update focus.
    ///   Clicking outside any focusable node clears focus.
    /// - `on_key` receives every [`KeyInput`] and the current [`FocusManager`];
    ///   return `true` to request a redraw.
    /// - IME commit events are synthesised as a `KeyInput` with an empty `code`
    ///   and the committed string in `text`.
    /// - A Catppuccin Mauve focus ring is painted over the focused widget's
    ///   bounds on every `RedrawRequested`.
    pub fn mount_root_with_focus(
        node: NativeNode,
        repaint_rx: std::sync::mpsc::Receiver<()>,
        hooks: Vec<(NativeNode, PaintHook)>,
        focus: FocusManager,
        on_key: OnKeyExtFn,
    ) {
        run_window_with_focus(node, Some(repaint_rx), hooks, focus, on_key, None);
    }

    /// Spawn a background thread that pre-decodes and caches images before they
    /// are needed at render time.  Call once at startup with all paths and pixel
    /// sizes that will be used — subsequent `data-image` paints will hit the warm
    /// cache instead of decoding on the render thread.
    ///
    /// Each entry is `(path, width, height)` matching the `data-w` / `data-h` of
    /// the node that will display the image.
    #[cfg(feature = "brick_native")]
    pub fn preload_images(entries: Vec<(String, u32, u32)>) {
        std::thread::Builder::new()
            .name("brick-img-preload".into())
            .spawn(move || {
                for (path, w, h) in entries {
                    load_scaled_image(&path, w, h);
                }
            })
            .ok();
    }
}

// ── Window loop ──────────────────────────────────────────────────────────────
//
// winit 0.29 API: EventLoop::new() → Result; closure takes (Event, &EventLoopWindowTarget).
// softbuffer 0.4 API: Context::new(D) and Surface::new(&ctx, W) — both safe, no `unsafe`.
//
// Rc<Window> is used so the window handle can be owned by both Context and Surface
// without lifetime complications.

fn run_window(
    root: NativeNode,
    repaint_rx: Option<std::sync::mpsc::Receiver<()>>,
    hooks: Vec<(NativeNode, PaintHook)>,
    mut on_key: Option<Box<dyn FnMut(char) -> bool>>,
    mut on_pre_paint: Option<Box<dyn FnMut(u32, u32)>>,
    options: WindowOptions,
    tick: Option<std::time::Duration>,
) {
    use std::rc::Rc;
    use winit::event::{ElementState, Event, MouseButton, StartCause, WindowEvent};
    use winit::event_loop::{ControlFlow, EventLoop, EventLoopBuilder};
    use winit::window::{WindowBuilder, WindowLevel};

    // Force X11 backend. Wayland via XDG portal is broken in WSL2 environments
    // (socket present but portal times out → Broken pipe on first event dispatch).
    // X11 works reliably on :0 via XWayland. When running on a native Linux
    // desktop this is still correct — X11 through XWayland has no overhead cost
    // at our scale and avoids the fragile portal dependency entirely.
    #[cfg(target_os = "linux")]
    let event_loop = {
        use winit::platform::x11::EventLoopBuilderExtX11;
        EventLoopBuilder::new()
            .with_x11()
            .build()
            .expect("event loop create (x11)")
    };
    #[cfg(not(target_os = "linux"))]
    let event_loop = EventLoop::new().expect("event loop create");
    let mut wb = WindowBuilder::new()
        .with_title(options.title.clone())
        .with_inner_size(winit::dpi::LogicalSize::new(options.width, options.height))
        .with_transparent(options.transparent)
        .with_decorations(options.decorations)
        .with_active(options.focus_on_open)
        .with_window_level(if options.always_on_top {
            WindowLevel::AlwaysOnTop
        } else {
            WindowLevel::Normal
        });
    if let Some((x, y)) = options.position {
        wb = wb.with_position(winit::dpi::LogicalPosition::new(x, y));
    }
    let window = Rc::new(wb.build(&event_loop).expect("window create"));

    // Whole-window click-through: when set, the window ignores pointer input so
    // clicks fall through to whatever is beneath it (the game). Best-effort —
    // some platforms/compositors do not support it; ignore the error rather than
    // panic so the overlay still renders.
    if options.click_through {
        let _ = window.set_cursor_hittest(false);
    }

    // Window icon (title bar + taskbar). Best-effort — ignore errors on platforms
    // that don't support it (e.g. macOS, some Wayland compositors).
    if let Some((rgba, w, h)) = options.icon_rgba.clone() {
        if let Ok(icon) = winit::window::Icon::from_rgba(rgba, w, h) {
            window.set_window_icon(Some(icon));
        }
    }

    let context = softbuffer::Context::new(Rc::clone(&window)).expect("softbuffer context create");
    let mut surface =
        softbuffer::Surface::new(&context, Rc::clone(&window)).expect("softbuffer surface create");

    let mut win_w: u32 = options.width;
    let mut win_h: u32 = options.height;
    let mut cursor = (0.0_f64, 0.0_f64);

    event_loop
        .run(move |event, elwt| {
            // Drain the repaint channel; if any pane sent a wakeup, schedule redraw.
            if let Some(ref rx) = repaint_rx {
                if rx.try_recv().is_ok() {
                    window.request_redraw();
                }
            }

            // With a `tick` cadence, arm a timer so the loop wakes on a fixed
            // interval even with no input; otherwise sleep until the next event.
            match tick {
                Some(d) => {
                    elwt.set_control_flow(ControlFlow::WaitUntil(std::time::Instant::now() + d))
                }
                None => elwt.set_control_flow(ControlFlow::Wait),
            }

            // The timer wake is delivered as a `NewEvents(ResumeTimeReached)`,
            // not a window event — turn it into a redraw request so `on_pre_paint`
            // runs and time-based UI (countdowns) advances. Requesting a redraw
            // here (rather than unconditionally) avoids a busy-loop on RedrawRequested.
            if let Event::NewEvents(StartCause::ResumeTimeReached { .. }) = event {
                // Re-assert topmost on every tick so the overlay stays above
                // game windows that claim HWND_TOPMOST after us (e.g. League
                // creating its in-game window after the overlay was created).
                if options.always_on_top {
                    window.set_window_level(WindowLevel::AlwaysOnTop);
                }
                window.request_redraw();
                return;
            }

            let Event::WindowEvent { event, .. } = event else {
                return;
            };

            match event {
                WindowEvent::CloseRequested => elwt.exit(),

                WindowEvent::Resized(size) => {
                    win_w = size.width.max(1);
                    win_h = size.height.max(1);
                    window.request_redraw();
                }

                WindowEvent::CursorMoved { position, .. } => {
                    cursor = (position.x, position.y);
                    window.request_redraw();
                }

                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    // `data-drag-window="true"` marks a region that initiates OS
                    // window-drag instead of firing a click.
                    if hit_test_attr(&root, cursor.0, cursor.1, "data-drag-window", "true") {
                        let _ = window.drag_window();
                    } else if let Some(hit) = hit_test(&root, cursor.0, cursor.1) {
                        fire_event(&hit, "click", BrickEvent::ClickAt(cursor.0, cursor.1));
                        // Execute any window command scheduled by the click handler.
                        if let Some(cmd) = PENDING_WINDOW_CMD.with(|p| p.take()) {
                            match cmd {
                                WindowCmd::Close         => elwt.exit(),
                                WindowCmd::Minimize      => window.set_minimized(true),
                                WindowCmd::ToggleMaximize =>
                                    window.set_maximized(!window.is_maximized()),
                            }
                        }
                    }
                    window.request_redraw();
                }

                WindowEvent::MouseWheel { delta, .. } => {
                    use winit::event::MouseScrollDelta;
                    let scroll_px = match delta {
                        MouseScrollDelta::LineDelta(_, y) => (-(y as f64) * 40.0) as i32,
                        MouseScrollDelta::PixelDelta(d) => -(d.y as i32),
                    };
                    if let Some(scroll_node) = find_scroll_ancestor(&root, cursor.0, cursor.1) {
                        let key = scroll_node.0.borrow().attrs.get("data-scroll-id")
                            .cloned()
                            .unwrap_or_else(|| format!("{}", Rc::as_ptr(&scroll_node.0) as usize));
                        SCROLL_OFFSETS.with(|s| {
                            let mut map = s.borrow_mut();
                            let offset = map.entry(key).or_insert(0);
                            *offset = (*offset + scroll_px).max(0);
                        });
                        window.request_redraw();
                    }
                }

                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed {
                        // Use the inserted text (handles Space, printable chars, and
                        // layout-specific characters correctly across keyboard layouts).
                        if let Some(text) = event.text.as_ref() {
                            // Always buffer for take_pending_chars().
                            PENDING_CHARS.with(|b| b.borrow_mut().push_str(text.as_str()));
                            // Request redraw immediately so search responds on the same frame.
                            window.request_redraw();
                            if let Some(c) = text.chars().next() {
                                if let Some(ref mut handler) = on_key {
                                    handler(c);
                                }
                            }
                        } else {
                            // Named keys with no text representation — buffer sentinels.
                            use winit::keyboard::{Key, NamedKey};
                            let mut pushed = false;
                            match &event.logical_key {
                                Key::Named(NamedKey::Backspace) => {
                                    PENDING_CHARS.with(|b| b.borrow_mut().push('\x08'));
                                    pushed = true;
                                }
                                Key::Named(NamedKey::Escape) => {
                                    PENDING_CHARS.with(|b| b.borrow_mut().push('\x1b'));
                                    pushed = true;
                                }
                                Key::Named(NamedKey::ArrowLeft) => {
                                    PENDING_CHARS.with(|b| b.borrow_mut().push('\x1c'));
                                    pushed = true;
                                }
                                Key::Named(NamedKey::ArrowRight) => {
                                    PENDING_CHARS.with(|b| b.borrow_mut().push('\x1d'));
                                    pushed = true;
                                }
                                Key::Named(NamedKey::ArrowUp) => {
                                    PENDING_CHARS.with(|b| b.borrow_mut().push('\x1e'));
                                    pushed = true;
                                }
                                Key::Named(NamedKey::ArrowDown) => {
                                    PENDING_CHARS.with(|b| b.borrow_mut().push('\x1f'));
                                    pushed = true;
                                }
                                _ => {}
                            }
                            if pushed { window.request_redraw(); }
                        }
                    }
                }

                WindowEvent::RedrawRequested => {
                    if let (Some(w), Some(h)) = (NonZeroU32::new(win_w), NonZeroU32::new(win_h)) {
                        surface.resize(w, h).expect("softbuffer resize");
                        let mut buf = surface.buffer_mut().expect("softbuffer buffer_mut");

                        // Let callers update layout attributes (e.g. dock flex fractions)
                        // before the layout pass so the new dimensions are reflected
                        // immediately on this frame.
                        if let Some(ref mut pre) = on_pre_paint {
                            pre(win_w, win_h);
                        }

                        layout_node(&root, 0, 0, win_w);
                        // Overlay-positioned children don't contribute to the
                        // parent's computed height, leaving root with h=0 when
                        // every direct child is an abs_panel. Force the root
                        // bounds to cover the full window so hit_test reaches
                        // those children.
                        root.0.borrow_mut().bounds = Rect { x: 0, y: 0, w: win_w, h: win_h };
                        // Update hover target after layout so paint_node sees
                        // the current frame's node for data-hover-fill.
                        let hover_ptr = hit_test(&root, cursor.0, cursor.1)
                            .map(|n| Rc::as_ptr(&n.0) as usize)
                            .unwrap_or(0);
                        HOVER_PTR.with(|p| p.set(hover_ptr));
                        buf.fill(options.background);
                        paint_node(&root, &mut buf, win_w);

                        for (hook_node, hook_fn) in &hooks {
                            let bounds = hook_node.0.borrow().bounds;
                            if bounds.w > 0 && bounds.h > 0 {
                                hook_fn(&mut buf, win_w, bounds);
                            }
                        }

                        // Per-pixel transparency: make the background show through
                        // and every painted pixel opaque. See WindowOptions docs.
                        if options.transparent {
                            apply_overlay_alpha(&mut buf, options.background);
                        }

                        if let Some(ref sink) = options.framebuffer_sink {
                            sink(&buf, win_w, win_h);
                        }

                        buf.present().expect("softbuffer present");
                    }
                }

                _ => {}
            }
        })
        .expect("event loop run");
}

// ── Layout-aware drag event loop ─────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_window_with_drag(
    root: NativeNode,
    repaint_rx: Option<std::sync::mpsc::Receiver<()>>,
    hooks: Vec<(NativeNode, PaintHook)>,
    mut on_press: OnPressFn,
    mut on_move: OnMoveFn,
    mut on_release: OnReleaseFn,
    mut on_key: OnKeyFn,
    mut on_paint: OnPaintFn,
    mut on_right_click: OnRightClickFn,
) {
    use std::rc::Rc;
    use winit::event::{ElementState, Event, MouseButton, WindowEvent};
    use winit::event_loop::{ControlFlow, EventLoop, EventLoopBuilder};
    use winit::window::WindowBuilder;

    #[cfg(target_os = "linux")]
    let event_loop = {
        use winit::platform::x11::EventLoopBuilderExtX11;
        EventLoopBuilder::new()
            .with_x11()
            .build()
            .expect("event loop create (x11)")
    };
    #[cfg(not(target_os = "linux"))]
    let event_loop = EventLoop::new().expect("event loop create");

    let window = Rc::new(
        WindowBuilder::new()
            .with_title("Brick")
            .with_inner_size(winit::dpi::LogicalSize::new(1200u32, 800u32))
            .build(&event_loop)
            .expect("window create"),
    );

    let context = softbuffer::Context::new(Rc::clone(&window)).expect("softbuffer context create");
    let mut surface =
        softbuffer::Surface::new(&context, Rc::clone(&window)).expect("softbuffer surface create");

    let mut win_w: u32 = 1200;
    let mut win_h: u32 = 800;
    let mut cursor = (0i32, 0i32);
    let mut drag_state = DragState::None;
    let mut drag_overlays: Vec<DragOverlay> = Vec::new();

    event_loop
        .run(move |event, elwt| {
            if let Some(ref rx) = repaint_rx {
                if rx.try_recv().is_ok() {
                    window.request_redraw();
                }
            }

            elwt.set_control_flow(ControlFlow::Wait);

            let Event::WindowEvent { event, .. } = event else {
                return;
            };

            match event {
                WindowEvent::CloseRequested => elwt.exit(),

                WindowEvent::Resized(size) => {
                    win_w = size.width.max(1);
                    win_h = size.height.max(1);
                    window.request_redraw();
                }

                WindowEvent::CursorMoved { position, .. } => {
                    cursor = (position.x as i32, position.y as i32);
                    let (new_state, overlays) = on_move(&drag_state, cursor.0, cursor.1);
                    let needs_redraw = !overlays.is_empty()
                        || matches!(
                            drag_state,
                            DragState::DraggingDivider { .. } | DragState::DraggingAZone { .. }
                        );
                    drag_state = new_state;
                    drag_overlays = overlays;
                    if needs_redraw {
                        window.request_redraw();
                    }
                }

                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    let new_state = on_press(cursor.0, cursor.1);
                    let started_drag = !matches!(new_state, DragState::None);
                    drag_state = new_state;
                    if !started_drag {
                        // Pass click to Brick scene tree.
                        if let Some(hit) = hit_test(&root, cursor.0 as f64, cursor.1 as f64) {
                            fire_event(
                                &hit,
                                "click",
                                BrickEvent::ClickAt(cursor.0 as f64, cursor.1 as f64),
                            );
                        }
                    }
                    window.request_redraw();
                }

                WindowEvent::MouseInput {
                    state: ElementState::Released,
                    button: MouseButton::Left,
                    ..
                } => {
                    on_release(&drag_state, cursor.0, cursor.1);
                    drag_state = DragState::None;
                    drag_overlays.clear();
                    window.request_redraw();
                }

                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Right,
                    ..
                } => {
                    let ctx = find_context_at(&root, cursor.0, cursor.1);
                    on_right_click(ctx, cursor.0, cursor.1);
                    window.request_redraw();
                }

                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } => {
                    use winit::keyboard::PhysicalKey;
                    if let PhysicalKey::Code(code) = key_event.physical_key {
                        let pressed = key_event.state == ElementState::Pressed;
                        // Use the debug name ("Space", "KeyV", "ArrowLeft", …) as the
                        // stable key identifier. Numeric discriminants are not stable
                        // across winit versions.
                        let key_name = format!("{code:?}");
                        let consumed = on_key(&key_name, pressed);
                        if consumed {
                            window.request_redraw();
                        }
                    }
                }

                WindowEvent::RedrawRequested => {
                    if let (Some(w), Some(h)) = (NonZeroU32::new(win_w), NonZeroU32::new(win_h)) {
                        surface.resize(w, h).expect("softbuffer resize");
                        let mut buf = surface.buffer_mut().expect("softbuffer buffer_mut");

                        layout_node(&root, 0, 0, win_w);
                        buf.fill(0x001E1E2E);
                        paint_node(&root, &mut buf, win_w);

                        for (hook_node, hook_fn) in &hooks {
                            let bounds = hook_node.0.borrow().bounds;
                            if bounds.w > 0 && bounds.h > 0 {
                                hook_fn(&mut buf, win_w, bounds);
                            }
                        }

                        // Paint drag overlays (AZone ghost, divider highlight).
                        for overlay in &drag_overlays {
                            paint_overlay(&mut buf, win_w, overlay);
                        }

                        // Extra paint pass (e.g. tab header strips).
                        on_paint(&mut buf, win_w, win_w, win_h);

                        buf.present().expect("softbuffer present");
                    }
                }

                _ => {}
            }
        })
        .expect("event loop run");
}

// ── Focus-aware event loop ────────────────────────────────────────────────────

fn run_window_with_focus(
    root: NativeNode,
    repaint_rx: Option<std::sync::mpsc::Receiver<()>>,
    hooks: Vec<(NativeNode, PaintHook)>,
    focus: FocusManager,
    mut on_key: OnKeyExtFn,
    mut on_pre_paint: Option<Box<dyn FnMut(u32, u32)>>,
) {
    use std::rc::Rc;
    use winit::event::{ElementState, Event, Ime, MouseButton, WindowEvent};
    use winit::event_loop::{ControlFlow, EventLoop, EventLoopBuilder};
    use winit::keyboard::{ModifiersState, PhysicalKey};
    use winit::window::WindowBuilder;

    #[cfg(target_os = "linux")]
    let event_loop = {
        use winit::platform::x11::EventLoopBuilderExtX11;
        EventLoopBuilder::new()
            .with_x11()
            .build()
            .expect("event loop create (x11)")
    };
    #[cfg(not(target_os = "linux"))]
    let event_loop = EventLoop::new().expect("event loop create");

    let window = Rc::new(
        WindowBuilder::new()
            .with_title("Brick")
            .with_inner_size(winit::dpi::LogicalSize::new(1200u32, 800u32))
            .build(&event_loop)
            .expect("window create"),
    );
    window.set_ime_allowed(true);

    let context = softbuffer::Context::new(Rc::clone(&window)).expect("softbuffer context create");
    let mut surface =
        softbuffer::Surface::new(&context, Rc::clone(&window)).expect("softbuffer surface create");

    let mut win_w: u32 = 1200;
    let mut win_h: u32 = 800;
    let mut cursor = (0.0_f64, 0.0_f64);
    let mut active_mods = KeyModifiers::default();

    event_loop
        .run(move |event, elwt| {
            if let Some(ref rx) = repaint_rx {
                if rx.try_recv().is_ok() {
                    window.request_redraw();
                }
            }

            elwt.set_control_flow(ControlFlow::Wait);

            let Event::WindowEvent { event, .. } = event else {
                return;
            };

            match event {
                WindowEvent::CloseRequested => elwt.exit(),

                WindowEvent::ModifiersChanged(mods) => {
                    let state = mods.state();
                    active_mods = KeyModifiers {
                        shift: state.contains(ModifiersState::SHIFT),
                        ctrl: state.contains(ModifiersState::CONTROL),
                        alt: state.contains(ModifiersState::ALT),
                        logo: state.contains(ModifiersState::SUPER),
                    };
                }

                WindowEvent::Resized(size) => {
                    win_w = size.width.max(1);
                    win_h = size.height.max(1);
                    window.request_redraw();
                }

                WindowEvent::CursorMoved { position, .. } => {
                    cursor = (position.x, position.y);
                }

                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    let fid = find_focus_id_at(&root, cursor.0 as i32, cursor.1 as i32);
                    match fid {
                        Some(id) => focus.request_focus(id),
                        None => focus.clear_focus(),
                    }
                    if let Some(hit) = hit_test(&root, cursor.0, cursor.1) {
                        fire_event(&hit, "click", BrickEvent::ClickAt(cursor.0, cursor.1));
                    }
                    window.request_redraw();
                }

                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } => {
                    let code = match key_event.physical_key {
                        PhysicalKey::Code(c) => format!("{c:?}"),
                        _ => String::new(),
                    };
                    let key_input = KeyInput {
                        code,
                        text: key_event.text.as_ref().map(|s| s.to_string()),
                        pressed: key_event.state == ElementState::Pressed,
                        repeat: key_event.repeat,
                        mods: active_mods,
                    };
                    if on_key(&key_input, &focus) {
                        window.request_redraw();
                    }
                }

                WindowEvent::Ime(Ime::Commit(text)) => {
                    let key_input = KeyInput {
                        code: String::new(),
                        text: Some(text),
                        pressed: true,
                        repeat: false,
                        mods: KeyModifiers::default(),
                    };
                    if on_key(&key_input, &focus) {
                        window.request_redraw();
                    }
                }

                WindowEvent::RedrawRequested => {
                    if let (Some(w), Some(h)) = (NonZeroU32::new(win_w), NonZeroU32::new(win_h)) {
                        surface.resize(w, h).expect("softbuffer resize");
                        let mut buf = surface.buffer_mut().expect("softbuffer buffer_mut");

                        if let Some(ref mut pre) = on_pre_paint {
                            pre(win_w, win_h);
                        }

                        layout_node(&root, 0, 0, win_w);
                        buf.fill(0x001E1E2E);
                        paint_node(&root, &mut buf, win_w);

                        for (hook_node, hook_fn) in &hooks {
                            let bounds = hook_node.0.borrow().bounds;
                            if bounds.w > 0 && bounds.h > 0 {
                                hook_fn(&mut buf, win_w, bounds);
                            }
                        }

                        // Paint a Mauve focus ring over the focused widget.
                        if let Some(id) = focus.focused() {
                            if let Some(bounds) = find_node_bounds_by_focus_id(&root, &id) {
                                paint_focus_ring(&mut buf, win_w, bounds, 0x00CBA6F7);
                            }
                        }

                        buf.present().expect("softbuffer present");
                    }
                }

                _ => {}
            }
        })
        .expect("event loop run");
}

/// Walk the scene tree, returning the `data-focus-id` of the nearest ancestor
/// (inclusive) containing `(x, y)`.  The outermost matching ancestor wins,
/// so setting `data-focus-id` on a container groups all its children under
/// that focus target without each child needing to repeat it.
fn find_focus_id_at(node: &NativeNode, x: i32, y: i32) -> Option<String> {
    fn walk(node: &NativeNode, x: i32, y: i32, current: Option<String>) -> Option<String> {
        let (bounds, attr, children) = {
            let inner = node.0.borrow();
            (
                inner.bounds,
                inner.attrs.get("data-focus-id").cloned(),
                inner.children.clone(),
            )
        };
        if x < bounds.x
            || x >= bounds.x + bounds.w as i32
            || y < bounds.y
            || y >= bounds.y + bounds.h as i32
        {
            return None;
        }
        let next = attr.or(current);
        for child in children.iter().rev() {
            if let Some(found) = walk(child, x, y, next.clone()) {
                return Some(found);
            }
        }
        next
    }
    walk(node, x, y, None)
}

/// DFS to find the bounds of the first node with `data-focus-id = id`.
fn find_node_bounds_by_focus_id(node: &NativeNode, id: &str) -> Option<Rect> {
    let (bounds, attr, children) = {
        let inner = node.0.borrow();
        (
            inner.bounds,
            inner.attrs.get("data-focus-id").cloned(),
            inner.children.clone(),
        )
    };
    if attr.as_deref() == Some(id) {
        return Some(bounds);
    }
    for child in &children {
        if let Some(found) = find_node_bounds_by_focus_id(child, id) {
            return Some(found);
        }
    }
    None
}

/// Alpha-blend a `DragOverlay` rect into the pixel buffer.
fn paint_overlay(buf: &mut [u32], buf_w: u32, overlay: &DragOverlay) {
    let x0 = overlay.rect.x.max(0) as u32;
    let y0 = overlay.rect.y.max(0) as u32;
    let x1 = (overlay.rect.x + overlay.rect.w as i32).max(0) as u32;
    let y1 = (overlay.rect.y + overlay.rect.h as i32).max(0) as u32;
    let r = (overlay.color >> 16) & 0xFF;
    let g = (overlay.color >> 8) & 0xFF;
    let b = overlay.color & 0xFF;
    let alpha = overlay.alpha as u32;
    for y in y0..y1 {
        let row = (y * buf_w) as usize;
        for x in x0..x1 {
            let idx = row + x as usize;
            if idx >= buf.len() {
                break;
            }
            let dst = buf[idx];
            let dr = (dst >> 16) & 0xFF;
            let dg = (dst >> 8) & 0xFF;
            let db = dst & 0xFF;
            let out_r = (r * alpha + dr * (255 - alpha)) / 255;
            let out_g = (g * alpha + dg * (255 - alpha)) / 255;
            let out_b = (b * alpha + db * (255 - alpha)) / 255;
            buf[idx] = (out_r << 16) | (out_g << 8) | out_b;
        }
    }
}

// ── Drag overlay constructors ─────────────────────────────────────────────────

/// AZone ghost overlay — semi-transparent white (20% opacity) over `ghost_rect`.
///
/// Shown while dragging a corner outward to preview where the new area will land.
pub fn azone_ghost_overlay(ghost_rect: Rect) -> DragOverlay {
    DragOverlay {
        rect: ghost_rect,
        color: 0x00FFFFFF,
        alpha: 51, // 0.2 * 255 ≈ 51
    }
}

/// Divider highlight overlay — a 2 px line painted in Catppuccin Mauve (#CBA6F7).
///
/// `split_rect` is the full rect of the Split node being dragged.
/// `axis_horizontal` true = left/right split (vertical line), false = top/bottom (horizontal line).
/// `pos` is the pixel position of the divider (x for horizontal axis, y for vertical).
pub fn divider_highlight_overlay(split_rect: Rect, axis_horizontal: bool, pos: i32) -> DragOverlay {
    let rect = if axis_horizontal {
        Rect {
            x: pos - 1,
            y: split_rect.y,
            w: 2,
            h: split_rect.h,
        }
    } else {
        Rect {
            x: split_rect.x,
            y: pos - 1,
            w: split_rect.w,
            h: 2,
        }
    };
    DragOverlay {
        rect,
        color: 0x00CBA6F7,
        alpha: 255,
    }
}

// ── Padding helper ────────────────────────────────────────────────────────────

/// Parse a CSS-style shorthand padding string into `[top, right, bottom, left]`
/// pixel values.
///
/// Accepted forms:
///   "8"           → all four sides = 8
///   "8 12"        → top+bottom = 8, left+right = 12
///   "8 12 8"      → top = 8, right+left = 12, bottom = 8
///   "8 12 8 12"   → top=8, right=12, bottom=8, left=12
///
/// Any other form (non-parseable tokens, >4 values) returns `[0, 0, 0, 0]`.
fn parse_pad(s: &str) -> [u32; 4] {
    let vals: Vec<u32> = s
        .split_whitespace()
        .filter_map(|v| v.parse::<u32>().ok())
        .collect();
    match vals.as_slice() {
        [a] => [*a, *a, *a, *a],
        [a, b] => [*a, *b, *a, *b],
        [a, b, c] => [*a, *b, *c, *b],
        [a, b, c, d] => [*a, *b, *c, *d],
        _ => [0, 0, 0, 0],
    }
}

// ── Layout pass ──────────────────────────────────────────────────────────────
//
// Supports two layout modes selected by the `data-layout` attr on a container:
//
//   column (default) — children stacked vertically, each filling available_w.
//   row              — children arranged left-to-right; each child's share of
//                      available_w is controlled by its `data-flex` attr (a
//                      float in 0.0–1.0). Children without `data-flex` share
//                      the remaining width equally.
//
// Supported spacing/decoration attrs on any container or text node:
//   data-pad          — inner padding, CSS shorthand "t r b l" / "t r" / "all"
//   data-border-top   — hex color for a top-side border stripe
//   data-border-left  — hex color for a left-side border stripe
//   data-border-left-width — width in px (default 1; overridden by the ":w" form)
//   data-align        — "left" (default) | "center" | "right" (text nodes)

fn layout_node(node: &NativeNode, x: i32, y: i32, available_w: u32) {
    // Overlay-positioned nodes (e.g. context menu panel) bypass normal flow:
    // their bounds come from explicit `data-x` / `data-y` / `data-w` / `data-h`
    // attrs so they can float on top of the scene tree at the cursor position.
    let overlay_rect = {
        let inner = node.0.borrow();
        if inner.attrs.contains_key("data-overlay") {
            let ox = inner
                .attrs
                .get("data-x")
                .and_then(|v| v.parse::<i32>().ok())
                .unwrap_or(x);
            let oy = inner
                .attrs
                .get("data-y")
                .and_then(|v| v.parse::<i32>().ok())
                .unwrap_or(y);
            let ow = inner
                .attrs
                .get("data-w")
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(available_w);
            let oh = inner
                .attrs
                .get("data-h")
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0);
            Some(Rect {
                x: ox,
                y: oy,
                w: ow,
                h: oh,
            })
        } else {
            None
        }
    };

    if let Some(rect) = overlay_rect {
        // Stack children vertically inside the overlay rect. Each child may
        // declare its own `data-h`; absent that, font_cache-derived text height
        // is used (same fallback the column branch applies).
        let children = node.0.borrow().children.clone();
        let mut cursor_y = rect.y;
        for child in &children {
            let child_h = child
                .0
                .borrow()
                .attrs
                .get("data-h")
                .and_then(|v| v.parse::<u32>().ok());
            layout_node(child, rect.x, cursor_y, rect.w);
            if let Some(h) = child_h {
                child.0.borrow_mut().bounds.h = h;
                cursor_y += h as i32;
            } else {
                cursor_y += child.0.borrow().bounds.h as i32;
            }
        }
        node.0.borrow_mut().bounds = rect;
        return;
    }

    let (children, is_text, is_row, pad) = {
        let inner = node.0.borrow();
        let children = inner.children.clone();
        let is_text = matches!(inner.kind, NodeKind::Text(_));
        let is_row = inner
            .attrs
            .get("data-layout")
            .map(|v| v == "row")
            .unwrap_or(false);
        let pad = inner
            .attrs
            .get("data-pad")
            .map(|s| parse_pad(s))
            .unwrap_or([0u32; 4]);
        (children, is_text, is_row, pad)
    };

    // Inner coordinates account for padding.
    let [pad_top, pad_right, pad_bottom, pad_left] = pad;
    let inner_x = x + pad_left as i32;
    let inner_y = y + pad_top as i32;
    let inner_w = available_w.saturating_sub(pad_left + pad_right);

    let content_h = if !children.is_empty() && is_row {
        // data-w: fixed pixel width (takes priority over data-flex).
        // data-flex: fraction of the width remaining after all fixed-width children are placed.
        // No data-w and no data-flex → share remaining equally.
        let fixed_widths: Vec<Option<u32>> = children
            .iter()
            .map(|c| c.0.borrow().attrs.get("data-w").and_then(|v| v.parse::<u32>().ok()))
            .collect();
        let flex_values: Vec<Option<f32>> = children
            .iter()
            .zip(fixed_widths.iter())
            .map(|(c, fw)| {
                if fw.is_some() {
                    Some(0.0) // fixed-width child: no flex claim
                } else {
                    c.0.borrow()
                        .attrs
                        .get("data-flex")
                        .and_then(|v| v.parse::<f32>().ok())
                }
            })
            .collect();

        let total_fixed: u32 = fixed_widths.iter().filter_map(|w| *w).sum();
        let remaining_for_flex = inner_w.saturating_sub(total_fixed) as f32;
        let claimed: f32 = flex_values.iter().filter_map(|f| *f).sum();
        let unclaimed_count = flex_values
            .iter()
            .zip(fixed_widths.iter())
            .filter(|(f, fw)| f.is_none() && fw.is_none())
            .count();
        let remaining_flex = (1.0_f32 - claimed).max(0.0);
        let share_each = if unclaimed_count > 0 {
            remaining_flex / unclaimed_count as f32
        } else {
            0.0
        };

        let mut cursor_x = inner_x;
        let mut max_h: u32 = 0;
        for ((child, flex), fixed_w) in
            children.iter().zip(flex_values.iter()).zip(fixed_widths.iter())
        {
            let child_w = if let Some(w) = fixed_w {
                *w
            } else {
                let frac = flex.unwrap_or(share_each);
                (frac * remaining_for_flex).round() as u32
            };
            layout_node(child, cursor_x, inner_y, child_w);
            max_h = max_h.max(child.0.borrow().bounds.h);
            cursor_x += child_w as i32;
        }
        max_h
    } else if !children.is_empty() {
        // Column layout (default).
        let is_scroll_y = node.0.borrow().attrs.get("data-scroll-y")
            .map(|v| v == "true").unwrap_or(false);
        let scroll_y = if is_scroll_y {
            let key = node.0.borrow().attrs.get("data-scroll-id")
                .cloned()
                .unwrap_or_else(|| format!("{}", Rc::as_ptr(&node.0) as usize));
            SCROLL_OFFSETS.with(|s| s.borrow().get(&key).copied().unwrap_or(0))
        } else {
            0
        };
        let mut cursor_y = inner_y - scroll_y;
        for child in &children {
            let is_overlay = child.0.borrow().attrs.contains_key("data-overlay");
            layout_node(child, inner_x, cursor_y, inner_w);
            // Overlay children are absolutely positioned and don't contribute to flow.
            if !is_overlay {
                cursor_y += child.0.borrow().bounds.h as i32;
            }
        }
        (cursor_y - inner_y + scroll_y).max(0) as u32
    } else if is_text {
        // Measure the actual wrapped height given the available (padded) width.
        let (text_content, size_px) = {
            let inner = node.0.borrow();
            let txt = match &inner.kind {
                NodeKind::Text(s) => s.clone(),
                _ => String::new(),
            };
            let sz = inner
                .attrs
                .get("data-text-size")
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(font_cache().size_px);
            (txt, sz)
        };
        measure_text_height(&text_content, inner_w, size_px)
    } else {
        0
    };

    // Total height includes padding on all sides.
    let h = content_h + pad_top + pad_bottom;

    // Allow callers to override the computed height with an explicit pixel value.
    // Children are still laid out at their natural positions within the available
    // width; only this node's reported height changes (paint clips to bounds).
    let final_h = node
        .0
        .borrow()
        .attrs
        .get("data-height")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(h);

    node.0.borrow_mut().bounds = Rect {
        x,
        y,
        w: available_w,
        h: final_h,
    };
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "brick_native")]
    use super::super::font::{FontFamily, FontWeight};
    use super::*;
    use crate::BrickRenderer;

    #[test]
    fn test_paint_hook_fires() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let node = NativeRenderer::element("div");
        // Manually assign bounds — simulates the post-layout state the render
        // loop would produce, without invoking font_cache in a test env.
        node.0.borrow_mut().bounds = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: 100,
        };

        let fired = Arc::new(AtomicBool::new(false));
        let fired_clone = Arc::clone(&fired);
        let hook: PaintHook = Arc::new(move |_buf, _w, _bounds| {
            fired_clone.store(true, Ordering::SeqCst);
        });

        let hooks = vec![(node.clone(), hook)];
        let mut buf = vec![0u32; 200 * 100];

        // Simulate the hook dispatch that run_window performs after paint_node.
        for (hook_node, hook_fn) in &hooks {
            let bounds = hook_node.0.borrow().bounds;
            if bounds.w > 0 && bounds.h > 0 {
                hook_fn(&mut buf, 200, bounds);
            }
        }

        assert!(fired.load(Ordering::SeqCst), "paint hook should have fired");
    }

    #[test]
    fn test_row_layout() {
        let row = NativeRenderer::element("div");
        NativeRenderer::set_attr(&row, "data-layout", "row");

        let left = NativeRenderer::element("div");
        NativeRenderer::set_attr(&left, "data-flex", "0.6");
        // Give it a child so it has nonzero height.
        NativeRenderer::append(&left, &NativeRenderer::text("left"));
        NativeRenderer::append(&row, &left);

        let right = NativeRenderer::element("div");
        NativeRenderer::set_attr(&right, "data-flex", "0.4");
        NativeRenderer::append(&right, &NativeRenderer::text("right"));
        NativeRenderer::append(&row, &right);

        layout_node(&row, 0, 0, 1000);

        let left_b = left.0.borrow().bounds;
        let right_b = right.0.borrow().bounds;

        assert_eq!(left_b.x, 0);
        assert_eq!(left_b.w, 600, "left child should be 60% of 1000");
        assert_eq!(right_b.x, 600);
        assert_eq!(right_b.w, 400, "right child should be 40% of 1000");
    }

    // ── G1 text-wrap tests ────────────────────────────────────────────────────

    #[test]
    fn test_wrap_paragraph_empty() {
        let lines = wrap_paragraph("", 200, 14.0);
        assert_eq!(lines.len(), 1, "empty string produces one (empty) line");
        assert_eq!(lines[0], "");
    }

    #[test]
    fn test_wrap_paragraph_single_word_fits() {
        // A single word narrower than avail_w stays on one line.
        let lines = wrap_paragraph("hello", 500, 14.0);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "hello");
    }

    #[test]
    fn test_wrap_paragraph_two_lines() {
        // "foo bar baz" at 14px in a very narrow box → multiple lines.
        // Exact split depends on font metrics, but we get more than one line.
        let lines = wrap_paragraph("foo bar baz", 20, 14.0);
        assert!(
            lines.len() > 1,
            "narrow box must produce multiple wrapped lines"
        );
        // All words must still be present.
        let rejoined = lines.join(" ");
        assert!(rejoined.contains("foo"));
        assert!(rejoined.contains("bar"));
        assert!(rejoined.contains("baz"));
    }

    #[test]
    fn test_measure_text_height_single_line() {
        let h = measure_text_height("hello", 500, 14.0);
        let expected_line_height = (14.0_f32 * 1.4).ceil() as u32;
        assert_eq!(h, expected_line_height, "single line = one line_height");
    }

    #[test]
    fn test_measure_text_height_hard_break() {
        // Two paragraphs joined by \n → height at least 2× line_height.
        let h = measure_text_height("hello\nworld", 500, 14.0);
        let lh = (14.0_f32 * 1.4).ceil() as u32;
        assert!(
            h >= lh * 2,
            "two hard-break lines must be >= 2 × line_height"
        );
    }

    #[test]
    fn test_measure_text_height_wrap() {
        // Very narrow box forces wrapping — height > single line.
        let h_narrow = measure_text_height("foo bar baz", 20, 14.0);
        let h_wide = measure_text_height("foo bar baz", 500, 14.0);
        assert!(
            h_narrow > h_wide,
            "wrapped text in narrow box must be taller than wide box"
        );
    }

    #[test]
    fn test_layout_text_node_height_grows_with_content() {
        let node = NativeRenderer::text("line one\nline two");
        layout_node(&node, 0, 0, 500);
        let h = node.0.borrow().bounds.h;
        let lh = (14.0_f32 * 1.4).ceil() as u32;
        assert!(
            h >= lh * 2,
            "two hard-break lines must produce height >= 2 × line_height"
        );
    }

    #[test]
    fn test_paint_text_does_not_panic_on_newline() {
        // Previously '\n' caused break; now it must not panic.
        let mut buf = vec![0u32; 800 * 100];
        let bounds = Rect {
            x: 0,
            y: 0,
            w: 800,
            h: 100,
        };
        let style = TextStyle::default();
        paint_text(
            &mut buf,
            800,
            &bounds,
            "hello\nworld",
            0x00FFFFFF,
            "left",
            &style,
        );
        // If we reached here without panic the test passes.
    }

    #[test]
    fn test_paint_text_align_does_not_panic() {
        let mut buf = vec![0u32; 400 * 50];
        let bounds = Rect {
            x: 0,
            y: 0,
            w: 400,
            h: 50,
        };
        let style = TextStyle::default();
        paint_text(
            &mut buf,
            400,
            &bounds,
            "centered text",
            0x00FFFFFF,
            "center",
            &style,
        );
        paint_text(
            &mut buf,
            400,
            &bounds,
            "right text",
            0x00FFFFFF,
            "right",
            &style,
        );
    }

    // ── G2 padding / border-left / align tests ────────────────────────────────

    #[test]
    fn test_parse_pad_single() {
        assert_eq!(parse_pad("8"), [8, 8, 8, 8]);
    }

    #[test]
    fn test_parse_pad_two() {
        assert_eq!(parse_pad("8 12"), [8, 12, 8, 12]);
    }

    #[test]
    fn test_parse_pad_four() {
        assert_eq!(parse_pad("4 8 4 8"), [4, 8, 4, 8]);
    }

    #[test]
    fn test_parse_pad_invalid() {
        assert_eq!(parse_pad(""), [0, 0, 0, 0]);
        assert_eq!(parse_pad("abc"), [0, 0, 0, 0]);
    }

    #[test]
    fn test_layout_padding_increases_height() {
        // A container with one text child and 16px uniform padding.
        let container = NativeRenderer::element("div");
        NativeRenderer::set_attr(&container, "data-pad", "16");
        NativeRenderer::append(&container, &NativeRenderer::text("hello"));

        layout_node(&container, 0, 0, 200);

        let h = container.0.borrow().bounds.h;
        let lh = (14.0_f32 * 1.4).ceil() as u32;
        // Height must be at least content_h + 32 (16 top + 16 bottom).
        assert!(
            h >= lh + 32,
            "padded container height must include pad_top + pad_bottom"
        );
    }

    #[test]
    fn test_layout_padding_offsets_children() {
        // Child should be laid out at (pad_left, pad_top) within the container.
        let container = NativeRenderer::element("div");
        NativeRenderer::set_attr(&container, "data-pad", "10 20 10 20");
        let child = NativeRenderer::text("child text");
        NativeRenderer::append(&container, &child);

        layout_node(&container, 0, 0, 200);

        let cb = child.0.borrow().bounds;
        assert_eq!(cb.x, 20, "child x should be offset by pad_left=20");
        assert_eq!(cb.y, 10, "child y should be offset by pad_top=10");
        assert_eq!(
            cb.w, 160,
            "child width should be 200 - pad_left(20) - pad_right(20)"
        );
    }

    #[test]
    fn test_paint_border_left_writes_pixels() {
        // A 100×100 buffer, container at (0,0,100,100), border-left green #A6E3A1 width 3.
        let mut buf = vec![0u32; 100 * 100];
        let r = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
        };
        paint_border_left(&mut buf, &r, 100, 3, 0x00A6E3A1);

        // The first 3 columns of every row should be #A6E3A1.
        for row in 0..100 {
            for col in 0..3 {
                let idx = row * 100 + col;
                assert_eq!(buf[idx], 0x00A6E3A1, "border pixel at row={row} col={col}");
            }
            // Column 3 should NOT be painted.
            let idx = row * 100 + 3;
            assert_eq!(
                buf[idx], 0,
                "non-border pixel at row={row} col=3 should be 0"
            );
        }
    }

    #[test]
    fn test_paint_node_border_left_combined_format() {
        // Verify "#RRGGBB:N" combined format (as used by fleet.rs) is handled.
        let node = NativeRenderer::element("div");
        NativeRenderer::set_attr(&node, "data-border-left", "#CBA6F7:2");
        node.0.borrow_mut().bounds = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 50,
        };

        let mut buf = vec![0u32; 100 * 50];
        paint_node(&node, &mut buf, 100);

        // First 2 columns should be #CBA6F7.
        assert_eq!(
            buf[0], 0x00CBA6F7,
            "first pixel should be border-left color"
        );
        assert_eq!(
            buf[1], 0x00CBA6F7,
            "second pixel should be border-left color"
        );
        assert_eq!(buf[2], 0, "third pixel should not be painted");
    }

    #[test]
    fn test_paint_node_border_left_separate_attrs() {
        // Verify separate data-border-left + data-border-left-width attrs.
        let node = NativeRenderer::element("div");
        NativeRenderer::set_attr(&node, "data-border-left", "#A6E3A1");
        NativeRenderer::set_attr(&node, "data-border-left-width", "3");
        node.0.borrow_mut().bounds = Rect {
            x: 0,
            y: 0,
            w: 80,
            h: 20,
        };

        let mut buf = vec![0u32; 80 * 20];
        paint_node(&node, &mut buf, 80);

        assert_eq!(buf[0], 0x00A6E3A1, "col 0 should be border-left color");
        assert_eq!(buf[2], 0x00A6E3A1, "col 2 should be border-left color");
        assert_eq!(buf[3], 0, "col 3 should be unpainted");
    }

    // ── G3 font-registry / data-text-* tests ─────────────────────────────────

    #[cfg(feature = "brick_native")]
    #[test]
    fn test_text_style_default() {
        let style = TextStyle::default();
        assert_eq!(style.family, FontFamily::Sans);
        assert_eq!(style.weight, FontWeight::Regular);
        assert!(!style.italic);
        assert!(style.size_px > 0.0);
    }

    #[cfg(feature = "brick_native")]
    #[test]
    fn test_text_style_from_attrs_mono() {
        let mut attrs = HashMap::new();
        attrs.insert("data-text-family".to_string(), "mono".to_string());
        attrs.insert("data-text-weight".to_string(), "bold".to_string());
        attrs.insert("data-text-italic".to_string(), "true".to_string());
        attrs.insert("data-text-size".to_string(), "18".to_string());
        let style = TextStyle::from_attrs(&attrs);
        assert_eq!(style.family, FontFamily::Mono);
        assert_eq!(style.weight, FontWeight::Bold);
        assert!(style.italic);
        assert_eq!(style.size_px, 18.0);
    }

    #[cfg(feature = "brick_native")]
    #[test]
    fn test_text_style_size_affects_layout() {
        // Nodes with data-text-size="20" must be taller than at the default size.
        let node_default = NativeRenderer::text("Hello");
        let node_large = NativeRenderer::text("Hello");
        NativeRenderer::set_attr(&node_large, "data-text-size", "20");

        layout_node(&node_default, 0, 0, 400);
        layout_node(&node_large, 0, 0, 400);

        let h_default = node_default.0.borrow().bounds.h;
        let h_large = node_large.0.borrow().bounds.h;
        assert!(
            h_large > h_default,
            "larger data-text-size must produce a taller text node ({h_large} vs {h_default})"
        );
    }

    #[cfg(feature = "brick_native")]
    #[test]
    fn test_paint_text_mono_family() {
        // Painting with mono family must not panic.
        let mut buf = vec![0u32; 400 * 50];
        let bounds = Rect {
            x: 0,
            y: 0,
            w: 400,
            h: 50,
        };
        let style = TextStyle {
            family: FontFamily::Mono,
            weight: FontWeight::Regular,
            italic: false,
            size_px: 14.0,
        };
        paint_text(
            &mut buf,
            400,
            &bounds,
            "fn main() {}",
            0x00FFFFFF,
            "left",
            &style,
        );
    }
}

// ── Image cache ──────────────────────────────────────────────────────────────
//
// Decoded + scaled images are cached by (path, w, h) so each image file is
// loaded from disk only once per unique display size per thread.
// Stored as packed 0xAARRGGBB so `blit_image` can alpha-composite without a
// second decode step.  The cache is thread-local because NativeRenderer runs
// entirely on the event-loop thread.

#[cfg(feature = "brick_native")]
static IMAGE_CACHE: std::sync::LazyLock<
    std::sync::RwLock<HashMap<(String, u32, u32), Arc<Vec<u32>>>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(HashMap::new()));

/// Resolve `path` to an existing file, trying (in order):
///   1. absolute path as-is
///   2. relative to the running executable's directory
///   3. relative to the current working directory
#[cfg(feature = "brick_native")]
fn resolve_asset(path: &str) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(path);
    if p.is_absolute() && p.exists() {
        return Some(p.to_path_buf());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let c = dir.join(p);
            if c.exists() {
                return Some(c);
            }
        }
    }
    if p.exists() {
        return Some(p.to_path_buf());
    }
    None
}

/// Load, scale, and cache an image for painting.
///
/// Returns a slice of `0xAARRGGBB` pixels sized `w × h`, or `None` if the file
/// cannot be found or decoded.
#[cfg(feature = "brick_native")]
fn load_scaled_image(path: &str, w: u32, h: u32) -> Option<Arc<Vec<u32>>> {
    let key = (path.to_string(), w, h);
    // Fast path — shared read lock; no allocation when cached.
    {
        if let Ok(cache) = IMAGE_CACHE.read() {
            if let Some(cached) = cache.get(&key) {
                return Some(Arc::clone(cached));
            }
        }
    }
    // Slow path — decode + scale, then insert under write lock.
    let resolved = resolve_asset(path)?;
    let img = image::open(&resolved).ok()?.to_rgba8();
    // Cover-fill: scale so the image fills both target dimensions, then center-crop.
    // Square source images (icons) are unaffected; landscape images (splashes) render
    // without stretching.
    let (iw, ih) = img.dimensions();
    let scale_x = w as f64 / iw as f64;
    let scale_y = h as f64 / ih as f64;
    let scale   = scale_x.max(scale_y);
    let cover_w = ((iw as f64 * scale).round() as u32).max(1);
    let cover_h = ((ih as f64 * scale).round() as u32).max(1);
    let upscaled = image::imageops::resize(
        &img, cover_w, cover_h,
        image::imageops::FilterType::Triangle,
    );
    let cx = cover_w.saturating_sub(w) / 2;
    let cy = cover_h.saturating_sub(h) / 2;
    let cropped = image::imageops::crop_imm(&upscaled, cx, cy, w, h).to_image();
    let pixels: Vec<u32> = cropped
        .pixels()
        .map(|p| {
            let [r, g, b, a] = p.0;
            ((a as u32) << 24)
                | ((r as u32) << 16)
                | ((g as u32) << 8)
                | (b as u32)
        })
        .collect();
    let arc = Arc::new(pixels);
    if let Ok(mut cache) = IMAGE_CACHE.write() {
        cache.insert(key, Arc::clone(&arc));
    }
    Some(arc)
}


/// Alpha-composite `pixels` (0xAARRGGBB, `r.w × r.h`) into the framebuffer.
/// `alpha_mul` (0–255) scales every pixel's alpha, enabling partial-opacity images.
#[cfg(feature = "brick_native")]
fn blit_image(buf: &mut [u32], buf_w: u32, r: &Rect, pixels: &[u32], alpha_mul: u8) {
    if r.w == 0 || r.h == 0 {
        return;
    }
    let clip = PAINT_CLIP.with(|c| c.get());
    let raw_x0 = r.x.max(0);
    let raw_y0 = r.y.max(0);
    let raw_x1 = (r.x + r.w as i32).min(buf_w as i32);
    let raw_y1 = r.y + r.h as i32;
    let (x0, y0, x1, y1) = if let Some((cx0, cy0, cx1, cy1)) = clip {
        (raw_x0.max(cx0).max(0) as u32,
         raw_y0.max(cy0).max(0) as u32,
         raw_x1.min(cx1).max(0) as u32,
         raw_y1.min(cy1).max(0) as u32)
    } else {
        (raw_x0 as u32, raw_y0 as u32, raw_x1.max(0) as u32, raw_y1.max(0) as u32)
    };
    if x0 >= x1 || y0 >= y1 { return; }
    for py in y0..y1 {
        for px in x0..x1 {
            let src_x = (px as i32 - r.x).max(0).min(r.w as i32 - 1) as u32;
            let src_y = (py as i32 - r.y).max(0).min(r.h as i32 - 1) as u32;
            let src = match pixels.get((src_y * r.w + src_x) as usize) {
                Some(&v) => v,
                None => continue,
            };
            // Apply alpha_mul to scale down the pixel's effective opacity.
            let raw_a = (src >> 24) & 0xFF;
            let a = raw_a * alpha_mul as u32 / 255;
            if a == 0 { continue; }
            let dst_idx = (py * buf_w + px) as usize;
            if dst_idx >= buf.len() { continue; }
            let sr = (src >> 16) & 0xFF;
            let sg = (src >> 8) & 0xFF;
            let sb = src & 0xFF;
            let dst = buf[dst_idx];
            let dr = (dst >> 16) & 0xFF;
            let dg = (dst >> 8) & 0xFF;
            let db = dst & 0xFF;
            let inv = 255 - a;
            buf[dst_idx] = (((sr * a + dr * inv) / 255) << 16)
                | (((sg * a + dg * inv) / 255) << 8)
                | ((sb * a + db * inv) / 255);
        }
    }
}

// ── Paint pass ───────────────────────────────────────────────────────────────
//
// Walks the scene tree and blits each node into a flat u32 pixel buffer.
// Pixel format: 0x00RRGGBB (xRGB, top byte unused — softbuffer 0.4 convention).
//
// Text nodes render as a dim-grey placeholder rectangle until fontdue is wired.
// Set `data-fill` attr on a Container to fill it with an #RRGGBB color.
// Set `data-image` attr to a PNG path to composite the image into the node bounds.

/// Blend each edge of `bounds` in the buffer toward `color`.
/// `fade_bottom/left/right` are pixel counts; 0 = no fade on that edge.
/// Pixel alpha is computed per-edge and max-blended so corners look correct.
#[cfg(feature = "brick_native")]
fn paint_edge_fades(
    buf: &mut [u32], buf_w: u32, bounds: &Rect, color: u32,
    fade_bottom: u32, fade_left: u32, fade_right: u32,
) {
    let tr = (color >> 16) & 0xFF;
    let tg = (color >> 8) & 0xFF;
    let tb = color & 0xFF;

    let clip = PAINT_CLIP.with(|c| c.get());
    let raw_x0 = bounds.x.max(0);
    let raw_y0 = bounds.y.max(0);
    let raw_x1 = (bounds.x + bounds.w as i32).min(buf_w as i32);
    let raw_y1 = bounds.y + bounds.h as i32;
    let (x0, y0, x1, y1) = if let Some((cx0, cy0, cx1, cy1)) = clip {
        (raw_x0.max(cx0), raw_y0.max(cy0), raw_x1.min(cx1), raw_y1.min(cy1))
    } else {
        (raw_x0, raw_y0, raw_x1, raw_y1)
    };
    if x0 >= x1 || y0 >= y1 { return; }

    for py in y0..y1 {
        for px in x0..x1 {
            let bx = (px - bounds.x) as u32;
            let by = (py - bounds.y) as u32;

            let a_bottom = if fade_bottom > 0 && bounds.h > fade_bottom {
                let start = bounds.h - fade_bottom;
                if by >= start { ((by - start) as f32 / fade_bottom as f32 * 255.0) as u32 } else { 0 }
            } else { 0 };

            let a_left = if fade_left > 0 && bx < fade_left {
                ((fade_left - bx) as f32 / fade_left as f32 * 255.0) as u32
            } else { 0 };

            let a_right = if fade_right > 0 && bounds.w > fade_right && bx >= bounds.w - fade_right {
                ((bx - (bounds.w - fade_right)) as f32 / fade_right as f32 * 255.0) as u32
            } else { 0 };

            let alpha = a_bottom.max(a_left).max(a_right).min(255);
            if alpha == 0 { continue; }

            let idx = (py as u32 * buf_w + px as u32) as usize;
            if idx >= buf.len() { continue; }
            let src = buf[idx];
            let sr = (src >> 16) & 0xFF;
            let sg = (src >> 8) & 0xFF;
            let sb = src & 0xFF;
            let r = (sr * (255 - alpha) + tr * alpha) / 255;
            let g = (sg * (255 - alpha) + tg * alpha) / 255;
            let b = (sb * (255 - alpha) + tb * alpha) / 255;
            buf[idx] = (r << 16) | (g << 8) | b;
        }
    }
}

fn paint_node(node: &NativeNode, buf: &mut [u32], buf_w: u32) {
    #[allow(clippy::type_complexity)]
    let (
        bounds,
        fill_color,
        text_color,
        border_color,
        border_top,
        border_left,
        border_bottom,
        border_right,
        children,
        text_content,
        align,
        pad,
        text_style,
        fade_color,
        fade_bottom,
        fade_left,
        fade_right,
        image_alpha,
    ) = {
        let inner = node.0.borrow();
        let fill = inner
            .attrs
            .get("data-fill")
            .and_then(|s| parse_hex_color(s));
        let color = inner
            .attrs
            .get("data-color")
            .and_then(|s| parse_hex_color(s));
        let border = inner
            .attrs
            .get("data-border")
            .and_then(|s| parse_hex_color(s));

        // data-border-top/left/bottom/right all support "#RRGGBB" or "#RRGGBB:N" format.
        let border_top = inner.attrs.get("data-border-top").and_then(|s| {
            if let Some((hex, w)) = s.split_once(':') {
                let color = parse_hex_color(hex.trim())?;
                let width = w.trim().parse::<u32>().unwrap_or(1);
                Some((color, width))
            } else {
                parse_hex_color(s).map(|c| (c, 1u32))
            }
        });
        let border_left = inner.attrs.get("data-border-left").and_then(|s| {
            if let Some((hex, w)) = s.split_once(':') {
                let color = parse_hex_color(hex.trim())?;
                let width = w.trim().parse::<u32>().unwrap_or(1);
                Some((color, width))
            } else {
                let color = parse_hex_color(s)?;
                let width = inner
                    .attrs
                    .get("data-border-left-width")
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(1);
                Some((color, width))
            }
        });
        let border_bottom = inner.attrs.get("data-border-bottom").and_then(|s| {
            if let Some((hex, w)) = s.split_once(':') {
                let color = parse_hex_color(hex.trim())?;
                let width = w.trim().parse::<u32>().unwrap_or(1);
                Some((color, width))
            } else {
                parse_hex_color(s).map(|c| (c, 1u32))
            }
        });
        let border_right = inner.attrs.get("data-border-right").and_then(|s| {
            if let Some((hex, w)) = s.split_once(':') {
                let color = parse_hex_color(hex.trim())?;
                let width = w.trim().parse::<u32>().unwrap_or(1);
                Some((color, width))
            } else {
                parse_hex_color(s).map(|c| (c, 1u32))
            }
        });

        let align = inner
            .attrs
            .get("data-align")
            .cloned()
            .unwrap_or_else(|| "left".to_string());
        let pad = inner
            .attrs
            .get("data-pad")
            .map(|s| parse_pad(s))
            .unwrap_or([0u32; 4]);
        let text_style = TextStyle::from_attrs(&inner.attrs);
        let bounds = inner.bounds;
        let children = inner.children.clone();
        let text = match &inner.kind {
            NodeKind::Text(s) => Some(s.clone()),
            _ => None,
        };
        let fade_color  = inner.attrs.get("data-fade-to").and_then(|s| parse_hex_color(s));
        let fade_bottom = inner.attrs.get("data-fade-bottom").and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        let fade_left   = inner.attrs.get("data-fade-left").and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        let fade_right  = inner.attrs.get("data-fade-right").and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        // data-image-alpha: 0.0–1.0 multiplier applied to every image pixel's alpha
        // before blending with the fill. Defaults to 1.0 (fully opaque).
        let image_alpha = inner.attrs.get("data-image-alpha")
            .and_then(|v| v.parse::<f32>().ok())
            .map(|f| (f.clamp(0.0, 1.0) * 255.0).round() as u8)
            .unwrap_or(255u8);
        (
            bounds,
            fill,
            color,
            border,
            border_top,
            border_left,
            border_bottom,
            border_right,
            children,
            text,
            align,
            pad,
            text_style,
            fade_color,
            fade_bottom,
            fade_left,
            fade_right,
            image_alpha,
        )
    };

    {
        let hover_fill = node.0.borrow().attrs
            .get("data-hover-fill")
            .and_then(|s| parse_hex_color(s));
        let is_hovered = hover_fill.is_some()
            && HOVER_PTR.with(|p| p.get()) == (Rc::as_ptr(&node.0) as usize);
        let fill = if is_hovered { hover_fill } else { fill_color };
        if let Some(color) = fill {
            fill_rect(buf, &bounds, buf_w, color);
        }
    }

    // Image: data-image = path to PNG, composited into node bounds after fill.
    #[cfg(feature = "brick_native")]
    if bounds.w > 0 && bounds.h > 0 {
        if let Some(path) = node.0.borrow().attrs.get("data-image").cloned() {
            if let Some(pixels) = load_scaled_image(&path, bounds.w, bounds.h) {
                blit_image(buf, buf_w, &bounds, &pixels, image_alpha);
            }
        }
        if let Some(fc) = fade_color {
            if fade_bottom > 0 || fade_left > 0 || fade_right > 0 {
                paint_edge_fades(buf, buf_w, &bounds, fc, fade_bottom, fade_left, fade_right);
            }
        }
    }

    // Edge borders paint on top of the fill, before children.
    if let Some((bt_color, bt_width)) = border_top {
        paint_border_top(buf, &bounds, buf_w, bt_width, bt_color);
    }
    if let Some((bl_color, bl_width)) = border_left {
        paint_border_left(buf, &bounds, buf_w, bl_width, bl_color);
    }
    if let Some((bb_color, bb_width)) = border_bottom {
        paint_border_bottom(buf, &bounds, buf_w, bb_width, bb_color);
    }
    if let Some((br_color, br_width)) = border_right {
        paint_border_right(buf, &bounds, buf_w, br_width, br_color);
    }

    if let Some(content) = text_content {
        let fg = text_color.unwrap_or(0x00CDD6F4); // catppuccin mocha text
        // For text nodes that carry padding attrs, shrink the painting rect.
        let [pad_top, pad_right, pad_bottom, pad_left] = pad;
        let paint_bounds = if pad_top > 0 || pad_right > 0 || pad_bottom > 0 || pad_left > 0 {
            Rect {
                x: bounds.x + pad_left as i32,
                y: bounds.y + pad_top as i32,
                w: bounds.w.saturating_sub(pad_left + pad_right),
                h: bounds.h.saturating_sub(pad_top + pad_bottom),
            }
        } else {
            bounds
        };
        paint_text(buf, buf_w, &paint_bounds, &content, fg, &align, &text_style);
    }

    let is_scroll_paint = node.0.borrow().attrs.get("data-scroll-y")
        .map(|v| v == "true").unwrap_or(false);
    if is_scroll_paint {
        let prev_clip = PAINT_CLIP.with(|c| c.get());
        PAINT_CLIP.with(|c| c.set(Some((bounds.x, bounds.y, bounds.x + bounds.w as i32, bounds.y + bounds.h as i32))));
        for child in &children {
            paint_node(child, buf, buf_w);
        }
        PAINT_CLIP.with(|c| c.set(prev_clip));
    } else {
        for child in &children {
            paint_node(child, buf, buf_w);
        }
    }

    // Stroke border paints after children so the frame sits on top of fills.
    if let Some(color) = border_color {
        stroke_rect(buf, &bounds, buf_w, color);
    }
}

/// Draw a 1-pixel border around `r` in `color`.
fn stroke_rect(buf: &mut [u32], r: &Rect, buf_w: u32, color: u32) {
    if r.w == 0 || r.h == 0 {
        return;
    }
    let x0 = r.x.max(0) as u32;
    let y0 = r.y.max(0) as u32;
    let x1 = (r.x + r.w as i32).max(0) as u32;
    let y1 = (r.y + r.h as i32).max(0) as u32;
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let bottom = y1.saturating_sub(1);
    let right = x1.saturating_sub(1);
    for x in x0..x1 {
        let top_idx = (y0 * buf_w) as usize + x as usize;
        if top_idx < buf.len() {
            buf[top_idx] = color;
        }
        let bot_idx = (bottom * buf_w) as usize + x as usize;
        if bot_idx < buf.len() {
            buf[bot_idx] = color;
        }
    }
    for y in y0..y1 {
        let left_idx = (y * buf_w) as usize + x0 as usize;
        if left_idx < buf.len() {
            buf[left_idx] = color;
        }
        let right_idx = (y * buf_w) as usize + right as usize;
        if right_idx < buf.len() {
            buf[right_idx] = color;
        }
    }
}

// ── Text style ────────────────────────────────────────────────────────────────

/// Combined font style for a single text node. Extracted from `data-text-*`
/// attrs in `paint_node` / `layout_node` and passed down to paint helpers.
#[derive(Clone, Debug)]
pub(super) struct TextStyle {
    pub(super) family: FontFamily,
    pub(super) weight: FontWeight,
    pub(super) italic: bool,
    pub(super) size_px: f32,
}

impl TextStyle {
    /// Read style attrs from a node's attribute map.
    fn from_attrs(attrs: &HashMap<String, String>) -> Self {
        let family = attrs
            .get("data-text-family")
            .map(|s| FontFamily::from_str(s))
            .unwrap_or(FontFamily::Sans);
        let weight = attrs
            .get("data-text-weight")
            .map(|s| FontWeight::from_str(s))
            .unwrap_or(FontWeight::Regular);
        let italic = attrs
            .get("data-text-italic")
            .map(|s| s == "true")
            .unwrap_or(false);
        let size_px = attrs
            .get("data-text-size")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or_else(|| font_cache().size_px);
        Self {
            family,
            weight,
            italic,
            size_px,
        }
    }

    fn default() -> Self {
        Self {
            family: FontFamily::Sans,
            weight: FontWeight::Regular,
            italic: false,
            size_px: font_cache().size_px,
        }
    }
}

// ── Text measurement + wrapping ───────────────────────────────────────────────

/// Measure the advance width of `ch` at `size_px` using the given font style.
fn char_advance_styled(ch: char, style: &TextStyle) -> f32 {
    font_registry()
        .metrics(ch, style.size_px, style.family, style.weight, style.italic)
        .advance_width
}

/// Measure the advance width of a single character at `size_px` using the
/// default sans-regular font. Kept for use by layout paths that don't yet have
/// a TextStyle in scope.
fn char_advance(ch: char, size_px: f32) -> f32 {
    font_registry()
        .metrics(ch, size_px, FontFamily::Sans, FontWeight::Regular, false)
        .advance_width
}

/// Break a single paragraph (no embedded `\n`) into wrapped display lines using
/// a greedy word-wrap algorithm. Each line fits within `avail_w` pixels at
/// `size_px`. Individual words wider than `avail_w` are placed on their own
/// line without splitting.
fn wrap_paragraph(para: &str, avail_w: u32, size_px: f32) -> Vec<String> {
    if para.is_empty() {
        return vec![String::new()];
    }
    let max_w = avail_w as f32;
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut current_w: f32 = 0.0;
    let space_w = char_advance(' ', size_px);

    for word in para.split_whitespace() {
        let word_w: f32 = word.chars().map(|ch| char_advance(ch, size_px)).sum();
        if current.is_empty() {
            current.push_str(word);
            current_w = word_w;
        } else if current_w + space_w + word_w <= max_w {
            current.push(' ');
            current.push_str(word);
            current_w += space_w + word_w;
        } else {
            lines.push(current.clone());
            current = word.to_string();
            current_w = word_w;
        }
    }
    lines.push(current);
    lines
}

/// Compute the total pixel height that `text` will occupy when painted at
/// `size_px` in a box of `avail_w` pixels wide. Honours hard `\n` breaks and
/// soft-wraps on whitespace.
///
/// `avail_w == 0` is treated as "no constraint" (one line per paragraph).
pub(crate) fn measure_text_height(text: &str, avail_w: u32, size_px: f32) -> u32 {
    let line_height = (size_px * 1.4).ceil() as u32;
    let mut total_lines: u32 = 0;
    for para in text.split('\n') {
        let n = if avail_w == 0 {
            1
        } else {
            wrap_paragraph(para, avail_w, size_px).len() as u32
        };
        total_lines += n;
    }
    (total_lines * line_height).max(line_height)
}

/// Paint `text` into `buf` within `bounds`, honouring hard `\n` line breaks,
/// soft-wrapping on whitespace, and the `align` parameter ("left", "center",
/// "right"). Font family/weight/italic/size are taken from `style`. Lines that
/// overflow the bottom of `bounds` are clipped.
fn paint_text(
    buf: &mut [u32],
    buf_w: u32,
    bounds: &Rect,
    text: &str,
    color: u32,
    align: &str,
    style: &TextStyle,
) {
    let size_px = style.size_px;
    let line_height = (size_px * 1.4).ceil() as i32;
    let r = ((color >> 16) & 0xFF) as u32;
    let g = ((color >> 8) & 0xFF) as u32;
    let b = (color & 0xFF) as u32;
    let reg = font_registry();

    // Collect all display lines (honouring both hard breaks and soft-wrap).
    let mut display_lines: Vec<String> = Vec::new();
    for para in text.split('\n') {
        if bounds.w == 0 {
            display_lines.push(para.to_string());
        } else {
            display_lines.extend(wrap_paragraph(para, bounds.w, size_px));
        }
    }

    let mut baseline_y = bounds.y + size_px.ceil() as i32;

    for line in &display_lines {
        // Stop if we've gone past the bottom of the bounds.
        if baseline_y - size_px.ceil() as i32 >= bounds.y + bounds.h as i32 {
            break;
        }

        // Compute line pixel width for alignment.
        let line_px_w: f32 = line
            .chars()
            .map(|ch| {
                reg.metrics(ch, size_px, style.family, style.weight, style.italic)
                    .advance_width
            })
            .sum();

        let cursor_x_start = match align {
            "center" => {
                let avail = bounds.w as f32;
                bounds.x + ((avail - line_px_w) / 2.0).max(0.0).floor() as i32
            }
            "right" => {
                let avail = bounds.w as f32;
                bounds.x + (avail - line_px_w).max(0.0).floor() as i32
            }
            _ => bounds.x, // "left" or default
        };

        let mut cursor_x = cursor_x_start;

        for ch in line.chars() {
            let (metrics, bitmap) =
                reg.rasterize(ch, size_px, style.family, style.weight, style.italic);
            let glyph_x = cursor_x + metrics.xmin;
            let glyph_y = baseline_y - metrics.height as i32 - metrics.ymin;

            for row in 0..metrics.height {
                for col in 0..metrics.width {
                    let coverage = bitmap[row * metrics.width + col] as u32;
                    if coverage == 0 {
                        continue;
                    }

                    let px = glyph_x + col as i32;
                    let py = glyph_y + row as i32;

                    // Clip to bounds rect, PAINT_CLIP scissor, and screen edges.
                    let (clip_y0, clip_y1) = PAINT_CLIP.with(|c| {
                        c.get().map_or((i32::MIN, i32::MAX), |(_, cy0, _, cy1)| (cy0, cy1))
                    });
                    if px < bounds.x
                        || px >= bounds.x + bounds.w as i32
                        || py < bounds.y
                        || py >= bounds.y + bounds.h as i32
                        || py < 0
                        || px < 0
                        || py < clip_y0
                        || py >= clip_y1
                    {
                        continue;
                    }

                    let idx = (py as u32 * buf_w + px as u32) as usize;
                    if idx >= buf.len() {
                        continue;
                    }

                    let dst = buf[idx];
                    let dr = (dst >> 16) & 0xFF;
                    let dg = (dst >> 8) & 0xFF;
                    let db = dst & 0xFF;

                    let out_r = (r * coverage + dr * (255 - coverage)) / 255;
                    let out_g = (g * coverage + dg * (255 - coverage)) / 255;
                    let out_b = (b * coverage + db * (255 - coverage)) / 255;

                    buf[idx] = (out_r << 16) | (out_g << 8) | out_b;
                }
            }

            cursor_x += metrics.advance_width.ceil() as i32;
        }

        baseline_y += line_height;
    }
}

/// Paint a vertical stripe on the left edge of `r` with the given `width` and
/// `color`. Used to render `data-border-left` semantic accents (green=ok,
/// red=err, blue=running, mauve=user, etc.).
fn paint_border_top(buf: &mut [u32], r: &Rect, buf_w: u32, width: u32, color: u32) {
    if width == 0 || r.w == 0 {
        return;
    }
    let stripe = Rect { x: r.x, y: r.y, w: r.w, h: width.min(r.h) };
    fill_rect(buf, &stripe, buf_w, color);
}

fn paint_border_left(buf: &mut [u32], r: &Rect, buf_w: u32, width: u32, color: u32) {
    if width == 0 || r.h == 0 {
        return;
    }
    let stripe = Rect { x: r.x, y: r.y, w: width.min(r.w), h: r.h };
    fill_rect(buf, &stripe, buf_w, color);
}

fn paint_border_bottom(buf: &mut [u32], r: &Rect, buf_w: u32, width: u32, color: u32) {
    if width == 0 || r.w == 0 {
        return;
    }
    let h = width.min(r.h);
    let stripe = Rect { x: r.x, y: r.y + r.h as i32 - h as i32, w: r.w, h };
    fill_rect(buf, &stripe, buf_w, color);
}

fn paint_border_right(buf: &mut [u32], r: &Rect, buf_w: u32, width: u32, color: u32) {
    if width == 0 || r.h == 0 {
        return;
    }
    let w = width.min(r.w);
    let stripe = Rect { x: r.x + r.w as i32 - w as i32, y: r.y, w, h: r.h };
    fill_rect(buf, &stripe, buf_w, color);
}

fn fill_rect(buf: &mut [u32], r: &Rect, buf_w: u32, color: u32) {
    let clip = PAINT_CLIP.with(|c| c.get());
    let raw_x0 = r.x.max(0);
    let raw_y0 = r.y.max(0);
    let raw_x1 = r.x + r.w as i32;
    let raw_y1 = r.y + r.h as i32;
    let (x0, y0, x1, y1) = if let Some((cx0, cy0, cx1, cy1)) = clip {
        (raw_x0.max(cx0).max(0) as u32,
         raw_y0.max(cy0).max(0) as u32,
         raw_x1.min(cx1).max(0) as u32,
         raw_y1.min(cy1).max(0) as u32)
    } else {
        (raw_x0.max(0) as u32, raw_y0.max(0) as u32,
         raw_x1.max(0) as u32, raw_y1.max(0) as u32)
    };
    if x0 >= x1 || y0 >= y1 { return; }
    for y in y0..y1 {
        let row = (y * buf_w) as usize;
        for x in x0..x1 {
            let idx = row + x as usize;
            if idx < buf.len() {
                buf[idx] = color;
            }
        }
    }
}

// ── Event dispatch ───────────────────────────────────────────────────────────

/// Walk the scene tree to find the nearest `data-context` attribute on the
/// path from the root to the deepest node containing `(x, y)`.
///
/// Used for right-click dispatch — a session message row tags itself with
/// `data-context="session-message"` once; any inner glyph or wrapper hit-tested
/// below it surfaces that tag without each child needing to re-declare it.
fn find_context_at(node: &NativeNode, x: i32, y: i32) -> Option<String> {
    fn walk(node: &NativeNode, x: i32, y: i32, current: Option<String>) -> Option<String> {
        let (bounds, attr_ctx, children) = {
            let inner = node.0.borrow();
            (
                inner.bounds,
                inner.attrs.get("data-context").cloned(),
                inner.children.clone(),
            )
        };

        if x < bounds.x
            || x >= bounds.x + bounds.w as i32
            || y < bounds.y
            || y >= bounds.y + bounds.h as i32
        {
            return None;
        }

        // The nearest data-context wins; carry it down as we descend.
        let next_ctx = attr_ctx.or(current);

        for child in children.iter().rev() {
            if let Some(found) = walk(child, x, y, next_ctx.clone()) {
                return Some(found);
            }
        }
        next_ctx
    }
    walk(node, x, y, None)
}

/// Returns `true` if any node in the tree rooted at `node` (within the cursor
/// bounds) carries `key = value`.  Used to detect `data-drag-window="true"` so
/// the event loop can call `window.drag_window()` before firing a click event.
fn hit_test_attr(node: &NativeNode, x: f64, y: f64, key: &str, value: &str) -> bool {
    let inner = node.0.borrow();
    let b = inner.bounds;
    if (x as i32) < b.x || (x as i32) >= b.x + b.w as i32
        || (y as i32) < b.y || (y as i32) >= b.y + b.h as i32
    {
        return false;
    }
    if inner.attrs.get(key).map(|v| v == value).unwrap_or(false) {
        return true;
    }
    inner.children.iter().any(|c| hit_test_attr(c, x, y, key, value))
}

fn hit_test(node: &NativeNode, x: f64, y: f64) -> Option<NativeNode> {
    let (bounds, children, is_text) = {
        let inner = node.0.borrow();
        let is_text = matches!(inner.kind, NodeKind::Text(_));
        (inner.bounds, inner.children.clone(), is_text)
    };

    if (x as i32) < bounds.x
        || (x as i32) >= bounds.x + bounds.w as i32
        || (y as i32) < bounds.y
        || (y as i32) >= bounds.y + bounds.h as i32
    {
        return None;
    }

    // Deepest child wins (painted last = on top).
    for child in children.iter().rev() {
        if let Some(hit) = hit_test(child, x, y) {
            return Some(hit);
        }
    }

    // Text nodes are rendering leaves — they never carry event handlers.
    // Return None so the parent container (e.g. the button) is the hit target.
    if is_text {
        return None;
    }

    Some(node.clone())
}

/// Walk the scene tree to find the innermost `data-scroll-y="true"` container
/// whose bounds contain `(x, y)`. Used to route MouseWheel events.
fn find_scroll_ancestor(node: &NativeNode, x: f64, y: f64) -> Option<NativeNode> {
    let (bounds, is_scroll, children) = {
        let inner = node.0.borrow();
        let b = inner.bounds;
        let scroll = inner.attrs.get("data-scroll-y").map(|v| v == "true").unwrap_or(false);
        let ch = inner.children.clone();
        (b, scroll, ch)
    };
    if (x as i32) < bounds.x || (x as i32) >= bounds.x + bounds.w as i32
        || (y as i32) < bounds.y || (y as i32) >= bounds.y + bounds.h as i32
    {
        return None;
    }
    for child in children.iter() {
        if let Some(found) = find_scroll_ancestor(child, x, y) {
            return Some(found);
        }
    }
    if is_scroll { Some(node.clone()) } else { None }
}

fn fire_event(node: &NativeNode, event_name: &str, brick_event: BrickEvent) {
    let handlers: Vec<Rc<dyn Fn(BrickEvent)>> = {
        let inner = node.0.borrow();
        inner
            .event_handlers
            .iter()
            .filter(|(name, _)| name == event_name)
            .map(|(_, h)| Rc::clone(h))
            .collect()
    };
    for h in handlers {
        h(brick_event.clone());
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn parse_hex_color(s: &str) -> Option<u32> {
    let s = s.strip_prefix('#')?;
    let v = u32::from_str_radix(s, 16).ok()?;
    if v <= 0x00FF_FFFF { Some(v) } else { None }
}

/// Draw a 1-pixel focus ring around `rect` using `color` (xRGB).
pub fn paint_focus_ring(buf: &mut [u32], buf_w: u32, rect: Rect, color: u32) {
    let x0 = rect.x.max(0) as u32;
    let y0 = rect.y.max(0) as u32;
    let x1 = (rect.x + rect.w as i32).max(0) as u32;
    let y1 = (rect.y + rect.h as i32).max(0) as u32;
    // Top and bottom edges
    for x in x0..x1 {
        let i = (y0 * buf_w + x) as usize;
        if i < buf.len() {
            buf[i] = color;
        }
    }
    for x in x0..x1 {
        let i = ((y1.saturating_sub(1)) * buf_w + x) as usize;
        if i < buf.len() {
            buf[i] = color;
        }
    }
    // Left and right edges
    for y in y0..y1 {
        let i = (y * buf_w + x0) as usize;
        if i < buf.len() {
            buf[i] = color;
        }
    }
    for y in y0..y1 {
        let i = (y * buf_w + x1.saturating_sub(1)) as usize;
        if i < buf.len() {
            buf[i] = color;
        }
    }
}

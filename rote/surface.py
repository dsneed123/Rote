"""How Rote perceives and acts on an application — the surface seam.

The recorded flow (schema.py) says *what* to do in surface-neutral terms: a
control with this role and name, the field labelled "Member Number", the
Balance cell of the PRIMARY SAVINGS row. A Surface turns that into *how* on
one kind of UI. WebSurface does it with Playwright; a desktop surface would do
the same over UI Automation or AT-SPI, whose trees also expose role, name and
neighbouring labels. Nothing above this module imports Playwright.

Three responsibilities:

  resolve   Target → exactly one live control, trying strategies in order.
            Zero matches or several both mean "not this strategy"; using a
            later strategy is reported so drift is visible.
  act       perform an action, after Guard.check() — the only way to the UI.
  observe   describe the screen for the model during discovery, and turn a
            control the model picked into ranked, verified locators.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urljoin, urlparse

from .policy import Guard, PolicyViolation, Redactor
from .schema import (
    LOCATOR_RANK, Action, Check, CssLocator, ElementCheck, FrameRef, LabelNearLocator,
    Locator, RegexCheck, Risk, RoleLocator, TableCellLocator, Target, TextCheck,
    TextLocator, UrlCheck,
)


class TargetNotFound(Exception):
    def __init__(self, target: Target, tried: list[str]) -> None:
        super().__init__(f"{target.description or 'target'} not found; tried: {'; '.join(tried)}")
        self.target = target
        self.tried = tried


class Interrupted(Exception):
    """Resolution or a checkpoint wait was cut short because the screen matched
    a known runtime condition (see replay.py)."""

    def __init__(self, detected: Any) -> None:
        super().__init__(str(detected))
        self.detected = detected


@dataclass
class Resolved:
    handle: Any                         # surface-specific (a Playwright Locator)
    strategy: Locator
    strategy_index: int
    control: str                        # visible name, for policy and logs


@dataclass
class Element:
    """One actionable control, as the model sees it."""

    id: int
    frame: str
    role: str
    name: str = ""
    label: str = ""                     # the label beside it, if any
    value: str = ""
    options: list[str] = field(default_factory=list)

    def line(self) -> str:
        parts = [f"[{self.id}] {self.role}"]
        if self.name:
            parts.append(f'"{self.name}"')
        if self.label and self.label != self.name:
            parts.append(f'label="{self.label}"')
        if self.value:
            parts.append(f'value="{self.value}"')
        if self.options:
            parts.append(f"options={self.options}")
        parts.append(f"(frame {self.frame})")
        return " ".join(parts)


@dataclass
class Observation:
    frames: dict[str, str]              # frame name → URL path
    elements: list[Element]
    text: dict[str, str]                # frame name → visible text
    screenshot: Path | None = None

    def element(self, element_id: int) -> Element | None:
        return next((e for e in self.elements if e.id == element_id), None)


class Surface(Protocol):
    """What the replay engine and the agent need from any surface."""

    def start(self) -> None: ...
    def close(self) -> None: ...
    def navigate(self, path: str) -> None: ...
    def resolve(self, target: Target, timeout_ms: int,
                interrupt: Callable[[], Any] | None = None) -> Resolved: ...
    def perform(self, action: Action, resolved: Resolved | None, value: str | None,
                declared: Risk | None = None, step_id: str = "",
                inputs: dict | None = None) -> Risk: ...
    def read(self, resolved: Resolved) -> str: ...
    def holds(self, check: Check, inputs: dict[str, Any] | None = None) -> bool: ...
    def visible_text(self) -> str: ...
    def location(self) -> dict[str, str]: ...
    def observe(self, screenshot: Path | None = None) -> Observation: ...
    def locators_for(self, element_id: int) -> tuple[Target, str]: ...
    def locators_for_value(self, value: str) -> tuple[Target, str]: ...
    def screenshot(self, path: Path) -> Path: ...
    def accessibility_snapshot(self) -> str: ...


def render(template: str, inputs: dict[str, Any] | None = None,
           secrets: Callable[[str], str] | None = None) -> str:
    """Fill {{inputs.x}} and {{secrets.x}}. Secrets resolve only at the last moment."""

    def sub(match: re.Match) -> str:
        kind, name = match.group(1), match.group(2)
        if kind == "inputs":
            if not inputs or name not in inputs:
                raise KeyError(f"no value for input {name!r}")
            return str(inputs[name])
        if secrets is None:
            raise KeyError(f"secret {name!r} requested but no secret source configured")
        return secrets(name)

    from .schema import TEMPLATE
    return TEMPLATE.sub(sub, template)


# --- in-page helpers ---------------------------------------------------------------
#
# The same JS computes a control's role, name and neighbouring label at record
# time and at replay time, so what was recorded is what gets matched.

_JS_HELPERS = r"""
const txt = n => ((n && (n.innerText || n.textContent)) || '').replace(/\s+/g, ' ').trim();
const norm = s => (s || '').replace(/\s+/g, ' ').trim().replace(/:$/, '').trim().toLowerCase();
const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
  return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
const roleOf = el => {
  const r = el.getAttribute('role'); if (r) return r;
  const t = el.tagName.toLowerCase();
  if (t === 'a') return 'link'; if (t === 'button') return 'button';
  if (t === 'select') return el.multiple ? 'listbox' : 'combobox';
  if (t === 'textarea') return 'textbox';
  if (t === 'input') { const ty = (el.type || 'text').toLowerCase();
    if (['submit','button','reset','image'].includes(ty)) return 'button';
    if (ty === 'checkbox') return 'checkbox'; if (ty === 'radio') return 'radio';
    return 'textbox'; }
  if (t === 'td' || t === 'th') return 'cell';
  return 'generic';
};
const nameOf = el => {
  const a = el.getAttribute('aria-label'); if (a) return a.trim();
  if (el.id) { const l = el.ownerDocument.querySelector('label[for="' + CSS.escape(el.id) + '"]'); if (l) return txt(l); }
  const wrap = el.closest('label'); if (wrap) return txt(wrap);
  const t = el.tagName.toLowerCase();
  if (t === 'input' && ['submit','button','reset'].includes((el.type||'').toLowerCase())) return (el.value || '').trim();
  if (t === 'input' && el.type === 'image') return el.alt || '';
  if (t === 'a' || t === 'button') return txt(el) || el.title || '';
  return el.title || el.placeholder || '';
};
const labelNear = el => {
  const cell = el.closest('td, th');
  if (cell) { let p = cell.previousElementSibling;
    while (p) { const s = txt(p); if (s) return s; p = p.previousElementSibling; } }
  return '';
};
const CONTROLS = 'a[href], button, input:not([type=hidden]), select, textarea, [onclick], [role=button], [role=link]';
"""

_JS_ENUMERATE = _JS_HELPERS + r"""
(startId) => {
  document.querySelectorAll('[data-rote-id]').forEach(e => e.removeAttribute('data-rote-id'));
  const out = []; let id = startId;
  for (const el of document.querySelectorAll(CONTROLS)) {
    if (!vis(el)) continue;
    el.setAttribute('data-rote-id', String(id));
    const role = roleOf(el);
    const item = {id, role, name: nameOf(el), label: labelNear(el), value: '', options: []};
    if (role === 'textbox') item.value = el.type === 'password' ? (el.value ? '••••' : '') : (el.value || '');
    if (el.tagName.toLowerCase() === 'select') {
      item.options = [...el.options].map(o => o.text.trim());
      item.value = el.selectedIndex >= 0 ? el.options[el.selectedIndex].text.trim() : '';
    }
    out.push(item); id++;
  }
  return out;
}
"""

_JS_LABEL_NEAR = _JS_HELPERS + r"""
([label, role, nonce]) => {
  const want = norm(label); let n = 0;
  const pool = role === 'cell' ? document.querySelectorAll('td, th') : document.querySelectorAll(CONTROLS);
  for (const el of pool) {
    if (roleOf(el) !== role || !vis(el)) continue;
    if (norm(labelNear(el)) === want) { el.setAttribute('data-rote-match', nonce); n++; }
  }
  return n;
}
"""

_JS_TABLE_CELL = _JS_HELPERS + r"""
([rowCol, rowVal, col, nonce]) => {
  let n = 0;
  for (const table of document.querySelectorAll('table')) {
    const rows = [...table.rows]; let hi = -1, rc = -1, cc = -1;
    for (let i = 0; i < rows.length && hi < 0; i++) {
      const cells = [...rows[i].cells].map(c => norm(txt(c)));
      const a = cells.indexOf(norm(rowCol)), b = cells.indexOf(norm(col));
      if (a >= 0 && b >= 0 && a !== b) { hi = i; rc = a; cc = b; }
    }
    if (hi < 0) continue;
    for (let i = hi + 1; i < rows.length; i++) {
      const cells = rows[i].cells;
      if (cells.length <= Math.max(rc, cc)) continue;
      if (norm(txt(cells[rc])) === norm(rowVal)) { cells[cc].setAttribute('data-rote-match', nonce); n++; }
    }
  }
  return n;
}
"""

# Describe one element (tagged data-rote-id) as candidate locators.
_JS_DESCRIBE = _JS_HELPERS + r"""
(id) => {
  const el = document.querySelector('[data-rote-id="' + id + '"]');
  if (!el) return null;
  const t = el.tagName.toLowerCase();
  const out = {role: roleOf(el), name: nameOf(el), label: labelNear(el), tag: t,
               text: (t === 'a' || t === 'button') ? txt(el) : '', css: '', cells: []};
  const nm = el.getAttribute('name');
  if (nm && document.querySelectorAll(t + '[name="' + nm + '"]').length === 1) out.css = t + '[name="' + nm + '"]';
  else { // structural path — the brittle last resort
    const parts = []; let n = el;
    while (n && n.nodeType === 1 && n !== document.body) {
      const p = n.parentElement; if (!p) break;
      const same = [...p.children].filter(c => c.tagName === n.tagName);
      parts.unshift(n.tagName.toLowerCase() + (same.length > 1 ? ':nth-of-type(' + (same.indexOf(n) + 1) + ')' : ''));
      n = p;
    }
    out.css = 'body > ' + parts.join(' > ');
  }
  const cell = el.closest('td, th');
  const table = cell && cell.closest('table');
  if (cell && table) {
    const rows = [...table.rows]; const row = cell.parentElement;
    const ci = [...row.cells].indexOf(cell); const header = rows[0];
    // A header row has to look like one: every cell a <th> or entirely bold.
    // Otherwise "Member | 12345" would pass as a header named "12345".
    const isHead = c => c.tagName === 'TH' ||
      [...c.querySelectorAll('b, strong')].some(b => txt(b) && txt(b) === txt(c));
    if (header !== row && header.cells.length === row.cells.length && [...header.cells].every(isHead)) {
      const heads = [...header.cells].map(c => txt(c));
      const data = rows.slice(1);
      const keys = [];
      heads.forEach((h, k) => {
        if (k === ci || !h) return;
        const v = txt(row.cells[k]); if (!v) return;
        const same = data.filter(r => r.cells[k] && txt(r.cells[k]) === v).length;
        if (same === 1) keys.push({rowColumn: h, rowValue: v, numeric: /^[\d$.,() -]+$/.test(v)});
      });
      // Descriptive keys first ("PRIMARY SAVINGS" over "S00"): they read well in review.
      keys.sort((a, b) => (a.numeric - b.numeric) || (b.rowValue.length - a.rowValue.length));
      out.cells = keys.slice(0, 2).map(k => ({row_column: k.rowColumn, row_value: k.rowValue, column: heads[ci]}));
    }
  }
  return out;
}
"""

_JS_FIND_VALUE = _JS_HELPERS + r"""
([value, startId]) => {
  // The deepest elements whose whole text is exactly the value.
  const want = value.replace(/\s+/g, ' ').trim(); const hits = [];
  for (const el of document.querySelectorAll('body *')) {
    if (!vis(el) || txt(el) !== want) continue;
    if ([...el.children].some(c => txt(c) === want)) continue;
    hits.push(el.closest('td, th') || el);
  }
  let id = startId;
  return [...new Set(hits)].map(el => { el.setAttribute('data-rote-id', String(id)); return id++; });
}
"""

# Recorder for human actions during a handoff: reports what the operator
# clicked or changed, described the same way the agent's targets are.
_JS_HUMAN_CAPTURE = _JS_HELPERS + r"""
(() => {
  if (window.__roteCapture) return; window.__roteCapture = true;
  const send = (kind, el) => {
    if (!window.__rote_event) return;
    const t = el.tagName.toLowerCase();
    const secret = t === 'input' && (el.type || '').toLowerCase() === 'password';
    window.__rote_event({kind, role: roleOf(el), name: nameOf(el), label: labelNear(el),
      value: kind === 'change' ? (secret ? '<password>' : (el.value || '')) : '',
      frame: window.name || 'top', path: location.pathname});
  };
  document.addEventListener('click', e => { const el = e.target.closest(CONTROLS); if (el) send('click', el); }, true);
  document.addEventListener('change', e => send('change', e.target), true);
})();
"""


def _xpath_free(text: str) -> str:
    return text.replace("\n", " ").strip()


class WebSurface:
    """A Chromium session driven through Playwright.

    `cdp_port` exposes the same browser over the DevTools protocol so an
    operator (or the mock operator console) can attach to *this* session during
    a handoff rather than starting a fresh one.
    """

    POLL_S = 0.15

    def __init__(self, base_url: str, guard: Guard, redactor: Redactor, *,
                 headless: bool = True, cdp_port: int | None = None,
                 secrets: Callable[[str], str] | None = None,
                 on_human_event: Callable[[dict], None] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.guard = guard
        self.redactor = redactor
        self.headless = headless
        self.cdp_port = cdp_port
        self.secrets = secrets
        self.on_human_event = on_human_event
        self.blocked: list[str] = []      # requests aborted by the allowlist
        self._inflight = 0
        self._text_cache: dict[Any, tuple[float, str]] = {}
        self._last_activity = 0.0
        self._pw = self._browser = self._context = self._page = None

    # -- lifecycle --

    def start(self) -> None:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        args = [f"--remote-debugging-port={self.cdp_port}"] if self.cdp_port else []
        self._browser = self._pw.chromium.launch(headless=self.headless, args=args)
        self._context = self._browser.new_context(viewport={"width": 1280, "height": 800})
        # Network-level allowlist: a click that would leave the allowlist is
        # stopped by the browser, not noticed afterwards.
        self._context.route("**/*", self._route)
        self._context.on("request", lambda r: self._activity(+1))
        self._context.on("requestfinished", lambda r: self._activity(-1))
        self._context.on("requestfailed", lambda r: self._activity(-1))
        self._context.expose_binding("__rote_event", self._human_event)
        self._context.add_init_script(_JS_HUMAN_CAPTURE)
        self._page = self._context.new_page()
        self._page.set_default_timeout(10_000)

    def close(self) -> None:
        for closer in (lambda: self._context.close(), lambda: self._browser.close(), lambda: self._pw.stop()):
            try:
                closer()
            except Exception:
                pass
        self._pw = self._browser = self._context = self._page = None

    @property
    def page(self):
        if self._page is None:
            raise RuntimeError("surface not started")
        return self._page

    @property
    def cdp_endpoint(self) -> str | None:
        return f"http://127.0.0.1:{self.cdp_port}" if self.cdp_port else None

    def _route(self, route, request) -> None:
        if self.guard.url_allowed(request.url):
            route.continue_()
        else:
            self.blocked.append(request.url)
            route.abort("blockedbyclient")

    def _activity(self, delta: int) -> None:
        self._inflight = max(0, self._inflight + delta)
        self._last_activity = time.monotonic()

    def _human_event(self, source, event: dict) -> None:
        if self.on_human_event:
            self.on_human_event(event)

    # -- frames and location --

    def _frames(self) -> list:
        return [f for f in self.page.frames if not f.is_detached()]

    @staticmethod
    def _frame_label(frame) -> str:
        return frame.name or ("top" if frame.parent_frame is None else urlparse(frame.url).path)

    def frame(self, ref: FrameRef | None):
        """The frame a ref names, or None. A ref of None means the top document."""
        frames = self._frames()
        if ref is None:
            return self.page.main_frame
        for frame in frames:
            if ref.name and frame.name == ref.name:
                return frame
        if ref.url_path:
            for frame in frames:
                if urlparse(frame.url).path == ref.url_path:
                    return frame
        return None

    def _search_frames(self, ref: FrameRef | None) -> list:
        """Frames to search for a target. With no ref, every frame: a frameset's
        top document holds no controls, and broke-out pages have no frames."""
        if ref is None:
            return self._frames()
        frame = self.frame(ref)
        if frame is not None:
            return [frame]
        # The named frame is gone — e.g. an expiry page broke out to the top.
        return []

    def location(self) -> dict[str, str]:
        return {self._frame_label(f): urlparse(f.url).path or "/" for f in self._frames()}

    def settle(self, timeout_ms: int = 5_000, quiet_ms: int = 250) -> None:  # noqa: C901
        """Let whatever the last action set off finish, in every frame.

        A click returns before the navigation it causes has begun, so "wait for
        load" alone passes too early. Instead wait for the network to be quiet
        (nothing in flight for quiet_ms), then for each frame's load event.
        Event processing needs the Playwright loop to run, hence the short
        page waits rather than time.sleep.
        """
        self._text_cache.clear()
        deadline = time.monotonic() + timeout_ms / 1000
        self._last_activity = max(self._last_activity, time.monotonic())
        while time.monotonic() < deadline:
            idle = time.monotonic() - self._last_activity
            if self._inflight == 0 and idle * 1000 >= quiet_ms:
                break
            self.page.wait_for_timeout(50)
        for frame in self._frames():
            try:
                frame.wait_for_load_state("load", timeout=timeout_ms)
            except Exception:
                pass
        self._raise_if_blocked()

    def _raise_if_blocked(self) -> None:
        if self.blocked:
            urls, self.blocked = self.blocked, []
            raise PolicyViolation(f"navigation outside the allowlist blocked: {urls[0]}", "allowlist")

    # -- resolution --

    def _match(self, frame, strategy: Locator):
        """A Playwright locator for a strategy within one frame."""
        if isinstance(strategy, RoleLocator):
            return frame.get_by_role(strategy.role, name=strategy.name, exact=strategy.exact)
        if isinstance(strategy, TextLocator):
            return frame.get_by_text(strategy.text, exact=strategy.exact)
        if isinstance(strategy, CssLocator):
            return frame.locator(strategy.css)
        nonce = uuid.uuid4().hex[:10]
        if isinstance(strategy, LabelNearLocator):
            frame.evaluate(_JS_LABEL_NEAR, [strategy.label, strategy.role, nonce])
        elif isinstance(strategy, TableCellLocator):
            frame.evaluate(_JS_TABLE_CELL, [strategy.row_column, strategy.row_value, strategy.column, nonce])
        return frame.locator(f"[data-rote-match='{nonce}']")

    def _unique(self, target: Target, strategy: Locator) -> tuple[Any, int]:
        """(locator, count of visible matches across the searched frames)."""
        found, count = None, 0
        for frame in self._search_frames(target.frame):
            try:
                locator = self._match(frame, strategy)
                matches = [m for m in locator.all() if m.is_visible()]
            except Exception:
                continue   # frame navigating mid-query; the next poll will see it settled
            count += len(matches)
            if matches:
                found = matches[0]
        return found, count

    def resolve(self, target: Target, timeout_ms: int = 10_000,
                interrupt: Callable[[], Any] | None = None) -> Resolved:
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            tried = []
            for index, strategy in enumerate(target.strategies):
                handle, count = self._unique(target, strategy)
                if count == 1:
                    return Resolved(handle, strategy, index, self._control_name(handle, target))
                tried.append(f"{strategy.by}: {count} match{'es' if count != 1 else ''}")
            if interrupt:
                detected = interrupt()
                if detected:
                    raise Interrupted(detected)
            if time.monotonic() >= deadline:
                raise TargetNotFound(target, tried)
            self.page.wait_for_timeout(self.POLL_S * 1000)

    @staticmethod
    def _control_name(handle, target: Target) -> str:
        try:
            name = handle.evaluate(_JS_HELPERS + "\n(el) => nameOf(el) || labelNear(el) || txt(el)")
        except Exception:
            name = ""
        return (name or target.description)[:80]

    # -- acting --

    def navigate(self, path: str) -> None:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        self.guard.check_url(url)
        self.guard.check(Action.navigate, path)
        self.page.goto(url, wait_until="load")
        self.settle()

    def perform(self, action: Action, resolved: Resolved | None, value: str | None,
                declared: Risk | None = None, step_id: str = "", inputs: dict | None = None) -> Risk:
        """Policy-check then execute one action. The only way to touch the UI."""
        control = resolved.control if resolved else (value or "")
        risk = self.guard.check(action, control, declared, step_id)
        text = render(value, inputs, self.secrets) if value is not None else None
        if action == Action.navigate:
            self.navigate(text or "/")
        elif action == Action.click:
            resolved.handle.click()
        elif action == Action.fill:
            resolved.handle.fill(text or "")
            # Blur now so the field's change event fires while automation owns
            # the session; otherwise it fires on the operator's first click
            # during a handoff and is misattributed to them.
            resolved.handle.evaluate("el => el.blur()")
        elif action == Action.select:
            try:
                resolved.handle.select_option(label=text)
            except Exception:
                resolved.handle.select_option(value=text)
        elif action == Action.press:
            (resolved.handle.press if resolved else self.page.keyboard.press)(text)
        elif action in (Action.extract, Action.wait_for):
            pass
        self.settle()
        return risk

    def read(self, resolved: Resolved) -> str:
        tag = resolved.handle.evaluate("el => el.tagName.toLowerCase()")
        if tag in ("input", "textarea", "select"):
            return resolved.handle.input_value()
        return " ".join(resolved.handle.inner_text().split())

    # -- checks --

    TEXT_TTL_S = 0.1

    def _body_text(self, frame) -> str | None:
        """A frame's visible text, cached briefly: one detection pass runs a
        dozen checks against the same screen. Any action clears the cache."""
        now = time.monotonic()
        cached = self._text_cache.get(frame)
        if cached and now - cached[0] < self.TEXT_TTL_S:
            return cached[1]
        try:
            # evaluate, not locator("body").inner_text(): a frameset document
            # has no <body>, and a locator would wait for one to appear.
            text = frame.evaluate("() => document.body ? document.body.innerText : ''")
        except Exception:
            text = None   # navigating; cached too, so a detection pass doesn't retry it 12 times
        self._text_cache[frame] = (now, text)
        return text

    def frame_text(self, ref: FrameRef | None) -> str | None:
        frames = self._frames() if ref is None else [f for f in [self.frame(ref)] if f]
        texts = [t for t in (self._body_text(f) for f in frames) if t is not None]
        if not texts:
            return None
        return "\n".join(texts)

    def visible_text(self) -> str:
        return self.frame_text(None) or ""

    def holds(self, check: Check, inputs: dict[str, Any] | None = None) -> bool:
        if isinstance(check, (TextCheck, RegexCheck)):
            text = self.frame_text(check.frame)
            if text is None:
                result = False
            elif isinstance(check, TextCheck):
                result = render(check.text, inputs) in text
            else:
                result = re.search(check.pattern, text) is not None
        elif isinstance(check, UrlCheck):
            frames = [self.frame(check.frame)] if check.frame else self._frames()
            result = any(f is not None and re.search(check.path_regex, urlparse(f.url).path or "/")
                         for f in frames)
        elif isinstance(check, ElementCheck):
            try:
                self.resolve(check.target, timeout_ms=0)
                result = True
            except TargetNotFound:
                result = False
        else:
            raise TypeError(f"unknown check {check!r}")
        return result != check.negate

    # -- observation (discovery) --

    def observe(self, screenshot: Path | None = None) -> Observation:
        self.settle()
        elements: list[Element] = []
        text: dict[str, str] = {}
        next_id = 1
        for frame in self._frames():
            label = self._frame_label(frame)
            try:
                found = frame.evaluate(_JS_ENUMERATE, next_id)
            except Exception:
                continue
            body = self._body_text(frame) or ""
            for item in found:
                elements.append(Element(frame=label, **item))
            next_id += len(found)
            if body.strip():
                text[label] = body.strip()
        shot = self.screenshot(screenshot) if screenshot else None
        return Observation(self.location(), elements, text, shot)

    def _frame_ref(self, frame) -> FrameRef | None:
        if frame.parent_frame is None:
            return None
        if frame.name:
            return FrameRef(name=frame.name)
        return FrameRef(url_path=urlparse(frame.url).path)

    def _tagged(self, element_id: int):
        for frame in self._frames():
            handle = frame.locator(f"[data-rote-id='{element_id}']")
            if handle.count():
                return frame, handle.first
        raise KeyError(f"no element {element_id} on the current screen")

    def locators_for(self, element_id: int) -> tuple[Target, str]:
        """Ranked locators for an observed element, each verified against the
        live page to resolve uniquely to that same element.

        Only verified strategies are recorded — so every locator in a fresh
        artifact is known to have worked at least once, on the real screen.
        """
        frame, handle = self._tagged(element_id)
        info = frame.evaluate(_JS_DESCRIBE, element_id)
        candidates: list[Locator] = []
        if info["name"] and info["role"] not in ("generic", "cell"):
            candidates.append(RoleLocator(role=info["role"], name=_xpath_free(info["name"])))
        if info["label"]:
            candidates.append(LabelNearLocator(label=_xpath_free(info["label"]), role=info["role"]))
        if info["text"]:
            candidates.append(TextLocator(text=_xpath_free(info["text"])))
        for cell in info["cells"]:
            candidates.append(TableCellLocator(**cell))
        if info["css"]:
            candidates.append(CssLocator(css=info["css"]))

        ref = self._frame_ref(frame)
        verified = []
        for candidate in candidates:
            probe = Target(frame=ref, strategies=[candidate])
            found, count = self._unique(probe, candidate)
            if count == 1 and found.evaluate(f"el => el.getAttribute('data-rote-id') === '{element_id}'"):
                verified.append(candidate)
        if not verified:
            raise TargetNotFound(Target(strategies=candidates or [CssLocator(css="*")]),
                                 ["no candidate locator was unique"])
        # A data cell is identified by its row and column, not by whatever
        # happens to sit in the cell to its left — so in a real table,
        # table_cell outranks label_near.
        in_table = any(isinstance(s, TableCellLocator) for s in verified)
        verified.sort(key=lambda s: (0.5 if in_table and s.by == "table_cell" else LOCATOR_RANK[s.by]))
        name = info["name"] or info["label"] or info["text"] or info["tag"]
        return Target(description=f"{info['role']} {name!r}", frame=ref, strategies=verified), name

    def locators_for_value(self, value: str) -> tuple[Target, str]:
        """Locate the screen element showing exactly `value`, for extraction."""
        hits: list[int] = []
        start = 10_000
        for frame in self._frames():
            try:
                ids = frame.evaluate(_JS_FIND_VALUE, [value, start])
            except Exception:
                continue
            hits += ids
            start += len(ids) + 1
        if len(hits) != 1:
            raise TargetNotFound(Target(strategies=[TextLocator(text=value)]),
                                 [f"value shown {len(hits)} times on screen; need exactly one"])
        target, _ = self.locators_for(hits[0])
        # Locate a value by *what it is*, not by the value itself: a text
        # locator for "$1,234.56" would only ever find that exact balance.
        strategies = [s for s in target.strategies if not isinstance(s, (TextLocator, RoleLocator))]
        if not strategies:
            raise TargetNotFound(target, ["only value-based locators found; nothing stable to record"])
        return target.model_copy(update={"strategies": strategies}), value

    # -- evidence --

    def screenshot(self, path: Path) -> Path:
        """A screenshot with sensitive text masked before it is written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        masks = []
        for frame in self._frames():
            for pattern in self.redactor.screen_patterns:
                masks.append(frame.get_by_text(re.compile(pattern)))
        try:
            self.page.screenshot(path=str(path), mask=masks, mask_color="#222222", timeout=5_000)
        except Exception:
            self.page.screenshot(path=str(path), timeout=5_000)   # e.g. a frame vanished mid-mask
        return path

    def accessibility_snapshot(self) -> str:
        """Every frame's ARIA snapshot, redacted — the DOM-level evidence."""
        parts = []
        for frame in self._frames():
            try:
                snap = frame.locator("body").aria_snapshot(timeout=2_000)
            except Exception:
                continue
            parts.append(f"# frame {self._frame_label(frame)} {urlparse(frame.url).path}\n{snap}")
        return self.redactor.text("\n\n".join(parts))

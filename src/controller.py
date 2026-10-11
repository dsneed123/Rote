"""Controllers expose a set of tools and execute plan steps against them.

Each controller declares what it can operate on and which tools it offers.
bridge.py picks one; analyze.py shows the model that controller's tools so the
plan it writes can only contain calls that actually exist.

Tools are plain annotated functions with a docstring, the same shape ai.py's
`with_tools` derives schemas from, so a controller's tools can be handed to the
model directly when a step needs the model in the loop.

A controller may be unusable on a given machine (no browser installed, no input
injection on Wayland). `is_available()` reports that as (ok, reason) so the
bridge can leave it out of the menu instead of planning steps that cannot run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .plan import Step, StepResult

# Anything that changes state asks first. Rote executes model-authored steps, so
# the default is to confirm rather than to trust.
MUTATING = {"run", "write_file", "click", "type_text", "key", "goto", "fill",
            "click_text", "press"}


def ask_terminal(step: Step) -> bool:
    """Default confirmation hook: prompt on the terminal before a mutating step.

    Ctrl-C and Ctrl-D decline the step rather than killing the program — the
    operator interrupting a confirmation means "no", not "crash".
    """
    try:
        answer = input(f"\n  run {step}?\n  [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def allow_all(step: Step) -> bool:
    """Confirmation hook for unattended runs. Use only with a trusted plan."""
    return True


class Controller:
    """Base class. Subclasses set `name` and implement `tools`."""

    name: str = "base"
    description: str = ""

    def __init__(self, confirm: Callable[[Step], bool] = ask_terminal) -> None:
        self.confirm = confirm

    def tools(self) -> list[Callable]:
        """The functions this controller exposes to the model."""
        raise NotImplementedError

    def registry(self) -> dict[str, Callable]:
        return {fn.__name__: fn for fn in self.tools()}

    def is_available(self) -> tuple[bool, str]:
        """Whether this controller can actually run here, and why not if it can't."""
        return True, ""

    def close(self) -> None:
        """Release anything held open. Safe to call when nothing is."""

    def execute(self, step: Step) -> StepResult:
        """Run one step, converting any failure into a StepResult."""
        fn = self.registry().get(step.tool)
        if fn is None:
            available_tools = ", ".join(self.registry()) or "none"
            return StepResult(step, False, f"no such tool {step.tool!r}; have: {available_tools}")

        if step.tool in MUTATING and not self.confirm(step):
            return StepResult(step, False, "declined by operator")

        try:
            output = fn(**step.args)
        except TypeError as e:
            # Wrong arguments from the model — recoverable, so report rather than raise.
            return StepResult(step, False, f"bad arguments: {e}")
        except Exception as e:
            return StepResult(step, False, f"{type(e).__name__}: {e}")
        return StepResult(step, True, "" if output is None else str(output))

    def run_plan(self, plan, stop_on_failure: bool = True) -> list[StepResult]:
        """Execute every step in order. Stops at the first failure by default."""
        results: list[StepResult] = []
        for step in plan.steps:
            result = self.execute(step)
            results.append(result)
            if not result.ok and stop_on_failure:
                break
        return results


class ShellController(Controller):
    """Commands and files. No extra dependencies, works under Wayland."""

    name = "shell"
    description = "Run shell commands and read or write files in the project."

    def __init__(self, root: Path | str = ".", **kwargs) -> None:
        super().__init__(**kwargs)
        self.root = Path(root).resolve()

    def _resolve(self, filename: str) -> Path:
        """Keep file tools inside the project root.

        The model chooses these paths, so "../../etc/passwd" is a case that
        will happen eventually, not a hypothetical.
        """
        target = (self.root / filename).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"path escapes the project root: {filename}")
        return target

    def tools(self) -> list[Callable]:
        return [self.run, self.read_file, self.write_file, self.list_dir]

    def run(self, command: str, timeout: int = 60) -> str:
        """Run a shell command in the project directory and return its output."""
        proc = subprocess.run(
            command, shell=True, cwd=self.root, timeout=timeout,
            capture_output=True, text=True,
        )
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0:
            raise RuntimeError(f"exit {proc.returncode}: {output[:500]}")
        return output or "(exit 0, no output)"

    def read_file(self, filename: str) -> str:
        """Read a text file from the project directory."""
        return self._resolve(filename).read_text()

    def write_file(self, filename: str, content: str) -> str:
        """Write text to a file in the project directory, replacing it."""
        target = self._resolve(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} bytes to {filename}"

    def list_dir(self, path: str = ".") -> str:
        """List the files in a directory of the project."""
        target = self._resolve(path)
        return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir()))


# evdev keycodes, for ydotool. It takes numeric codes, not key names.
KEY_CODES: dict[str, int] = {
    "esc": 1, "escape": 1, "backspace": 14, "tab": 15, "enter": 28, "return": 28,
    "ctrl": 29, "control": 29, "shift": 42, "alt": 56, "space": 57,
    "super": 125, "meta": 125, "win": 125,
    "up": 103, "down": 108, "left": 105, "right": 106,
    "home": 102, "end": 107, "pageup": 104, "pagedown": 109,
    "insert": 110, "delete": 111, "del": 111,
    "minus": 12, "equal": 13, "comma": 51, "dot": 52, "period": 52, "slash": 53,
}
# Letters and digits are not alphabetical in evdev order — they follow the
# physical keyboard rows — so they are listed explicitly.
KEY_CODES.update({
    "q": 16, "w": 17, "e": 18, "r": 19, "t": 20, "y": 21, "u": 22, "i": 23,
    "o": 24, "p": 25, "a": 30, "s": 31, "d": 32, "f": 33, "g": 34, "h": 35,
    "j": 36, "k": 37, "l": 38, "z": 44, "x": 45, "c": 46, "v": 47, "b": 48,
    "n": 49, "m": 50,
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
})
KEY_CODES.update({f"f{n}": 58 + n for n in range(1, 11)})  # f1..f10
KEY_CODES.update({"f11": 87, "f12": 88})


class DesktopController(Controller):
    """Mouse and keyboard on the real desktop.

    Input goes through ydotool, which writes to /dev/uinput and so works under
    Wayland (unlike xdotool, which is X11-only). xdotool is used instead when
    the session is X11 and ydotool is absent.

    Setup on Wayland:
        sudo apt install ydotool
        sudo systemctl enable --now ydotoold

    Coordinates come from the model reading a screenshot, so they are pixels of
    the captured image. If the capture is scaled relative to the display, scale
    them back before injecting.
    """

    name = "desktop"
    description = "Click and type on the desktop, driven by what is on screen."

    def __init__(self, scale: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    def _backend(self) -> str:
        if shutil.which("ydotool"):
            return "ydotool"
        if shutil.which("xdotool"):
            return "xdotool"
        return ""

    def is_available(self) -> tuple[bool, str]:
        backend = self._backend()
        if not backend:
            return False, (
                "no input injection tool. On Wayland:\n"
                "      sudo apt install ydotool && sudo systemctl enable --now ydotoold"
            )
        if backend == "ydotool":
            # The daemon owns /dev/uinput; without it every call fails at the socket.
            probe = subprocess.run(["ydotool", "mousemove", "--absolute", "-x", "0", "-y", "0"],
                                   capture_output=True, text=True)
            if probe.returncode != 0:
                return False, (
                    f"ydotool present but not working ({probe.stderr.strip()[:120]}). "
                    "Is ydotoold running?"
                )
        return True, ""

    def _ydotool(self, *args: str) -> str:
        proc = subprocess.run(["ydotool", *args], capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            raise RuntimeError(f"ydotool {' '.join(args)}: {proc.stderr.strip()[:200]}")
        return proc.stdout.strip()

    def _xdotool(self, *args: str) -> str:
        proc = subprocess.run(["xdotool", *args], capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            raise RuntimeError(f"xdotool {' '.join(args)}: {proc.stderr.strip()[:200]}")
        return proc.stdout.strip()

    def tools(self) -> list[Callable]:
        return [self.click, self.type_text, self.key]

    def click(self, x: int, y: int, button: str = "left") -> str:
        """Click at a pixel coordinate on the screen."""
        x, y = int(x * self.scale), int(y * self.scale)
        if self._backend() == "ydotool":
            codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
            if button not in codes:
                raise ValueError(f"button must be left, right or middle, not {button!r}")
            self._ydotool("mousemove", "--absolute", "-x", str(x), "-y", str(y))
            self._ydotool("click", codes[button])
        else:
            numbers = {"left": "1", "middle": "2", "right": "3"}
            if button not in numbers:
                raise ValueError(f"button must be left, right or middle, not {button!r}")
            self._xdotool("mousemove", str(x), str(y), "click", numbers[button])
        return f"clicked {button} at ({x}, {y})"

    def type_text(self, text: str) -> str:
        """Type a string on the keyboard."""
        if self._backend() == "ydotool":
            self._ydotool("type", text)
        else:
            self._xdotool("type", "--delay", "12", text)
        return f"typed {len(text)} characters"

    def key(self, keys: str) -> str:
        """Press a key combination, e.g. 'ctrl+s' or 'alt+Tab'."""
        parts = [k.strip().lower() for k in keys.replace(" ", "+").split("+") if k.strip()]
        if not parts:
            raise ValueError("no keys given")

        if self._backend() == "ydotool":
            unknown = [k for k in parts if k not in KEY_CODES]
            if unknown:
                raise ValueError(f"unknown key(s): {', '.join(unknown)}")
            codes = [KEY_CODES[k] for k in parts]
            # Press in order, release in reverse, as a single ydotool invocation.
            sequence = [f"{c}:1" for c in codes] + [f"{c}:0" for c in reversed(codes)]
            self._ydotool("key", *sequence)
        else:
            self._xdotool("key", "+".join(parts))
        return f"pressed {'+'.join(parts)}"


class BrowserController(Controller):
    """A real browser via Playwright.

    Setup:
        pip install playwright && playwright install chromium

    Preferred over the desktop controller where the task is a web one: it
    addresses elements by visible text rather than screen pixels, so a recorded
    workflow survives the window moving, the theme changing, or a reboot.

    The browser is opened lazily on first use and stays open across steps, so a
    plan can navigate and then act on the page it landed on. Call close() when
    finished.
    """

    name = "browser"
    description = "Drive a web browser: navigate, click, fill, and read pages."

    def __init__(self, headless: bool = False, timeout_ms: int = 15000, **kwargs) -> None:
        super().__init__(**kwargs)
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._playwright = None
        self._browser = None
        self._page = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import playwright  # noqa: F401
        except ImportError:
            return False, "playwright is not installed. pip install playwright && playwright install chromium"
        return True, ""

    def page(self):
        """The live page, starting the browser on first use."""
        if self._page is None:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page()
            self._page.set_default_timeout(self.timeout_ms)
        return self._page

    def close(self) -> None:
        for resource in (self._browser, self._playwright):
            try:
                resource.close() if resource is self._browser else resource.stop()
            except Exception:
                pass   # best-effort teardown; the browser may already be gone
        self._playwright = self._browser = self._page = None

    def tools(self) -> list[Callable]:
        return [self.goto, self.click_text, self.fill, self.read_page, self.press]

    def goto(self, url: str) -> str:
        """Navigate the browser to a URL."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        self.page().goto(url, wait_until="domcontentloaded")
        return f"at {self.page().url}: {self.page().title()}"

    def click_text(self, text: str) -> str:
        """Click the button, link or element showing this visible text."""
        page = self.page()
        # get_by_text matches substrings and is the closest thing to what the
        # model saw on screen; fall back to a CSS/XPath selector if given one.
        try:
            page.get_by_text(text, exact=False).first.click()
        except Exception:
            page.click(text)
        return f"clicked {text!r}; now at {page.url}"

    def fill(self, label: str, value: str) -> str:
        """Type a value into the form field with this label, placeholder or name."""
        page = self.page()
        for locate in (
            lambda: page.get_by_label(label, exact=False).first,
            lambda: page.get_by_placeholder(label).first,
            lambda: page.locator(f"[name='{label}']").first,
        ):
            try:
                locate().fill(value)
                return f"filled {label!r}"
            except Exception:
                continue   # try the next way of finding the field; the raise below covers all failing
        raise RuntimeError(f"no field matching {label!r}")

    def read_page(self, selector: str = "body") -> str:
        """Return the visible text of the current page, or of one element."""
        text = self.page().locator(selector).first.inner_text()
        return text[:4000]

    def press(self, keys: str) -> str:
        """Press a key in the browser, e.g. 'Enter' or 'Control+a'."""
        self.page().keyboard.press(keys)
        return f"pressed {keys}"


def available(only_usable: bool = False, **kwargs) -> dict[str, Controller]:
    """Every controller Rote knows about, by name.

    With only_usable, leaves out the ones that cannot run here, so the bridge
    never picks a controller whose every step would fail.
    """
    controllers = [ShellController(**kwargs), DesktopController(**kwargs),
                   BrowserController(**kwargs)]
    if only_usable:
        controllers = [c for c in controllers if c.is_available()[0]]
    return {c.name: c for c in controllers}


def status(**kwargs) -> list[tuple[str, bool, str]]:
    """(name, usable, reason) for each controller — for a startup report."""
    report = []
    for controller in available(**kwargs).values():
        ok, reason = controller.is_available()
        report.append((controller.name, ok, reason))
    return report

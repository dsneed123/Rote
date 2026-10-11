"""CoreLink Teller — a deliberately legacy-looking core banking app.

A stand-in for the back-office systems Rote targets. It is hostile on purpose:

  * a frameset (banner / nav / main), so nothing useful is in the top document
  * table layouts, <font> tags, field names like f_0012, no ids, no test ids
  * form labels in the neighbouring table cell rather than <label for>, so a
    control's accessible name is often empty
  * runtime conditions a replay must deal with: record not found, validation
    errors, restricted accounts, a system-notice interstitial, session expiry,
    slowness and outright system errors

Faults are injected through /__control/faults (only when started with
--faults), so a test or a demo can make the next request slow, expire the
session, or show the notice, without the app itself knowing it is under test.

Tenant "b" is the same product configured differently: other labels, other
column order, other branding. It stands in for a second institution running
the same vendor product.

All data is fictional. Run:  python -m mockapp.corelink --port 8765 --faults
"""

from __future__ import annotations

import argparse
import html
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Demo-only operator credentials. Rote reads them from the environment
# (ROTE_SECRET_CORELINK_USER / _PASSWORD), never from an artifact.
OPERATOR_USER = "teller01"
OPERATOR_PASSWORD = "demo-pass-123"

# --- data ---------------------------------------------------------------------


def _member(name, ssn, dob, shares, restricted=False):
    return {"name": name, "ssn": ssn, "dob": dob, "shares": shares, "restricted": restricted}


def seed_members() -> dict[str, dict]:
    return {
        "12345": _member("JANE Q SAMPLE", "900-12-3456", "1984-03-02", [
            {"id": "S00", "type": "SAVINGS", "desc": "PRIMARY SAVINGS", "balance": 1234.56},
            {"id": "S10", "type": "CHECKING", "desc": "SHARE DRAFT", "balance": 310.02},
        ]),
        "20001": _member("ROBERT T EXAMPLE", "900-55-0101", "1971-11-19", [
            {"id": "S00", "type": "SAVINGS", "desc": "PRIMARY SAVINGS", "balance": 87012.00},
            {"id": "S05", "type": "SAVINGS", "desc": "HOLIDAY CLUB", "balance": 450.00},
        ]),
        "30003": _member("MARIA L PLACEHOLDER", "900-77-4242", "1990-06-30", [
            {"id": "S00", "type": "SAVINGS", "desc": "PRIMARY SAVINGS", "balance": 25.00},
        ]),
        # An employee account: tellers may not view it.
        "77777": _member("EMPLOYEE ACCOUNT", "900-00-7777", "1980-01-01", [
            {"id": "S00", "type": "SAVINGS", "desc": "PRIMARY SAVINGS", "balance": 5000.00},
        ], restricted=True),
    }


SHARE_TYPES = {  # code: (description, minimum opening deposit)
    "SC": ("SAVINGS CLUB", 5.00),
    "CD": ("SHARE CERTIFICATE", 500.00),
    "MM": ("MONEY MARKET", 2500.00),
}


@dataclass
class Tenant:
    code: str
    institution: str
    colour: str
    member_label: str
    inquire_button: str
    balance_header: str
    columns: tuple[str, ...]           # order of the share table columns


TENANTS = {
    "a": Tenant("a", "FIRST COMMUNITY FCU", "#003366", "Member Number:", "Inquire",
                "Balance", ("id", "type", "desc", "balance")),
    "b": Tenant("b", "LAKESHORE CREDIT UNION", "#5a2d0c", "Acct Holder No.:", "Search",
                "Avail Bal", ("desc", "id", "balance", "type")),
}


@dataclass
class Faults:
    """Injected runtime conditions. Each is consumed or applies until cleared."""

    slow_ms: int = 0                  # delay every response by this much
    notice_once: bool = False         # next inquiry shows the system notice
    error_once: bool = False          # next inquiry returns a system error
    expire_session: bool = False      # current sessions are treated as expired

    def update(self, raw: dict) -> None:
        for key, value in raw.items():
            if hasattr(self, key):
                setattr(self, key, type(getattr(self, key))(value))


@dataclass
class State:
    tenant: Tenant
    members: dict[str, dict] = field(default_factory=seed_members)
    sessions: dict[str, float] = field(default_factory=dict)   # token -> last seen
    session_timeout_s: float = 600.0
    faults: Faults = field(default_factory=Faults)
    faults_enabled: bool = False
    next_confirmation: int = 48210
    lock: threading.Lock = field(default_factory=threading.Lock)


# --- rendering ----------------------------------------------------------------

e = html.escape


def money(amount: float) -> str:
    return f"${amount:,.2f}"


def page(state: State, title: str, body: str) -> str:
    # No doctype, no lang, no landmarks, presentational markup. On purpose.
    return (
        f"<html><head><title>{e(title)}</title></head>"
        f"<body bgcolor='#e8e8e0'><font face='Courier New' size='2'>{body}</font></body></html>"
    )


def panel(heading: str, inner: str, colour: str = "#003366") -> str:
    return (
        "<table width='100%' border='0' cellpadding='2' cellspacing='0'>"
        f"<tr><td bgcolor='{colour}'><font color='white'><b>{e(heading)}</b></font></td></tr>"
        f"<tr><td>{inner}</td></tr></table>"
    )


def message_box(text: str, colour: str = "#aa0000") -> str:
    return (
        "<table border='1' cellpadding='4' cellspacing='0'><tr>"
        f"<td bgcolor='#ffffcc'><font color='{colour}'><b>{e(text)}</b></font></td>"
        "</tr></table><br>"
    )


# --- request handling -----------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    state: State  # set on the subclass made by make_server()

    server_version = "CoreLink/7.4"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    # -- plumbing --

    def _session(self) -> str | None:
        jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        token = jar["CLSESSION"].value if "CLSESSION" in jar else None
        return token if token in self.state.sessions else None

    def _send(self, body: str, status: int = 200, headers: dict | None = None) -> None:
        if self.state.faults.slow_ms and not self.path.startswith("/__control"):
            time.sleep(self.state.faults.slow_ms / 1000)
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str, headers: dict | None = None) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode() if length else ""
        return {k: v[0] for k, v in parse_qs(raw, keep_blank_values=True).items()}

    def _query(self) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    def _require_session(self) -> str | None:
        """The session token, or None after rendering the expiry page."""
        token = self._session()
        state = self.state
        if token and not state.faults.expire_session:
            last = state.sessions[token]
            if time.time() - last <= state.session_timeout_s:
                state.sessions[token] = time.time()
                return token
        if token:
            state.sessions.pop(token, None)
        # The legacy behaviour: a page inside the frame that says so, with a
        # link that breaks out to the top-level sign-on.
        self._send(page(state, "Session Expired", panel(
            "CORELINK TELLER - SESSION",
            message_box("SESSION EXPIRED. PLEASE SIGN ON AGAIN. (CL-0401)")
            + "<a href='/login' target='_top'>Sign On</a>",
            state.tenant.colour,
        )))
        return None

    # -- routes --

    def do_GET(self):  # noqa: N802 — http.server naming
        path = urlparse(self.path).path
        routes = {
            "/": self.get_root, "/login": self.get_login, "/main": self.get_main,
            "/banner": self.get_banner, "/nav": self.get_nav, "/home": self.get_home,
            "/inquiry": self.get_inquiry, "/member": self.get_member,
            "/share/new": self.get_share_new, "/logout": self.get_logout,
            "/__control/faults": self.get_faults,
        }
        handler = routes.get(path)
        if handler is None:
            self._send(page(self.state, "Not Found", message_box("CL-0404 FUNCTION NOT AVAILABLE")), 404)
            return
        handler()

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        routes = {
            "/login": self.post_login, "/inquiry": self.post_inquiry,
            "/notice": self.post_notice, "/share/review": self.post_share_review,
            "/share/post": self.post_share_post, "/__control/faults": self.post_faults,
        }
        handler = routes.get(path)
        if handler is None:
            self._send(page(self.state, "Not Found", message_box("CL-0404 FUNCTION NOT AVAILABLE")), 404)
            return
        handler()

    def get_root(self):
        self._redirect("/main" if self._session() else "/login")

    def get_login(self, error: str = ""):
        t = self.state.tenant
        body = panel(f"{t.institution} - CORELINK TELLER SIGN ON", (
            (message_box(error) if error else "")
            + "<form method='post' action='/login'><table cellpadding='3'>"
            "<tr><td>Operator ID</td><td><input type='text' name='f_0001' size='12'></td></tr>"
            "<tr><td>Password</td><td><input type='password' name='f_0002' size='12'></td></tr>"
            "<tr><td></td><td><input type='submit' value='Sign On'></td></tr>"
            "</table></form>"
        ), t.colour)
        self._send(page(self.state, "CoreLink Teller", body))

    def post_login(self):
        form = self._form()
        if form.get("f_0001") != OPERATOR_USER or form.get("f_0002") != OPERATOR_PASSWORD:
            self.get_login("INVALID OPERATOR ID OR PASSWORD (CL-0101)")
            return
        token = secrets.token_hex(12)
        with self.state.lock:
            self.state.sessions[token] = time.time()
            self.state.faults.expire_session = False
        self._redirect("/main", {"Set-Cookie": f"CLSESSION={token}; Path=/; HttpOnly"})

    def get_main(self):
        if not self._session():
            self._redirect("/login")
            return
        self._send(
            "<html><head><title>CoreLink Teller</title></head>"
            "<frameset rows='48,*' border='1'>"
            "<frame name='banner' src='/banner' scrolling='no'>"
            "<frameset cols='170,*'>"
            "<frame name='nav' src='/nav'>"
            "<frame name='main' src='/home'>"
            "</frameset></frameset></html>"
        )

    def get_banner(self):
        t = self.state.tenant
        self._send(page(self.state, "banner", (
            f"<table width='100%' bgcolor='{t.colour}'><tr>"
            f"<td><font color='white' size='4'><b>{e(t.institution)}</b></font></td>"
            "<td align='right'><font color='white'>CoreLink Teller 7.4 | Operator TELLER01</font></td>"
            "</tr></table>"
        )))

    def get_nav(self):
        # Plain links; target=main so they load in the content frame.
        links = [("/inquiry", "Member Inquiry"), ("/home", "Teller Home"),
                 ("/logout", "Sign Off")]
        if self.state.tenant.code == "b":
            links.reverse()
        items = "".join(
            f"<tr><td><a href='{href}' target='{'_top' if href == '/logout' else 'main'}'>{e(label)}</a></td></tr>"
            for href, label in links
        )
        self._send(page(self.state, "nav", f"<table cellpadding='4'>{items}</table>"))

    def get_home(self):
        if not self._require_session():
            return
        self._send(page(self.state, "home", panel(
            "TELLER HOME", "Select a function from the menu.", self.state.tenant.colour)))

    def get_logout(self):
        token = self._session()
        if token:
            self.state.sessions.pop(token, None)
        self._redirect("/login", {"Set-Cookie": "CLSESSION=; Path=/; Max-Age=0"})

    # -- member inquiry --

    def get_inquiry(self, error: str = "", value: str = ""):
        if not self._require_session():
            return
        t = self.state.tenant
        body = panel("MEMBER INQUIRY", (
            (message_box(error) if error else "")
            + "<form method='post' action='/inquiry'><table cellpadding='3'>"
            f"<tr><td>{e(t.member_label)}</td>"
            f"<td><input type='text' name='f_0012' size='10' value='{e(value)}'></td></tr>"
            f"<tr><td></td><td><input type='submit' value='{e(t.inquire_button)}'></td></tr>"
            "</table></form>"
        ), t.colour)
        self._send(page(self.state, "inquiry", body))

    def post_inquiry(self):
        if not self._require_session():
            return
        faults, state = self.state.faults, self.state
        number = self._form().get("f_0012", "").strip()

        if faults.error_once:
            faults.error_once = False
            self._send(page(state, "System Error", panel("CORELINK TELLER", message_box(
                "SYSTEM ERROR 0x5E1 - HOST TRANSACTION ABORTED. CONTACT SUPPORT."), state.tenant.colour)), 500)
            return

        if faults.notice_once:
            faults.notice_once = False
            self._send(page(state, "System Notice", panel("SYSTEM NOTICE", (
                "<p>Scheduled maintenance tonight 23:00-01:00. Teller functions will be unavailable.</p>"
                "<form method='post' action='/notice'>"
                f"<input type='hidden' name='f_0012' value='{e(number)}'>"
                "<input type='submit' value='Acknowledge'></form>"
            ), "#886600")))
            return

        self._inquiry_result(number)

    def post_notice(self):
        if not self._require_session():
            return
        self._inquiry_result(self._form().get("f_0012", "").strip())

    def _inquiry_result(self, number: str):
        if not number.isdigit() or not 4 <= len(number) <= 9:
            self.get_inquiry("INVALID MEMBER NUMBER FORMAT (CL-0210)", number)
            return
        member = self.state.members.get(number)
        if member is None:
            self.get_inquiry(f"NO RECORD FOUND FOR MEMBER {number} (CL-0204)", number)
            return
        if member["restricted"]:
            self._send(page(self.state, "Access Denied", panel("MEMBER INQUIRY", message_box(
                "ACCESS DENIED - RESTRICTED ACCOUNT. SUPERVISOR OVERRIDE REQUIRED (CL-0403)"),
                self.state.tenant.colour)), 403)
            return
        self._redirect(f"/member?m={number}")

    def get_member(self):
        if not self._require_session():
            return
        number = self._query().get("m", "")
        member = self.state.members.get(number)
        if member is None or member["restricted"]:
            self._send(page(self.state, "Error", message_box("CL-0204 RECORD NOT AVAILABLE")), 404)
            return
        t = self.state.tenant
        headers = {"id": "Share", "type": "Type", "desc": "Description", "balance": t.balance_header}
        head = "".join(f"<td bgcolor='#cccccc'><b>{headers[c]}</b></td>" for c in t.columns)
        rows = ""
        for share in member["shares"]:
            cells = {"id": share["id"], "type": share["type"], "desc": share["desc"],
                     "balance": money(share["balance"])}
            rows += "<tr>" + "".join(
                f"<td{' align=right' if c == 'balance' else ''}>{e(cells[c])}</td>" for c in t.columns
            ) + "</tr>"
        body = panel("MEMBER INQUIRY - DETAIL", (
            "<table cellpadding='2'>"
            f"<tr><td>Member</td><td><b>{e(number)}</b></td></tr>"
            f"<tr><td>Name</td><td>{e(member['name'])}</td></tr>"
            f"<tr><td>SSN</td><td>{e(member['ssn'])}</td></tr>"
            f"<tr><td>Birth Date</td><td>{e(member['dob'])}</td></tr>"
            "</table><br>"
            f"<table border='1' cellspacing='0' cellpadding='3'><tr>{head}</tr>{rows}</table><br>"
            f"<a href='/share/new?m={e(number)}'>Open Share</a> &nbsp; "
            "<a href='/inquiry'>New Inquiry</a>"
        ), t.colour)
        self._send(page(self.state, "member detail", body))

    # -- open a share (sub-account) --

    def get_share_new(self, error: str = "", number: str | None = None, form: dict | None = None):
        if not self._require_session():
            return
        number = number if number is not None else self._query().get("m", "")
        if number not in self.state.members:
            self._send(page(self.state, "Error", message_box("CL-0204 RECORD NOT AVAILABLE")), 404)
            return
        form = form or {}
        options = "".join(
            f"<option value='{code}'{' selected' if form.get('f_0031') == code else ''}>{e(desc)}</option>"
            for code, (desc, _) in SHARE_TYPES.items()
        )
        body = panel("SHARE MAINTENANCE - OPEN SHARE", (
            (message_box(error) if error else "")
            + "<form method='post' action='/share/review'>"
            f"<input type='hidden' name='m' value='{e(number)}'>"
            "<table cellpadding='3'>"
            f"<tr><td>Member</td><td>{e(number)}</td></tr>"
            f"<tr><td>Share Type</td><td><select name='f_0031'><option value=''>--</option>{options}</select></td></tr>"
            f"<tr><td>Nickname</td><td><input type='text' name='f_0032' value='{e(form.get('f_0032', ''))}'></td></tr>"
            f"<tr><td>Opening Deposit</td><td><input type='text' name='f_0033' value='{e(form.get('f_0033', ''))}'></td></tr>"
            "<tr><td></td><td><input type='submit' value='Continue'></td></tr>"
            "</table></form>"
        ), self.state.tenant.colour)
        self._send(page(self.state, "open share", body))

    def _validate_share(self, form: dict) -> str:
        code = form.get("f_0031", "")
        if code not in SHARE_TYPES:
            return "SHARE TYPE IS REQUIRED (CL-0311)"
        try:
            deposit = float(form.get("f_0033", "").replace("$", "").replace(",", ""))
        except ValueError:
            return "OPENING DEPOSIT MUST BE NUMERIC (CL-0312)"
        desc, minimum = SHARE_TYPES[code]
        if deposit < minimum:
            return f"MINIMUM OPENING DEPOSIT FOR {desc} IS {money(minimum)} (CL-0313)"
        return ""

    def post_share_review(self):
        if not self._require_session():
            return
        form = self._form()
        number = form.get("m", "")
        error = self._validate_share(form)
        if error:
            self.get_share_new(error, number, form)
            return
        desc, _ = SHARE_TYPES[form["f_0031"]]
        deposit = float(form["f_0033"].replace("$", "").replace(",", ""))
        hidden = "".join(
            f"<input type='hidden' name='{k}' value='{e(v)}'>" for k, v in form.items()
        )
        body = panel("SHARE MAINTENANCE - REVIEW", (
            "<table cellpadding='3'>"
            f"<tr><td>Member</td><td>{e(number)}</td></tr>"
            f"<tr><td>Share Type</td><td>{e(desc)}</td></tr>"
            f"<tr><td>Nickname</td><td>{e(form.get('f_0032', ''))}</td></tr>"
            f"<tr><td>Opening Deposit</td><td>{money(deposit)}</td></tr>"
            "</table><p>Review the details. Posting opens the share and cannot be undone.</p>"
            f"<form method='post' action='/share/post'>{hidden}"
            "<input type='submit' value='Post'></form>"
            f"<a href='/member?m={e(number)}'>Cancel</a>"
        ), self.state.tenant.colour)
        self._send(page(self.state, "review share", body))

    def post_share_post(self):
        if not self._require_session():
            return
        form = self._form()
        number = form.get("m", "")
        member = self.state.members.get(number)
        if member is None or self._validate_share(form):
            self._send(page(self.state, "Error", message_box("CL-0500 POSTING REJECTED")), 400)
            return
        with self.state.lock:
            existing = {s["id"] for s in member["shares"]}
            share_id = next(f"S{n:02d}" for n in range(1, 100) if f"S{n:02d}" not in existing)
            desc, _ = SHARE_TYPES[form["f_0031"]]
            deposit = float(form["f_0033"].replace("$", "").replace(",", ""))
            member["shares"].append({"id": share_id, "type": "SAVINGS", "desc": desc, "balance": deposit})
            self.state.next_confirmation += 1
            confirmation = self.state.next_confirmation
        body = panel("SHARE MAINTENANCE - CONFIRMATION", (
            message_box(f"SHARE {share_id} OPENED. CONFIRMATION {confirmation}", "#006600")
            + f"<a href='/member?m={e(number)}'>Return to Member</a>"
        ), self.state.tenant.colour)
        self._send(page(self.state, "share opened", body))

    # -- fault injection (test harness only) --

    def get_faults(self):
        if not self.state.faults_enabled:
            self._send("disabled", 404)
            return
        self._send(json.dumps(self.state.faults.__dict__), headers={"Content-Type": "application/json"})

    def post_faults(self):
        if not self.state.faults_enabled:
            self._send("disabled", 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = json.loads(self.rfile.read(length) or b"{}")
        if raw.pop("reset", False):
            self.state.faults = Faults()
            self.state.members = seed_members()
        self.state.faults.update(raw)
        self._send(json.dumps(self.state.faults.__dict__))


def make_server(port: int = 8765, tenant: str = "a", faults: bool = False,
                session_timeout_s: float = 600.0, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    state = State(tenant=TENANTS[tenant], faults_enabled=faults, session_timeout_s=session_timeout_s)
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve_in_thread(**kwargs) -> tuple[ThreadingHTTPServer, str]:
    """Start the app on a background thread. Returns (server, base_url)."""
    server = make_server(**kwargs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tenant", choices=sorted(TENANTS), default="a")
    parser.add_argument("--faults", action="store_true", help="enable /__control/faults")
    parser.add_argument("--session-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    server = make_server(args.port, args.tenant, args.faults, args.session_timeout)
    print(f"CoreLink Teller (tenant {args.tenant}) on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

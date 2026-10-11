"""A minimal operator console: the human side of a handoff, over HTTP.

Deliberately thin — a mock of the real co-browsing console — but the control
transfer behind it is the real one: the buttons call SessionControl.take()
and .resolve() on the live run, and the page shows how to reach the *same*
browser session:

  * headed run   the browser window on the machine is the session
  * headless run attach over CDP: open chrome://inspect, add the endpoint
                 under "Configure...", then "inspect" the page — DevTools'
                 screencast is interactive, so the operator can click and type
                 into the live session from another machine.

    GET  /            the console page (polls /api/state)
    GET  /api/state   owner, open intervention, resolved history
    GET  /shot/<id>   the intervention's (masked) screenshot
    POST /take        operator takes control
    POST /resolve     {"action": "...", "note": "..."} hands control back
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .evidence import RUNS
from .handoff import Intervention, InteractiveEscalator, Owner, Resolution, SessionControl

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Rote operator console</title>
<style>
 body{font:14px system-ui,sans-serif;margin:24px;max-width:1000px;color:#1b1b1b;background:#f6f6f3}
 h1{font-size:18px} .owner{display:inline-block;padding:2px 8px;border-radius:4px;background:#ddd}
 .owner.human{background:#ffd27a}.owner.awaiting_human{background:#ff9d9d}.owner.automation{background:#b8e0b8}
 .card{background:#fff;border:1px solid #ccc;border-radius:6px;padding:16px;margin:16px 0}
 img{max-width:100%;border:1px solid #999} code{background:#eee;padding:1px 4px}
 button{font-size:14px;margin-right:8px;padding:6px 14px} textarea{width:100%;height:48px}
 table{border-collapse:collapse} td{padding:2px 8px;vertical-align:top}
</style></head><body>
<h1>Rote operator console</h1>
<p>Session owner: <span id="owner" class="owner">…</span></p>
<div id="open"></div><h2>Resolved</h2><div id="history"></div>
<script>
async function post(path, body){await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});refresh();}
function esc(s){const d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML;}
async function refresh(){
  const s=await (await fetch('/api/state')).json();
  const o=document.getElementById('owner'); o.textContent=s.owner; o.className='owner '+s.owner;
  const i=s.open, el=document.getElementById('open');
  if(!i){el.innerHTML='<div class="card">No open intervention. Automation is in control.</div>';}
  else{
    const opts=i.options.map(a=>`<button onclick="post('/resolve',{action:'${a}',note:document.getElementById('note').value})" ${s.owner!=='human'?'disabled':''}>${a}</button>`).join('');
    el.innerHTML=`<div class="card"><h2>${esc(i.reason)} — ${esc(i.capability||i.goal||'')}</h2>
     <p>${esc(i.message)}</p>
     <table><tr><td>step</td><td>${esc(i.step_id||'-')} ${esc(i.step_description)}</td></tr>
     <tr><td>where</td><td>${esc(Object.entries(i.location).map(([k,v])=>k+'='+v).join('  '))}</td></tr>
     <tr><td>run</td><td>${esc(i.run_id)}</td></tr>
     <tr><td>live session</td><td>${i.cdp_endpoint?`attach over CDP at <code>${esc(i.cdp_endpoint)}</code> (chrome://inspect → Configure)`:'the headed browser window on the automation host'}</td></tr></table>
     ${i.screenshot?`<p><img src="/shot/${i.id}"></p>`:''}
     ${s.owner==='awaiting_human'?`<button onclick="post('/take')">Take control</button>`:
       `<p><b>You have control.</b> Work in the live session; your clicks are recorded. Then resolve:</p>
        <textarea id="note" placeholder="note for the record"></textarea><p>${opts}</p>`}
    </div>`;}
  document.getElementById('history').innerHTML=s.history.map(h=>`<div class="card">${esc(h.reason)} at ${esc(h.step_id||'-')} → <b>${esc(h.action)}</b> by ${esc(h.by)}; ${h.human_events} human action(s). ${esc(h.note)}</div>`).join('')||'<p>none</p>';
}
refresh(); setInterval(()=>{ if(!document.activeElement||document.activeElement.id!=='note') refresh(); },1500);
</script></body></html>"""


class OperatorConsole:
    """Serves the console for one live session."""

    def __init__(self, session: SessionControl, port: int = 8770, host: str = "127.0.0.1") -> None:
        self.session = session
        self.history: list[dict] = []
        self._open: Intervention | None = None
        console = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body: bytes, ctype: str = "application/json", status: int = 200):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path == "/":
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/api/state":
                    self._send(json.dumps(console.state()).encode())
                elif self.path.startswith("/shot/"):
                    shot = console.screenshot_path(self.path.rsplit("/", 1)[-1])
                    if shot and shot.exists():
                        self._send(shot.read_bytes(), "image/png")
                    else:
                        self._send(b"{}", status=404)
                else:
                    self._send(b"{}", status=404)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                try:
                    if self.path == "/take":
                        console.session.take("console-operator")
                    elif self.path == "/resolve":
                        console.resolve(body.get("action", ""), body.get("note", ""))
                    else:
                        self._send(b"{}", status=404)
                        return
                except (RuntimeError, ValueError) as e:
                    self._send(json.dumps({"error": str(e)}).encode(), status=409)
                    return
                self._send(json.dumps(console.state()).encode())

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://{host}:{self.server.server_address[1]}"

    def start(self) -> "OperatorConsole":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.server.shutdown()

    def publish(self, intervention: Intervention) -> None:
        self._open = intervention

    def resolve(self, action: str, note: str) -> None:
        intervention = self.session.current
        if self.session.owner != Owner.human:
            raise RuntimeError("take control before resolving")
        events = len(self.session.human_events)
        # Stored as typed; the page escapes when it renders.
        self.session.resolve(Resolution(action=action, by="console-operator", note=note[:500]))  # type: ignore[arg-type]
        if intervention:
            self.history.append({"reason": intervention.reason, "step_id": intervention.step_id,
                                 "action": action, "by": "console-operator", "note": note,
                                 "human_events": events})

    def screenshot_path(self, intervention_id: str) -> Path | None:
        current = self.session.current
        if current and current.id == intervention_id and current.screenshot:
            return RUNS.parent / current.screenshot
        return None

    def state(self) -> dict:
        current = self.session.current
        return {
            "owner": self.session.owner.value,
            "open": current.to_dict() if current else None,
            "human_events": len(self.session.human_events),
            "history": self.history,
        }


class ConsoleEscalator(InteractiveEscalator):
    """Routes interventions to the operator console and waits (pumping)."""

    def __init__(self, console: OperatorConsole, pump=None, timeout_s: float = 900,
                 notify=print) -> None:
        super().__init__(pump, timeout_s)
        self.console = console
        self.notify = notify

    def present(self, intervention: Intervention, session: SessionControl) -> None:
        self.console.publish(intervention)
        self.notify(f"  ⚑ intervention {intervention.id} ({intervention.reason}): {intervention.message}\n"
                    f"    open the operator console: {self.console.url}")

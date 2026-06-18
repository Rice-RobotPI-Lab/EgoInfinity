"""Lightweight HTTP server for real-time visual filter review."""
import json
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

_VIZ_DIR: Path | None = None
_ITEMS: list[dict] = []  # [{id, filename, video_uid, status, reason, metrics, desc}]
_LOCK = threading.Lock()


def init(viz_dir: str):
    global _VIZ_DIR
    _VIZ_DIR = Path(viz_dir)
    _VIZ_DIR.mkdir(parents=True, exist_ok=True)
    (_VIZ_DIR / "imgs").mkdir(exist_ok=True)


def add_item(item: dict):
    with _LOCK:
        _ITEMS.append(item)


class _Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence logs

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/items":
            self._serve_json()
        elif self.path.startswith("/imgs/"):
            self._serve_file()
        else:
            self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(_HTML.encode())

    def _serve_json(self):
        with _LOCK:
            data = list(_ITEMS)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _serve_file(self):
        fpath = _VIZ_DIR / self.path.lstrip("/")
        if fpath.exists():
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(fpath.read_bytes())
        else:
            self.send_error(404)


def start_server(port: int = 8899):
    server = HTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


_HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Visual Filter Review</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; color:#eee; font-family:system-ui,sans-serif; padding:16px; }
  h1 { font-size:18px; margin-bottom:8px; color:#888; }
  #stats { margin-bottom:12px; font-size:14px; color:#aaa; }
  #filters { margin-bottom:12px; display:flex; gap:8px; }
  #filters button { padding:4px 12px; border:1px solid #555; background:#222;
    color:#ccc; border-radius:4px; cursor:pointer; font-size:13px; }
  #filters button.active { background:#446; border-color:#88f; color:#fff; }

  /* Video folder card */
  .video-card { background:#1a1a1a; border-radius:8px; margin-bottom:10px;
    border:1px solid #333; overflow:hidden; }
  .video-header { padding:10px 14px; cursor:pointer; display:flex;
    justify-content:space-between; align-items:center; user-select:none; gap:12px; }
  .video-header:hover { background:#222; }
  .video-thumb { flex-shrink:0; display:flex; gap:3px; }
  .video-thumb img { height:48px; border-radius:3px; opacity:0.85; }
  .video-header-info { flex:1; min-width:0; }
  .video-title { font-size:14px; font-weight:600; }
  .video-stats { display:flex; gap:14px; font-size:12px; color:#aaa; flex-wrap:wrap; }
  .video-stats .chip { padding:2px 8px; border-radius:3px; font-size:11px; }
  .chip-accept { background:#1a3a1a; color:#4f4; }
  .chip-reject { background:#3a1a1a; color:#f66; }
  .chip-metric { background:#1a2a3a; color:#8cf; }
  .chip-cut { background:#3a3a1a; color:#fc0; }
  .video-arrow { font-size:16px; color:#666; transition:transform 0.2s; }
  .video-card.open .video-arrow { transform:rotate(90deg); }

  /* Segments inside a video folder */
  .video-segments { display:none; padding:0 8px 8px 8px; }
  .video-card.open .video-segments { display:block; }
  .seg-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(700px, 1fr)); gap:8px; }
  .card { background:#222; border-radius:6px; overflow:hidden; border:2px solid transparent; }
  .card.accept { border-color:#2a2; }
  .card.reject { border-color:#a22; }
  .card img { width:100%; display:block; }
  .card-info { padding:6px 10px; font-size:12px; }
  .card-info .status { font-weight:bold; font-size:13px; }
  .card-info .status.accept { color:#4f4; }
  .card-info .status.reject { color:#f44; }
  .card-info .metrics { color:#aaa; margin-top:2px; }
  .card-info .desc { color:#777; margin-top:2px; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap; }
</style>
</head><body>
<h1>Visual Filter Review</h1>
<div id="stats">Loading...</div>
<div id="filters">
  <button class="active" data-filter="all">All</button>
  <button data-filter="accept">Accept</button>
  <button data-filter="reject">Reject</button>
</div>
<div id="videos"></div>
<script>
let items = [], filter = 'all', openVideos = new Set();

document.querySelectorAll('#filters button').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('#filters button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    filter = btn.dataset.filter;
    render();
  };
});

function parseMetrics(metricsStr) {
  const m = {};
  const flowMatch = metricsStr.match(/flow=([\d.]+)/);
  const sizeMatch = metricsStr.match(/size=([\d.]+)/);
  const handMatch = metricsStr.match(/hand=([\d]+)%/);
  if (flowMatch) m.flow = parseFloat(flowMatch[1]);
  if (sizeMatch) m.size = parseFloat(sizeMatch[1]);
  if (handMatch) m.hand = parseInt(handMatch[1]);
  return m;
}

function render() {
  // Group by video_uid
  const groups = {};
  for (const it of items) {
    if (!groups[it.video_uid]) groups[it.video_uid] = [];
    groups[it.video_uid].push(it);
  }

  const totalAccept = items.filter(i => i.status === 'accept').length;
  document.getElementById('stats').textContent =
    `${Object.keys(groups).length} videos, ${items.length} segments | ${totalAccept} accept | ${items.length - totalAccept} reject`;

  let html = '';
  for (const [uid, segs] of Object.entries(groups)) {
    const accept = segs.filter(s => s.status === 'accept').length;
    const reject = segs.length - accept;

    // Filter: skip video if no matching segments
    const filtered = segs.filter(s =>
      filter === 'all' || s.status === filter
    );
    if (filtered.length === 0) continue;

    // Compute averages
    let totalFlow = 0, totalSize = 0, flowCount = 0, sizeCount = 0;
    let cutSegs = 0;
    for (const s of segs) {
      const m = parseMetrics(s.metrics);
      if (m.flow !== undefined && m.flow < 900) { totalFlow += m.flow; flowCount++; }
      if (m.size !== undefined) { totalSize += m.size; sizeCount++; }
      if ((s.n_cuts || 0) > 0) cutSegs++;
    }
    const avgFlow = flowCount > 0 ? (totalFlow / flowCount).toFixed(3) : '?';
    const avgSize = sizeCount > 0 ? (totalSize / sizeCount * 100).toFixed(1) + '%' : '?';
    const isOpen = openVideos.has(uid);

    // Pick up to 4 thumbnails from different segments
    const thumbSegs = segs.slice(0, 4);

    html += `
    <div class="video-card ${isOpen ? 'open' : ''}" data-uid="${uid}">
      <div class="video-header" onclick="toggleVideo('${uid}')">
        <div class="video-thumb">
          ${thumbSegs.map(s => `<img src="/imgs/${s.filename}" loading="lazy" />`).join('')}
        </div>
        <div class="video-header-info">
          <span class="video-title">${uid}</span>
          <span class="video-stats">
            <span class="chip chip-accept">${accept} accept</span>
            <span class="chip chip-reject">${reject} reject</span>
            <span class="chip chip-metric">flow ${avgFlow}</span>
            <span class="chip chip-metric">size ${avgSize}</span>
            ${cutSegs > 0 ? `<span class="chip chip-cut">${cutSegs} cut</span>` : ''}
            <span style="color:#888">${segs.length} segs</span>
          </span>
        </div>
        <span class="video-arrow">&#9654;</span>
      </div>
      <div class="video-segments">
        <div class="seg-grid">
          ${filtered.map(it => `
            <div class="card ${it.status}">
              <img src="/imgs/${it.filename}" loading="lazy" />
              <div class="card-info">
                <div class="status ${it.status}">${it.status.toUpperCase()}${it.reason ? ': ' + it.reason : ''}</div>
                <div class="metrics">${it.metrics}</div>
                <div class="desc">${it.desc}</div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    </div>`;
  }
  document.getElementById('videos').innerHTML = html;
}

function toggleVideo(uid) {
  if (openVideos.has(uid)) openVideos.delete(uid);
  else openVideos.add(uid);
  render();
}

let prevLen = 0;
async function poll() {
  try {
    const r = await fetch('/api/items');
    items = await r.json();
    if (items.length !== prevLen) { prevLen = items.length; render(); }
  } catch {}
  setTimeout(poll, 1000);
}
poll();
</script>
</body></html>"""

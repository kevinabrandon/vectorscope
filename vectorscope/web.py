"""Minimal web-based oscilloscope viewer for vectorscope.

Zero external dependencies — uses only Python stdlib (hashlib, struct,
threading, socket).  Serves an embedded HTML/JS page over HTTP and streams
binary XY data over WebSocket (RFC 6455 server-to-client framing only), all
on a single port.
"""

import base64
import hashlib
import json
import socket
import struct
import threading
import time

# ---------------------------------------------------------------------------
# WebSocket helpers (RFC 6455, server-to-client only)
# ---------------------------------------------------------------------------

_WS_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_accept_key(client_key):
    """Compute Sec-WebSocket-Accept from client's Sec-WebSocket-Key."""
    digest = hashlib.sha1(client_key.encode() + _WS_MAGIC).digest()
    return base64.b64encode(digest).decode()


def _ws_frame(payload, opcode=0x02):
    """Build a server-to-client WebSocket frame (no masking)."""
    header = bytes([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header += bytes([length])
    elif length < 65536:
        header += struct.pack("!BH", 126, length)
    else:
        header += struct.pack("!BQ", 127, length)
    return header + payload


def _recv_until(sock, sentinel):
    """Read from sock until sentinel is found. Return all bytes read."""
    buf = b""
    while sentinel not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("connection closed during HTTP read")
        buf += chunk
    return buf


# ---------------------------------------------------------------------------
# VectorscopeWebServer
# ---------------------------------------------------------------------------

class VectorscopeWebServer:
    """Serves a browser-based oscilloscope viewer and streams XY data.

    Uses raw sockets — no http.server / BaseHTTPRequestHandler — to avoid
    framework-level socket lifecycle issues with long-lived WebSocket
    connections.
    """

    def __init__(self, port=8080):
        self._port = port
        self._frame_data = bytearray() # accumulated binary data
        self._metadata = None        # latest JSON metadata string
        self._metadata_sent = True   # tracks whether latest metadata was sent
        self._clients = []           # list of WebSocket sockets
        self._lock = threading.Lock()
        self._listen_sock = None
        self._running = False

    def start(self):
        """Launch accept loop + push loop in daemon threads."""
        try:
            self._listen_sock = socket.socket(socket.AF_INET,
                                              socket.SOCK_STREAM)
            self._listen_sock.setsockopt(socket.SOL_SOCKET,
                                         socket.SO_REUSEADDR, 1)
            self._listen_sock.bind(("", self._port))
            self._listen_sock.listen(5)
        except OSError as e:
            print(f"Web server error: {e}")
            print(f"(port {self._port} may be in use)")
            return

        self._running = True

        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._push_loop, daemon=True).start()

        print(f"Web viewer: http://localhost:{self._port}/")

    def stop(self):
        """Shutdown server and close client sockets."""
        self._running = False
        if self._listen_sock:
            try:
                self._listen_sock.close()
            except OSError:
                pass
        with self._lock:
            for sock in self._clients:
                try:
                    sock.close()
                except OSError:
                    pass
            self._clients.clear()

    # ------------------------------------------------------------------
    # Accept loop — one thread per connection
    # ------------------------------------------------------------------

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._listen_sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle_conn,
                             args=(conn, addr), daemon=True).start()

    def _handle_conn(self, conn, addr):
        try:
            raw = _recv_until(conn, b"\r\n\r\n")
            header_end = raw.index(b"\r\n\r\n")
            header_text = raw[:header_end].decode("utf-8", errors="replace")
            lines = header_text.split("\r\n")

            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            if headers.get("upgrade", "").lower() == "websocket":
                self._handle_ws(conn, addr, headers)
            else:
                self._serve_html(conn)
        except (ConnectionError, OSError):
            pass
        except Exception as e:
            print(f"[web] handler error: {e}")
        finally:
            # For HTTP connections we close; for WS, socket is already
            # removed from _clients and we close here too.
            try:
                conn.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # HTTP — serve the embedded HTML page
    # ------------------------------------------------------------------

    def _serve_html(self, conn):
        body = _HTML_PAGE.encode("utf-8")
        response = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode() + body
        conn.sendall(response)

    # ------------------------------------------------------------------
    # WebSocket — handshake + hold connection open
    # ------------------------------------------------------------------

    def _handle_ws(self, conn, addr, headers):
        client_key = headers.get("sec-websocket-key", "").strip()
        accept = _ws_accept_key(client_key)

        print(f"[web] WS connect from {addr[0]}")

        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(2.0) # Prevent slow clients from hanging the push loop

        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode()
        conn.sendall(response)

        self._add_client(conn)

        # Block reading to detect disconnect
        try:
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                    if len(data) >= 2 and (data[0] & 0x0F) == 0x08:
                        break
                except socket.timeout:
                    # Timeout is fine, it just means the client hasn't sent 
                    # anything (or a close frame). Keep the connection alive.
                    continue
        except (ConnectionError, OSError):
            pass
        finally:
            self._remove_client(conn)
            print(f"[web] WS disconnect from {addr[0]}")

    # ------------------------------------------------------------------
    # Client registry
    # ------------------------------------------------------------------

    def push_frame(self, data):
        """Called from audio thread — accumulate frame as bytes."""
        with self._lock:
            # Limit buffer to ~1 second of 192kHz audio (1.5MB)
            if len(self._frame_data) < 2000000:
                self._frame_data.extend(data.tobytes())

    def push_metadata(self, meta):
        """Called when command/params change — store JSON metadata."""
        # Include current hardware-specific settings for web normalization
        if hasattr(self, '_z_amp'):
            meta['z_amp'] = self._z_amp
        if hasattr(self, '_web_scale_factor'):
            meta['scale_factor'] = self._web_scale_factor
        self._metadata = json.dumps(meta)
        self._metadata_sent = False

    def set_z_amp(self, z_amp):
        """Allow player to notify web server of intensity scaling."""
        self._z_amp = z_amp
        self._metadata_sent = False

    def set_web_scale_factor(self, factor):
        """Allow player to notify web server of scaling factor."""
        self._web_scale_factor = factor
        self._metadata_sent = False

    def _add_client(self, sock):
        with self._lock:
            self._clients.append(sock)
            self._metadata_sent = False

    def _remove_client(self, sock):
        with self._lock:
            try:
                self._clients.remove(sock)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Push loop — ~60 fps broadcast to all clients
    # ------------------------------------------------------------------

    def _push_loop(self):
        interval = 1.0 / 60
        while self._running:
            start = time.monotonic()

            # Send metadata if changed
            if not self._metadata_sent and self._metadata is not None:
                frame = _ws_frame(self._metadata.encode("utf-8"), opcode=0x01)
                self._broadcast(frame)
                self._metadata_sent = True

            # Send accumulated audio data
            data_to_send = None
            with self._lock:
                if self._frame_data:
                    data_to_send = bytes(self._frame_data)
                    self._frame_data.clear()

            if data_to_send:
                frame = _ws_frame(data_to_send, opcode=0x02)
                self._broadcast(frame)

            elapsed = time.monotonic() - start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _broadcast(self, frame_bytes):
        """Send to all clients, remove dead ones."""
        with self._lock:
            clients = list(self._clients)
        
        dead = []
        for sock in clients:
            try:
                sock.sendall(frame_bytes)
            except (BrokenPipeError, ConnectionError, OSError, socket.timeout):
                dead.append(sock)
        
        if dead:
            with self._lock:
                for sock in dead:
                    if sock in self._clients:
                        self._clients.remove(sock)
                    try:
                        sock.close()
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# Embedded HTML/JS oscilloscope page
# ---------------------------------------------------------------------------

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vectorscope</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #000; color: #b0b0b0; font-family: 'Courier New', monospace;
       display: flex; flex-direction: column; align-items: center; height: 100vh; overflow: hidden; }
#container { position: relative; flex: 1; display: flex; align-items: center;
             justify-content: center; width: 100%; background: #000; }
#scope-body { display: flex; align-items: stretch; background: #ccc; border-radius: 4px;
              box-shadow: 0 10px 40px rgba(0,0,0,0.8); overflow: visible; }
#bezel { position: relative; background: #333; border-radius: 4px;
         box-shadow: inset 0 2px 15px rgba(0,0,0,0.6), 0 1px 2px rgba(255,255,255,0.1); }
#bezel-title { position: absolute; left: 0; width: 100%;
               text-align: center; color: #aaa; font-weight: bold;
               letter-spacing: 2px; text-transform: uppercase; }
#bezel-link { position: absolute; right: 0; color: #888; text-decoration: none;
              cursor: pointer; opacity: 0.6; }
canvas { display: block; background: #000; border-radius: 2px; }
#bezel-status { position: absolute; left: 0; color: #888; opacity: 0.6; }
#controls { display: flex; background: transparent; overflow: visible; }
.ctrl-section { position: relative; border: 1px solid rgba(0,0,0,0.05);
                background: rgba(0,0,0,0.02); border-radius: 4px; }
.ctrl-title { position: absolute; width: 100%; color: #222; font-weight: bold; letter-spacing: 1px;
              text-transform: uppercase; text-align: center; white-space: nowrap; }
.dial-group { position: absolute; left: 50%; transform: translate(-50%, -50%);
              display: flex; flex-direction: column; align-items: center; }
.dial { border-radius: 50%; position: relative; cursor: grab; user-select: none;
        background: radial-gradient(circle at 38% 38%, #555, #1a1a1a 65%, #000);
        box-shadow: 0 3px 8px rgba(0,0,0,0.5), inset 0 1px 2px rgba(255,255,255,0.1);
        border: 1px solid #111; }
.dial-pointer { position: absolute; top: 50%; left: 50%; width: 2px;
                background: #ff9d00; transform-origin: bottom center;
                border-radius: 1px; }
.dial-label { color: #333; text-align: center; text-transform: uppercase;
              white-space: nowrap; font-weight: bold; }
.dial-tick { position: absolute; color: #444; white-space: nowrap;
             transform: translate(-50%, -50%); pointer-events: none; font-weight: bold; }
</style>
</head>
<body>
<div id="container">
  <div id="scope-body">
    <div id="bezel">
      <div id="bezel-title">vectorscope</div>
      <canvas id="scope"></canvas>
      <a id="bezel-link" href="https://github.com/kevinabrandon/vectorscope" target="_blank">github.com/kevinabrandon/vectorscope</a>
      <div id="bezel-status">connecting...</div>
    </div>
    <div id="controls">
      <div class="ctrl-section">
        <div class="ctrl-title" data-grid="0">DISPLAY</div>
        <div class="dial-group" data-grid="2" data-param="intensity">
          <div class="dial-label">INTENSITY</div>
          <div class="dial dial-sm"><div class="dial-pointer"></div></div>
        </div>
        <div class="dial-group" data-grid="6" data-param="focus">
          <div class="dial-label">FOCUS</div>
          <div class="dial dial-sm"><div class="dial-pointer"></div></div>
        </div>
      </div>
      <div class="ctrl-section">
        <div class="ctrl-title" data-grid="0">CH1 &mdash; X</div>
        <div class="dial-group" data-grid="2" data-param="pos_x">
          <div class="dial-label">POSITION</div>
          <div class="dial dial-sm"><div class="dial-pointer"></div></div>
        </div>
        <div class="dial-group" data-grid="5" data-param="volts_x">
          <div class="dial-label">VOLTS/DIV</div>
          <div class="dial dial-lg">
            <div class="dial-pointer"></div>
            <span class="dial-tick" data-angle="-135">2</span>
            <span class="dial-tick" data-angle="-67.5">1</span>
            <span class="dial-tick" data-angle="0">.5</span>
            <span class="dial-tick" data-angle="67.5">.2</span>
            <span class="dial-tick" data-angle="135">.1</span>
          </div>
        </div>
      </div>
      <div class="ctrl-section">
        <div class="ctrl-title" data-grid="0">CH2 &mdash; Y</div>
        <div class="dial-group" data-grid="2" data-param="pos_y">
          <div class="dial-label">POSITION</div>
          <div class="dial dial-sm"><div class="dial-pointer"></div></div>
        </div>
        <div class="dial-group" data-grid="5" data-param="volts_y">
          <div class="dial-label">VOLTS/DIV</div>
          <div class="dial dial-lg">
            <div class="dial-pointer"></div>
            <span class="dial-tick" data-angle="-135">2</span>
            <span class="dial-tick" data-angle="-67.5">1</span>
            <span class="dial-tick" data-angle="0">.5</span>
            <span class="dial-tick" data-angle="67.5">.2</span>
            <span class="dial-tick" data-angle="135">.1</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
(function() {
  const canvas = document.getElementById('scope');
  const ctx = canvas.getContext('2d');
  const statusEl = document.getElementById('bezel-status');

  let channels = 2;
  let zAmp = 1.0;
  let scaleFactor = 1.0;
  let pendingFrames = [];
  let intensityScale = 0.75; // Initial centered value
  let focusScale = 0.0; // 0.0 is focused, higher is blurred
  let voltsX = 0.5;
  let voltsY = 0.5;
  let posX = 0.0;
  let posY = 0.0;

  const bezel = document.getElementById('bezel');
  const controlsEl = document.getElementById('controls');
  function resize() {
    const container = document.getElementById('container');
    const maxW = container.clientWidth;
    const maxH = container.clientHeight;

    // Sizing constant: pad = ch / 10
    let ch = Math.min(maxH / 1.25, maxW / 2.85);
    let cw = Math.max(200, Math.round(ch * 1.25));
    ch = Math.max(160, Math.round(ch));
    const pad = Math.round(ch / 10);
    const secW = Math.round(ch * 0.25);
    
    canvas.width = cw;
    canvas.height = ch;
    
    container.style.padding = `${pad}px`;
    const scopeBody = document.getElementById('scope-body');
    scopeBody.style.padding = `${pad}px`;
    scopeBody.style.overflow = 'visible';
    bezel.style.padding = `${pad}px`;

    // Internal spacing between sections
    bezel.style.marginRight = `${pad}px`;
    const sections = document.querySelectorAll('.ctrl-section');
    const titleFontSize = Math.max(8, Math.round(pad * 0.45));
    
    sections.forEach((s, i) => {
      s.style.width = `${secW}px`;
      s.style.height = `${ch}px`;
      s.style.marginTop = `${pad}px`; // Move box down
      
      const title = s.querySelector('.ctrl-title');
      title.style.fontSize = `${titleFontSize}px`;
      title.style.top = `${-Math.round(pad * 0.75)}px`; // Place above box
      
      if (i < sections.length - 1) {
        s.style.marginRight = `${pad}px`;
      } else {
        s.style.marginRight = '0';
      }
    });

    const bezelTitle = document.getElementById('bezel-title');
    bezelTitle.style.fontSize = `${Math.round(pad * 0.5)}px`;
    bezelTitle.style.top = `${Math.round(pad * 0.2)}px`;
    const link = document.getElementById('bezel-link');
    link.style.fontSize = `${Math.round(pad * 0.35)}px`;
    link.style.bottom = `${Math.round(pad * 0.2)}px`;
    link.style.right = `${Math.round(pad * 0.5)}px`;
    const status = document.getElementById('bezel-status');
    status.style.fontSize = `${Math.round(pad * 0.35)}px`;
    status.style.bottom = `${Math.round(pad * 0.2)}px`;
    status.style.left = `${Math.round(pad * 0.5)}px`;

    const lgSize = Math.round(secW * 0.7);
    const smSize = Math.round(secW * 0.35);
    document.querySelectorAll('.dial-lg').forEach(d => {
      d.style.width = `${lgSize}px`; d.style.height = `${lgSize}px`;
    });
    document.querySelectorAll('.dial-sm').forEach(d => {
      d.style.width = `${smSize}px`; d.style.height = `${smSize}px`;
    });
    // Position titles and dials on grid lines (8 rows, 9 lines 0-8)
    document.querySelectorAll('[data-grid]').forEach(el => {
      if (el.classList.contains('ctrl-title')) return;
      const line = parseInt(el.dataset.grid);
      el.style.top = `${Math.round((line / 8) * ch)}px`;
    });
    // Pointer: thin line from center to near edge
    document.querySelectorAll('.dial-pointer').forEach(p => {
      const dial = p.parentElement;
      const r = dial.offsetWidth / 2;
      const ptrH = Math.round(r * 0.75);
      p.style.height = `${ptrH}px`;
      p.style.marginTop = `${-ptrH}px`;
    });
    // Position tick labels around large dials
    const tickFontSize = Math.max(7, Math.round(pad * 0.35));
    document.querySelectorAll('.dial-tick').forEach(t => {
      const dial = t.parentElement;
      const r = dial.offsetWidth / 2;
      const dist = r + tickFontSize * 0.8;
      const ang = parseFloat(t.dataset.angle) * Math.PI / 180;
      t.style.left = `${Math.round(r + Math.sin(ang) * dist)}px`;
      t.style.top = `${Math.round(r - Math.cos(ang) * dist)}px`;
      t.style.fontSize = `${tickFontSize}px`;
    });
    // Font sizes
    const labelFontSize = `${Math.max(7, Math.round(pad * 0.35))}px`;
    document.querySelectorAll('.dial-label').forEach(l => {
      l.style.fontSize = labelFontSize;
      const isLg = l.parentElement.querySelector('.dial-lg');
      l.style.marginBottom = `${Math.round(pad * (isLg ? 0.8 : 0.35))}px`;
    });
    drawGraticule();
  }
  window.addEventListener('resize', resize);
  resize();

  function drawGraticule() {
    const w = canvas.width, h = canvas.height;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    const cols = 10, rows = 8;
    for (let i = 0; i <= cols; i++) {
      const x = (i / cols) * w;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let i = 0; i <= rows; i++) {
      const y = (i / rows) * h;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    
    // Draw Center Axes with Ticks
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 1.5;
    const cx = w / 2;
    const cy = h / 2;
    
    // Horizontal center line
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
    // Vertical center line
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
    
    // Ticks on horizontal center line (vertical marks)
    const tickH = h / 100;
    for (let i = 0; i <= cols * 5; i++) {
      if (i % 5 === 0) continue; // Skip major grid lines
      const x = (i / (cols * 5)) * w;
      ctx.beginPath();
      ctx.moveTo(x, cy - tickH);
      ctx.lineTo(x, cy + tickH);
      ctx.stroke();
    }
    
    // Ticks on vertical center line (horizontal marks)
    const tickW = w / 125;
    for (let i = 0; i <= rows * 5; i++) {
      if (i % 5 === 0) continue; // Skip major grid lines
      const y = (i / (rows * 5)) * h;
      ctx.beginPath();
      ctx.moveTo(cx - tickW, y);
      ctx.lineTo(cx + tickW, y);
      ctx.stroke();
    }
  }

  function drawTrace(floats) {
    const w = canvas.width, h = canvas.height;
    const s = h, ox = (w - h) / 2;
    // Server sends 2 (XY) or 3 (XY+Z) floats per sample
    const stride = Math.min(channels, 3);
    const numPts = Math.floor(floats.length / stride);
    if (numPts < 2) return;

    // Scale intensity: 0.0 to 1.0 (with 0.75 center)
    // Adjust level based on focus (dimmer when out of focus)
    const level = intensityScale * (1.0 - (focusScale * 0.5));

    // Width adjustments for focus
    const coreWidth = 1.5 + (1.0 * (1.0 - focusScale)) + (focusScale * 8.0);
    const glowWidth = 6.0 + (focusScale * 25.0);

    // Volts/Div scaling: true to grid (10 cols X, 8 rows Y)
    const scaleX = (0.25 * scaleFactor) / voltsX;
    const scaleY = (0.25 * scaleFactor) / voltsY;

    if (stride < 3) {
      // Draw glow pass (wide, dim)
      ctx.strokeStyle = `rgba(0, 255, 65, ${0.12 * level})`;
      ctx.lineWidth = glowWidth;
      ctx.beginPath();
      for (let i = 0; i < numPts; i++) {
        const x = (floats[i * stride] * scaleX + posX + 1) * 0.5 * s + ox;
        const y = (1 - (floats[i * stride + 1] * scaleY + posY + 1) * 0.5) * s;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      // Draw bright core
      ctx.strokeStyle = '#00ff41';
      ctx.globalAlpha = Math.min(1, level);
      ctx.lineWidth = coreWidth;
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    } else {
      // Z-channel: batch visible segments into subpaths, skip blanked
      let prevX = (floats[0] * scaleX + posX + 1) * 0.5 * s + ox;
      let prevY = (1 - (floats[1] * scaleY + posY + 1) * 0.5) * s;
      let hasPartial = false;

      ctx.beginPath();
      let inSub = false;
      for (let i = 1; i < numPts; i++) {
        const x = (floats[i * stride] * scaleX + posX + 1) * 0.5 * s + ox;
        const y = (1 - (floats[i * stride + 1] * scaleY + posY + 1) * 0.5) * s;
        const zRaw = floats[i * stride + 2];
        
        const zNorm = Math.max(-1, Math.min(1, zRaw / zAmp));
        const intensity = Math.max(0, Math.min(1, (1 - zNorm) * 0.5)) * level;
        
        if (intensity > 0.005) {
          if (!inSub) { ctx.moveTo(prevX, prevY); inSub = true; }
          ctx.lineTo(x, y);
          if (intensity < 0.995) hasPartial = true;
        } else {
          inSub = false;
        }
        prevX = x;
        prevY = y;
      }
      // Glow pass (always batched)
      ctx.strokeStyle = `rgba(0, 255, 65, ${0.12 * level})`;
      ctx.lineWidth = glowWidth;
      ctx.stroke();

      if (!hasPartial && level >= 0.99) {
        // All visible at full intensity — batched core (fast path)
        ctx.strokeStyle = '#00ff41';
        ctx.lineWidth = coreWidth;
        ctx.stroke();
      } else {
        // Variable intensity — per-segment core over batched glow
        prevX = (floats[0] * scaleX + posX + 1) * 0.5 * s + ox;
        prevY = (1 - (floats[1] * scaleY + posY + 1) * 0.5) * s;
        for (let i = 1; i < numPts; i++) {
          const x = (floats[i * stride] * scaleX + posX + 1) * 0.5 * s + ox;
          const y = (1 - (floats[i * stride + 1] * scaleY + posY + 1) * 0.5) * s;
          const zRaw = floats[i * stride + 2];
          
          const zNorm = Math.max(-1, Math.min(1, zRaw / zAmp));
          const intensity = Math.max(0, Math.min(1, (1 - zNorm) * 0.5)) * level;
          
          if (intensity > 0.005) {
            const adj = Math.min(1, intensity * intensity);
            const green = Math.round(40 + 215 * adj);
            const alpha = Math.min(1, 0.03 + 0.97 * adj);
            
            // Scaled core width based on per-segment intensity and global focus
            const segCoreWidth = (1.0 + (1.0 * adj)) * (coreWidth / 2.0);
            
            ctx.lineWidth = segCoreWidth;
            ctx.strokeStyle = `rgba(0, ${green}, 40, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(x, y);
            ctx.stroke();
          }
          prevX = x;
          prevY = y;
        }
      }
    }
  }

  let lastAnimTime = 0;
  function animate(now) {
    requestAnimationFrame(animate);

    const dt = lastAnimTime ? (now - lastAnimTime) / 1000 : 0.016;
    lastAnimTime = now;

    // Time-based phosphor decay: ~0.5s to fade to near-zero
    const decay = 1 - Math.pow(0.02, dt / 0.5);
    ctx.fillStyle = `rgba(40, 120, 110, ${decay})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawGraticule();

    while (pendingFrames.length > 0) {
      drawTrace(pendingFrames.shift());
    }

  }
  requestAnimationFrame(animate);

  // Dial interaction — click and drag to rotate
  let activeDial = null;
  let activePointer = null;
  let isDiscrete = false;

  function updateDialAngle(clientX, clientY) {
    const rect = activeDial.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    let angle = Math.atan2(clientX - cx, -(clientY - cy)) * (180 / Math.PI);
    angle = Math.max(-135, Math.min(135, angle));
    if (isDiscrete) {
      const step = 270 / 4;
      angle = Math.round((angle + 135) / step) * step - 135;
    }
    activePointer.style.transform = `rotate(${angle}deg)`;

    // Handle parameter changes
    const group = activeDial.closest('.dial-group');
    if (group && group.dataset.param === 'intensity') {
      // Map [-135, 135] to [0.0, 1.0] where center (0) is 0.75
      if (angle <= 0) {
        intensityScale = (angle + 135) / 135 * 0.75;
      } else {
        intensityScale = 0.75 + (angle / 135 * 0.25);
      }
    } else if (group && group.dataset.param === 'focus') {
      focusScale = Math.abs(angle) / 135;
    } else if (group && (group.dataset.param === 'volts_x' || group.dataset.param === 'volts_y')) {
      const values = [2.0, 1.0, 0.5, 0.2, 0.1];
      const step = 270 / 4;
      const idx = Math.round((angle + 135) / step);
      const val = values[Math.max(0, Math.min(values.length - 1, idx))];
      if (group.dataset.param === 'volts_x') voltsX = val;
      else voltsY = val;
    } else if (group && (group.dataset.param === 'pos_x' || group.dataset.param === 'pos_y')) {
      const val = angle / 135;
      if (group.dataset.param === 'pos_x') posX = val;
      else posY = val;
    }
  }

  document.querySelectorAll('.dial').forEach(dial => {
    dial.addEventListener('mousedown', e => {
      e.preventDefault();
      activeDial = dial;
      activePointer = dial.querySelector('.dial-pointer');
      isDiscrete = dial.classList.contains('dial-lg');
      dial.style.cursor = 'grabbing';
      updateDialAngle(e.clientX, e.clientY);
    });
    dial.addEventListener('touchstart', e => {
      e.preventDefault();
      activeDial = dial;
      activePointer = dial.querySelector('.dial-pointer');
      isDiscrete = dial.classList.contains('dial-lg');
      const t = e.touches[0];
      updateDialAngle(t.clientX, t.clientY);
    });
  });

  document.addEventListener('mousemove', e => {
    if (activeDial) updateDialAngle(e.clientX, e.clientY);
  });
  document.addEventListener('touchmove', e => {
    if (activeDial) {
      const t = e.touches[0];
      updateDialAngle(t.clientX, t.clientY);
    }
  });
  document.addEventListener('mouseup', () => {
    if (activeDial) activeDial.style.cursor = 'grab';
    activeDial = null;
  });
  document.addEventListener('touchend', () => { activeDial = null; });

  function connect() {
    statusEl.textContent = 'connecting...';
    const ws = new WebSocket(`ws://${location.host}/`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = function() {
      statusEl.textContent = 'connected';
    };

    ws.onmessage = function(evt) {
      if (typeof evt.data === 'string') {
        try {
          const meta = JSON.parse(evt.data);
          if (meta.channels) { channels = meta.channels; }
          if (meta.z_amp) { zAmp = meta.z_amp; }
          if (meta.scale_factor) { scaleFactor = meta.scale_factor; }
        } catch(e) {}
      } else {
        pendingFrames.push(new Float32Array(evt.data));
        if (pendingFrames.length > 100) { pendingFrames.shift(); }
      }
    };

    ws.onclose = function(evt) {
      pendingFrames = [];
      statusEl.textContent = 'disconnected — reconnecting...';
      setTimeout(connect, 2000);
    };

    ws.onerror = function(evt) {
      console.log('WS error', evt);
      ws.close();
    };
  }

  connect();
})();
</script>
</body>
</html>
"""

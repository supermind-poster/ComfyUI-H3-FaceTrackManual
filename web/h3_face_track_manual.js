import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "H3FaceTrackManual";

// ---------------------------------------------------------------- parsing / formatting
const nums = (s) => (s || "").match(/-?\d+(?:\.\d+)?/g) || [];
function parseKeyframes(text) {
  const kf = {};
  for (const raw of (text || "").split("\n")) {
    const line = raw.split("#")[0].trim();
    if (!line) continue;
    const n = nums(line);
    if (n.length === 5) kf[parseInt(n[0])] = n.slice(1).map(Number);
  }
  return kf;
}
function fmtKeyframes(kf, sig, frames) {
  const keys = Object.keys(kf).map(Number).sort((a, b) => a - b);
  return (sig ? `# video: ${sig}\n` : "") + (frames ? `# frames: ${frames}\n` : "") +
    "# frame: x1 y1 x2 y2   (CROP box, source-frame pixels, origin top-left)\n" +
    keys.map((f) => `${f}: ${kf[f].map((v) => Math.round(v * 10) / 10).join(" ")}`).join("\n") + "\n";
}
const videoMarker = (text) => ((text || "").match(/^[ \t]*#[ \t]*video:[ \t]*(.+?)[ \t]*$/m) || [])[1] || null;
const parseInts = (t) => ((t || "").match(/\d+/g) || []).map(Number);
function parseRanges(t) {
  const out = [];
  const re = /(\d+)\s*-\s*(\d+)|(\d+)/g;
  let m;
  while ((m = re.exec(t || ""))) {
    if (m[3] !== undefined) out.push([+m[3], +m[3]]);
    else out.push([Math.min(+m[1], +m[2]), Math.max(+m[1], +m[2])]);
  }
  return out;
}
const fmtRanges = (r) => r.map(([a, b]) => (a === b ? `${a}` : `${a}-${b}`)).join(", ");

// ---------------------------------------------------------------- interpolation (mirrors the node)
function endSlope(h0, h1, d0, d1) {
  let s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
  if (Math.sign(s) !== Math.sign(d0)) s = 0;
  else if (Math.sign(d0) !== Math.sign(d1) && Math.abs(s) > 3 * Math.abs(d0)) s = 3 * d0;
  return s;
}
function interp(xs, ys, x, mode) {
  const n = xs.length;
  if (n === 1 || x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 0;
  while (i < n - 2 && x >= xs[i + 1]) i++;
  if (mode === "hold") return ys[i];
  const h = [], d = [];
  for (let k = 0; k < n - 1; k++) { h[k] = xs[k + 1] - xs[k]; d[k] = (ys[k + 1] - ys[k]) / h[k]; }
  const t = (x - xs[i]) / h[i];
  if (mode === "linear" || n < 3) return ys[i] + (ys[i + 1] - ys[i]) * t;
  const m = new Array(n);
  for (let k = 1; k < n - 1; k++) {
    if (d[k - 1] * d[k] <= 0) m[k] = 0;
    else {
      const w1 = 2 * h[k] + h[k - 1], w2 = h[k] + 2 * h[k - 1];
      m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k]);
    }
  }
  m[0] = endSlope(h[0], h[1], d[0], d[1]);
  m[n - 1] = endSlope(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
  const t2 = t * t, t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i] +
    (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1];
}

// ---------------------------------------------------------------- widget helpers
const getW = (node, name) => node.widgets?.find((x) => x.name === name);
function setW(node, name, val) {
  const x = getW(node, name);
  if (!x) return;
  x.value = val;
  try { x.callback?.(val); } catch (e) { /* ignore */ }
}

function toast(msg, undoNode) {
  const d = document.createElement("div");
  d.textContent = msg;
  if (undoNode && undoNode.__h3undo) {                       // an automatic clearing can be taken back
    const b = document.createElement("button");
    b.textContent = "Undo";
    Object.assign(b.style, { marginLeft: "12px", cursor: "pointer", background: "#2c2c32", color: "#eee",
      border: "1px solid #666", borderRadius: "4px", padding: "2px 10px" });
    b.onclick = () => {
      const u = undoNode.__h3undo;
      for (const k of ["keyframes", "cut_frames", "absent_ranges"]) setW(undoNode, k, u[k]);
      app.graph.setDirtyCanvas(true, true);
      d.remove();
    };
    d.appendChild(b);
  }
  Object.assign(d.style, { position: "fixed", left: "50%", bottom: "40px", transform: "translateX(-50%)",
    background: "#222", color: "#eee", padding: "10px 16px", border: "1px solid #555", borderRadius: "6px",
    zIndex: 10001, font: "13px sans-serif" });
  document.body.appendChild(d);
  return () => d.remove();
}
function linkOrigin(node, inputName) {
  const inp = node.inputs?.find((i) => i.name === inputName);
  if (!inp || inp.link == null) return null;
  const g = app.graph;
  const link = g.links?.get ? g.links.get(inp.link) : g.links?.[inp.link];
  return link ? g.getNodeById(link.origin_id) : null;
}
// The video plugged into `video`: a Load Video node directly (its file is read straight from disk), or
// anything else (Trim Video, Crop Video, ...), in which case the frames the node itself receives are used.
function upstreamVideo(node) {
  const inp = node.inputs?.find((i) => i.name === "video");
  if (!inp || inp.link == null) return { connected: false, file: null, direct: false };
  const origin = linkOrigin(node, "video");
  const fileW = origin && origin.type === "LoadVideo" ? origin.widgets?.find((w) => w.name === "file") : null;
  return { connected: true, file: fileW?.value || null, direct: !!fileW };
}
// A string that changes whenever anything feeding `video` (file, trim times, crop values, ...) changes.
function chainKey(node) {
  const parts = [], seen = new Set(), g = app.graph;
  const walk = (n) => {
    if (!n || seen.has(n.id)) return;
    seen.add(n.id);
    parts.push(`${n.type}:` + (n.widgets || []).filter((w) => w.type !== "button" && typeof w.value !== "object")
      .map((w) => `${w.name}=${w.value}`).join(","));
    for (const inp of n.inputs || []) {
      if (inp.link == null) continue;
      const link = g.links?.get ? g.links.get(inp.link) : g.links?.[inp.link];
      if (link) walk(g.getNodeById(link.origin_id));
    }
  };
  walk(linkOrigin(node, "video"));
  return parts.join("|");
}
// Run only this node and what feeds it, so the frames after Trim Video / Crop Video / ... get registered.
function prunePrompt(output, id) {
  const keep = new Set(), stack = [String(id)];
  while (stack.length) {
    const k = stack.pop();
    if (keep.has(k) || !output[k]) continue;
    keep.add(k);
    for (const v of Object.values(output[k].inputs || {}))
      if (Array.isArray(v) && v.length === 2 && output[String(v[0])] && typeof v[1] === "number") stack.push(String(v[0]));
  }
  const pruned = {};
  for (const k of keep) pruned[k] = output[k];
  return pruned;
}
async function loadNodeFrames(node) {
  const id = String(node.id), nid = encodeURIComponent(id);
  let prev = 0;
  try { const r = await fetch(`/h3manual/vinfo?node=${nid}`); if (r.ok) prev = (await r.json()).time || 0; } catch (e) { /* none yet */ }
  const p = await app.graphToPrompt();
  if (!p.output[id]) throw new Error("this node is muted/bypassed or not part of the graph");
  const pruned = prunePrompt(p.output, id);
  pruned[id] = { ...pruned[id], inputs: { ...pruned[id].inputs, keyframes: `#H3MANUAL_PREVIEW_ONLY ${Date.now()}` } };
  let failed = false;
  const onErr = () => { failed = true; };
  api.addEventListener("execution_error", onErr);
  try {
    await api.queuePrompt(0, { output: pruned, workflow: p.workflow });
    const t0 = Date.now();
    while (Date.now() - t0 < 10 * 60 * 1000) {
      if (failed) throw new Error("the nodes feeding this one failed - see the error dialog");
      const r = await fetch(`/h3manual/vinfo?node=${nid}`);
      if (r.ok && ((await r.json()).time || 0) > prev) return;
      await new Promise((res) => setTimeout(res, 800));
    }
    throw new Error("timed out waiting for the video");
  } finally { api.removeEventListener("execution_error", onErr); }
}
// One keyframe after YOLO has found the face in it: the crop box is moved so the face is in its centre
// (when `center`), and enlarged when it is smaller than crop_factor asks for (never shrunk).
// `fitFn(cx, cy, h)` builds the box (right aspect ratio, inside the frame).
function calibrateBox(box, face, cf, aspect, center, fitFn) {
  const fh = face[3] - face[1], fw = face[2] - face[0];
  const need = cf * Math.max(fh, fw / aspect);                 // crop height asked for by crop_factor
  const have = box[3] - box[1];
  const h = Math.max(have, need);
  const cx = center ? (face[0] + face[2]) / 2 : (box[0] + box[2]) / 2;
  const cy = center ? (face[1] + face[3]) / 2 : (box[1] + box[3]) / 2;
  const nb = fitFn(cx, cy, h);
  const shift = Math.hypot((nb[0] + nb[2]) / 2 - (box[0] + box[2]) / 2, (nb[1] + nb[3]) / 2 - (box[1] + box[3]) / 2);
  const off = Math.hypot((nb[0] + nb[2]) / 2 - (face[0] + face[2]) / 2, (nb[1] + nb[3]) / 2 - (face[1] + face[3]) / 2);
  return { box: nb, grown: h > have + 0.5, ratio: h / have, moved: center && shift > 0.5,
    notCentered: center && off > 0.01 * (nb[3] - nb[1]) };     // the frame edge stopped it
}
const splitMarker = (m) => {
  if (!m) return { sig: null, fps: 0 };
  const i = m.lastIndexOf("@");
  return i < 0 ? { sig: m, fps: 0 } : { sig: m.slice(0, i), fps: parseFloat(m.slice(i + 1)) || 0 };
};
function hasData(node) {
  return Object.keys(parseKeyframes(getW(node, "keyframes")?.value)).length > 0 ||
    !!(getW(node, "cut_frames")?.value || "").trim() || !!(getW(node, "absent_ranges")?.value || "").trim();
}
// the frame rate changed but the file did not: frame numbers scale, so keyframes / cuts / absent follow
function rescaleNodeData(node, factor, newFps) {
  node.__h3undo = { keyframes: getW(node, "keyframes")?.value, cut_frames: getW(node, "cut_frames")?.value,
    absent_ranges: getW(node, "absent_ranges")?.value };
  const text = getW(node, "keyframes")?.value;
  const { sig } = splitMarker(videoMarker(text));
  const kf = parseKeyframes(text), kf2 = {};
  for (const [k, v] of Object.entries(kf)) kf2[Math.round(k * factor)] = v;
  setW(node, "keyframes", fmtKeyframes(kf2, sig ? `${sig}@${newFps.toFixed(3)}` : null, 0));
  setW(node, "cut_frames", parseInts(getW(node, "cut_frames")?.value).map((c) => Math.max(1, Math.round(c * factor))).join(", "));
  setW(node, "absent_ranges", fmtRanges(parseRanges(getW(node, "absent_ranges")?.value)
    .map(([a, b]) => [Math.round(a * factor), Math.round(b * factor)])));
  app.graph.setDirtyCanvas(true, true);
}
const DEFAULT_KEYFRAMES = "# frame: x1 y1 x2 y2   (CROP box, source-frame pixels, origin top-left)\n" +
  "# filled by the 'Edit keyframes (graphic)' button\n";
function resetNodeData(node) {
  node.__h3undo = { keyframes: getW(node, "keyframes")?.value, cut_frames: getW(node, "cut_frames")?.value,
    absent_ranges: getW(node, "absent_ranges")?.value };
  setW(node, "keyframes", DEFAULT_KEYFRAMES);
  setW(node, "cut_frames", "");
  setW(node, "absent_ranges", "");
  app.graph.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------- editor
async function openEditor(node) {
  // Load Video plugged in directly: its file is read straight from disk (nothing decoded up front).
  // Anything else (Trim Video, Crop Video, ...): the frames this very node receives, which a short
  // run of just this node and what feeds it makes available.
  const up = upstreamVideo(node);
  if (!up.connected) { alert("Connect a video (a Load Video node) to this node's 'video' input first."); return; }
  const nid = encodeURIComponent(node.id);
  const source = up.direct && up.file
    ? { kind: "file", q: `name=${encodeURIComponent(up.file)}`, fps: Number(getW(node, "fps")?.value ?? 24) || 0 }
    : { kind: "node", q: `node=${nid}`, fps: 0 };                  // those frames are already at the node's fps
  const fpsParam = source.fps;
  let info = null;
  {
    const hide = toast(source.kind === "file" ? "Opening the video..." : "Reading the video after the nodes in front of this one...");
    try {
      if (source.kind === "node") await loadNodeFrames(node);
      const r = await fetch(`/h3manual/vinfo?${source.q}&fps=${fpsParam}`);
      if (!r.ok) throw new Error(await r.text());
      const m = await r.json();
      const conv = Math.abs(m.fps - m.src_fps) >= 0.001;
      info = { frames: m.frames, width: m.src_width, height: m.src_height, sig: m.sig, fps: m.fps,
        marker: `${m.sig}@${Number(m.fps).toFixed(3)}`,
        label: `${source.kind === "file" ? up.file : "video from the node"}: ${m.frames} frames @ ${Number(m.fps).toFixed(2)} fps` +
          (conv ? ` (source ${Number(m.src_fps).toFixed(2)} fps, ${m.src_frames} frames)` : "") + `, ${m.src_width}x${m.src_height}` };
    } catch (e) { alert("Cannot open the video: " + e.message); return; }
    finally { hide(); }
  }
  const N = info.frames, VW = info.width, VH = info.height;
  if (!N || !VW) { alert("No frames available."); return; }
  const frameUrl = (idx) => `/h3manual/vframe?${source.q}&index=${idx}&fps=${fpsParam}`;

  // keyframes made for another file do not belong to this video: start clean. Same file at another
  // frame rate: the numbers are scaled instead.
  const saved = splitMarker(videoMarker(getW(node, "keyframes")?.value));
  const staleClip = !!saved.sig && saved.sig !== info.sig;
  let rescaled = false;
  if (staleClip) resetNodeData(node);
  else if (saved.sig && saved.fps && Math.abs(saved.fps - info.fps) > 0.0005) {
    rescaleNodeData(node, info.fps / saved.fps, info.fps);
    rescaled = true;
  }
  let kf = parseKeyframes(getW(node, "keyframes")?.value);
  for (const k of Object.keys(kf)) if (+k >= N) delete kf[k];
  const cuts = new Set(parseInts(getW(node, "cut_frames")?.value).filter((c) => c > 0 && c < N));
  let absent = parseRanges(getW(node, "absent_ranges")?.value)
    .filter(([a]) => a < N).map(([a, b]) => [a, Math.min(b, N - 1)]);
  let note = staleClip ? "new video: previous keyframes were cleared" : rescaled ? "frame rate changed: keyframes, cuts and absent were rescaled" : "", busy = false, frame = 0, img = null, token = 0, pendingAbsent = null, drag = null;
  const cache = new Map();

  // ---- DOM
  const css = (el, s) => Object.assign(el.style, s);
  const overlay = document.createElement("div");
  overlay.tabIndex = 0;
  css(overlay, { position: "fixed", inset: "0", zIndex: 10000, background: "rgba(10,10,12,.96)",
    display: "flex", flexDirection: "column", color: "#ddd", font: "13px sans-serif", outline: "none" });
  const bar = document.createElement("div");
  css(bar, { display: "flex", gap: "16px", padding: "8px", alignItems: "stretch", borderBottom: "1px solid #333" });
  const rows = document.createElement("div");
  css(rows, { flex: "1", display: "flex", flexDirection: "column", gap: "6px", minWidth: "0" });
  const mkRow = (title, color) => {
    const r = document.createElement("div");
    css(r, { display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center", paddingLeft: "6px",
      borderLeft: `3px solid ${color}` });
    const l = document.createElement("span");
    l.textContent = title;
    css(l, { minWidth: "92px", fontWeight: "bold", color, letterSpacing: ".5px" });
    r.appendChild(l); rows.appendChild(r);
    return r;
  };
  const rowKey = mkRow("KEYFRAMES", "#e0e"), rowTrk = mkRow("TRACKING", "#4ac"), rowCut = mkRow("CUTS", "#fa0"), rowAbs = mkRow("ABSENT", "#f55");
  const side = document.createElement("div");
  css(side, { display: "flex", flexDirection: "column", justifyContent: "space-between", alignItems: "flex-end",
    gap: "6px", minWidth: "150px" });
  const sideTop = document.createElement("div");
  css(sideTop, { display: "flex", gap: "6px" });
  side.appendChild(sideTop);
  bar.append(rows, side);
  let target = rowKey;
  const area = document.createElement("div");
  css(area, { flex: "1", position: "relative", minHeight: "0" });
  const canvas = document.createElement("canvas");
  css(canvas, { position: "absolute", inset: "0", cursor: "crosshair" });
  area.appendChild(canvas);
  const foot = document.createElement("div");
  css(foot, { padding: "8px", borderTop: "1px solid #333" });
  const strip = document.createElement("canvas");
  strip.height = 18;
  css(strip, { width: "100%", height: "18px", display: "block" });
  const slider = document.createElement("input");
  slider.type = "range"; slider.min = 0; slider.max = N - 1; slider.value = 0;
  css(slider, { width: "100%", margin: "4px 0" });
  const status = document.createElement("div");
  css(status, { fontFamily: "monospace", opacity: ".85" });
  foot.append(strip, slider, status);
  overlay.append(bar, area, foot);

  const btn = (label, title, fn, color) => {
    const b = document.createElement("button");
    b.textContent = label; b.title = title || "";
    css(b, { padding: "5px 10px", background: color || "#2c2c32", color: "#eee",
      border: "1px solid #4a4a52", borderRadius: "4px", cursor: "pointer" });
    b.onclick = (e) => { e.preventDefault(); fn(); overlay.focus(); };
    target.appendChild(b);
    return b;
  };

  // ---- geometry
  const ctx = canvas.getContext("2d");
  let scale = 1, offX = 0, offY = 0;
  function layout() {
    canvas.width = area.clientWidth; canvas.height = area.clientHeight;
    strip.width = strip.clientWidth || 800;
    scale = Math.min(canvas.width / VW, canvas.height / VH);
    offX = (canvas.width - VW * scale) / 2; offY = (canvas.height - VH * scale) / 2;
  }
  const toScr = (x, y) => [offX + x * scale, offY + y * scale];
  const clampX = (v) => Math.max(0, Math.min(VW, v));
  const clampY = (v) => Math.max(0, Math.min(VH, v));

  // ---- box model
  function shotOf(f) {
    const b = [0, ...[...cuts].sort((a, c) => a - c), N];
    for (let i = 0; i < b.length - 1; i++) if (f >= b[i] && f < b[i + 1]) return [b[i], b[i + 1]];
    return [0, N];
  }
  const keys = () => Object.keys(kf).map(Number).sort((a, b) => a - b);
  function boxAt(f) {
    if (kf[f]) return kf[f].slice();
    const all = keys();
    if (!all.length) return null;
    const [s, e] = shotOf(f);
    let ks = all.filter((k) => k >= s && k < e);
    if (!ks.length) {
      const dist = (k) => Math.min(Math.abs(k - s), Math.abs(k - (e - 1)));
      ks = [all.reduce((a, b) => (dist(b) < dist(a) ? b : a))];
    }
    const mode = getW(node, "interpolation")?.value || "smooth";
    return [0, 1, 2, 3].map((c) => interp(ks, ks.map((k) => kf[k][c]), f, mode));
  }
  const r32 = (v) => Math.max(32, Math.round(v / 32) * 32);
  function cfg() {
    const mode = getW(node, "canvas_mode")?.value;
    const cw = r32(getW(node, "canvas_width")?.value || 768), ch = r32(getW(node, "canvas_height")?.value || 768);
    return { aspect: mode === "manual" || !mode ? cw / ch : 1,
      snap: false };                        // the crop box is free (no snapping to multiples of 32)
  }
  function place(cx, cy, w, h) {
    const x = Math.min(Math.max(cx - w / 2, 0), Math.max(0, VW - w));
    const y = Math.min(Math.max(cy - h / 2, 0), Math.max(0, VH - h));
    return [x, y, x + w, y + h];
  }
  // a crop box of height h centred on (cx, cy): same rules as the node (multiples of 32, inside the frame)
  function fit(cx, cy, h) {
    const c = cfg();
    let w;
    if (c.snap) {
      const Hm = Math.max(32, Math.floor(VH / 32) * 32), Wm = Math.max(32, Math.floor(VW / 32) * 32);
      h = Math.min(r32(h), Hm); w = r32(h * c.aspect);
      if (w > Wm) { w = Wm; h = Math.min(Hm, r32(w / c.aspect)); }
    } else {
      w = h * c.aspect;
      if (w > VW) { w = VW; h = VW / c.aspect; }
      if (h > VH) { h = VH; w = VH * c.aspect; }
    }
    return place(cx, cy, w, h).map((v) => Math.round(v * 10) / 10);
  }
  // what is shown / what a keyframe would pin on this frame
  function viewBox(f) {
    if (kf[f]) return kf[f].slice();
    const b = boxAt(f);
    return b ? fit((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1]) : null;
  }
  function defaultH() {
    const ks = keys();
    return ks.length ? kf[ks[0]][3] - kf[ks[0]][1] : r32(Math.min(VW, VH) * 0.3);
  }
  const isAbsent = (f) => absent.some(([a, b]) => f >= a && f <= b);

  // ---- drawing
  function draw() {
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (img) ctx.drawImage(img, offX, offY, VW * scale, VH * scale);
    const b = viewBox(frame);
    if (b) {
      const isKey = !!kf[frame];
      const [x0, y0] = toScr(b[0], b[1]), [x1, y1] = toScr(b[2], b[3]);
      ctx.setLineDash(isKey ? [] : [5, 4]);
      ctx.strokeStyle = isAbsent(frame) ? "#f44" : (isKey ? "#f0f" : "#0ff");
      ctx.lineWidth = 2; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.setLineDash([]); ctx.fillStyle = ctx.strokeStyle;
      for (const [hx, hy] of [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) ctx.fillRect(hx - 5, hy - 5, 10, 10);
    }
    drawStrip();
    const kcount = keys().length;
    status.textContent =
      `[${info.label}]   frame ${frame}/${N - 1}   keyframes: ${kcount}   cuts: ${cuts.size}   absent ranges: ${absent.length}` +
      (kf[frame] ? "   [KEYFRAME]" : (viewBox(frame) ? "   [interpolated - drag to make it a keyframe]" : "   [no box yet - click on the image]")) +
      (isAbsent(frame) ? "   [ABSENT]" : "") + (note ? `   | ${note}` : "") +
      (pendingAbsent !== null ? `   (absent range started at ${pendingAbsent})` : "") +
      (viewBox(frame) ? `   crop: ${Math.round(viewBox(frame)[2] - viewBox(frame)[0])}x${Math.round(viewBox(frame)[3] - viewBox(frame)[1])}  at ${viewBox(frame).slice(0, 2).map((v) => Math.round(v)).join(",")}` : "");
  }
  function drawStrip() {
    const c = strip.getContext("2d"), W = strip.width, H = strip.height;
    c.clearRect(0, 0, W, H); c.fillStyle = "#1b1b20"; c.fillRect(0, 0, W, H);
    const px = (f) => (f / Math.max(1, N - 1)) * (W - 1);
    c.fillStyle = "rgba(255,60,60,.45)";
    for (const [a, b] of absent) c.fillRect(px(a), 0, Math.max(2, px(b) - px(a)), H);
    c.fillStyle = "#fa0";
    for (const f of cuts) c.fillRect(px(f) - 1, 0, 2, H);
    c.fillStyle = "#f0f";
    for (const f of keys()) c.fillRect(px(f) - 1.5, 4, 3, H - 8);
    c.fillStyle = "#fff"; c.fillRect(px(frame) - 1, 0, 2, H);
  }

  // ---- frames
  function load(f) {
    frame = Math.max(0, Math.min(N - 1, f));
    slider.value = frame;
    const idx = frame, t = ++token;
    const hit = cache.get(idx);
    if (hit) { img = hit; draw(); return; }
    const im = new Image();
    im.onload = () => {
      cache.set(idx, im);
      if (cache.size > 80) cache.delete(cache.keys().next().value);
      if (t === token) { img = im; draw(); }
    };
    im.src = frameUrl(idx);
    draw();
  }

  // ---- playback (the frames are shown like a video)
  let playing = false, timer = null, playBtn = null, fpsSel = null;
  const pending = new Set();
  function prefetch(from, n) {
    for (let k = 1; k <= n; k++) {
      const idx = from + k;
      if (idx >= N || cache.has(idx) || pending.has(idx)) continue;
      pending.add(idx);
      const im = new Image();
      im.onload = () => { pending.delete(idx); cache.set(idx, im); if (cache.size > 160) cache.delete(cache.keys().next().value); };
      im.onerror = () => pending.delete(idx);
      im.src = frameUrl(idx);
    }
  }
  function stopPlay() {
    playing = false;
    if (timer) { clearInterval(timer); timer = null; }
    if (playBtn) playBtn.textContent = "▶ Play";
  }
  function startPlay() {
    if (playing) return;
    if (frame >= N - 1) load(0);
    playing = true; playBtn.textContent = "⏸ Pause";
    timer = setInterval(() => {
      if (frame >= N - 1) { stopPlay(); return; }
      load(frame + 1); prefetch(frame, 8);
    }, 1000 / Number(fpsSel.value || 24));
  }

  // ---- mouse
  const pos = (e) => {
    const r = canvas.getBoundingClientRect();
    return [clampX((e.clientX - r.left - offX) / scale), clampY((e.clientY - r.top - offY) / scale)];
  };
  canvas.addEventListener("pointerdown", (e) => {
    stopPlay();
    const [px, py] = pos(e);
    const b = viewBox(frame);
    drag = null;
    if (b) {
      const corners = [[b[0], b[1], b[2], b[3]], [b[2], b[1], b[0], b[3]],
        [b[0], b[3], b[2], b[1]], [b[2], b[3], b[0], b[1]]];
      for (const [cx, cy, fx, fy] of corners) {
        if (Math.hypot((cx - px) * scale, (cy - py) * scale) < 12) { drag = { mode: "resize", fx, fy }; break; }
      }
      if (!drag && px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3])
        drag = { mode: "move", dx: px - b[0], dy: py - b[1], w: b[2] - b[0], h: b[3] - b[1] };
    }
    if (!drag) {                       // click on empty image: drop a crop box there, then drag to place it
      const nb = fit(px, py, b ? b[3] - b[1] : defaultH());
      kf[frame] = nb;
      drag = { mode: "move", dx: px - nb[0], dy: py - nb[1], w: nb[2] - nb[0], h: nb[3] - nb[1] };
      draw();
    }
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const [px, py] = pos(e);
    if (drag.mode === "move") {
      const x = Math.round(Math.min(Math.max(px - drag.dx, 0), VW - drag.w));
      const y = Math.round(Math.min(Math.max(py - drag.dy, 0), VH - drag.h));
      kf[frame] = [x, y, x + drag.w, y + drag.h];
    } else {                           // resize: keep aspect, size in steps of 32, opposite corner stays put
      const c = cfg(), step = c.snap ? 32 : 4, min = c.snap ? 32 : 8;
      const fx = drag.fx, fy = drag.fy;
      const sx = px >= fx ? 1 : -1, sy = py >= fy ? 1 : -1;
      const maxW = sx > 0 ? VW - fx : fx, maxH = sy > 0 ? VH - fy : fy;
      let h0 = Math.max(Math.abs(py - fy), Math.abs(px - fx) / c.aspect);
      let h = c.snap ? r32(h0) : Math.max(min, h0);
      let w = c.snap ? r32(h * c.aspect) : h * c.aspect;
      while ((w > maxW || h > maxH) && h > min) { h -= step; w = c.snap ? r32(h * c.aspect) : h * c.aspect; }
      if (w > maxW || h > maxH) return;
      const x2 = fx + sx * w, y2 = fy + sy * h;
      kf[frame] = [Math.min(fx, x2), Math.min(fy, y2), Math.max(fx, x2), Math.max(fy, y2)].map((v) => Math.round(v * 10) / 10);
    }
    draw();
  });
  const endDrag = () => { drag = null; draw(); };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("wheel", (e) => {      // wheel over the image: grow / shrink the box by one step
    const b = viewBox(frame);
    if (!b) return;
    e.preventDefault();
    const step = cfg().snap ? 32 : 8;
    kf[frame] = fit((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[3] - b[1]) + (e.deltaY < 0 ? step : -step));
    draw();
  }, { passive: false });

  // ---- timeline
  slider.addEventListener("input", () => { stopPlay(); load(+slider.value); });
  strip.addEventListener("pointerdown", (e) => {
    stopPlay();
    const r = strip.getBoundingClientRect();
    load(Math.round(((e.clientX - r.left) / r.width) * (N - 1)));
  });

  // ---- actions
  const jump = (dir) => {
    const ks = keys();
    const t = dir > 0 ? ks.find((k) => k > frame) : [...ks].reverse().find((k) => k < frame);
    if (t !== undefined) load(t);
  };
  btn("⏮ prev key", "Previous keyframe", () => jump(-1));
  btn("◀", "Previous frame  (←)", () => load(frame - 1));
  btn("▶", "Next frame  (→)", () => load(frame + 1));
  btn("next key ⏭", "Next keyframe", () => jump(1));
  playBtn = btn("▶ Play", "Play the frames like a video (Space)", () => (playing ? stopPlay() : startPlay()), "#1f4a5a");
  fpsSel = document.createElement("select");
  const vfps = Math.round(info.fps || 0) || 24;
  for (const f of [...new Set([6, 12, 24, 30, vfps])].sort((x, y) => x - y)) {
    const o = document.createElement("option"); o.value = f; o.textContent = `${f} fps`; if (f === vfps) o.selected = true; fpsSel.appendChild(o);
  }
  css(fpsSel, { background: "#2c2c32", color: "#eee", border: "1px solid #4a4a52", borderRadius: "4px", padding: "4px" });
  fpsSel.title = "Playback speed in the editor (does not change the video)";
  fpsSel.onchange = () => { if (playing) { stopPlay(); startPlay(); } };
  target.appendChild(fpsSel);
  btn("◆ Set keyframe", "Pin the current (interpolated) box as a keyframe  (K)", () => {
    const b = viewBox(frame); if (b) { kf[frame] = b; draw(); }
  }, "#4a2a5a");
  btn("✕ Delete keyframe", "Remove the keyframe on this frame  (Delete)", () => { delete kf[frame]; draw(); });
  btn("✕ Delete scene keyframes", "Remove the keyframes of the scene around this frame only: from cut to cut, from the video start to the first cut, or from the last cut to the end", () => {
    const [s, e] = shotOf(frame);
    let n = 0;
    for (const k of keys()) if (k >= s && k < e) { delete kf[k]; n++; }
    note = n ? `removed ${n} keyframe(s) of the scene ${s}..${e - 1}` : `no keyframes in the scene ${s}..${e - 1}`;
    draw();
  }, "#6a2020");
  btn("🗑 Delete ALL keyframes", "Remove every keyframe (cuts and absent ranges stay)", () => {
    if (keys().length && confirm(`Delete all ${keys().length} keyframes?`)) { kf = {}; draw(); }
  }, "#6a2020");

  // ---- tracking row: track a box forwards / backwards, or let YOLO enlarge boxes that are too small
  target = rowTrk;
  const lbl = (txt) => { const l = document.createElement("span"); l.textContent = txt; l.style.opacity = ".7"; target.appendChild(l); };
  lbl("Tracker:");
  const trkSel = document.createElement("select");
  for (const n of ["csrt", "kcf", "mil"]) { const o = document.createElement("option"); o.value = o.textContent = n; trkSel.appendChild(o); }
  css(trkSel, { background: "#2c2c32", color: "#eee", border: "1px solid #4a4a52", borderRadius: "4px", padding: "4px" });
  trkSel.title = "csrt = most accurate (needs opencv-contrib, otherwise falls back to the next one); kcf = fast but loses the head when it changes size; mil = always available";
  target.appendChild(trkSel);
  lbl("key every");
  const strideIn = document.createElement("input");
  strideIn.type = "number"; strideIn.min = 1; strideIn.max = 60; strideIn.value = 4;
  css(strideIn, { width: "46px", background: "#2c2c32", color: "#eee", border: "1px solid #4a4a52", borderRadius: "4px", padding: "4px" });
  strideIn.title = "Place a keyframe every N frames of the tracked span (the node interpolates and smooths between them)";
  target.appendChild(strideIn);
  lbl("frames");
  const lockLbl = document.createElement("label");
  const lockCb = document.createElement("input"); lockCb.type = "checkbox"; lockCb.checked = true;
  lockLbl.append(lockCb, " lock size");
  lockLbl.title = "Applies to Track back / both / fwd only. Ticked: every keyframe keeps the size of the box you started from (only the position follows). Unticked: the box grows and shrinks with the head.";
  css(lockLbl, { opacity: ".85", cursor: "pointer" });
  target.appendChild(lockLbl);

  async function runTrack(dir) {
    if (busy) return;
    // the three settings are read NOW, at the click, not when the tracking finishes
    const opts = { tracker: trkSel.value, stride: Math.max(1, parseInt(strideIn.value) || 4), lock: lockCb.checked };
    const b = viewBox(frame);
    if (!b) { alert("Draw a crop box on this frame first (click on the image)."); return; }
    if (isAbsent(frame)) { alert("This frame is inside an 'absent' range."); return; }
    const cf = getW(node, "crop_factor")?.value || 2.5, fr = getW(node, "face_width_ratio")?.value || 0.8;
    const ch = b[3] - b[1], cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
    const fh = ch / cf, fw = fh * fr;                      // the head inside the crop is what gets tracked
    const face = [cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2];
    const [s, e] = shotOf(frame);
    let lo = s, hi = e - 1;                                // never cross a cut or an absent range
    for (const [a, z] of absent) { if (z < frame) lo = Math.max(lo, z + 1); if (a > frame) hi = Math.min(hi, a - 1); }
    if (dir === "fwd") lo = frame;
    if (dir === "back") hi = frame;
    if (lo === hi) { alert("Nothing to track in that direction (cut / absent range / end of video)."); return; }
    const start = frame;
    kf[start] = kf[start] || b;                            // the start box becomes a keyframe
    busy = true; note = `tracking (${opts.tracker}, every ${opts.stride}, size ${opts.lock ? "locked" : "follows the head"})...`; draw();
    try {
      const r = await fetch("/h3manual/track", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...(source.kind === "file" ? { name: up.file } : { node: String(node.id) }), fps: fpsParam,
          start, box: face, lo, hi, direction: dir, tracker: opts.tracker }) });
      if (!r.ok) throw new Error(await r.text());
      const res = await r.json();
      const fs = Object.keys(res.boxes).map(Number).sort((x, y) => x - y);
      if (!fs.length) { note = "tracker lost the target immediately"; return; }
      const a0 = Math.min(start, fs[0]), a1 = Math.max(start, fs[fs.length - 1]);
      for (const k of keys()) if (k >= a0 && k <= a1 && k !== start) delete kf[k];   // replace old keys in the span
      let n = 0;
      for (const f of fs) {
        if ((Math.abs(f - start) % opts.stride) && f !== fs[0] && f !== fs[fs.length - 1]) continue;
        const t2 = res.boxes[f];                           // [x1, y1, x2, y2, scale relative to the start]
        const nh = opts.lock ? ch : ch * Math.min(2.0, Math.max(0.5, t2[4] || 1));
        kf[f] = fit((t2[0] + t2[2]) / 2, (t2[1] + t2[3]) / 2, nh); n++;
      }
      const lost = [res.lost.fwd !== null && res.lost.fwd !== undefined ? `lost forward at ${res.lost.fwd}` : "",
        res.lost.back !== null && res.lost.back !== undefined ? `lost backward at ${res.lost.back}` : ""].filter(Boolean).join(", ");
      note = `${res.tracker}: tracked ${a0}..${a1}, ${n} keyframes` + (lost ? `  (${lost} - fix that frame and track again)` : "");
    } catch (err) {
      note = ""; alert("Tracking failed: " + err.message);
    } finally { busy = false; draw(); }
  }
  btn("◀ Track back", "Track the box backward from this frame, until a cut / absent range / lost target. Uses the tracker, key-every and lock-size values as they are when you click.", () => runTrack("back"), "#1f4a5a");
  btn("Track both ◀▶", "Track the box in both directions from this frame. Uses the tracker, key-every and lock-size values as they are when you click.", () => runTrack("both"), "#1f4a5a");
  btn("Track fwd ▶", "Track the box forward from this frame, until a cut / absent range / lost target. Uses the tracker, key-every and lock-size values as they are when you click.", () => runTrack("fwd"), "#1f4a5a");

  // YOLO Calibrate: look at every keyframe with the face detector and enlarge boxes that are smaller than
  // crop_factor asks for (crop height >= face height x crop_factor). Boxes that are big enough stay as they are.
  const ctrLbl = document.createElement("label");
  const ctrCb = document.createElement("input"); ctrCb.type = "checkbox"; ctrCb.checked = true;
  ctrLbl.append(ctrCb, " center on face");
  ctrLbl.title = "YOLO Calibrate only. Ticked: the crop box is moved so the face YOLO found sits in its centre (as the original pack does). Unticked: the box keeps its centre and is only enlarged when needed.";
  css(ctrLbl, { opacity: ".85", cursor: "pointer" });
  target.appendChild(ctrLbl);

  async function runCalibrate() {
    if (busy) return;
    const center = ctrCb.checked;                               // read now, at the click
    const det = getW(node, "detector")?.value, conf = Number(getW(node, "confidence")?.value ?? 0.35);
    const cf = Number(getW(node, "crop_factor")?.value) || 2.5;
    const items = keys().filter((k) => !isAbsent(k)).map((k) => ({ frame: k, box: kf[k] }));
    if (!items.length) { alert("There are no keyframes to calibrate (keyframes inside absent ranges are skipped)."); return; }
    if (!det) { alert("Choose a face detector in the node's `detector` field first."); return; }
    busy = true; note = `YOLO: checking ${items.length} keyframe(s)...`; draw();
    try {
      const r = await fetch("/h3manual/calibrate", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...(source.kind === "file" ? { name: up.file } : { node: String(node.id) }), fps: fpsParam,
          detector: det, confidence: conf, items }) });
      if (!r.ok) throw new Error(await r.text());
      const res = (await r.json()).results;
      const aspect = cfg().aspect;
      let grown = 0, moved = 0, edge = 0, none = 0, biggest = 1;
      for (const it of res) {
        if (!it.face) { none++; continue; }
        const bx = kf[it.frame];
        if (!bx) continue;
        const o = calibrateBox(bx, it.face, cf, aspect, center, fit);
        kf[it.frame] = o.box;
        if (o.grown) { grown++; biggest = Math.max(biggest, o.ratio); }
        if (o.moved) moved++;
        if (o.notCentered) edge++;
      }
      note = `YOLO Calibrate: ${res.length} keyframe(s): ` + (center ? `${moved} re-centred on the face, ` : "") +
        `${grown} enlarged` + (grown ? ` (up to x${biggest.toFixed(2)})` : "") +
        (edge ? `, ${edge} could not be centred (frame edge)` : "") +
        `, ${none} without a face found` + (none ? " (left as they are)" : "");
    } catch (err) {
      note = ""; alert("YOLO Calibrate failed: " + err.message);
    } finally { busy = false; draw(); }
  }
  btn("YOLO Calibrate", "Checks every keyframe (except those in absent ranges) with the face detector chosen in the node. With 'center on face' the box is moved so the face is in its centre; a box smaller than the face needs (crop height = face height x crop_factor) is enlarged around that centre, a bigger one is never shrunk. Keyframes where no face is found stay as they are. The tracker settings do not apply.", runCalibrate, "#4a3a1a");

  // ---- cuts row
  target = rowCut;
  const jumpCut = (dir) => {
    const cs = [...cuts].sort((a, b) => a - b);
    const tgt = dir > 0 ? cs.find((c) => c > frame) : [...cs].reverse().find((c) => c < frame);
    if (tgt !== undefined) load(tgt);
  };
  btn("⏮ prev cut", "Jump to the previous cut (first frame of that shot)", () => jumpCut(-1));
  btn("next cut ⏭", "Jump to the next cut (first frame of the next shot)", () => jumpCut(1));
  btn("✂ Cut here", "Mark / unmark a hard cut: a new shot starts on this frame", () => {
    if (frame > 0) { cuts.has(frame) ? cuts.delete(frame) : cuts.add(frame); draw(); }
  }, "#5a4a1a");
  btn("🗑 Delete all cuts", "Remove every cut", () => {
    if (cuts.size && confirm(`Delete all ${cuts.size} cuts?`)) { cuts.clear(); draw(); }
  }, "#6a2020");
  const thrIn = document.createElement("input");
  thrIn.type = "number"; thrIn.min = 0.5; thrIn.max = 30; thrIn.step = 0.5; thrIn.value = 3;
  css(thrIn, { width: "52px", background: "#2c2c32", color: "#eee", border: "1px solid #4a4a52", borderRadius: "4px", padding: "4px" });
  thrIn.title = "Sensitivity: how far a frame must stand out from its neighbours to count as a cut. Lower = more cuts. 3.0 is PySceneDetect's default.";
  async function autoCuts() {
    if (busy) return;
    if (cuts.size && !confirm(`Replace the ${cuts.size} existing cut(s) with automatically detected ones?`)) return;
    busy = true; note = "detecting cuts (decodes the whole video)..."; draw();
    try {
      const r = await fetch(`/h3manual/cuts?${source.q}&fps=${fpsParam}&threshold=${encodeURIComponent(thrIn.value || 3)}`);
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();
      cuts.clear();
      for (const c of j.cuts) if (c > 0 && c < N) cuts.add(c);
      note = `${j.method}: ${cuts.size} cut(s) found` + (cuts.size ? ` at ${[...cuts].sort((x, y) => x - y).slice(0, 12).join(", ")}${cuts.size > 12 ? " ..." : ""}` : "");
    } catch (err) { note = ""; alert("Cut detection failed: " + err.message); }
    finally { busy = false; draw(); }
  }
  btn("⚡ Auto-detect cuts", "Find hard cuts automatically (PySceneDetect if installed, otherwise a built-in detector)", autoCuts, "#5a4a1a");
  const l2 = document.createElement("span"); l2.textContent = "sensitivity"; l2.style.opacity = ".7";
  target.append(l2, thrIn);

  // ---- absent row
  target = rowAbs;
  const mergeAbsent = () => {
    absent.sort((x, y) => x[0] - y[0]);
    const out = [];
    for (const r of absent) {
      const l = out[out.length - 1];
      if (l && r[0] <= l[1] + 1) l[1] = Math.max(l[1], r[1]); else out.push([r[0], r[1]]);
    }
    absent = out;
  };
  const jumpAbs = (dir) => {
    const starts = absent.map((r) => r[0]).sort((a, b) => a - b);
    const tgt = dir > 0 ? starts.find((s) => s > frame) : [...starts].reverse().find((s) => s < frame);
    if (tgt !== undefined) load(tgt);
  };
  btn("⏮ prev absent", "Jump to the start of the previous absent range", () => jumpAbs(-1));
  btn("next absent ⏭", "Jump to the start of the next absent range", () => jumpAbs(1));
  const absBtn = btn("Absent: start", "Two clicks: start frame, then end frame of a range where the subject is not visible", () => {
    if (pendingAbsent === null) { pendingAbsent = frame; absBtn.textContent = "Absent: end here"; }
    else {
      absent.push([Math.min(pendingAbsent, frame), Math.max(pendingAbsent, frame)]); mergeAbsent();
      pendingAbsent = null; absBtn.textContent = "Absent: start";
    }
    draw();
  }, "#5a2a2a");
  btn("Absent fragment", "Mark the whole shot around this frame as absent: from cut to cut, or from the video start / to the video end", () => {
    const [s, e] = shotOf(frame);
    absent.push([s, e - 1]); mergeAbsent(); draw();
  }, "#5a2a2a");
  btn("Clear absent here", "Remove the absent range that contains this frame", () => {
    absent = absent.filter(([a, b]) => !(frame >= a && frame <= b)); draw();
  }, "#5a2a2a");
  btn("🗑 Delete all absent", "Remove every absent range", () => {
    if ((absent.length || pendingAbsent !== null) && confirm(`Delete all ${absent.length} absent range(s)?`)) {
      absent = []; pendingAbsent = null; absBtn.textContent = "Absent: start"; draw();
    }
  }, "#6a2020");

  // ---- right side: finish / reset
  const close = () => { stopPlay(); window.removeEventListener("resize", onResize); overlay.remove(); };
  target = sideTop;
  btn("Cancel", "Close without saving", close);
  btn("✔ Use these", "Write keyframes, cuts and absent ranges into the node and close", () => {
    if (!keys().length) { alert("Add at least one keyframe (click on the image to place a box)."); return; }
    setW(node, "keyframes", fmtKeyframes(kf, info.marker, N));
    setW(node, "cut_frames", [...cuts].sort((a, b) => a - b).join(", "));
    setW(node, "absent_ranges", fmtRanges(absent));
    app.graph.setDirtyCanvas(true, true);
    close();
  }, "#1f5a2a");
  target = side;
  const resetBtn = btn("⟲ Reset ALL", "Clear every keyframe, cut and absent range", () => {
    if (confirm("Reset everything: all keyframes, cuts and absent ranges?")) {
      kf = {}; cuts.clear(); absent = []; pendingAbsent = null; absBtn.textContent = "Absent: start"; note = ""; draw();
    }
  }, "#7a1a1a");
  resetBtn.style.fontWeight = "bold";

  overlay.addEventListener("keydown", (e) => {
    const k = e.key;
    if (k === " ") { playing ? stopPlay() : startPlay(); e.preventDefault(); e.stopPropagation(); return; }
    if (k === "ArrowLeft" || k === "ArrowRight") stopPlay();
    if (k === "ArrowLeft") load(frame - (e.shiftKey ? 10 : 1));
    else if (k === "ArrowRight") load(frame + (e.shiftKey ? 10 : 1));
    else if (k === "k" || k === "K") { const b = viewBox(frame); if (b) { kf[frame] = b; draw(); } }
    else if (k === "Delete" || k === "Backspace") { delete kf[frame]; draw(); }
    else if (k === "Escape") close();
    else return;
    e.preventDefault(); e.stopPropagation();
  });
  const onResize = () => { layout(); draw(); };
  window.addEventListener("resize", onResize);

  document.body.appendChild(overlay);
  layout();
  overlay.focus();
  const first = keys()[0] ?? 0;
  load(first);
}

// ---------------------------------------------------------------- register
app.registerExtension({
  name: "h3.FaceTrackManual",
  async setup() {
    // On the fly: the moment another file is chosen in the Load Video node feeding one of our nodes,
    // its keyframes / cuts / absent ranges (made for the previous file) are cleared.
    setInterval(() => {
      for (const n of app.graph?._nodes || []) {
        if (n.type !== NODE) continue;
        const cur = chainKey(n);
        const curFps = Number(getW(n, "fps")?.value) || 0;
        if (n.__h3file === undefined) { n.__h3file = cur; n.__h3fps = curFps; continue; }
        if (cur !== n.__h3file) {                                   // another file: the data no longer fits
          n.__h3file = cur; n.__h3fps = curFps;
          if (hasData(n)) {
            resetNodeData(n);
            const hide = toast("The video (or something in front of it) changed: the previous keyframes, cuts and absent ranges were cleared.", n);
            setTimeout(hide, 20000);
          }
        } else if (curFps !== n.__h3fps) {                           // same file, other frame rate
          const old = splitMarker(videoMarker(getW(n, "keyframes")?.value));
          n.__h3fps = curFps;
          if (!hasData(n)) continue;
          if (old.fps > 0 && curFps > 0) {
            rescaleNodeData(n, curFps / old.fps, curFps);
            const hide = toast(`Frame rate changed to ${curFps}: keyframes, cuts and absent ranges were rescaled.`, n);
            setTimeout(hide, 20000);
          } else {
            resetNodeData(n);
            const hide = toast("Frame rate changed: the previous keyframes were cleared (their frame numbers no longer apply).", n);
            setTimeout(hide, 20000);
          }
        }
      }
    }, 1000);
    api.addEventListener("h3manual.notice", (ev) => {
      const d = ev.detail || {};
      const node = app.graph.getNodeById(Number(d.node));
      if (d.kind === "reset" && node) resetNodeData(node);
      const hide = toast(d.text || "", d.kind === "reset" ? node : null);
      setTimeout(hide, d.kind === "reset" ? 20000 : 8000);
    });
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    // Workflows saved before ruler_frame / ruler_step / crop_snap were removed carry their values at
    // fixed positions; drop those three so the remaining widgets get the right values.
    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      try {
        const w = info && info.widgets_values;
        if (Array.isArray(w) && (w[13] === "multiple_of_32" || w[13] === "off") &&
            ["pad_end", "trim_end", "off"].includes(w[14])) {
          info.widgets_values = w.filter((_, i) => i !== 10 && i !== 11 && i !== 13);
        }
      } catch (e) { /* leave the values alone */ }
      return origConfigure.apply(this, arguments);
    };
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = orig ? orig.apply(this, arguments) : undefined;
      this.addWidget("button", "Edit keyframes (graphic)", null, () => openEditor(this));
      return r;
    };
  },
});

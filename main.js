import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const DECART_SDK_URL = "https://esm.sh/@decartai/sdk@0.1.17";

// Demo mode (?demo): synthetic video + fake landmarks, for testing without a camera.
const DEMO = new URLSearchParams(location.search).has("demo");

const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const INDEX_MCP = 5;

const video = document.getElementById("video");
const lucyVid = document.getElementById("lucy");
const shell = document.getElementById("shell");
const canvas = document.getElementById("canvas");
// `desynchronized` opts into a low-latency present path (skips a compositor
// frame where supported); `alpha` is required now that the camera passthrough
// lives in the video element behind us rather than in the canvas itself.
const ctx = canvas.getContext("2d", { desynchronized: true, alpha: true });
const hudEl = document.getElementById("hud");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const hintEl = document.getElementById("hint");
const toolbar = document.getElementById("toolbar");
const livePill = document.getElementById("live-pill");
const liveText = document.getElementById("live-text");

// Each effect is a live style prompt for Lucy 2.5, phrased per Decart's
// prompt templates ("Change the style of the video to <description>." with
// concrete visual specifics — vague or non-template phrasing degrades output).
const EFFECTS = [
  {
    id: "movie3d",
    label: "3D Movie",
    prompt:
      "Change the style of the video to a 3D animated movie: stylized CGI " +
      "animation, the person as an animated character with expressive big " +
      "eyes and smooth skin, soft cinematic lighting.",
  },
  {
    id: "anime",
    label: "Anime",
    prompt:
      "Change the style of the video to hand-drawn anime: clean black line " +
      "art, flat cel shading, vibrant colors, large expressive eyes.",
  },
  {
    id: "cyberpunk",
    label: "Cyberpunk",
    prompt:
      "Change the style of the video to neon cyberpunk: glowing pink and " +
      "cyan neon light on the person and walls, rain-slick reflective " +
      "surfaces, holographic signs in the background.",
  },
  {
    id: "watercolor",
    label: "Watercolor",
    prompt:
      "Change the style of the video to a watercolor painting: soft loose " +
      "brushstrokes, gentle color bleeds, visible paper texture, muted " +
      "pastel palette.",
  },
  {
    id: "lego",
    label: "LEGO",
    prompt:
      "Change the style of the video to a LEGO stop-motion animation: the " +
      "person is a yellow LEGO minifigure with a cylindrical head, painted " +
      "face, and claw hands, and the room is built entirely from glossy " +
      "plastic LEGO bricks with visible round studs on every surface.",
  },
  { id: "custom", label: "Custom ✨", prompt: null },
];
let effect = "movie3d";

let apiKey =
  localStorage.getItem("decart-key") || sessionStorage.getItem("decart-key") || "";
let customPrompt = localStorage.getItem("lucy-custom") || "";
let realtimeClient = null;
let lucyLive = false;
let cameraStream = null;

// Filtered quad corners ({x, y, vx, vy} per corner) + presence fade (0..1).
let corners = null;
let presence = 0;

// ---- Latency budget ------------------------------------------------------
// One Euro filter (Casiez et al., CHI 2012). A fixed-alpha EMA has to pick one
// point on the jitter/lag curve and live with it; One Euro moves along that
// curve with hand speed — heavy smoothing when still (no visible jitter), and
// the cutoff opens up the instant the hands move (no visible trailing).
const MIN_CUTOFF = 1.1; // Hz, at rest
const CUTOFF_SLOPE = 14; // Hz per screen-width/second of corner speed
const DERIV_CUTOFF = 1.0; // Hz, velocity estimate is noisy — smooth it hard

// Everything the filter can't remove is *transport* lag: the camera frame is
// already ~1 frame old when MediaPipe sees it, inference costs some more, and
// the pixels we draw light up a compositor frame later. That lag is knowable,
// so extrapolate along the filtered velocity to where the corner will be when
// it is actually seen, rather than drawing where it used to be.
const PRESENT_LOOKAHEAD_MS = 8; // draw -> photons
const MAX_EXTRAPOLATE_MS = 55; // never predict further than this
const MAX_EXTRAPOLATE_FRAC = 0.05; // ...nor further than 5% of the screen
// Presence fade, in seconds-to-converge rather than per-frame steps, so a
// 120 Hz display doesn't fade twice as fast as a 60 Hz one.
const PRESENCE_RISE = 0.09;
const PRESENCE_FALL = 0.22;

// Timestamp of the camera frame the current `corners` were measured from, and
// the wall clock when that measurement landed — the extrapolation baseline.
let lastDetectAt = 0;
let lastDrawAt = 0;

// Latency HUD (press L).
let hudOn = false;
const detectMs = { last: 0, avg: 0 };
const drawMs = { last: 0, avg: 0 };
let detectHz = 0;

function ema(acc, v) {
  acc.last = v;
  acc.avg = acc.avg ? acc.avg + (v - acc.avg) * 0.08 : v;
}

// One Euro low-pass coefficient for a given cutoff and timestep.
function lpAlpha(dt, cutoff) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}
// True while a frame is being shown — relaxes the gesture gate (hysteresis).
let frameActive = false;
// Frames since the quad was last seen; short dropouts hold the last quad.
let lostFrames = 0;
// Crossing/overlapping hands often occlude each other and break detection
// for a while — hold the last quad through a moderate dropout window.
const MAX_LOST_FRAMES = 25;
// Frames in a row a far-jumped quad must persist before we accept it as a
// real reposition rather than a mis-detection during hand overlap.
const JUMP_CONFIRM_FRAMES = 2;
let jumpFrames = 0;

let landmarker = null;

function currentPrompt() {
  const e = EFFECTS.find((x) => x.id === effect);
  if (e?.prompt) return e.prompt;
  return (
    customPrompt.trim() ||
    "Transform the person into a 3D animated movie character."
  );
}

function buildToolbar() {
  EFFECTS.forEach((e, i) => {
    const btn = document.createElement("button");
    btn.innerHTML = `<span class="key">${i + 1}</span>${e.label}`;
    btn.dataset.id = e.id;
    if (e.id === effect) btn.classList.add("active");
    btn.addEventListener("click", () => setEffect(e.id));
    toolbar.appendChild(btn);
  });
  window.addEventListener("keydown", (ev) => {
    const idx = parseInt(ev.key, 10) - 1;
    if (idx >= 0 && idx < EFFECTS.length) setEffect(EFFECTS[idx].id);
  });
}

function setEffect(id) {
  effect = id;
  toolbar.querySelectorAll("button").forEach((b) => {
    b.classList.toggle("active", b.dataset.id === id);
  });
  if (id === "custom" && !customPrompt.trim()) {
    document.getElementById("key-panel").classList.remove("hidden");
  }
  pushPrompt();
}

// Send the active style to the live Lucy session (no reconnect needed).
async function pushPrompt() {
  if (!realtimeClient || !lucyLive) return;
  const text = currentPrompt();
  try {
    // SDK versions differ on the exact shape — try the documented forms.
    try {
      await realtimeClient.set({ prompt: text, enhance: true });
    } catch {
      await realtimeClient.set({ prompt: { text }, enhance: true });
    }
  } catch (err) {
    console.error("prompt update failed:", err);
  }
}

function setPill(state, text) {
  livePill.className = state ? `on ${state}` : "";
  if (state) livePill.classList.add("on");
  liveText.textContent = text;
}

// ---- Decart Lucy 2.5 (realtime video-to-video over WebRTC) ----

async function connectLucy() {
  if (!apiKey || !cameraStream || DEMO) return;
  try {
    setPill("connecting", "CONNECTING…");
    const { createDecartClient, models } = await import(DECART_SDK_URL);
    const model = models.realtime("lucy-2.5");
    const client = createDecartClient({ apiKey });
    realtimeClient = await client.realtime.connect(cameraStream, {
      model,
      initialState: { prompt: { text: currentPrompt(), enhance: true } },
      onRemoteStream: (stream) => {
        lucyVid.srcObject = stream;
        lucyVid.play().catch(() => {});
        lucyLive = true;
        setPill("", "LIVE");
      },
    });
    console.log("Lucy connected", realtimeClient);
  } catch (err) {
    console.error("Lucy connect failed:", err);
    lucyLive = false;
    setPill("error", "AI OFFLINE — " + (err.message || "connect failed").slice(0, 60));
  }
}

async function disconnectLucy() {
  lucyLive = false;
  setPill(null, "");
  try {
    await realtimeClient?.disconnect?.();
    realtimeClient?.close?.();
  } catch {}
  realtimeClient = null;
  lucyVid.srcObject = null;
}

function setupKeyPanel() {
  const btn = document.getElementById("key-btn");
  const panel = document.getElementById("key-panel");
  const input = document.getElementById("key-input");
  const remember = document.getElementById("key-remember");
  const custom = document.getElementById("style-custom");

  input.value = apiKey;
  remember.checked = !!localStorage.getItem("decart-key");
  custom.value = customPrompt;

  btn.addEventListener("click", () => panel.classList.toggle("hidden"));
  document.getElementById("key-save").addEventListener("click", async () => {
    apiKey = input.value.trim();
    localStorage.removeItem("decart-key");
    sessionStorage.removeItem("decart-key");
    if (apiKey) {
      (remember.checked ? localStorage : sessionStorage).setItem("decart-key", apiKey);
    }
    customPrompt = custom.value;
    localStorage.setItem("lucy-custom", customPrompt);
    panel.classList.add("hidden");
    await disconnectLucy();
    if (apiKey) connectLucy();
    else pushPrompt();
  });
  document.getElementById("key-clear").addEventListener("click", async () => {
    apiKey = "";
    input.value = "";
    localStorage.removeItem("decart-key");
    sessionStorage.removeItem("decart-key");
    await disconnectLucy();
  });
}

async function init() {
  buildToolbar();
  setupKeyPanel();

  let stream;
  if (DEMO) {
    stream = makeDemoStream();
  } else {
    statusText.textContent = "Loading hand tracker…";
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.3,
      minHandPresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
    });

    statusText.textContent = "Requesting camera…";
    // Lucy 2.5 expects 1280x720 @ 30fps landscape input.
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
        facingMode: "user",
      },
      audio: false,
    });
    cameraStream = stream;
  }
  video.srcObject = stream;
  await new Promise((res) => (video.onloadedmetadata = res));
  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  shell.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;

  statusEl.classList.add("hidden");
  if (apiKey && !DEMO) connectLucy();

  // Detection is driven by the camera, drawing by the display. Coupling them
  // (the old single rAF loop) meant a fresh camera frame could sit unlooked-at
  // for a whole display frame, and that between detections the quad was frozen
  // — on a 120Hz panel, three of every four drawn frames were stale.
  startDetectLoop();
  requestAnimationFrame(drawLoop);

  window.addEventListener("keydown", (ev) => {
    if (ev.key === "l" || ev.key === "L") {
      hudOn = !hudOn;
      hudEl.classList.toggle("hidden", !hudOn);
    }
  });
}

// Run the landmarker once per *camera* frame, as soon as that frame exists.
function startDetectLoop() {
  // requestVideoFrameCallback fires on frame arrival and hands us `mediaTime`,
  // the frame's own presentation timestamp. MediaPipe's VIDEO mode uses that
  // timestamp to drive its internal tracker, so feeding it wall-clock time
  // (as this used to) misreports the frame interval and degrades tracking.
  if (typeof video.requestVideoFrameCallback === "function") {
    const onFrame = (now, meta) => {
      detect(meta.mediaTime * 1000, now);
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
    return;
  }
  // Firefox has no rVFC yet: fall back to polling currentTime off rAF.
  let lastVideoTime = -1;
  const poll = () => {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      detect(video.currentTime * 1000, performance.now());
    }
    requestAnimationFrame(poll);
  };
  requestAnimationFrame(poll);
}

// Draw a (mirrored) source onto any 2d context, filling w x h.
function drawMirrored(c, w, h, src = video) {
  c.save();
  c.translate(w, 0);
  c.scale(-1, 1);
  c.drawImage(src, 0, 0, w, h);
  c.restore();
}

function toPixel(lm) {
  // Mirror x so coordinates match the mirrored canvas.
  return { x: (1 - lm.x) * canvas.width, y: lm.y * canvas.height };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Given landmark sets for exactly two hands, return the 4 frame corners in
// ANATOMICAL order: [left.index, right.index, right.thumb, left.thumb]
// ("left"/"right" = on-screen wrist position). Each corner belongs to a
// specific finger, so the edge cycle is honest geometry: two upright "L"s
// trace a rectangle, and flipping one hand's fingers makes the edges cross
// into a bowtie of two triangles — and uncrossing recovers by itself, since
// nothing about the ordering is stateful.
function computeQuad(hands) {
  const info = hands.map((lm) => ({
    index: toPixel(lm[INDEX_TIP]),
    thumb: toPixel(lm[THUMB_TIP]),
    wristX: toPixel(lm[WRIST]).x,
    // Hand size from wrist -> middle knuckle: stable regardless of which way
    // the fingers point (unlike finger-based measures, which foreshorten).
    scale: dist(toPixel(lm[WRIST]), toPixel(lm[MIDDLE_MCP])) + 1,
  }));
  // Require thumb and index spread apart (an open "L"). Hysteresis: easy to
  // keep once active, so rotating/foreshortening fingers doesn't drop it.
  const needed = frameActive ? 0.2 : 0.75;
  for (const hd of info) {
    if (dist(hd.thumb, hd.index) < hd.scale * needed) return null;
  }
  info.sort((a, b) => a.wristX - b.wristX);
  const [A, B] = info;
  // Standard gesture holds both index fingers up and thumbs down, so this
  // cycle traces a rectangle; flipping one hand crosses it into a bowtie.
  const pts = [A.index, B.index, B.thumb, A.thumb];
  // Degenerate-frame gate on the spanned extent (angle-sorted area) — the
  // traced area is near zero for a legitimate crossed (bowtie) frame.
  const cx = pts.reduce((s, p) => s + p.x, 0) / 4;
  const cy = pts.reduce((s, p) => s + p.y, 0) / 4;
  const hull = [...pts].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  );
  const minArea = frameActive ? 0.0005 : 0.005;
  if (polygonArea(hull) < canvas.width * canvas.height * minArea) return null;
  return pts;
}

function polygonArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a / 2);
}

function quadPath(c, q) {
  c.beginPath();
  c.moveTo(q[0].x, q[0].y);
  for (let i = 1; i < 4; i++) c.lineTo(q[i].x, q[i].y);
  c.closePath();
}

function applyEffect(q) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.save();
  quadPath(ctx, q);
  ctx.clip();
  ctx.globalAlpha = presence;

  if (lucyLive && lucyVid.readyState >= 2) {
    // The live AI stream is a full-frame transform of the same camera feed —
    // draw it mirrored and screen-aligned so the finger frame is a window
    // into the AI world, staying registered as hands move.
    drawMirrored(ctx, w, h, lucyVid);
  } else {
    // Keyless fallback: local color shift so the window still does something.
    ctx.filter = "hue-rotate(140deg) saturate(1.6) contrast(1.1)";
    drawMirrored(ctx, w, h);
    ctx.filter = "none";
    if (!apiKey && !DEMO) {
      const cx = q.reduce((s, p) => s + p.x, 0) / 4;
      const cy = q.reduce((s, p) => s + p.y, 0) / 4;
      ctx.font = `600 ${Math.round(w / 55)}px -apple-system, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.shadowColor = "rgba(0,0,0,0.7)";
      ctx.shadowBlur = 8;
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.fillText("🔑 Add your Decart key for the live AI world", cx, cy);
      ctx.shadowBlur = 0;
    }
  }

  ctx.restore();
  ctx.globalAlpha = 1;
}

function drawFrameOutline(q) {
  const t = performance.now() / 1000;
  ctx.save();
  ctx.globalAlpha = presence;

  quadPath(ctx, q);
  ctx.setLineDash([10, 8]);
  // Marching ants: slide the dash pattern along the outline.
  ctx.lineDashOffset = -t * 40;
  // A wider dark stroke underneath reads the same as a drop shadow against a
  // busy camera feed, without shadowBlur's per-frame software blur pass.
  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(0,0,0,0.45)";
  ctx.stroke();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.lineDashOffset = 0;
  q.forEach((p, i) => {
    const r = 7 + Math.sin(t * 3 + i * 1.5) * 1.5;
    // Soft expanding halo behind each corner dot.
    const halo = (t * 0.8 + i * 0.25) % 1;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r + halo * 14, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,255,255,${0.5 * (1 - halo) * presence})`;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0,0,0,0.25)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
  ctx.restore();
}

// Fold one new measurement of a corner into its filtered position + velocity.
function updateCorner(c, target, dt) {
  const rawVx = (target.x - c.x) / dt;
  const rawVy = (target.y - c.y) / dt;
  const aD = lpAlpha(dt, DERIV_CUTOFF);
  c.vx += (rawVx - c.vx) * aD;
  c.vy += (rawVy - c.vy) * aD;
  // Cutoff rises with speed: still hands get smoothed, moving hands get followed.
  const speed = Math.hypot(c.vx, c.vy) / canvas.width;
  const a = lpAlpha(dt, MIN_CUTOFF + CUTOFF_SLOPE * speed);
  c.x += (target.x - c.x) * a;
  c.y += (target.y - c.y) * a;
}

// Called once per camera frame. `mediaMs` is the frame's own timestamp (what
// MediaPipe wants); `nowMs` is the wall clock (the extrapolation baseline).
function detect(mediaMs, nowMs) {
  const t0 = performance.now();
  let results;
  if (DEMO) {
    results = { landmarks: fakeHands(mediaMs / 1000) };
  } else {
    results = landmarker.detectForVideo(video, mediaMs);
  }
  ema(detectMs, performance.now() - t0);
  if (lastDetectAt) detectHz = 1000 / Math.max(1, nowMs - lastDetectAt);

  const targetQuad =
    results?.landmarks?.length === 2 ? computeQuad(results.landmarks) : null;

  // Seconds since the previous measurement — the filter's true timestep, which
  // is a camera interval and not a display interval.
  const dt = lastDetectAt ? Math.min(0.2, (nowMs - lastDetectAt) / 1000) : 1 / 30;
  lastDetectAt = nowMs;

  if (!targetQuad) {
    // Brief tracking dropout: hold the last quad instead of fading. Bleed
    // velocity off so a held quad coasts to a stop rather than flying away.
    if (corners && ++lostFrames <= MAX_LOST_FRAMES) {
      for (const c of corners) {
        c.vx *= 0.6;
        c.vy *= 0.6;
      }
    } else if (corners) {
      corners = null;
      frameActive = false;
      jumpFrames = 0;
    }
    return;
  }

  if (!corners) {
    lostFrames = 0;
    frameActive = true;
    jumpFrames = 0;
    corners = targetQuad.map((p) => ({ x: p.x, y: p.y, vx: 0, vy: 0 }));
    return;
  }

  const moved = targetQuad.reduce((s, p, i) => s + dist(p, corners[i]), 0) / 4;
  // Only quads that genuinely teleport (≥30% of the screen in one frame,
  // beyond any real hand motion) are treated as suspect mis-detections.
  if (moved > canvas.width * 0.3 && ++jumpFrames < JUMP_CONFIRM_FRAMES) {
    lostFrames++;
    return;
  }
  lostFrames = 0;
  frameActive = true;
  jumpFrames = 0;
  corners.forEach((c, i) => updateCorner(c, targetQuad[i], dt));
}

// Where a corner will be when the pixels we are about to draw actually appear.
function extrapolate(c, aheadS) {
  let dx = c.vx * aheadS;
  let dy = c.vy * aheadS;
  // A constant-velocity model overshoots hard on direction reversal, so cap
  // how far it is ever allowed to run ahead of the last real measurement.
  const cap = canvas.width * MAX_EXTRAPOLATE_FRAC;
  const mag = Math.hypot(dx, dy);
  if (mag > cap) {
    dx *= cap / mag;
    dy *= cap / mag;
  }
  return { x: c.x + dx, y: c.y + dy };
}

function drawLoop(now) {
  const t0 = performance.now();
  const dt = lastDrawAt ? Math.min(0.1, (now - lastDrawAt) / 1000) : 1 / 60;
  lastDrawAt = now;

  const visible = corners && lostFrames <= MAX_LOST_FRAMES;
  // Exponential approach with a time constant, so the fade is the same wall
  // clock duration at 60Hz and 120Hz (per-frame steps were not).
  const tau = visible ? PRESENCE_RISE : PRESENCE_FALL;
  presence += ((visible ? 1 : 0) - presence) * (1 - Math.exp(-dt / tau));
  if (!visible && presence < 0.01) {
    presence = 0;
    corners = null;
  }

  // The canvas is now overlay-only — the camera passthrough is the video
  // element behind it, composited without a round trip through here.
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (corners && presence > 0.01) {
    const aheadS = Math.min(
      MAX_EXTRAPOLATE_MS,
      now - lastDetectAt + PRESENT_LOOKAHEAD_MS
    ) / 1000;
    const quad = corners.map((c) => extrapolate(c, aheadS));
    applyEffect(quad);
    drawFrameOutline(quad);
  }

  hintEl.classList.toggle("hidden", presence > 0.5);
  ema(drawMs, performance.now() - t0);
  if (hudOn) {
    hudEl.textContent =
      `detect ${detectMs.avg.toFixed(1)}ms @ ${detectHz.toFixed(0)}Hz   ` +
      `draw ${drawMs.avg.toFixed(2)}ms   ` +
      `predicted ${Math.max(0, now - lastDetectAt).toFixed(0)}ms ahead`;
  }

  requestAnimationFrame(drawLoop);
}

// ---- Demo mode helpers ----

function makeDemoStream() {
  const demo = document.createElement("canvas");
  demo.width = 1280;
  demo.height = 720;
  const d = demo.getContext("2d");
  function paint() {
    const t = performance.now() / 1000;
    const g = d.createLinearGradient(0, 0, demo.width, demo.height);
    g.addColorStop(0, "#1c2a4a");
    g.addColorStop(1, "#3a1c4a");
    d.fillStyle = g;
    d.fillRect(0, 0, demo.width, demo.height);
    for (let i = 0; i < 6; i++) {
      const x = demo.width * (0.15 + 0.14 * i) + Math.sin(t * 0.8 + i) * 60;
      const y = demo.height * 0.5 + Math.cos(t * 0.6 + i * 1.7) * 160;
      d.beginPath();
      d.arc(x, y, 50 + 18 * Math.sin(t + i), 0, Math.PI * 2);
      d.fillStyle = `hsl(${(i * 60 + t * 30) % 360}, 75%, 62%)`;
      d.fill();
    }
    d.fillStyle = "rgba(255,255,255,0.9)";
    d.font = "bold 56px sans-serif";
    d.textAlign = "center";
    // Draw mirrored so it reads correctly after the canvas flips it back.
    d.save();
    d.translate(demo.width, 0);
    d.scale(-1, 1);
    d.fillText("DEMO FEED", demo.width / 2, demo.height / 2);
    d.restore();
    requestAnimationFrame(paint);
  }
  paint();
  return demo.captureStream(30);
}

function fakeHand(indexTip, thumbTip, indexMcp) {
  const lm = Array.from({ length: 21 }, () => ({ ...indexMcp, z: 0 }));
  lm[INDEX_TIP] = { ...indexTip, z: 0 };
  lm[THUMB_TIP] = { ...thumbTip, z: 0 };
  lm[INDEX_MCP] = { ...indexMcp, z: 0 };
  return lm;
}

function fakeHands(t) {
  const ox = Math.sin(t * 0.9) * 0.02;
  const oy = Math.cos(t * 0.7) * 0.02;
  return [
    fakeHand(
      { x: 0.74 + ox, y: 0.26 + oy },
      { x: 0.8 + ox, y: 0.56 + oy },
      { x: 0.75 + ox, y: 0.4 + oy }
    ),
    fakeHand(
      { x: 0.26 - ox, y: 0.28 - oy },
      { x: 0.2 - ox, y: 0.58 - oy },
      { x: 0.25 - ox, y: 0.44 - oy }
    ),
  ];
}

init().catch((err) => {
  console.error(err);
  statusEl.classList.remove("hidden");
  statusEl.querySelector(".spinner")?.remove();
  statusText.textContent =
    err.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access and reload."
      : `Failed to start: ${err.message}`;
});

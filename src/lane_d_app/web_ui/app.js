/* PathSnap evaluator UI (Laptop 2).
 *
 * Talks to the FastAPI server on Laptop 1. Supports the team API
 * (GET /trips -> {student_test, evaluator_test}, GET /evaluate/{id}) and the
 * lightweight dev API (GET /trips -> {trips: [...]}, POST /predict).
 *
 * API address: ?api=... URL param > saved value > same host on :8000.
 * The UI never receives reference routes: only GPS input, the predicted
 * route and metric values computed on the server.
 */
(() => {
  "use strict";

  if (typeof L === "undefined") {
    const msg = document.createElement("div");
    msg.className = "toast err";
    msg.textContent = "The map library could not load. Laptop 2 needs internet access for Leaflet and the map tiles.";
    document.body.append(msg);
    return;
  }

  // ------------------------------------------------------------------ formatting
  const STORE_KEY = "pathsnap.apiUrl";
  const REQUEST_TIMEOUT_MS = 300000;
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const int = (v) => Math.round(v).toLocaleString();
  const dec3 = (v) => v.toFixed(3);
  const dist = (v) => (v >= 1000 ? `${(v / 1000).toFixed(2)} km` : `${Math.round(v)} m`);
  const dur = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

  // Display order and formatting for known metrics. Unknown numeric metrics are appended.
  const METRICS = [
    ["route_continuity", "Route continuity", pct],
    ["macro_f1", "Macro F1", dec3],
    ["log_loss", "Log loss", dec3],
    ["candidate_miss_rate", "Candidate miss rate", pct],
    ["top1_accuracy_undirected", "Top-1 (any direction)", pct],
    ["label_coverage", "Labelled points", pct],
    ["reference_coverage", "Reference coverage", pct],
    ["n_points", "GPS points", int],
    ["n_unmatched", "Unmatched points", int],
    ["n_route_arcs", "Route segments", int],
    ["route_length_m", "Route length", dist],
    ["inference_ms", "Inference time", dur],
  ];
  const FRACTION_KEYS = new Set(["top1_accuracy", "route_continuity", "candidate_miss_rate", "top1_accuracy_undirected", "label_coverage", "reference_coverage", "macro_f1"]);
  const HIDDEN = new Set(["reference_available", "top1_accuracy", "inference_time_sec"]);

  // ------------------------------------------------------------------ state
  const state = {
    api: resolveApi(),
    flavor: null,            // "team" | "dev"
    partitions: {},          // partition -> [{trip_id, ...}]
    partition: null,
    selected: null,
    running: false,
    runs: new Map(),         // key "partition/trip" -> run record
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const trips = () => state.partitions[state.partition] || [];
  const runKey = (id) => `${state.partition}/${id}`;

  function resolveApi() {
    const q = new URLSearchParams(location.search).get("api");
    if (q) return q.replace(/\/+$/, "");
    try { const s = localStorage.getItem(STORE_KEY); if (s) return s; } catch (_) { /* unavailable */ }
    return `http://${location.hostname || "localhost"}:8000`;
  }

  async function request(path, opts = {}) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), opts.timeout || REQUEST_TIMEOUT_MS);
    try {
      const res = await fetch(state.api + path, {
        ...opts, signal: ctrl.signal,
        headers: opts.body ? { "Content-Type": "application/json" } : undefined,
      });
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`;
        try { detail = (await res.json()).detail || detail; } catch (_) { /* not json */ }
        throw new Error(detail);
      }
      return { data: await res.json(), requestId: res.headers.get("X-Request-ID") };
    } catch (e) {
      if (e.name === "AbortError") throw new Error("The API did not answer in time.");
      if (e instanceof TypeError) throw new Error(`Cannot reach the API at ${state.api}`);
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }

  // ------------------------------------------------------------------ toast
  let toastTimer;
  function toast(msg, isErr = false) {
    const t = $("toast");
    t.textContent = msg;
    t.className = "toast" + (isErr ? " err" : "");
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), isErr ? 6000 : 3000);
  }

  // ------------------------------------------------------------------ connection
  async function checkHealth() {
    $("connHost").textContent = state.api.replace(/^https?:\/\//, "");
    try {
      const { data } = await request("/health", { timeout: 6000 });
      const ok = ["ok", "healthy"].includes(String(data.status).toLowerCase());
      $("connDot").className = "dot " + (ok ? "ok" : "bad");
      $("connText").textContent = ok ? "Connected" : "API unhealthy";
      const model = data.model?.name;
      $("connInfo").textContent = model ? `Model: ${model} v${data.model.version}` : (data.service || "API reachable");
      if (model) { $("modelTag").textContent = model; $("modelTag").hidden = false; }
      return ok;
    } catch (e) {
      $("connDot").className = "dot bad";
      $("connText").textContent = "Offline";
      $("connInfo").textContent = e.message;
      return false;
    }
  }

  function setupConnPanel() {
    const btn = $("connBtn"), panel = $("connPanel"), input = $("apiUrl");
    input.value = state.api;
    const close = () => { panel.hidden = true; btn.setAttribute("aria-expanded", "false"); };
    btn.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      btn.setAttribute("aria-expanded", String(!panel.hidden));
      if (!panel.hidden) input.focus();
    });
    document.addEventListener("click", (e) => { if (!panel.hidden && !panel.contains(e.target) && !btn.contains(e.target)) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) { close(); btn.focus(); } });
    panel.addEventListener("submit", async (e) => {
      e.preventDefault();
      const v = input.value.trim().replace(/\/+$/, "");
      if (!v) return;
      state.api = v;
      try { localStorage.setItem(STORE_KEY, v); } catch (_) { /* ignore */ }
      if (await checkHealth()) { close(); toast("Connected to Laptop 1"); loadTrips(); }
      else toast(`Cannot reach ${v}`, true);
    });
  }

  // ------------------------------------------------------------------ trips
  async function loadTrips() {
    try {
      const { data } = await request("/trips");
      if (Array.isArray(data.trips)) {
        state.flavor = "dev";
        state.partitions = { trips: data.trips };
      } else {
        state.flavor = "team";
        state.partitions = {};
        for (const [part, ids] of Object.entries(data)) {
          if (Array.isArray(ids)) state.partitions[part] = ids.map((id) => ({ trip_id: id }));
        }
      }
      const parts = Object.keys(state.partitions).filter((p) => state.partitions[p].length);
      if (!parts.includes(state.partition)) {
        state.partition = parts.includes("evaluator_test") ? "evaluator_test" : parts[0] || null;
      }
    } catch (e) {
      state.partitions = {};
      toast(e.message, true);
    }
    renderTabs();
    renderTrips();
    updateButtons();
  }

  const PART_LABEL = { evaluator_test: "Evaluator", student_test: "Student test", trips: "Trips" };
  function renderTabs() {
    const tabs = $("partitionTabs");
    const parts = Object.keys(state.partitions).filter((p) => state.partitions[p].length);
    tabs.replaceChildren();
    tabs.hidden = parts.length < 2;
    parts.forEach((p) => {
      const b = el("button", "tab", `${PART_LABEL[p] || p} (${state.partitions[p].length})`);
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(p === state.partition));
      b.addEventListener("click", () => { state.partition = p; state.selected = null; renderTabs(); renderTrips(); updateButtons(); });
      tabs.append(b);
    });
  }

  function fmtMinutes(s) {
    const m = Math.round(s / 60);
    return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`;
  }

  function renderTrips() {
    const list = $("tripList");
    list.replaceChildren();
    const ts = trips();
    $("tripCount").textContent = ts.length;
    if (!ts.length) {
      list.append(el("li", "trip-meta", "No trips found. Check the API address in the top right."));
      renderSession();
      return;
    }
    ts.forEach((t, i) => {
      const li = el("li", "trip");
      li.setAttribute("role", "option");
      li.tabIndex = 0;
      li.dataset.id = t.trip_id;
      li.setAttribute("aria-selected", String(state.selected === t.trip_id));
      const body = el("div");
      body.append(el("div", "trip-id", t.trip_id));
      const run = state.runs.get(runKey(t.trip_id));
      const meta = t.n_points != null
        ? `${int(t.n_points)} pts · ${fmtMinutes(t.duration_s)} · ${dist(t.gps_distance_m)}`
        : run?.metrics.n_points != null ? `${int(run.metrics.n_points)} pts` : "";
      if (meta) body.append(el("div", "trip-meta", meta));
      const score = run?.metrics.top1_accuracy;
      li.append(
        el("span", "trip-idx", String(i + 1).padStart(2, "0")),
        body,
        el("span", "trip-score" + (score != null ? "" : " pending"), score != null ? pct(score) : "-")
      );
      li.addEventListener("click", () => selectTrip(t.trip_id));
      li.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectTrip(t.trip_id); }
        if (e.key === "ArrowDown") { e.preventDefault(); li.nextElementSibling?.focus(); }
        if (e.key === "ArrowUp") { e.preventDefault(); li.previousElementSibling?.focus(); }
      });
      list.append(li);
    });
    renderSession();
  }

  function selectTrip(id) {
    state.selected = id;
    document.querySelectorAll(".trip").forEach((n) => n.setAttribute("aria-selected", String(n.dataset.id === id)));
    updateButtons();
    const t = trips().find((x) => x.trip_id === id);
    if (t?.center) map.flyTo(t.center, 13, { duration: 0.6 });
  }

  function updateButtons() {
    $("runBtn").disabled = !state.selected || state.running;
    $("runLabel").textContent = state.running ? "Running" : state.selected ? `Run trip ${state.selected}` : "Select a trip";
    $("runAllBtn").disabled = state.running || !trips().length;
    $("runAllBtn").textContent = trips().length ? `Run all ${trips().length} trips` : "Run all trips";
  }

  // ------------------------------------------------------------------ map
  const map = L.map("map", { preferCanvas: true, zoomControl: false }).setView([30, 10], 2);
  L.control.zoom({ position: "topright" }).addTo(map);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, className: "base-tiles",
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  map.createPane("routePane").style.zIndex = 420;
  map.createPane("gpsPane").style.zIndex = 410;
  const routeRenderer = L.canvas({ pane: "routePane" });
  const gpsRenderer = L.canvas({ pane: "gpsPane" });
  const layers = { route: L.layerGroup().addTo(map), gps: L.layerGroup().addTo(map), ends: L.layerGroup().addTo(map) };
  [["tgRoute", "route"], ["tgGps", "gps"], ["tgEnds", "ends"]].forEach(([id, key]) =>
    $(id).addEventListener("change", (e) => (e.target.checked ? layers[key].addTo(map) : layers[key].remove())));

  function drawResult(r) {
    Object.values(layers).forEach((l) => l.clearLayers());
    const gps = r.gps || [];
    gps.forEach((p, i) => L.circleMarker(p, {
      renderer: gpsRenderer, radius: 2.6, weight: 0, fillColor: "#2F6FB3", fillOpacity: 0.45,
    }).bindTooltip(`GPS #${i}`, { direction: "top" }).addTo(layers.gps));
    if (r.route.length) {
      L.polyline(r.route, { renderer: routeRenderer, color: "#ffffff", weight: 8, opacity: 0.9, lineCap: "round", lineJoin: "round" }).addTo(layers.route);
      L.polyline(r.route, { renderer: routeRenderer, color: "#D9480F", weight: 4, opacity: 0.95, lineCap: "round", lineJoin: "round" })
        .bindTooltip("Predicted route", { sticky: true }).addTo(layers.route);
    }
    if (gps.length) {
      const pin = (cls) => L.divIcon({ className: "", html: `<div class="pin ${cls}"></div>`, iconSize: [14, 14], iconAnchor: [7, 7] });
      L.marker(gps[0], { icon: pin("start"), keyboard: false }).bindTooltip("Start").addTo(layers.ends);
      L.marker(gps[gps.length - 1], { icon: pin("end"), keyboard: false }).bindTooltip("End").addTo(layers.ends);
      map.fitBounds(L.latLngBounds(gps), { padding: [40, 40], maxZoom: 17 });
    } else if (r.route.length) {
      map.fitBounds(L.latLngBounds(r.route.flat()), { padding: [40, 40], maxZoom: 17 });
    }
  }

  // ------------------------------------------------------------------ results
  function normalize(data) {
    const raw = { ...(data.metrics || {}) };
    const m = {};
    for (const [k, v] of Object.entries(raw)) {
      if (typeof v !== "number" && v !== null) continue;
      m[k] = v;
    }
    // Team API reports continuity in percent; everything else is a 0-1 fraction.
    for (const k of FRACTION_KEYS) if (typeof m[k] === "number" && m[k] > 1.0001) m[k] = m[k] / 100;
    if (typeof m.inference_time_sec === "number" && m.inference_ms == null) m.inference_ms = m.inference_time_sec * 1000;
    if (m.n_points == null && typeof data.num_gps_points === "number") m.n_points = data.num_gps_points;
    if (m.n_route_arcs == null && Array.isArray(data.reconstructed_route)) m.n_route_arcs = data.reconstructed_route.length;
    return {
      trip_id: data.trip_id,
      gps: data.gps || [],
      route: data.predicted_route || [],
      model: data.model ? `${data.model.name} v${data.model.version}` : "",
      metrics: m,
    };
  }

  function renderResults(r) {
    $("resEmpty").hidden = true;
    $("resBody").hidden = false;
    if (r.model) { $("modelTag").textContent = r.model; $("modelTag").hidden = false; }
    const top1 = r.metrics.top1_accuracy;
    $("heroVal").textContent = typeof top1 === "number" ? pct(top1) : "N/A";
    $("heroBar").style.width = typeof top1 === "number" ? `${Math.max(0, Math.min(1, top1)) * 100}%` : "0";

    const kpis = $("kpis");
    kpis.replaceChildren();
    const shown = new Set();
    const add = (key, label, fmt) => {
      if (!(key in r.metrics) || HIDDEN.has(key)) return;
      shown.add(key);
      const v = r.metrics[key];
      const box = el("div", "kpi");
      box.append(el("dt", "", label), el("dd", v == null ? "na" : "", v == null ? "N/A" : fmt(v)));
      kpis.append(box);
    };
    METRICS.forEach(([k, label, fmt]) => add(k, label, fmt));
    for (const k of Object.keys(r.metrics)) {
      if (!shown.has(k) && !HIDDEN.has(k)) add(k, k.replace(/_/g, " "), (v) => (Number.isInteger(v) ? int(v) : dec3(v)));
    }
  }

  function renderSession() {
    const rows = [...state.runs.entries()].filter(([k]) => k.startsWith(`${state.partition}/`));
    $("sessHint").textContent = `${rows.length} / ${trips().length}`;
    $("csvBtn").disabled = !state.runs.size;
    const body = $("sessRows");
    body.replaceChildren();
    if (!rows.length) {
      const tr = el("tr", "empty-row");
      const td = el("td", "", "No runs yet");
      td.colSpan = 4;
      tr.append(td);
      body.append(tr);
      $("sessFoot").hidden = true;
      return;
    }
    rows.sort((a, b) => b[1].at - a[1].at).forEach(([, run]) => {
      const tr = el("tr");
      const m = run.metrics;
      tr.append(
        el("td", "", run.trip_id),
        el("td", "num", m.top1_accuracy != null ? pct(m.top1_accuracy) : "N/A"),
        el("td", "num", m.route_continuity != null ? pct(m.route_continuity) : "N/A"),
        el("td", "num", dur(run.ms))
      );
      tr.title = run.requestId ? `Request ${run.requestId}` : "";
      tr.addEventListener("click", () => selectTrip(run.trip_id));
      body.append(tr);
    });
    const mean = (f) => {
      const v = rows.map(([, r]) => f(r)).filter((x) => typeof x === "number");
      return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
    };
    const t1 = mean((r) => r.metrics.top1_accuracy), c = mean((r) => r.metrics.route_continuity), t = mean((r) => r.ms);
    $("meanTop1").textContent = t1 != null ? pct(t1) : "N/A";
    $("meanCont").textContent = c != null ? pct(c) : "N/A";
    $("meanTime").textContent = t != null ? dur(t) : "";
    $("sessFoot").hidden = rows.length < 2;
  }

  // ------------------------------------------------------------------ run
  let tick;
  function beginBusy(text) {
    state.running = true;
    updateButtons();
    $("emptyMap").hidden = true;
    $("progress").hidden = false;
    $("progressText").textContent = text;
    $("mapRegion").setAttribute("aria-busy", "true");
    const t0 = performance.now();
    clearInterval(tick);
    tick = setInterval(() => ($("progressTime").textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`), 100);
  }
  function endBusy() {
    clearInterval(tick);
    state.running = false;
    $("progress").hidden = true;
    $("mapRegion").setAttribute("aria-busy", "false");
    updateButtons();
  }

  async function runTrip(id) {
    const t0 = performance.now();
    const { data, requestId } = state.flavor === "team"
      ? await request(`/evaluate/${encodeURIComponent(id)}?partition=${encodeURIComponent(state.partition)}`)
      : await request("/predict", { method: "POST", body: JSON.stringify({ trip_id: id }) });
    const ms = performance.now() - t0;
    const r = normalize(data);
    $("mapTitle").hidden = false;
    $("mapTrip").textContent = r.trip_id;
    drawResult(r);
    renderResults(r);
    state.runs.set(runKey(id), { trip_id: id, partition: state.partition, ms, at: Date.now(), requestId, model: r.model, metrics: r.metrics });
    renderTrips();
    return ms;
  }

  async function runSelected() {
    if (!state.selected || state.running) return;
    const id = state.selected;
    beginBusy(`Processing trip ${id} on Laptop 1`);
    try { toast(`Trip ${id} done in ${dur(await runTrip(id))}`); }
    catch (e) { toast(e.message, true); if (!state.runs.size) $("emptyMap").hidden = false; }
    finally { endBusy(); }
  }

  async function runAll() {
    if (state.running || !trips().length) return;
    const ids = trips().map((t) => t.trip_id);
    let ok = 0;
    for (let i = 0; i < ids.length; i++) {
      beginBusy(`Trip ${i + 1} of ${ids.length} (${ids[i]})`);
      state.selected = ids[i];
      document.querySelectorAll(".trip").forEach((n) => n.setAttribute("aria-selected", String(n.dataset.id === ids[i])));
      try { await runTrip(ids[i]); ok++; } catch (e) { toast(`${ids[i]}: ${e.message}`, true); }
    }
    endBusy();
    toast(ok === ids.length ? `All ${ok} trips processed` : `${ok} of ${ids.length} trips processed`, ok !== ids.length);
  }

  function exportCsv() {
    if (!state.runs.size) return;
    const rows = [...state.runs.values()].sort((a, b) => (a.partition + a.trip_id).localeCompare(b.partition + b.trip_id));
    const keys = [...new Set(rows.flatMap((r) => Object.keys(r.metrics)))];
    const esc = (v) => { const s = v == null ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const lines = [["partition", "trip_id", "timestamp", "request_id", "model", "round_trip_ms", ...keys].join(",")]
      .concat(rows.map((r) => [r.partition, r.trip_id, new Date(r.at).toISOString(), r.requestId || "", r.model, r.ms.toFixed(1),
        ...keys.map((k) => r.metrics[k])].map(esc).join(",")));
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" })),
      download: `pathsnap_session_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`,
    });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  $("runBtn").addEventListener("click", runSelected);
  $("runAllBtn").addEventListener("click", runAll);
  $("csvBtn").addEventListener("click", exportCsv);
  document.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runSelected(); });

  // ------------------------------------------------------------------ boot
  setupConnPanel();
  checkHealth().then(loadTrips).then(() => {
    const q = new URLSearchParams(location.search);
    if (q.get("partition") && state.partitions[q.get("partition")]) { state.partition = q.get("partition"); renderTabs(); renderTrips(); updateButtons(); }
    if (q.get("all") === "1") return runAll();
    const want = q.get("trip");
    if (want && trips().some((t) => t.trip_id === want)) { selectTrip(want); runSelected(); }
  });
  setInterval(checkHealth, 15000);
})();

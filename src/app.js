(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const f3 = (v) => { const s = (+v).toFixed(3); return s === "-0.000" ? "0.000" : s; };
  const f2 = (v) => { const s = (+v).toFixed(2); return s === "-0.00" ? "0.00" : s; };
  const fe = (v) => (v === 0 ? "0" : Math.abs(v) < 1e-3 ? v.toExponential(1) : f3(v));
  const par = (v) => (v < 0 ? "(" + f3(v) + ")" : f3(v));
  const cssv = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- data
  const SIZES = [13, 16, 12, 8, 4, 1];
  const NPARAM = SIZES.slice(0, -1).reduce((s, n, k) => s + n * SIZES[k + 1] + SIZES[k + 1], 0);
  const FEATS = [
    ["age", "Age", "years"], ["sex", "Sex", "1 = male, 0 = female"], ["cp", "Chest-pain type", "0–3"],
    ["trestbps", "Resting blood pressure", "mm Hg"], ["chol", "Serum cholesterol", "mg/dl"],
    ["fbs", "Fasting blood sugar > 120", "1 = yes"], ["restecg", "Resting ECG result", "0–2"],
    ["thalach", "Max heart rate reached", "bpm"], ["exang", "Angina during exercise", "1 = yes"],
    ["oldpeak", "ST depression after exercise", "mm"], ["slope", "Slope of peak ST segment", "0–2"],
    ["ca", "Major vessels on fluoroscopy", "0–4"], ["thal", "Thalassemia test result", "0–3"],
  ];
  const rows = HEART_CSV.trim().split("\n").slice(1).map((l) => l.split(",").map(Number));
  const order = rows.map((_, i) => i);
  { const r = Core.rng(42); for (let i = order.length - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)); [order[i], order[k]] = [order[k], order[i]]; } }
  const trIdx = order.slice(0, 242), teIdx = order.slice(242);
  const MU = Array(13).fill(0), SD = Array(13).fill(0);
  trIdx.forEach((i) => rows[i].slice(0, 13).forEach((v, j) => (MU[j] += v / trIdx.length)));
  trIdx.forEach((i) => rows[i].slice(0, 13).forEach((v, j) => (SD[j] += (v - MU[j]) ** 2 / trIdx.length)));
  for (let j = 0; j < 13; j++) SD[j] = Math.sqrt(SD[j]);
  const std = (i) => Float64Array.from(rows[i].slice(0, 13), (v, j) => (v - MU[j]) / SD[j]);
  const Xtr = trIdx.map(std), Ytr = trIdx.map((i) => rows[i][13]);
  const Xte = teIdx.map(std), Yte = teIdx.map((i) => rows[i][13]);

  // ---------------------------------------------------------------- state
  const S = {
    patient: 0, weights: "init", act: "relu", init: "he",
    layer: 0, neuron: 0, bnLayer: 0, bnNeuron: 0, gamma: 1, beta: 0,
    dropLayer: 0, dropP: 0.3, dropTrain: true, dropSeed: 3,
    bpLayer: 1, bpNeuron: 0, lrUpd: 0.1,
  };
  let netInit, netTrained;
  const TRAINED_EPOCHS = 15;
  function buildNets() {
    netInit = Core.create(SIZES, { init: S.init, act: S.act, seed: 7 });
    netTrained = Core.clone(netInit);
    const opt = Core.makeOptimizer("adam", 0.01), r = Core.rng(1);
    for (let e = 0; e < TRAINED_EPOCHS; e++) Core.trainEpoch(netTrained, opt, Xtr, Ytr, 32, r);
  }
  const net = () => (S.weights === "trained" ? netTrained : netInit);
  const xP = () => Xte[S.patient];
  const yP = () => Yte[S.patient];
  const fwP = () => Core.forward(net(), [xP()], { train: false });
  const LAST = SIZES.length - 2;
  const inName = (l, i) => (l === 0 ? FEATS[i][0] : `h${l}·${i + 1}`);
  const outName = (l, j) => (l === LAST ? "ŷ" : `h${l + 1}·${j + 1}`);
  const layerLabel = (l) => (l === LAST ? `Layer ${l + 1} · output (${SIZES[l]} → 1)` : `Layer ${l + 1} · hidden ${l + 1} (${SIZES[l]} → ${SIZES[l + 1]})`);
  const actOf = (l) => (l === LAST ? { name: "sigmoid", f: Core.sigmoid, d: Core.ACT.sigmoid.d } : Core.ACT[S.act]);

  // ---------------------------------------------------------------- svg helpers
  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const clear = (svg) => { while (svg.firstChild) svg.removeChild(svg.firstChild); };
  const vb = (svg) => [+svg.viewBox.baseVal.width, +svg.viewBox.baseVal.height];
  const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  function niceTicks(a, b, n = 5) {
    const span = b - a || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function frame(svg, W, H, m, xd, yd, opt = {}) {
    const x = lin(xd[0], xd[1], m.l, W - m.r);
    const y = opt.ylog ? (v) => lin(Math.log10(yd[0]), Math.log10(yd[1]), H - m.b, m.t)(Math.log10(Math.max(v, yd[0]))) : lin(yd[0], yd[1], H - m.b, m.t);
    const yt = opt.ylog ? niceTicks(Math.log10(yd[0]), Math.log10(yd[1]), 5).filter((v) => Number.isInteger(v)).map((v) => 10 ** v) : niceTicks(yd[0], yd[1], opt.yn || 4);
    yt.forEach((v) => {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid" }, svg);
      el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, svg, opt.ylog ? "1e" + Math.round(Math.log10(v)) : (opt.yfmt || ((t) => +t.toFixed(3)))(v));
    });
    (opt.xticks || niceTicks(xd[0], xd[1], opt.xn || 5)).forEach((v) => el("text", { x: x(v), y: H - m.b + 15, "text-anchor": "middle" }, svg, opt.xfmt ? opt.xfmt(v) : +v.toFixed(3)));
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "ax" }, svg);
    el("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "ax" }, svg);
    if (opt.xlabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle" }, svg, opt.xlabel);
    if (opt.ylabel) el("text", { x: 11, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 11 ${(m.t + H - m.b) / 2})` }, svg, opt.ylabel);
    return { x, y };
  }
  const path = (pts) => pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join("");
  const kpis = (id, items) => { $(id).innerHTML = items.map(([b, s]) => `<div class="kpi"><b>${b}</b><span>${s}</span></div>`).join(""); };


  // ---------------------------------------------------------------- live TeX
  const texQ = new Set(); let texT = null;
  function flushTex() {
    if (texT) return;
    texT = setTimeout(() => {
      texT = null;
      const MJ = window.MathJax;
      if (!MJ || !MJ.typesetPromise) { setTimeout(flushTex, 250); return; }
      const els = [...texQ]; texQ.clear();
      if (MJ.typesetClear) MJ.typesetClear(els);
      MJ.typesetPromise(els).catch(() => {});
    }, 40);
  }
  function tex(id, s) { const e = $(id); if (!e) return; e.textContent = "\\[" + s + "\\]"; texQ.add(e); flushTex(); }
  const tn = (v) => { if (v === 0) return "0"; if (Math.abs(v) < 1e-3) { const [m, ex] = v.toExponential(1).split("e"); return `${m}\\times 10^{${+ex}}`; } return f3(v); };
  const tp = (v) => (v < 0 ? `(${tn(v)})` : tn(v));
  const actTex = (l) => (l === LAST ? "\\sigma" : { relu: "\\mathrm{ReLU}", leaky: "\\mathrm{LeakyReLU}", tanh: "\\tanh", sigmoid: "\\sigma" }[S.act]);
  const initTex = (fi, fo) => ({ he: `\\sqrt{2/${fi}}`, xavier: `\\sqrt{2/(${fi}+${fo})}`, small: "0.01", zeros: "0" }[S.init]);
  // ---------------------------------------------------------------- plotly helpers
  const hasPlotly = () => typeof Plotly !== "undefined";
  function plotBase() {
    const panel = cssv("--panel"), muted = cssv("--muted"), rule = cssv("--rule");
    return { panel, muted, rule, ink: cssv("--ink"), fwd: cssv("--fwd"), bwd: cssv("--bwd"), soft: cssv("--soft"), hl: cssv("--hl"), pos: cssv("--pos"), neg: cssv("--neg"), font: { family: "JetBrains Mono, monospace", size: 11, color: muted } };
  }
  const axis3 = (t, c) => ({ title: { text: t }, gridcolor: c.rule, backgroundcolor: c.panel, showbackground: true, color: c.muted, zerolinecolor: c.rule });
  const PCFG = { displayModeBar: false, responsive: true };
  function noPlotly(id) { $(id).innerHTML = '<p class="cap" style="padding:20px">This 3D figure needs the Plotly library, which did not load. Check your connection and reload.</p>'; }

  // ---------------------------------------------------------------- controls
  function initControls() {
    const ps = $("patientSel");
    teIdx.forEach((ri, k) => {
      const r = rows[ri], o = document.createElement("option");
      o.value = k; o.textContent = `#${k + 1} · ${r[0]} y · ${r[1] ? "male" : "female"} · ${r[13] ? "disease" : "no disease"}`;
      ps.appendChild(o);
    });
    ps.addEventListener("change", () => { S.patient = +ps.value; renderPatient(); });
    $("prevPatient").addEventListener("click", () => { S.patient = (S.patient + teIdx.length - 1) % teIdx.length; ps.value = S.patient; renderPatient(); });
    $("nextPatient").addEventListener("click", () => { S.patient = (S.patient + 1) % teIdx.length; ps.value = S.patient; renderPatient(); });
    $("weightsSel").addEventListener("change", (e) => { S.weights = e.target.value; renderNetDependent(); });
    $("actSel").addEventListener("change", (e) => { S.act = e.target.value; buildNets(); renderNetDependent(); renderProp(); renderBatchCurve(); runTraining(); });
    $("initSel").addEventListener("change", (e) => { S.init = e.target.value; buildNets(); renderNetDependent(); renderGradByAct(); renderBatchCurve(); runTraining(); });
    const fillLayers = (sel, hiddenOnly) => {
      sel.innerHTML = "";
      for (let l = 0; l < SIZES.length - (hiddenOnly ? 2 : 1); l++) { const o = document.createElement("option"); o.value = l; o.textContent = layerLabel(l); sel.appendChild(o); }
    };
    const fillNeurons = (sel, l, cur) => {
      sel.innerHTML = "";
      for (let j = 0; j < SIZES[l + 1]; j++) { const o = document.createElement("option"); o.value = j; o.textContent = outName(l, j); sel.appendChild(o); }
      sel.value = Math.min(cur, SIZES[l + 1] - 1);
    };
    fillLayers($("layerSel")); fillLayers($("bpLayerSel")); fillLayers($("bnLayerSel"), true); fillLayers($("dropLayerSel"), true); fillLayers($("wdLayer"));
    const bindPair = (ls, ns, lk, nk, render) => {
      $(ls).value = S[lk]; fillNeurons($(ns), S[lk], S[nk]);
      $(ls).addEventListener("change", (e) => { S[lk] = +e.target.value; S[nk] = 0; fillNeurons($(ns), S[lk], 0); render(); });
      $(ns).addEventListener("change", (e) => { S[nk] = +e.target.value; render(); });
    };
    bindPair("layerSel", "neuronSel", "layer", "neuron", () => { renderLayer(); renderNet(); });
    bindPair("bpLayerSel", "bpNeuronSel", "bpLayer", "bpNeuron", () => { renderBackprop(); renderUpdate(); });
    bindPair("bnLayerSel", "bnNeuronSel", "bnLayer", "bnNeuron", renderBN);
    S.selectLayer = (l) => { S.layer = l; S.neuron = 0; $("layerSel").value = l; fillNeurons($("neuronSel"), l, 0); renderLayer(); renderNet(); };
    $("bnGamma").addEventListener("input", (e) => { S.gamma = +e.target.value; renderBN(); });
    $("bnBeta").addEventListener("input", (e) => { S.beta = +e.target.value; renderBN(); });
    $("dropLayerSel").addEventListener("change", (e) => { S.dropLayer = +e.target.value; renderDrop(); });
    $("dropP").addEventListener("input", (e) => { S.dropP = +e.target.value; renderDrop(); });
    $("dropResample").addEventListener("click", () => { S.dropSeed++; renderDrop(); });
    $("dropMode").addEventListener("click", (e) => {
      S.dropTrain = !S.dropTrain; e.target.setAttribute("aria-pressed", String(S.dropTrain));
      e.target.textContent = S.dropTrain ? "Mode: training" : "Mode: evaluation"; renderDrop();
    });
    ["gx", "gy", "gz"].forEach((id) => $(id).addEventListener("input", renderGate));
    $("lrUpd").addEventListener("input", (e) => { S.lrUpd = +e.target.value; renderUpdate(); });
    $("wdLayer").addEventListener("change", renderResults);
    $("resRun").addEventListener("change", renderResults);
    $("gnRun").addEventListener("change", drawGradNorms);
    $("trEpochs").addEventListener("input", (e) => { $("trEpochsV").textContent = e.target.value; });
    $("trRun").addEventListener("click", runTraining);
    $("playFwd").addEventListener("click", () => playPass("netSvg", 1, "playFwd"));
    $("playBwd").addEventListener("click", () => playPass("deltaFlow", -1, "playBwd"));
    $("rollGD").addEventListener("click", rollGD);
  }

  // ---------------------------------------------------------------- 1. data
  function renderData() {
    const ri = teIdx[S.patient], r = rows[ri], x = xP();
    $("profileTitle").textContent = `Patient #${S.patient + 1}: ${r[0]} y, ${r[1] ? "male" : "female"}, ${r[13] ? "heart disease" : "no heart disease"}`;
    const svg = $("profileSvg"); clear(svg);
    const [W, H] = vb(svg), top = 30, rh = (H - top - 26) / 13, x0 = lin(-3, 3, 150, W - 120);
    [-3, -2, -1, 0, 1, 2, 3].forEach((t) => { el("line", { x1: x0(t), x2: x0(t), y1: top - 6, y2: H - 22, class: t === 0 ? "ax" : "grid" }, svg); el("text", { x: x0(t), y: H - 6, "text-anchor": "middle" }, svg, (t > 0 ? "+" : "") + t + " sd"); });
    el("text", { x: W - 8, y: 18, "text-anchor": "end" }, svg, "raw value");
    FEATS.forEach(([k], j) => {
      const y = top + j * rh + rh / 2, v = Math.max(-3.2, Math.min(3.2, x[j]));
      el("text", { x: 140, y: y + 4, "text-anchor": "end", style: "fill:var(--ink);font-size:12px" }, svg, k);
      el("line", { x1: x0(0), x2: x0(v), y1: y, y2: y, style: `stroke:var(${v < 0 ? "--neg" : "--pos"});stroke-width:7;stroke-linecap:round` }, svg);
      el("circle", { cx: x0(v), cy: y, r: 5.5, style: `fill:var(--panel);stroke:var(${v < 0 ? "--neg" : "--pos"});stroke-width:2.5` }, svg);
      el("text", { x: W - 8, y: y + 4, "text-anchor": "end", style: "fill:var(--ink)" }, svg, String(r[j]));
    });
    tex("zscoreBox", `x_{\\text{age}} = \\frac{${r[0]} - ${f2(MU[0])}}{${f2(SD[0])}} = ${f3(x[0])}, \\qquad x_{\\text{chol}} = \\frac{${r[4]} - ${f2(MU[4])}}{${f2(SD[4])}} = ${f3(x[4])}`);
    const sc = $("scatterSvg"); clear(sc);
    const pairs = [[7, 9, "max heart rate (bpm)", "ST depression (mm)"], [0, 4, "age (years)", "cholesterol (mg/dl)"], [0, 7, "age (years)", "max heart rate (bpm)"]];
    pairs.forEach(([a, b, la, lb], p) => {
      const g = el("g", { transform: `translate(${p * 367},0)` }, sc);
      const va = rows.map((q) => q[a]), vbv = rows.map((q) => q[b]);
      const pad = (lo, hi) => [lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05];
      const F = frame(g, 357, 330, { l: 46, r: 8, t: 10, b: 36 }, pad(Math.min(...va), Math.max(...va)), pad(Math.min(...vbv), Math.max(...vbv)), { xlabel: la, ylabel: lb, xn: 4, yn: 4 });
      rows.forEach((q) => el("circle", { cx: F.x(q[a]), cy: F.y(q[b]), r: 3.2, style: `fill:var(${q[13] ? "--neg" : "--fwd"});fill-opacity:.55` }, g));
      el("circle", { cx: F.x(r[a]), cy: F.y(r[b]), r: 8, style: "fill:none;stroke:var(--hl);stroke-width:3" }, g);
    });
    let h = `<tr><th>Input</th><th>Feature</th><th>Unit</th><th class="num">Raw</th><th class="num">μ</th><th class="num">σ</th><th class="num">Standardised</th></tr>`;
    FEATS.forEach(([k, name, unit], j) => { h += `<tr><td>x${j + 1}</td><td><b>${k}</b> · ${name}</td><td>${unit}</td><td class="num">${r[j]}</td><td class="num">${f2(MU[j])}</td><td class="num">${f2(SD[j])}</td><td class="num">${f3(x[j])}</td></tr>`; });
    $("featTable").innerHTML = h;
  }
  function renderSpread() {
    const svg = $("spreadSvg"); clear(svg);
    const [W, H] = vb(svg), top = 14, rh = (H - top - 40) / 13;
    const x = (v) => lin(Math.log10(0.1), Math.log10(100), 100, W - 14)(Math.log10(v));
    [0.1, 1, 10, 100].forEach((t) => { el("line", { x1: x(t), x2: x(t), y1: top, y2: H - 34, class: t === 1 ? "ax" : "grid" }, svg); el("text", { x: x(t), y: H - 20, "text-anchor": "middle" }, svg, String(t)); });
    el("text", { x: (100 + W) / 2, y: H - 4, "text-anchor": "middle" }, svg, "standard deviation (log scale)");
    FEATS.forEach(([k], j) => {
      const y = top + j * rh + rh / 2;
      el("text", { x: 92, y: y + 4, "text-anchor": "end", style: "fill:var(--ink)" }, svg, k);
      el("line", { x1: x(0.1), x2: x(SD[j]), y1: y - 3, y2: y - 3, style: "stroke:var(--c1);stroke-width:6;stroke-opacity:.7" }, svg);
      el("line", { x1: x(0.1), x2: x(1), y1: y + 4, y2: y + 4, style: "stroke:var(--fwd);stroke-width:4" }, svg);
      el("text", { x: x(Math.max(SD[j], 1.2)) + 4, y: y + 1, style: "font-size:10px" }, svg, f2(SD[j]));
    });
    el("text", { x: W - 14, y: 26, "text-anchor": "end", style: "fill:var(--c1)" }, svg, "grey: raw");
    el("text", { x: W - 14, y: 40, "text-anchor": "end", style: "fill:var(--fwd)" }, svg, "blue: standardised = 1");
  }

  // ---------------------------------------------------------------- 2. network
  function netGeom(W, H) {
    const xs = SIZES.map((_, i) => 110 + i * ((W - 290) / 5));
    const ys = SIZES.map((n) => { const sp = Math.min(27, (H - 90) / Math.max(1, n - 1)); const top = (H - 40) / 2 - (sp * (n - 1)) / 2; return Array.from({ length: n }, (_, k) => top + k * sp); });
    return { xs, ys };
  }
  function drawNetwork(svg, vals, opt) {
    clear(svg); svg.dataset.playing = "";
    const [W, H] = vb(svg), { xs, ys } = netGeom(W, H), nt = net();
    for (let l = 0; l < SIZES.length - 1; l++) {
      const Wl = nt.layers[l].W, mx = Math.max(1e-9, ...Wl.flatMap((r) => Array.from(r, Math.abs)));
      const sel = opt.selLayer === l;
      for (let j = 0; j < SIZES[l + 1]; j++) for (let i = 0; i < SIZES[l]; i++) {
        const w = Wl[j][i], o = (Math.abs(w) / mx) * (sel ? 0.85 : 0.32);
        if (o < 0.02) continue;
        el("line", { x1: xs[l], y1: ys[l][i], x2: xs[l + 1], y2: ys[l + 1][j], "data-l": l, "data-a": (Math.abs(w) / mx).toFixed(4), style: `stroke:var(${w >= 0 ? "--pos" : "--neg"});stroke-opacity:${o.toFixed(3)};stroke-width:${sel && j === opt.selNeuron ? 2.2 : 1}` }, svg);
      }
    }
    SIZES.forEach((n, l) => {
      const a = vals[l], mx = Math.max(1e-9, ...a.map(Math.abs));
      for (let k = 0; k < n; k++) {
        const isSel = opt.selLayer + 1 === l && k === opt.selNeuron;
        const fo = (0.06 + (0.88 * Math.abs(a[k])) / mx).toFixed(3);
        const cc = el("circle", { cx: xs[l], cy: ys[l][k], r: l === SIZES.length - 1 ? 14 : 9, "data-l": l, "data-o": fo, style: `fill:var(${opt.color});fill-opacity:${fo};stroke:var(${isSel ? "--hl" : "--muted"});stroke-width:${isSel ? 3 : 1}` }, svg);
        el("title", {}, cc, `${l === 0 ? FEATS[k][0] : l === SIZES.length - 1 ? "ŷ" : `h${l}·${k + 1}`}: ${opt.tipLabel || "value"} ${f3(a[k])}`);
        if (l === 0 && opt.labels) el("text", { x: xs[0] - 16, y: ys[0][k] + 4, "text-anchor": "end" }, svg, `${FEATS[k][0]} ${f2(a[k])}`);
      }
      const lbl = l === 0 ? "Input · 13" : l === SIZES.length - 1 ? "Output · σ" : `Hidden ${l} · ${n}`;
      el("text", { x: xs[l], y: H - 18, "text-anchor": "middle", style: "fill:var(--ink);font-size:12px" }, svg, lbl);
    });
    if (opt.outLabel) el("text", { x: xs[5] + 22, y: ys[5][0] + 4, style: "fill:var(--ink);font-size:13px" }, svg, opt.outLabel);
    if (opt.arrow) el("text", { x: W / 2, y: 18, "text-anchor": "middle", style: `fill:var(${opt.color});font-size:13px;font-weight:600` }, svg, opt.arrow);
    return { xs, ys };
  }
  function renderNet() {
    const fw = fwP();
    const vals = [Array.from(xP())].concat(fw.caches.map((c) => Array.from(c.A[0])));
    const { xs } = drawNetwork($("netSvg"), vals, { color: "--fwd", selLayer: S.layer, selNeuron: S.neuron, labels: true, outLabel: `ŷ = ${f3(fw.out[0])}` });
    const svg = $("netSvg"), [, H] = vb(svg);
    for (let l = 1; l < SIZES.length; l++) {
      const r = el("rect", { x: xs[l] - 55, y: 10, width: 110, height: H - 40, class: "layerpick", style: `fill:var(--hl);fill-opacity:${l === S.layer + 1 ? 0.07 : 0}`, tabindex: 0, role: "button", "aria-label": "Inspect " + layerLabel(l - 1) }, svg);
      const go = () => { S.selectLayer(l - 1); $("layer").scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth" }); };
      r.addEventListener("click", go);
      r.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    }
  }
  function drawNet3d() {
    if (!hasPlotly()) return noPlotly("net3d");
    const c = plotBase(), nt = net(), fw = fwP();
    const vals = [Array.from(xP())].concat(fw.caches.map((q) => Array.from(q.A[0])));
    const P = SIZES.map((n, l) => Array.from({ length: n }, (_, k) => { const R = n === 1 ? 0 : 0.6 + Math.sqrt(n) * 0.35, t = (2 * Math.PI * k) / n; return [l * 2.2, R * Math.cos(t), R * Math.sin(t)]; }));
    const segs = { pos: [[], [], []], neg: [[], [], []] };
    for (let l = 0; l < SIZES.length - 1; l++) {
      const Wl = nt.layers[l].W, all = Wl.flatMap((r) => Array.from(r, Math.abs)).sort((a, b) => b - a), thr = all[Math.floor(all.length * 0.35)] || 0;
      Wl.forEach((row, j) => row.forEach((w, i) => {
        if (Math.abs(w) < thr || w === 0) return;
        const s = w >= 0 ? segs.pos : segs.neg, a = P[l][i], b = P[l + 1][j];
        s[0].push(a[0], b[0], null); s[1].push(a[1], b[1], null); s[2].push(a[2], b[2], null);
      }));
    }
    const nodes = P.flat(), nv = vals.flat().map(Math.abs), mx = Math.max(...nv, 1e-9);
    const tr = [
      { type: "scatter3d", mode: "lines", x: segs.pos[0], y: segs.pos[1], z: segs.pos[2], line: { color: c.pos, width: 2 }, opacity: 0.45, hoverinfo: "skip" },
      { type: "scatter3d", mode: "lines", x: segs.neg[0], y: segs.neg[1], z: segs.neg[2], line: { color: c.neg, width: 2 }, opacity: 0.45, hoverinfo: "skip" },
      { type: "scatter3d", mode: "markers", x: nodes.map((p) => p[0]), y: nodes.map((p) => p[1]), z: nodes.map((p) => p[2]), marker: { size: nodes.map((_, k) => (k === nodes.length - 1 ? 12 : 7)), color: nv.map((v) => v / mx), colorscale: [[0, c.soft], [1, c.fwd]], cmin: 0, cmax: 1, line: { color: c.muted, width: 1 } }, text: SIZES.flatMap((n, l) => Array.from({ length: n }, (_, k) => (l === 0 ? FEATS[k][0] : l === 5 ? "ŷ" : `h${l}·${k + 1}`) + " = " + f3(vals[l][k]))), hoverinfo: "text" },
    ];
    const ax = { showticklabels: false, showgrid: false, zeroline: false, showbackground: false, title: { text: "" } };
    Plotly.react("net3d", tr, { margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: c.panel, font: c.font, showlegend: false, scene: { xaxis: ax, yaxis: ax, zaxis: ax, camera: { eye: { x: 0.2, y: -2.1, z: 0.9 } }, aspectmode: "manual", aspectratio: { x: 2.2, y: 1, z: 1 } }, uirevision: "net3d" }, PCFG);
  }
  function renderParams() {
    const svg = $("paramSvg"); clear(svg);
    const [W, H] = vb(svg);
    const data = SIZES.slice(0, -1).map((fi, l) => ({ w: fi * SIZES[l + 1], b: SIZES[l + 1], bn: l < LAST ? 2 * SIZES[l + 1] : 0 }));
    const mx = Math.max(...data.map((d) => d.w + d.b + d.bn));
    const x = lin(0, mx, 70, W - 50), rh = (H - 40) / 5;
    data.forEach((d, l) => {
      const y = 10 + l * rh;
      el("text", { x: 62, y: y + rh / 2 + 4, "text-anchor": "end", style: "fill:var(--ink)" }, svg, `layer ${l + 1}`);
      let x0 = x(0);
      [[d.w, "--fwd"], [d.b, "--bwd"], [d.bn, "--pos"]].forEach(([v, c]) => { if (!v) return; el("rect", { x: x0, y: y + 5, width: x(v) - x(0), height: rh - 12, style: `fill:var(${c})` }, svg); x0 += x(v) - x(0); });
      el("text", { x: x0 + 5, y: y + rh / 2 + 4 }, svg, String(d.w + d.b + d.bn));
    });
    el("text", { x: 70, y: H - 8 }, svg, "blue: weights · orange: biases · green: batch-norm γ, β");
    let h = `<tr><th>Layer</th><th>Shape</th><th class="num">Weights</th><th class="num">Biases</th><th class="num">BN γ, β</th><th class="num">Total</th></tr>`;
    let tw = 0, tb = 0, tn = 0;
    data.forEach((d, l) => { tw += d.w; tb += d.b; tn += d.bn; h += `<tr><td>${layerLabel(l)}</td><td>${SIZES[l + 1]} × ${SIZES[l]}</td><td class="num">${d.w}</td><td class="num">${d.b}</td><td class="num">${d.bn || "–"}</td><td class="num">${d.w + d.b + d.bn}</td></tr>`; });
    h += `<tr class="sel"><td><b>Total</b></td><td></td><td class="num"><b>${tw}</b></td><td class="num"><b>${tb}</b></td><td class="num"><b>${tn}</b></td><td class="num"><b>${tw + tb + tn}</b></td></tr>`;
    $("paramTable").innerHTML = h;
  }

  // ---------------------------------------------------------------- 3. neuron
  function renderLayer() {
    const l = S.layer, j = S.neuron, nt = net(), L = nt.layers[l], fw = fwP(), c = fw.caches[l];
    const xin = c.Ain[0], w = L.W[j], b = L.b[j], z = c.Z[0][j], act = actOf(l), a = act.f(z), n = xin.length;
    const prods = Array.from(w, (wi, i) => wi * xin[i]);
    $("neuronTitle").textContent = `Anatomy of neuron ${outName(l, j)} in ${layerLabel(l).toLowerCase()}`;
    {
      const svg = $("neuronSvg"); clear(svg);
      const [W, H] = vb(svg), sp = Math.min(27, (H - 70) / Math.max(1, n - 1)), top = H / 2 - (sp * (n - 1)) / 2;
      const cx = Math.round(W * 0.58), cy = H / 2, mw = Math.max(1e-9, ...Array.from(w, Math.abs)), mxv = Math.max(1e-9, ...Array.from(xin, Math.abs));
      el("text", { x: 10, y: 16 }, svg, "input i · value xᵢ");
      el("text", { x: 300, y: 16, "text-anchor": "middle" }, svg, "weight wᵢ");
      for (let i = 0; i < n; i++) {
        const y = top + i * sp, wi = w[i];
        el("line", { x1: 196, y1: y, x2: cx - 42, y2: cy, style: `stroke:var(${wi >= 0 ? "--pos" : "--neg"});stroke-width:${(0.6 + (5 * Math.abs(wi)) / mw).toFixed(2)};stroke-opacity:.75` }, svg);
        el("text", { x: 10, y: y + 4, style: "fill:var(--ink)" }, svg, `x${i + 1} ${inName(l, i)}`);
        el("text", { x: 168, y: y + 4, "text-anchor": "end", style: "fill:var(--fwd)" }, svg, f2(xin[i]));
        el("circle", { cx: 186, cy: y, r: 8, style: `fill:var(--fwd);fill-opacity:${(0.1 + (0.8 * Math.abs(xin[i])) / mxv).toFixed(2)};stroke:var(--muted)` }, svg);
        const ly = y + (cy - y) * 0.2, lx = 196 + (cx - 42 - 196) * 0.2;
        el("rect", { x: lx - 25, y: ly - 9, width: 50, height: 15, rx: 3, style: "fill:var(--panel);fill-opacity:.85" }, svg);
        el("text", { x: lx, y: ly + 3, "text-anchor": "middle", style: `fill:var(${wi >= 0 ? "--pos" : "--neg"});font-size:10.5px` }, svg, f2(wi));
      }
      el("line", { x1: cx, y1: 40, x2: cx, y2: cy - 42, style: "stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 3" }, svg);
      el("text", { x: cx, y: 32, "text-anchor": "middle", style: "fill:var(--ink)" }, svg, `bias b = ${f3(b)}`);
      el("circle", { cx, cy, r: 42, style: "fill:var(--soft);stroke:var(--ink);stroke-width:1.5" }, svg);
      el("text", { x: cx, y: cy - 2, "text-anchor": "middle", style: "fill:var(--ink);font-size:24px" }, svg, "Σ");
      el("text", { x: cx, y: cy + 20, "text-anchor": "middle", style: "fill:var(--fwd);font-size:12px;font-weight:600" }, svg, `z = ${f3(z)}`);
      el("line", { x1: cx + 42, y1: cy, x2: cx + 92, y2: cy, style: "stroke:var(--ink);stroke-width:1.5" }, svg);
      el("rect", { x: cx + 94, y: cy - 24, width: 78, height: 48, rx: 8, style: "fill:var(--panel);stroke:var(--ink);stroke-width:1.5" }, svg);
      el("text", { x: cx + 133, y: cy + 5, "text-anchor": "middle", style: "fill:var(--ink);font-size:13px" }, svg, act.name);
      el("line", { x1: cx + 172, y1: cy, x2: cx + 228, y2: cy, style: "stroke:var(--ink);stroke-width:1.5" }, svg);
      el("path", { d: `M${cx + 228},${cy - 6} L${cx + 240},${cy} L${cx + 228},${cy + 6} Z`, style: "fill:var(--ink)" }, svg);
      el("text", { x: cx + 234, y: cy - 16, "text-anchor": "middle", style: "fill:var(--fwd);font-size:14px;font-weight:600" }, svg, `a = ${f3(a)}`);
      el("text", { x: cx + 234, y: cy + 30, "text-anchor": "middle" }, svg, l === LAST ? "ŷ = P(disease)" : "to the next layer");
    }
    heatmap($("heatSvg"), L.W, j, { pos: "--pos", neg: "--neg", rowName: (k) => outName(l, k), colName: (i) => inName(l, i) });
    {
      const svg = $("waterSvg"); clear(svg);
      const [W, H] = vb(svg), steps = prods.concat([b]);
      let cum = 0; const segs = steps.map((v) => { const s = cum; cum += v; return [s, cum]; });
      const all = segs.flat().concat([0, z]), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.12 || 1;
      const nb = steps.length + 1, F = frame(svg, W, H, { l: 48, r: 10, t: 22, b: 46 }, [0, nb], [lo - pad, hi + pad], { xticks: [] });
      const bw = (W - 58) / nb;
      el("line", { x1: 48, x2: W - 10, y1: F.y(0), y2: F.y(0), style: "stroke:var(--muted)" }, svg);
      segs.forEach(([s, e], k) => {
        const x = F.x(k) + bw * 0.15, v = e - s, isB = k === steps.length - 1;
        el("rect", { class: "gb", x, y: Math.min(F.y(s), F.y(e)), width: bw * 0.7, height: Math.max(1.5, Math.abs(F.y(e) - F.y(s))), style: `fill:var(${isB ? "--c1" : v >= 0 ? "--pos" : "--neg"});fill-opacity:.85` }, svg);
        if (k < steps.length - 1) el("line", { x1: x + bw * 0.7, x2: x + bw, y1: F.y(e), y2: F.y(e), style: "stroke:var(--muted);stroke-dasharray:2 2" }, svg);
        el("text", { x: x + bw * 0.35, y: Math.min(F.y(s), F.y(e)) - 5, "text-anchor": "middle", style: "font-size:10px" }, svg, (v >= 0 ? "+" : "") + f2(v));
        el("text", { x: x + bw * 0.35, y: H - 28, "text-anchor": "middle", style: "fill:var(--ink);font-size:10.5px" }, svg, isB ? "bias" : inName(l, k));
        if (!isB) el("text", { x: x + bw * 0.35, y: H - 14, "text-anchor": "middle", style: "font-size:9.5px" }, svg, `w${k + 1}·x${k + 1}`);
      });
      const xz = F.x(nb - 1) + bw * 0.15;
      el("rect", { x: xz, y: Math.min(F.y(0), F.y(z)), width: bw * 0.7, height: Math.max(1.5, Math.abs(F.y(z) - F.y(0))), style: "fill:var(--hl)" }, svg);
      el("text", { x: xz + bw * 0.35, y: Math.min(F.y(0), F.y(z)) - 5, "text-anchor": "middle", style: "fill:var(--hl);font-weight:600" }, svg, `z = ${f2(z)}`);
      el("text", { x: xz + bw * 0.35, y: H - 28, "text-anchor": "middle", style: "fill:var(--ink);font-size:10.5px" }, svg, "total z");
    }
    const sum = prods.reduce((s, p) => s + p, 0), terms = prods.map((p, i) => `${tp(w[i])}\,${tp(xin[i])}`);
    let rowsT = []; for (let i = 0; i < terms.length; i += 4) rowsT.push(terms.slice(i, i + 4).join(" + "));
    tex("sumBox", `\\begin{aligned} z^{(${l + 1})}_{${j + 1}} &= ${rowsT.join(" \\\\ &\\quad + ")} \\\\ &= ${f3(sum)} + ${tp(b)} = ${f3(z)} \\\\ a^{(${l + 1})}_{${j + 1}} &= ${actTex(l)}(${f3(z)}) = ${f3(a)} \\end{aligned}`);
    let h = `<tr><th>i</th><th>Input</th><th class="num">xᵢ</th><th class="num">wᵢ</th><th class="num">wᵢ·xᵢ</th></tr>`;
    prods.forEach((p, i) => { h += `<tr><td>${i + 1}</td><td>input ${i + 1} · ${inName(l, i)}</td><td class="num fwd-c">${f3(xin[i])}</td><td class="num">${f3(w[i])}</td><td class="num">${f3(p)}</td></tr>`; });
    h += `<tr><td></td><td>bias</td><td></td><td class="num">${f3(b)}</td><td class="num">${f3(b)}</td></tr><tr class="sel"><td></td><td><b>z</b></td><td></td><td></td><td class="num"><b>${f3(z)}</b></td></tr>`;
    $("inTable").innerHTML = h;
    {
      const svg = $("fanSvg"); clear(svg);
      const fi = L.fanIn, fo = L.fanOut, cx = 180, cy = 120, nIn = Math.min(fi, 9), nOut = Math.min(fo, 9);
      for (let k = 0; k < nIn; k++) { const y = 30 + (180 * k) / Math.max(1, nIn - 1); el("line", { x1: 40, y1: y, x2: cx - 22, y2: cy, style: "stroke:var(--fwd);stroke-width:1.5" }, svg); el("circle", { cx: 40, cy: y, r: 5, style: "fill:var(--fwd)" }, svg); }
      for (let k = 0; k < nOut; k++) { const y = nOut === 1 ? cy : 30 + (180 * k) / (nOut - 1); el("line", { x1: cx + 22, y1: cy, x2: 320, y2: y, style: "stroke:var(--bwd);stroke-width:1.5" }, svg); el("circle", { cx: 320, cy: y, r: 5, style: "fill:var(--bwd)" }, svg); }
      el("circle", { cx, cy, r: 22, style: "fill:var(--soft);stroke:var(--ink);stroke-width:1.5" }, svg);
      el("text", { x: cx, y: cy + 5, "text-anchor": "middle", style: "fill:var(--ink);font-size:13px" }, svg, "layer");
      el("text", { x: 60, y: 248, "text-anchor": "middle", style: "fill:var(--fwd);font-size:13px;font-weight:600" }, svg, `fan-in = ${fi}`);
      el("text", { x: 300, y: 248, "text-anchor": "middle", style: "fill:var(--bwd);font-size:13px;font-weight:600" }, svg, `fan-out = ${fo}`);
      if (fi > nIn) el("text", { x: 22, y: 128, "text-anchor": "middle" }, svg, "⋮");
      if (fo > nOut) el("text", { x: 338, y: 128, "text-anchor": "middle" }, svg, "⋮");
      $("fanCap").textContent = `${layerLabel(l)}: each neuron receives ${fi} inputs (fan-in); the layer sends ${fo} outputs onward (fan-out).`;
    }
    const init = Core.INIT[S.init], sd = init.std(L.fanIn, L.fanOut);
    const allW = L.W.flatMap((r) => Array.from(r)), emp = Math.sqrt(allW.reduce((s, v) => s + v * v, 0) / allW.length);
    tex("initBox", `\\begin{gathered} n_{\\text{in}} = ${L.fanIn},\\;\\; n_{\\text{out}} = ${L.fanOut}, \\qquad \\sigma_w = ${initTex(L.fanIn, L.fanOut)} = ${f3(sd)} \\\\ \\widehat{\\sigma}_w = ${f3(emp)}\\;\\;\\text{(${S.weights === "trained" ? "after training" : "measured on the " + allW.length + " weights"})} \\end{gathered}`);
    histogram($("wHist"), [{ values: allW, color: "--fwd" }], { pdfStd: sd, label: "weight value" });
    {
      const svg = $("actGrid"); clear(svg);
      const keys = ["relu", "leaky", "tanh", "sigmoid"], cur = l === LAST ? "sigmoid" : S.act;
      keys.forEach((k, p) => {
        const g = el("g", { transform: `translate(${p * 225},0)` }, svg), A = Core.ACT[k];
        if (k === cur) el("rect", { x: 2, y: 2, width: 219, height: 236, rx: 8, style: "fill:none;stroke:var(--hl);stroke-width:2" }, g);
        el("text", { x: 112, y: 22, "text-anchor": "middle", style: `fill:var(--ink);font-size:13px;font-weight:${k === cur ? 700 : 400}` }, g, A.name);
        const zr = 4, zz = Array.from({ length: 121 }, (_, i) => -zr + (2 * zr * i) / 120);
        const F = frame(g, 221, 236, { l: 30, r: 10, t: 32, b: 28 }, [-zr, zr], [-1.2, 2.2], { xticks: [-4, 0, 4], yn: 3 });
        el("path", { d: path(zz.map((v) => [F.x(v), F.y(Math.min(2.2, A.f(v)))])), class: "draw", pathLength: 1, style: "fill:none;stroke:var(--fwd);stroke-width:2.4" }, g);
        el("path", { d: path(zz.map((v) => [F.x(v), F.y(A.d(v))])), style: "fill:none;stroke:var(--bwd);stroke-width:1.8;stroke-dasharray:4 3" }, g);
        if (k === cur) { const zc = Math.max(-zr, Math.min(zr, z)); el("circle", { cx: F.x(zc), cy: F.y(Math.min(2.2, A.f(zc))), r: 6, style: "fill:var(--hl)" }, g); }
      });
    }
    tex("actBox", `\\begin{aligned} a &= ${actTex(l)}(${f3(z)}) = ${f3(a)} \\\\ ${actTex(l)}'(z) &= ${f3(act.d(z))}${S.act === "relu" && l < LAST && z <= 0 ? "\\quad \\text{(inactive)}" : ""} \\end{aligned}`);
    {
      const outs = Array.from(c.A[0]), svg = $("layerOut"); clear(svg);
      const [W, H] = vb(svg), lo = Math.min(0, ...outs), hi = Math.max(0.1, ...outs);
      $("layerOutTitle").textContent = l === LAST ? "Output of the network: ŷ" : `Output of hidden ${l + 1}: the vector a⁽${l + 1}⁾ for this patient`;
      const F = frame(svg, W, H, { l: 44, r: 10, t: 14, b: 30 }, [0, outs.length], [lo, hi * 1.12], { xticks: [] });
      const bw = (W - 54) / outs.length;
      outs.forEach((v, k) => {
        el("rect", { class: "gb", x: F.x(k) + bw * 0.15, y: Math.min(F.y(v), F.y(0)), width: bw * 0.7, height: Math.max(1, Math.abs(F.y(v) - F.y(0))), style: `fill:var(${k === j ? "--hl" : "--fwd"})` }, svg);
        el("text", { x: F.x(k) + bw / 2, y: Math.min(F.y(v), F.y(0)) - 5, "text-anchor": "middle", style: "font-size:10.5px" }, svg, f2(v));
        el("text", { x: F.x(k) + bw / 2, y: H - 12, "text-anchor": "middle" }, svg, outName(l, k));
      });
    }
  }
  function heatmap(svg, M, selRow, o) {
    clear(svg);
    const [W, H] = vb(svg), R = M.length, Cn = M[0].length;
    const m = { l: 56, t: 10, r: 10, b: 54 }, cw = (W - m.l - m.r) / Cn, ch = Math.min(26, (H - m.t - m.b) / R);
    const mx = Math.max(1e-12, ...M.flatMap((r) => Array.from(r, Math.abs)));
    M.forEach((row, r) => row.forEach((v, c) => el("rect", { x: m.l + c * cw, y: m.t + r * ch, width: cw - 1, height: ch - 1, style: `fill:var(${v >= 0 ? o.pos : o.neg});fill-opacity:${(0.06 + (0.9 * Math.abs(v)) / mx).toFixed(3)}` }, svg)));
    for (let r = 0; r < R; r++) el("text", { x: m.l - 5, y: m.t + r * ch + ch / 2 + 4, "text-anchor": "end", style: r === selRow ? "fill:var(--hl);font-weight:600;font-size:10px" : "font-size:10px" }, svg, o.rowName(r));
    for (let c = 0; c < Cn; c++) el("text", { x: m.l + c * cw + cw / 2, y: m.t + R * ch + 8, transform: `rotate(-60 ${m.l + c * cw + cw / 2} ${m.t + R * ch + 8})`, "text-anchor": "end", style: "font-size:9.5px" }, svg, o.colName(c));
    if (selRow !== null) el("rect", { x: m.l - 1, y: m.t + selRow * ch - 1, width: Cn * cw + 1, height: ch + 1, style: "fill:none;stroke:var(--hl);stroke-width:2.5" }, svg);
  }
  function histogram(svg, sets, opt = {}) {
    clear(svg);
    const [W, H] = vb(svg), all = sets.flatMap((s) => s.values);
    let lo = Math.min(...all), hi = Math.max(...all);
    if (opt.pdfStd) { lo = Math.min(lo, -3 * opt.pdfStd); hi = Math.max(hi, 3 * opt.pdfStd); }
    if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
    const nb = opt.bins || 26, bwid = (hi - lo) / nb;
    const counts = sets.map((s) => { const c = Array(nb).fill(0); s.values.forEach((v) => c[Math.max(0, Math.min(nb - 1, Math.floor((v - lo) / bwid)))]++); return c.map((k) => k / (s.values.length * bwid)); });
    let ymax = Math.max(...counts.flat());
    const pdf = opt.pdfStd ? (v) => Math.exp(-0.5 * (v / opt.pdfStd) ** 2) / (opt.pdfStd * Math.sqrt(2 * Math.PI)) : null;
    if (pdf) ymax = Math.max(ymax, pdf(0));
    const F = frame(svg, W, H, { l: 40, r: 10, t: 10, b: 32 }, [lo, hi], [0, ymax * 1.1 || 1], { xlabel: opt.label || "", yfmt: (v) => +v.toFixed(2) });
    counts.forEach((c, si) => c.forEach((v, k) => { if (v) el("rect", { class: "gb", x: F.x(lo + k * bwid) + 1, y: F.y(v), width: Math.max(1, F.x(lo + bwid) - F.x(lo) - 2), height: F.y(0) - F.y(v), style: `fill:var(${sets[si].color});fill-opacity:${sets.length > 1 ? 0.55 : 0.8}` }, svg); }));
    if (pdf) {
      const xs = Array.from({ length: 101 }, (_, k) => lo + ((hi - lo) * k) / 100);
      el("path", { d: path(xs.map((v) => [F.x(v), F.y(pdf(v))])), style: "fill:none;stroke:var(--bwd);stroke-width:2;stroke-dasharray:5 4" }, svg);
    }
    return F;
  }
  function renderProp() {
    const svg = $("propSvg"); clear(svg);
    const depth = 12, width = 64, sizes = [13].concat(Array(depth).fill(width)).concat([1]);
    const lines = [["he", "--fwd", "He"], ["xavier", "--pos", "Xavier"], ["small", "--neg", "N(0, 0.01²)"]].map(([k, c, name]) => {
      const n2 = Core.create(sizes, { init: k, act: S.act, seed: 5 }), fw = Core.forward(n2, Xtr, { train: false });
      return { c, name, values: fw.caches.slice(0, depth).map((q) => { const v = q.A.flatMap((r) => Array.from(r)); const m = v.reduce((s, t) => s + t, 0) / v.length; return Math.sqrt(v.reduce((s, t) => s + (t - m) ** 2, 0) / v.length); }) };
    });
    const [W, H] = vb(svg);
    const F = frame(svg, W, H, { l: 46, r: 96, t: 10, b: 32 }, [1, depth], [1e-12, 10], { ylog: true, xlabel: `layer (${Core.ACT[S.act].name})`, ylabel: "spread of activations", xticks: [1, 3, 6, 9, 12] });
    lines.forEach((s) => {
      const pts = s.values.map((v, i) => [F.x(i + 1), F.y(Math.max(1e-12, v))]);
      el("path", { d: path(pts), class: "draw", pathLength: 1, style: `fill:none;stroke:var(${s.c});stroke-width:2.5` }, svg);
      pts.forEach((p) => el("circle", { cx: p[0], cy: p[1], r: 3, style: `fill:var(${s.c})` }, svg));
      el("text", { x: W - 92, y: pts[pts.length - 1][1] + 4, style: `fill:var(${s.c});font-weight:600` }, svg, s.name);
    });
  }

  // ---------------------------------------------------------------- 4. batch norm
  function renderBN() {
    $("bnGammaV").textContent = S.gamma.toFixed(1); $("bnBetaV").textContent = S.beta.toFixed(1);
    const l = S.bnLayer, j = S.bnNeuron, batch = [xP()].concat(Xtr.slice(0, 31));
    const Zl = Core.forward(net(), batch, { train: false }).caches[l].Z, z = Zl.map((r) => r[j]);
    const N = z.length, mu = z.reduce((s, v) => s + v, 0) / N, vr = z.reduce((s, v) => s + (v - mu) ** 2, 0) / N, sd = Math.sqrt(vr + Core.EPS);
    const zh = z.map((v) => (v - mu) / sd), y = zh.map((v) => S.gamma * v + S.beta);
    bnHist($("bnBefore"), z, mu, Math.sqrt(vr), "before: z", "--c1");
    bnHist($("bnAfter"), y, S.beta, S.gamma, "after: γ·ẑ + β", "--fwd");
    const act = Core.ACT[S.act];
    tex("bnMath", `\\mu_B = ${f3(mu)},\\;\\; \\sigma_B = ${f3(Math.sqrt(vr))}; \\qquad z = ${f3(z[0])} \\;\\to\\; \\hat z = \\frac{${f3(z[0])} - ${tp(mu)}}{\\sqrt{${f3(vr)} + \\varepsilon}} = ${f3(zh[0])} \\;\\to\\; y = ${S.gamma.toFixed(1)}\\cdot${tp(zh[0])} + ${tp(S.beta)} = ${f3(y[0])}`);
    const svg = $("bnLayer"); clear(svg);
    const n = SIZES[l + 1], stats = Array.from({ length: n }, (_, k) => { const v = Zl.map((r) => r[k]), m = v.reduce((s, q) => s + q, 0) / N; return { m, s: Math.sqrt(v.reduce((s, q) => s + (q - m) ** 2, 0) / N) }; });
    const [W, H] = vb(svg), lo = Math.min(S.beta - S.gamma, ...stats.map((t) => t.m - t.s)) - 0.2, hi = Math.max(S.beta + S.gamma, ...stats.map((t) => t.m + t.s)) + 0.2;
    const F = frame(svg, W, H, { l: 46, r: 12, t: 12, b: 30 }, [0, n], [lo, hi], { xticks: [] });
    const cw = (W - 58) / n;
    stats.forEach((t, k) => {
      const cx = F.x(k) + cw / 2;
      [[t.m, t.s, "--c1", -7], [S.beta, S.gamma, "--fwd", 7]].forEach(([m, s, c, dx]) => {
        el("line", { x1: cx + dx, x2: cx + dx, y1: F.y(m - s), y2: F.y(m + s), style: `stroke:var(${c});stroke-width:4;stroke-linecap:round` }, svg);
        el("circle", { cx: cx + dx, cy: F.y(m), r: 4.5, style: `fill:var(--panel);stroke:var(${c});stroke-width:2.5` }, svg);
      });
      el("text", { x: cx, y: H - 10, "text-anchor": "middle", style: k === j ? "fill:var(--hl);font-weight:600" : "" }, svg, outName(l, k));
    });
  }
  function bnHist(svg, vals, m, s, label, color) {
    clear(svg);
    const [W, H] = vb(svg), lo = Math.min(...vals, m - 2 * s) - 0.1, hi = Math.max(...vals, m + 2 * s) + 0.1;
    const nb = 14, bw = (hi - lo) / nb, cnt = Array(nb).fill(0);
    vals.forEach((v) => cnt[Math.min(nb - 1, Math.floor((v - lo) / bw))]++);
    const F = frame(svg, W, H, { l: 30, r: 8, t: 26, b: 30 }, [lo, hi], [0, Math.max(...cnt) + 1], { yn: 3, xn: 5 });
    el("rect", { x: F.x(m - s), y: 26, width: Math.max(1, F.x(m + s) - F.x(m - s)), height: H - 56, style: "fill:var(--soft)" }, svg);
    cnt.forEach((c, k) => { if (c) el("rect", { x: F.x(lo + k * bw) + 1, y: F.y(c), width: F.x(lo + bw) - F.x(lo) - 2, height: F.y(0) - F.y(c), style: `fill:var(${color});fill-opacity:.85` }, svg); });
    el("line", { x1: F.x(m), x2: F.x(m), y1: 22, y2: H - 30, style: "stroke:var(--ink);stroke-width:1.5" }, svg);
    el("path", { d: `M${F.x(vals[0]) - 7},${H - 44} L${F.x(vals[0]) + 7},${H - 44} L${F.x(vals[0])},${H - 32} Z`, style: "fill:var(--hl)" }, svg);
    el("text", { x: 30, y: 14, style: "fill:var(--ink);font-size:12px" }, svg, `${label} · mean ${f2(m)} · sd ${f2(s)}`);
  }

  // ---------------------------------------------------------------- 5. dropout
  function dropMasks() {
    const keep = 1 - S.dropP;
    return SIZES.slice(1, -1).map((n, l) => { const r = Core.rng(S.dropSeed * 97 + l); return Array.from({ length: n }, () => r() < keep); });
  }
  function renderDrop() {
    $("dropPV").textContent = S.dropP.toFixed(2);
    const masks = dropMasks(), keep = 1 - S.dropP;
    {
      const svg = $("dropNetSvg"); clear(svg);
      const [W, H] = vb(svg), xs = SIZES.map((_, i) => 50 + i * ((W - 100) / 5));
      const ys = SIZES.map((n) => { const sp = Math.min(17, (H - 60) / Math.max(1, n - 1)); const top = (H - 30) / 2 - (sp * (n - 1)) / 2; return Array.from({ length: n }, (_, k) => top + k * sp); });
      const on = (l, k) => !S.dropTrain || l === 0 || l === SIZES.length - 1 || masks[l - 1][k];
      for (let l = 0; l < SIZES.length - 1; l++) for (let i = 0; i < SIZES[l]; i++) for (let j = 0; j < SIZES[l + 1]; j++) {
        if (!on(l, i) || !on(l + 1, j)) continue;
        el("line", { x1: xs[l], y1: ys[l][i], x2: xs[l + 1], y2: ys[l + 1][j], style: "stroke:var(--muted);stroke-opacity:.22" }, svg);
      }
      SIZES.forEach((n, l) => {
        if (l === S.dropLayer + 1) el("rect", { x: xs[l] - 16, y: 4, width: 32, height: H - 38, rx: 6, style: "fill:var(--hl);fill-opacity:.08" }, svg);
        for (let k = 0; k < n; k++) {
          const o = on(l, k), x = xs[l], y = ys[l][k];
          el("circle", { cx: x, cy: y, r: 6.5, style: `fill:var(${o ? "--fwd" : "--soft"});fill-opacity:${o ? 0.75 : 1};stroke:var(--muted)` }, svg);
          if (!o) { el("line", { x1: x - 6, y1: y - 6, x2: x + 6, y2: y + 6, style: "stroke:var(--neg);stroke-width:2" }, svg); el("line", { x1: x - 6, y1: y + 6, x2: x + 6, y2: y - 6, style: "stroke:var(--neg);stroke-width:2" }, svg); }
        }
        el("text", { x: xs[l], y: H - 10, "text-anchor": "middle" }, svg, l === 0 ? "input" : l === SIZES.length - 1 ? "output" : `hidden ${l}`);
      });
    }
    const l = S.dropLayer, h = Array.from(fwP().caches[l].A[0]), m = masks[l];
    const after = h.map((v, k) => (S.dropTrain ? (m[k] ? v / keep : 0) : v));
    {
      const svg = $("dropBars"); clear(svg);
      const [W, H] = vb(svg), hi = Math.max(0.1, ...h, ...after), F = frame(svg, W, H, { l: 38, r: 8, t: 10, b: 28 }, [0, h.length], [Math.min(0, ...h, ...after), hi * 1.1], { xticks: [] });
      const bw = (W - 46) / h.length;
      h.forEach((v, k) => {
        const x = F.x(k);
        el("rect", { x: x + bw * 0.12, y: Math.min(F.y(v), F.y(0)), width: bw * 0.36, height: Math.max(1, Math.abs(F.y(v) - F.y(0))), style: "fill:var(--c1);fill-opacity:.7" }, svg);
        el("rect", { x: x + bw * 0.5, y: Math.min(F.y(after[k]), F.y(0)), width: bw * 0.36, height: Math.max(1, Math.abs(F.y(after[k]) - F.y(0))), style: "fill:var(--fwd)" }, svg);
        if (S.dropTrain && !m[k]) el("text", { x: x + bw * 0.68, y: F.y(0) - 4, "text-anchor": "middle", style: "fill:var(--neg);font-weight:700" }, svg, "×");
        el("text", { x: x + bw / 2, y: H - 10, "text-anchor": "middle", style: "font-size:9.5px" }, svg, String(k + 1));
      });
    }
    const sb = h.reduce((s, v) => s + v, 0), sa = after.reduce((s, v) => s + v, 0), kept = S.dropTrain ? m.filter(Boolean).length : h.length;
    tex("dropMath", S.dropTrain
      ? `\\text{kept } ${kept}/${h.length}, \\quad \\tilde a_j = \\frac{m_j a_j}{1 - ${S.dropP.toFixed(2)}}, \\quad \\textstyle\\sum_j a_j = ${f3(sb)} \\;\\to\\; \\sum_j \\tilde a_j = ${f3(sa)}`
      : `\\text{evaluation mode: } \\tilde a_j = a_j, \\quad \\textstyle\\sum_j a_j = ${f3(sb)}`);
    {
      const n2 = Core.clone(net()); n2.drop = S.dropP;
      const r = Core.rng(77), preds = [];
      for (let t = 0; t < 300; t++) preds.push(Core.forward(n2, [xP()], { train: true, r }).out[0]);
      const ev = fwP().out[0], mean = preds.reduce((s, v) => s + v, 0) / preds.length;
      const svg = $("mcSvg"); clear(svg);
      const [W, H] = vb(svg), nb = 50, cnt = Array(nb).fill(0);
      preds.forEach((p) => cnt[Math.min(nb - 1, Math.floor(p * nb))]++);
      const F = frame(svg, W, H, { l: 40, r: 12, t: 22, b: 32 }, [0, 1], [0, Math.max(...cnt) * 1.15 + 1], { xlabel: "predicted probability of disease ŷ", yn: 3 });
      cnt.forEach((c, k) => { if (c) el("rect", { class: "gb", x: F.x(k / nb) + 1, y: F.y(c), width: F.x(1 / nb) - F.x(0) - 2, height: F.y(0) - F.y(c), style: "fill:var(--fwd);fill-opacity:.75" }, svg); });
      el("line", { x1: F.x(ev), x2: F.x(ev), y1: 18, y2: H - 32, style: "stroke:var(--hl);stroke-width:3" }, svg);
      el("line", { x1: F.x(mean), x2: F.x(mean), y1: 18, y2: H - 32, style: "stroke:var(--ink);stroke-width:2;stroke-dasharray:5 4" }, svg);
      el("text", { x: F.x(ev) + (ev > 0.6 ? -6 : 6), y: 14, "text-anchor": ev > 0.6 ? "end" : "start", style: "fill:var(--hl);font-weight:600" }, svg, `full network ŷ = ${f3(ev)} · average over masks ${f3(mean)}`);
    }
  }

  // ---------------------------------------------------------------- 6. loss
  function renderLoss() {
    const yh = fwP().out[0], y = yP(), L = Core.bce([yh], [y]);
    kpis("lossKpis", [[f3(yh), "ŷ = P(disease)"], [String(y), "true label y"], [f3(L), "loss"]]);
    tex("lossMath", `\\mathcal{L} = -\\bigl[\\,${y}\\log ${f3(yh)} + ${1 - y}\\log(1 - ${f3(yh)})\\,\\bigr] = ${f3(L)}, \\qquad \\frac{\\partial \\mathcal{L}}{\\partial z^{(5)}} = \\hat y - y = ${f3(yh - y)}`);
    const svg = $("lossPlot"); clear(svg);
    const [W, H] = vb(svg), F = frame(svg, W, H, { l: 40, r: 12, t: 12, b: 32 }, [0, 1], [0, 4.6], { xlabel: "predicted probability ŷ", ylabel: "loss" });
    const ps = Array.from({ length: 99 }, (_, k) => 0.01 + k * 0.01);
    el("path", { d: path(ps.map((p) => [F.x(p), F.y(-Math.log(p))])), class: "draw", pathLength: 1, style: "fill:none;stroke:var(--neg);stroke-width:2.5" }, svg);
    el("path", { d: path(ps.map((p) => [F.x(p), F.y(-Math.log(1 - p))])), class: "draw", pathLength: 1, style: "fill:none;stroke:var(--fwd);stroke-width:2.5" }, svg);
    el("circle", { cx: F.x(yh), cy: F.y(Math.min(4.6, L)), r: 7, style: "fill:var(--hl)" }, svg);
    el("text", { x: F.x(yh) + (yh > 0.6 ? -10 : 10), y: F.y(Math.min(4.6, L)) - 10, "text-anchor": yh > 0.6 ? "end" : "start", style: "fill:var(--ink);font-size:12px" }, svg, `L = ${f3(L)}`);
  }
  function drawLossSurface() {
    if (!hasPlotly()) { noPlotly("loss3d"); noPlotly("loss2d"); return; }
    const nt = net(), Lo = nt.layers[LAST], W = Lo.W[0];
    const ord = [0, 1, 2, 3].sort((a, b) => Math.abs(W[b]) - Math.abs(W[a])), ia = ord[0], ib = ord[1];
    const A4 = Core.forward(nt, Xtr, { train: false }).caches[LAST].Ain;
    const lossAt = (wa, wb) => { let s = 0; A4.forEach((a, n) => { let z = Lo.b[0]; for (let i = 0; i < 4; i++) z += (i === ia ? wa : i === ib ? wb : W[i]) * a[i]; const p = Math.min(1 - 1e-7, Math.max(1e-7, Core.sigmoid(z))); s += -(Ytr[n] * Math.log(p) + (1 - Ytr[n]) * Math.log(1 - p)); }); return s / A4.length; };
    const n = 41, ra = Array.from({ length: n }, (_, k) => W[ia] - 3 + (6 * k) / (n - 1)), rb = Array.from({ length: n }, (_, k) => W[ib] - 3 + (6 * k) / (n - 1));
    const Z = rb.map((b) => ra.map((a) => lossAt(a, b))), cur = lossAt(W[ia], W[ib]);
    const g0a = ra[3], g0b = rb[rb.length - 4], g0z = lossAt(g0a, g0b) + 0.004;
    const c = plotBase(), na = `w(h4·${ia + 1} → ŷ)`, nb = `w(h4·${ib + 1} → ŷ)`;
    Plotly.react("loss3d", [
      { type: "surface", x: ra, y: rb, z: Z, showscale: false, opacity: 0.92, colorscale: [[0, c.fwd], [0.5, c.soft], [1, c.bwd]], contours: { z: { show: true, usecolormap: true, project: { z: true }, width: 1 } }, hovertemplate: "%{x:.2f}, %{y:.2f}: loss %{z:.3f}<extra></extra>" },
      { type: "scatter3d", mode: "markers", x: [W[ia]], y: [W[ib]], z: [cur], marker: { size: 7, color: c.hl, line: { color: c.ink, width: 1 } }, hoverinfo: "skip" },
      { type: "scatter3d", mode: "lines", x: [g0a], y: [g0b], z: [g0z], line: { color: c.ink, width: 6 }, hoverinfo: "skip" },
      { type: "scatter3d", mode: "markers", x: [g0a], y: [g0b], z: [g0z], marker: { size: 8, color: c.bwd, line: { color: c.panel, width: 2 } }, hoverinfo: "skip" },
    ], { margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: c.panel, font: c.font, showlegend: false, scene: { xaxis: axis3(na, c), yaxis: axis3(nb, c), zaxis: axis3("training loss", c), camera: { eye: { x: 1.5, y: -1.5, z: 0.9 } } }, uirevision: "loss3d" }, PCFG);
    const fw = Core.forward(nt, Xtr, { train: false }), g = Core.backward(nt, fw, Ytr).grads[LAST].dW[0];
    const gn = Math.hypot(g[ia], g[ib]) || 1, sc = 1.2 / gn;
    Plotly.react("loss2d", [
      { type: "contour", x: ra, y: rb, z: Z, showscale: false, ncontours: 24, contours: { coloring: "heatmap" }, colorscale: [[0, c.fwd], [0.5, c.soft], [1, c.bwd]], line: { color: c.panel, width: 0.6 }, hovertemplate: "%{x:.2f}, %{y:.2f}: %{z:.3f}<extra></extra>" },
      { type: "scatter", mode: "lines+markers", x: [W[ia], W[ia] - g[ia] * sc], y: [W[ib], W[ib] - g[ib] * sc], line: { color: c.ink, width: 3 }, marker: { size: [12, 7], color: [c.hl, c.ink] }, hoverinfo: "skip" },
      { type: "scatter", mode: "lines+markers", x: [g0a], y: [g0b], line: { color: c.bwd, width: 2.5 }, marker: { size: 5, color: c.bwd }, hoverinfo: "skip" },
      { type: "scatter", mode: "markers", x: [g0a], y: [g0b], marker: { size: 14, color: c.bwd, line: { color: c.panel, width: 2.5 } }, hoverinfo: "skip" },
    ], { margin: { l: 54, r: 10, t: 10, b: 44 }, paper_bgcolor: c.panel, plot_bgcolor: c.panel, font: c.font, showlegend: false, xaxis: { title: { text: na }, gridcolor: c.rule }, yaxis: { title: { text: nb }, gridcolor: c.rule } }, PCFG);
    lossCtx = { ia, ib, W: Array.from(W), b: Lo.b[0], A4, ra, rb, lossAt };
    $("loss3dCap").textContent = `Average training loss as the two largest output weights (${na}, ${nb}) vary, all other ${NPARAM - 2} parameters fixed. Purple: the network now (loss ${f3(cur)}). The line in Fig 6.3 points along the negative gradient: where gradient descent will step.`;
  }

  // ---------------------------------------------------------------- 7. backprop
  function renderGate() {
    const x = +$("gx").value, y = +$("gy").value, z = +$("gz").value;
    $("gxV").textContent = x; $("gyV").textContent = y; $("gzV").textContent = z;
    const q = x + y, f = q * z, svg = $("gateSvg"); clear(svg);
    const pr = (v) => (v < 0 ? `(${v})` : v);
    const node = (cx, cy, label) => { el("circle", { cx, cy, r: 26, style: "fill:var(--soft);stroke:var(--ink);stroke-width:1.5" }, svg); el("text", { x: cx, y: cy + 6, "text-anchor": "middle", style: "fill:var(--ink);font-size:18px" }, svg, label); };
    const edge = (x1, y1, x2, y2, fv, gv) => {
      el("line", { x1, y1, x2, y2, style: "stroke:var(--muted);stroke-width:1.5" }, svg);
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      el("text", { x: mx, y: my - 9, "text-anchor": "middle", style: "fill:var(--fwd);font-size:15px;font-weight:600" }, svg, String(+fv.toFixed(2)));
      el("text", { x: mx, y: my + 21, "text-anchor": "middle", style: "fill:var(--bwd);font-size:15px;font-weight:600" }, svg, String(+gv.toFixed(2)));
    };
    edge(90, 36, 334, 110, x, z); edge(90, 184, 334, 110, y, z); edge(386, 110, 594, 170, q, z); edge(90, 240, 594, 176, z, q); edge(646, 170, 820, 170, f, 1);
    [["x", 60, 36], ["y", 60, 184], ["z", 60, 240]].forEach(([t, cx, cy]) => el("text", { x: cx - 4, y: cy + 6, "text-anchor": "end", style: "fill:var(--ink);font-size:18px" }, svg, t));
    node(360, 110, "+"); node(620, 170, "×");
    el("text", { x: 836, y: 176, style: "fill:var(--ink);font-size:18px" }, svg, "f");
    el("text", { x: 360, y: 158, "text-anchor": "middle" }, svg, "q = x + y");
    el("text", { x: 620, y: 218, "text-anchor": "middle" }, svg, "f = q · z");
    tex("gateMath", `\\begin{aligned} q &= x + y = ${x} + ${tp(y)} = ${q} \\\\ f &= q\\,z = ${q}\\cdot${tp(z)} = ${f} \\\\ \\tfrac{\\partial f}{\\partial z} &= q = ${q}, \\quad \\tfrac{\\partial f}{\\partial q} = z = ${z} \\\\ \\tfrac{\\partial f}{\\partial x} &= \\tfrac{\\partial f}{\\partial q}\\,\\tfrac{\\partial q}{\\partial x} = ${z}\\cdot 1 = ${z} \\end{aligned}`);
  }
  function renderBackprop() {
    const nt = net(), fw = fwP(), y = yP(), bw = Core.backward(nt, fw, [y]);
    const l = S.bpLayer, j = S.bpNeuron, delta = bw.deltas[l][0][j], z = fw.caches[l].Z[0][j], act = actOf(l);
    const dvals = [Array(13).fill(0)].concat(bw.deltas.map((d) => Array.from(d[0], Math.abs)));
    drawNetwork($("deltaFlow"), dvals, { tipLabel: "|δ| =", color: "--bwd", selLayer: l, selNeuron: j, labels: false, outLabel: `δ = ŷ − y = ${f3(fw.out[0] - y)}`, arrow: "← gradients flow from the loss back toward the inputs" });
    heatmap($("gradHeat"), bw.grads[l].dW, j, { pos: "--bwd", neg: "--fwd", rowName: (k) => outName(l, k), colName: (i) => inName(l, i) });
    let up;
    if (l === LAST) {
      tex("deltaBox", `\\delta^{(5)} = \\hat y - y = ${f3(fw.out[0])} - ${y} = ${tn(delta)}`);
      up = `<tr><th>Quantity</th><th class="num">Value</th></tr><tr><td>ŷ</td><td class="num">${f3(fw.out[0])}</td></tr><tr><td>y</td><td class="num">${y}</td></tr><tr class="sel"><td>δ = ŷ − y</td><td class="num">${f3(delta)}</td></tr>`;
    } else {
      const Wn = nt.layers[l + 1].W, dn = bw.deltas[l + 1][0];
      let s = 0;
      up = `<tr><th>Feeds k</th><th class="num">Wₖⱼ</th><th class="num">δₖ</th><th class="num">Wₖⱼ·δₖ</th></tr>`;
      for (let k = 0; k < SIZES[l + 2]; k++) { const p = Wn[k][j] * dn[k]; s += p; up += `<tr><td>${outName(l + 1, k)}</td><td class="num">${f3(Wn[k][j])}</td><td class="num bwd-c">${fe(dn[k])}</td><td class="num">${fe(p)}</td></tr>`; }
      up += `<tr class="sel"><td><b>Σ</b></td><td></td><td></td><td class="num"><b>${fe(s)}</b></td></tr>`;
      const d = act.d(z);
      tex("deltaBox", `\\delta^{(${l + 1})}_{${j + 1}} = \\phi'(z_{${j + 1}})\\sum_k W^{(${l + 2})}_{k${j + 1}}\\,\\delta^{(${l + 2})}_k = ${tn(d)} \\times ${tp(s)} = ${tn(delta)}${d === 0 ? "\\qquad \\text{(inactive unit: } z \\le 0)" : ""}`);
    }
    $("upTable").innerHTML = up;
    const ain = fw.caches[l].Ain[0], g = bw.grads[l];
    let h = `<tr><th>Weight</th><th class="num">aᵢ</th><th class="num">∂L/∂w = δ·aᵢ</th></tr>`;
    for (let i = 0; i < SIZES[l]; i++) h += `<tr><td>w${i + 1} · ${inName(l, i)}</td><td class="num fwd-c">${f3(ain[i])}</td><td class="num bwd-c">${fe(g.dW[j][i])}</td></tr>`;
    h += `<tr class="sel"><td>bias</td><td class="num">1</td><td class="num bwd-c">${fe(g.db[j])}</td></tr>`;
    $("gradTable").innerHTML = h;
  }
  function renderGradByAct() {
    const svg = $("gradBars"); clear(svg);
    const keys = [["relu", "--fwd"], ["leaky", "--pos"], ["tanh", "--hl"], ["sigmoid", "--neg"]];
    const init = S.init === "zeros" ? "he" : S.init, Xb = Xtr.slice(0, 64), Yb = Ytr.slice(0, 64);
    const res = keys.map(([k]) => {
      const n2 = Core.create(SIZES, { init, act: k, seed: 7 }), fw = Core.forward(n2, Xb, { train: false }), gr = Core.backward(n2, fw, Yb).grads;
      return gr.map((q) => { const v = q.dW.flatMap((r) => Array.from(r, Math.abs)); return v.reduce((a, b) => a + b, 0) / v.length; });
    });
    const all = res.flat().filter((v) => v > 0);
    const lo = 10 ** Math.floor(Math.log10(Math.min(...all))), hi = 10 ** Math.ceil(Math.log10(Math.max(...all)));
    const [W, H] = vb(svg), F = frame(svg, W, H, { l: 56, r: 12, t: 14, b: 30 }, [0, 5], [lo, hi], { ylog: true, xticks: [] });
    const gw = (W - 68) / 5, bw = (gw - 40) / 4;
    for (let L = 0; L < 5; L++) {
      keys.forEach(([, c], a) => { const v = Math.max(res[a][L], lo), x = F.x(L) + 20 + a * bw; el("rect", { class: "gb", x, y: F.y(v), width: bw - 4, height: Math.max(1, F.y(lo) - F.y(v)), style: `fill:var(${c});fill-opacity:.85` }, svg); });
      el("text", { x: F.x(L) + gw / 2, y: H - 10, "text-anchor": "middle", style: "fill:var(--ink)" }, svg, `layer ${L + 1}${L === 0 ? " (first)" : L === 4 ? " (output)" : ""}`);
    }
    $("gradLegend").innerHTML = keys.map(([k, c]) => `<span><i style="background:var(${c})"></i>${Core.ACT[k].name}</span>`).join("") + (S.init === "zeros" ? "<span>(He init used here: zero weights give no gradient)</span>" : "");
  }

  // ---------------------------------------------------------------- 8. update
  function renderUpdate() {
    $("lrUpdV").textContent = S.lrUpd.toFixed(2);
    const nt = net(), fw = fwP(), y = yP(), bw = Core.backward(nt, fw, [y]);
    const l = S.bpLayer, j = S.bpNeuron, Wr = nt.layers[l].W[j], g = bw.grads[l].dW[j], eta = S.lrUpd, n = SIZES[l];
    const olds = Array.from(Wr).concat([nt.layers[l].b[j]]), grads = Array.from(g).concat([bw.grads[l].db[j]]), news = olds.map((v, i) => v - eta * grads[i]);
    {
      const svg = $("dumbSvg"); clear(svg);
      const [W, H] = vb(svg), mx = Math.max(0.5, ...olds.map(Math.abs), ...news.map(Math.abs)) * 1.1;
      const x = lin(-mx, mx, 86, W - 12), rh = (H - 40) / olds.length;
      niceTicks(-mx, mx, 4).forEach((t) => { el("line", { x1: x(t), x2: x(t), y1: 6, y2: H - 30, class: t === 0 ? "ax" : "grid" }, svg); el("text", { x: x(t), y: H - 16, "text-anchor": "middle" }, svg, String(t)); });
      olds.forEach((o, i) => {
        const yy = 10 + i * rh + rh / 2, nv = news[i];
        el("text", { x: 80, y: yy + 4, "text-anchor": "end", style: "fill:var(--ink);font-size:10.5px" }, svg, i === n ? "bias" : `w${i + 1} ${inName(l, i)}`);
        el("line", { x1: x(o), x2: x(nv), y1: yy, y2: yy, style: "stroke:var(--bwd);stroke-width:2" }, svg);
        el("circle", { cx: x(o), cy: yy, r: 4, style: "fill:var(--c1)" }, svg);
        if (Math.abs(x(nv) - x(o)) > 3) el("path", { d: `M${x(nv)},${yy} l${nv > o ? -7 : 7},-4 l0,8 Z`, style: "fill:var(--bwd)" }, svg);
        else el("circle", { cx: x(nv), cy: yy, r: 3, style: "fill:var(--bwd)" }, svg);
      });
      el("text", { x: W - 12, y: H - 2, "text-anchor": "end" }, svg, "grey: old · orange arrow: new");
    }
    const n2 = Core.clone(nt);
    Core.makeOptimizer("sgd", eta).step(n2, bw.grads);
    const yh0 = fw.out[0], yh1 = Core.forward(n2, [xP()], { train: false }).out[0], L0 = Core.bce([yh0], [y]), L1 = Core.bce([yh1], [y]);
    kpis("updStats", [[`${f3(L0)} → ${f3(L1)}`, "loss, before → after one step"]]);
    $("updNote").textContent = L1 < L0 ? `All ${NPARAM} parameters stepped at once; ŷ moved from ${f3(yh0)} to ${f3(yh1)}.` : `η = ${eta.toFixed(2)} overshot: the loss went up. Lower η.`;
    {
      const svg = $("lineSearch"); clear(svg);
      const tmax = Math.max(2 * eta, 0.6), ts = Array.from({ length: 41 }, (_, k) => (tmax * k) / 40);
      const Ls = ts.map((t) => { const q = Core.clone(nt); Core.makeOptimizer("sgd", t).step(q, bw.grads); return Core.bce(Core.forward(q, [xP()], { train: false }).out, [y]); });
      const [W, H] = vb(svg), hi = Math.max(...Ls), lo = Math.min(...Ls);
      const F = frame(svg, W, H, { l: 44, r: 12, t: 16, b: 34 }, [0, tmax], [Math.max(0, lo - (hi - lo) * 0.1), hi + (hi - lo) * 0.15 + 1e-3], { xlabel: "step size t", ylabel: "loss" });
      el("path", { d: path(ts.map((t, k) => [F.x(t), F.y(Ls[k])])), style: "fill:none;stroke:var(--bwd);stroke-width:2.5" }, svg);
      el("circle", { cx: F.x(0), cy: F.y(L0), r: 5, style: "fill:var(--c1)" }, svg);
      el("circle", { cx: F.x(eta), cy: F.y(L1), r: 7, style: "fill:var(--hl)" }, svg);
      el("text", { x: F.x(eta) + (eta > tmax * 0.6 ? -8 : 8), y: F.y(L1) - 8, "text-anchor": eta > tmax * 0.6 ? "end" : "start", style: "fill:var(--hl);font-weight:600" }, svg, `η = ${eta.toFixed(2)}`);
    }
    let h = `<tr><th>Parameter</th><th class="num">old</th><th class="num">∂L/∂w</th><th class="num">new = old − η·∂L/∂w</th></tr>`;
    olds.forEach((o, i) => { h += `<tr><td>${i === n ? "bias" : `w${i + 1} · ${inName(l, i)}`}</td><td class="num">${f3(o)}</td><td class="num bwd-c">${fe(grads[i])}</td><td class="num"><b>${f3(news[i])}</b></td></tr>`; });
    $("updTable").innerHTML = h;
  }

  // ---------------------------------------------------------------- 9. optimizers
  const SURF = {
    ravine: { f: (x, y) => 0.1 * x * x + 2 * y * y, g: (x, y) => [0.2 * x, 4 * y], start: [-4.5, 1.6], xr: [-5, 5], yr: [-2, 2], mins: [[0, 0]], cap: 12, lr: 0.02, steps: 150,
      note: "A long, narrow valley: steep across, gentle along. SGD bounces between the walls; momentum builds speed along the floor." },
    saddle: { f: (x, y) => 0.5 * x * x - 0.5 * y * y + 0.125 * y ** 4, g: (x, y) => [x, -y + 0.5 * y ** 3], start: [-2.5, 0.002], xr: [-3, 3], yr: [-2.2, 2.2], mins: [[0, Math.SQRT2], [0, -Math.SQRT2]], cap: 4, lr: 0.05, steps: 160,
      note: "A saddle point at the centre: a minimum along w₁ but a maximum along w₂. The gradient there is almost zero, so plain SGD lingers for 130+ steps; momentum, RMSprop and Adam escape fast." },
    plateau: { f: (x, y) => 1 - Math.exp(-(x * x + 2 * y * y) / 2), g: (x, y) => { const e = Math.exp(-(x * x + 2 * y * y) / 2); return [x * e, 2 * y * e]; }, start: [3.4, 2.0], xr: [-4, 4], yr: [-3, 3], mins: [[0, 0]], cap: 1.05, lr: 0.1, steps: 150,
      note: "A plateau: the start is on a nearly flat table where the gradient is about 0.0001. SGD and momentum barely move; RMSprop and Adam divide by the gradient's own size, so they keep walking." },
    twowell: { f: (x, y) => (x * x - 1) ** 2 + 0.5 * y * y + 0.25 * x, g: (x, y) => [4 * x * (x * x - 1) + 0.25, y], start: [2, 1.6], xr: [-2, 2.3], yr: [-2, 2], mins: [[-1.03, 0]], cap: 12, lr: 0.02, steps: 150,
      note: "Two valleys: a shallow local minimum on the right and a deeper global one on the left. Momentum carries enough speed to roll over the ridge; the others settle in the nearer valley." },
    himmel: { f: (x, y) => ((x * x + y - 11) ** 2 + (x + y * y - 7) ** 2) / 100, g: (x, y) => [(4 * x * (x * x + y - 11) + 2 * (x + y * y - 7)) / 100, (2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)) / 100], start: [-0.4, -0.6], xr: [-5, 5], yr: [-5, 5], mins: [[3, 2], [-2.805, 3.131], [-3.779, -3.283], [3.584, -1.848]], cap: 9, lr: 0.05, steps: 150,
      note: "Himmelblau's function has four equally good minima around a central hill. Which one an optimizer reaches depends on its path; real networks have many equivalent solutions too." },
    banana: { f: (x, y) => (1 - x) ** 2 + 5 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 20 * x * (y - x * x), 10 * (y - x * x)], start: [-1.5, 2.2], xr: [-2, 2], yr: [-1, 3.2], mins: [[1, 1]], cap: 40, lr: 0.02, steps: 200,
      note: "Rosenbrock's curved valley: finding the valley is easy, following its flat, curving floor to (1, 1) is slow." },
    bumpy: { f: (x, y) => 0.05 * (x * x + y * y) + 0.12 * (2 - Math.cos(3 * x) - Math.cos(3 * y)), g: (x, y) => [0.1 * x + 0.36 * Math.sin(3 * x), 0.1 * y + 0.36 * Math.sin(3 * y)], start: [4.2, 3.6], xr: [-5, 5], yr: [-5, 5], mins: [[0, 0]], cap: 4, lr: 0.1, steps: 200,
      note: "Many small local minima on a big bowl. SGD stops in the first dip, RMSprop and Adam in a nearby one; momentum and Nesterov roll over the bumps to the bottom." },
  };
  const OPTS = [["sgd", "--c1"], ["momentum", "--fwd"], ["nesterov", "--hl"], ["rmsprop", "--pos"], ["adam", "--bwd"]];
  const optOn = { sgd: true, momentum: true, nesterov: false, rmsprop: true, adam: true };
  let optPaths = {}, optFrame = 0, optTimer = null, optField = true, optMap = null;
  const surf = () => SURF[$("surfSel").value];
  function computePaths() {
    const s = surf(), lr = +$("optLr").value, beta = +$("optBeta").value, steps = +$("optSteps").value;
    optPaths = {};
    OPTS.forEach(([k]) => {
      const p = Float64Array.from(s.start), o = Core.makeOptimizer(k, lr, beta), pts = [[p[0], p[1]]];
      for (let t = 0; t < steps; t++) { o.stepArrays([[p, Float64Array.from(s.g(p[0], p[1]))]]); if (!isFinite(p[0]) || Math.abs(p[0]) > 50 || Math.abs(p[1]) > 50) break; pts.push([p[0], p[1]]); }
      optPaths[k] = pts;
    });
    $("surfNote").textContent = s.note;
  }
  const clipPt = (s, q) => [Math.max(s.xr[0], Math.min(s.xr[1], q[0])), Math.max(s.yr[0], Math.min(s.yr[1], q[1]))];
  function drawOptBase() {
    if (!hasPlotly()) { noPlotly("opt2d"); noPlotly("opt3d"); return; }
    const s = surf(), c = plotBase(), n = 80;
    const xs = Array.from({ length: n }, (_, i) => s.xr[0] + ((s.xr[1] - s.xr[0]) * i) / (n - 1)), ys = Array.from({ length: n }, (_, i) => s.yr[0] + ((s.yr[1] - s.yr[0]) * i) / (n - 1));
    const Z = ys.map((y) => xs.map((x) => Math.min(s.cap, s.f(x, y))));
    const on = OPTS.filter(([k]) => optOn[k]);
    const t2 = [{ type: "contour", x: xs, y: ys, z: Z, showscale: false, ncontours: 26, contours: { coloring: "heatmap" }, colorscale: [[0, c.fwd], [0.45, c.soft], [1, c.panel]], line: { color: c.rule, width: 0.6 }, hoverinfo: "skip" }];
    // gradient arrow field
    const fx = [], fy = [];
    if (optField) {
      const m = 15, dx = (s.xr[1] - s.xr[0]) / m, dy = (s.yr[1] - s.yr[0]) / m;
      for (let i = 0; i <= m; i++) for (let k = 0; k <= m; k++) {
        const x = s.xr[0] + i * dx, y = s.yr[0] + k * dy, g = s.g(x, y), gn = Math.hypot(g[0], g[1]);
        if (gn < 1e-9) continue;
        const L = 0.38 * Math.min(dx, dy) * (0.35 + 0.65 * Math.min(1, gn / 2));
        const ex = x - (g[0] / gn) * L * (dx / Math.min(dx, dy)), ey = y - (g[1] / gn) * L * (dy / Math.min(dx, dy));
        fx.push(x, ex, null); fy.push(y, ey, null);
      }
    }
    t2.push({ type: "scatter", mode: "lines", x: fx, y: fy, line: { color: c.muted, width: 1 }, opacity: 0.5, hoverinfo: "skip", showlegend: false });
    t2.push({ type: "scatter", mode: "markers", x: s.mins.map((m) => m[0]), y: s.mins.map((m) => m[1]), marker: { size: 13, symbol: "star", color: c.ink }, hoverinfo: "skip", showlegend: false });
    t2.push({ type: "scatter", mode: "markers", x: [s.start[0]], y: [s.start[1]], marker: { size: 13, color: c.ink, symbol: "circle-open", line: { width: 2.5 } }, hoverinfo: "skip", showlegend: false });
    const base2 = t2.length;
    on.forEach(([k, col]) => {
      t2.push({ type: "scatter", mode: "lines", x: [s.start[0]], y: [s.start[1]], name: Core.OPT[k].name, line: { color: cssv(col), width: 3 }, hoverinfo: "skip" });
      t2.push({ type: "scatter", mode: "markers", x: [s.start[0]], y: [s.start[1]], marker: { size: 15, color: cssv(col), line: { color: c.panel, width: 2.5 } }, showlegend: false, hoverinfo: "skip" });
    });
    Plotly.react("opt2d", t2, { margin: { l: 44, r: 10, t: 10, b: 40 }, paper_bgcolor: c.panel, plot_bgcolor: c.panel, font: c.font, xaxis: { title: { text: "w₁" }, gridcolor: c.rule, zeroline: false, range: s.xr }, yaxis: { title: { text: "w₂" }, gridcolor: c.rule, zeroline: false, range: s.yr }, legend: { orientation: "h", y: -0.2, font: c.font } }, PCFG);
    const t3 = [{ type: "surface", x: xs, y: ys, z: Z, showscale: false, opacity: 0.9, colorscale: [[0, c.fwd], [0.5, c.soft], [1, c.bwd]], contours: { z: { show: true, usecolormap: true, project: { z: true }, width: 1 } }, lighting: { ambient: 0.7, diffuse: 0.6, specular: 0.25, roughness: 0.6 }, hoverinfo: "skip" }];
    t3.push({ type: "scatter3d", mode: "markers", x: s.mins.map((m) => m[0]), y: s.mins.map((m) => m[1]), z: s.mins.map((m) => Math.min(s.cap, s.f(m[0], m[1])) + s.cap * 0.02), marker: { size: 5, symbol: "diamond", color: c.ink }, hoverinfo: "skip", showlegend: false });
    const base3 = t3.length;
    on.forEach(([, col]) => {
      const z0 = [Math.min(s.cap, s.f(s.start[0], s.start[1])) + s.cap * 0.015];
      t3.push({ type: "scatter3d", mode: "lines", x: [s.start[0]], y: [s.start[1]], z: z0, line: { color: cssv(col), width: 7 }, showlegend: false, hoverinfo: "skip" });
      t3.push({ type: "scatter3d", mode: "markers", x: [s.start[0]], y: [s.start[1]], z: z0, marker: { size: 7, color: cssv(col), line: { color: c.panel, width: 2 } }, showlegend: false, hoverinfo: "skip" });
    });
    Plotly.react("opt3d", t3, { margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: c.panel, font: c.font, scene: { xaxis: axis3("w₁", c), yaxis: axis3("w₂", c), zaxis: axis3("loss", c), camera: { eye: { x: -1.45, y: -1.55, z: 1.1 } } }, uirevision: $("surfSel").value }, PCFG);
    optMap = { on: on.map(([k]) => k), base2, base3 };
    setOptFrame(optFrame);
  }
  function setOptFrame(k) {
    optFrame = k;
    const s = surf(), steps = +$("optSteps").value;
    $("optStepV").textContent = `step ${Math.min(k, steps)} / ${steps}`;
    if (hasPlotly() && optMap) {
      const d2 = { x: [], y: [] }, i2 = [], d3 = { x: [], y: [], z: [] }, i3 = [];
      optMap.on.forEach((key, n) => {
        const p = optPaths[key].slice(0, Math.min(optPaths[key].length, k + 1)).map((q) => clipPt(s, q)), h = p[p.length - 1];
        const zz = (q) => Math.min(s.cap, s.f(q[0], q[1])) + s.cap * 0.015;
        d2.x.push(p.map((q) => q[0]), [h[0]]); d2.y.push(p.map((q) => q[1]), [h[1]]); i2.push(optMap.base2 + 2 * n, optMap.base2 + 2 * n + 1);
        d3.x.push(p.map((q) => q[0]), [h[0]]); d3.y.push(p.map((q) => q[1]), [h[1]]); d3.z.push(p.map(zz), [zz(h)]); i3.push(optMap.base3 + 2 * n, optMap.base3 + 2 * n + 1);
      });
      if (i2.length) { Plotly.restyle("opt2d", d2, i2); Plotly.restyle("opt3d", d3, i3); }
    }
    // scoreboard
    let h = `<tr><th>Optimizer</th><th class="num">Step</th><th class="num">Loss</th><th class="num">|gradient|</th><th>Status</th></tr>`;
    OPTS.filter(([key]) => optOn[key]).forEach(([key, c]) => {
      const P = optPaths[key], idx = Math.min(P.length - 1, k), q = P[idx], g = s.g(q[0], q[1]), gn = Math.hypot(g[0], g[1]);
      const nearMin = s.mins.some((m) => Math.hypot(q[0] - m[0], q[1] - m[1]) < 0.15);
      const status = P.length - 1 < steps && idx >= P.length - 1 ? "diverged" : nearMin ? "at a global minimum" : gn < 2e-3 ? (idx < 3 ? "flat start" : "stuck: flat or local min") : "moving";
      h += `<tr><td><span class="bar" style="width:14px;height:4px;background:var(${c});margin-right:8px"></span>${Core.OPT[key].name}</td><td class="num">${idx}</td><td class="num">${f3(s.f(q[0], q[1]))}</td><td class="num">${gn < 1e-3 ? gn.toExponential(1) : f3(gn)}</td><td>${status}</td></tr>`;
    });
    $("optTable").innerHTML = h;
    drawOptCurve(k);
  }
  function drawOptCurve(upto) {
    const svg = $("optCurve"); clear(svg);
    const s = surf(), steps = +$("optSteps").value;
    let fmin = Infinity;
    for (let i = 0; i <= 120; i++) for (let k = 0; k <= 120; k++) fmin = Math.min(fmin, s.f(s.xr[0] + ((s.xr[1] - s.xr[0]) * i) / 120, s.yr[0] + ((s.yr[1] - s.yr[0]) * k) / 120));
    s.mins.forEach((m) => (fmin = Math.min(fmin, s.f(m[0], m[1]))));
    fmin -= 1e-4;
    const series = OPTS.filter(([k]) => optOn[k]).map(([k, c]) => ({ c, values: optPaths[k].map((q) => Math.max(1e-4, s.f(q[0], q[1]) - fmin)) }));
    if (!series.length) return;
    const all = series.flatMap((q) => q.values);
    const [W, H] = vb(svg), F = frame(svg, W, H, { l: 50, r: 10, t: 10, b: 32 }, [0, steps], [10 ** Math.floor(Math.log10(Math.min(...all))), 10 ** Math.ceil(Math.log10(Math.max(...all)))], { ylog: true, xlabel: "step", ylabel: "loss − best" });
    series.forEach((q) => {
      const v = q.values.slice(0, Math.min(q.values.length, upto + 1));
      el("path", { d: path(v.map((t, i) => [F.x(i), F.y(t)])), style: `fill:none;stroke:var(${q.c});stroke-width:2.2` }, svg);
      if (v.length) el("circle", { cx: F.x(v.length - 1), cy: F.y(v[v.length - 1]), r: 4, style: `fill:var(${q.c})` }, svg);
    });
  }
  function optPlay() {
    const btn = $("optPlay"), steps = +$("optSteps").value;
    if (optTimer) { cancelAnimationFrame(optTimer); optTimer = null; btn.textContent = "▶ Race"; return; }
    if (reduceMotion()) { setOptFrame(steps); return; }
    if (optFrame >= steps) optFrame = 0;
    btn.textContent = "❚❚ Pause";
    let last = 0;
    const tick = (t) => {
      if (t - last > 33) { last = t; setOptFrame(optFrame + Math.max(1, Math.round(steps / 150))); }
      if (optFrame >= steps) { optTimer = null; btn.textContent = "▶ Race"; return; }
      optTimer = requestAnimationFrame(tick);
    };
    optTimer = requestAnimationFrame(tick);
  }
  function drawCrit() {
    const svg = $("critSvg"); clear(svg);
    const panels = [["Minimum", "λ₁ > 0, λ₂ > 0", (x, y) => x * x + 1.6 * y * y], ["Saddle point", "λ₁ > 0, λ₂ < 0", (x, y) => x * x - y * y], ["Plateau", "λ₁ ≈ 0, λ₂ ≈ 0", (x, y) => -Math.exp(-((x + 0.55) ** 2 + (y + 0.45) ** 2) * 7)], ["Maximum", "λ₁ < 0, λ₂ < 0", (x, y) => -(x * x + 1.6 * y * y)]];
    panels.forEach(([name, ev, f], p) => {
      const g = el("g", { transform: `translate(${p * 260},0)` }, svg), cx = 125, cy = 105, R = 80;
      // contour sketch by marching squares-lite: draw iso-points
      const levels = name === "Saddle point" ? [-0.6, -0.3, -0.1, 0.1, 0.3, 0.6] : name === "Plateau" ? [-0.85, -0.55, -0.3, -0.12] : [0.1, 0.3, 0.6, 1.0].map((v) => (name === "Maximum" ? -v : v));
      const N = 60;
      levels.forEach((lv) => {
        for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) {
          const x = -1 + (2 * i) / N, y = -1 + (2 * k) / N, d = 2 / N;
          const a = f(x, y) - lv, b = f(x + d, y) - lv, c2 = f(x, y + d) - lv;
          if (a * b < 0 || a * c2 < 0) el("rect", { x: cx + x * R, y: cy - y * R, width: 1.6, height: 1.6, style: `fill:var(${lv > 0 ? "--bwd" : "--fwd"})` }, g);
        }
      });
      el("circle", { cx, cy, r: 5, style: "fill:var(--ink)" }, g);
      el("rect", { x: cx - R - 6, y: cy - R - 6, width: 2 * R + 12, height: 2 * R + 12, rx: 10, style: "fill:none;stroke:var(--rule)" }, g);
      el("text", { x: cx, y: 214, "text-anchor": "middle", style: "fill:var(--ink);font-size:14px;font-weight:600" }, g, name);
      el("text", { x: cx, y: 232, "text-anchor": "middle" }, g, ev);
    });
  }
  function initOpt() {
    $("optToggles").innerHTML = OPTS.map(([k, c]) => `<button class="btn" data-opt="${k}" aria-pressed="${optOn[k]}"><span class="bar" style="width:14px;height:4px;background:var(${c});margin-right:6px"></span>${Core.OPT[k].name}</button>`).join("");
    $("optToggles").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { optOn[b.dataset.opt] = !optOn[b.dataset.opt]; b.setAttribute("aria-pressed", String(optOn[b.dataset.opt])); drawOptBase(); }));
    const stop = () => { if (optTimer) { cancelAnimationFrame(optTimer); optTimer = null; $("optPlay").textContent = "▶ Race"; } };
    const redo = (full) => { stop(); $("optBetaV").textContent = (+$("optBeta").value).toFixed(2); $("optStepsV").textContent = $("optSteps").value; computePaths(); optFrame = full ? +$("optSteps").value : 0; drawOptBase(); };
    $("surfSel").addEventListener("change", () => { const s = surf(); $("optLr").value = String(s.lr); $("optSteps").value = s.steps; redo(false); optPlay(); });
    $("optLr").addEventListener("change", () => redo(false));
    ["optBeta", "optSteps"].forEach((id) => $(id).addEventListener("input", () => redo(true)));
    $("optPlay").addEventListener("click", optPlay);
    $("optReset").addEventListener("click", () => { stop(); setOptFrame(0); });
    $("optField").addEventListener("click", (e) => { optField = !optField; e.target.setAttribute("aria-pressed", String(optField)); drawOptBase(); });
    const s0 = surf(); $("optLr").value = String(s0.lr); $("optSteps").value = s0.steps;
    redo(true);
    drawCrit();
    $("batchTable").innerHTML = `<tr><th>Method</th><th class="num">|B|</th><th class="num">Updates / epoch</th></tr>` + [["Batch GD", 242, 1], ["Mini-batch", 32, 8], ["Mini-batch", 8, 31], ["SGD", 1, 242]].map(([a, b, c]) => `<tr><td>${a}</td><td class="num">${b}</td><td class="num">${c}</td></tr>`).join("");
    // start the race when the section first comes into view
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); if (!reduceMotion()) { optFrame = 0; optPlay(); } } }, { threshold: 0.35 });
      io.observe($("opt3d"));
    }
  }
  function renderBatchCurve() {
    const runs = [[242, "--c1"], [1, "--bwd"], [32, "--fwd"]].map(([bs, c]) => {
      const n2 = Core.create(SIZES, { init: S.init, act: S.act, seed: 7 }), opt = Core.makeOptimizer("sgd", 0.05), r = Core.rng(3), vals = [Core.evaluate(n2, Xtr, Ytr).loss];
      for (let e = 0; e < 20; e++) { Core.trainEpoch(n2, opt, Xtr, Ytr, bs, r); vals.push(Core.evaluate(n2, Xtr, Ytr).loss); }
      return { c, values: vals };
    });
    lineChart($("batchCurve"), runs, { xMax: 20, yMin: 0, yMax: Math.min(3, Math.max(0.8, ...runs.flatMap((r) => r.values).filter(isFinite)) * 1.05), ylabel: "training loss", draw: true });
  }

  // ---------------------------------------------------------------- 10. training
  const RUNS = [{ name: "Plain", c: "--c1", bn: false, drop: 0 }, { name: "+ Batch norm", c: "--fwd", bn: true, drop: 0 }, { name: "+ Dropout", c: "--bwd", bn: false, drop: 0.3 }, { name: "+ Both", c: "--pos", bn: true, drop: 0.3 }];
  let TR = null, trToken = 0;
  function lineChart(svg, series, opt) {
    clear(svg);
    const [W, H] = vb(svg), vals = series.flatMap((s) => s.values).filter((v) => isFinite(v));
    let lo = opt.yMin !== undefined ? opt.yMin : Math.min(...vals), hi = opt.yMax !== undefined ? opt.yMax : Math.max(...vals);
    if (opt.ylog) { lo = 10 ** Math.floor(Math.log10(Math.max(1e-9, Math.min(...vals.filter((v) => v > 0))))); hi = 10 ** Math.ceil(Math.log10(Math.max(...vals, 1e-9))); }
    if (!(hi > lo)) hi = lo + 1;
    const F = frame(svg, W, H, { l: 44, r: 10, t: 10, b: 32 }, [0, opt.xMax], [lo, hi], { xlabel: "epoch", ylabel: opt.ylabel, ylog: opt.ylog, yfmt: opt.yfmt, xn: 5 });
    series.forEach((s) => {
      const pts = s.values.map((v, i) => [i, v]).filter((q) => isFinite(q[1])).map(([i, v]) => [F.x(i), F.y(Math.min(hi, Math.max(lo, v)))]);
      if (!pts.length) return;
      el("path", { d: path(pts), class: opt.draw ? "draw" : "", pathLength: 1, style: `fill:none;stroke:var(${s.c});stroke-width:2.2` }, svg);
      el("circle", { cx: pts[pts.length - 1][0], cy: pts[pts.length - 1][1], r: 3.5, style: `fill:var(${s.c})` }, svg);
    });
  }
  function runTraining() {
    const token = ++trToken;
    const E = +$("trEpochs").value, B = +$("trBatch").value, ot = $("trOpt").value, lr = +$("trLr").value, R = +$("trRows").value;
    const X = Xtr.slice(0, R), Y = Ytr.slice(0, R), bs = Math.min(B, R);
    TR = RUNS.map((cfg) => {
      const n = Core.create(SIZES, { init: S.init, act: S.act, seed: 7, bn: cfg.bn, drop: cfg.drop });
      const e0 = Core.evaluate(n, X, Y), v0 = Core.evaluate(n, Xte, Yte);
      return { cfg, net: n, before: Core.clone(n), opt: Core.makeOptimizer(ot, lr), r: Core.rng(1), trL: [e0.loss], trA: [e0.acc], vaL: [v0.loss], vaA: [v0.acc], gn: [[], [], [], [], []] };
    });
    $("trRun").disabled = true;
    let e = 0;
    const step = () => {
      if (token !== trToken) return;
      e++;
      TR.forEach((t) => {
        const n = t.net, idx = X.map((_, i) => i);
        for (let i = idx.length - 1; i > 0; i--) { const k = Math.floor(t.r() * (i + 1)); [idx[i], idx[k]] = [idx[k], idx[i]]; }
        const acc = [0, 0, 0, 0, 0]; let nb = 0;
        for (let s = 0; s < idx.length; s += bs) {
          const b = idx.slice(s, s + bs);
          if (n.bn && b.length < 2) continue;
          const fw = Core.forward(n, b.map((i) => X[i]), { train: true, r: t.r });
          const { grads } = Core.backward(n, fw, b.map((i) => Y[i]));
          grads.forEach((g, l) => { let q = 0; g.dW.forEach((row) => row.forEach((v) => (q += v * v))); acc[l] += Math.sqrt(q); });
          nb++; t.opt.step(n, grads);
        }
        acc.forEach((v, l) => t.gn[l].push(v / Math.max(1, nb)));
        const et = Core.evaluate(n, X, Y), ev = Core.evaluate(n, Xte, Yte);
        t.trL.push(et.loss); t.trA.push(et.acc); t.vaL.push(ev.loss); t.vaA.push(ev.acc);
      });
      $("trEpochLbl").textContent = `epoch ${e} / ${E}`;
      drawTraining(E);
      if (e < E) setTimeout(step, reduceMotion() ? 0 : 90);
      else { $("trRun").disabled = false; summarise(); drawGradNorms(); renderResults(); }
    };
    drawTraining(E);
    setTimeout(step, 30);
  }
  function drawTraining(E) {
    const top = Math.max(1, ...TR.flatMap((t) => t.vaL.concat(t.trL)).filter(isFinite));
    lineChart($("chTrain"), TR.map((t) => ({ c: t.cfg.c, values: t.trL })), { xMax: E, yMin: 0, yMax: Math.min(3, top) });
    lineChart($("chVal"), TR.map((t) => ({ c: t.cfg.c, values: t.vaL })), { xMax: E, yMin: 0, yMax: Math.min(3, top) });
    lineChart($("chAcc"), TR.map((t) => ({ c: t.cfg.c, values: t.vaA })), { xMax: E, yMin: 0.4, yMax: 1, yfmt: (v) => Math.round(v * 100) + "%" });
    let h = `<tr><th>Run</th><th class="num">Train loss</th><th class="num">Test loss</th><th class="num">Best test (epoch)</th><th class="num">Train acc</th><th class="num">Test acc</th></tr>`;
    TR.forEach((t) => {
      const n = t.trL.length - 1, best = t.vaL.reduce((b, v, i) => (v < t.vaL[b] ? i : b), 0);
      h += `<tr><td><span class="bar" style="width:14px;height:4px;background:var(${t.cfg.c});margin-right:8px"></span>${t.cfg.name}</td><td class="num">${f3(t.trL[n])}</td><td class="num">${f3(t.vaL[n])}</td><td class="num">${f3(t.vaL[best])} (${best})</td><td class="num">${(t.trA[n] * 100).toFixed(1)}%</td><td class="num">${(t.vaA[n] * 100).toFixed(1)}%</td></tr>`;
    });
    $("trTable").innerHTML = h;
  }
  function summarise() {
    const p = TR[0], b = TR[3], n = p.trL.length - 1, pb = p.vaL.reduce((bi, v, i) => (v < p.vaL[bi] ? i : bi), 0);
    const rising = p.vaL[n] > p.vaL[pb] * 1.15;
    $("trSummary").innerHTML = rising
      ? `<b>Overfitting, visible.</b> Plain: test loss bottomed at ${f3(p.vaL[pb])} (epoch ${pb}) then rose to ${f3(p.vaL[n])} while training accuracy reached ${(p.trA[n] * 100).toFixed(0)}%. With batch norm and dropout: test loss ${f3(b.vaL[n])}, test accuracy ${(b.vaA[n] * 100).toFixed(0)}%.`
      : `Plain ends at test loss ${f3(p.vaL[n])}; batch norm + dropout at ${f3(b.vaL[n])}. Try 60 training patients or more epochs to provoke overfitting.`;
  }
  function drawGradNorms() {
    if (!TR) return;
    const t = TR[+$("gnRun").value], cols = ["--c1", "--fwd", "--hl", "--pos", "--bwd"];
    lineChart($("chGrad"), t.gn.map((v, l) => ({ c: cols[l], values: [NaN].concat(v) })), { xMax: Math.max(1, t.gn[0].length), ylog: true });
    $("gnLegend").innerHTML = cols.map((c, l) => `<span><i style="background:var(${c})"></i>layer ${l + 1}</span>`).join("");
  }

  // ---------------------------------------------------------------- 11. results
  function pca(Xs, k) {
    const n = Xs.length, d = Xs[0].length, mu = Array(d).fill(0);
    Xs.forEach((x) => x.forEach((v, j) => (mu[j] += v / n)));
    const Xc = Xs.map((x) => Array.from(x, (v, j) => v - mu[j]));
    const C = Array.from({ length: d }, () => Array(d).fill(0));
    Xc.forEach((x) => { for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += (x[a] * x[b]) / n; });
    const comps = [];
    for (let c = 0; c < Math.min(k, d); c++) {
      let v = Array.from({ length: d }, (_, i) => 1 + i * 0.37 + c);
      for (let it = 0; it < 300; it++) {
        let w = C.map((row) => row.reduce((s, q, j) => s + q * v[j], 0));
        comps.forEach((u) => { const p = w.reduce((s, q, j) => s + q * u[j], 0); w = w.map((q, j) => q - p * u[j]); });
        const nr = Math.hypot(...w) || 1; v = w.map((q) => q / nr);
      }
      comps.push(v);
    }
    return Xc.map((x) => comps.map((u) => x.reduce((s, q, j) => s + q * u[j], 0)));
  }
  function scatter(svg, P, Y) {
    clear(svg);
    const xs = P.map((p) => p[0]), ys = P.map((p) => p[1]), pad = (a, b) => { const s = b - a || 1; return [a - s * 0.08, b + s * 0.08]; };
    const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
    const x = lin(x0, x1, 12, 218), y = lin(y0, y1, 218, 12);
    el("rect", { x: 6, y: 6, width: 218, height: 218, rx: 6, style: "fill:none;stroke:var(--rule)" }, svg);
    P.forEach((p, i) => el("circle", { cx: x(p[0]), cy: y(p[1]), r: 4, style: `fill:var(${Y[i] ? "--neg" : "--fwd"});fill-opacity:.75` }, svg));
  }
  function renderResults() {
    if (!TR) return;
    const t = TR[+$("resRun").value];
    $("resRunName").textContent = t.cfg.name;
    const p = Core.forward(t.net, Xte, { train: false }).out;
    let TP = 0, FN = 0, FP = 0, TN = 0;
    p.forEach((q, i) => { const pr = q >= 0.5 ? 1 : 0; if (Yte[i] && pr) TP++; else if (Yte[i]) FN++; else if (pr) FP++; else TN++; });
    const svg = $("cmSvg"); clear(svg);
    const mx = Math.max(TN, FP, FN, TP, 1);
    [[TN, "TN", 0, 0], [FP, "FP", 1, 0], [FN, "FN", 0, 1], [TP, "TP", 1, 1]].forEach(([v, lab, cx, cy]) => {
      const good = lab === "TP" || lab === "TN";
      el("rect", { x: 110 + cx * 100, y: 30 + cy * 100, width: 96, height: 96, rx: 6, style: `fill:var(${good ? "--fwd" : "--neg"});fill-opacity:${(0.12 + (0.7 * v) / mx).toFixed(2)}` }, svg);
      el("text", { x: 158 + cx * 100, y: 82 + cy * 100, "text-anchor": "middle", style: "fill:var(--ink);font-size:24px;font-weight:600" }, svg, String(v));
      el("text", { x: 158 + cx * 100, y: 104 + cy * 100, "text-anchor": "middle" }, svg, lab);
    });
    el("text", { x: 208, y: 18, "text-anchor": "middle" }, svg, "predicted: none · disease");
    el("text", { x: 102, y: 82, "text-anchor": "end" }, svg, "true: none");
    el("text", { x: 102, y: 182, "text-anchor": "end" }, svg, "true: disease");
    const pairs = p.map((q, i) => [q, Yte[i]]).sort((a, b) => b[0] - a[0]);
    const P1 = Yte.filter((v) => v).length, N0 = Yte.length - P1;
    let tp = 0, fp = 0, auc = 0, prevF = 0, prevT = 0; const pts = [[0, 0]];
    pairs.forEach(([, yv], i) => { if (yv) tp++; else fp++; if (i === pairs.length - 1 || pairs[i + 1][0] !== pairs[i][0]) { const F = fp / N0, T = tp / P1; auc += ((F - prevF) * (T + prevT)) / 2; prevF = F; prevT = T; pts.push([F, T]); } });
    const rs = $("rocSvg"); clear(rs);
    const FR = frame(rs, 300, 260, { l: 48, r: 10, t: 10, b: 34 }, [0, 1], [0, 1], { xlabel: "false positive rate", ylabel: "true positive rate", xn: 4, yn: 4 });
    el("path", { d: path(pts.map(([a, b]) => [FR.x(a), FR.y(b)])) + `L${FR.x(1)},${FR.y(0)} L${FR.x(0)},${FR.y(0)} Z`, style: "fill:var(--bwd);fill-opacity:.12;stroke:none" }, rs);
    el("line", { x1: FR.x(0), y1: FR.y(0), x2: FR.x(1), y2: FR.y(1), style: "stroke:var(--muted);stroke-dasharray:4 4" }, rs);
    el("path", { d: path(pts.map(([a, b]) => [FR.x(a), FR.y(b)])), style: "fill:none;stroke:var(--bwd);stroke-width:2.5" }, rs);
    el("text", { x: FR.x(0.95), y: FR.y(0.08), "text-anchor": "end", style: "fill:var(--ink);font-size:13px;font-weight:600" }, rs, `AUC = ${auc.toFixed(3)}`);
    const pr1 = TP / Math.max(1, TP + FP), rc1 = TP / Math.max(1, TP + FN), pr0 = TN / Math.max(1, TN + FN), rc0 = TN / Math.max(1, TN + FP), f1 = (a, b) => (a + b ? (2 * a * b) / (a + b) : 0);
    $("repTable").innerHTML = `<tr><th>Class (n)</th><th class="num">Prec.</th><th class="num">Recall</th><th class="num">F1</th></tr><tr><td>disease (${P1})</td><td class="num">${f2(pr1)}</td><td class="num">${f2(rc1)}</td><td class="num">${f2(f1(pr1, rc1))}</td></tr><tr><td>none (${N0})</td><td class="num">${f2(pr0)}</td><td class="num">${f2(rc0)}</td><td class="num">${f2(f1(pr0, rc0))}</td></tr><tr class="sel"><td>accuracy (${Yte.length})</td><td></td><td></td><td class="num">${f2((TP + TN) / Yte.length)}</td></tr>`;
    const l = +$("wdLayer").value;
    histogram($("wdSvg"), [{ values: t.before.layers[l].W.flatMap((r) => Array.from(r)), color: "--c1" }, { values: t.net.layers[l].W.flatMap((r) => Array.from(r)), color: "--bwd" }], { label: `weights of ${layerLabel(l)}` });
    const hid = (n2) => Core.forward(n2, Xte, { train: false }).caches[LAST].Ain;
    scatter($("pcaBefore"), pca(hid(t.before), 2), Yte);
    const after = hid(t.net);
    scatter($("pcaAfter"), pca(after, 2), Yte);
    drawPca3d(pca(after, 3));
  }
  let lastPca = null;
  function drawPca3d(P) {
    if (P) lastPca = P;
    if (!lastPca) return;
    if (!hasPlotly()) return noPlotly("pca3d");
    const c = plotBase(), P3 = lastPca;
    const tr = [1, 0].map((cls) => { const idx = Yte.map((y, i) => (y === cls ? i : -1)).filter((i) => i >= 0); return { type: "scatter3d", mode: "markers", name: cls ? "heart disease" : "no disease", x: idx.map((i) => P3[i][0]), y: idx.map((i) => P3[i][1]), z: idx.map((i) => P3[i][2] || 0), marker: { size: 5, color: cls ? c.neg : c.fwd, opacity: 0.85, line: { color: c.panel, width: 0.5 } } }; });
    Plotly.react("pca3d", tr, { margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: c.panel, font: c.font, legend: { x: 0, y: 1, font: c.font }, scene: { xaxis: axis3("PC 1", c), yaxis: axis3("PC 2", c), zaxis: axis3("PC 3", c), camera: { eye: { x: 1.6, y: 1.3, z: 0.8 } } }, uirevision: "pca3d" }, PCFG);
  }

  // ---------------------------------------------------------------- motion: hero, playback, GD ball, scroll
  function hero() {
    const cv = $("heroCanvas"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    const P = SIZES.map((n, l) => Array.from({ length: n }, (_, k) => { const R = n === 1 ? 0 : 55 + Math.sqrt(n) * 40, t = (2 * Math.PI * k) / n; return [(l - 2.5) * 245, R * Math.cos(t), R * Math.sin(t)]; }));
    const r = Core.rng(11), edges = [];
    for (let l = 0; l < SIZES.length - 1; l++) for (let i = 0; i < SIZES[l]; i++) for (let j = 0; j < SIZES[l + 1]; j++) if (r() < 0.42) edges.push([l, i, j]);
    let col = {};
    const refresh = () => { col = { rule: cssv("--rule"), fwd: cssv("--fwd"), bwd: cssv("--bwd"), ink: cssv("--muted"), panel: cssv("--panel"), soft: cssv("--soft") }; };
    refresh();
    const yaw = -0.42, cyw = Math.cos(yaw), syw = Math.sin(yaw), Dp = 1100;
    const proj = (p, a) => {
      const ca = Math.cos(a), sa = Math.sin(a);
      const y1 = p[1] * ca - p[2] * sa, z1 = p[1] * sa + p[2] * ca;
      const x2 = p[0] * cyw + z1 * syw, z2 = -p[0] * syw + z1 * cyw;
      const s = Dp / (Dp + z2 + 260);
      return [W / 2 + x2 * s * 1.18, H / 2 + y1 * s * 1.18, s];
    };
    const T = 7000;
    function draw(t) {
      const a = t * 0.00022, ph = (t % T) / T, forward = ph < 0.5, lp = forward ? ph * 2 : (ph - 0.5) * 2;
      const Q = P.map((layer) => layer.map((p) => proj(p, a)));
      ctx.fillStyle = col.panel; ctx.fillRect(0, 0, W, H);
      ctx.lineWidth = 1; ctx.strokeStyle = col.rule;
      ctx.beginPath(); edges.forEach(([l, i, j]) => { const A = Q[l][i], B = Q[l + 1][j]; ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); }); ctx.stroke();
      const front = lp * 5, c = forward ? col.fwd : col.bwd;
      ctx.fillStyle = c;
      edges.forEach(([l, i, j], k) => {
        const seg = forward ? l : 4 - l, u = front - seg;
        if (u < 0 || u > 1 || (k * 7) % 5 > 1) return;
        const s = forward ? u : 1 - u, A = Q[l][i], B = Q[l + 1][j];
        ctx.beginPath(); ctx.arc(A[0] + (B[0] - A[0]) * s, A[1] + (B[1] - A[1]) * s, 3.6, 0, Math.PI * 2); ctx.fill();
      });
      const nodes = [];
      Q.forEach((layer, l) => layer.forEach((q) => nodes.push([q, l])));
      nodes.sort((u, v) => u[0][2] - v[0][2]);
      nodes.forEach(([q, l]) => {
        const lit = forward ? Math.max(0, 1 - Math.abs(front - l) * 1.3) : Math.max(0, 1 - Math.abs(front - (5 - l)) * 1.3);
        const rr = (l === 5 ? 15 : 9.5) * q[2] * 1.15;
        ctx.globalAlpha = 1; ctx.fillStyle = col.panel; ctx.beginPath(); ctx.arc(q[0], q[1], rr, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 0.18 + 0.82 * lit; ctx.fillStyle = c; ctx.beginPath(); ctx.arc(q[0], q[1], rr, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 0.55 + 0.45 * q[2]; ctx.strokeStyle = col.ink; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(q[0], q[1], rr, 0, Math.PI * 2); ctx.stroke();
      });
      ctx.globalAlpha = 1; ctx.font = "600 22px JetBrains Mono, monospace"; ctx.textAlign = "left"; ctx.fillStyle = c;
      ctx.fillText(forward ? "→ forward pass" : "← backward pass", 26, 42);
      ctx.font = "18px JetBrains Mono, monospace"; ctx.fillStyle = col.ink;
      ctx.fillText(forward ? "weighted sums and activations, layer by layer" : "gradients from the loss back to every weight", 26, 70);
    }
    if (reduceMotion()) { draw(1500); return () => { refresh(); draw(1500); }; }
    let visible = true;
    if ("IntersectionObserver" in window) new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(cv);
    const loop = (t) => { if (visible) draw(t); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    return refresh;
  }

  function playPass(svgId, dir, btnId) {
    const svg = $(svgId), btn = $(btnId);
    if (svg.dataset.playing === "1" || reduceMotion()) return;
    svg.dataset.playing = "1"; btn.disabled = true;
    const circles = [...svg.querySelectorAll("circle[data-l]")], edges = [...svg.querySelectorAll("line[data-l]")];
    const last = SIZES.length - 1;
    circles.forEach((c) => { c.style.transition = "fill-opacity .35s ease"; const l = +c.dataset.l; if (dir > 0 ? l > 0 : l < last) c.style.fillOpacity = "0.03"; });
    const overlay = el("g", {}, svg);
    let step = 0;
    const done = () => { overlay.remove(); svg.dataset.playing = ""; btn.disabled = false; };
    const run = () => {
      if (!svg.isConnected || step >= last) { done(); return; }
      const edgeLayer = dir > 0 ? step : last - 1 - step, target = dir > 0 ? step + 1 : last - 1 - step;
      const es = edges.filter((e) => +e.dataset.l === edgeLayer).sort((a, b) => b.dataset.a - a.dataset.a).slice(0, 28);
      const dots = es.map(() => el("circle", { r: 3.6, style: `fill:var(${dir > 0 ? "--fwd" : "--bwd"})` }, overlay));
      const t0 = performance.now(), D = 620;
      const anim = (t) => {
        const u = Math.min(1, (t - t0) / D), e2 = u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2;
        es.forEach((e, i) => {
          const s = dir > 0 ? e2 : 1 - e2;
          dots[i].setAttribute("cx", +e.getAttribute("x1") + (+e.getAttribute("x2") - +e.getAttribute("x1")) * s);
          dots[i].setAttribute("cy", +e.getAttribute("y1") + (+e.getAttribute("y2") - +e.getAttribute("y1")) * s);
        });
        if (u < 1) requestAnimationFrame(anim);
        else {
          dots.forEach((d) => d.remove());
          circles.filter((c) => +c.dataset.l === target).forEach((c) => (c.style.fillOpacity = c.dataset.o));
          step++; setTimeout(run, 140);
        }
      };
      requestAnimationFrame(anim);
    };
    setTimeout(run, 250);
  }

  let lossCtx = null, gdTimer = null;
  function rollGD() {
    if (!lossCtx || !hasPlotly()) return;
    const L = lossCtx, btn = $("rollGD");
    if (gdTimer) { cancelAnimationFrame(gdTimer); gdTimer = null; }
    const grad = (wa, wb) => { let ga = 0, gb = 0; L.A4.forEach((a, n) => { let z = L.b; for (let i = 0; i < 4; i++) z += (i === L.ia ? wa : i === L.ib ? wb : L.W[i]) * a[i]; const e = Core.sigmoid(z) - Ytr[n]; ga += e * a[L.ia]; gb += e * a[L.ib]; }); return [ga / L.A4.length, gb / L.A4.length]; };
    // start from a deliberately poor corner of the grid so the roll is visible
    let wa = L.ra[3], wb = L.rb[L.rb.length - 4];
    const g0 = grad(wa, wb), lr = 0.35 / Math.max(1e-6, Math.hypot(g0[0], g0[1]));
    const pts = [[wa, wb]];
    for (let t = 0; t < 70; t++) { const g = grad(wa, wb); wa -= lr * g[0]; wb -= lr * g[1]; wa = Math.max(L.ra[0], Math.min(L.ra[L.ra.length - 1], wa)); wb = Math.max(L.rb[0], Math.min(L.rb[L.rb.length - 1], wb)); pts.push([wa, wb]); if (Math.hypot(g[0], g[1]) * lr < 1e-3) break; }
    const zs = pts.map(([a, b]) => L.lossAt(a, b) + 0.004);
    btn.disabled = true;
    let k = 1, lastT = 0;
    const tick = (t) => {
      if (t - lastT > 60 || reduceMotion()) {
        lastT = t; k = reduceMotion() ? pts.length : k + 1;
        const p = pts.slice(0, k), h = p[p.length - 1];
        Plotly.restyle("loss3d", { x: [p.map((q) => q[0]), [h[0]]], y: [p.map((q) => q[1]), [h[1]]], z: [zs.slice(0, k), [zs[k - 1]]] }, [2, 3]);
        Plotly.restyle("loss2d", { x: [p.map((q) => q[0]), [h[0]]], y: [p.map((q) => q[1]), [h[1]]] }, [2, 3]);
        $("gdLbl").textContent = `step ${k - 1}: loss ${f3(zs[k - 1] - 0.004)}`;
      }
      if (k < pts.length) gdTimer = requestAnimationFrame(tick); else { gdTimer = null; btn.disabled = false; }
    };
    gdTimer = requestAnimationFrame(tick);
  }

  function initScrollFx() {
    const secs = [...document.querySelectorAll("section[id]")];
    const bar = $("progBar"), nav = $("secNav"), topbar = document.querySelector(".topbar");
    const PART1 = new Set(["history", "bio", "perceptron", "mlp", "pipeline"]);
    nav.innerHTML = '<option value="">Overview</option>' + secs.map((s) => {
      const t = s.querySelector("h2"), num = t.querySelector(".num");
      return `<option value="${s.id}">${num.textContent} · ${t.textContent.slice(num.textContent.length)}</option>`;
    }).join("");
    nav.addEventListener("change", () => {
      const s = nav.value ? $(nav.value) : null;
      window.scrollTo({ top: s ? s.getBoundingClientRect().top + window.scrollY - topbar.offsetHeight - 12 : 0, behavior: reduceMotion() ? "auto" : "smooth" });
    });
    let ticking = false;
    const upd = () => {
      ticking = false;
      const h = document.documentElement, max = h.scrollHeight - h.clientHeight;
      bar.style.transform = `scaleX(${max > 0 ? Math.min(1, h.scrollTop / max) : 0})`;
      let cur = null;
      secs.forEach((s) => { if (s.getBoundingClientRect().top < 140) cur = s; });
      if (document.activeElement !== nav) nav.value = cur ? cur.id : "";
      topbar.classList.toggle("p1", !cur || PART1.has(cur.id));
      document.querySelectorAll("#toc a").forEach((a) => a.classList.toggle("on", !!cur && a.getAttribute("href") === "#" + cur.id));
    };
    window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(upd); } }, { passive: true });
    upd();
    if (!("IntersectionObserver" in window) || reduceMotion()) return;
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return;
      io.unobserve(e.target);
      e.target.classList.add("reveal");
      setTimeout(() => e.target.classList.remove("reveal"), 1600);
    }), { threshold: 0.18 });
    document.querySelectorAll("figure, .box").forEach((f) => io.observe(f));
    const auto = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return;
      auto.unobserve(e.target);
      if (e.target.id === "netSvg") playPass("netSvg", 1, "playFwd");
      if (e.target.id === "deltaFlow") playPass("deltaFlow", -1, "playBwd");
      if (e.target.id === "loss3d") rollGD();
    }), { threshold: 0.45 });
    ["netSvg", "deltaFlow", "loss3d"].forEach((id) => auto.observe($(id)));
  }

  function initMode() {
    const set = (plain) => {
      document.body.classList.toggle("plain", plain);
      $("modeFull").setAttribute("aria-pressed", String(!plain));
      $("modePlain").setAttribute("aria-pressed", String(plain));
      try { localStorage.setItem("mlpMode", plain ? "plain" : "full"); } catch (e) {}
    };
    let saved = null;
    try { saved = localStorage.getItem("mlpMode"); } catch (e) {}
    set(saved === "plain");
    $("modeFull").addEventListener("click", () => set(false));
    $("modePlain").addEventListener("click", () => set(true));
    $("ctlBtn").addEventListener("click", () => {
      const on = document.querySelector(".topbar").classList.toggle("showctl");
      $("ctlBtn").setAttribute("aria-expanded", String(on));
    });
    document.querySelectorAll(".box.deriv .box-h").forEach((h) => {
      h.setAttribute("tabindex", "0");
      const tog = () => { if (document.body.classList.contains("plain")) h.parentElement.classList.toggle("open"); };
      h.addEventListener("click", tog);
      h.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tog(); } });
    });
  }

  // ---------------------------------------------------------------- wiring
  function renderPatient() { renderData(); renderNet(); drawNet3d(); renderLayer(); renderBN(); renderDrop(); renderLoss(); renderBackprop(); renderUpdate(); }
  function renderNetDependent() { renderPatient(); drawLossSurface(); }
  let heroRefresh = () => {};
  // ---------------------------------------------------------------- part I: history, gates, XOR, baseline
  // narrow containers (phones) get stacked or vertical layouts instead of a shrunken wide drawing
  const isNarrow = (svg) => (svg.parentElement.clientWidth || 1000) < 560;
  const label = (g, x, y, s, st = "", anc = "start") => el("text", { x, y, "text-anchor": anc, style: st }, g, s);
  const T_INK = "fill:var(--ink);font-family:var(--sans);font-weight:600;";

  const EVENTS = [
    [1943, "u1", "McCulloch–Pitts neuron", "B", "A threshold unit computing logic; weights set by hand"],
    [1949, "d1", "Hebb's learning rule", "B", "Connections strengthen when both cells fire together"],
    [1958, "u2", "Rosenblatt's perceptron", "C", "The first neuron that learns its weights from data"],
    [1960, "d2", "ADALINE (Widrow–Hoff)", "", "Learns by gradient descent on squared error"],
    [1969, "u1", "Minsky & Papert: XOR", "C", "A single-layer perceptron cannot compute XOR"],
    [1970, "d1", "Reverse-mode autodiff", "7", "Linnainmaa: the algorithm behind backpropagation"],
    [1986, "u2", "Backpropagation", "7", "Rumelhart, Hinton & Williams train hidden layers"],
    [1989, "d1", "Universal approximation", "2", "Cybenko; also LeCun's convolutional nets"],
    [1997, "u1", "LSTM", "", "Hochreiter & Schmidhuber: memory for sequences"],
    [2006, "u1", "Deep belief nets", "", "Hinton: layer-wise pre-training revives 'deep' nets"],
    [2010, "d1", "Xavier init · ReLU", "3", "Glorot & Bengio; Nair & Hinton", "end"],
    [2012, "u2", "AlexNet wins ImageNet", "", "Top-5 error 15.3% vs 26.2%: deep learning goes mainstream", "end"],
    [2014, "d2", "Dropout · Adam", "5, 9", "Srivastava et al.; Kingma & Ba", "start"],
    [2015, "u3", "Batch norm · He init", "3, 4", "Ioffe & Szegedy; He et al. (also ResNet)", "start"],
    [2017, "u1", "Transformer", "", "Attention is all you need", "start"],
    [2022, "d1", "ChatGPT", "", "Large language models reach the public"],
  ];
  const WINTERS = [[1974, 1980, "1st AI winter"], [1987, 1993, "2nd AI winter"]];

  function drawTimeline() {
    const svg = $("timeSvg"); clear(svg);
    if (isNarrow(svg)) return drawTimelineVertical(svg);
    svg.setAttribute("viewBox", "0 0 1060 344");
    // piecewise scale: the crowded 2000s get twice the room of earlier decades
    const AX = 210, x = (y) => (y <= 2000 ? lin(1940, 2000, 60, 560)(y) : lin(2000, 2025, 560, 1020)(y));
    WINTERS.forEach(([a, b, t]) => {
      el("rect", { x: x(a), y: 30, width: x(b) - x(a), height: 296, style: "fill:var(--soft)" }, svg);
      label(svg, (x(a) + x(b)) / 2, 338, t, "font-size:10.5px;fill:var(--faint)", "middle");
    });
    el("line", { x1: 50, x2: 1030, y1: AX, y2: AX, style: "stroke:var(--ink);stroke-width:1.2" }, svg);
    for (let y = 1940; y <= 2020; y += 10) el("line", { x1: x(y), x2: x(y), y1: AX - 4, y2: AX + 4, style: "stroke:var(--ink)" }, svg);
    const LANE = { u1: 176, u2: 136, u3: 96, d1: 264, d2: 306 };
    EVENTS.forEach(([yr, lane, title, sec, det, anc = "middle"]) => {
      const X = x(yr), Y = LANE[lane], up = lane[0] === "u", used = !!sec, TX = X + (anc === "end" ? 6 : anc === "start" ? -6 : 0);
      const g = el("g", {}, svg);
      el("title", {}, g, `${yr} · ${title}. ${det}${used ? ` (section ${sec})` : ""}`);
      el("line", { x1: X, x2: X, y1: AX, y2: up ? Y + 8 : Y - 26, style: `stroke:var(${used ? "--hl" : "--faint"});stroke-dasharray:2 3` }, g);
      el("circle", { cx: X, cy: AX, r: used ? 6 : 4.5, style: `fill:var(${used ? "--hl" : "--panel"});stroke:var(${used ? "--hl" : "--ink"});stroke-width:1.4` }, g);
      label(g, TX, up ? Y - 6 : Y - 12, title, T_INK + "font-size:12px", anc);
      label(g, TX, up ? Y + 7 : Y + 1, used ? `${yr} · §${sec}` : String(yr), `fill:var(${used ? "--hl" : "--muted"});font-size:10.5px`, anc);
    });
    for (let y = 1940; y <= 2020; y += 10) label(svg, x(y), AX + 17, y, "font-size:10.5px;fill:var(--faint);paint-order:stroke;stroke:var(--panel);stroke-width:5px", "middle");
    label(svg, 60, 22, "● violet: used on this page (section)    ○ context", "font-size:11px;fill:var(--muted)");
  }

  function drawTimelineVertical(svg) {
    // one row per event, winters as shaded rows; not to scale, but readable at phone width
    const rows = [];
    EVENTS.forEach((e) => {
      WINTERS.forEach((w) => { if (e[0] > w[0] && !rows.some((r) => r.w === w)) rows.push({ w }); });
      rows.push({ e });
    });
    const RH = 58, W = 360, H = rows.length * RH + 20, AXX = 58;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el("line", { x1: AXX, x2: AXX, y1: 16, y2: H - 10, style: "stroke:var(--ink);stroke-width:1.2" }, svg);
    rows.forEach((r, i) => {
      const y = 16 + i * RH + RH / 2;
      if (r.w) {
        el("rect", { x: 0, y: y - RH / 2 + 6, width: W, height: RH - 12, rx: 6, style: "fill:var(--soft)" }, svg);
        label(svg, AXX + 18, y + 5, `${r.w[2]}, ${r.w[0]}–${r.w[1]}`, "font-size:13px;fill:var(--muted);font-family:var(--sans)");
        return;
      }
      const [yr, , title, sec, det] = r.e, used = !!sec, g = el("g", {}, svg);
      el("title", {}, g, `${yr} · ${title}. ${det}`);
      label(g, AXX - 14, y + 5, yr, `font-size:12.5px;fill:var(${used ? "--hl" : "--muted"})`, "end");
      el("circle", { cx: AXX, cy: y, r: used ? 6 : 4.5, style: `fill:var(${used ? "--hl" : "--panel"});stroke:var(${used ? "--hl" : "--ink"});stroke-width:1.4` }, g);
      label(g, AXX + 18, y - 3, title + (used ? `  §${sec}` : ""), T_INK + "font-size:14px");
      label(g, AXX + 18, y + 15, det.length > 44 ? det.slice(0, 42) + "…" : det, "font-size:11.5px;fill:var(--muted);font-family:var(--sans)");
    });
  }

  let gatesGen = 0;
  function drawGates() {
    const svg = $("gatesSvg"); clear(svg);
    const gen = ++gatesGen, narrow = isNarrow(svg);
    const pw = 300, ph = 230;
    svg.setAttribute("viewBox", narrow ? "0 0 360 900" : "0 0 1060 300");
    const P = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const G = [["AND", [0, 0, 0, 1]], ["OR", [0, 1, 1, 1]], ["XOR", [0, 1, 1, 0]]];
    let sweep = null;
    G.forEach(([name, Y], k) => {
      const ox = narrow ? 30 : 25 + k * 355, top = narrow ? 50 + k * 300 : 44, g = el("g", {}, svg);
      const sx = lin(-0.5, 1.5, ox, ox + pw), sy = lin(-0.5, 1.5, top + ph, top);
      el("rect", { x: ox, y: top, width: pw, height: ph, rx: 6, style: "fill:none;stroke:var(--rule)" }, g);
      const clip = el("clipPath", { id: "gclip" + k }, g); el("rect", { x: ox, y: top, width: pw, height: ph }, clip);
      const shade = el("path", { "clip-path": `url(#gclip${k})`, style: "fill:var(--bwd);fill-opacity:.09" }, g);
      const ln = el("line", { "clip-path": `url(#gclip${k})`, style: "stroke:var(--ink);stroke-width:2" }, g);
      label(g, ox, top - 20, name, T_INK + "font-size:16px");
      const st = label(g, ox + 50, top - 20, k < 2 ? "✓ one line separates the classes" : "✗ no line gets all four right", `fill:var(${k < 2 ? "--pos" : "--neg"});font-size:13px`);
      const rings = P.map(([a, b]) => el("circle", { cx: sx(a), cy: sy(b), r: 18, style: "fill:none;stroke:var(--neg);stroke-width:2.2;opacity:0" }, g));
      P.forEach(([a, b], i) => {
        el("circle", { cx: sx(a), cy: sy(b), r: 11, style: `fill:var(${Y[i] ? "--bwd" : "--fwd"});stroke:var(--panel);stroke-width:2` }, g);
        label(g, sx(a), sy(b) + (b ? -20 : 30), `(${a},${b})`, "font-size:12.5px", "middle");
      });
      // line n·x = c with unit normal n = (cos t, sin t); the positive side is shaded and predicts 1
      const setLine = (t, c) => {
        const n = [Math.cos(t), Math.sin(t)], d = [-n[1], n[0]], p0 = [n[0] * c, n[1] * c];
        const A = [p0[0] - 4 * d[0], p0[1] - 4 * d[1]], B = [p0[0] + 4 * d[0], p0[1] + 4 * d[1]];
        ln.setAttribute("x1", sx(A[0])); ln.setAttribute("y1", sy(A[1])); ln.setAttribute("x2", sx(B[0])); ln.setAttribute("y2", sy(B[1]));
        const C = [B[0] + 4 * n[0], B[1] + 4 * n[1]], D = [A[0] + 4 * n[0], A[1] + 4 * n[1]];
        shade.setAttribute("d", path([A, B, C, D].map(([u, v]) => [sx(u), sy(v)])) + "Z");
        let err = 0;
        P.forEach(([a, b], i) => { const wrong = (a * n[0] + b * n[1] - c >= 0 ? 1 : 0) !== Y[i]; rings[i].style.opacity = wrong ? 1 : 0; err += wrong; });
        return err;
      };
      const r2 = Math.SQRT1_2;
      setLine(Math.PI / 4, (k === 0 ? 1.5 : 0.5) * r2);
      if (k === 2) sweep = (tm) => { st.textContent = `✗ this line: ${setLine(Math.PI / 4 + 1.1 * Math.sin(tm * 0.00045), 0.707 + 0.55 * Math.sin(tm * 0.00071))} of 4 wrong`; };
    });
    if (!sweep || reduceMotion()) return;
    let vis = false, raf = null;
    const loop = (t) => { if (!vis || gen !== gatesGen) { raf = null; return; } sweep(t); raf = requestAnimationFrame(loop); };
    if ("IntersectionObserver" in window) new IntersectionObserver((es) => { vis = es[0].isIntersecting; if (vis && !raf && gen === gatesGen) raf = requestAnimationFrame(loop); }).observe(svg);
  }

  function drawXor() {
    const svg = $("xorSvg"); clear(svg);
    const narrow = isNarrow(svg), pw = 300, ph = 240;
    svg.setAttribute("viewBox", narrow ? "0 0 340 680" : "0 0 700 310");
    const panel = (ox, oy, title, xl, yl) => {
      const sx = lin(-0.4, 1.4, ox + 30, ox + pw), sy = lin(-0.4, 1.4, oy + ph, oy);
      el("rect", { x: ox + 30, y: oy, width: pw - 30, height: ph, rx: 6, style: "fill:none;stroke:var(--rule)" }, svg);
      label(svg, ox + 30, oy - 14, title, T_INK + "font-size:14px");
      label(svg, ox + 30 + (pw - 30) / 2, oy + ph + 22, xl, "font-size:12.5px", "middle");
      el("text", { x: ox + 14, y: oy + ph / 2, "text-anchor": "middle", transform: `rotate(-90 ${ox + 14} ${oy + ph / 2})`, style: "font-size:12.5px" }, svg, yl);
      [0, 1].forEach((v) => { label(svg, sx(v), oy + ph - 5, v, "font-size:11px;fill:var(--faint)", "middle"); label(svg, ox + 36, sy(v) + 4, v, "font-size:11px;fill:var(--faint)"); });
      return { sx, sy };
    };
    const L = panel(0, 36, "Input space: two hidden lines", "x₁", "x₂");
    const band = [[-0.4, 0.9], [0.9, -0.4], [1.4, -0.4], [1.4, 0.1], [0.1, 1.4], [-0.4, 1.4]];
    el("path", { d: path(band.map(([a, b]) => [L.sx(a), L.sy(b)])) + "Z", style: "fill:var(--bwd);fill-opacity:.1" }, svg);
    [[[0.9, -0.4], [-0.4, 0.9], "h₁ OR", [-0.36, 0.62], "start"], [[1.4, 0.1], [0.1, 1.4], "h₂ AND", [1.36, 0.17], "end"]].forEach(([A, B, t, Q, anc]) => {
      el("line", { x1: L.sx(A[0]), y1: L.sy(A[1]), x2: L.sx(B[0]), y2: L.sy(B[1]), style: "stroke:var(--ink);stroke-width:1.6;stroke-dasharray:5 4" }, svg);
      label(svg, L.sx(Q[0]), L.sy(Q[1]), t, T_INK + "font-size:13px", anc);
    });
    [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]].forEach(([a, b, y]) => el("circle", { cx: L.sx(a), cy: L.sy(b), r: 10, style: `fill:var(${y ? "--bwd" : "--fwd"});stroke:var(--panel);stroke-width:2` }, svg));
    const R = narrow ? panel(0, 400, "Hidden space: the line h₁ − h₂ = 0.5", "h₁ = OR", "h₂ = AND") : panel(380, 36, "Hidden space: the line h₁ − h₂ = 0.5", "h₁ = OR", "h₂ = AND");
    el("path", { d: path([[0.1, -0.4], [1.4, -0.4], [1.4, 0.9]].map(([a, b]) => [R.sx(a), R.sy(b)])) + "Z", style: "fill:var(--bwd);fill-opacity:.1" }, svg);
    el("line", { x1: R.sx(0.1), y1: R.sy(-0.4), x2: R.sx(1.4), y2: R.sy(0.9), style: "stroke:var(--ink);stroke-width:2" }, svg);
    [[0, 0, 0, "from (0,0)"], [1, 0, 1, "from (0,1) and (1,0)"], [1, 1, 0, "from (1,1)"]].forEach(([a, b, y, t]) => {
      el("circle", { cx: R.sx(a), cy: R.sy(b), r: y ? 13 : 10, style: `fill:var(${y ? "--bwd" : "--fwd"});stroke:var(--panel);stroke-width:2` }, svg);
      if (y) label(svg, R.sx(a), R.sy(b) + 4, "×2", "fill:var(--panel);font-size:10.5px;font-weight:700", "middle");
      label(svg, R.sx(a) + (a ? 0 : -6), R.sy(b) + (b || !a ? -19 : 30), t, "font-size:12px", a ? "middle" : "start");
    });
    const arrow = narrow ? "M165,322 L165,356 M159,348 L165,356 L171,348" : "M318,156 L354,156 M346,150 L354,156 L346,162";
    el("path", { d: arrow, style: "fill:none;stroke:var(--faint);stroke-width:1.5" }, svg);
  }

  // ---- the blog's neuron and network diagrams, redrawn natively (Towards AI series, Parts 2-4)
  function neuronFig(id, kind) {
    const svg = $(id); clear(svg);
    const g = el("g", {}, svg), CX = 330, CY = 150, R = 50;
    const ins = [["x₁", 52], ["x₂", 108], ["x₃", 164], ["xₙ", 246]];
    if (kind === "lr") ins.push(["1", 296]);
    ins.forEach(([t, y], i) => {
      const bias = t === "1";
      // the edge meets the neuron's rim
      const ang = Math.atan2(y - CY, 82 - CX), ex = CX + R * Math.cos(ang), ey = CY + R * Math.sin(ang);
      el("line", { x1: 82, y1: y, x2: ex - 6 * Math.cos(ang + Math.PI), y2: ey - 6 * Math.sin(ang + Math.PI), style: `stroke:var(${bias ? "--bwd" : "--muted"});stroke-width:${bias ? 1.8 : 1.4}` }, g);
      el("circle", { cx: 62, cy: y, r: 19, style: `fill:color-mix(in srgb, var(${bias ? "--bwd" : "--fwd"}) 12%, var(--panel));stroke:var(${bias ? "--bwd" : "--fwd"});stroke-width:1.5` }, g);
      label(g, 62, y + 5, t, "fill:var(--ink);font-size:14px;font-family:var(--serif);font-style:italic", "middle");
      const wx = 82 + (ex - 82) * 0.42, wy = y + (ey - y) * 0.42 - 7;
      label(g, wx, wy, bias ? "b" : ["w₁", "w₂", "w₃", "wₙ"][i], `fill:var(${bias ? "--bwd" : "--muted"});font-size:13px;font-family:var(--serif);font-style:italic;paint-order:stroke;stroke:var(--panel);stroke-width:4px`, "middle");
    });
    label(g, 62, 212, "⋮", "fill:var(--muted);font-size:20px", "middle");
    el("circle", { cx: CX, cy: CY, r: R, style: "fill:var(--panel);stroke:var(--ink);stroke-width:1.8" }, g);
    el("line", { x1: CX, y1: CY - R, x2: CX, y2: CY + R, style: "stroke:var(--ink);stroke-width:1.2" }, g);
    label(g, CX - 24, CY + 9, "Σ", "fill:var(--ink);font-size:26px;font-family:var(--serif)", "middle");
    // activation glyph in the right half
    const gx = (u) => CX + 8 + (u + 1) * 15, gy = (v) => CY + 14 - v * 28;
    if (kind === "perc") el("path", { d: `M${gx(-1)},${gy(0)} L${gx(0)},${gy(0)} L${gx(0)},${gy(1)} L${gx(1)},${gy(1)}`, style: "fill:none;stroke:var(--hl);stroke-width:2.2" }, g);
    else if (kind === "lr") el("path", { d: path(Array.from({ length: 31 }, (_, k) => { const u = -1 + k / 15; return [gx(u), gy(1 / (1 + Math.exp(-6 * u)))]; })), style: "fill:none;stroke:var(--hl);stroke-width:2.2" }, g);
    else label(g, CX + 24, CY + 9, "f", "fill:var(--hl);font-size:24px;font-family:var(--serif);font-style:italic", "middle");
    if (kind !== "lr") {
      el("line", { x1: CX - 34, y1: CY - R - 30, x2: CX - 18, y2: CY - R + 2, style: "stroke:var(--bwd);stroke-width:1.6" }, g);
      label(g, CX - 40, CY - R - 36, "+ b", "fill:var(--bwd);font-size:14px;font-family:var(--serif);font-style:italic", "middle");
    }
    el("path", { d: `M${CX + R},${CY} L${CX + R + 92},${CY} M${CX + R + 84},${CY - 6} L${CX + R + 92},${CY} L${CX + R + 84},${CY + 6}`, style: "fill:none;stroke:var(--ink);stroke-width:1.6" }, g);
    label(g, CX + R + 102, CY + 6, "ŷ", "fill:var(--ink);font-size:20px;font-family:var(--serif);font-style:italic");
    const out = { ann: "a = f(z)", perc: "ŷ ∈ {0, 1}", lr: "ŷ ∈ (0, 1)" }[kind];
    label(g, CX + R + 46, CY - 14, out, "fill:var(--muted);font-size:12.5px", "middle");
    label(g, CX - 8, CY + R + 26, "sum", "fill:var(--muted);font-size:12px;font-family:var(--sans)", "end");
    label(g, CX + 8, CY + R + 26, { ann: "activation", perc: "step", lr: "sigmoid" }[kind], "fill:var(--hl);font-size:12px;font-family:var(--sans)", "start");
    label(g, 290, 300, { ann: "z = w₁x₁ + w₂x₂ + … + wₙxₙ + b", perc: "", lr: "" }[kind], "fill:var(--muted);font-size:12.5px", "middle");
  }

  function deepFig() {
    const svg = $("deepSvg"); clear(svg);
    const narrow = isNarrow(svg);
    svg.setAttribute("viewBox", narrow ? "0 0 420 620" : "0 0 1000 310");
    const net = (ox, oy, cols, title) => {
      const g = el("g", {}, svg), w = 400, xs = cols.map((_, i) => ox + 30 + (i * (w - 60)) / (cols.length - 1));
      label(g, ox + w / 2, oy, title, T_INK + "font-size:14px", "middle");
      const pos = cols.map((c, i) => (c.n ? Array.from({ length: c.n }, (_, k) => [xs[i], oy + 130 + (k - (c.n - 1) / 2) * 52]) : null));
      for (let i = 0; i < cols.length - 1; i++) {
        const A = pos[i], B = pos[i + 1];
        if (!A || !B) { // ellipsis between two hidden layers
          const a = A || pos[i - 1], b = B || pos[i + 2];
          (A || []).forEach((p) => el("line", { x1: p[0] + 13, y1: p[1], x2: xs[i + 1] - 10, y2: p[1], style: "stroke:var(--faint);stroke-dasharray:3 4" }, g));
          (B || []).forEach((p) => el("line", { x1: xs[i] + 10, y1: p[1], x2: p[0] - 13, y2: p[1], style: "stroke:var(--faint);stroke-dasharray:3 4" }, g));
          continue;
        }
        A.forEach((p) => B.forEach((q) => el("line", { x1: p[0], y1: p[1], x2: q[0], y2: q[1], style: "stroke:var(--muted);stroke-opacity:.45" }, g)));
      }
      pos.forEach((P, i) => {
        if (!P) { label(g, xs[i], oy + 136, "⋯", "fill:var(--muted);font-size:22px", "middle"); }
        else P.forEach((p, k) => {
          const c = cols[i];
          el("circle", { cx: p[0], cy: p[1], r: c.out ? 15 : 13, style: `fill:color-mix(in srgb, var(${c.in ? "--fwd" : c.out ? "--hl" : "--muted"}) 14%, var(--panel));stroke:var(${c.in ? "--fwd" : c.out ? "--hl" : "--ink"});stroke-width:1.5` }, g);
          if (c.in) label(g, p[0], p[1] + 5, `x${"₁₂₃"[k]}`, "fill:var(--ink);font-size:12px;font-family:var(--serif);font-style:italic", "middle");
          if (c.out) label(g, p[0] + 26, p[1] + 6, "ŷ", "fill:var(--ink);font-size:16px;font-family:var(--serif);font-style:italic");
        });
        label(g, xs[i], oy + 236, cols[i].t, "fill:var(--muted);font-size:12px;font-family:var(--sans)", "middle");
      });
    };
    net(0, 24, [{ n: 3, in: 1, t: "input" }, { n: 3, t: "hidden 1" }, { n: 2, t: "hidden 2" }, { n: 1, out: 1, t: "output" }], "Two hidden layers");
    if (!narrow) el("line", { x1: 500, x2: 500, y1: 20, y2: 270, style: "stroke:var(--rule)" }, svg);
    net(narrow ? 0 : 560, narrow ? 300 : 24, [{ n: 3, in: 1, t: "input" }, { n: 3, t: "hidden 1" }, { t: "" }, { n: 3, t: "hidden N" }, { n: 1, out: 1, t: "output" }], "N hidden layers");
    label(svg, narrow ? 210 : 780, narrow ? 600 : 296, "the deeper the stack, the further the gradient must travel (§7)", "fill:var(--bwd);font-size:12px;font-family:var(--sans)", "middle");
  }

  function drawBlogFigs() { neuronFig("annSvg", "ann"); neuronFig("percSvg", "perc"); neuronFig("lrSvg", "lr"); deepFig(); }

  // re-layout the Part I figures when crossing the phone breakpoint
  let narrowState = null;
  window.addEventListener("resize", () => {
    clearTimeout(window.__p1t);
    window.__p1t = setTimeout(() => { const n = isNarrow($("timeSvg")); if (n !== narrowState) { narrowState = n; drawTimeline(); drawGates(); drawXor(); deepFig(); } }, 150);
  });
  function aucOf(p, y) {
    const pr = p.map((v, i) => [v, y[i]]).sort((a, b) => a[0] - b[0]);
    let rank = 0, sumPos = 0, nPos = 0;
    pr.forEach(([, yy], i) => { rank = i + 1; if (yy) { sumPos += rank; nPos++; } });
    const nNeg = pr.length - nPos;
    return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
  }
  function lrBaseline() {
    const lr = Core.create([13, 1], { seed: 7, init: "xavier" }), opt = Core.makeOptimizer("adam", 0.01), r = Core.rng(1);
    for (let e = 0; e < 100; e++) Core.trainEpoch(lr, opt, Xtr, Ytr, 32, r);
    const p = Core.forward(lr, Xte, { train: false }).out;
    const acc = p.filter((v, i) => (v >= 0.5 ? 1 : 0) === Yte[i]).length / Yte.length;
    $("lrKpi").innerHTML = `<strong>${(acc * 100).toFixed(1)}% accuracy</strong> and <strong>AUC ${aucOf(p, Yte).toFixed(3)}</strong>`;
  }

  function rethemed() { heroRefresh(); drawOptBase(); drawNet3d(); drawLossSurface(); drawPca3d(); }
  function boot() {
    buildNets();
    initControls();
    heroRefresh = hero();
    kpis("heroKpis", [["303", "patients"], ["13", "inputs"], ["5", "weight layers"], [String(NPARAM), "parameters"]]);
    const nPos = rows.filter((q) => q[13] === 1).length;
    kpis("dataKpis", [["303", "patients"], [`${nPos} · ${303 - nPos}`, "disease · no disease"], ["242 · 61", "train · test"], ["13", "inputs per patient"]]);
    initMode();
    drawTimeline(); drawGates(); drawXor(); drawBlogFigs();
    renderSpread(); renderParams();
    renderNetDependent(); renderGate(); initOpt();
    initScrollFx();
    setTimeout(() => { lrBaseline(); renderProp(); renderGradByAct(); renderBatchCurve(); runTraining(); }, 60);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rethemed);
    new MutationObserver(rethemed).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();

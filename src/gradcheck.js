// Numerical gradient check for core.js, with and without batch norm and dropout.
const C = require("./core.js");
function check(label, opts) {
  const r = C.rng(3);
  const net = C.create([4, 5, 3, 1], Object.assign({ seed: 11 }, opts));
  const X = Array.from({ length: 6 }, () => Float64Array.from({ length: 4 }, () => C.gauss(r)));
  const Y = [1, 0, 1, 1, 0, 0];
  const fw = C.forward(net, X, { train: true, r: C.rng(5), updateRunning: false });
  const masks = fw.caches.map((c) => c.mask || null);
  const { grads } = C.backward(net, fw, Y);
  const lossAt = () => C.bce(C.forward(net, X, { train: true, masks, updateRunning: false, r: C.rng(5) }).out, Y);
  let worst = 0, maxAbs = 0, maxG = 0;
  const probe = (arr, i, ga) => {
    const h = 1e-6, o = arr[i];
    arr[i] = o + h; const lp = lossAt(); arr[i] = o - h; const lm = lossAt(); arr[i] = o;
    const num = (lp - lm) / (2 * h), an = ga[i];
    const rel = Math.abs(num - an) / Math.max(1e-8, Math.abs(num) + Math.abs(an));
    worst = Math.max(worst, Math.abs(num - an) < 1e-9 ? 0 : rel); maxAbs = Math.max(maxAbs, Math.abs(num - an)); maxG = Math.max(maxG, Math.abs(an));
  };
  net.layers.forEach((L, l) => {
    L.W.forEach((row, j) => row.forEach((_, i) => probe(row, i, grads[l].dW[j])));
    L.b.forEach((_, j) => probe(L.b, j, grads[l].db));
    if (opts.bn && l < net.layers.length - 1) {
      L.gamma.forEach((_, j) => probe(L.gamma, j, grads[l].dgamma));
      L.beta.forEach((_, j) => probe(L.beta, j, grads[l].dbeta));
    }
  });
  console.log(label.padEnd(28), "max rel", worst.toExponential(2), "max |num-an|", maxAbs.toExponential(2), "max |grad|", maxG.toFixed(3), worst < 1e-5 ? "OK" : "FAIL");
}
check("relu", { act: "relu" });
check("tanh", { act: "tanh" });
check("sigmoid", { act: "sigmoid" });
check("relu + bn", { act: "relu", bn: true });
check("tanh + bn + dropout 0.3", { act: "tanh", bn: true, drop: 0.3 });
check("leaky + dropout 0.5", { act: "leaky", drop: 0.5 });

// quick training sanity on the real data
const fs = require("fs");
const rows = fs.readFileSync(process.argv[2], "utf8").trim().split("\n").slice(1).map((l) => l.split(",").map(Number));
const r = C.rng(42); const idx = rows.map((_, i) => i);
for (let i = idx.length - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)); [idx[i], idx[k]] = [idx[k], idx[i]]; }
const tr = idx.slice(0, 242), te = idx.slice(242);
const mu = Array(13).fill(0), sd = Array(13).fill(0);
tr.forEach((i) => rows[i].slice(0, 13).forEach((v, j) => (mu[j] += v / tr.length)));
tr.forEach((i) => rows[i].slice(0, 13).forEach((v, j) => (sd[j] += (v - mu[j]) ** 2 / tr.length)));
const std = sd.map(Math.sqrt);
const prep = (ids) => [ids.map((i) => Float64Array.from(rows[i].slice(0, 13), (v, j) => (v - mu[j]) / std[j])), ids.map((i) => rows[i][13])];
const [Xtr, Ytr] = prep(tr), [Xte, Yte] = prep(te);
for (const cfg of [{ bn: false, drop: 0 }, { bn: true, drop: 0 }, { bn: false, drop: 0.3 }, { bn: true, drop: 0.3 }]) {
  const net = C.create([13, 16, 12, 8, 4, 1], Object.assign({ seed: 7 }, cfg));
  const opt = C.makeOptimizer("adam", 0.01); const rr = C.rng(1);
  let line = [];
  for (let e = 1; e <= 30; e++) { const tl = C.trainEpoch(net, opt, Xtr, Ytr, 32, rr); if (e % 10 === 0) { const ev = C.evaluate(net, Xte, Yte); const et = C.evaluate(net, Xtr, Ytr); line.push(`e${e} train ${tl.toFixed(3)} trAcc ${et.acc.toFixed(2)} teLoss ${ev.loss.toFixed(3)} teAcc ${ev.acc.toFixed(2)}`); } }
  console.log(JSON.stringify(cfg), "\n  " + line.join("\n  "));
}
console.log("classes", rows.filter((r) => r[13] === 1).length, rows.length);

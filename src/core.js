// MLP engine: dense layers, optional batch norm and dropout, sigmoid + BCE output, five optimizers.
// Plain JS with no dependencies; runs in the browser and in Node (for gradient checks).
const Core = (() => {
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0;
    while (!u) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const ACT = {
    relu: { name: "ReLU", f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0) },
    leaky: { name: "Leaky ReLU", f: (z) => (z > 0 ? z : 0.1 * z), d: (z) => (z > 0 ? 1 : 0.1) },
    tanh: { name: "tanh", f: (z) => Math.tanh(z), d: (z) => 1 - Math.tanh(z) ** 2 },
    sigmoid: { name: "sigmoid", f: sigmoid, d: (z) => sigmoid(z) * (1 - sigmoid(z)) },
  };
  const INIT = {
    he: { name: "He (Kaiming)", std: (fi) => Math.sqrt(2 / fi), formula: "√(2 / fan_in)" },
    xavier: { name: "Xavier (Glorot)", std: (fi, fo) => Math.sqrt(2 / (fi + fo)), formula: "√(2 / (fan_in + fan_out))" },
    small: { name: "Too small: N(0, 0.01²)", std: () => 0.01, formula: "0.01" },
    zeros: { name: "All zeros", std: () => 0, formula: "0" },
  };
  const EPS = 1e-5;

  function create(sizes, opts = {}) {
    const o = Object.assign({ init: "he", seed: 7, act: "relu", bn: false, drop: 0 }, opts);
    const r = rng(o.seed);
    const layers = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fi = sizes[l], fo = sizes[l + 1];
      const std = INIT[o.init].std(fi, fo);
      const W = [];
      for (let j = 0; j < fo; j++) {
        const row = new Float64Array(fi);
        for (let i = 0; i < fi; i++) row[i] = gauss(r) * std;
        W.push(row);
      }
      layers.push({
        W, b: new Float64Array(fo),
        gamma: new Float64Array(fo).fill(1), beta: new Float64Array(fo),
        rm: new Float64Array(fo), rv: new Float64Array(fo).fill(1),
        fanIn: fi, fanOut: fo, std,
      });
    }
    return { sizes, layers, act: o.act, bn: o.bn, drop: o.drop, init: o.init, seed: o.seed };
  }

  function clone(net) {
    return {
      ...net,
      layers: net.layers.map((L) => ({
        ...L, W: L.W.map((r) => Float64Array.from(r)), b: Float64Array.from(L.b),
        gamma: Float64Array.from(L.gamma), beta: Float64Array.from(L.beta),
        rm: Float64Array.from(L.rm), rv: Float64Array.from(L.rv),
      })),
    };
  }

  // X: array of Float64Array rows. opts: train (bool), r (rng for dropout), masks (reuse), updateRunning (bool)
  function forward(net, X, opts = {}) {
    const train = !!opts.train;
    const nL = net.layers.length, act = ACT[net.act];
    const caches = [];
    let A = X;
    for (let l = 0; l < nL; l++) {
      const L = net.layers[l], last = l === nL - 1, fo = L.fanOut, N = A.length;
      const Z = A.map((row) => {
        const out = new Float64Array(fo);
        for (let j = 0; j < fo; j++) {
          let s = L.b[j];
          const w = L.W[j];
          for (let i = 0; i < row.length; i++) s += w[i] * row[i];
          out[j] = s;
        }
        return out;
      });
      const c = { Ain: A, Z };
      let Y = Z;
      if (!last && net.bn) {
        let mean, vr;
        if (train) {
          mean = new Float64Array(fo); vr = new Float64Array(fo);
          for (const z of Z) for (let j = 0; j < fo; j++) mean[j] += z[j] / N;
          for (const z of Z) for (let j = 0; j < fo; j++) vr[j] += (z[j] - mean[j]) ** 2 / N;
          if (opts.updateRunning !== false) {
            for (let j = 0; j < fo; j++) {
              L.rm[j] = 0.9 * L.rm[j] + 0.1 * mean[j];
              L.rv[j] = 0.9 * L.rv[j] + 0.1 * (N > 1 ? (vr[j] * N) / (N - 1) : vr[j]);
            }
          }
        } else { mean = L.rm; vr = L.rv; }
        const inv = new Float64Array(fo);
        for (let j = 0; j < fo; j++) inv[j] = 1 / Math.sqrt(vr[j] + EPS);
        const Zhat = Z.map((z) => z.map((v, j) => (v - mean[j]) * inv[j]));
        Y = Zhat.map((zh) => zh.map((v, j) => L.gamma[j] * v + L.beta[j]));
        Object.assign(c, { mean, vr, inv, Zhat });
      }
      c.Y = Y;
      if (last) {
        A = Y.map((y) => y.map(sigmoid));
      } else {
        const H = Y.map((y) => y.map(act.f));
        c.H = H;
        if (train && net.drop > 0) {
          const keep = 1 - net.drop;
          const masks = opts.masks && opts.masks[l] ? opts.masks[l]
            : H.map((h) => h.map(() => (opts.r() < keep ? 1 / keep : 0)));
          c.mask = masks;
          A = H.map((h, n) => h.map((v, j) => v * masks[n][j]));
        } else A = H;
      }
      c.A = A;
      caches.push(c);
    }
    return { caches, out: A.map((a) => a[0]) };
  }

  function bce(yhat, y) {
    let s = 0;
    for (let n = 0; n < y.length; n++) {
      const p = Math.min(Math.max(yhat[n], 1e-7), 1 - 1e-7);
      s += -(y[n] * Math.log(p) + (1 - y[n]) * Math.log(1 - p));
    }
    return s / y.length;
  }

  function backward(net, fw, y) {
    const nL = net.layers.length, act = ACT[net.act], N = y.length;
    const grads = new Array(nL);
    const deltas = new Array(nL);
    let dA = null;
    for (let l = nL - 1; l >= 0; l--) {
      const L = net.layers[l], c = fw.caches[l], fo = L.fanOut, fi = L.fanIn;
      let dZ;
      const g = { dgamma: null, dbeta: null };
      if (l === nL - 1) {
        dZ = fw.out.map((p, n) => Float64Array.of((p - y[n]) / N));
      } else {
        const dH = c.mask ? dA.map((d, n) => d.map((v, j) => v * c.mask[n][j])) : dA;
        const dY = dH.map((d, n) => d.map((v, j) => v * act.d(c.Y[n][j])));
        if (net.bn) {
          g.dgamma = new Float64Array(fo); g.dbeta = new Float64Array(fo);
          const sdz = new Float64Array(fo), sdzz = new Float64Array(fo);
          const dZhat = dY.map((d) => d.map((v, j) => v * L.gamma[j]));
          for (let n = 0; n < N; n++) for (let j = 0; j < fo; j++) {
            g.dgamma[j] += dY[n][j] * c.Zhat[n][j];
            g.dbeta[j] += dY[n][j];
            sdz[j] += dZhat[n][j];
            sdzz[j] += dZhat[n][j] * c.Zhat[n][j];
          }
          dZ = dZhat.map((d, n) => d.map((v, j) => (c.inv[j] / N) * (N * v - sdz[j] - c.Zhat[n][j] * sdzz[j])));
        } else dZ = dY;
      }
      const dW = L.W.map(() => new Float64Array(fi)), db = new Float64Array(fo);
      for (let n = 0; n < N; n++) {
        const a = c.Ain[n], d = dZ[n];
        for (let j = 0; j < fo; j++) {
          db[j] += d[j];
          const r = dW[j];
          for (let i = 0; i < fi; i++) r[i] += d[j] * a[i];
        }
      }
      dA = c.Ain.map((_, n) => {
        const out = new Float64Array(fi);
        for (let j = 0; j < fo; j++) {
          const w = L.W[j], d = dZ[n][j];
          for (let i = 0; i < fi; i++) out[i] += d * w[i];
        }
        return out;
      });
      g.dW = dW; g.db = db;
      grads[l] = g;
      deltas[l] = dZ;
    }
    return { grads, deltas };
  }

  // Optimizers work on a flat list of [param, grad] Float64Array pairs.
  function paramList(net, grads) {
    const P = [];
    net.layers.forEach((L, l) => {
      const g = grads ? grads[l] : null;
      L.W.forEach((row, j) => P.push([row, g ? g.dW[j] : null]));
      P.push([L.b, g ? g.db : null]);
      if (net.bn && l < net.layers.length - 1) {
        P.push([L.gamma, g ? g.dgamma : null]);
        P.push([L.beta, g ? g.dbeta : null]);
      }
    });
    return P;
  }

  const OPT = {
    sgd: { name: "SGD" }, momentum: { name: "SGD + momentum" }, nesterov: { name: "Nesterov" },
    rmsprop: { name: "RMSprop" }, adam: { name: "Adam" },
  };
  function makeOptimizer(type, lr, beta = 0.9) {
    const st = { t: 0, m: [], v: [] };
    function stepArrays(pairs) {
      st.t++;
      pairs.forEach(([p, g], k) => {
        if (!g) return;
        if (!st.m[k]) { st.m[k] = new Float64Array(p.length); st.v[k] = new Float64Array(p.length); }
        const m = st.m[k], v = st.v[k];
        for (let i = 0; i < p.length; i++) {
          const gi = g[i];
          if (type === "sgd") p[i] -= lr * gi;
          else if (type === "momentum") { m[i] = beta * m[i] + gi; p[i] -= lr * m[i]; }
          else if (type === "nesterov") { m[i] = beta * m[i] + gi; p[i] -= lr * (gi + beta * m[i]); }
          else if (type === "rmsprop") { v[i] = 0.99 * v[i] + 0.01 * gi * gi; p[i] -= (lr * gi) / (Math.sqrt(v[i]) + 1e-8); }
          else if (type === "adam") {
            m[i] = 0.9 * m[i] + 0.1 * gi; v[i] = 0.999 * v[i] + 0.001 * gi * gi;
            const mh = m[i] / (1 - 0.9 ** st.t), vh = v[i] / (1 - 0.999 ** st.t);
            p[i] -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
          }
        }
      });
    }
    return { type, lr, beta, stepArrays, step: (net, grads) => stepArrays(paramList(net, grads)) };
  }

  function evaluate(net, X, Y) {
    const fw = forward(net, X, { train: false });
    let correct = 0;
    fw.out.forEach((p, n) => { if ((p >= 0.5 ? 1 : 0) === Y[n]) correct++; });
    return { loss: bce(fw.out, Y), acc: correct / Y.length };
  }

  function trainEpoch(net, opt, X, Y, batch, r) {
    const idx = X.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)); [idx[i], idx[k]] = [idx[k], idx[i]]; }
    let tot = 0;
    for (let s = 0; s < idx.length; s += batch) {
      const b = idx.slice(s, s + batch);
      if (net.bn && b.length < 2) continue;
      const xb = b.map((i) => X[i]), yb = b.map((i) => Y[i]);
      const fw = forward(net, xb, { train: true, r });
      tot += bce(fw.out, yb) * b.length;
      const { grads } = backward(net, fw, yb);
      opt.step(net, grads);
    }
    return tot / idx.length;
  }

  return { rng, gauss, sigmoid, ACT, INIT, OPT, EPS, create, clone, forward, backward, bce, paramList, makeOptimizer, evaluate, trainEpoch };
})();
if (typeof module !== "undefined") module.exports = Core;

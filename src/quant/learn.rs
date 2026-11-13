//! Learned strategies, trained in Rust on the lookback: gradient-boosted trees, a small LSTM and
//! tabular Q-learning. Each candle's score comes from a model that never saw that candle's future
//! (4 time-blocked folds, purged by the horizon); the newest candle uses a model trained on all of it.
use super::*;

const F: usize = 10;
const START: usize = 60;

/// Features at candle i (all from candles <= i): 1/3/5/10/20-candle log returns, RSI14, ATR%,
/// distance from SMA20 and SMA50, position in the 20-candle range.
fn features(b: &Bars) -> Vec<[f64; F]> {
    let (r14, a14, m20, m50) = (rsi(&b.c, 14), atr(b, 14), sma(&b.c, 20), sma(&b.c, 50));
    (0..b.n())
        .map(|i| {
            if i < START {
                return [NAN; F];
            }
            let lr = |k: usize| (b.c[i] / b.c[i - k]).ln();
            let (lo, hi) = (b.l[i - 19..=i].iter().copied().fold(f64::MAX, f64::min), b.h[i - 19..=i].iter().copied().fold(f64::MIN, f64::max));
            [lr(1), lr(3), lr(5), lr(10), lr(20), r14[i] / 100.0 - 0.5, a14[i] / b.c[i], b.c[i] / m20[i] - 1.0, b.c[i] / m50[i] - 1.0, if hi > lo { (b.c[i] - lo) / (hi - lo) - 0.5 } else { 0.0 }]
        })
        .collect()
}

/// Out-of-fold scores (NaN where no model applies). `train` gets labelled candle indices.
fn oof<M>(n: usize, h: usize, train: impl Fn(&[usize]) -> Option<M>, score: impl Fn(&M, usize) -> f64) -> Result<Vec<f64>, String> {
    const K: usize = 4;
    let labelled: Vec<usize> = (START..n.saturating_sub(1 + h)).collect();
    if labelled.len() < 200 {
        return Err(format!("needs {} more candles to learn from (has {} labelled, needs 200)", 200 - labelled.len(), labelled.len()));
    }
    let mut s = vec![NAN; n];
    let span = n - 1 - START;
    for k in 0..K {
        let (lo, hi) = (START + span * k / K, START + span * (k + 1) / K);
        // purge: no training label may reach into the block; embargo h after it
        let idx: Vec<usize> = labelled.iter().copied().filter(|&i| i + h < lo || i >= hi + h).collect();
        if let Some(m) = train(&idx) {
            for (i, v) in s.iter_mut().enumerate().take(hi).skip(lo) {
                *v = score(&m, i);
            }
        }
    }
    let m = train(&labelled).ok_or("training failed")?;
    s[n - 1] = score(&m, n - 1);
    Ok(s)
}

/// Fires where the score is in the top 20% of its out-of-fold history.
fn top_fifth(s: &[f64]) -> Vec<bool> {
    let mut v: Vec<f64> = s[..s.len() - 1].iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return vec![false; s.len()];
    }
    v.sort_by(f64::total_cmp);
    let thr = v[v.len() * 4 / 5];
    s.iter().map(|x| *x > thr).collect()
}

fn up(b: &Bars, i: usize, h: usize) -> f64 {
    (b.c[i + h] > b.c[i]) as u8 as f64
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------- gradient-boosted trees

enum Node {
    Leaf(f64),
    Split(usize, f64, Box<Node>, Box<Node>),
}

impl Node {
    fn eval(&self, x: &[f64; F]) -> f64 {
        match self {
            Node::Leaf(v) => *v,
            Node::Split(f, t, l, r) => if x[*f] <= *t { l.eval(x) } else { r.eval(x) },
        }
    }
}

const LAMBDA: f64 = 1.0;
const MIN_LEAF: usize = 20;

/// One XGBoost-style tree: Newton leaves, gain = G_L^2/(H_L+l) + G_R^2/(H_R+l) - G^2/(H+l).
fn grow(x: &[[f64; F]], g: &[f64], hs: &[f64], idx: &[usize], cuts: &[Vec<f64>], depth: usize) -> Node {
    let (gs, hsum) = idx.iter().fold((0.0, 0.0), |a, &i| (a.0 + g[i], a.1 + hs[i]));
    let leaf = Node::Leaf(gs / (hsum + LAMBDA));
    if depth == 0 || idx.len() < 2 * MIN_LEAF {
        return leaf;
    }
    let base = gs * gs / (hsum + LAMBDA);
    let mut best = (1e-9, 0, 0.0);
    for f in 0..F {
        for &t in &cuts[f] {
            let (mut gl, mut hl, mut nl) = (0.0, 0.0, 0);
            for &i in idx {
                if x[i][f] <= t {
                    gl += g[i];
                    hl += hs[i];
                    nl += 1;
                }
            }
            if nl < MIN_LEAF || idx.len() - nl < MIN_LEAF {
                continue;
            }
            let (gr, hr) = (gs - gl, hsum - hl);
            let gain = gl * gl / (hl + LAMBDA) + gr * gr / (hr + LAMBDA) - base;
            if gain > best.0 {
                best = (gain, f, t);
            }
        }
    }
    if best.0 <= 1e-9 {
        return leaf;
    }
    let (f, t) = (best.1, best.2);
    let (l, r): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| x[i][f] <= t);
    Node::Split(f, t, Box::new(grow(x, g, hs, &l, cuts, depth - 1)), Box::new(grow(x, g, hs, &r, cuts, depth - 1)))
}

struct Gbt {
    base: f64,
    trees: Vec<Node>,
}

const LR: f64 = 0.1;

impl Gbt {
    fn raw(&self, x: &[f64; F]) -> f64 {
        self.base + LR * self.trees.iter().map(|t| t.eval(x)).sum::<f64>()
    }
}

fn fit_gbt(x: &[[f64; F]], y: &[f64], idx: &[usize]) -> Gbt {
    // candidate thresholds: 16 quantiles per feature of the training rows
    let cuts: Vec<Vec<f64>> = (0..F)
        .map(|f| {
            let mut v: Vec<f64> = idx.iter().map(|&i| x[i][f]).collect();
            v.sort_by(f64::total_cmp);
            let mut c: Vec<f64> = (1..16).map(|q| v[v.len() * q / 16]).collect();
            c.dedup();
            c
        })
        .collect();
    let p0 = (idx.iter().map(|&i| y[i]).sum::<f64>() / idx.len() as f64).clamp(0.01, 0.99);
    let mut m = Gbt { base: (p0 / (1.0 - p0)).ln(), trees: vec![] };
    let mut raw: Vec<f64> = vec![m.base; x.len()];
    let (mut g, mut hs) = (vec![0.0; x.len()], vec![0.0; x.len()]);
    for _ in 0..60 {
        for &i in idx {
            let p = sigmoid(raw[i]);
            g[i] = y[i] - p;
            hs[i] = p * (1.0 - p);
        }
        let t = grow(x, &g, &hs, idx, &cuts, 3);
        for &i in idx {
            raw[i] += LR * t.eval(&x[i]);
        }
        m.trees.push(t);
    }
    m
}

pub(super) fn trees(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    let x = features(b);
    let n = b.n();
    let y: Vec<f64> = (0..n).map(|i| if i + cx.h < n { up(b, i, cx.h) } else { NAN }).collect();
    let s = oof(n, cx.h, |idx| Some(fit_gbt(&x, &y, idx)), |m, i| sigmoid(m.raw(&x[i])))?;
    Ok(top_fifth(&s))
}

// ---------------------------------------------------------------- LSTM

const SEQ: usize = 16;
const D: usize = 3; // inputs per step: 1-candle return, distance from SMA20, range position (scaled)
const HID: usize = 8;

struct Lstm {
    w: Vec<f64>, // 4HID x (D + HID), row-major
    b: Vec<f64>, // 4HID
    wo: Vec<f64>,
    bo: f64,
}

struct Trace {
    xs: Vec<[f64; D + HID]>,
    gates: Vec<[f64; 4 * HID]>, // i f g o (activated)
    cs: Vec<[f64; HID]>,        // c_t (index 0 = c_0 = 0)
    hs: Vec<[f64; HID]>,
}

impl Lstm {
    fn new(rng: &mut Rng) -> Self {
        let s = 1.0 / ((D + HID) as f64).sqrt();
        let mut b = vec![0.0; 4 * HID];
        b[HID..2 * HID].iter_mut().for_each(|v| *v = 1.0); // forget-gate bias 1
        Lstm { w: (0..4 * HID * (D + HID)).map(|_| s * normal(rng)).collect(), b, wo: (0..HID).map(|_| 0.1 * normal(rng)).collect(), bo: 0.0 }
    }

    fn forward(&self, seq: &[[f64; D]]) -> (f64, Trace) {
        let mut tr = Trace { xs: vec![], gates: vec![], cs: vec![[0.0; HID]], hs: vec![[0.0; HID]] };
        for x in seq {
            let mut xh = [0.0; D + HID];
            xh[..D].copy_from_slice(x);
            xh[D..].copy_from_slice(tr.hs.last().unwrap());
            let mut gt = [0.0; 4 * HID];
            for (r, g) in gt.iter_mut().enumerate() {
                let z = self.b[r] + (0..D + HID).map(|k| self.w[r * (D + HID) + k] * xh[k]).sum::<f64>();
                *g = if (2 * HID..3 * HID).contains(&r) { z.tanh() } else { sigmoid(z) };
            }
            let cp = *tr.cs.last().unwrap();
            let (mut c, mut h) = ([0.0; HID], [0.0; HID]);
            for j in 0..HID {
                c[j] = gt[HID + j] * cp[j] + gt[j] * gt[2 * HID + j];
                h[j] = gt[3 * HID + j] * c[j].tanh();
            }
            tr.xs.push(xh);
            tr.gates.push(gt);
            tr.cs.push(c);
            tr.hs.push(h);
        }
        let hl = tr.hs.last().unwrap();
        (self.bo + (0..HID).map(|j| self.wo[j] * hl[j]).sum::<f64>(), tr)
    }

    /// gradients of the logistic loss for one sequence, added into `gr` (same layout as the params)
    fn backward(&self, tr: &Trace, dlogit: f64, gr: &mut Lstm) {
        let hl = tr.hs.last().unwrap();
        for j in 0..HID {
            gr.wo[j] += dlogit * hl[j];
        }
        gr.bo += dlogit;
        let mut dh: [f64; HID] = std::array::from_fn(|j| dlogit * self.wo[j]);
        let mut dc = [0.0; HID];
        for t in (0..tr.xs.len()).rev() {
            let (gt, c, cp) = (&tr.gates[t], &tr.cs[t + 1], &tr.cs[t]);
            let mut dz = [0.0; 4 * HID];
            for j in 0..HID {
                let (ig, fg, gg, og) = (gt[j], gt[HID + j], gt[2 * HID + j], gt[3 * HID + j]);
                let tc = c[j].tanh();
                dc[j] += dh[j] * og * (1.0 - tc * tc);
                dz[j] = dc[j] * gg * ig * (1.0 - ig);
                dz[HID + j] = dc[j] * cp[j] * fg * (1.0 - fg);
                dz[2 * HID + j] = dc[j] * ig * (1.0 - gg * gg);
                dz[3 * HID + j] = dh[j] * tc * og * (1.0 - og);
                dc[j] *= fg;
            }
            let mut dhp = [0.0; HID];
            for r in 0..4 * HID {
                gr.b[r] += dz[r];
                for k in 0..D + HID {
                    gr.w[r * (D + HID) + k] += dz[r] * tr.xs[t][k];
                }
                for j in 0..HID {
                    dhp[j] += self.w[r * (D + HID) + D + j] * dz[r];
                }
            }
            dh = dhp;
        }
    }

    fn params(&mut self) -> Vec<&mut f64> {
        self.w.iter_mut().chain(self.b.iter_mut()).chain(self.wo.iter_mut()).chain(std::iter::once(&mut self.bo)).collect()
    }

    fn zero() -> Self {
        Lstm { w: vec![0.0; 4 * HID * (D + HID)], b: vec![0.0; 4 * HID], wo: vec![0.0; HID], bo: 0.0 }
    }
}

fn fit_lstm(seqs: &[Vec<[f64; D]>], y: &[f64], idx: &[usize]) -> Lstm {
    let mut rng = Rng::new(11);
    let mut m = Lstm::new(&mut rng);
    let np = m.params().len();
    let (mut m1, mut m2, mut t) = (vec![0.0; np], vec![0.0; np], 0);
    let mut order = idx.to_vec();
    for _epoch in 0..8 {
        for i in (1..order.len()).rev() {
            order.swap(i, pick(&mut rng, i + 1));
        }
        for batch in order.chunks(32) {
            let mut g = Lstm::zero();
            for &i in batch {
                let (logit, tr) = m.forward(&seqs[i]);
                m.backward(&tr, (sigmoid(logit) - y[i]) / batch.len() as f64, &mut g);
            }
            // Adam, lr 0.01
            t += 1;
            let (b1, b2) = (0.9f64, 0.999f64);
            for (k, (p, gk)) in m.params().into_iter().zip(g.params()).enumerate() {
                m1[k] = b1 * m1[k] + (1.0 - b1) * *gk;
                m2[k] = b2 * m2[k] + (1.0 - b2) * *gk * *gk;
                *p -= 0.01 * (m1[k] / (1.0 - b1.powi(t))) / ((m2[k] / (1.0 - b2.powi(t))).sqrt() + 1e-8);
            }
        }
    }
    m
}

pub(super) fn lstm(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    let x = features(b);
    let n = b.n();
    // inputs scaled by their spread over the lookback (scale only, no direction leaks)
    let cols = [0usize, 7, 9];
    let sd: Vec<f64> = cols.iter().map(|&c| mean_var(&x[START..].iter().map(|r| r[c]).collect::<Vec<_>>()).1.sqrt().max(1e-9)).collect();
    let seqs: Vec<Vec<[f64; D]>> = (0..n).map(|i| if i < START + SEQ { vec![] } else { (i + 1 - SEQ..=i).map(|t| std::array::from_fn(|k| x[t][cols[k]] / sd[k])).collect() }).collect();
    let y: Vec<f64> = (0..n).map(|i| if i + cx.h < n { up(b, i, cx.h) } else { NAN }).collect();
    let usable = |idx: &[usize]| idx.iter().copied().filter(|&i| !seqs[i].is_empty()).collect::<Vec<_>>();
    let s = oof(n, cx.h, |idx| Some(fit_lstm(&seqs, &y, &usable(idx))), |m, i| if seqs[i].is_empty() { NAN } else { sigmoid(m.forward(&seqs[i]).0) })?;
    Ok(top_fifth(&s))
}

// ---------------------------------------------------------------- Q-learning

const NS: usize = 24; // market states: RSI bucket (4) x above SMA20 (2) x 5-candle return bucket (3)

pub(super) fn rl(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    let n = b.n();
    let (r14, m20) = (rsi(&b.c, 14), sma(&b.c, 20));
    let r5: Vec<f64> = (0..n).map(|i| if i >= 5 { (b.c[i] / b.c[i - 5]).ln() } else { NAN }).collect();
    let sd5 = mean_var(&r5[START..].to_vec()).1.sqrt();
    let st: Vec<usize> = (0..n)
        .map(|i| {
            if i < START {
                return 0;
            }
            let rb = match r14[i] { x if x < 30.0 => 0, x if x < 50.0 => 1, x if x < 70.0 => 2, _ => 3 };
            let tb = (b.c[i] > m20[i]) as usize;
            let mb = if r5[i] < -sd5 { 0 } else if r5[i] > sd5 { 2 } else { 1 };
            rb * 6 + tb * 3 + mb
        })
        .collect();
    const COST: f64 = 0.0005; // per position change, as a return
    let train = |idx: &[usize]| -> Option<[[[f64; 2]; 2]; NS]> {
        let mut q = [[[0.0f64; 2]; 2]; NS]; // q[state][position][action]
        let mut rng = Rng::new(17);
        // contiguous training stretches (folds cut the index list)
        let mut runs: Vec<&[usize]> = vec![];
        let mut s0 = 0;
        for k in 1..=idx.len() {
            if k == idx.len() || idx[k] != idx[k - 1] + 1 {
                runs.push(&idx[s0..k]);
                s0 = k;
            }
        }
        for _ in 0..30 {
            for run in &runs {
                let mut pos = 0;
                for &t in run.iter().filter(|&&t| t + 1 < n - 1) {
                    let s = st[t];
                    let a = if (rng.unit() as f64) < 0.1 { pick(&mut rng, 2) } else { (q[s][pos][1] > q[s][pos][0]) as usize };
                    let reward = a as f64 * (b.c[t + 1] / b.c[t]).ln() - COST * (a != pos) as u8 as f64;
                    let next = q[st[t + 1]][a][0].max(q[st[t + 1]][a][1]);
                    q[s][pos][a] += 0.1 * (reward + 0.95 * next - q[s][pos][a]);
                    pos = a;
                }
            }
        }
        Some(q)
    };
    // score = advantage of going long from flat
    let s = oof(n, cx.h, train, |q, i| q[st[i]][0][1] - q[st[i]][0][0])?;
    Ok(each(b, START + 1, |i| s[i] > 0.0 && !(s[i - 1] > 0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lstm_gradient_matches_numeric() {
        let mut rng = Rng::new(3);
        let mut m = Lstm::new(&mut rng);
        let seq: Vec<[f64; D]> = (0..5).map(|_| std::array::from_fn(|_| normal(&mut rng))).collect();
        let loss = |m: &Lstm| {
            let p = sigmoid(m.forward(&seq).0);
            -(p.ln()) // label 1
        };
        let (logit, tr) = m.forward(&seq);
        let mut g = Lstm::zero();
        m.backward(&tr, sigmoid(logit) - 1.0, &mut g);
        let analytic: Vec<f64> = g.params().into_iter().map(|x| *x).collect();
        for k in [0, 7, 40, 100, 4 * HID * (D + HID) + 3, 4 * HID * (D + HID) + 4 * HID + 2, analytic.len() - 1] {
            let base = *m.params()[k];
            *m.params()[k] = base + 1e-6;
            let up = loss(&m);
            *m.params()[k] = base - 1e-6;
            let dn = loss(&m);
            *m.params()[k] = base;
            let num = (up - dn) / 2e-6;
            assert!((num - analytic[k]).abs() < 1e-6 + 1e-4 * num.abs(), "param {k}: numeric {num} analytic {}", analytic[k]);
        }
    }

    #[test]
    fn gbt_learns_a_simple_rule() {
        // label = feature 0 > 0: trees must separate it
        let mut rng = Rng::new(1);
        let x: Vec<[f64; F]> = (0..400).map(|_| std::array::from_fn(|_| normal(&mut rng))).collect();
        let y: Vec<f64> = x.iter().map(|r| (r[0] > 0.0) as u8 as f64).collect();
        let idx: Vec<usize> = (0..300).collect();
        let m = fit_gbt(&x, &y, &idx);
        let acc = (300..400).filter(|&i| (sigmoid(m.raw(&x[i])) > 0.5) == (y[i] > 0.5)).count();
        assert!(acc >= 90, "{acc}");
    }
}

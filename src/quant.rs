//! MATH FORECAST strategy library (also listed in the MATHS tab). Every runnable strategy gives the
//! same Prediction the AI models give, so the chart overlay, leaderboard, COMPARE, TEST and paper
//! trading all work unchanged.
//! - PATTERN strategies say WHEN to look (a rule on the candles). Their forecast is what actually
//!   happened on this stock after every earlier time the rule fired in the lookback: those forward
//!   candles, scaled to today's price, are the samples (mean path, P(up), range ...). No lookahead:
//!   an earlier case counts only once all h of its following candles have closed. No signal on the
//!   latest candle = no forecast ("no signal").
//! - MODEL strategies (AR, GARCH, OU, HMM) fit their equation to the lookback and simulate it forward.
use crate::kronos::Rng;
use crate::shared::Candle;

mod cross;
mod learn;
use cross::*;
use learn::*;

/// Earlier cases a pattern needs before its history counts as a forecast.
pub const MIN_CASES: usize = 5;
/// Most recent earlier cases used per forecast.
const MAX_CASES: usize = 300;
/// Simulated paths per model forecast.
const SIM_PATHS: usize = 500;
const NAN: f64 = f64::NAN;

pub type Samples = Vec<Vec<[f64; 6]>>;

pub enum Run {
    /// fires[i] = rule true on candle i, using candles 0..=i only. None = not applicable (e.g. no volume)
    Pattern(fn(&Bars) -> Option<Vec<bool>>),
    /// fit + simulate h candles ahead
    Model(fn(&Bars, usize, &mut Rng) -> Result<Samples, String>),
    /// rule needing other stocks, fundamentals or the horizon (learned rules); Err = why it can't run now
    Rule(fn(&Bars, &Ctx) -> Result<Vec<bool>, String>),
    /// model needing other series (factor ETFs)
    CtxModel(fn(&Bars, &Ctx, &mut Rng) -> Result<(Samples, String), String>),
    /// position-sizing rule for paper trading, not a forecast
    Sizing,
    /// listed for reference; can't run on this app's data (reason)
    Na(&'static str),
}

/// One quarter of company accounts (Yahoo fundamentals timeseries). NaN = not reported.
#[derive(Clone, Copy, Debug)]
pub struct FundQ {
    pub end_t: i64, // fiscal quarter end, unix seconds
    pub eps: f64,
    pub equity: f64,
    pub net_income: f64,
    pub shares: f64,
}

/// What a strategy needs besides the stock's own candles.
#[derive(Default)]
pub struct Extra {
    pub uni: Vec<(String, Vec<Candle>)>, // other symbols, same timeframe
    pub funds: std::collections::HashMap<String, Vec<FundQ>>,
}

/// Factor ETF proxies used by Fama-French: market, small caps, value, growth, quality, momentum.
pub const FACTOR_ETFS: [&str; 6] = ["SPY", "IWM", "IWD", "IWF", "QUAL", "MTUM"];

/// Context a Rule / CtxModel gets: other series aligned to this stock's candle times.
pub struct Ctx<'a> {
    pub h: usize,
    pub sym: &'a str,
    /// other symbols' closes at each of this stock's candles (last close at or before; NaN before their first)
    pub uni: Vec<(&'a str, Vec<f64>)>,
    pub funds: &'a std::collections::HashMap<String, Vec<FundQ>>,
}

impl Ctx<'_> {
    pub fn get(&self, sym: &str) -> Option<&[f64]> {
        self.uni.iter().find(|u| u.0 == sym).map(|u| &u.1[..])
    }
}

/// `other` closes forward-filled onto the times `t`.
fn align(t: &[i64], other: &[Candle]) -> Vec<f64> {
    let mut j = 0;
    let mut last = NAN;
    t.iter()
        .map(|&ti| {
            while j < other.len() && other[j].t <= ti {
                last = other[j].c;
                j += 1;
            }
            last
        })
        .collect()
}

pub struct Strat {
    pub id: &'static str,
    pub name: &'static str,
    pub cat: &'static str,
    pub formula: &'static str,
    pub what: &'static str,
    pub example: &'static str,
    pub used: &'static str,
    pub run: Run,
}

impl Strat {
    /// can make a forecast in MATH FORECAST
    pub fn runnable(&self) -> bool {
        !matches!(self.run, Run::Na(_) | Run::Sizing)
    }
    /// needs the tracked stocks' candles
    pub fn needs_universe(&self) -> bool {
        matches!(self.id, "q:pairs" | "q:kalman_pairs" | "q:johansen" | "q:xs_momentum" | "q:pca" | "q:factor")
    }
    pub fn needs_etfs(&self) -> bool {
        self.id == "q:ff5"
    }
    /// needs company accounts (for the stock itself, plus the universe for factor)
    pub fn needs_funds(&self) -> bool {
        matches!(self.id, "q:factor" | "q:pead")
    }
}

pub fn find(id: &str) -> Option<&'static Strat> {
    STRATS.iter().find(|s| s.id == id)
}

/// Samples for strategy `st` on `cs`, h candles ahead. Ok((samples, signal text)) or Err(why no forecast).
pub fn samples(st: &Strat, sym: &str, cs: &[Candle], h: usize, seed: u64, extra: &Extra) -> Result<(Samples, String), String> {
    let b = Bars::new(cs);
    let n = b.n();
    let ctx = || Ctx { h, sym, uni: extra.uni.iter().filter(|u| u.0 != sym).map(|u| (u.0.as_str(), align(&b.t, &u.1))).collect(), funds: &extra.funds };
    let mut rng = Rng::new(seed);
    let fires = match &st.run {
        Run::Na(why) => return Err(format!("not runnable: {why}")),
        Run::Sizing => return Err("a paper-trading SIZE rule, not a forecast: pick it in a strategy window".into()),
        Run::Model(f) => return f(&b, h, &mut rng).map(|s| (s, format!("MODEL fitted on {n} candles, {SIM_PATHS} simulated paths"))),
        Run::CtxModel(f) => return f(&b, &ctx(), &mut rng),
        Run::Rule(f) => f(&b, &ctx())?,
        Run::Pattern(f) => f(&b).ok_or("needs volume data (this symbol has none)")?,
    };
    if !fires[n - 1] {
        return Err("no signal on the latest candle".into());
    }
    let last = b.c[n - 1];
    // earlier firings whose h following candles have all closed (the newest candle may be live)
    let cases: Vec<usize> = (0..n - 1).filter(|&i| fires[i] && i + h < n - 1).collect();
    if cases.len() < MIN_CASES {
        return Err(format!("SIGNAL NOW, but only {} earlier cases in the lookback (needs {MIN_CASES})", cases.len()));
    }
    let s: Samples = cases
        .iter()
        .rev()
        .take(MAX_CASES)
        .map(|&i| {
            let k = last / b.c[i];
            (1..=h).map(|d| [b.o[i + d] * k, b.h[i + d] * k, b.l[i + d] * k, b.c[i + d] * k, b.v[i + d], 0.0]).collect()
        })
        .collect();
    Ok((s, format!("SIGNAL NOW | forecast = what followed the {} earlier signals", cases.len().min(MAX_CASES))))
}

// ---------------------------------------------------------------- candles + indicators

pub struct Bars {
    pub t: Vec<i64>,
    pub o: Vec<f64>,
    pub h: Vec<f64>,
    pub l: Vec<f64>,
    pub c: Vec<f64>,
    pub v: Vec<f64>,
}

impl Bars {
    pub fn new(cs: &[Candle]) -> Self {
        Bars { t: cs.iter().map(|c| c.t).collect(), o: cs.iter().map(|c| c.o).collect(), h: cs.iter().map(|c| c.h).collect(), l: cs.iter().map(|c| c.l).collect(), c: cs.iter().map(|c| c.c).collect(), v: cs.iter().map(|c| c.v as f64).collect() }
    }
    pub fn n(&self) -> usize {
        self.c.len()
    }
    fn body(&self, i: usize) -> f64 {
        (self.c[i] - self.o[i]).abs()
    }
    fn range(&self, i: usize) -> f64 {
        self.h[i] - self.l[i]
    }
    fn top(&self, i: usize) -> f64 {
        self.o[i].max(self.c[i])
    }
    fn bot(&self, i: usize) -> f64 {
        self.o[i].min(self.c[i])
    }
    fn upper(&self, i: usize) -> f64 {
        self.h[i] - self.top(i)
    }
    fn lower(&self, i: usize) -> f64 {
        self.bot(i) - self.l[i]
    }
    fn bull(&self, i: usize) -> bool {
        self.c[i] > self.o[i]
    }
    fn bear(&self, i: usize) -> bool {
        self.c[i] < self.o[i]
    }
    fn mid(&self, i: usize) -> f64 {
        (self.o[i] + self.c[i]) / 2.0
    }
    /// body / range; NaN for a flat candle so every rule on it is false
    fn ratio(&self, i: usize) -> f64 {
        let r = self.range(i);
        if r > 0.0 { self.body(i) / r } else { NAN }
    }
    /// fell over the 3 candles before i
    fn declined(&self, i: usize) -> bool {
        i >= 4 && self.c[i - 1] < self.c[i - 4]
    }
    fn advanced(&self, i: usize) -> bool {
        i >= 4 && self.c[i - 1] > self.c[i - 4]
    }
    pub(crate) fn has_volume(&self) -> bool {
        self.v.iter().filter(|v| **v > 0.0).count() * 2 > self.n()
    }
    fn heikin(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.n();
        let hc: Vec<f64> = (0..n).map(|i| (self.o[i] + self.h[i] + self.l[i] + self.c[i]) / 4.0).collect();
        let mut ho = vec![0.0; n];
        for i in 0..n {
            ho[i] = if i == 0 { (self.o[0] + self.c[0]) / 2.0 } else { (ho[i - 1] + hc[i - 1]) / 2.0 };
        }
        (ho, hc)
    }
}

fn each(b: &Bars, start: usize, f: impl Fn(usize) -> bool) -> Vec<bool> {
    (0..b.n()).map(|i| i >= start && f(i)).collect()
}

/// Windowed mean (NaN until n values, or if any value in the window is NaN).
fn sma(x: &[f64], n: usize) -> Vec<f64> {
    (0..x.len()).map(|i| if i + 1 < n { NAN } else { x[i + 1 - n..=i].iter().sum::<f64>() / n as f64 }).collect()
}

fn rsum(x: &[f64], n: usize) -> Vec<f64> {
    (0..x.len()).map(|i| if i + 1 < n { NAN } else { x[i + 1 - n..=i].iter().sum::<f64>() }).collect()
}

fn stdev(x: &[f64], n: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            if i + 1 < n {
                return NAN;
            }
            let w = &x[i + 1 - n..=i];
            let m = w.iter().sum::<f64>() / n as f64;
            (w.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt()
        })
        .collect()
}

/// Exponential smoothing with factor a, starting at the first non-NaN value.
fn smooth(x: &[f64], a: f64) -> Vec<f64> {
    let mut out = vec![NAN; x.len()];
    let mut prev = NAN;
    for i in 0..x.len() {
        if x[i].is_nan() {
            continue;
        }
        prev = if prev.is_nan() { x[i] } else { prev + a * (x[i] - prev) };
        out[i] = prev;
    }
    out
}

fn ema(x: &[f64], n: usize) -> Vec<f64> {
    smooth(x, 2.0 / (n as f64 + 1.0))
}

fn wilder(x: &[f64], n: usize) -> Vec<f64> {
    smooth(x, 1.0 / n as f64)
}

fn true_range(b: &Bars) -> Vec<f64> {
    (0..b.n()).map(|i| if i == 0 { b.range(0) } else { b.range(i).max((b.h[i] - b.c[i - 1]).abs()).max((b.l[i] - b.c[i - 1]).abs()) }).collect()
}

fn atr(b: &Bars, n: usize) -> Vec<f64> {
    let mut a = wilder(&true_range(b), n);
    a.iter_mut().take(n).for_each(|v| *v = NAN);
    a
}

fn rsi(x: &[f64], n: usize) -> Vec<f64> {
    let len = x.len();
    let mut g = vec![NAN; len];
    let mut l = vec![NAN; len];
    for i in 1..len {
        let d = x[i] - x[i - 1];
        g[i] = d.max(0.0);
        l[i] = (-d).max(0.0);
    }
    let (ag, al) = (wilder(&g, n), wilder(&l, n));
    (0..len).map(|i| if i < n { NAN } else if al[i] == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + ag[i] / al[i]) }).collect()
}

fn roc(x: &[f64], n: usize) -> Vec<f64> {
    (0..x.len()).map(|i| if i < n || x[i - n] == 0.0 { NAN } else { x[i] / x[i - n] - 1.0 }).collect()
}

/// a crosses above b at i
fn cross_up(a: &[f64], b: &[f64], i: usize) -> bool {
    i > 0 && a[i] > b[i] && a[i - 1] <= b[i - 1]
}

/// series crosses above a constant level at i
fn up_through(x: &[f64], k: f64, i: usize) -> bool {
    i > 0 && x[i] > k && x[i - 1] <= k
}

fn down_through(x: &[f64], k: f64, i: usize) -> bool {
    i > 0 && x[i] < k && x[i - 1] >= k
}

/// highest of the n values before i (not including i)
fn prior_max(x: &[f64], i: usize, n: usize) -> f64 {
    if i < n { NAN } else { x[i - n..i].iter().copied().fold(f64::MIN, f64::max) }
}

/// least-squares slope of y against 0..y.len()
fn slope(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mx = (n - 1.0) / 2.0;
    let my = y.iter().sum::<f64>() / n;
    let (num, den) = y.iter().enumerate().fold((0.0, 0.0), |(a, b), (i, v)| (a + (i as f64 - mx) * (v - my), b + (i as f64 - mx).powi(2)));
    num / den
}

/// log returns, ret[i] = ln(c[i] / c[i-1]); ret[0] = NaN
fn log_ret(b: &Bars) -> Vec<f64> {
    (0..b.n()).map(|i| if i == 0 || b.c[i - 1] <= 0.0 || b.c[i] <= 0.0 { NAN } else { (b.c[i] / b.c[i - 1]).ln() }).collect()
}

fn mean_var(x: &[f64]) -> (f64, f64) {
    let n = x.len().max(1) as f64;
    let m = x.iter().sum::<f64>() / n;
    (m, x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n)
}

/// standard normal CDF (Abramowitz-Stegun 7.1.26, error < 1.5e-7)
fn phi(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs() / std::f64::consts::SQRT_2);
    let y = 1.0 - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592) * t * (-(x * x) / 2.0).exp();
    0.5 * (1.0 + if x >= 0.0 { y } else { -y })
}

fn normal(rng: &mut Rng) -> f64 {
    let (u1, u2) = (1.0 - rng.unit() as f64, rng.unit() as f64);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn pick(rng: &mut Rng, n: usize) -> usize {
    ((rng.unit() as f64 * n as f64) as usize).min(n - 1)
}

/// one simulated candle from a return
fn step(price: &mut f64, r: f64) -> [f64; 6] {
    let o = *price;
    *price *= r.exp();
    [o, o.max(*price), o.min(*price), *price, 0.0, 0.0]
}

// ---------------------------------------------------------------- candlestick patterns

fn body_range(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.bull(i) && b.ratio(i) >= 0.8))
}
fn doji(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 4, |i| b.ratio(i) < 0.05 && b.declined(i)))
}
fn hammer(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 4, |i| b.range(i) > 0.0 && b.lower(i) >= 2.0 * b.body(i) && b.upper(i) < 0.1 * b.range(i) && b.declined(i)))
}
fn engulf(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.bear(i - 1) && b.bull(i) && b.c[i] > b.o[i - 1] && b.o[i] < b.c[i - 1]))
}
fn inside(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 2, |i| b.h[i - 1] < b.h[i - 2] && b.l[i - 1] > b.l[i - 2] && b.c[i] > b.h[i - 1]))
}
fn heikin(b: &Bars) -> Option<Vec<bool>> {
    let (ho, hc) = b.heikin();
    Some(each(b, 1, |i| hc[i] > ho[i] && hc[i - 1] <= ho[i - 1]))
}
fn marubozu(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.bull(i) && b.ratio(i) > 0.9 && b.upper(i) < 0.05 * b.range(i) && b.lower(i) < 0.05 * b.range(i)))
}
fn spinning(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 4, |i| b.ratio(i) < 0.25 && b.upper(i) > 0.3 * b.range(i) && b.lower(i) > 0.3 * b.range(i) && b.declined(i)))
}
fn dragonfly(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.ratio(i) < 0.05 && b.bot(i) >= b.l[i] + 0.9 * b.range(i)))
}
fn gravestone(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.ratio(i) < 0.05 && b.top(i) <= b.l[i] + 0.1 * b.range(i)))
}
fn pin(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.range(i) > 0.0 && b.lower(i) >= 0.66 * b.range(i) && b.bot(i) >= b.l[i] + 2.0 / 3.0 * b.range(i)))
}
fn harami(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.bear(i - 1) && b.bull(i) && b.top(i) < b.top(i - 1) && b.bot(i) > b.bot(i - 1)))
}
fn piercing(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.bear(i - 1) && b.o[i] < b.l[i - 1] && b.c[i] > b.mid(i - 1) && b.c[i] < b.o[i - 1]))
}
fn darkcloud(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.bull(i - 1) && b.o[i] > b.h[i - 1] && b.c[i] < b.mid(i - 1) && b.c[i] > b.o[i - 1]))
}
fn morning(b: &Bars) -> Option<Vec<bool>> {
    let a = atr(b, 14);
    Some(each(b, 16, |i| b.bear(i - 2) && b.body(i - 2) > a[i - 2] && b.body(i - 1) < 0.3 * a[i - 1] && b.bull(i) && b.c[i] > b.mid(i - 2)))
}
fn evening(b: &Bars) -> Option<Vec<bool>> {
    let a = atr(b, 14);
    Some(each(b, 16, |i| b.bull(i - 2) && b.body(i - 2) > a[i - 2] && b.body(i - 1) < 0.3 * a[i - 1] && b.bear(i) && b.c[i] < b.mid(i - 2)))
}
fn soldiers(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 3, |i| {
        (0..3).all(|k| b.bull(i - k) && b.range(i - k) > 0.0 && b.c[i - k] >= b.h[i - k] - 0.2 * b.range(i - k))
            && (0..2).all(|k| b.o[i - k] > b.o[i - k - 1] && b.o[i - k] < b.c[i - k - 1] && b.c[i - k] > b.c[i - k - 1])
    }))
}
fn crows(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 3, |i| {
        (0..3).all(|k| b.bear(i - k) && b.range(i - k) > 0.0 && b.c[i - k] <= b.l[i - k] + 0.2 * b.range(i - k))
            && (0..2).all(|k| b.o[i - k] < b.o[i - k - 1] && b.o[i - k] > b.c[i - k - 1] && b.c[i - k] < b.c[i - k - 1])
    }))
}
fn tweezer_bottom(b: &Bars) -> Option<Vec<bool>> {
    let a = atr(b, 14);
    Some(each(b, 16, |i| (b.l[i] - b.l[i - 1]).abs() < 0.05 * a[i] && b.declined(i - 1) && b.bull(i)))
}
fn tweezer_top(b: &Bars) -> Option<Vec<bool>> {
    let a = atr(b, 14);
    Some(each(b, 16, |i| (b.h[i] - b.h[i - 1]).abs() < 0.05 * a[i] && b.advanced(i - 1) && b.bear(i)))
}
fn kicker(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 1, |i| b.bear(i - 1) && b.bull(i) && b.o[i] > b.o[i - 1] && b.l[i] > b.o[i - 1]))
}
fn narrow(b: &Bars, k: usize) -> Vec<bool> {
    each(b, k + 1, |i| {
        let r = b.range(i - 1);
        r > 0.0 && (i - k..i).all(|j| b.range(j) >= r) && b.c[i] > b.h[i - 1]
    })
}
fn nr4(b: &Bars) -> Option<Vec<bool>> {
    Some(narrow(b, 4))
}
fn nr7(b: &Bars) -> Option<Vec<bool>> {
    Some(narrow(b, 7))
}
fn linreg(b: &Bars) -> Option<Vec<bool>> {
    let s: Vec<f64> = (0..b.n()).map(|i| if i < 13 { NAN } else { slope(&b.c[i - 13..=i]) }).collect();
    Some(each(b, 14, |i| up_through(&s, 0.0, i)))
}
fn vol_candle(b: &Bars) -> Option<Vec<bool>> {
    if !b.has_volume() {
        return None;
    }
    let vs = sma(&b.v, 20);
    Some(each(b, 20, |i| b.bull(i) && b.ratio(i) > 0.6 && b.v[i] > 1.5 * vs[i - 1]))
}

// ---------------------------------------------------------------- mean reversion / statistical

fn zscore(b: &Bars) -> Option<Vec<bool>> {
    let (m, s) = (sma(&b.c, 20), stdev(&b.c, 20));
    let z: Vec<f64> = (0..b.n()).map(|i| (b.c[i] - m[i]) / s[i]).collect();
    Some(each(b, 21, |i| down_through(&z, -2.0, i)))
}

/// rescaled-range Hurst exponent of `x` from lags 8, 16, 32, 64
fn hurst_rs(x: &[f64]) -> f64 {
    let pts: Vec<(f64, f64)> = [8usize, 16, 32, 64]
        .iter()
        .filter_map(|&m| {
            let rs: Vec<f64> = x
                .chunks_exact(m)
                .filter_map(|c| {
                    let (mean, var) = mean_var(c);
                    let (mut cum, mut hi, mut lo) = (0.0, f64::MIN, f64::MAX);
                    for v in c {
                        cum += v - mean;
                        hi = hi.max(cum);
                        lo = lo.min(cum);
                    }
                    (var > 0.0).then(|| (hi - lo) / var.sqrt())
                })
                .collect();
            (!rs.is_empty()).then(|| ((m as f64).ln(), (rs.iter().sum::<f64>() / rs.len() as f64).ln()))
        })
        .collect();
    if pts.len() < 3 {
        return NAN;
    }
    let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64, pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64);
    pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>()
}

fn hurst(b: &Bars) -> Option<Vec<bool>> {
    let r = log_ret(b);
    let h: Vec<f64> = (0..b.n()).map(|i| if i < 129 { NAN } else { hurst_rs(&r[i - 127..=i]) }).collect();
    let m = sma(&b.c, 20);
    Some(each(b, 130, |i| up_through(&h, 0.55, i) && b.c[i] > m[i]))
}

fn var_ratio(b: &Bars) -> Option<Vec<bool>> {
    let r = log_ret(b);
    let vr: Vec<f64> = (0..b.n())
        .map(|i| {
            if i < 101 {
                return NAN;
            }
            let w = &r[i - 99..=i];
            let r4: Vec<f64> = w.windows(4).map(|x| x.iter().sum()).collect();
            let (v1, v4) = (mean_var(w).1, mean_var(&r4).1);
            if v1 > 0.0 { v4 / (4.0 * v1) } else { NAN }
        })
        .collect();
    Some(each(b, 102, |i| up_through(&vr, 1.2, i) && b.c[i] > b.c[i - 4]))
}

fn rsi2(b: &Bars) -> Option<Vec<bool>> {
    let r = rsi(&b.c, 2);
    Some(each(b, 3, |i| down_through(&r, 10.0, i)))
}

fn connors(b: &Bars) -> Option<Vec<bool>> {
    let n = b.n();
    let r3 = rsi(&b.c, 3);
    let mut streak = vec![0.0; n];
    for i in 1..n {
        let p = streak[i - 1];
        streak[i] = if b.c[i] > b.c[i - 1] { if p > 0.0 { p + 1.0 } else { 1.0 } } else if b.c[i] < b.c[i - 1] { if p < 0.0 { p - 1.0 } else { -1.0 } } else { 0.0 };
    }
    let rs = rsi(&streak, 2);
    let r1 = roc(&b.c, 1);
    let pr: Vec<f64> = (0..n).map(|i| if i < 101 { NAN } else { r1[i - 100..i].iter().filter(|x| **x < r1[i]).count() as f64 }).collect();
    let crsi: Vec<f64> = (0..n).map(|i| (r3[i] + rs[i] + pr[i]) / 3.0).collect();
    Some(each(b, 102, |i| down_through(&crsi, 10.0, i)))
}

fn bb_pctb(b: &Bars) -> Option<Vec<bool>> {
    let (m, s) = (sma(&b.c, 20), stdev(&b.c, 20));
    let pb: Vec<f64> = (0..b.n()).map(|i| (b.c[i] - (m[i] - 2.0 * s[i])) / (4.0 * s[i])).collect();
    Some(each(b, 21, |i| down_through(&pb, 0.0, i)))
}

// ---------------------------------------------------------------- trend / momentum / volatility

fn ts_momentum(b: &Bars) -> Option<Vec<bool>> {
    let r = roc(&b.c, 63);
    Some(each(b, 64, |i| up_through(&r, 0.0, i)))
}
fn atr_breakout(b: &Bars) -> Option<Vec<bool>> {
    let (m, a) = (sma(&b.c, 20), atr(b, 14));
    let band: Vec<f64> = (0..b.n()).map(|i| m[i] + 1.5 * a[i]).collect();
    Some(each(b, 21, |i| cross_up(&b.c, &band, i)))
}
fn donchian(b: &Bars) -> Option<Vec<bool>> {
    Some(each(b, 22, |i| b.c[i] > prior_max(&b.h, i, 20) && b.c[i - 1] <= prior_max(&b.h, i - 1, 20)))
}
fn squeeze(b: &Bars) -> Option<Vec<bool>> {
    let (m, s, e, a) = (sma(&b.c, 20), stdev(&b.c, 20), ema(&b.c, 20), atr(b, 20));
    let on = |i: usize| m[i] + 2.0 * s[i] < e[i] + 1.5 * a[i] && m[i] - 2.0 * s[i] > e[i] - 1.5 * a[i];
    Some(each(b, 27, |i| !on(i) && (1..=5).all(|k| on(i - k)) && b.c[i] > m[i]))
}
fn macd(b: &Bars) -> Option<Vec<bool>> {
    let (f, s) = (ema(&b.c, 12), ema(&b.c, 26));
    let m: Vec<f64> = (0..b.n()).map(|i| f[i] - s[i]).collect();
    let sig = ema(&m, 9);
    Some(each(b, 35, |i| cross_up(&m, &sig, i)))
}
fn dual_ma(b: &Bars) -> Option<Vec<bool>> {
    let (f, s) = (sma(&b.c, 10), sma(&b.c, 30));
    Some(each(b, 31, |i| cross_up(&f, &s, i)))
}
fn kama(b: &Bars) -> Option<Vec<bool>> {
    let n = b.n();
    let mut k = vec![NAN; n];
    let (fast, slow) = (2.0 / 3.0, 2.0 / 31.0);
    for i in 10..n {
        let change = (b.c[i] - b.c[i - 10]).abs();
        let noise: f64 = (i - 9..=i).map(|j| (b.c[j] - b.c[j - 1]).abs()).sum();
        let er = if noise > 0.0 { change / noise } else { 0.0 };
        let sc = (er * (fast - slow) + slow).powi(2);
        k[i] = if i == 10 { b.c[i] } else { k[i - 1] + sc * (b.c[i] - k[i - 1]) };
    }
    Some(each(b, 12, |i| cross_up(&b.c, &k, i)))
}
fn cmo(b: &Bars) -> Option<Vec<bool>> {
    let x: Vec<f64> = (0..b.n())
        .map(|i| {
            if i < 14 {
                return NAN;
            }
            let (u, d) = (i - 13..=i).fold((0.0, 0.0), |(u, d), j| {
                let m = b.c[j] - b.c[j - 1];
                (u + m.max(0.0), d + (-m).max(0.0))
            });
            if u + d > 0.0 { 100.0 * (u - d) / (u + d) } else { 0.0 }
        })
        .collect();
    Some(each(b, 15, |i| up_through(&x, -50.0, i)))
}
fn roc12(b: &Bars) -> Option<Vec<bool>> {
    let r = roc(&b.c, 12);
    Some(each(b, 13, |i| up_through(&r, 0.0, i)))
}
fn adx(b: &Bars) -> Option<Vec<bool>> {
    let n = b.n();
    let mut pdm = vec![0.0; n];
    let mut mdm = vec![0.0; n];
    for i in 1..n {
        let (up, dn) = (b.h[i] - b.h[i - 1], b.l[i - 1] - b.l[i]);
        pdm[i] = if up > dn && up > 0.0 { up } else { 0.0 };
        mdm[i] = if dn > up && dn > 0.0 { dn } else { 0.0 };
    }
    let (tr, sp, sm) = (wilder(&true_range(b), 14), wilder(&pdm, 14), wilder(&mdm, 14));
    let pdi: Vec<f64> = (0..n).map(|i| if tr[i] > 0.0 { 100.0 * sp[i] / tr[i] } else { NAN }).collect();
    let mdi: Vec<f64> = (0..n).map(|i| if tr[i] > 0.0 { 100.0 * sm[i] / tr[i] } else { NAN }).collect();
    let dx: Vec<f64> = (0..n).map(|i| if pdi[i] + mdi[i] > 0.0 { 100.0 * (pdi[i] - mdi[i]).abs() / (pdi[i] + mdi[i]) } else { NAN }).collect();
    let ax = wilder(&dx, 14);
    Some(each(b, 30, |i| ax[i] > 25.0 && cross_up(&pdi, &mdi, i)))
}

// ---------------------------------------------------------------- volume / order flow

fn vwap(b: &Bars) -> Option<Vec<bool>> {
    if !b.has_volume() {
        return None;
    }
    let pv: Vec<f64> = (0..b.n()).map(|i| (b.h[i] + b.l[i] + b.c[i]) / 3.0 * b.v[i]).collect();
    let (num, den) = (rsum(&pv, 20), rsum(&b.v, 20));
    let dev: Vec<f64> = (0..b.n()).map(|i| if den[i] > 0.0 { b.c[i] - num[i] / den[i] } else { NAN }).collect();
    let sd = stdev(&dev, 20);
    let band: Vec<f64> = (0..b.n()).map(|i| -2.0 * sd[i]).collect();
    Some(each(b, 41, |i| dev[i] < band[i] && dev[i - 1] >= band[i - 1]))
}
fn vpin(b: &Bars) -> Option<Vec<bool>> {
    if !b.has_volume() {
        return None;
    }
    let n = b.n();
    let dp: Vec<f64> = (0..n).map(|i| if i == 0 { NAN } else { b.c[i] - b.c[i - 1] }).collect();
    let sd = stdev(&dp, 50);
    // bulk volume classification: buy share of a bar = PHI(price change / its typical size)
    let imb: Vec<f64> = (0..n).map(|i| if sd[i] > 0.0 { (2.0 * phi(dp[i] / sd[i]) - 1.0).abs() * b.v[i] } else { NAN }).collect();
    let (si, sv) = (rsum(&imb, 50), rsum(&b.v, 50));
    let vp: Vec<f64> = (0..n).map(|i| if sv[i] > 0.0 { si[i] / sv[i] } else { NAN }).collect();
    // threshold = 90th percentile of the previous 200 values (no lookahead)
    let thr: Vec<f64> = (0..n)
        .map(|i| {
            let mut w: Vec<f64> = vp[i.saturating_sub(200)..i].iter().copied().filter(|x| !x.is_nan()).collect();
            if w.len() < 50 {
                return NAN;
            }
            w.sort_by(f64::total_cmp);
            w[w.len() * 9 / 10]
        })
        .collect();
    Some(each(b, 101, |i| vp[i] > thr[i] && vp[i - 1] <= thr[i]))
}
fn obv(b: &Bars) -> Option<Vec<bool>> {
    if !b.has_volume() {
        return None;
    }
    let mut o = vec![0.0; b.n()];
    for i in 1..b.n() {
        o[i] = o[i - 1] + if b.c[i] > b.c[i - 1] { b.v[i] } else if b.c[i] < b.c[i - 1] { -b.v[i] } else { 0.0 };
    }
    let m = sma(&o, 20);
    Some(each(b, 21, |i| cross_up(&o, &m, i)))
}
fn cmf(b: &Bars) -> Option<Vec<bool>> {
    if !b.has_volume() {
        return None;
    }
    let mfv: Vec<f64> = (0..b.n()).map(|i| if b.range(i) > 0.0 { ((b.c[i] - b.l[i]) - (b.h[i] - b.c[i])) / b.range(i) * b.v[i] } else { 0.0 }).collect();
    let (s, v) = (rsum(&mfv, 20), rsum(&b.v, 20));
    let x: Vec<f64> = (0..b.n()).map(|i| if v[i] > 0.0 { s[i] / v[i] } else { NAN }).collect();
    Some(each(b, 21, |i| up_through(&x, 0.1, i)))
}
fn vpoc(b: &Bars) -> Option<Vec<bool>> {
    if !b.has_volume() {
        return None;
    }
    const W: usize = 50;
    const BINS: usize = 24;
    let poc: Vec<f64> = (0..b.n())
        .map(|i| {
            if i < W {
                return NAN;
            }
            let lo = b.l[i - W..i].iter().copied().fold(f64::MAX, f64::min);
            let hi = b.h[i - W..i].iter().copied().fold(f64::MIN, f64::max);
            if hi <= lo {
                return NAN;
            }
            let mut hist = [0.0; BINS];
            for j in i - W..i {
                let tp = (b.h[j] + b.l[j] + b.c[j]) / 3.0;
                hist[(((tp - lo) / (hi - lo)) * BINS as f64).min(BINS as f64 - 1.0) as usize] += b.v[j];
            }
            let k = (0..BINS).max_by(|x, y| hist[*x].total_cmp(&hist[*y])).unwrap();
            lo + (k as f64 + 0.5) * (hi - lo) / BINS as f64
        })
        .collect();
    Some(each(b, W + 1, |i| cross_up(&b.c, &poc, i)))
}

// ---------------------------------------------------------------- econometric models (simulated)

fn clean_rets(b: &Bars) -> Vec<f64> {
    log_ret(b).into_iter().filter(|r| !r.is_nan()).collect()
}

/// Gaussian elimination with partial pivoting; None if singular.
fn solve(mut a: Vec<Vec<f64>>, mut y: Vec<f64>) -> Option<Vec<f64>> {
    let n = y.len();
    for col in 0..n {
        let p = (col..n).max_by(|i, j| a[*i][col].abs().total_cmp(&a[*j][col].abs()))?;
        if a[p][col].abs() < 1e-18 {
            return None;
        }
        a.swap(col, p);
        y.swap(col, p);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            y[r] -= f * y[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        x[r] = (y[r] - (r + 1..n).map(|c| a[r][c] * x[c]).sum::<f64>()) / a[r][r];
    }
    Some(x)
}

/// ARIMA(5,1,0): AR(5) on log returns (OLS), simulated with bootstrapped residuals.
fn arima(b: &Bars, h: usize, rng: &mut Rng) -> Result<Samples, String> {
    const P: usize = 5;
    let r = clean_rets(b);
    if r.len() < 60 {
        return Err("needs 60+ candles".into());
    }
    let mut xtx = vec![vec![0.0; P + 1]; P + 1];
    let mut xty = vec![0.0; P + 1];
    let row = |t: usize| -> Vec<f64> { std::iter::once(1.0).chain((1..=P).map(|k| r[t - k])).collect() };
    for t in P..r.len() {
        let x = row(t);
        for i in 0..=P {
            xty[i] += x[i] * r[t];
            for j in 0..=P {
                xtx[i][j] += x[i] * x[j];
            }
        }
    }
    let coef = solve(xtx, xty).ok_or("AR fit failed (flat prices?)")?;
    let resid: Vec<f64> = (P..r.len()).map(|t| r[t] - row(t).iter().zip(&coef).map(|(a, b)| a * b).sum::<f64>()).collect();
    let last = b.c[b.n() - 1];
    Ok((0..SIM_PATHS)
        .map(|_| {
            let mut hist: Vec<f64> = r[r.len() - P..].to_vec();
            let mut price = last;
            (0..h)
                .map(|_| {
                    let x = coef[0] + (1..=P).map(|k| coef[k] * hist[hist.len() - k]).sum::<f64>() + resid[pick(rng, resid.len())];
                    hist.push(x);
                    step(&mut price, x)
                })
                .collect()
        })
        .collect())
}

/// GARCH(1,1) by grid-search maximum likelihood (variance targeting), simulated forward.
fn garch(b: &Bars, h: usize, rng: &mut Rng) -> Result<Samples, String> {
    let r = clean_rets(b);
    if r.len() < 100 {
        return Err("needs 100+ candles".into());
    }
    let (mu, s2) = mean_var(&r);
    if s2 <= 0.0 {
        return Err("flat prices".into());
    }
    let e: Vec<f64> = r.iter().map(|x| x - mu).collect();
    let run = |a: f64, bb: f64| -> (f64, f64) {
        let w = s2 * (1.0 - a - bb);
        let (mut v, mut ll) = (s2, 0.0);
        for x in &e {
            ll -= v.ln() + x * x / v;
            v = w + a * x * x + bb * v;
        }
        (ll, v) // v = variance forecast for the next candle
    };
    let mut best = (f64::MIN, 0.05, 0.9, s2);
    for ai in 1..=10 {
        for bi in 0..=14 {
            let (a, bb) = (ai as f64 * 0.02, 0.70 + bi as f64 * 0.02);
            if a + bb < 0.999 {
                let (ll, v) = run(a, bb);
                if ll > best.0 {
                    best = (ll, a, bb, v);
                }
            }
        }
    }
    let (_, a, bb, v0) = best;
    let w = s2 * (1.0 - a - bb);
    let last = b.c[b.n() - 1];
    Ok((0..SIM_PATHS)
        .map(|_| {
            let (mut v, mut price) = (v0, last);
            (0..h)
                .map(|_| {
                    let x = v.sqrt() * normal(rng);
                    v = w + a * x * x + bb * v;
                    step(&mut price, mu + x)
                })
                .collect()
        })
        .collect())
}

/// Ornstein-Uhlenbeck on log price via its exact AR(1) form x' = a + b x + e.
fn ou(b: &Bars, h: usize, rng: &mut Rng) -> Result<Samples, String> {
    let x: Vec<f64> = b.c.iter().filter(|c| **c > 0.0).map(|c| c.ln()).collect();
    if x.len() < 60 {
        return Err("needs 60+ candles".into());
    }
    let (x0, x1) = (&x[..x.len() - 1], &x[1..]);
    let (m0, v0) = mean_var(x0);
    let m1 = x1.iter().sum::<f64>() / x1.len() as f64;
    let cov = x0.iter().zip(x1).map(|(a, c)| (a - m0) * (c - m1)).sum::<f64>() / x0.len() as f64;
    let beta = cov / v0;
    // b >= 0.999 = half-life over ~700 candles: effectively a random walk / trend
    if !(beta > 0.0 && beta < 0.999) {
        return Err(format!("no mean reversion in this lookback (AR(1) b = {beta:.3}, needs 0 < b < 0.999)"));
    }
    let a = m1 - beta * m0;
    let sd = mean_var(&x0.iter().zip(x1).map(|(p, c)| c - a - beta * p).collect::<Vec<_>>()).1.sqrt();
    let last = *x.last().unwrap();
    Ok((0..SIM_PATHS)
        .map(|_| {
            let (mut lx, mut price) = (last, last.exp());
            (0..h)
                .map(|_| {
                    let nx = a + beta * lx + sd * normal(rng);
                    let r = nx - lx;
                    lx = nx;
                    step(&mut price, r)
                })
                .collect()
        })
        .collect())
}

/// 2-state Gaussian hidden Markov model on returns (Baum-Welch), simulated from today's state odds.
fn hmm(b: &Bars, h: usize, rng: &mut Rng) -> Result<Samples, String> {
    let r = clean_rets(b);
    let n = r.len();
    if n < 100 {
        return Err("needs 100+ candles".into());
    }
    let mut sorted = r.clone();
    sorted.sort_by(f64::total_cmp);
    let v0 = mean_var(&r).1.max(1e-12);
    let mut mu = [sorted[n / 4], sorted[3 * n / 4]];
    let mut var = [v0, v0];
    let mut tr = [[0.9, 0.1], [0.1, 0.9]];
    let mut pi = [0.5, 0.5];
    let mut alpha = vec![[0.0; 2]; n];
    for _ in 0..30 {
        let em = |s: usize, x: f64| (-(x - mu[s]).powi(2) / (2.0 * var[s])).exp() / (std::f64::consts::TAU * var[s]).sqrt() + 1e-300;
        let mut scale = vec![0.0; n];
        for t in 0..n {
            for s in 0..2 {
                alpha[t][s] = em(s, r[t]) * if t == 0 { pi[s] } else { alpha[t - 1][0] * tr[0][s] + alpha[t - 1][1] * tr[1][s] };
            }
            scale[t] = alpha[t][0] + alpha[t][1];
            alpha[t][0] /= scale[t];
            alpha[t][1] /= scale[t];
        }
        let mut beta = vec![[1.0; 2]; n];
        for t in (0..n - 1).rev() {
            for s in 0..2 {
                beta[t][s] = (0..2).map(|k| tr[s][k] * em(k, r[t + 1]) * beta[t + 1][k]).sum::<f64>() / scale[t + 1];
            }
        }
        let gamma: Vec<[f64; 2]> = (0..n)
            .map(|t| {
                let g = [alpha[t][0] * beta[t][0], alpha[t][1] * beta[t][1]];
                let z = g[0] + g[1];
                [g[0] / z, g[1] / z]
            })
            .collect();
        let mut xi = [[0.0; 2]; 2];
        for t in 0..n - 1 {
            let mut m = [[0.0; 2]; 2];
            let mut z = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    m[i][j] = alpha[t][i] * tr[i][j] * em(j, r[t + 1]) * beta[t + 1][j];
                    z += m[i][j];
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    xi[i][j] += m[i][j] / z;
                }
            }
        }
        pi = gamma[0];
        for i in 0..2 {
            let row = xi[i][0] + xi[i][1];
            if row > 0.0 {
                tr[i] = [xi[i][0] / row, xi[i][1] / row];
            }
            let g: f64 = gamma.iter().map(|g| g[i]).sum();
            if g > 0.0 {
                mu[i] = gamma.iter().zip(&r).map(|(g, x)| g[i] * x).sum::<f64>() / g;
                var[i] = (gamma.iter().zip(&r).map(|(g, x)| g[i] * (x - mu[i]).powi(2)).sum::<f64>() / g).max(1e-12);
            }
        }
    }
    let now = alpha[n - 1];
    let last = b.c[b.n() - 1];
    Ok((0..SIM_PATHS)
        .map(|_| {
            let mut s = if (rng.unit() as f64) < now[0] { 0 } else { 1 };
            let mut price = last;
            (0..h)
                .map(|_| {
                    s = if (rng.unit() as f64) < tr[s][0] { 0 } else { 1 };
                    step(&mut price, mu[s] + var[s].sqrt() * normal(rng))
                })
                .collect()
        })
        .collect())
}

// ---------------------------------------------------------------- the library

const C: &str = "CANDLESTICK";
const M: &str = "MEAN REVERSION / STATISTICAL";
const T: &str = "TREND / MOMENTUM / VOLATILITY";
const V: &str = "VOLUME / ORDER FLOW / MICROSTRUCTURE";
const E: &str = "ECONOMETRIC / MACHINE LEARNING / FACTOR";

macro_rules! s {
    ($id:literal, $name:literal, $cat:expr, $run:expr, $f:literal, $w:literal, $e:literal, $u:literal) => {
        Strat { id: concat!("q:", $id), name: $name, cat: $cat, formula: $f, what: $w, example: $e, used: $u, run: $run }
    };
}

pub static STRATS: &[Strat] = &[
    // candlestick maths
    s!("body_range", "Body-to-Range momentum", C, Run::Pattern(body_range), "|C - O| / (H - L) >= 0.80, C > O",
        "How much of the candle's total range the body fills. Near 0 = indecision (doji), above 0.8 = strong conviction; this one fires on strong UP candles.",
        "AAPL opens 180, high 184.2, low 179.8, closes 184.0: body 4.0 / range 4.4 = 0.91 -> fires.",
        "Momentum entries on intraday and daily bars; a filter for 'is this move real'."),
    s!("doji", "Doji after a decline", C, Run::Pattern(doji), "|C - O| / (H - L) < 0.05 after 3 falling candles",
        "Open and close almost equal: buyers and sellers in balance. After a fall it can mark the selling running out.",
        "Stock falls 3 days, then O 50.02 C 50.00, H 51 L 49: body 0.02 / range 2 = 0.01 -> fires.",
        "Reversal watch on daily charts; usually confirmed by the next candle."),
    s!("hammer", "Hammer (shadow-to-body)", C, Run::Pattern(hammer), "lower = min(O,C) - L >= 2 x body,  upper = H - max(O,C) < 0.1 x (H - L), after a decline",
        "Long lower shadow: price was pushed down hard and then bought back up. Classic bullish rejection.",
        "O 100 C 100.5 H 100.6 L 98: lower 2.0 >= 2 x 0.5, upper 0.1 < 0.26 -> hammer.",
        "Bottom-fishing after sell-offs, daily and 1-hour bars."),
    s!("engulf", "Bullish engulfing", C, Run::Pattern(engulf), "C(t-1) < O(t-1),  C(t) > O(t-1)  and  O(t) < C(t-1)",
        "Today's up body completely covers yesterday's down body: buyers took over.",
        "Yesterday O 52 C 50; today O 49.8 C 52.5 -> engulfs -> fires.",
        "Swing-trade reversals on daily bars; intraday on liquid stocks."),
    s!("inside", "Inside bar breakout", C, Run::Pattern(inside), "H(t-1) < H(t-2), L(t-1) > L(t-2), then C(t) > H(t-1)",
        "A candle inside the previous one = volatility squeezing; fires when price breaks out upward.",
        "Mother bar 95-105, inside bar 97-103, today closes 103.4 -> breakout.",
        "Breakout trading after quiet periods, daily/4h bars."),
    s!("heikin", "Heikin-Ashi trend turn", C, Run::Pattern(heikin), "HA_C = (O+H+L+C)/4,  HA_O = (HA_O(t-1) + HA_C(t-1))/2; fires when HA turns up",
        "Averaged candles that filter noise; a switch from red to green HA candles marks a new up-swing.",
        "HA_C 101.2 > HA_O 100.9 after a red HA candle -> trend turned up.",
        "Trend following on noisy intraday charts."),
    s!("marubozu", "Bullish Marubozu", C, Run::Pattern(marubozu), "|C - O| / (H - L) > 0.90, both shadows < 5% of range, C > O",
        "A candle that is almost all body: buyers in control from open to close.",
        "O 20.00 L 19.98 H 21.02 C 21.00: ratio 0.96 -> fires.",
        "Momentum continuation, opening-range plays."),
    s!("spinning", "Spinning top after a decline", C, Run::Pattern(spinning), "|C - O| / (H - L) < 0.25 and both shadows > 30% of range, after a decline",
        "Small body with long shadows both ways: the down move lost its push.",
        "After a fall: O 30.1 C 30.0 H 30.8 L 29.3 -> fires.",
        "Early reversal warning, daily bars."),
    s!("dragonfly", "Dragonfly doji", C, Run::Pattern(dragonfly), "|C - O| / (H - L) < 0.05, open and close in the top 10% of the range",
        "Price fell far within the candle but closed back at the high: strong rejection of lower prices.",
        "O 40.00 C 40.01 H 40.02 L 38.5 -> fires.",
        "Support tests, end of pull-backs."),
    s!("gravestone", "Gravestone doji", C, Run::Pattern(gravestone), "|C - O| / (H - L) < 0.05, open and close in the bottom 10% of the range",
        "Price rallied within the candle but closed at the low: rejection of higher prices (bearish).",
        "O 60.00 C 59.99 H 61.6 L 59.98 -> fires; history usually shows a fall after.",
        "Spotting tops / exits (this app is long-only: shows the expected drop)."),
    s!("pin", "Pin bar (bullish)", C, Run::Pattern(pin), "lower shadow >= 66% of (H - L) and the body in the top third",
        "One long wick = price rejection at that level.",
        "H 10.0 L 9.0, body 9.75-9.9: lower wick 0.75 = 75% -> fires.",
        "Price-action traders at support levels, forex and indices."),
    s!("harami", "Bullish harami", C, Run::Pattern(harami), "down candle, then max(O,C)(t) < max(O,C)(t-1) and min(O,C)(t) > min(O,C)(t-1)",
        "A small up body inside a big down body: the selling is contracting.",
        "Big red O 80 C 75, then green O 76 C 77.5 -> inside -> fires.",
        "Reversal setups, combined with support."),
    s!("piercing", "Piercing line", C, Run::Pattern(piercing), "down candle, O(t) < L(t-1), C(t) > (O(t-1)+C(t-1))/2 and C(t) < O(t-1)",
        "Gaps below yesterday's low, then closes back above the middle of yesterday's body.",
        "Yesterday O 50 C 46 L 45.5; today O 45 C 48.6 (> 48 midpoint) -> fires.",
        "Daily reversal trades after a down day."),
    s!("darkcloud", "Dark cloud cover", C, Run::Pattern(darkcloud), "up candle, O(t) > H(t-1), C(t) < midpoint of prior body, C(t) > O(t-1)",
        "Bearish mirror of the piercing line: gap up that fails below the prior midpoint.",
        "Yesterday O 46 C 50 H 50.5; today O 51 C 47.5 -> fires (expect weakness).",
        "Exits / avoiding buys at tops."),
    s!("morning", "Morning star (3 candles)", C, Run::Pattern(morning), "|C-O|(t-2) > ATR14 down, |C-O|(t-1) < 0.3 ATR, up candle closing > midpoint of candle t-2",
        "Big down candle, a small pause candle, then a strong up candle: a 3-step reversal.",
        "ATR 2: red 100->96, small 95.6->95.8, green 96->98.5 (> 98 midpoint) -> fires.",
        "Classic bottom reversal on daily charts."),
    s!("evening", "Evening star (3 candles)", C, Run::Pattern(evening), "big up candle (> ATR14), small middle body (< 0.3 ATR), down candle closing below the first's midpoint",
        "The bearish mirror of the morning star.",
        "Green 96->100, small 100.4->100.2, red 100->97.5 -> fires (expect a drop).",
        "Tops, profit-taking."),
    s!("soldiers", "Three white soldiers", C, Run::Pattern(soldiers), "3 up candles, each opening inside the prior body and closing in the top 20% of its range",
        "Three strong up candles in a row with steady buying.",
        "10->10.8, 10.5->11.4, 11.1->12.0 all near their highs -> fires.",
        "Trend start confirmation."),
    s!("crows", "Three black crows", C, Run::Pattern(crows), "3 down candles, each opening inside the prior body and closing in the bottom 20%",
        "Three strong down candles: sellers in control.",
        "12->11.2, 11.5->10.6, 10.9->10.0 -> fires (expect weakness).",
        "Trend-end warning, exits."),
    s!("tweezer_bottom", "Tweezer bottom", C, Run::Pattern(tweezer_bottom), "|L(t) - L(t-1)| < 0.05 x ATR14 after a decline, then an up candle",
        "Two candles with the same low: that price level held twice.",
        "Lows 49.98 and 50.00 with ATR 1.2 -> fires.",
        "Double-bottom style support entries."),
    s!("tweezer_top", "Tweezer top", C, Run::Pattern(tweezer_top), "|H(t) - H(t-1)| < 0.05 x ATR14 after a rise, then a down candle",
        "Two candles with the same high: the level rejected price twice (bearish).",
        "Highs 75.01 and 75.00 -> fires (expect a pull-back).",
        "Resistance / exits."),
    s!("kicker", "Bullish kicker", C, Run::Pattern(kicker), "down candle, then an up candle with O(t) > O(t-1) and L(t) > O(t-1) (gap not filled)",
        "Sentiment flips overnight: the stock gaps above yesterday's open and never looks back.",
        "Yesterday O 30 C 28; today O 30.5 L 30.2 C 32 -> fires.",
        "News-driven gap trades."),
    s!("nr4", "NR4 breakout", C, Run::Pattern(nr4), "range of candle t-1 = smallest of the last 4, then C(t) > H(t-1)",
        "The narrowest candle in 4 = calm before a move; enter on the upward break.",
        "Ranges 2.1, 1.8, 1.6, 0.9 -> next close above that 0.9 candle's high -> fires.",
        "Volatility-expansion breakouts (Toby Crabel)."),
    s!("nr7", "NR7 breakout", C, Run::Pattern(nr7), "range of candle t-1 = smallest of the last 7, then C(t) > H(t-1)",
        "Same idea over 7 candles: a rarer, stronger squeeze.",
        "Smallest range in 7 days, then a close above its high -> fires.",
        "Swing breakouts on daily bars."),
    s!("linreg", "Linear regression candles", C, Run::Pattern(linreg), "slope of the 14-candle least-squares line through closes turns from <= 0 to > 0",
        "Fits a straight line through recent closes; the slope turning positive marks a new up-trend.",
        "Slope -0.02 yesterday, +0.01 today -> fires.",
        "Smoothed trend signals, less noise than raw candles."),
    s!("vol_candle", "Volume-weighted candle", C, Run::Pattern(vol_candle), "up candle, |C-O|/(H-L) > 0.6 and V(t) / SMA20(V) > 1.5",
        "A strong up candle on 1.5x normal volume: the move is backed by real money.",
        "Body 70% of range on 3.1M shares vs 1.8M average -> fires.",
        "Filtering out low-liquidity false signals."),
    // mean reversion / statistical
    s!("zscore", "Z-score mean reversion", M, Run::Pattern(zscore), "Z = (P - SMA20) / SD20 crosses below -2",
        "Price stretched 2 standard deviations below its average; bets it snaps back.",
        "SMA 100, SD 2, price 95.5 -> Z = -2.25 -> fires.",
        "Range-bound stocks, ETFs, pairs legs."),
    s!("ou", "Ornstein-Uhlenbeck reversion", M, Run::Model(ou), "dX = theta (mu - X) dt + sigma dW on log price; fitted as X' = a + b X + e, theta = -ln b",
        "Models price being pulled back to a long-run level; simulates that pull forward. Half-life = ln 2 / theta.",
        "b = 0.95 -> theta 0.051 -> half-life about 13.5 candles.",
        "Spread trading, mean-reverting assets; tells how fast reversion should happen."),
    s!("hurst", "Hurst exponent regime", M, Run::Pattern(hurst), "rescaled range R/S ~ n^H over lags 8..64 of 128 returns; fires when H crosses above 0.55 with price above SMA20",
        "H < 0.5 = mean-reverting, 0.5 = random walk, > 0.5 = trending. Fires when an up-trend regime starts.",
        "H rises 0.52 -> 0.58 while above the 20-average -> fires.",
        "Choosing between trend and mean-reversion systems."),
    s!("var_ratio", "Variance ratio (Lo-MacKinlay)", M, Run::Pattern(var_ratio), "VR(4) = Var(4-candle returns) / (4 x Var(1-candle returns)) over 100 candles crosses above 1.2, price up",
        "VR > 1 means returns keep going the same way (momentum); < 1 means they reverse.",
        "Var1 = 1.0, Var4 = 5.0 -> VR 1.25 -> momentum regime -> fires.",
        "Testing if a market is a random walk."),
    s!("rsi2", "RSI-2", M, Run::Pattern(rsi2), "RSI(2) = 100 - 100 / (1 + avg gain / avg loss) over 2 candles crosses below 10",
        "Ultra-short oversold: 2 sharp down candles. Larry Connors' dip-buying rule.",
        "Two down days of -2% and -3% -> RSI(2) = 4 -> fires.",
        "Buying dips in up-trending stocks / indices."),
    s!("connors", "Connors RSI", M, Run::Pattern(connors), "CRSI = (RSI(3) + RSI(streak, 2) + percent rank of 1-candle change over 100) / 3 < 10",
        "Combines price RSI, how long a down-streak has run and how extreme today's move is.",
        "RSI3 8, streak RSI 5, rank 3 -> CRSI 5.3 -> fires.",
        "Short-term mean reversion in ETFs and large caps."),
    s!("bb_pctb", "Bollinger %B", M, Run::Pattern(bb_pctb), "%B = (P - (SMA20 - 2 SD)) / (4 SD) crosses below 0",
        "Where price sits inside the bands: 0 = lower band, 1 = upper. Below 0 = statistically stretched.",
        "Bands 96-104, price 95.2 -> %B = -0.1 -> fires.",
        "Contrarian entries; bandwidth also flags squeezes."),
    s!("pairs", "Pairs trading (Engle-Granger)", M, Run::Rule(pairs), "log A = a + b log B over 60 candles (OLS); spread z = residual / SD; trade when ADF t < -3.04 (cointegrated) and z crosses below -2",
        "Two stocks that move together drift apart and come back. Long-only here: fires when THIS stock is the cheap leg (you'd buy it; the short leg is skipped). Partner = the tracked stock moving most like it.",
        "KO vs PEP: spread z = -2.3 with ADF t = -3.6 -> KO unusually cheap vs PEP -> buy KO.",
        "Statistical arbitrage desks."),
    s!("kalman_pairs", "Kalman filter pairs", M, Run::Rule(kalman_pairs), "state (a, b) random walk; log A(t) = a + b log B(t) + e; innovation e / sqrt(Q) crosses below -2",
        "Pairs trading where the hedge ratio adapts every candle (Kalman filter, Chan's version). Long-only: fires when this stock is unexpectedly cheap vs its partner.",
        "b drifts 0.82 -> 0.87; today's innovation is -2.4 SD -> buy.",
        "Dynamic stat-arb."),
    s!("johansen", "Johansen cointegration basket", M, Run::Rule(johansen_basket), "VECM(1) on log prices of this stock + its 2 closest partners (120 candles): eigenvalues of S11^-1 S10 S00^-1 S01; trace > 29.80 -> basket w . log P, z crosses below -2",
        "Pairs trading extended to a 3-stock basket. Johansen finds the weights that make the basket mean-revert; fires when the basket is low, i.e. this stock (weight 1) is cheap vs the other two.",
        "w = (1, -0.6, -0.3): basket z = -2.2 -> buy this stock.",
        "Sector stat-arb."),
    // trend / momentum / volatility
    s!("ts_momentum", "Time-series momentum", T, Run::Pattern(ts_momentum), "return over the last 63 candles crosses above 0",
        "Things that have gone up tend to keep going up for a while.",
        "63 daily candles (~3 months): -1% -> +0.5% -> fires.",
        "Managed futures / CTAs, trend-following funds."),
    s!("xs_momentum", "Cross-sectional momentum", T, Run::Rule(xs_momentum), "rank the tracked symbols by C(t-5)/C(t-63) - 1; fires when this one enters the top 3",
        "Buy the recent winners relative to everything else you track (the classic 12-1 month idea, in candles). Long-only: the 'short the losers' half is skipped.",
        "NVDA moves from 4th to 2nd by 63-candle return -> fires.",
        "Equity factor funds."),
    s!("atr_breakout", "Volatility (ATR) breakout", T, Run::Pattern(atr_breakout), "C crosses above SMA20 + 1.5 x ATR14",
        "A move bigger than normal noise: the market wants to go somewhere.",
        "SMA 50, ATR 1 -> close 51.6 > 51.5 -> fires.",
        "Breakout systems, volatility-adjusted entries."),
    s!("donchian", "Donchian breakout (Turtles)", T, Run::Pattern(donchian), "C(t) > highest high of the previous 20 candles",
        "New 20-candle high = start of a trend. The famous Turtle traders' rule.",
        "20-day high 120, close 121 -> fires.",
        "Trend following on futures, stocks."),
    s!("squeeze", "Keltner / TTM squeeze", T, Run::Pattern(squeeze), "Bollinger(20, 2) inside Keltner(EMA20 +- 1.5 ATR20) for 5 candles, then outside, with C > SMA20",
        "Volatility compressed (bands inside channels) and now releasing upward.",
        "5 days of squeeze, today bands expand and price is above the average -> fires.",
        "Breakout timing (John Carter's TTM squeeze)."),
    s!("macd", "MACD crossover", T, Run::Pattern(macd), "MACD = EMA12 - EMA26 crosses above its 9-period EMA (signal line)",
        "Fast trend overtaking slow trend.",
        "MACD -0.05 -> +0.02 crossing a signal of 0.00 -> fires.",
        "Classic trend entries on any timeframe."),
    s!("dual_ma", "Dual moving average crossover", T, Run::Pattern(dual_ma), "SMA10 crosses above SMA30",
        "Short average crossing the long average = trend turning up ('golden cross' idea).",
        "SMA10 101.2 > SMA30 101.0 after being below -> fires.",
        "Simple trend systems."),
    s!("kama", "Kaufman adaptive moving average", T, Run::Pattern(kama), "ER = |C - C(t-10)| / sum|dC|; sc = (ER (2/3 - 2/31) + 2/31)^2; KAMA += sc (C - KAMA); C crosses above KAMA",
        "An average that speeds up in trends and stays flat in chop.",
        "ER 0.8 in a clean trend -> KAMA follows fast; price crosses above -> fires.",
        "Trend filters with fewer whipsaws."),
    s!("cmo", "Chande momentum oscillator", T, Run::Pattern(cmo), "CMO = 100 (up moves - down moves) / (up + down) over 14 candles crosses above -50",
        "Pure velocity from -100 to +100; leaving -50 = coming out of oversold.",
        "Up 2, down 7 -> CMO -55.6; next day -45 -> fires.",
        "Momentum turns."),
    s!("roc", "Rate of change", T, Run::Pattern(roc12), "ROC = C / C(t-12) - 1 crosses above 0",
        "Price is now higher than 12 candles ago.",
        "C 103 vs 102.8 twelve days ago -> +0.2% -> fires.",
        "Momentum filters."),
    s!("adx", "ADX / directional movement", T, Run::Pattern(adx), "ADX(14) > 25 and +DI crosses above -DI",
        "ADX measures trend strength; +DI over -DI says the strong trend is up.",
        "ADX 31, +DI 24 crosses -DI 22 -> fires.",
        "Trend-strength filter (Welles Wilder)."),
    s!("vol_target", "Volatility targeting", T, Run::Sizing, "weight = target vol / stock vol, target = median predicted vol of the tracked stocks, capped at 1 (the max-position limit)",
        "Hold less of wild stocks and full size of calm ones so each position carries similar risk. A sizing rule: switch SIZE to VOL TARGET in a strategy window and AUTO buys use it.",
        "Median vol 2%, NVDA 4% -> NVDA bought at half size.",
        "Risk parity and portfolio sizing."),
    // volume / order flow
    s!("vwap", "VWAP deviation", V, Run::Pattern(vwap), "VWAP20 = sum(typical price x V) / sum V; fires when C - VWAP crosses below -2 SD",
        "Volume-weighted fair price; far below it = stretched, bet on a return.",
        "VWAP 50.0, SD of deviation 0.3, close 49.3 -> fires.",
        "Intraday execution and mean reversion (institutions benchmark to VWAP)."),
    s!("vpin", "VPIN (bulk volume, approx.)", V, Run::Pattern(vpin), "buy share = PHI(dP / SD50(dP)); VPIN = sum |buy - sell| / sum V over 50 candles; fires above its 90th percentile",
        "Estimates 'toxic', one-sided order flow. Real VPIN uses tick trades; this uses candles (bulk volume classification).",
        "VPIN 0.42 > its 90th percentile 0.38 -> fires.",
        "Flash-crash / volatility warnings."),
    s!("obv", "On-balance volume", V, Run::Pattern(obv), "OBV += V if C up, -V if C down; fires when OBV crosses above its SMA20",
        "Running total of volume on up vs down candles: accumulation shows before price.",
        "OBV turns up through its average while price is flat -> fires.",
        "Spotting institutional accumulation."),
    s!("cmf", "Chaikin money flow", V, Run::Pattern(cmf), "CMF = sum(((C-L) - (H-C)) / (H-L) x V) / sum V over 20, crosses above +0.10",
        "Closes near the highs on volume = money flowing in.",
        "CMF 0.08 -> 0.12 -> fires.",
        "Confirming breakouts with volume."),
    s!("vpoc", "Volume point of control", V, Run::Pattern(vpoc), "histogram of volume by price over the last 50 candles (24 bins); POC = busiest price; C crosses above POC",
        "The price with the most trading = fair value; reclaiming it is bullish.",
        "POC 101.5, price moves 101.2 -> 101.8 -> fires.",
        "Market Profile traders, futures."),
    s!("obi", "Order book imbalance", V, Run::Na("needs bid/ask sizes from the order book; Yahoo only gives them with a login, and a live book feed (IBKR) isn't connected yet"), "OBI = (V_bid - V_ask) / (V_bid + V_ask) over the top N book levels",
        "More size waiting to buy than sell predicts the next tick.",
        "Bid 9,000 vs ask 3,000 -> OBI 0.5 -> next tick likely up.",
        "High-frequency market making."),
    s!("hawkes", "Hawkes self-exciting moves", V, Run::Model(hawkes), "events = candles moving > 2 SD; intensity lambda(t) = mu + alpha sum e^(-beta (t - t_i)), maximum likelihood (grid); simulated with resampled jump / calm returns",
        "Big moves come in bursts: each one raises the odds of another. Fitted on big candle moves (individual orders would need tick data), so the forecast carries today's 'burstiness'.",
        "Branching 0.6, beta 0.3: after two shocks the next-candle jump odds are 3x normal.",
        "HFT, execution, volatility risk."),
    // econometric / ML / factor
    s!("arima", "ARIMA(5,1,0)", E, Run::Model(arima), "r(t) = a0 + a1 r(t-1) + ... + a5 r(t-5) + e, r = log return (OLS), simulated with resampled errors",
        "Forecasts the next returns from the last few returns.",
        "a1 = 0.08: a +1% candle adds +0.08% to the next forecast.",
        "Classic econometric forecasting."),
    s!("garch", "GARCH(1,1)", E, Run::Model(garch), "sigma^2(t) = w + a e^2(t-1) + b sigma^2(t-1), fitted by maximum likelihood (grid)",
        "Forecasts volatility: calm follows calm, storms follow storms. Direction = average return.",
        "a 0.08, b 0.90 -> a shock fades by 2% per candle.",
        "Option pricing, risk limits, VaR."),
    s!("hmm", "Hidden Markov regimes", E, Run::Model(hmm), "2 hidden states, returns ~ N(mu_s, sigma_s^2), switching matrix A, fitted by Baum-Welch",
        "Infers whether the market is in a calm-up or a volatile-down regime and simulates regime switches.",
        "State 1: mu +0.1%, sd 0.8%; state 2: mu -0.2%, sd 2.1%; today 85% state 1.",
        "Regime detection, switching strategies on/off."),
    s!("pca", "PCA statistical arbitrage", E, Run::Rule(pca), "top 3 eigenvectors of the other tracked stocks' return correlations (60 candles); regress this stock on them; s = z of the cumulative residual crosses below -1.5",
        "Removes the market/sector moves (principal components) and trades what is left of the stock (Avellaneda-Lee). Fires when the stock has fallen behind its factors.",
        "NVDA residual s-score -1.7 vs the tracked stocks' factors -> buy.",
        "Quant equity market-neutral funds."),
    s!("factor", "Factor investing (smart beta)", E, Run::Rule(factor), "composite = mean z of value (book/price), quality (4 x quarterly net income / equity), size (-ln market cap), momentum (63-5 return), low vol (-SD 60); fires on entering the top 3",
        "Ranks the tracked companies by measurable traits. Accounts from Yahoo (last ~5 quarters, used 45 days after quarter end, so no peeking); older candles rank on the price factors only. Companies only.",
        "Low price-to-book + high ROE + steady price -> ranks 2nd of 12 -> fires.",
        "Smart-beta ETFs."),
    s!("ff5", "Fama-French factors + momentum", E, Run::CtxModel(ff5), "r = a + b1 MKT + b2 SMB + b3 HML + b4 RMW + b5 UMD + e, factors from ETFs: SPY, IWM-SPY, IWD-IWF, QUAL-SPY, MTUM-SPY; simulated with a = 0",
        "Explains the stock's moves by market, size, value, profitability and momentum exposures, then simulates futures from resampled factor days. ETF proxies stand in for the academic factor files; CMA (investment) has no clean ETF and is left out.",
        "b(MKT) 1.3, b(HML) -0.4: a growth-heavy, high-beta stock.",
        "Academic and institutional portfolio construction."),
    s!("pead", "Post-earnings drift (PEAD)", E, Run::Rule(pead), "earnings day = first day 10-75 days after a fiscal quarter end with V > 1.8 x SMA20(V); fires if C > C(t-1) that day and EPS > EPS a year earlier (when known)",
        "Stocks keep drifting for weeks after a good earnings surprise. The surprise is measured by the day's price reaction (analyst forecasts aren't free); quarter ends come from Yahoo's accounts. 1 DAY candles only.",
        "Reports, jumps +6% on 3x volume, EPS 1.20 vs 1.00 a year ago -> fires.",
        "Event-driven funds."),
    s!("trees", "Gradient-boosted trees (XGBoost-style)", E, Run::Rule(trees), "60 depth-3 trees, logistic loss, Newton leaves, lr 0.1; features: 1-20 candle returns, RSI14, ATR%, distance from SMA20/50, range position; label = up after h candles",
        "Learns non-linear rules from technical features, trained in Rust on the lookback. Scores come from models that never saw that candle's future (4 purged time folds). Fires when the score is in its top 20%.",
        "Features RSI 28, -4% vs SMA20 -> score 0.63 (top 20%) -> fires.",
        "Modern quant desks."),
    s!("deep", "LSTM (deep learning)", E, Run::Rule(lstm), "LSTM, 8 hidden units, 16-candle sequences of (return, distance from SMA20, range position) -> P(up after h); Adam, 8 epochs, BPTT",
        "A small recurrent neural network trained in Rust on the lookback, out-of-fold like the trees. Fires when its score is in the top 20%. (Temporal Fusion Transformers need far more data; Kronos in AI FORECAST is the big transformer.)",
        "Sequence of 3 down candles near the range low -> P(up) 0.58 (top 20%) -> fires.",
        "Research desks."),
    s!("rl", "Reinforcement learning (Q-learning)", E, Run::Rule(rl), "Q(s, pos, a) += 0.1 (reward + 0.95 max Q(s', a') - Q); reward = position x next return - 0.05% per change; states = RSI bucket x trend x 5-candle return bucket",
        "An agent learns by trial and error when to be long, paying trading costs. Tabular Q-learning (a DQN/PPO would replace the table with a neural net). Fires when it would switch from flat to long.",
        "State 'RSI < 30, below SMA20, sharp drop': Q(long) > Q(flat) -> fires.",
        "Execution optimisation."),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(closes: &[f64]) -> Vec<Candle> {
        closes.iter().enumerate().map(|(i, c)| Candle { t: i as i64 * 60, o: c - 0.1, h: c + 0.5, l: c - 0.6, c: *c, v: 1000 + (i as u64 % 7) * 100 }).collect()
    }

    /// A daily market: a common factor + sector cycles, 12 "companies" with accounts, factor ETFs,
    /// earnings-day volume spikes. Every runnable strategy must produce a forecast or a reason.
    #[test]
    fn every_runnable_strategy_runs_on_real_looking_data() {
        const N: usize = 700;
        let mut rng = Rng::new(42);
        let mkt: Vec<f64> = (0..N).map(|_| 0.01 * normal(&mut rng)).collect();
        let series = |beta: f64, seed: u64| -> Vec<Candle> {
            let mut r = Rng::new(seed);
            let mut p = 100.0;
            (0..N)
                .map(|i| {
                    let o = p;
                    p *= (0.0003 + beta * mkt[i] + 0.012 * normal(&mut r) + 0.004 * (i as f64 / 9.0 + seed as f64).sin()).exp();
                    let spike = if (i * 86_400 + 40 * 86_400) % 7_889_000 < 86_400 { 4 } else { 1 };
                    Candle { t: i as i64 * 86_400, o, h: o.max(p) * 1.004, l: o.min(p) * 0.996, c: p, v: spike * (1_000_000 + (r.unit() * 300_000.0) as u64) }
                })
                .collect()
        };
        let cs = series(1.0, 1);
        let mut ex = Extra::default();
        // two partners cointegrated with X: X's price times a mean-reverting (AR 0.8) wobble
        for (name, seed) in [("PA1", 7u64), ("PA2", 8)] {
            let (mut r, mut e) = (Rng::new(seed), 0.0);
            ex.uni.push((name.into(), cs.iter().map(|c| {
                e = 0.8 * e + 0.01 * normal(&mut r);
                Candle { c: c.c * e.exp(), ..*c }
            }).collect()));
        }
        for (k, s) in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK"].iter().enumerate() {
            ex.uni.push((s.to_string(), series(0.6 + 0.08 * k as f64, 10 + k as u64)));
        }
        for (k, s) in FACTOR_ETFS.iter().enumerate() {
            ex.uni.push((s.to_string(), series(0.9 + 0.05 * k as f64, 50 + k as u64)));
        }
        let names: Vec<String> = std::iter::once("X".to_string()).chain(ex.uni.iter().skip(2).take(11).map(|u| u.0.clone())).collect();
        for (k, s) in names.iter().enumerate() {
            let q = (0..8).map(|j| FundQ { end_t: (j as i64 + 1) * 7_889_000 + 86_400 * 200, eps: 1.0 + 0.1 * j as f64, equity: 1e9 * (1.0 + k as f64), net_income: 5e7 * (1.0 + j as f64 * 0.1), shares: 1e7 }).collect();
            ex.funds.insert(s.clone(), q);
        }
        let mut ran = vec![];
        for s in STRATS.iter().filter(|s| s.runnable()) {
            match samples(s, "X", &cs, 5, 1, &ex) {
                Ok((v, _)) => {
                    assert!(!v.is_empty() && v.iter().all(|p| p.len() == 5), "{}", s.id);
                    assert!(v.iter().flatten().all(|c| c.iter().all(|x| x.is_finite())), "{} not finite", s.id);
                }
                Err(e) => assert!(!e.is_empty(), "{}", s.id),
            }
            // rules must compute their full signal history (a reason only for missing data)
            if let Run::Rule(f) = s.run {
                let b = Bars::new(&cs);
                let ctx = Ctx { h: 5, sym: "X", uni: ex.uni.iter().map(|u| (u.0.as_str(), align(&b.t, &u.1))).collect(), funds: &ex.funds };
                let fires = f(&b, &ctx).unwrap_or_else(|e| panic!("{}: {e}", s.id));
                assert_eq!(fires.len(), N, "{}", s.id);
                ran.push((s.id, fires.iter().filter(|x| **x).count()));
            }
        }
        println!("rule firings: {ran:?}");
        assert!(ran.iter().all(|r| r.1 > 0), "a rule never fired: {ran:?}");
        assert_eq!(STRATS.iter().map(|s| s.id).collect::<std::collections::HashSet<_>>().len(), STRATS.len(), "duplicate id");
    }

    #[test]
    fn pattern_forecast_is_what_followed_without_lookahead() {
        // price steps: every 10th candle a +5% jump is followed by a rise; check the forecast uses only closed cases
        let mut c = vec![100.0; 200];
        for i in 1..200 {
            c[i] = c[i - 1] * if i % 10 == 0 { 1.05 } else { 1.001 };
        }
        let cs: Vec<Candle> = c.iter().enumerate().map(|(i, x)| Candle { t: i as i64, o: if i % 10 == 0 { x / 1.05 } else { x / 1.001 }, h: x * 1.0001, l: if i % 10 == 0 { x / 1.0501 } else { x / 1.01 }, c: *x, v: 0 }).collect();
        let b = Bars::new(&cs);
        let f = body_range(&b).unwrap();
        assert!(f[190] && f[180] && !f[185]);
        let (s, txt) = samples(find("q:body_range").unwrap(), "X", &cs[..191], 3, 1, &Extra::default()).unwrap();
        // cases 10..=180 whose 3 next candles closed before candle 190: 10,20,...,180 = 18 cases
        assert_eq!(s.len(), 18, "{txt}");
        let last = cs[190].c;
        assert!(s.iter().all(|p| (p[2][3] / last - 1.001f64.powi(3)).abs() < 1e-9));
    }

    #[test]
    fn models_fit_known_processes() {
        // OU: strongly mean-reverting log price around ln 50
        let mut rng = Rng::new(3);
        let mut x = 50f64.ln();
        let c: Vec<f64> = (0..400).map(|_| { x = 0.2 * 50f64.ln() + 0.8 * x + 0.01 * normal(&mut rng); x.exp() }).collect();
        let b = Bars::new(&bars(&c));
        let s = ou(&b, 20, &mut Rng::new(9)).unwrap();
        let mean_end = s.iter().map(|p| p[19][3]).sum::<f64>() / s.len() as f64;
        assert!((mean_end - 50.0).abs() < 1.0, "{mean_end}");
        // a random walk with drift is not mean-reverting enough for OU but GARCH/ARIMA/HMM still run
        let w: Vec<f64> = (0..300).map(|i| 100.0 * (1.0 + 0.001 * i as f64)).collect();
        assert!(ou(&Bars::new(&bars(&w)), 5, &mut Rng::new(1)).is_err());
        for f in [arima, garch, hmm] {
            assert_eq!(f(&b, 5, &mut Rng::new(2)).unwrap().len(), SIM_PATHS);
        }
        assert!((phi(0.0) - 0.5).abs() < 1e-7 && (phi(1.96) - 0.975).abs() < 1e-3);
    }
}

/// Live check against Yahoo: `cargo test --release live_yahoo -- --ignored --nocapture`
#[cfg(test)]
mod live {
    use super::*;

    /// Which strategies had an edge: every tracked stock, real Yahoo data, h = 5 candles ahead.
    /// Signals: average 5-candle return after each firing minus that stock's average 5-candle return
    /// (excess), win rate, split into older / newer half of the history. Models: walk-forward, refit
    /// every 10 candles on the 500 before, forecast direction vs what happened.
    /// `cargo test --release rank_strategies -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn rank_strategies() {
        const H: usize = 5;
        let y = crate::yahoo::Yahoo::new();
        let watch = crate::shared::Settings::default().watchlist;
        let mut funds = std::collections::HashMap::new();
        for s in &watch {
            if let Ok(f) = y.fundamentals(s) {
                funds.insert(s.clone(), f);
            }
        }
        for tf in [4usize, 3] {
            let (iv, rg, name) = crate::shared::TIMEFRAMES[tf];
            let mut uni: Vec<(String, Vec<Candle>)> = vec![];
            for s in watch.iter().map(|s| s.as_str()).chain(FACTOR_ETFS) {
                if let Ok((c, _)) = y.chart(s, iv, rg) {
                    uni.push((s.to_string(), c[c.len().saturating_sub(1024)..].to_vec()));
                }
            }
            println!("
===== {name} candles, next {H} candles, {} stocks =====", watch.len());
            println!("SIGNALS{:<34}{:>7}{:>9}{:>8}{:>10}{:>10}{:>8}", "", "n", "excess", "win%", "old half", "new half", "stocks+");
            let mut rows = vec![];
            for st in STRATS.iter().filter(|s| matches!(s.run, Run::Pattern(_) | Run::Rule(_))) {
                let (mut ex_all, mut ex_old, mut ex_new, mut wins, mut pos_stocks) = (vec![], vec![], vec![], 0, 0);
                for sym in &watch {
                    let Some(cs) = uni.iter().find(|u| &u.0 == sym).map(|u| &u.1) else { continue };
                    let b = Bars::new(cs);
                    let n = b.n();
                    if n < 200 {
                        continue;
                    }
                    let fires = match &st.run {
                        Run::Pattern(f) => f(&b),
                        Run::Rule(f) => {
                            let ctx = Ctx { h: H, sym, uni: uni.iter().filter(|u| &u.0 != sym).map(|u| (u.0.as_str(), align(&b.t, &u.1))).collect(), funds: &funds };
                            f(&b, &ctx).ok()
                        }
                        _ => None,
                    };
                    let Some(fires) = fires else { continue };
                    let fwd = |i: usize| (b.c[i + H] / b.c[i]).ln();
                    let base = (0..n - H).map(fwd).sum::<f64>() / (n - H) as f64;
                    let mut mine = vec![];
                    for i in (0..n - H).filter(|&i| fires[i]) {
                        let e = fwd(i) - base;
                        mine.push(e);
                        if fwd(i) > 0.0 {
                            wins += 1;
                        }
                        if i < n / 2 { ex_old.push(e) } else { ex_new.push(e) }
                    }
                    if !mine.is_empty() && mine.iter().sum::<f64>() > 0.0 {
                        pos_stocks += 1;
                    }
                    ex_all.extend(mine);
                }
                let avg = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 * 100.0 };
                let n = ex_all.len();
                rows.push((avg(&ex_all), format!("{:<40}{n:>7}{:>+8.2}%{:>7.0}%{:>+9.2}%{:>+9.2}%{:>8}", st.name, avg(&ex_all), wins as f64 * 100.0 / n.max(1) as f64, avg(&ex_old), avg(&ex_new), pos_stocks)));
            }
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            rows.iter().for_each(|r| println!("  {}", r.1));
            println!("MODELS{:<35}{:>7}{:>9}{:>8}", "", "n", "dir hit", "corr");
            for st in STRATS.iter().filter(|s| matches!(s.run, Run::Model(_) | Run::CtxModel(_))) {
                let (mut hit, mut tot, mut xs, mut ys) = (0, 0, vec![], vec![]);
                for sym in &watch {
                    let Some(cs) = uni.iter().find(|u| &u.0 == sym).map(|u| &u.1) else { continue };
                    let n = cs.len();
                    for i in (500..n - H).step_by(10) {
                        let w = &cs[i - 500..=i];
                        let ex = Extra { uni: uni.iter().filter(|u| FACTOR_ETFS.contains(&u.0.as_str())).map(|u| (u.0.clone(), u.1.iter().filter(|c| c.t <= cs[i].t).copied().collect())).collect(), funds: Default::default() };
                        let Ok((v, _)) = samples(st, sym, w, H, i as u64, &ex) else { continue };
                        let f = v.iter().map(|p| (p[H - 1][3] / cs[i].c).ln()).sum::<f64>() / v.len() as f64;
                        let a = (cs[i + H].c / cs[i].c).ln();
                        tot += 1;
                        hit += ((f > 0.0) == (a > 0.0)) as u32;
                        xs.push(f);
                        ys.push(a);
                    }
                }
                let (mx, my) = (mean_var(&xs), mean_var(&ys));
                let cov = xs.iter().zip(&ys).map(|(x, y)| (x - mx.0) * (y - my.0)).sum::<f64>() / xs.len().max(1) as f64;
                println!("  {:<40}{tot:>7}{:>8.1}%{:>+8.3}", st.name, hit as f64 * 100.0 / tot.max(1) as f64, cov / (mx.1 * my.1).sqrt());
            }
        }
    }

    #[test]
    #[ignore]
    fn live_yahoo() {
        let y = crate::yahoo::Yahoo::new();
        let watch = crate::shared::Settings::default().watchlist;
        for tf in [4usize, 3] {
            let (iv, rg, _) = crate::shared::TIMEFRAMES[tf];
            let mut ex = Extra::default();
            for s in watch.iter().map(|s| s.as_str()).chain(FACTOR_ETFS) {
                if let Ok((c, _)) = y.chart(s, iv, rg) {
                    ex.uni.push((s.to_string(), c));
                }
            }
            for s in &watch {
                if let Ok(f) = y.fundamentals(s) {
                    ex.funds.insert(s.clone(), f);
                }
            }
            println!("== {} : {} series, {} with accounts", crate::shared::TIMEFRAMES[tf].2, ex.uni.len(), ex.funds.values().filter(|f| !f.is_empty()).count());
            let target = ex.uni.iter().find(|u| u.0 == "AAPL").unwrap().1.clone();
            let w = &target[target.len().saturating_sub(1024)..];
            for s in STRATS.iter().filter(|s| s.runnable() && !matches!(s.run, Run::Pattern(_))) {
                let t0 = std::time::Instant::now();
                let r = samples(s, "AAPL", w, 5, 1, &ex);
                let ms = t0.elapsed().as_millis();
                // the signal history too (how often it fires on real data)
                let fires = if let Run::Rule(f) = s.run {
                    let b = Bars::new(w);
                    let ctx = Ctx { h: 5, sym: "AAPL", uni: ex.uni.iter().filter(|u| u.0 != "AAPL").map(|u| (u.0.as_str(), align(&b.t, &u.1))).collect(), funds: &ex.funds };
                    f(&b, &ctx).map_or_else(|e| e, |v| format!("{} firings", v.iter().filter(|x| **x).count()))
                } else {
                    String::new()
                };
                match r {
                    Ok((v, txt)) => println!("  {:<16}{ms:>6} ms  {} paths  {txt}  {fires}", s.id, v.len()),
                    Err(e) => println!("  {:<16}{ms:>6} ms  -- {e}  {fires}", s.id),
                }
            }
        }
    }
}

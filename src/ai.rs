//! Background forecast loops in the engine, one per desk: AI FORECAST (Kronos, Black-Scholes) and
//! MATH FORECAST (quant.rs strategies). Both turn sampled futures into the same statistics.
use crate::engine::Shared;
use crate::kronos::{self, Kronos};
use crate::shared::*;
use crate::yahoo::Yahoo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Sampled futures per prediction; their spread gives P(up) and the p10..p90 range.
pub const SAMPLES: usize = 16;
/// Simulated price paths per Black-Scholes Monte Carlo prediction (cheap: ~ms).
pub const MC_PATHS: usize = 1000;
/// "Unusually high increase": expected rise at least this many typical moves (sigma) ...
pub const UNUSUAL_Z: f64 = 2.0;
// ... and P(up) at or above the user's "Min confidence" setting.

const MIN_CANDLES: usize = 32;
/// Re-run on unchanged candles only after this long; keeps CPU use down.
const SINGLE_REFRESH: Duration = Duration::from_secs(5);
/// Each leaderboard symbol is re-predicted about once a minute (30 symbols x ~1 s each fits).
const BOARD_REFRESH: Duration = Duration::from_secs(60);

fn status(s: &mut crate::engine::State, math: bool) -> &mut String {
    if math { &mut s.math_status } else { &mut s.ai_status }
}

/// Sets a desk's status line; `clear` also drops that desk's predictions (the other desk keeps its own).
fn set_status(st: &Shared, math: bool, msg: &str, clear: bool) {
    let mut s = st.lock().unwrap();
    let has = s.predictions.keys().any(|k| k.0 == math) || s.board.keys().any(|k| is_math(&k.0) == math);
    if status(&mut s, math) != msg || (clear && has) {
        *status(&mut s, math) = msg.to_string();
        if clear {
            s.predictions.retain(|k, _| k.0 != math);
            s.board.retain(|k, _| is_math(&k.0) != math);
        }
        s.bump();
    }
}

/// What makes a desk's samples.
#[derive(Clone, Copy)]
enum Eng<'a> {
    Kronos(&'a Kronos),
    Mc,
    Quant(&'static crate::quant::Strat),
}

/// Which panel a prediction feeds: one-stock PREDICTION OUTPUT or the LEADERBOARD / COMPARE.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Single,
    Board,
}

/// What the last run of (kind, model, symbol) used: last candle time + close, timeframe, horizon, when.
type LastRun = HashMap<(Kind, String, String, usize, usize), (i64, f64, usize, usize, Instant)>;

/// Leaderboard candles are reused for this long, so two compared models share one download.
const CANDLE_CACHE: Duration = Duration::from_secs(30);
/// Other stocks' candles for cross-stock MATH strategies (pairs, PCA, ...).
/// shortcut: up to 2 min old, so the partner's newest candle can lag on 1 MIN charts; refetch per job if it matters.
const UNI_CACHE: Duration = Duration::from_secs(120);
/// Company accounts change quarterly.
const FUND_CACHE: Duration = Duration::from_secs(6 * 3600);

type Cache = HashMap<(String, usize), (Instant, Vec<Candle>)>;

/// Other symbols' candles and company accounts a MATH strategy needs (cached).
#[allow(clippy::too_many_arguments)]
fn gather(y: &Yahoo, st: &Shared, cache: &mut Cache, funds: &mut HashMap<String, (Instant, Vec<crate::quant::FundQ>)>, q: &crate::quant::Strat, sym: &str, tf: usize, watch: &[String]) -> crate::quant::Extra {
    let mut ex = crate::quant::Extra::default();
    let mut syms: Vec<String> = vec![];
    if q.needs_universe() {
        syms.extend(watch.iter().filter(|s| *s != sym).cloned());
    }
    if q.needs_etfs() {
        syms.extend(crate::quant::FACTOR_ETFS.iter().map(|s| s.to_string()));
    }
    for s in syms {
        let fresh = cache.get(&(s.clone(), tf)).filter(|c| c.0.elapsed() < UNI_CACHE).map(|c| c.1.clone());
        let c = fresh.or_else(|| {
            let c = fetch(y, st, &s, tf)?;
            cache.insert((s.clone(), tf), (Instant::now(), c.clone()));
            Some(c)
        });
        if let Some(c) = c {
            ex.uni.push((s, c));
        }
    }
    if q.needs_funds() {
        let who: Vec<String> = std::iter::once(sym.to_string()).chain(if q.needs_universe() { watch.to_vec() } else { vec![] }).collect();
        for s in who {
            let f = match funds.get(&s) {
                Some((t, f)) if t.elapsed() < FUND_CACHE => f.clone(),
                _ => match y.fundamentals(&s) {
                    Ok(f) => {
                        funds.insert(s.clone(), (Instant::now(), f.clone()));
                        f
                    }
                    Err(e) => {
                        st.lock().unwrap().log(format!("MATH accounts {s}: {e}"));
                        vec![]
                    }
                },
            };
            ex.funds.insert(s, f);
        }
    }
    ex
}

fn fetch(y: &Yahoo, st: &Shared, sym: &str, tf: usize) -> Option<Vec<Candle>> {
    match y.chart(sym, TIMEFRAMES[tf].0, TIMEFRAMES[tf].1) {
        Ok((c, _)) => Some(c),
        Err(e) => {
            st.lock().unwrap().log(format!("AI {sym}: {e}"));
            None
        }
    }
}

/// Kronos is loaded on first use; Err = can't run.
fn load_model<'a>(models: &'a mut HashMap<String, Kronos>, id: &str, st: &Shared) -> Result<Eng<'a>, String> {
    if id == MC_MODEL {
        return Ok(Eng::Mc);
    }
    if is_math(id) {
        return match crate::quant::find(id) {
            Some(q) => match q.run {
                crate::quant::Run::Na(why) => Err(format!("{} can't run here: {why}", q.name)),
                _ => Ok(Eng::Quant(q)),
            },
            None => Err(format!("unknown strategy {id}")),
        };
    }
    let Some((pdir, tdir, ctx)) = kronos_files(id) else {
        return Err(format!("{id} has no Rust implementation yet - pick Kronos or Black-Scholes"));
    };
    if model_status(id) != "READY" {
        return Err(format!("model files missing: put {pdir} + {tdir} in {}", data_dir().join("models").display()));
    }
    if !models.contains_key(id) {
        set_status(st, false, &format!("loading {id}..."), false);
        let m = data_dir().join("models");
        let k = Kronos::load(&m.join(pdir), &m.join(tdir), ctx).map_err(|e| format!("{id} failed to load: {e}"))?;
        models.insert(id.to_string(), k);
    }
    Ok(Eng::Kronos(&models[id]))
}

/// A leaderboard prediction waiting for its last predicted candle to close.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
struct Pending {
    model: String,
    sym: String,
    tf: usize,
    h: usize,
    made_t: i64,   // open time of the last candle the model saw
    target_t: i64, // open time of the last predicted candle
    last: f64,     // close the prediction started from
    up: bool,      // predicted direction (mean path ends higher)
}

/// Track record, saved to data_dir()/track.json: pending checks + direction hit counts.
#[derive(Serialize, Deserialize, Default)]
pub struct Track {
    pending: Vec<Pending>,
    scores: Vec<TrackScore>,
}

impl Track {
    pub fn load() -> Self {
        std::fs::read_to_string(data_dir().join("track.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }
    fn save(&self) {
        if let Ok(s) = serde_json::to_string(self) {
            let _ = std::fs::write(data_dir().join("track.json"), s);
        }
    }
    /// One record per model/symbol/setting/candle (re-runs on the same candle don't count twice).
    fn add(&mut self, p: Pending) -> bool {
        let dup = self.pending.iter().any(|q| (&q.model, &q.sym, q.tf, q.h, q.made_t) == (&p.model, &p.sym, p.tf, p.h, p.made_t));
        if !dup {
            self.pending.push(p);
        }
        !dup
    }
    /// Checks every pending prediction of `sym`/`tf` whose target candle has closed in `candles`.
    fn resolve(&mut self, sym: &str, tf: usize, candles: &[Candle]) -> bool {
        let (Some(first), Some(newest)) = (candles.first(), candles.last()) else { return false };
        let n = self.pending.len();
        let mut done = vec![];
        self.pending.retain(|p| {
            // target candle closed = a newer candle exists after it
            if p.sym != sym || p.tf != tf || p.target_t >= newest.t {
                return true;
            }
            if p.target_t >= first.t
                && let Some(c) = candles.iter().find(|c| c.t >= p.target_t)
            {
                done.push((p.model.clone(), p.h, (c.c > p.last) == p.up));
            } // else: older than the downloaded range, can't be checked -> dropped
            false
        });
        for (model, h, hit) in done {
            match self.scores.iter_mut().find(|t| t.model == model && t.sym == sym && t.tf == tf && t.h == h) {
                Some(t) => {
                    t.total += 1;
                    t.hits += hit as u32;
                }
                None => self.scores.push(TrackScore { model, sym: sym.to_string(), tf, h, hits: hit as u32, total: 1 }),
            }
        }
        self.pending.len() != n
    }
}

/// One desk's loop (math = MATH FORECAST). Both desks share `track` (one track.json).
pub fn ai_loop(st: Shared, track: Arc<Mutex<Track>>, math: bool) {
    let y = Yahoo::new();
    let mut models: HashMap<String, Kronos> = HashMap::new();
    let mut next_board = 0usize;
    let mut last: LastRun = HashMap::new();
    let mut was_unusual: HashMap<(Kind, String, String, usize, usize), bool> = HashMap::new();
    let mut streams_was: Vec<(String, usize, usize)> = vec![];
    let mut cur_model = String::new();
    let mut cache: Cache = HashMap::new();
    let mut funds = HashMap::new();
    st.lock().unwrap().track = track.lock().unwrap().scores.clone();
    let desk = if math { "MATH" } else { "AI" };
    loop {
        std::thread::sleep(Duration::from_millis(250));
        let (ai, sel, chart_tf, chart, watch, clients, off, strat_streams) = {
            let s = st.lock().unwrap();
            let ss: Vec<(String, usize, usize)> = s.book.iter().filter(|b| b.ended_ms == 0).flat_map(|b| b.models.iter().filter(|m| is_math(m) == math).map(|m| (m.clone(), b.tf, b.h))).collect();
            (if math { s.settings.math.clone() } else { s.settings.ai.clone() }, s.selected.clone(), s.candles_tf, s.candles.clone(), s.settings.watchlist.clone(), s.clients, s.session.map_or(-18_000, |x| x.gmtoffset), ss)
        };
        if !ai.enabled {
            set_status(&st, math, "OFF", true);
            streams_was.clear();
            continue;
        }
        if ai.model != cur_model {
            // the other model's one-stock numbers must not linger or block a re-run
            cur_model = ai.model.clone();
            last.retain(|k, _| k.0 != Kind::Single);
            let mut s = st.lock().unwrap();
            s.predictions.retain(|k, _| k.0 != math);
            s.bump();
        }
        if !ai.background && clients == 0 {
            set_status(&st, math, "paused - window closed and 'run in background' is OFF", false);
            continue;
        }
        // prediction streams (model, timeframe, horizon) over the tracked stocks: the leaderboard's
        // (both compared models in COMPARE mode, else the picked one) + every paper strategy's
        let (btf, bh) = (ai.board_tf.min(TIMEFRAMES.len() - 1), ai.board_horizon.max(1) as usize);
        let mut streams: Vec<(String, usize, usize)> = if ai.compare {
            vec![(ai.compare_a.clone(), btf, bh), (ai.compare_b.clone(), btf, bh)]
        } else if ai.board {
            vec![(ai.model.clone(), btf, bh)]
        } else {
            vec![]
        };
        for x in strat_streams {
            if !streams.contains(&x) {
                streams.push(x);
            }
        }
        models.retain(|id, _| *id == ai.model || streams.iter().any(|x| &x.0 == id)); // free unused model memory
        if streams != streams_was {
            // drop predictions no stream needs any more (e.g. leaderboard timeframe changed)
            let mut s = st.lock().unwrap();
            s.board.retain(|k, _| is_math(&k.0) != math || streams.iter().any(|x| (&x.0, x.1, x.2) == (&k.0, k.1, k.2)));
            s.bump();
            streams_was = streams.clone();
        }

        // 1) the charted symbol whenever its candles moved, 2) otherwise the next due leaderboard (model, symbol)
        let stf = if ai.input_tf == 0 { chart_tf } else { (ai.input_tf - 1).min(TIMEFRAMES.len() - 1) };
        let sh = ai.horizon.max(1) as usize;
        let ran = |kind: Kind, m: &str, sym: &str, tf: usize, h: usize, every: Duration| {
            last.get(&(kind, m.to_string(), sym.to_string(), tf, h)).is_some_and(|e| e.2 == tf && e.3 == h && e.4.elapsed() < every)
        };
        let single = if stf == chart_tf && !chart.is_empty() {
            let lc = chart[chart.len() - 1];
            let same = last.get(&(Kind::Single, ai.model.clone(), sel.clone(), stf, sh)).is_some_and(|e| e.0 == lc.t && e.1 == lc.c);
            (!(same && ran(Kind::Single, &ai.model, &sel, stf, sh, SINGLE_REFRESH))).then_some(chart)
        } else if !ran(Kind::Single, &ai.model, &sel, stf, sh, SINGLE_REFRESH) {
            fetch(&y, &st, &sel, stf)
        } else {
            None
        };
        let nm = streams.len();
        let job = match single {
            Some(c) => Some((Kind::Single, ai.model.clone(), sel.clone(), c, stf, sh)),
            None if nm > 0 && !watch.is_empty() => {
                // symbol-major so both compared models fill the same stock back to back
                let n = watch.len() * nm;
                let due = (0..n).map(|i| (next_board + i) % n).find(|&i| {
                    let x = &streams[i % nm];
                    !ran(Kind::Board, &x.0, &watch[i / nm], x.1, x.2, BOARD_REFRESH)
                });
                due.and_then(|i| {
                    next_board = i + 1;
                    let ((m, btf, bh), sym) = (streams[i % nm].clone(), watch[i / nm].clone());
                    let fresh = cache.get(&(sym.clone(), btf)).filter(|c| c.0.elapsed() < CANDLE_CACHE).map(|c| c.1.clone());
                    let c = fresh.or_else(|| {
                        let c = fetch(&y, &st, &sym, btf)?;
                        cache.insert((sym.clone(), btf), (Instant::now(), c.clone()));
                        let mut t = track.lock().unwrap();
                        if t.resolve(&sym, btf, &c) {
                            t.save();
                            st.lock().unwrap().track = t.scores.clone();
                        }
                        Some(c)
                    })?;
                    Some((Kind::Board, m, sym, c, btf, bh))
                })
            }
            None => None,
        };
        let Some((kind, mid, sym, candles, tf, h)) = job else { continue };
        let lc = candles.last().copied();
        let key = (kind, mid.clone(), sym.clone(), tf, h);
        let k = match load_model(&mut models, &mid, &st) {
            Ok(k) => k,
            Err(e) => {
                if kind == Kind::Single {
                    set_status(&st, math, &e, false);
                } else {
                    st.lock().unwrap().log(format!("{desk} COMPARE/LEADERBOARD: {e}"));
                }
                if let Some(lc) = lc {
                    last.insert(key, (lc.t, lc.c, tf, h, Instant::now()));
                }
                continue;
            }
        };
        let t0 = Instant::now();
        let extra = match crate::quant::find(&mid) {
            Some(q) if q.needs_universe() || q.needs_etfs() || q.needs_funds() => gather(&y, &st, &mut cache, &mut funds, q, &sym, tf, &watch),
            _ => crate::quant::Extra::default(),
        };
        let Some(mut p) = predict_one(k, &sym, &candles, tf, h, ai.lookback as usize, ai.min_conf, off, &extra) else {
            if kind == Kind::Single {
                set_status(&st, math, &format!("waiting for candles ({} of {MIN_CANDLES} needed)", candles.len()), false);
            }
            if let Some(lc) = lc {
                last.insert(key, (lc.t, lc.c, tf, h, Instant::now())); // don't hammer short histories
            }
            continue;
        };
        p.ms = t0.elapsed().as_millis() as u64;
        p.model = mid.clone();
        let lc = lc.unwrap();
        last.insert(key.clone(), (lc.t, lc.c, tf, h, Instant::now()));
        if kind == Kind::Board && p.samples > 0 && track.lock().unwrap().add(Pending { model: mid.clone(), sym: sym.clone(), tf, h, made_t: lc.t, target_t: p.steps[h - 1].t, last: lc.c, up: p.exp_pct > 0.0 }) {
            track.lock().unwrap().save();
        }

        let mut s = st.lock().unwrap();
        let gpu = if ai.gpu { " | GPU not supported yet, using CPU" } else { "" };
        *status(&mut s, math) = format!("RUNNING - last: {sym} ({mid}) in {} ms{gpu}", p.ms);
        let newly = p.unusual && !was_unusual.get(&key).copied().unwrap_or(false);
        was_unusual.insert(key, p.unusual);
        if newly && ai.notify {
            s.alert(&format!("{desk}: unusual rise predicted for {sym} ({mid}): {:+.2}% over next {h} x {}, P(up) {:.0}%", p.exp_pct, TIMEFRAMES[tf].2, p.p_up * 100.0));
        }
        match kind {
            Kind::Single => s.predictions.insert((math, sym), p),
            Kind::Board => s.board.insert((mid, tf, h, sym), p),
        };
        s.bump();
    }
}

/// One prediction from raw candles; None when there is too little history.
#[allow(clippy::too_many_arguments)]
fn predict_one(k: Eng, sym: &str, candles: &[Candle], tf: usize, h: usize, lookback: usize, min_conf: u32, off: i64, extra: &crate::quant::Extra) -> Option<Prediction> {
    let ctx = if let Eng::Kronos(k) = k { k.max_context - h } else { usize::MAX };
    let lookback = lookback.min(ctx).min(candles.len());
    if lookback < MIN_CANDLES {
        return None;
    }
    let window = &candles[candles.len() - lookback..];
    let lc = window[lookback - 1];
    let x: Vec<[f64; 6]> = window.iter().map(|c| [c.o, c.h, c.l, c.c, c.v as f64, c.v as f64 * (c.o + c.h + c.l + c.c) / 4.0]).collect();
    let xs: Vec<[usize; 5]> = window.iter().map(|c| kronos::stamp(c.t + off)).collect();
    let times = future_times(lc.t, tf, h);
    let ys: Vec<[usize; 5]> = times.iter().map(|t| kronos::stamp(t + off)).collect();
    let mut hs = std::collections::hash_map::DefaultHasher::new();
    (sym, lc.t, tf, h).hash(&mut hs);
    let closes: Vec<f64> = window.iter().map(|c| c.c).collect();
    let (samples, signal) = match k {
        Eng::Kronos(k) => (k.predict(&x, &xs, &ys, SAMPLES, hs.finish()).ok()?, String::new()),
        Eng::Mc => (monte_carlo(&closes, h, MC_PATHS, hs.finish()), String::new()),
        Eng::Quant(q) => match crate::quant::samples(q, sym, window, h, hs.finish(), extra) {
            Ok(x) => x,
            Err(why) => (vec![], why),
        },
    };
    let mut p = if samples.is_empty() { no_signal(lc.c, &times) } else { stats(&samples, &closes, &times, min_conf) };
    p.signal = signal;
    p.sym = sym.to_string();
    p.tf = tf;
    p.made_ms = now_ms();
    p.lookback = lookback;
    Some(p)
}

/// Stocks both models predicted, best agreement first: ordered by the WORSE of the two models'
/// ranks (by expected %), ties by the sum, so both must rate it high. Also returns the Spearman
/// rank correlation of the two models' expected moves (3+ shared stocks).
pub fn agreement<'a>(pa: &[&'a Prediction], pb: &[&'a Prediction]) -> (Vec<(&'a Prediction, &'a Prediction)>, Option<f64>) {
    let both: Vec<(&'a Prediction, &'a Prediction)> = pa.iter().filter_map(|x| pb.iter().find(|y| y.sym == x.sym).map(|y| (*x, *y))).collect();
    let rank = |side: fn(&(&'a Prediction, &'a Prediction)) -> f64| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..both.len()).collect();
        idx.sort_by(|&i, &j| side(&both[j]).total_cmp(&side(&both[i])));
        let mut r = vec![0; both.len()];
        for (k, i) in idx.into_iter().enumerate() {
            r[i] = k;
        }
        r
    };
    let (ra, rb) = (rank(|x| x.0.exp_pct), rank(|x| x.1.exp_pct));
    let mut order: Vec<usize> = (0..both.len()).collect();
    order.sort_by_key(|&i| (ra[i].max(rb[i]), ra[i] + rb[i]));
    let n = both.len() as f64;
    let rho = (both.len() >= 3).then(|| 1.0 - 6.0 * (0..both.len()).map(|i| (ra[i] as f64 - rb[i] as f64).powi(2)).sum::<f64>() / (n * (n * n - 1.0)));
    (order.into_iter().map(|i| both[i]).collect(), rho)
}

/// Black-Scholes Monte Carlo: `paths` geometric-Brownian-motion price paths, h candles each.
/// Drift + volatility = mean / std of the lookback's 1-candle log returns (historical drift, not the
/// risk-free rate: we forecast the stock, not price an option). 4 sub-steps per candle give each
/// simulated candle a high/low. Same output shape as Kronos samples: [o, h, l, c, vol, amount].
pub fn monte_carlo(closes: &[f64], h: usize, paths: usize, seed: u64) -> Vec<Vec<[f64; 6]>> {
    const SUB: usize = 4;
    let rets: Vec<f64> = closes.windows(2).filter(|w| w[0] > 0.0 && w[1] > 0.0).map(|w| (w[1] / w[0]).ln()).collect();
    let n = rets.len().max(1) as f64;
    let m = rets.iter().sum::<f64>() / n;
    let sd = (rets.iter().map(|r| (r - m).powi(2)).sum::<f64>() / n).sqrt();
    let (m, sd) = (m / SUB as f64, sd / (SUB as f64).sqrt());
    let mut rng = kronos::Rng::new(seed);
    let mut normal = || {
        // Box-Muller; 1 - u keeps ln() away from 0
        let (u1, u2) = (1.0 - rng.unit() as f64, rng.unit() as f64);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let last = *closes.last().unwrap_or(&0.0);
    (0..paths)
        .map(|_| {
            let mut s = last;
            (0..h)
                .map(|_| {
                    let (o, mut hi, mut lo) = (s, s, s);
                    for _ in 0..SUB {
                        s *= (m + sd * normal()).exp();
                        hi = hi.max(s);
                        lo = lo.min(s);
                    }
                    [o, hi, lo, s, 0.0, 0.0]
                })
                .collect()
        })
        .collect()
}

/// Timestamps of the next `n` candles. Daily skips weekends; intraday ignores session gaps (approximate).
fn future_times(last: i64, tf: usize, n: usize) -> Vec<i64> {
    let step = tf_secs(tf);
    let mut t = last;
    (0..n)
        .map(|_| {
            t += step;
            if step == 86_400 {
                // 1970-01-01 was a Thursday: day index 2 = Saturday, 3 = Sunday
                while matches!((t.div_euclid(86_400) % 7) as u8, 2 | 3) {
                    t += 86_400;
                }
            }
            t
        })
        .collect()
}

/// A MATH strategy with no forecast right now: flat path, 0 samples, P(up) 0 (never traded or flagged).
fn no_signal(last: f64, times: &[i64]) -> Prediction {
    let steps = times.iter().map(|&t| PredStep { t, o: last, h: last, l: last, c: last, c_lo: last, c_hi: last, p_up: 0.0 }).collect();
    Prediction { last_close: last, steps, ..Default::default() }
}

/// Turns sampled futures into the numbers shown in the forecast tabs.
pub fn stats(samples: &[Vec<[f64; 6]>], closes: &[f64], times: &[i64], min_conf: u32) -> Prediction {
    let last = *closes.last().unwrap();
    let n = samples.len() as f64;
    let h = times.len();
    let steps: Vec<PredStep> = (0..h)
        .map(|i| {
            let col = |c: usize| samples.iter().map(|s| s[i][c]).sum::<f64>() / n;
            let cl = samples.iter().map(|s| s[i][3]);
            PredStep {
                t: times[i],
                o: col(0),
                h: col(1),
                l: col(2),
                c: col(3),
                c_lo: cl.clone().fold(f64::MAX, f64::min),
                c_hi: cl.clone().fold(f64::MIN, f64::max),
                // Laplace's rule of succession: 16/16 samples up is ~94%, not certainty
                p_up: (cl.filter(|c| *c > last).count() as f64 + 1.0) / (n + 2.0),
            }
        })
        .collect();
    let mut finals: Vec<f64> = samples.iter().map(|s| s[h - 1][3]).collect();
    finals.sort_by(f64::total_cmp);
    let pct = |v: f64| (v / last - 1.0) * 100.0;
    let q = |f: f64| finals[((finals.len() - 1) as f64 * f).round() as usize];
    let fin = steps[h - 1].c;
    // typical move over h candles: std of 1-candle log returns, scaled by sqrt(h)
    let rets: Vec<f64> = closes.windows(2).filter(|w| w[0] > 0.0 && w[1] > 0.0).map(|w| (w[1] / w[0]).ln()).collect();
    let mu = rets.iter().sum::<f64>() / rets.len().max(1) as f64;
    let sd = (rets.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / rets.len().max(1) as f64).sqrt() * (h as f64).sqrt();
    let z = if sd > 0.0 && fin > 0.0 { (fin / last).ln() / sd } else { 0.0 };
    let p_up = steps[h - 1].p_up;
    Prediction {
        exp_pct: pct(fin),
        p_up,
        p10_pct: pct(q(0.1)),
        p90_pct: pct(q(0.9)),
        sigma_pct: (sd.exp() - 1.0) * 100.0,
        z,
        unusual: z >= UNUSUAL_Z && p_up * 100.0 >= min_conf as f64,
        samples: samples.len(),
        last_close: last,
        steps,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(c: f64) -> Vec<[f64; 6]> {
        vec![[c, c, c, c, 0.0, 0.0]; 2]
    }

    #[test]
    fn track_checks_direction_once_target_candle_closed() {
        let c = |t: i64, c: f64| Candle { t, o: c, h: c, l: c, c, v: 0 };
        let mut tr = Track::default();
        let p = |m: &str, up: bool| Pending { model: m.into(), sym: "X".into(), tf: 3, h: 2, made_t: 100, target_t: 300, last: 10.0, up };
        assert!(tr.add(p("a", true)));
        assert!(!tr.add(p("a", true))); // same candle again: not counted twice
        assert!(tr.add(p("b", false)));
        // target candle (t=300) is still the newest -> not closed yet
        assert!(!tr.resolve("X", 3, &[c(100, 10.0), c(200, 10.5), c(300, 11.0)]));
        assert!(tr.resolve("X", 3, &[c(100, 10.0), c(200, 10.5), c(300, 11.0), c(400, 9.0)]));
        assert!(tr.pending.is_empty());
        let s = |m: &str| tr.scores.iter().find(|t| t.model == m).map(|t| (t.hits, t.total));
        assert_eq!((s("a"), s("b")), (Some((1, 1)), Some((0, 1)))); // rose to 11: up right, down wrong
    }

    #[test]
    fn monte_carlo_matches_gbm() {
        // +0.1% drift per candle with +-1% zigzag -> after 10 candles mean log move ~ +1%, spread ~ 1% * sqrt(10)
        let closes: Vec<f64> = (0..400).scan(100.0, |p, i| { *p *= 1.001 * if i % 2 == 0 { 1.01 } else { 1.0 / 1.01 }; Some(*p) }).collect();
        let s = monte_carlo(&closes, 10, 20_000, 7);
        assert_eq!((s.len(), s[0].len()), (20_000, 10));
        let last = closes[399];
        let lr: Vec<f64> = s.iter().map(|p| (p[9][3] / last).ln()).collect();
        let mean = lr.iter().sum::<f64>() / lr.len() as f64;
        let sd = (lr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lr.len() as f64).sqrt();
        assert!((mean - 0.01).abs() < 0.002, "mean {mean}");
        assert!((sd - 0.01 * 10f64.sqrt()).abs() < 0.002, "sd {sd}");
        assert!(s.iter().all(|p| p.iter().all(|c| c[1] >= c[0].max(c[3]) && c[2] <= c[0].min(c[3]))));
    }

    #[test]
    fn flags_only_big_confident_rises() {
        let closes: Vec<f64> = (0..50).map(|i| 100.0 + if i % 2 == 0 { 0.1 } else { -0.1 }).collect(); // ~0.2% typical move
        let t = [1, 2];
        // every sample +3%: way above 2 sigma, P(up) 17/18 (never a flat 100%)
        let p = stats(&vec![path(102.9); 16], &closes, &t, 70);
        assert!(p.unusual && p.z > UNUSUAL_Z && p.p_up == 17.0 / 18.0);
        // tiny rise: normal
        assert!(!stats(&vec![path(99.95); 16], &closes, &t, 70).unusual);
        // big mean rise but only half the samples up: below confidence
        let mixed: Vec<_> = (0..16).map(|i| path(if i < 8 { 110.0 } else { 99.0 })).collect();
        assert!(!stats(&mixed, &closes, &t, 70).unusual);
    }

    #[test]
    fn daily_steps_skip_weekends() {
        let fri = 1_794_528_000; // 2026-11-13 00:00 UTC (Friday)
        let t = future_times(fri, 4, 2);
        assert_eq!((t[0] - fri) / 86_400, 3); // -> Monday
        assert_eq!((t[1] - fri) / 86_400, 4);
    }
}

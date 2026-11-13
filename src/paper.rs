//! Paper trading book: strategies trading fake money, kept by the engine and saved to
//! data_dir()/paper.json so positions survive closing the app. Money is integer cents,
//! share counts are integer micro-shares (fractional shares allowed).
use crate::ai::agreement;
use crate::shared::*;
use std::collections::HashMap;

/// Micro-shares per share.
pub const SHARE: i64 = 1_000_000;

/// Predictions by (model, timeframe, horizon, symbol), as the engine's leaderboard keeps them.
pub type Preds = HashMap<(String, usize, usize, String), Prediction>;

pub fn load() -> Vec<Strategy> {
    std::fs::read_to_string(data_dir().join("paper.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

pub fn save(book: &[Strategy]) {
    // write + rename so a crash mid-write can't wipe the book
    let (p, tmp) = (data_dir().join("paper.json"), data_dir().join("paper.json.tmp"));
    if let Ok(s) = serde_json::to_string(book)
        && std::fs::write(&tmp, s).is_ok()
    {
        let _ = std::fs::rename(tmp, p);
    }
}

/// Quote price in cents.
pub fn cents(q: &Quote) -> i64 {
    (q.price * 100.0).round() as i64
}

/// History sampling for trades with no planned length: a point is recorded when the time since the
/// last one reaches clamp(age / 60, 10 s, 12 h), age = time since the strategy's first trade.
/// The gap grows with age, so points are log-spaced: every 10 s for a few-minute trade, about
/// every 12 h after a month; ~60 points per e-fold of age, ~1000 for a month-long trade.
pub fn record_gap_ms(age_ms: i64) -> i64 {
    (age_ms / 60).clamp(10_000, 43_200_000)
}

/// Cash + positions at current prices (a position without a quote counts at its cost).
pub fn value(s: &Strategy, q: &HashMap<String, Quote>) -> i64 {
    s.cash + s.positions.iter().map(|p| q.get(&p.sym).map_or(p.cost, |q| p.micro * cents(q) / SHARE)).sum::<i64>()
}

/// Risk limits from settings: max % of the account in one stock, daily loss that stops buying.
#[derive(Clone, Copy)]
pub struct Limits {
    pub max_pos_pct: i64,
    pub day_stop_pct: i64,
}

/// A new strategy with the starting paper cash.
pub fn new_strategy(id: u64, group: u64, name: String, kind: &str, models: Vec<String>, tf: usize, h: usize, start: i64) -> Strategy {
    Strategy { id, group, name, kind: kind.into(), mode: "PAPER".into(), models, tf, h, start, cash: start, created_ms: now_ms(), ..Default::default() }
}

/// Buys `amount` cents of `sym` at `price` cents per share.
#[allow(clippy::too_many_arguments)]
pub fn buy(s: &mut Strategy, sym: &str, amount: i64, price: i64, value_now: i64, lim: Limits, why: &str, pred: Option<&Prediction>, exit_t: i64, now: i64) -> Result<(), String> {
    if amount <= 0 || price <= 0 {
        return Err("nothing to buy (no price yet?)".into());
    }
    if amount > s.cash {
        return Err(format!("not enough cash: have {}, need {}", fmt_cents(s.cash), fmt_cents(amount)));
    }
    if s.day.1 > 0 && (s.day.1 - value_now) * 100 >= s.day.1 * lim.day_stop_pct {
        return Err(format!("daily loss stop: down {}%+ today, buying paused until tomorrow", lim.day_stop_pct));
    }
    let held = s.positions.iter().find(|p| p.sym == sym).map_or(0, |p| p.micro * price / SHARE);
    let cap = value_now * lim.max_pos_pct / 100;
    if held + amount > cap {
        return Err(format!("max {}% per stock = {} (already {} in {sym})", lim.max_pos_pct, fmt_cents(cap), fmt_cents(held)));
    }
    let micro = amount * SHARE / price;
    if micro == 0 {
        return Err("amount too small for one micro-share".into());
    }
    let (exp_pct, p_up) = pred.map_or((0.0, 0.0), |p| (p.exp_pct, p.p_up));
    match s.positions.iter_mut().find(|p| p.sym == sym) {
        Some(p) => {
            p.micro += micro;
            p.cost += amount;
            p.exit_t = p.exit_t.max(exit_t);
            p.peak = p.micro * price / SHARE; // trailing stop restarts from the enlarged position
        }
        None => s.positions.push(Position { sym: sym.into(), micro, cost: amount, opened_ms: now, exp_pct, p_up, exit_t, peak: amount }),
    }
    s.cash -= amount;
    s.trades.push(Trade { t_ms: now, sym: sym.into(), buy: true, micro, price, cents: amount, why: why.into(), exp_pct, p_up });
    Ok(())
}

/// Sells `amount` cents worth of `sym` (None = everything) at `price` cents per share.
pub fn sell(s: &mut Strategy, sym: &str, amount: Option<i64>, price: i64, why: &str, pred: Option<&Prediction>, now: i64) -> Result<(), String> {
    if price <= 0 {
        return Err("no price yet".into());
    }
    let Some(i) = s.positions.iter().position(|p| p.sym == sym) else { return Err(format!("no {sym} shares held")) };
    let p = &mut s.positions[i];
    let micro = amount.map_or(p.micro, |a| (a * SHARE / price).min(p.micro));
    if micro <= 0 {
        return Err("nothing to sell".into());
    }
    let proceeds = micro * price / SHARE;
    let cost_part = p.cost * micro / p.micro;
    p.micro -= micro;
    p.cost -= cost_part;
    if p.micro == 0 {
        s.positions.remove(i);
    }
    s.cash += proceeds;
    let (exp_pct, p_up) = pred.map_or((0.0, 0.0), |p| (p.exp_pct, p.p_up));
    s.trades.push(Trade { t_ms: now, sym: sym.into(), buy: false, micro, price, cents: proceeds, why: why.into(), exp_pct, p_up });
    Ok(())
}

/// Terminates a run: sells every position at its quote (at average cost if there is no quote),
/// records the final value and stops it for good.
pub fn end(s: &mut Strategy, q: &HashMap<String, Quote>, now: i64) {
    for p in s.positions.clone() {
        let price = q.get(&p.sym).map_or(0, cents);
        let price = if price > 0 { price } else { p.cost * SHARE / p.micro.max(1) };
        let _ = sell(s, &p.sym, None, price, "terminated", None, now);
    }
    s.history.push(HistPoint { t_ms: now, value: value(s, q), cash: s.cash });
    s.auto = Auto::Off;
    s.ended_ms = now;
    s.note = "TERMINATED".into();
}

/// The strategy's prediction for `sym`: its model's, or for a pair the first model's.
pub fn pred<'a>(s: &Strategy, preds: &'a Preds, sym: &str) -> Option<&'a Prediction> {
    preds.get(&(s.models.first()?.clone(), s.tf, s.h, sym.to_string()))
}

/// AUTO "top picks": up to 3 stocks with the highest predicted rise and P(up) >= min_conf
/// (a pair needs BOTH models to agree; ranked by the worse of the two ranks). None until each
/// model has predicted at least half of the tracked stocks, so a half-filled board can't trade.
pub fn top_picks(s: &Strategy, preds: &Preds, watch: &[String], min_conf: u32) -> Option<Vec<String>> {
    let of = |m: &String| -> Vec<&Prediction> { watch.iter().filter_map(|w| preds.get(&(m.clone(), s.tf, s.h, w.clone()))).collect() };
    let sets: Vec<Vec<&Prediction>> = s.models.iter().map(of).collect();
    if sets.is_empty() || sets.iter().any(|v| v.len() * 2 < watch.len()) {
        return None;
    }
    let ok = |p: &Prediction| p.exp_pct > 0.0 && p.p_up * 100.0 >= min_conf as f64;
    let picks = if sets.len() == 1 {
        let mut v: Vec<&Prediction> = sets[0].iter().copied().filter(|p| ok(p)).collect();
        v.sort_by(|a, b| b.exp_pct.total_cmp(&a.exp_pct));
        v.iter().take(3).map(|p| p.sym.clone()).collect()
    } else {
        agreement(&sets[0], &sets[1]).0.into_iter().filter(|(a, b)| ok(a) && ok(b)).take(3).map(|(a, _)| a.sym.clone()).collect()
    };
    Some(picks)
}

/// Volatility targeting: buy weight = median predicted move of the tracked stocks / this stock's,
/// capped at 1 (full size is already the max-position limit). 1 when off or unknown.
pub fn vol_weight(s: &Strategy, preds: &Preds, watch: &[String], sym: &str) -> f64 {
    let Some(m) = s.models.first().filter(|_| s.vol_target) else { return 1.0 };
    let sig = |x: &str| preds.get(&(m.clone(), s.tf, s.h, x.to_string())).map(|p| p.sigma_pct).filter(|v| *v > 0.0);
    let mut all: Vec<f64> = watch.iter().filter_map(|x| sig(x)).collect();
    let (Some(me), false) = (sig(sym), all.is_empty()) else { return 1.0 };
    all.sort_by(f64::total_cmp);
    (all[all.len() / 2] / me).clamp(0.1, 1.0)
}

/// One second of a strategy's life: daily-loss reference, AUTO trades, history points.
/// Returns true when anything changed (caller saves the book).
pub fn tick(s: &mut Strategy, q: &HashMap<String, Quote>, preds: &Preds, watch: &[String], min_conf: u32, lim: Limits, now: i64) -> bool {
    let mut changed = false;
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    if s.day.0 != today {
        s.day = (today, value(s, q));
        changed = true;
    }
    let mut note = |s: &mut Strategy, r: Result<(), String>| {
        match r {
            Ok(()) => changed = true,
            Err(e) if s.note != e => {
                s.note = e;
                changed = true;
            }
            Err(_) => {}
        }
    };
    // indexes (^VIX, ^GSPC...) can't be bought; AUTO only trades real symbols
    let watch: &[String] = &watch.iter().filter(|w| !w.starts_with('^')).cloned().collect::<Vec<_>>();
    // protective exits, in every AUTO mode (OFF too): stop-loss, trailing stop, max hold
    let now_s = now / 1000;
    let hold_s = s.h as i64 * tf_secs(s.tf);
    // TOP PICKS would buy a still-picked stock straight back, so MAX HOLD leaves those alone
    let keep = if s.auto == Auto::TopPicks { top_picks(s, preds, watch, min_conf).unwrap_or_default() } else { vec![] };
    for i in (0..s.positions.len()).rev() {
        let p = &s.positions[i];
        let Some(c) = q.get(&p.sym).map(cents).filter(|c| *c > 0) else { continue };
        let mv = p.micro * c / SHARE;
        let peak = p.peak.max(p.cost).max(mv);
        let exit = if p.exit_t > 0 { p.exit_t } else { p.opened_ms / 1000 + hold_s };
        // widen toward the stock's typical move so normal noise doesn't trigger a stop, but at most 2x the setting
        let noise = pred(s, preds, &p.sym).map_or(0, |x| x.sigma_pct.ceil() as i64).max(0);
        let widen = |pct: u32| if pct > 0 { (pct as i64).max(noise.min(pct as i64 * 2)) } else { 0 };
        let (stop, trail) = (widen(s.stop_pct), widen(s.trail_pct));
        let why = if stop > 0 && mv * 100 <= p.cost * (100 - stop) {
            Some((format!("STOP-LOSS: down {stop}% from cost"), true))
        } else if trail > 0 && mv * 100 <= peak * (100 - trail) {
            Some((format!("TRAILING STOP: {trail}% below its best {}", fmt_cents(peak)), true))
        } else if (s.max_hold || s.auto == Auto::Signal) && now_s >= exit && !keep.contains(&p.sym) {
            Some(("MAX HOLD: predicted horizon reached".to_string(), false))
        } else {
            None
        };
        let sym = p.sym.clone();
        s.positions[i].peak = peak;
        if let Some((why, loss)) = why {
            let r = sell(s, &sym, None, c, &why, pred(s, preds, &sym), now);
            if loss && r.is_ok() {
                // don't let AUTO buy the same falling stock straight back: wait h candles
                s.cooldown.retain(|x| x.0 != sym);
                s.cooldown.push((sym, now + hold_s * 1000));
            }
            note(s, r);
        }
    }
    if s.cooldown.iter().any(|c| c.1 <= now) {
        s.cooldown.retain(|c| c.1 > now);
        note(s, Ok(())); // = changed
    }
    let cooling = |s: &Strategy, sym: &str| s.cooldown.iter().any(|c| c.0 == sym);
    match s.auto {
        Auto::Off => {}
        Auto::Signal => {
            for sym in watch {
                if cooling(s, sym) || s.positions.iter().any(|p| &p.sym == sym) {
                    continue;
                }
                let all = s.models.iter().map(|m| preds.get(&(m.clone(), s.tf, s.h, sym.clone()))).collect::<Option<Vec<_>>>();
                // a pair only buys when BOTH models signal
                let Some(ps) = all.filter(|ps| !ps.is_empty() && ps.iter().all(|p| p.samples > 0 && p.exp_pct > 0.0 && p.p_up * 100.0 >= min_conf as f64)) else { continue };
                let Some(qq) = q.get(sym) else { continue };
                let v = value(s, q);
                let amount = (((v * lim.max_pos_pct / 100) as f64 * vol_weight(s, preds, watch, sym)) as i64).min(s.cash);
                let exit = ps[0].steps.last().map_or(now_s + hold_s, |x| x.t);
                let r = buy(s, sym, amount, cents(qq), v, lim, "AUTO: signal", Some(ps[0]), exit, now);
                note(s, r);
            }
        }
        Auto::TopPicks => {
            let picks = top_picks(s, preds, watch, min_conf);
            // say why nothing happens instead of sitting silent
            match &picks {
                None => note(s, Err("AUTO waiting: models still predicting the tracked stocks".into())),
                Some(v) if v.is_empty() && s.positions.is_empty() => note(s, Err(format!("AUTO waiting: no stock has a predicted rise with P(up) >= {min_conf}% (Min confidence)"))),
                _ => {}
            }
            if let Some(picks) = picks {
                for sym in s.positions.iter().map(|p| p.sym.clone()).filter(|p| !picks.contains(p)).collect::<Vec<_>>() {
                    if let Some(qq) = q.get(&sym) {
                        let r = sell(s, &sym, None, cents(qq), "AUTO: left top picks", pred(s, preds, &sym), now);
                        note(s, r);
                    }
                }
                let new: Vec<&String> = picks.iter().filter(|p| !s.positions.iter().any(|x| &&x.sym == p) && !cooling(s, p)).collect();
                for sym in new {
                    if let Some(qq) = q.get(sym) {
                        let exit = pred(s, preds, sym).and_then(|p| p.steps.last()).map_or(0, |x| x.t);
                        let v = value(s, q);
                        let w = vol_weight(s, preds, watch, sym);
                        let amount = ((v * lim.max_pos_pct.min(33) / 100) as f64 * w) as i64;
                        let amount = amount.min(s.cash);
                        let r = buy(s, sym, amount, cents(qq), v, lim, "AUTO: top pick", pred(s, preds, sym), exit, now);
                        note(s, r);
                    }
                }
            }
        }
        Auto::Unusual => {
            let now_s = now / 1000;
            for sym in s.positions.iter().filter(|p| p.exit_t > 0 && now_s >= p.exit_t).map(|p| p.sym.clone()).collect::<Vec<_>>() {
                if let Some(qq) = q.get(&sym) {
                    let r = sell(s, &sym, None, cents(qq), "AUTO: predicted horizon reached", pred(s, preds, &sym), now);
                    note(s, r);
                }
            }
            for sym in watch {
                let all = s.models.iter().map(|m| preds.get(&(m.clone(), s.tf, s.h, sym.clone()))).collect::<Option<Vec<_>>>();
                // a pair only buys when BOTH models flag the unusual rise
                let Some(ps) = all.filter(|ps| !ps.is_empty() && ps.iter().all(|p| p.unusual)) else { continue };
                let (Some(qq), false) = (q.get(sym), s.positions.iter().any(|p| &p.sym == sym) || cooling(s, sym)) else { continue };
                let v = value(s, q);
                let amount = (((v * lim.max_pos_pct / 100) as f64 * vol_weight(s, preds, watch, sym)) as i64).min(s.cash);
                let exit = ps[0].steps.last().map_or(0, |x| x.t);
                let r = buy(s, sym, amount, cents(qq), v, lim, "AUTO: unusual rise", Some(ps[0]), exit, now);
                note(s, r);
            }
        }
    }
    // history: one point per trade, plus log-spaced points while anything is held
    let v = value(s, q);
    let traded = s.trades.last().is_some_and(|t| s.history.last().is_none_or(|h| h.t_ms < t.t_ms));
    let due = match (s.trades.first(), s.history.last()) {
        (Some(first), Some(h)) => !s.positions.is_empty() && now - h.t_ms >= record_gap_ms(now - first.t_ms),
        _ => false,
    };
    if traded || due {
        s.history.push(HistPoint { t_ms: now, value: v, cash: s.cash });
        changed = true;
    }
    changed
}

/// "$1,234.56" / "-$12.00"
pub fn fmt_cents(c: i64) -> String {
    let d = (c.abs() / 100).to_string();
    let mut out = String::new();
    for (i, ch) in d.chars().enumerate() {
        if i > 0 && (d.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    format!("{}${out}.{:02}", if c < 0 { "-" } else { "" }, c.abs() % 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quote(sym: &str, price: f64) -> (String, Quote) {
        (sym.into(), Quote { sym: sym.into(), price, ..Default::default() })
    }
    const LIM: Limits = Limits { max_pos_pct: 5, day_stop_pct: 2 };

    #[test]
    fn buy_sell_keeps_money_exact() {
        let mut s = new_strategy(1, 1, "t".into(), "AI", vec!["m".into()], 3, 5, 10_000_000); // $100k
        buy(&mut s, "X", 500_000, 12_345, 10_000_000, LIM, "test", None, 0, 1).unwrap(); // $5,000 @ $123.45
        assert_eq!(s.cash, 9_500_000);
        assert!(buy(&mut s, "X", 1_000, 12_345, 10_000_000, LIM, "test", None, 0, 2).is_err()); // over 5%
        let q: HashMap<String, Quote> = [quote("X", 135.80)].into(); // ~+10%: 40.502227 sh x $135.80 = $5,500.20
        let v = value(&s, &q);
        assert_eq!(v, 10_050_020);
        sell(&mut s, "X", Some(275_000), 13_580, "half", None, 3).unwrap();
        assert_eq!(s.positions[0].cost, 500_000 - 500_000 * s.trades[1].micro / s.trades[0].micro);
        sell(&mut s, "X", None, 13_580, "rest", None, 4).unwrap();
        assert!(s.positions.is_empty());
        assert!((s.cash - 10_050_020).abs() <= 2, "{}", s.cash); // +$500.20, sub-cent rounding only
        assert_eq!(fmt_cents(-123_456), "-$1,234.56");
    }

    #[test]
    fn stops_and_max_hold_sell_and_cool_down() {
        let watch: Vec<String> = vec!["X".into()];
        let p = Prediction { sym: "X".into(), samples: 10, exp_pct: 2.0, p_up: 0.9, steps: vec![PredStep { t: 10_000, ..Default::default() }], ..Default::default() };
        let preds: Preds = [(("m".to_string(), 4, 5, "X".to_string()), p)].into();
        let mut s = new_strategy(1, 1, "t".into(), "AI", vec!["m".into()], 4, 5, 10_000_000);
        s.auto = Auto::Signal;
        let q = |px: f64| -> HashMap<String, Quote> { [quote("X", px)].into() };
        tick(&mut s, &q(100.0), &preds, &watch, 60, LIM, 1_000);
        assert_eq!((s.positions.len(), s.positions[0].exit_t), (1, 10_000), "SIGNAL buys, exit = predicted horizon");
        // trailing stop 5%: rises to 110 (best), falls to 104 -> sold, then no re-buy while cooling down
        s.trail_pct = 5;
        tick(&mut s, &q(110.0), &preds, &watch, 60, LIM, 2_000);
        assert_eq!(s.positions.len(), 1);
        tick(&mut s, &q(104.0), &preds, &watch, 60, LIM, 3_000);
        assert!(s.positions.is_empty() && s.trades.last().unwrap().why.starts_with("TRAILING STOP"));
        assert_eq!(s.cooldown[0].0, "X");
        tick(&mut s, &q(104.0), &preds, &watch, 60, LIM, 4_000);
        assert!(s.positions.is_empty(), "cooling down");
        // cooldown over (5 daily candles) -> buys again; stop-loss 3% sells it
        let later = 3_000 + 5 * 86_400_000 + 1;
        s.trail_pct = 0;
        s.stop_pct = 3;
        tick(&mut s, &q(104.0), &preds, &watch, 60, LIM, later);
        assert_eq!(s.positions.len(), 1);
        tick(&mut s, &q(100.0), &preds, &watch, 60, LIM, later + 1);
        assert!(s.trades.last().unwrap().why.starts_with("STOP-LOSS"));
        // max hold: a manual buy (exit 0) is sold h candles after opening, even with AUTO off
        let mut m = new_strategy(2, 2, "t".into(), "AI", vec!["m".into()], 4, 5, 10_000_000);
        m.max_hold = true;
        buy(&mut m, "X", 100_000, 10_000, 10_000_000, LIM, "manual", None, 0, 0).unwrap();
        tick(&mut m, &q(100.0), &preds, &watch, 60, LIM, 5 * 86_400_000 - 1);
        assert_eq!(m.positions.len(), 1);
        tick(&mut m, &q(100.0), &preds, &watch, 60, LIM, 5 * 86_400_000);
        assert!(m.positions.is_empty() && m.trades.last().unwrap().why.starts_with("MAX HOLD"));
    }

    #[test]
    fn stops_widen_to_typical_move_and_indexes_are_skipped() {
        let watch: Vec<String> = vec!["X".into(), "^VIX".into()];
        let pr = |sym: &str| (("m".to_string(), 4, 5, sym.to_string()), Prediction { sym: sym.into(), samples: 10, exp_pct: 2.0, p_up: 0.9, sigma_pct: 7.2, ..Default::default() });
        let preds: Preds = [pr("X"), pr("^VIX")].into();
        let mut s = new_strategy(1, 1, "t".into(), "AI", vec!["m".into()], 4, 5, 10_000_000);
        s.auto = Auto::Signal;
        s.trail_pct = 3;
        let q = |px: f64| -> HashMap<String, Quote> { [quote("X", px), quote("^VIX", 20.0)].into() };
        tick(&mut s, &q(100.0), &preds, &watch, 60, LIM, 1_000);
        assert_eq!(s.positions.len(), 1, "X bought, ^VIX not");
        tick(&mut s, &q(110.0), &preds, &watch, 60, LIM, 2_000);
        tick(&mut s, &q(104.0), &preds, &watch, 60, LIM, 3_000); // -5.5%: typical move 8%, capped at 2x3% = 6%
        assert_eq!(s.positions.len(), 1);
        tick(&mut s, &q(103.0), &preds, &watch, 60, LIM, 4_000); // -6.4%
        assert!(s.trades.last().unwrap().why.starts_with("TRAILING STOP: 6%"));
        // max hold leaves a stock alone while it is still a top pick
        let mut t = new_strategy(2, 2, "t".into(), "AI", vec!["m".into()], 4, 5, 10_000_000);
        (t.auto, t.max_hold) = (Auto::TopPicks, true);
        tick(&mut t, &q(100.0), &preds, &watch, 60, LIM, 1_000);
        tick(&mut t, &q(100.0), &preds, &watch, 60, LIM, 6 * 86_400_000);
        assert_eq!((t.positions.len(), t.trades.len()), (1, 1));
    }

    #[test]
    fn vol_target_halves_a_twice_as_wild_stock() {
        let mut s = new_strategy(1, 1, "t".into(), "AI", vec!["m".into()], 3, 5, 10_000_000);
        let p = |sym: &str, sig: f64| (("m".to_string(), 3, 5, sym.to_string()), Prediction { sym: sym.into(), sigma_pct: sig, ..Default::default() });
        let preds: Preds = [p("A", 1.0), p("B", 2.0), p("C", 2.0), p("D", 4.0)].into();
        let watch: Vec<String> = ["A", "B", "C", "D"].map(String::from).to_vec();
        assert_eq!(vol_weight(&s, &preds, &watch, "D"), 1.0); // off
        s.vol_target = true;
        assert_eq!(vol_weight(&s, &preds, &watch, "D"), 0.5);
        assert_eq!(vol_weight(&s, &preds, &watch, "A"), 1.0); // calm: capped at full size
    }

    #[test]
    fn end_sells_everything_and_freezes() {
        let mut s = new_strategy(1, 1, "t".into(), "AI", vec!["m".into()], 3, 5, 10_000_000);
        buy(&mut s, "X", 500_000, 10_000, 10_000_000, LIM, "", None, 0, 1).unwrap(); // 50 sh @ $100
        buy(&mut s, "Y", 200_000, 5_000, 10_000_000, LIM, "", None, 0, 1).unwrap(); // 40 sh @ $50, no quote later
        s.auto = Auto::TopPicks;
        let q: HashMap<String, Quote> = [quote("X", 110.0)].into();
        end(&mut s, &q, 9);
        assert!(s.positions.is_empty());
        assert_eq!(s.cash, 10_000_000 + 50_000); // X +$500, Y closed at cost
        assert_eq!((s.ended_ms, s.auto, s.history.last().unwrap().value), (9, Auto::Off, 10_050_000));
    }

    #[test]
    fn daily_stop_blocks_buys() {
        let mut s = new_strategy(1, 1, "t".into(), "AI", vec!["m".into()], 3, 5, 10_000_000);
        s.day = ("d".into(), 10_000_000);
        assert!(buy(&mut s, "X", 100, 100, 9_790_000, LIM, "", None, 0, 1).unwrap_err().contains("daily loss"));
        assert!(buy(&mut s, "X", 100, 100, 9_810_000, LIM, "", None, 0, 1).is_ok());
    }

    #[test]
    fn history_gap_is_log_spaced() {
        assert_eq!(record_gap_ms(60_000), 10_000); // 1 minute in: every 10 s
        assert_eq!(record_gap_ms(3_600_000), 60_000); // 1 hour in: every minute
        assert_eq!(record_gap_ms(30 * 86_400_000), 43_200_000); // a month in: every 12 h
    }

    #[test]
    fn auto_top_picks_trades_and_records() {
        let watch: Vec<String> = ["A", "B", "C", "D"].map(String::from).to_vec();
        let mut preds = Preds::new();
        for (i, w) in watch.iter().enumerate() {
            let p = Prediction { sym: w.clone(), exp_pct: 4.0 - i as f64, p_up: 0.9, ..Default::default() };
            preds.insert(("m".into(), 3, 5, w.clone()), p);
        }
        let q: HashMap<String, Quote> = watch.iter().map(|w| quote(w, 10.0)).collect();
        let mut s = new_strategy(1, 1, "t".into(), "AI", vec!["m".into()], 3, 5, 10_000_000);
        s.auto = Auto::TopPicks;
        assert!(tick(&mut s, &q, &preds, &watch, 70, LIM, 1_000));
        let held: Vec<&str> = s.positions.iter().map(|p| p.sym.as_str()).collect();
        assert_eq!(held, ["A", "B", "C"]);
        assert_eq!(s.history.len(), 1);
        // D jumps to the top -> C (now 4th) sold, D bought
        preds.get_mut(&("m".into(), 3, 5, "D".into())).unwrap().exp_pct = 9.0;
        tick(&mut s, &q, &preds, &watch, 70, LIM, 2_000);
        let mut held: Vec<&str> = s.positions.iter().map(|p| p.sym.as_str()).collect();
        held.sort();
        assert_eq!(held, ["A", "B", "D"]);
        // no trade, gap not reached -> no point; after 10 s -> point
        let n = s.history.len();
        tick(&mut s, &q, &preds, &watch, 70, LIM, 5_000);
        assert_eq!(s.history.len(), n);
        tick(&mut s, &q, &preds, &watch, 70, LIM, 13_000);
        assert_eq!(s.history.len(), n + 1);
    }
}

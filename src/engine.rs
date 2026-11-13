//! Background engine: pulls data, keeps it in memory + on disk, serves the viewer.
//! Keeps running when the viewer window closes; only `:stop` (or a restart) ends it.
use crate::shared::*;
use crate::yahoo::{self, Yahoo};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::{fs, thread, time::Duration};

pub struct State {
    pub settings: Settings,
    pub selected: String,
    pub quotes: HashMap<String, Quote>,
    pub candles: Vec<Candle>,
    pub tf: usize,         // timeframe the viewer asked for
    pub candles_tf: usize, // timeframe `candles` actually holds
    pub session: Option<Session>,
    pub log: VecDeque<String>,
    pub alert: Option<(i64, String)>,
    pub last_ok_ms: i64,
    pub healthy: bool,
    pub symbols: Arc<Vec<String>>,
    pub predictions: HashMap<(bool, String), Prediction>, // one-stock panel by (MATH desk?, symbol)
    pub board: crate::paper::Preds, // (model, timeframe, horizon, symbol)
    pub book: Vec<Strategy>,         // paper strategies
    pub track: Vec<TrackScore>,
    pub ai_status: String,
    pub math_status: String,
    pub clients: usize,
    pub version: u64, // bumped on every change so clients only resend when needed
}

pub type Shared = Arc<Mutex<State>>;

impl State {
    pub fn bump(&mut self) {
        self.version += 1;
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        self.log.push_back(format!("{} {}", chrono::Local::now().format("%H:%M:%S"), msg.into()));
        if self.log.len() > 200 {
            self.log.pop_front();
        }
        self.version += 1;
    }

    /// Status-line flash in the viewer + Windows notification.
    pub fn alert(&mut self, msg: &str) {
        self.log(format!("ALERT {msg}"));
        self.alert = Some((now_ms(), msg.to_string()));
        let msg = msg.to_string();
        thread::spawn(move || {
            let _ = notify_rust::Notification::new().summary("WetIronTrading").body(&msg).show();
        });
    }

    fn snapshot(&self) -> Snapshot {
        let mut order: Vec<&String> = self.settings.watchlist.iter().collect();
        if !order.contains(&&self.selected) {
            order.insert(0, &self.selected);
        }
        for p in self.book.iter().flat_map(|b| &b.positions) {
            if !order.contains(&&p.sym) {
                order.push(&p.sym);
            }
        }
        Snapshot {
            source: self.settings.data_source.clone(),
            mode: self.settings.mode.clone(),
            selected: self.selected.clone(),
            tf: self.candles_tf,
            quotes: order.iter().filter_map(|s| self.quotes.get(*s).cloned()).collect(),
            candles: self.candles.clone(),
            session: self.session,
            log: self.log.iter().cloned().collect(),
            alert: self.alert.clone(),
            predictions: self.predictions.values().cloned().collect(),
            board: self.board.values().filter(|p| self.settings.watchlist.contains(&p.sym)).cloned().collect(),
            // shortcut: whole book (incl. history) in every snapshot; send deltas if it gets slow
            strategies: self.book.clone(),
            track: self.track.clone(),
            ai_status: self.ai_status.clone(),
            math_status: self.math_status.clone(),
            last_ok_ms: self.last_ok_ms,
            healthy: self.healthy,
        }
    }
}

pub fn run() {
    let Ok(listener) = TcpListener::bind(ADDR) else { return }; // already running
    let (settings, warn) = Settings::load();
    let selected = settings.watchlist.first().cloned().unwrap_or_else(|| "AAPL".into());
    let source = settings.data_source.clone();
    let st: Shared = Arc::new(Mutex::new(State {
        settings,
        selected,
        quotes: HashMap::new(),
        candles: vec![],
        tf: 0,
        candles_tf: 0,
        session: None,
        log: VecDeque::new(),
        alert: None,
        last_ok_ms: 0,
        healthy: false,
        symbols: Arc::new(vec![]),
        predictions: HashMap::new(),
        board: HashMap::new(),
        book: crate::paper::load(),
        track: vec![],
        ai_status: "starting".into(),
        math_status: "starting".into(),
        clients: 0,
        version: 0,
    }));
    {
        let mut s = st.lock().unwrap();
        s.log(format!("engine started, source={source}"));
        if let Some(w) = warn {
            s.log(w);
        }
    }
    let track = Arc::new(Mutex::new(crate::ai::Track::load()));
    for math in [false, true] {
        let (st, track) = (st.clone(), track.clone());
        thread::spawn(move || crate::ai::ai_loop(st, track, math));
    }
    {
        let st = st.clone();
        thread::spawn(move || paper_loop(st));
    }
    if source == "yahoo" {
        let st = st.clone();
        thread::spawn(move || fetch_loop(st));
    } else {
        st.lock().unwrap().log(format!("data source '{source}' not implemented"));
    }
    for conn in listener.incoming().flatten() {
        let st = st.clone();
        thread::spawn(move || client(conn, st));
    }
}

fn client(stream: TcpStream, st: Shared) {
    let Ok(read) = stream.try_clone() else { return };
    let st2 = st.clone();
    thread::spawn(move || {
        for line in BufReader::new(read).lines().map_while(Result::ok) {
            if let Ok(cmd) = serde_json::from_str::<ToEngine>(&line) {
                handle(cmd, &st2);
            }
        }
    });
    let mut w = BufWriter::new(stream);
    let (mut sent_syms, mut sent_ver) = (0, u64::MAX);
    st.lock().unwrap().clients += 1;
    // counts open windows so the AI can pause when "run in background" is off
    struct Leave(Shared);
    impl Drop for Leave {
        fn drop(&mut self) {
            self.0.lock().unwrap().clients -= 1;
        }
    }
    let _leave = Leave(st.clone());
    loop {
        let mut out = String::new();
        {
            let s = st.lock().unwrap();
            if s.symbols.len() != sent_syms {
                sent_syms = s.symbols.len();
                out += &serde_json::to_string(&ToViewer::Symbols(s.symbols.to_vec())).unwrap();
                out.push('\n');
            }
            if s.version != sent_ver {
                sent_ver = s.version;
                out += &serde_json::to_string(&ToViewer::Snap(s.snapshot())).unwrap();
                out.push('\n');
            }
        }
        if !out.is_empty() && (w.write_all(out.as_bytes()).is_err() || w.flush().is_err()) {
            return; // viewer closed
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn handle(cmd: ToEngine, st: &Shared) {
    let mut s = st.lock().unwrap();
    match cmd {
        ToEngine::Select(sym) => {
            s.selected = sym.trim().to_uppercase();
            s.candles.clear();
            s.version += 1;
        }
        ToEngine::Timeframe(tf) => {
            s.tf = tf.min(TIMEFRAMES.len() - 1);
            s.candles.clear();
            s.version += 1;
        }
        // restart = exit; the viewer sees the drop and starts a fresh engine with the saved settings
        ToEngine::Restart | ToEngine::Stop => std::process::exit(0),
        ToEngine::TestAlert => s.alert("test alert"),
        ToEngine::Ai(ai) => {
            s.settings.ai = ai;
            s.version += 1;
        }
        ToEngine::Math(m) => {
            s.settings.math = m;
            s.version += 1;
        }
        ToEngine::NewTest { group, a, b, tf, h } => {
            if s.book.iter().any(|x| x.group == group) {
                return;
            }
            let start = s.settings.start_cash_cents;
            let kind = if is_math(&a) { "MATHS" } else { "AI" };
            let tfn = TIMEFRAMES[tf.min(TIMEFRAMES.len() - 1)].2;
            let (na, nb) = (model_label(&a), model_label(&b));
            let new = [
                crate::paper::new_strategy(group, group, format!("{na} + {nb} ({tfn} x{h})"), &format!("{kind} PAIR"), vec![a.clone(), b.clone()], tf, h, start),
                crate::paper::new_strategy(group + 1, group, format!("{na} ({tfn} x{h})"), kind, vec![a], tf, h, start),
                crate::paper::new_strategy(group + 2, group, format!("{nb} ({tfn} x{h})"), kind, vec![b], tf, h, start),
            ];
            s.book.extend(new);
            s.log(format!("TEST started: {na} vs {nb}, {tfn} x{h}, {} paper each", crate::paper::fmt_cents(start)));
            crate::paper::save(&s.book);
        }
        ToEngine::Buy { id, sym, cents } => order(&mut s, id, &sym, Some(cents), true),
        ToEngine::Sell { id, sym, cents } => order(&mut s, id, &sym, cents, false),
        ToEngine::End { id } => {
            let State { book, quotes, .. } = &mut *s;
            let Some(x) = book.iter_mut().find(|x| x.id == id && x.ended_ms == 0) else { return };
            crate::paper::end(x, quotes, now_ms());
            let msg = format!("TERMINATED {}: final value {}", x.name, crate::paper::fmt_cents(crate::paper::value(x, quotes)));
            crate::paper::save(&s.book);
            s.log(msg);
        }
        ToEngine::Delete { id } => {
            s.book.retain(|x| x.id != id || x.ended_ms == 0); // only ended runs can be deleted
            crate::paper::save(&s.book);
            s.version += 1;
        }
        ToEngine::SetRisk { id, stop_pct, trail_pct, max_hold } => {
            if let Some(x) = s.book.iter_mut().find(|x| x.id == id && x.ended_ms == 0) {
                (x.stop_pct, x.trail_pct, x.max_hold) = (stop_pct, trail_pct, max_hold);
                x.note = format!("RISK stop {stop_pct}% | trail {trail_pct}% | max hold {}", if max_hold { "ON" } else { "OFF" });
            }
            crate::paper::save(&s.book);
            s.version += 1;
        }
        ToEngine::SetSize { id, vol_target } => {
            if let Some(x) = s.book.iter_mut().find(|x| x.id == id && x.ended_ms == 0) {
                x.vol_target = vol_target;
                x.note = format!("SIZE {}", if vol_target { "VOL TARGET" } else { "EQUAL $" });
            }
            crate::paper::save(&s.book);
            s.version += 1;
        }
        ToEngine::SetAuto { id, auto } => {
            if let Some(x) = s.book.iter_mut().find(|x| x.id == id && x.ended_ms == 0) {
                x.auto = auto;
                x.note = format!("AUTO {auto:?}");
            }
            crate::paper::save(&s.book);
            s.version += 1;
        }
    }
}

/// Manual buy/sell from a strategy window at the latest quote; refusals land in the strategy's note.
fn order(s: &mut State, id: u64, sym: &str, cents: Option<i64>, buy: bool) {
    let lim = crate::paper::Limits { max_pos_pct: s.settings.max_position_pct as i64, day_stop_pct: s.settings.daily_loss_stop_pct as i64 };
    let price = s.quotes.get(sym).map_or(0, crate::paper::cents);
    let State { book, quotes, board, .. } = s;
    let Some(x) = book.iter_mut().find(|x| x.id == id) else { return };
    if x.ended_ms > 0 {
        x.note = "this run has ended - it can't trade".into();
        s.version += 1;
        return;
    }
    let v = crate::paper::value(x, quotes);
    let p = crate::paper::pred(x, board, sym).cloned();
    let now = now_ms();
    let r = if buy {
        crate::paper::buy(x, sym, cents.unwrap_or(0), price, v, lim, "manual", p.as_ref(), 0, now)
    } else {
        crate::paper::sell(x, sym, cents, price, "manual", p.as_ref(), now)
    };
    x.note = match r {
        Ok(()) => format!("{} {sym} at {}", if buy { "bought" } else { "sold" }, crate::paper::fmt_cents(price)),
        Err(e) => e,
    };
    let _ = crate::paper::tick(x, quotes, board, &[], 0, lim, now); // history point for the trade
    crate::paper::save(book);
    s.version += 1;
}

/// Once a second: AUTO trades, daily-loss reference and P/L history for every strategy.
fn paper_loop(st: Shared) {
    loop {
        thread::sleep(Duration::from_secs(1));
        let mut s = st.lock().unwrap();
        let lim = crate::paper::Limits { max_pos_pct: s.settings.max_position_pct as i64, day_stop_pct: s.settings.daily_loss_stop_pct as i64 };
        let now = now_ms();
        let State { book, quotes, board, settings, .. } = &mut *s;
        let mut changed = false;
        for x in book.iter_mut().filter(|x| x.ended_ms == 0) {
            let min_conf = if x.models.first().is_some_and(|m| is_math(m)) { settings.math.min_conf } else { settings.ai.min_conf };
            changed |= crate::paper::tick(x, quotes, board, &settings.watchlist, min_conf, lim, now);
        }
        if changed {
            crate::paper::save(book);
            s.version += 1;
        }
    }
}

fn fetch_loop(st: Shared) {
    let y = Yahoo::new();
    let syms = load_symbols(&y, &st);
    {
        let mut s = st.lock().unwrap();
        s.log(format!("{} symbols in picker", syms.len()));
        s.symbols = Arc::new(syms);
    }
    let (mut batch, mut fails) = (0usize, 0u32);
    loop {
        let (sel, tf, watch, poll) = {
            let s = st.lock().unwrap();
            let mut w = s.settings.watchlist.clone();
            for p in s.book.iter().flat_map(|b| &b.positions) {
                if !w.contains(&p.sym) {
                    w.push(p.sym.clone());
                }
            }
            (s.selected.clone(), s.tf, w, s.settings.poll_ms.max(250))
        };
        // one slot of each quote batch is kept for the charted symbol so its quote is always fresh
        let chunks: Vec<&[String]> = watch.chunks(yahoo::MAX_BATCH - 1).collect();
        let mut batch_syms: Vec<String> = chunks.get(batch % chunks.len().max(1)).map_or(vec![], |c| c.to_vec());
        if !batch_syms.contains(&sel) {
            batch_syms.push(sel.clone());
        }
        batch += 1;

        let (interval, range, _) = TIMEFRAMES[tf];
        let chart = y.chart(&sel, interval, range);
        let quotes = y.quotes(&batch_syms);
        if let Ok((c, _)) = &chart {
            save_candles(&sel, interval, c);
        }

        let mut s = st.lock().unwrap();
        let mut ok = true;
        match chart {
            Ok((c, sess)) => {
                if s.selected == sel && s.tf == tf {
                    s.candles = c;
                    s.candles_tf = tf;
                }
                s.session = sess.or(s.session);
            }
            Err(e) => {
                ok = false;
                s.log(format!("chart {sel}: {e}"));
            }
        }
        match quotes {
            Ok((qs, sess)) => {
                for q in qs {
                    s.quotes.insert(q.sym.clone(), q);
                }
                s.session = s.session.or(sess);
            }
            Err(e) => {
                ok = false;
                s.log(format!("quotes: {e}"));
            }
        }
        if ok {
            if fails >= 3 {
                s.alert("Yahoo data recovered");
            }
            fails = 0;
            s.healthy = true;
            s.last_ok_ms = now_ms();
        } else {
            fails += 1;
            if fails == 3 {
                s.healthy = false;
                s.alert("Yahoo data failing");
            }
        }
        s.version += 1;
        drop(s);

        // wait, but react at once when the viewer picks another symbol
        for _ in 0..poll / 50 {
            thread::sleep(Duration::from_millis(50));
            let s = st.lock().unwrap();
            if s.selected != sel || s.tf != tf {
                break;
            }
        }
    }
}

/// Cached for a day so restarts are instant; falls back to the extras only if Nasdaq is unreachable.
fn load_symbols(y: &Yahoo, st: &Shared) -> Vec<String> {
    let path = data_dir().join("symbols.txt");
    let fresh = fs::metadata(&path).and_then(|m| m.modified()).is_ok_and(|t| t.elapsed().is_ok_and(|e| e.as_secs() < 86_400));
    let mut list: Vec<String> = match fs::read_to_string(&path) {
        Ok(s) if fresh => s.lines().map(String::from).collect(),
        _ => match y.symbol_list() {
            Ok(v) => {
                let _ = fs::write(&path, v.join("\n"));
                v
            }
            Err(e) => {
                st.lock().unwrap().log(format!("symbol list: {e}"));
                fs::read_to_string(&path).map(|s| s.lines().map(String::from).collect()).unwrap_or_default()
            }
        },
    };
    list.splice(0..0, yahoo::EXTRA.iter().map(|s| s.to_string()));
    list
}

/// Keeps every day's candles: data/yahoo/<date>/<SYM>_<interval>.json
fn save_candles(sym: &str, interval: &str, c: &[Candle]) {
    if c.is_empty() {
        return;
    }
    let dir = data_dir().join("yahoo").join(chrono::Utc::now().format("%Y-%m-%d").to_string());
    let _ = fs::create_dir_all(&dir);
    let _ = fs::write(dir.join(format!("{}_{interval}.json", sym.replace(['^', '='], "_"))), serde_json::to_string(c).unwrap());
}

//! The green window. Connects to the engine (starting it if needed); closing it leaves the engine running.
use crate::shared::*;
use crate::ui;
use ratatui::crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::crossterm::{execute, terminal::SetTitle};
use ratatui::layout::Rect;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};
use std::{fs, thread};

pub const TABS: [&str; 12] = ["MAIN", "CURRENT SHARES", "HISTORY", "SCAN", "STRAT", "AI FORECAST", "MATH FORECAST", "BACKTEST", "DATA", "STORAGE", "MATHS", "SETTINGS"];
/// Same tabs, shorter, for narrow windows.
pub const TABS_SHORT: [&str; 12] = ["MAIN", "SHARES", "HISTORY", "SCAN", "STRAT", "AI FC", "MATH FC", "BACKTEST", "DATA", "STORAGE", "MATHS", "SETTINGS"];
pub const T_MAIN: usize = 0;
pub const T_SHARES: usize = 1;
pub const T_HISTORY: usize = 2;
pub const T_SCAN: usize = 3;
pub const T_STRAT: usize = 4;
pub const T_AI: usize = 5;
pub const T_MATHF: usize = 6;
pub const T_BACKTEST: usize = 7;
pub const T_DATA: usize = 8;
pub const T_STORAGE: usize = 9;
pub const T_MATHS: usize = 10;
pub const T_SETTINGS: usize = 11;
/// Strategy window tabs.
pub const STRAT_TABS: [&str; 2] = ["BUY/SELL", "HISTORY"];
/// Order amount choices in the strategy window, cents ($100 .. $25,000).
pub const AMOUNTS: [i64; 8] = [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000];
pub const AI_ROWS: usize = 8; // candlestick prediction model settings rows
pub const HORIZONS: [u32; 5] = [1, 3, 5, 10, 20]; // "next N candles" choices
pub const SETTING_ROWS: usize = 8;

#[derive(Clone)]
pub enum Click {
    Tab(usize),
    Pick(String),
    PickScroll(i64),
    Focus(usize),
    Row(usize),
    Sym(String),
    Chart,
    /// clickable value in an AI panel's top line: 0 one-stock timeframe, 1 one-stock horizon,
    /// 2 leaderboard on/off, 3 leaderboard timeframe, 4 leaderboard horizon, 5 COMPARE on/off,
    /// 6 compared model A, 7 compared model B; strategy window: 10 AUTO, 11 amount down,
    /// 12 amount up, 13 BUY, 14 SELL, 15 SELL ALL, 16 TERMINATE, 17 SIZE, 18 STOP-LOSS,
    /// 19 TRAILING STOP, 20 MAX HOLD
    Ctl(u8),
    /// FORECAST tab TEST button
    Test,
    /// a strategy: main window opens its window, a strategy window switches to it
    Strat(u64),
    /// a P/L history point in the strategy window
    Hist(usize),
    /// TERMINATE a running strategy (click twice)
    End(u64),
    /// DELETE an ended run from HISTORY (click twice)
    Del(u64),
}

#[derive(PartialEq, Clone, Copy)]
pub enum Conn {
    Connecting,
    Up,
    Stopped,
}

#[derive(Default)]
pub struct Storage {
    pub disks: Vec<(String, u64, u64)>, // mount, total, available
    pub data: Vec<(String, u64)>,
    pub total: u64,
    pub first: Option<(Instant, u64)>,
    pub at: Option<Instant>,
}

pub struct App {
    pub settings: Settings,
    pub tab: usize,
    pub symbols: Vec<String>,
    pub snap: Snapshot,
    pub cmd: Option<String>,
    pub picker_off: usize,
    pub focus: usize,
    pub maxed: bool,
    pub help: bool,
    pub mouse: (u16, u16),
    pub flashes: HashMap<String, (f64, Option<Instant>)>,
    pub status_flash: Option<(String, Instant)>,
    pub sel_row: usize,
    pub tf: usize, // requested chart timeframe (index into TIMEFRAMES)
    pub hits: Vec<(Rect, Click)>,
    pub conn: Conn,
    pub started: Instant,
    pub storage: Storage,
    pub maths: Result<Vec<MathEntry>, String>,
    /// Some(id) = this is a strategy window, not the main app
    pub strat: Option<u64>,
    pub sym_sel: String, // strategy window: picked stock
    pub amount: i64,     // strategy window: order size, cents
    pub hist_sel: usize, // strategy window: picked history point (0 = newest)
    pub confirm: Option<(u64, Instant)>, // TERMINATE / DELETE armed by a first click
    quit: bool,
    writer: Arc<Mutex<Option<TcpStream>>>,
    stopped: Arc<AtomicBool>,
}

enum Msg {
    Up,
    Down,
    Data(ToViewer),
}

/// Re-opens the app in its own Windows Terminal profile (VT323 font, black/green) via a WT fragment,
/// which leaves the user's own terminal settings untouched. False = no WT, run in the current window.
fn open_in_own_window() -> bool {
    let (Ok(exe), Some(local)) = (std::env::current_exe(), std::env::var_os("LOCALAPPDATA")) else { return false };
    let dir = std::path::PathBuf::from(local).join("Microsoft/Windows Terminal/Fragments/WetIronTrading");
    let (r, g, b) = ui::GREEN;
    let green = format!("#{r:02X}{g:02X}{b:02X}");
    let profile = serde_json::json!({ "profiles": [{
        "guid": "{6f1d6b0e-8f8a-4c43-9a57-3c1f7e2a9b11}",
        "name": "WetIronTrading",
        "commandline": format!("\"{}\" ui", exe.display()),
        "font": { "face": "VT323", "size": 14 },
        "background": "#000000",
        "foreground": green,
        "cursorColor": green,
    }]});
    if fs::create_dir_all(&dir).and_then(|_| fs::write(dir.join("profile.json"), profile.to_string())).is_err() {
        return false;
    }
    Command::new("wt.exe").args(["-w", "new", "--maximized", "new-tab", "-p", "WetIronTrading"]).spawn().is_ok()
}

/// A strategy's own window: same WT profile (font/colours), running `wetiron strat <id>`.
fn open_strategy_window(id: u64) -> bool {
    let Ok(exe) = std::env::current_exe() else { return false };
    let exe = exe.display().to_string();
    let id = id.to_string();
    Command::new("wt.exe").args(["-w", "new", "--maximized", "new-tab", "-p", "WetIronTrading", "--title", "WetIronTrading STRATEGY", &exe, "strat", &id]).spawn().is_ok()
}

pub fn run() -> std::io::Result<()> {
    // `ui` = already inside our own window, `strat <id>` = a strategy window;
    // anything else (double-click, cargo run) hops into our own window
    let args: Vec<String> = std::env::args().collect();
    let strat = (args.get(1).map(String::as_str) == Some("strat")).then(|| args.get(2).and_then(|s| s.parse().ok())).flatten();
    if args.get(1).map(String::as_str) != Some("ui") && strat.is_none() && open_in_own_window() {
        return Ok(());
    }
    let (settings, warn) = Settings::load();
    let (tx, rx) = mpsc::channel();
    let mut app = App::new(settings);
    app.strat = strat;
    {
        let (w, s) = (app.writer.clone(), app.stopped.clone());
        thread::spawn(move || connect_loop(tx, w, s));
    }
    if let Some(w) = warn {
        app.flash(w);
    }
    let mut term = ratatui::init();
    execute!(std::io::stdout(), EnableMouseCapture, SetTitle(if strat.is_some() { "WetIronTrading STRATEGY" } else { "WetIronTrading" }))?;
    let res = (|| {
        while !app.quit {
            while let Ok(m) = rx.try_recv() {
                app.on_msg(m);
            }
            if app.tab == T_STORAGE {
                app.refresh_storage();
            }
            term.draw(|f| app.hits = ui::draw(f, &app))?;
            // ~25 fps; only the viewer waits here, never the engine
            if event::poll(Duration::from_millis(40))? {
                match event::read()? {
                    Event::Key(k) if k.kind == KeyEventKind::Press => app.on_key(k),
                    Event::Mouse(m) => app.on_mouse(m),
                    _ => {}
                }
            }
        }
        Ok(())
    })();
    let _ = execute!(std::io::stdout(), DisableMouseCapture);
    ratatui::restore();
    res
}

impl App {
    fn new(settings: Settings) -> Self {
        App {
        settings,
        tab: T_MAIN,
        symbols: vec![],
        snap: Snapshot::default(),
        cmd: None,
        picker_off: 0,
        focus: 0,
        maxed: false,
        help: false,
        mouse: (0, 0),
        flashes: HashMap::new(),
        status_flash: None,
        sel_row: 0,
        tf: 0,
        hits: vec![],
        conn: Conn::Connecting,
        started: Instant::now(),
        storage: Storage::default(),
        maths: Ok(vec![]),
        strat: None,
        sym_sel: String::new(),
        amount: 100_000,
        hist_sel: 0,
        confirm: None,
        quit: false,
        writer: Arc::new(Mutex::new(None)),
        stopped: Arc::new(AtomicBool::new(false)),
        }
    }
}

fn connect_loop(tx: mpsc::Sender<Msg>, writer: Arc<Mutex<Option<TcpStream>>>, stopped: Arc<AtomicBool>) {
    let addr = ADDR.parse().unwrap();
    let mut last_spawn: Option<Instant> = None;
    loop {
        if stopped.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(200));
            continue;
        }
        // short timeout: Windows otherwise retries a refused localhost connect for ~2s
        match TcpStream::connect_timeout(&addr, Duration::from_millis(300)) {
            Ok(s) => {
                let _ = s.set_nodelay(true);
                *writer.lock().unwrap() = s.try_clone().ok();
                if tx.send(Msg::Up).is_err() {
                    return;
                }
                for line in BufReader::new(s).lines().map_while(Result::ok) {
                    if let Ok(m) = serde_json::from_str(&line) {
                        if tx.send(Msg::Data(m)).is_err() {
                            return;
                        }
                    }
                }
                *writer.lock().unwrap() = None;
                if tx.send(Msg::Down).is_err() {
                    return;
                }
            }
            Err(_) => {
                if last_spawn.is_none_or(|t| t.elapsed() > Duration::from_secs(5)) {
                    spawn_engine();
                    last_spawn = Some(Instant::now());
                }
                thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

/// Starts this same exe as `engine`, detached so it outlives the window.
fn spawn_engine() {
    let Ok(exe) = std::env::current_exe() else { return };
    let cmd = || {
        let mut c = Command::new(&exe);
        c.arg("engine").stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        c
    };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED: u32 = 0x8 | 0x200; // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        const BREAKAWAY: u32 = 0x0100_0000; // leave the terminal's job so closing the window can't kill it
        if cmd().creation_flags(DETACHED | BREAKAWAY).spawn().is_ok() {
            return;
        }
        let _ = cmd().creation_flags(DETACHED).spawn(); // breakaway not allowed by the job
    }
    #[cfg(not(windows))]
    let _ = cmd().spawn();
}

/// Candlestick prediction model rows: (label, value). Enter/click cycles the value.
/// (Timeframe / horizon live in the PREDICTION OUTPUT and LEADERBOARD panels' top lines.)
pub fn ai_row(a: &AiSettings, i: usize) -> (&'static str, String) {
    let on = |b: bool| if b { "ON" } else { "OFF" }.to_string();
    match i {
        0 => ("Prediction model", on(a.enabled)),
        1 => {
            // the model only sees `max context` candles in total, including the predicted ones
            let cap = kronos_files(&a.model).map(|k| k.2.saturating_sub(a.horizon.max(a.board_horizon) as usize));
            match cap {
                Some(c) if (a.lookback as usize) > c => ("Lookback", format!("{} candles (model max {c})", a.lookback)),
                _ => ("Lookback", format!("{} candles", a.lookback)),
            }
        }
        2 => ("Show on chart", on(a.overlay)),
        3 => ("Show confidence", on(a.show_conf)),
        4 => ("Min confidence", format!("{}%", a.min_conf)),
        5 => ("Run in background", on(a.background)),
        6 => ("Notify strong signal", on(a.notify)),
        _ => ("Compute", if a.gpu { "GPU" } else { "CPU" }.into()),
    }
}

/// Next value after `v` in `opts`, wrapping.
fn cycle(v: u32, opts: &[u32]) -> u32 {
    opts[(opts.iter().position(|o| *o == v).map_or(0, |i| i + 1)) % opts.len()]
}

pub fn setting_row(s: &Settings, i: usize) -> (&'static str, String) {
    let on = |b: bool| if b { "ON" } else { "OFF" }.to_string();
    match i {
        0 => ("Mode", s.mode.clone()),
        1 => ("Motion", if s.lively { "LIVELY" } else { "STATIC" }.into()),
        2 => ("Ticker tapes", on(s.tapes)),
        3 => ("Flash on change", on(s.flash)),
        4 => ("Live candle", on(s.live_candle)),
        5 => ("Trade pre-market", on(s.trade_pre)),
        6 => ("Trade regular hours", on(s.trade_regular)),
        _ => ("Trade after-hours", on(s.trade_post)),
    }
}

impl App {
    pub fn flash(&mut self, msg: impl Into<String>) {
        self.status_flash = Some((msg.into(), Instant::now()));
    }

    /// An unusually high rise is currently predicted by that desk -> its FORECAST tab flashes.
    pub fn ai_alert(&self, math: bool) -> bool {
        let d = if math { &self.settings.math } else { &self.settings.ai };
        d.enabled && self.snap.predictions.iter().chain(&self.snap.board).any(|p| p.unusual && is_math(&p.model) == math)
    }

    /// AI FORECAST or MATH FORECAST is showing (same page, different models).
    pub fn fc_tab(&self) -> bool {
        self.tab == T_AI || self.tab == T_MATHF
    }

    pub fn math(&self) -> bool {
        self.tab == T_MATHF
    }

    /// Settings of the forecast desk on screen (AI anywhere but MATH FORECAST).
    pub fn desk(&self) -> &AiSettings {
        if self.math() { &self.settings.math } else { &self.settings.ai }
    }

    /// Saves and sends the desk's settings; the engine applies them live.
    fn send_desk(&mut self) {
        self.settings.save();
        let d = self.desk().clone();
        self.send(if self.math() { ToEngine::Math(d) } else { ToEngine::Ai(d) });
    }

    /// The desk's forecast for the charted stock.
    pub fn single_pred(&self) -> Option<&Prediction> {
        let m = &self.desk().model;
        self.snap.predictions.iter().find(|p| p.sym == self.snap.selected && &p.model == m)
    }

    /// MAIN has 5 panels, the forecast tabs 6 (the leaderboard).
    pub fn panel_count(&self) -> usize {
        if self.fc_tab() { 6 } else { 5 }
    }

    /// Steps a clickable AI-panel value (see Click::Ctl); d = +1 next, -1 previous.
    fn adjust(&mut self, id: u8, d: i64) {
        if id >= 10 {
            return self.strat_ctl(id, d);
        }
        let math = self.math();
        // compared models cycle through the ones that can run right now
        let ready: Vec<&str> = desk_models(math).into_iter().map(|m| m.0).filter(|id| model_status(id) == "READY").collect();
        let a = if math { &mut self.settings.math } else { &mut self.settings.ai };
        let step = |v: usize, n: usize| (v as i64 + d).rem_euclid(n as i64) as usize;
        let hstep = |v: u32| HORIZONS[step(HORIZONS.iter().position(|h| *h == v).unwrap_or(2), HORIZONS.len())];
        match id {
            0 => a.input_tf = step(a.input_tf, TIMEFRAMES.len() + 1),
            1 => a.horizon = hstep(a.horizon),
            2 => a.board = !a.board,
            3 => a.board_tf = step(a.board_tf, TIMEFRAMES.len()),
            4 => a.board_horizon = hstep(a.board_horizon),
            5 => a.compare = !a.compare,
            _ => {
                let m = if id == 6 { &mut a.compare_a } else { &mut a.compare_b };
                if let Some(next) = ready.get(step(ready.iter().position(|r| r == m).unwrap_or(0), ready.len().max(1))) {
                    *m = next.to_string();
                }
            }
        }
        self.send_desk();
    }

    /// TEST: paper strategies for compared models A, B and the pair at the leaderboard's
    /// timeframe/horizon (re-used if this exact test exists), then the pair's window.
    fn open_test(&mut self) {
        let ai = self.desk();
        let (a, b, tf, h) = (ai.compare_a.clone(), ai.compare_b.clone(), ai.board_tf.min(TIMEFRAMES.len() - 1), ai.board_horizon.max(1) as usize);
        let found = self.snap.strategies.iter().find(|s| s.ended_ms == 0 && s.kind.ends_with("PAIR") && s.models == [a.clone(), b.clone()] && s.tf == tf && s.h == h).map(|s| s.id);
        let id = match found {
            Some(id) => id,
            None => {
                let group = now_ms() as u64 * 10; // ids group, +1, +2
                self.send(ToEngine::NewTest { group, a, b, tf, h });
                group
            }
        };
        if !open_strategy_window(id) {
            self.flash("could not open a window (Windows Terminal needed)");
        }
    }

    /// The strategy this window shows (None until the engine has sent it).
    pub fn strategy(&self) -> Option<&Strategy> {
        self.strat.and_then(|id| self.snap.strategies.iter().find(|s| s.id == id))
    }

    /// Stocks a strategy window lists: tracked ones, plus anything it still holds.
    pub fn strat_syms(&self) -> Vec<String> {
        let mut v = self.settings.watchlist.clone();
        for p in self.strategy().map_or(&[][..], |s| &s.positions) {
            if !v.contains(&p.sym) {
                v.push(p.sym.clone());
            }
        }
        v
    }

    fn step_sym(&mut self, d: i64) {
        let v = self.strat_syms();
        let i = v.iter().position(|s| *s == self.sym_sel).map_or(0, |i| (i as i64 + d).clamp(0, v.len() as i64 - 1) as usize);
        if let Some(s) = v.get(i) {
            self.sym_sel = s.clone();
        }
    }

    /// TERMINATE (del = false) or DELETE needs a second click on the same strategy within 5 s.
    pub fn armed(&self, id: u64) -> bool {
        self.confirm.is_some_and(|(c, t)| c == id && t.elapsed() < Duration::from_secs(5))
    }

    fn confirm_then(&mut self, id: u64, del: bool) {
        if self.armed(id) {
            self.confirm = None;
            self.send(if del { ToEngine::Delete { id } } else { ToEngine::End { id } });
            self.flash(if del { "run deleted" } else { "strategy terminated - everything sold, see F3 HISTORY" });
        } else {
            self.confirm = Some((id, Instant::now()));
            self.flash(if del { "click DELETE again within 5 s to delete this run for good" } else { "click TERMINATE again within 5 s: sells everything at market and ends the strategy" });
        }
    }

    fn strat_ctl(&mut self, id: u8, d: i64) {
        let Some(sid) = self.strat else { return };
        let sym = self.sym_sel.clone();
        if id != 11 && id != 12 && self.strategy().is_some_and(|s| s.ended_ms > 0) {
            return self.flash("this run has ended (see F3 HISTORY in the main window)");
        }
        match id {
            16 => self.confirm_then(sid, false),
            18..=20 => {
                let Some(s) = self.strategy() else { return };
                let (mut stop_pct, mut trail_pct, mut max_hold) = (s.stop_pct, s.trail_pct, s.max_hold);
                let pcts = [0u32, 1, 2, 3, 5, 8, 10, 15];
                let step = |v: u32| pcts[(pcts.iter().position(|x| *x == v).unwrap_or(0) as i64 + d).rem_euclid(pcts.len() as i64) as usize];
                match id {
                    18 => stop_pct = step(stop_pct),
                    19 => trail_pct = step(trail_pct),
                    _ => max_hold = !max_hold,
                }
                self.send(ToEngine::SetRisk { id: sid, stop_pct, trail_pct, max_hold });
            }
            17 => {
                let vol_target = !self.strategy().is_some_and(|s| s.vol_target);
                self.send(ToEngine::SetSize { id: sid, vol_target });
            }
            10 => {
                let order = [Auto::Off, Auto::TopPicks, Auto::Unusual, Auto::Signal];
                let cur = self.strategy().map_or(Auto::Off, |s| s.auto);
                let i = order.iter().position(|x| *x == cur).unwrap_or(0) as i64;
                let auto = order[(i + d).rem_euclid(order.len() as i64) as usize];
                self.send(ToEngine::SetAuto { id: sid, auto });
            }
            11 | 12 => {
                let dir = if id == 11 { -d.abs() } else { d.abs() };
                let i = AMOUNTS.iter().position(|x| *x == self.amount).unwrap_or(3) as i64;
                self.amount = AMOUNTS[(i + dir).clamp(0, AMOUNTS.len() as i64 - 1) as usize];
            }
            _ if sym.is_empty() => self.flash("pick a stock first"),
            13 => self.send(ToEngine::Buy { id: sid, sym, cents: self.amount }),
            14 => self.send(ToEngine::Sell { id: sid, sym, cents: Some(self.amount) }),
            _ => self.send(ToEngine::Sell { id: sid, sym, cents: None }),
        }
    }

    fn strat_key(&mut self, k: KeyEvent) {
        match k.code {
            KeyCode::F(n) if (1..=STRAT_TABS.len() as u8).contains(&n) => self.tab = n as usize - 1,
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            KeyCode::Up if self.tab == 0 => self.step_sym(-1),
            KeyCode::Down if self.tab == 0 => self.step_sym(1),
            KeyCode::Up => self.hist_sel = self.hist_sel.saturating_sub(1),
            KeyCode::Down => self.hist_sel += 1,
            KeyCode::Char('b') => self.strat_ctl(13, 1),
            KeyCode::Char('s') => self.strat_ctl(14, 1),
            KeyCode::Char('+' | '=') => self.strat_ctl(12, 1),
            KeyCode::Char('-') => self.strat_ctl(11, 1),
            _ => {}
        }
    }

    /// Picker contents: everything, or what starts with the text typed in the command bar.
    pub fn picker_items(&self) -> Vec<&String> {
        match self.cmd.as_deref().map(|c| c.trim().to_uppercase()) {
            Some(f) if !f.is_empty() => self.symbols.iter().filter(|s| s.starts_with(&f)).collect(),
            _ => self.symbols.iter().collect(),
        }
    }

    fn row_count(&self) -> usize {
        match self.tab {
            T_SHARES => ui::shares_lines(self).len(),
            T_HISTORY => ui::history_lines(self).len(),
            T_SCAN => self.snap.quotes.len(),
            T_DATA => DATA_SOURCES.len() + EXECUTION.len(),
            T_AI | T_MATHF => desk_models(self.math()).len() + AI_ROWS,
            T_SETTINGS => SETTING_ROWS,
            T_MATHS => ui::maths_lines(self).len(),
            _ => 0,
        }
    }

    fn on_msg(&mut self, m: Msg) {
        match m {
            Msg::Up => {
                self.conn = Conn::Up;
                if self.tf != 0 {
                    self.send(ToEngine::Timeframe(self.tf)); // a restarted engine starts at 1 MIN
                }
            }
            Msg::Down => self.conn = if self.stopped.load(Ordering::Relaxed) { Conn::Stopped } else { Conn::Connecting },
            Msg::Data(ToViewer::Symbols(v)) => self.symbols = v,
            Msg::Data(ToViewer::Snap(s)) => {
                for q in &s.quotes {
                    let changed = self.flashes.get(&q.sym).is_some_and(|(p, _)| *p != q.price);
                    let prev = self.flashes.get(&q.sym).and_then(|x| x.1);
                    self.flashes.insert(q.sym.clone(), (q.price, if changed { Some(Instant::now()) } else { prev }));
                }
                if let Some((t, msg)) = &s.alert {
                    let newer = self.snap.alert.as_ref().is_none_or(|(old, _)| old != t);
                    if newer && *t > now_ms() - 5000 {
                        self.flash(msg.clone());
                    }
                }
                self.snap = s;
            }
        }
    }

    fn send(&mut self, cmd: ToEngine) {
        let sent = match self.writer.lock().unwrap().as_mut() {
            Some(w) => writeln!(w, "{}", serde_json::to_string(&cmd).unwrap()).is_ok(),
            None => false,
        };
        if !sent {
            self.flash("engine not connected");
        }
    }

    fn restart_engine(&mut self) {
        self.settings.save();
        self.send(ToEngine::Restart);
        self.flash("settings saved - restarting engine");
    }

    fn select(&mut self, sym: &str) {
        self.send(ToEngine::Select(sym.trim().to_uppercase()));
        if !self.fc_tab() {
            self.tab = T_MAIN;
        }
    }

    fn set_tab(&mut self, t: usize) {
        self.tab = t;
        self.sel_row = 0;
        if t == T_MATHS {
            self.maths = load_maths();
        }
        if t == T_STORAGE {
            self.storage.at = None;
        }
    }

    fn exec(&mut self, line: &str) {
        let lc = line.trim().to_lowercase();
        match lc.as_str() {
            "" => {}
            "q" | "quit" => self.quit = true,
            "help" => self.help = true,
            "alert" => self.send(ToEngine::TestAlert),
            "stop" => {
                self.stopped.store(true, Ordering::Relaxed);
                self.send(ToEngine::Stop);
                self.flash("engine stopped - :start to start it again");
            }
            "start" => {
                self.stopped.store(false, Ordering::Relaxed);
                self.flash("starting engine");
            }
            "static" | "lively" => {
                self.settings.lively = lc == "lively";
                self.settings.save();
            }
            "mode signals" | "mode paper" => {
                self.settings.mode = if lc.ends_with("signals") { "SIGNALS" } else { "AUTO-PAPER" }.into();
                self.restart_engine();
            }
            _ if lc.starts_with("mode") => self.flash("usage: mode signals | mode paper"),
            _ if lc.contains(' ') => self.flash(format!("unknown command: {lc}")),
            _ => self.select(line),
        }
    }

    fn activate(&mut self, i: usize) {
        match self.tab {
            T_SCAN => {
                if let Some(q) = self.snap.quotes.get(i) {
                    let s = q.sym.clone();
                    self.select(&s);
                }
            }
            T_DATA => {
                let nd = DATA_SOURCES.len();
                let (entry, is_data) = match (DATA_SOURCES.get(i), i.checked_sub(nd).and_then(|j| EXECUTION.get(j))) {
                    (Some(e), _) => (e, true),
                    (None, Some(e)) => (e, false),
                    _ => return,
                };
                let (id, label, built) = *entry;
                if !built {
                    self.flash(format!("{label}: not built yet [N/A]"));
                    return;
                }
                let cur = if is_data { &mut self.settings.data_source } else { &mut self.settings.execution };
                if cur != id {
                    *cur = id.into();
                    self.restart_engine();
                }
            }
            T_AI | T_MATHF => {
                let math = self.math();
                let models = desk_models(math);
                let a = if math { &mut self.settings.math } else { &mut self.settings.ai };
                if let Some((id, label)) = models.get(i) {
                    a.model = id.to_string();
                    let msg = match (model_status(id), crate::quant::find(id).map(|q| &q.run)) {
                        ("READY", _) => format!("{label} selected"),
                        ("NOT DOWNLOADED", _) => format!("{label}: model files not downloaded"),
                        (_, Some(crate::quant::Run::Na(why))) => format!("{label}: can't run here - {why} (see MATHS tab)"),
                        _ => format!("{label}: no Rust implementation yet, nothing will run"),
                    };
                    self.flash(msg);
                } else {
                    match i - models.len() {
                        0 => a.enabled = !a.enabled,
                        1 => a.lookback = cycle(a.lookback, &[64, 128, 256, 512, 1024, 2048]),
                        2 => a.overlay = !a.overlay,
                        3 => a.show_conf = !a.show_conf,
                        4 => a.min_conf = cycle(a.min_conf, &[50, 60, 70, 80, 90]),
                        5 => a.background = !a.background,
                        6 => a.notify = !a.notify,
                        7 => a.gpu = !a.gpu,
                        _ => return,
                    }
                }
                self.send_desk(); // engine applies it live, no restart
            }
            T_SETTINGS => {
                let s = &mut self.settings;
                match i {
                    0 => {
                        s.mode = if s.mode == "SIGNALS" { "AUTO-PAPER" } else { "SIGNALS" }.into();
                        return self.restart_engine();
                    }
                    1 => s.lively = !s.lively,
                    2 => s.tapes = !s.tapes,
                    3 => s.flash = !s.flash,
                    4 => s.live_candle = !s.live_candle,
                    5 => s.trade_pre = !s.trade_pre,
                    6 => s.trade_regular = !s.trade_regular,
                    7 => s.trade_post = !s.trade_post,
                    _ => return,
                }
                s.save();
            }
            _ => {}
        }
    }

    /// MAIN and AI share the 5-panel grid (focus, maximise, chart zoom).
    fn grid_tab(&self) -> bool {
        self.tab == T_MAIN || self.fc_tab()
    }

    fn on_key(&mut self, k: KeyEvent) {
        if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return;
        }
        if self.strat.is_some() {
            return self.strat_key(k);
        }
        if self.help {
            self.help = false;
            return;
        }
        if let Some(buf) = &mut self.cmd {
            match k.code {
                KeyCode::Esc => self.cmd = None,
                KeyCode::Enter => {
                    let line = std::mem::take(buf);
                    self.cmd = None;
                    self.exec(&line);
                }
                KeyCode::Backspace => {
                    buf.pop();
                    self.picker_off = 0;
                }
                KeyCode::Char(c) => {
                    buf.push(c);
                    self.picker_off = 0;
                }
                _ => {}
            }
            return;
        }
        match k.code {
            KeyCode::Char(':') => self.cmd = Some(String::new()),
            KeyCode::Char('?') => self.help = true,
            KeyCode::F(n) if (1..=TABS.len() as u8).contains(&n) => self.set_tab(n as usize - 1),
            KeyCode::Tab if self.grid_tab() => self.focus = (self.focus + 1) % self.panel_count(),
            KeyCode::BackTab if self.grid_tab() => self.focus = (self.focus + self.panel_count() - 1) % self.panel_count(),
            KeyCode::Char(' ') if self.grid_tab() => self.maxed = !self.maxed,
            KeyCode::Esc => self.maxed = false,
            KeyCode::Char('+' | '=') if self.grid_tab() => self.zoom(-1),
            KeyCode::Char('-') if self.grid_tab() => self.zoom(1),
            KeyCode::Up => self.sel_row = self.sel_row.saturating_sub(1),
            KeyCode::Down => self.sel_row = (self.sel_row + 1).min(self.row_count().saturating_sub(1)),
            KeyCode::Enter => self.activate(self.sel_row),
            _ => {}
        }
    }

    fn on_mouse(&mut self, m: MouseEvent) {
        self.mouse = (m.column, m.row);
        let inside = |r: &Rect| m.column >= r.x && m.column < r.right() && m.row >= r.y && m.row < r.bottom();
        let hit = self.hits.iter().rev().find(|(r, _)| inside(r)).map(|(_, c)| c.clone());
        match (m.kind, hit) {
            (MouseEventKind::Down(MouseButton::Left), Some(c)) => match c {
                Click::Tab(t) => self.set_tab(t),
                Click::Sym(s) if self.strat.is_some() => self.sym_sel = s,
                Click::Pick(s) | Click::Sym(s) => self.select(&s),
                Click::PickScroll(d) => self.scroll_picker(d),
                Click::Focus(i) => self.focus = i,
                Click::Chart => self.focus = 0,
                Click::Ctl(id) => self.adjust(id, 1),
                Click::Test => self.open_test(),
                Click::Strat(id) if self.strat.is_some() => {
                    self.strat = Some(id);
                    self.hist_sel = 0;
                }
                Click::Strat(id) => {
                    if !open_strategy_window(id) {
                        self.flash("could not open a window (Windows Terminal needed)");
                    }
                }
                Click::Hist(i) => self.hist_sel = i,
                Click::End(id) => self.confirm_then(id, false),
                Click::Del(id) => self.confirm_then(id, true),
                Click::Row(i) => {
                    self.sel_row = i;
                    self.activate(i);
                }
            },
            (MouseEventKind::ScrollDown | MouseEventKind::ScrollUp, Some(c)) => {
                let d: i64 = if m.kind == MouseEventKind::ScrollDown { 1 } else { -1 };
                match c {
                    Click::Pick(_) | Click::PickScroll(_) => self.scroll_picker(d),
                    Click::Chart => self.zoom(d), // wheel up = zoom in
                    Click::Ctl(id) => self.adjust(id, d),
                    Click::Hist(_) => self.hist_sel = (self.hist_sel as i64 + d).max(0) as usize,
                    Click::Sym(_) if self.strat.is_some() => self.step_sym(d),
                    _ => self.sel_row = (self.sel_row as i64 + d).clamp(0, self.row_count().saturating_sub(1) as i64) as usize,
                }
            }
            _ => {}
        }
    }

    /// Step the chart timeframe; -1 = zoom in (smaller candles), +1 = zoom out.
    fn zoom(&mut self, d: i64) {
        let n = (self.tf as i64 + d).clamp(0, TIMEFRAMES.len() as i64 - 1) as usize;
        if n != self.tf {
            self.tf = n;
            self.send(ToEngine::Timeframe(n));
        }
    }

    fn scroll_picker(&mut self, d: i64) {
        let max = self.picker_items().len().saturating_sub(1) as i64;
        self.picker_off = (self.picker_off as i64 + d).clamp(0, max) as usize;
    }

    fn refresh_storage(&mut self) {
        if self.storage.at.is_some_and(|t| t.elapsed() < Duration::from_secs(5)) {
            return;
        }
        let disks = sysinfo::Disks::new_with_refreshed_list();
        self.storage.disks = disks.list().iter().map(|d| (d.mount_point().display().to_string(), d.total_space(), d.available_space())).collect();
        let mut data: Vec<(String, u64)> = fs::read_dir(data_dir())
            .map(|rd| rd.flatten().map(|e| (e.file_name().to_string_lossy().into_owned(), size_of(&e.path()))).collect())
            .unwrap_or_default();
        data.sort_by(|a, b| b.1.cmp(&a.1));
        self.storage.total = data.iter().map(|d| d.1).sum();
        self.storage.data = data;
        self.storage.first.get_or_insert((Instant::now(), self.storage.total));
        self.storage.at = Some(Instant::now());
    }
}

fn size_of(p: &std::path::Path) -> u64 {
    match fs::metadata(p) {
        Ok(m) if m.is_dir() => fs::read_dir(p).map(|rd| rd.flatten().map(|e| size_of(&e.path())).sum()).unwrap_or(0),
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn test_strategy() -> Strategy {
        let pos = Position { sym: "AAPL".into(), micro: 2_500_000, cost: 25_000, opened_ms: now_ms() - 3_600_000, exp_pct: 1.2, p_up: 0.7, exit_t: 0, peak: 25_000 };
        let trade = Trade { t_ms: now_ms() - 3_600_000, sym: "AAPL".into(), buy: true, micro: 2_500_000, price: 10_000, cents: 25_000, why: "manual".into(), exp_pct: 1.2, p_up: 0.7 };
        let history = (0..40).map(|i| HistPoint { t_ms: now_ms() - (40 - i) * 60_000, value: 10_000_000 + (i % 7) * 1_000 - 3_000, cash: 9_975_000 }).collect();
        Strategy { id: 1, group: 1, name: "Kronos-small + Black-Scholes (1 HOUR x5)".into(), kind: "AI PAIR".into(), mode: "PAPER".into(), models: vec!["kronos_small".into(), MC_MODEL.into()], tf: 3, h: 5, start: 10_000_000, cash: 9_975_000, positions: vec![pos], trades: vec![trade], history, note: "bought AAPL".into(), ..Default::default() }
    }

    /// Click + wheel every clickable spot on every tab (both COMPARE states), redrawing after each.
    #[test]
    fn every_click_and_scroll_survives() {
        // settings.save() writes data_dir(): keep the real settings.json out of it
        let tmp = std::env::temp_dir().join("wetiron_click_test");
        unsafe { std::env::set_var("LOCALAPPDATA", &tmp) };
        let mut a = App::new(Settings::default());
        a.symbols = vec!["AAPL".into(), "NVDA".into()];
        a.snap.selected = "AAPL".into();
        a.snap.candles = (0..300).map(|i| Candle { t: i * 60, o: 100.0, h: 101.0, l: 99.0, c: 100.5, v: 1 }).collect();
        let steps: Vec<PredStep> = (1..=5).map(|i| PredStep { t: 300 * 60 + i * 60, o: 100.0, h: 101.0, l: 99.0, c: 100.0, c_lo: 99.0, c_hi: 101.0, p_up: 0.5 }).collect();
        let (ai, m) = (a.settings.ai.clone(), a.settings.math.clone());
        a.snap.predictions = [&ai.model, &m.model].map(|id| Prediction { sym: "AAPL".into(), model: id.clone(), steps: steps.clone(), samples: 16, last_close: 100.0, ..Default::default() }).to_vec();
        a.snap.board = [&ai.model, &ai.compare_a, &ai.compare_b, &m.model, &m.compare_a, &m.compare_b].iter().flat_map(|m| ["AAPL", "NVDA"].map(|s| Prediction { sym: s.into(), model: m.to_string(), steps: steps.clone(), exp_pct: 1.0, p_up: 0.6, ..Default::default() })).collect();
        let mut ended = test_strategy();
        (ended.id, ended.ended_ms) = (7, now_ms());
        let mut risky = test_strategy();
        (risky.id, risky.stop_pct, risky.trail_pct, risky.max_hold, risky.auto, risky.cooldown) = (8, 3, 5, true, Auto::Signal, vec![("NVDA".into(), now_ms() + 60_000)]);
        a.snap.strategies = vec![test_strategy(), ended, risky];
        let mut t = Terminal::new(TestBackend::new(220, 60)).unwrap();
        // strategy windows first (plain + one with every risk setting on), then every main tab
        for (strat, tab) in [(Some(1), 0), (Some(1), 1), (Some(8), 0)] {
            a.strat = strat;
            a.tab = tab;
            t.draw(|f| a.hits = ui::draw(f, &a)).unwrap();
            let spots: Vec<(u16, u16)> = a.hits.iter().map(|(r, _)| (r.x + r.width / 2, r.y + r.height / 2)).collect();
            for (x, y) in spots {
                for kind in [MouseEventKind::Moved, MouseEventKind::Down(MouseButton::Left), MouseEventKind::ScrollDown, MouseEventKind::ScrollUp] {
                    a.strat = strat;
                    a.tab = tab;
                    a.on_mouse(MouseEvent { kind, column: x, row: y, modifiers: KeyModifiers::NONE });
                    t.draw(|f| a.hits = ui::draw(f, &a)).unwrap();
                }
            }
        }
        a.strat = None;
        for tab in 0..TABS.len() {
            for compare in [false, true] {
                a.set_tab(tab);
                a.settings.ai.compare = compare;
                a.settings.math.compare = compare;
                t.draw(|f| a.hits = ui::draw(f, &a)).unwrap();
                let spots: Vec<(u16, u16)> = a.hits.iter().map(|(r, _)| (r.x + r.width / 2, r.y + r.height / 2)).collect();
                for (x, y) in spots {
                    for kind in [MouseEventKind::Moved, MouseEventKind::Down(MouseButton::Left), MouseEventKind::ScrollDown, MouseEventKind::ScrollUp] {
                        a.tab = tab;
                        a.on_mouse(MouseEvent { kind, column: x, row: y, modifiers: KeyModifiers::NONE });
                        t.draw(|f| a.hits = ui::draw(f, &a)).unwrap();
                    }
                }
            }
        }
    }

    /// Every tab, overlay and window size must draw without panicking.
    #[test]
    fn renders_everything_at_any_size() {
        let mut a = App::new(Settings::default());
        a.symbols = vec!["AAPL".into(), "CL=F".into()];
        a.snap.selected = "AAPL".into();
        a.snap.quotes = vec![Quote { sym: "AAPL".into(), name: "Apple Inc.".into(), price: 355.6, change_pct: 2.0, prev_close: 348.5, volume: 10, time: 0 }];
        a.snap.candles = (0..300).map(|i| { let p = 100.0 + (i % 7) as f64; Candle { t: i * 60, o: p, h: p + 2.0, l: p - 2.0, c: p + if i % 2 == 0 { 1.0 } else { -1.0 }, v: 1 } }).collect();
        a.snap.log = vec!["x".into(); 50];
        a.snap.predictions = vec![Prediction { sym: "AAPL".into(), steps: (1..=5).map(|i| PredStep { t: 300 * 60 + i * 60, o: 104.0, h: 107.0, l: 103.0, c: 106.0, c_lo: 101.0, c_hi: 109.0, p_up: 0.8 }).collect(), unusual: true, samples: 16, last_close: 104.0, ..Default::default() }];
        a.snap.ai_status = "RUNNING".into();
        let ai = a.settings.ai.clone();
        let mut no_sig = a.snap.predictions[0].clone();
        (no_sig.model, no_sig.samples, no_sig.signal) = (a.settings.math.model.clone(), 0, "no signal on the latest candle".into());
        a.snap.predictions[0].model = ai.model.clone();
        a.snap.predictions.push(no_sig);
        let m = a.settings.math.clone();
        a.snap.board = [&ai.model, &ai.compare_a, &m.model, &m.compare_a].iter().flat_map(|m| ["NVDA", "CL=F", "AMD"].iter().enumerate().map(|(i, s)| Prediction { sym: s.to_string(), model: m.to_string(), tf: 3, samples: i * 8, signal: "SIGNAL NOW".into(), exp_pct: 2.0 - i as f64, p_up: 0.7, unusual: i == 0, made_ms: now_ms(), steps: a.snap.predictions[0].steps.clone(), ..Default::default() }).collect::<Vec<_>>()).collect();
        a.snap.track = vec![TrackScore { model: ai.compare_a.clone(), sym: "AMD".into(), tf: 3, h: 5, hits: 4, total: 6 }, TrackScore { model: ai.compare_b.clone(), sym: "AMD".into(), tf: 3, h: 5, hits: 5, total: 6 }];
        let mut ended = test_strategy();
        (ended.id, ended.ended_ms, ended.positions) = (7, now_ms(), vec![]);
        a.snap.strategies = vec![test_strategy(), ended];
        a.maths = load_maths();
        a.refresh_storage();
        a.flash("hello");
        let sizes: Vec<(u16, u16)> = [(220, 60), (80, 24), (40, 10), (45, 12), (30, 8), (5, 3), (1, 1)].into_iter().chain((20..=400).step_by(7).flat_map(|w| (6..=120).step_by(9).map(move |h| (w, h)))).collect();
        for (w, h) in sizes {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            for tab in 0..TABS.len() {
                for (help, maxed, tapes) in [(false, false, true), (true, true, false)] {
                    a.strat = None;
                    a.tab = tab;
                    a.help = help;
                    a.maxed = maxed;
                    a.settings.tapes = tapes;
                    a.settings.ai.compare = help; // second pass draws COMPARE mode
                    a.settings.math.compare = help;
                    a.cmd = if help { Some("A".into()) } else { None };
                    t.draw(|f| a.hits = ui::draw(f, &a)).unwrap();
                    if std::env::var("DUMP").is_ok_and(|d| d == format!("{tab}{w}")) && !help {
                        println!("{}", t.backend());
                    }
                    if tab < STRAT_TABS.len() {
                        // the same tab number in a strategy window
                        a.strat = Some(1);
                        t.draw(|f| a.hits = ui::draw(f, &a)).unwrap();
                        if std::env::var("DUMP").is_ok_and(|d| d == format!("s{tab}{w}")) && !help {
                            println!("{}", t.backend());
                        }
                    }
                }
            }
        }
    }
}

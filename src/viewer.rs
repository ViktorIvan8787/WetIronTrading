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

pub const TABS: [&str; 9] = ["MAIN", "SCAN", "STRAT", "FORECAST", "BACKTEST", "DATA", "STORAGE", "MATHS", "SETTINGS"];
pub const T_MAIN: usize = 0;
pub const T_SCAN: usize = 1;
pub const T_STRAT: usize = 2;
pub const T_AI: usize = 3;
pub const T_BACKTEST: usize = 4;
pub const T_DATA: usize = 5;
pub const T_STORAGE: usize = 6;
pub const T_MATHS: usize = 7;
pub const T_SETTINGS: usize = 8;
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
    /// 6 compared model A, 7 compared model B
    Ctl(u8),
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

pub fn run() -> std::io::Result<()> {
    // `ui` = already inside our own window; anything else (double-click, cargo run) hops into it
    if std::env::args().nth(1).as_deref() != Some("ui") && open_in_own_window() {
        return Ok(());
    }
    let (settings, warn) = Settings::load();
    let (tx, rx) = mpsc::channel();
    let mut app = App::new(settings);
    {
        let (w, s) = (app.writer.clone(), app.stopped.clone());
        thread::spawn(move || connect_loop(tx, w, s));
    }
    if let Some(w) = warn {
        app.flash(w);
    }
    let mut term = ratatui::init();
    execute!(std::io::stdout(), EnableMouseCapture, SetTitle("WetIronTrading"))?;
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

    /// An unusually high rise is currently predicted -> F4 FORECAST flashes.
    pub fn ai_alert(&self) -> bool {
        self.settings.ai.enabled && self.snap.predictions.iter().chain(&self.snap.board).any(|p| p.unusual)
    }

    /// MAIN has 5 panels, AI has 6 (the leaderboard).
    pub fn panel_count(&self) -> usize {
        if self.tab == T_AI { 6 } else { 5 }
    }

    /// Steps a clickable AI-panel value (see Click::Ctl); d = +1 next, -1 previous.
    fn adjust(&mut self, id: u8, d: i64) {
        let a = &mut self.settings.ai;
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
                // compared models cycle through the ones that can run right now
                let ready: Vec<&str> = AI_MODELS.iter().map(|m| m.0).filter(|id| model_status(id) == "READY").collect();
                let m = if id == 6 { &mut a.compare_a } else { &mut a.compare_b };
                if let Some(next) = ready.get(step(ready.iter().position(|r| r == m).unwrap_or(0), ready.len().max(1))) {
                    *m = next.to_string();
                }
            }
        }
        self.settings.save();
        let ai = self.settings.ai.clone();
        self.send(ToEngine::Ai(ai));
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
            T_SCAN => self.snap.quotes.len(),
            T_DATA => DATA_SOURCES.len() + EXECUTION.len(),
            T_AI => AI_MODELS.len() + AI_ROWS,
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
        if self.tab != T_AI {
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
            T_AI => {
                let a = &mut self.settings.ai;
                if let Some((id, label, _)) = AI_MODELS.get(i) {
                    a.model = id.to_string();
                    let msg = match model_status(id) {
                        "READY" => format!("{label} selected"),
                        "NOT DOWNLOADED" => format!("{label}: model files not downloaded"),
                        _ => format!("{label}: no Rust implementation yet, nothing will run"),
                    };
                    self.flash(msg);
                } else {
                    match i - AI_MODELS.len() {
                        0 => a.enabled = !a.enabled,
                        1 => a.lookback = cycle(a.lookback, &[64, 128, 256, 512, 1024]),
                        2 => a.overlay = !a.overlay,
                        3 => a.show_conf = !a.show_conf,
                        4 => a.min_conf = cycle(a.min_conf, &[50, 60, 70, 80, 90]),
                        5 => a.background = !a.background,
                        6 => a.notify = !a.notify,
                        7 => a.gpu = !a.gpu,
                        _ => return,
                    }
                }
                self.settings.save();
                let ai = self.settings.ai.clone();
                self.send(ToEngine::Ai(ai)); // engine applies it live, no restart
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
        self.tab == T_MAIN || self.tab == T_AI
    }

    fn on_key(&mut self, k: KeyEvent) {
        if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return;
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
                Click::Pick(s) | Click::Sym(s) => self.select(&s),
                Click::PickScroll(d) => self.scroll_picker(d),
                Click::Focus(i) => self.focus = i,
                Click::Chart => self.focus = 0,
                Click::Ctl(id) => self.adjust(id, 1),
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
        a.snap.predictions = vec![Prediction { sym: "AAPL".into(), steps: steps.clone(), samples: 16, last_close: 100.0, ..Default::default() }];
        let ai = a.settings.ai.clone();
        a.snap.board = [&ai.model, &ai.compare_a, &ai.compare_b].iter().flat_map(|m| ["AAPL", "NVDA"].map(|s| Prediction { sym: s.into(), model: m.to_string(), steps: steps.clone(), exp_pct: 1.0, p_up: 0.6, ..Default::default() })).collect();
        let mut t = Terminal::new(TestBackend::new(220, 60)).unwrap();
        for tab in 0..TABS.len() {
            for compare in [false, true] {
                a.set_tab(tab);
                a.settings.ai.compare = compare;
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
        a.snap.board = [&ai.model, &ai.compare_a].iter().flat_map(|m| ["NVDA", "CL=F", "AMD"].iter().enumerate().map(|(i, s)| Prediction { sym: s.to_string(), model: m.to_string(), tf: 3, exp_pct: 2.0 - i as f64, p_up: 0.7, unusual: i == 0, made_ms: now_ms(), steps: a.snap.predictions[0].steps.clone(), ..Default::default() }).collect::<Vec<_>>()).collect();
        a.snap.track = vec![TrackScore { model: ai.compare_a.clone(), sym: "AMD".into(), tf: 3, h: 5, hits: 4, total: 6 }, TrackScore { model: ai.compare_b.clone(), sym: "AMD".into(), tf: 3, h: 5, hits: 5, total: 6 }];
        a.maths = load_maths();
        a.refresh_storage();
        a.flash("hello");
        let sizes: Vec<(u16, u16)> = [(220, 60), (80, 24), (40, 10), (45, 12), (30, 8), (5, 3), (1, 1)].into_iter().chain((20..=400).step_by(7).flat_map(|w| (6..=120).step_by(9).map(move |h| (w, h)))).collect();
        for (w, h) in sizes {
            let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
            for tab in 0..TABS.len() {
                for (help, maxed, tapes) in [(false, false, true), (true, true, false)] {
                    a.tab = tab;
                    a.help = help;
                    a.maxed = maxed;
                    a.settings.tapes = tapes;
                    a.settings.ai.compare = help; // second pass draws COMPARE mode
                    a.cmd = if help { Some("A".into()) } else { None };
                    t.draw(|f| a.hits = ui::draw(f, &a)).unwrap();
                    if std::env::var("DUMP").is_ok_and(|d| d == format!("{tab}{w}")) && !help {
                        println!("{}", t.backend());
                    }
                }
            }
        }
    }
}

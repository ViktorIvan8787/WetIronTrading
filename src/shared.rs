//! Types and files shared by the engine and the viewer.
use serde::{Deserialize, Serialize};
use std::{env, fs, path::PathBuf};

/// Engine <-> viewer channel. Localhost only.
pub const ADDR: &str = "127.0.0.1:47821";

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Quote {
    pub sym: String,
    #[serde(default)]
    pub name: String,
    pub price: f64, // display only; accounting will use integer cents
    pub change_pct: f64, // straight from the source, not computed by us
    pub prev_close: f64,
    pub volume: u64,
    pub time: i64,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Candle {
    pub t: i64,
    pub o: f64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    pub v: u64,
}

/// Unix-second windows for today's sessions, from the data source.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Session {
    pub pre: (i64, i64),
    pub regular: (i64, i64),
    pub post: (i64, i64),
    pub gmtoffset: i64,
}

/// Chart zoom levels, smallest first: (Yahoo interval, Yahoo range, axis label).
pub const TIMEFRAMES: &[(&str, &str, &str)] = &[
    ("1m", "1d", "1 MIN"),
    ("5m", "5d", "5 MIN"),
    ("15m", "1mo", "15 MIN"),
    ("60m", "3mo", "1 HOUR"),
    ("1d", "1y", "1 DAY"),
    ("1wk", "5y", "1 WEEK"),
    ("1mo", "max", "1 MONTH"),
];

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Snapshot {
    pub source: String,
    pub mode: String,
    pub selected: String,
    pub tf: usize, // index into TIMEFRAMES of the candles below
    pub quotes: Vec<Quote>,
    pub candles: Vec<Candle>,
    pub session: Option<Session>,
    pub log: Vec<String>,
    pub alert: Option<(i64, String)>,
    pub predictions: Vec<Prediction>, // one-stock panel
    pub board: Vec<Prediction>,       // leaderboard, tracked symbols only (one per model in compare mode)
    pub track: Vec<TrackScore>,       // how often past leaderboard predictions got the direction right
    pub ai_status: String,
    pub last_ok_ms: i64,
    pub healthy: bool,
}

#[derive(Serialize, Deserialize)]
pub enum ToViewer {
    Symbols(Vec<String>),
    Snap(Snapshot),
}

#[derive(Serialize, Deserialize)]
pub enum ToEngine {
    Select(String),
    Timeframe(usize),
    Restart,
    Stop,
    TestAlert,
    Ai(AiSettings),
}

/// (id, label, implemented)
pub const DATA_SOURCES: &[(&str, &str, bool)] = &[
    ("yahoo", "Yahoo Finance", true),
    ("alpaca_iex", "Alpaca IEX (free)", false),
    ("alpaca_sip", "Alpaca SIP ($99/mo)", false),
    ("databento", "Databento", false),
    ("massive", "Massive (ex-Polygon)", false),
    ("ibkr", "Interactive Brokers", false),
    ("tiingo", "Tiingo", false),
    ("finnhub", "Finnhub", false),
    ("alphavantage", "Alpha Vantage", false),
    ("twelvedata", "Twelve Data", false),
];

/// (id, label, implemented). When a LIVE entry is implemented, activating it must require typing CONFIRM.
pub const EXECUTION: &[(&str, &str, bool)] = &[
    ("paper", "Paper (built-in simulator)", true),
    ("alpaca_paper", "Alpaca paper", false),
    ("alpaca_live", "Alpaca LIVE", false),
    ("ibkr_paper", "IBKR paper", false),
    ("ibkr_live", "IBKR LIVE", false),
];

/// Candlestick prediction models: (id, label, runnable in this app). Kronos runs natively in Rust (kronos.rs).
pub const AI_MODELS: &[(&str, &str, bool)] = &[
    ("kronos_small", "Kronos-small (candlestick, 25M)", true),
    ("kronos_base", "Kronos-base (candlestick, 102M)", true),
    ("kronos_mini", "Kronos-mini (candlestick, 4M)", true),
    (MC_MODEL, "Black-Scholes Monte Carlo (GBM)", true),
    ("chronos", "Chronos (pretrained, Amazon)", false),
    ("timesfm", "TimesFM (pretrained, Google)", false),
    ("moirai", "Moirai (pretrained, Salesforce)", false),
    ("lag_llama", "Lag-Llama (pretrained)", false),
    ("patchtst", "PatchTST (train your own)", false),
    ("tft", "Temporal Fusion Transformer", false),
    ("nbeats", "N-BEATS", false),
    ("lstm", "LSTM", false),
    ("tcn", "TCN", false),
    ("gbm", "Gradient boosting (LightGBM-style)", false),
    ("custom", "Custom (your own model)", false),
];

/// Candlestick prediction model settings. Stored only; nothing runs until a model is installed.
#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct AiSettings {
    pub model: String, // AI_MODELS id, "" = none picked
    pub enabled: bool,
    pub input_tf: usize, // one-stock panel: 0 = follow chart, else TIMEFRAMES index + 1
    pub lookback: u32,   // candles fed in
    pub horizon: u32,    // one-stock panel: candles predicted ahead
    pub board: bool,     // leaderboard: predict every tracked symbol
    pub board_tf: usize, // leaderboard timeframe (TIMEFRAMES index)
    pub board_horizon: u32,
    pub compare: bool,      // COMPARE mode: leaderboard shows compare_a vs compare_b side by side
    pub compare_a: String,  // AI_MODELS id
    pub compare_b: String,
    pub overlay: bool,   // draw prediction on chart
    pub show_conf: bool,
    pub min_conf: u32, // % needed to count as a signal
    pub background: bool,
    pub notify: bool,
    pub gpu: bool,
}

impl Default for AiSettings {
    fn default() -> Self {
        Self { model: "kronos_small".into(), enabled: true, input_tf: 0, lookback: 256, horizon: 5, overlay: true, show_conf: true, min_conf: 70, background: true, board: true, board_tf: 3, board_horizon: 5, compare: false, compare_a: MC_MODEL.into(), compare_b: "kronos_small".into(), notify: true, gpu: false }
    }
}

/// Simulation model (no files): Black-Scholes price paths, see ai::monte_carlo.
pub const MC_MODEL: &str = "bs_mc";

/// Kronos files under data_dir()/models: (predictor folder, tokenizer folder, max context in candles).
pub fn kronos_files(id: &str) -> Option<(&'static str, &'static str, usize)> {
    match id {
        "kronos_small" => Some(("Kronos-small", "Kronos-Tokenizer-base", 512)),
        "kronos_base" => Some(("Kronos-base", "Kronos-Tokenizer-base", 512)),
        "kronos_mini" => Some(("Kronos-mini", "Kronos-Tokenizer-2k", 2048)),
        _ => None,
    }
}

/// READY (runnable + files present), NOT DOWNLOADED, or N/A (no Rust implementation yet).
pub fn model_status(id: &str) -> &'static str {
    match kronos_files(id) {
        None if id == MC_MODEL => "READY",
        None => "N/A",
        Some((p, t, _)) => {
            let m = data_dir().join("models");
            if m.join(p).join("model.safetensors").exists() && m.join(t).join("model.safetensors").exists() { "READY" } else { "NOT DOWNLOADED" }
        }
    }
}

/// Track record of one (model, symbol, timeframe, horizon): direction right `hits` of `total` checked.
#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Debug)]
pub struct TrackScore {
    pub model: String,
    pub sym: String,
    pub tf: usize,
    pub h: usize,
    pub hits: u32,
    pub total: u32,
}

/// One forecast step: mean of all samples, plus the spread of the sampled closes.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct PredStep {
    pub t: i64,
    pub o: f64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    pub c_lo: f64, // lowest sampled close
    pub c_hi: f64, // highest sampled close
    pub p_up: f64, // share of samples closing above the last real close
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Prediction {
    pub sym: String,
    pub tf: usize,
    pub model: String,
    pub made_ms: i64,
    pub last_close: f64,
    pub lookback: usize,
    pub samples: usize,
    pub ms: u64,
    pub steps: Vec<PredStep>,
    pub exp_pct: f64,   // mean final close vs last close, %
    pub p_up: f64,      // share of samples ending higher
    pub p10_pct: f64,   // 10th / 90th percentile of the final close, % vs last close
    pub p90_pct: f64,
    pub sigma_pct: f64, // typical move over the same number of candles, from recent history
    pub z: f64,         // expected move in units of sigma
    pub unusual: bool,  // unusually high predicted rise (see ai::UNUSUAL_Z)
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Settings {
    pub data_source: String,
    pub execution: String,
    pub mode: String, // "AUTO-PAPER" | "SIGNALS"
    pub lively: bool,
    pub tapes: bool,
    pub flash: bool,
    pub live_candle: bool,
    pub trade_pre: bool,
    pub trade_regular: bool,
    pub trade_post: bool,
    pub start_cash_cents: i64,
    pub max_position_pct: u32,
    pub daily_loss_stop_pct: u32,
    pub poll_ms: u64,
    pub watchlist: Vec<String>,
    pub ai: AiSettings,
}

impl Default for Settings {
    fn default() -> Self {
        let w = "AAPL MSFT NVDA AMZN GOOGL META TSLA AMD AVGO NFLX JPM XOM SPY QQQ IWM \
                 CL=F BZ=F NG=F GC=F SI=F HG=F ZC=F ZW=F EURUSD=X GBPUSD=X USDJPY=X BTC-USD ^GSPC ^IXIC ^VIX";
        Self {
            data_source: "yahoo".into(),
            execution: "paper".into(),
            mode: "AUTO-PAPER".into(),
            lively: true,
            tapes: true,
            flash: true,
            live_candle: true,
            trade_pre: true,
            trade_regular: true,
            trade_post: true,
            start_cash_cents: 100_000_00,
            max_position_pct: 5,
            daily_loss_stop_pct: 2,
            poll_ms: 2000,
            watchlist: w.split_whitespace().map(String::from).collect(),
            ai: AiSettings::default(),
        }
    }
}

impl Settings {
    /// Err carries a message when the file exists but can't be parsed (defaults are used).
    pub fn load() -> (Self, Option<String>) {
        match fs::read_to_string(data_dir().join("settings.json")) {
            Err(_) => (Self::default(), None),
            Ok(s) => match serde_json::from_str(&s) {
                Ok(v) => (v, None),
                Err(e) => (Self::default(), Some(format!("settings.json invalid ({e}), using defaults"))),
            },
        }
    }
    pub fn save(&self) {
        let _ = fs::write(data_dir().join("settings.json"), serde_json::to_string_pretty(self).unwrap());
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(default)]
pub struct MathEntry {
    pub name: String,
    pub equation: String,
    pub contributes: Vec<String>,
    pub part_of: String,
    pub applies_to: String,
    pub notes: String,
    pub added: String,
}

pub fn load_maths() -> Result<Vec<MathEntry>, String> {
    let p = data_dir().join("maths.json");
    if !p.exists() {
        let example = vec![MathEntry {
            name: "EXAMPLE - replace me".into(),
            equation: "y = a * x + b".into(),
            contributes: vec!["what this feeds into".into(), "one short bullet per point".into()],
            part_of: "strategy name".into(),
            applies_to: "e.g. large-cap tech, oil producers".into(),
            notes: "edit this file to add entries".into(),
            added: chrono::Utc::now().format("%Y-%m-%d").to_string(),
        }];
        let _ = fs::write(&p, serde_json::to_string_pretty(&example).unwrap());
    }
    let s = fs::read_to_string(&p).map_err(|e| e.to_string())?;
    serde_json::from_str(&s).map_err(|e| format!("maths.json invalid: {e}"))
}

/// %LOCALAPPDATA%\WetIronTrading - settings, caches and downloaded data live here.
pub fn data_dir() -> PathBuf {
    let p = env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|| ".".into()).join("WetIronTrading");
    let _ = fs::create_dir_all(&p);
    p
}

pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

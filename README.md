# WetIronTrading

<!-- AI-NOTES: dense context for Claude. Human-readable not required. Update as decisions land. -->

## WHAT
- Rust trading platform. Systematic/algorithmic. Equities (stocks) first.
- Goal: predict good entry (buy) timing + exit; short-term AND long-term horizons.
- Live: ingest data "from everywhere", analyse ALL stocks live (universe-wide scan, not watchlist).
- Backtesting = first-class, not afterthought.
- Long-term project: user will add prediction sources one by one over time.
- Status 2026-11-11: PHASE 1 BUILT (user said "now build what we talked about"). Display-only app, Yahoo live data.

## RUN
- `cargo build --release` then double-click `target\release\wetiron.exe` (opens in Windows Terminal). Viewer auto-starts engine if not running.
- Look: font VT323 (installed per-user 2026-11-11, %LOCALAPPDATA%\Microsoft\Windows\Fonts), green #26CC26 (ui::GREEN). App writes a Windows Terminal fragment profile "WetIronTrading" (%LOCALAPPDATA%\Microsoft\Windows Terminal\Fragments\WetIronTrading\profile.json) and relaunches itself into it as `wetiron ui`; user's WT settings.json untouched. No WT -> runs in current window.
- `wetiron engine` = background engine alone (no window). Closing window leaves engine running; `:stop` ends it.
- REBUILD GOTCHA: a running engine locks the exe on Windows -> `:stop` (or kill wetiron.exe) before `cargo build`.
- Data/settings: %LOCALAPPDATA%\WetIronTrading\ (settings.json, maths.json, symbols.txt, yahoo\<date>\<SYM>_1m.json).
- Tests: `cargo test` (format helpers + renders every tab at many window sizes). `DUMP=<tab><width> cargo test -- --nocapture` prints a frame, e.g. DUMP=0220.

## CODE MAP (single crate, one exe, two modes)
- src/main.rs dispatch | shared.rs types, settings, protocol, source catalog | yahoo.rs adapter | engine.rs background engine (TCP 127.0.0.1:47821, JSON lines, snapshot pushed on change) | viewer.rs window state/input/IPC/engine spawn | ui.rs all drawing.
- Adding a data source = new adapter like yahoo.rs + flip `implemented` in shared::DATA_SOURCES + branch in engine::run.
- PHASE 1 SCOPE (user, 2026-11-11): build the APP DESIGN + DISPLAYING DATA only. NO maths, calculations, predictions, strategies, signals logic. Panels for those exist but show placeholders/empty. User adds the maths themself later.

## ENV
- Win11, PowerShell primary (bash avail). rustc/cargo 1.99.1.
- Owner: Viktor Ivanov. Branch main. No remote known yet.

## LANGUAGE RULE (user, emphatic)
- Rust everywhere. Python ONLY where 100% necessary. If it can be done in Rust, it's in Rust.

## DECISIONS (user, 2026-11-11)
- Mode: BOTH signals-only AND auto paper trading. Settings toggle between them.
- Horizons: short = intraday (1-5 min bars), long = days-weeks (daily bars).
- Universe: ALL stocks (no price/volume filter by default).
- Hours: all sessions (pre, regular, after). UI shows which session is active; user can change which sessions are traded.
- Paper account defaults: $100k, max 5% per position, stop trading for day after -2%. Editable.
- Long only (default; shorting later).
- Storage: keep most data downloaded locally; revisit when too big.
- Source switching: change in settings -> app restarts engine (no hot-swap).

## PREDICTION SOURCES (add one at a time, long term)
- Planned list (user): systematic math strategies; AI models (on candles + world data, running in background); news; live simulated city traffic; heat/weather forecasts -> oil physical properties; wildfires; wars; education; commodities/currency/asset news; house prices; sea levels; climate change. "as much as possible".
- Implication: each = plugin with same interface: inputs (time series, point-in-time) -> prediction(s) per symbol w/ confidence + horizon. A combiner merges them. Adding one = new plugin, no core change.
- Alt-data must still obey NO-LOOKAHEAD (store when we learned it, not just event time).

## UI (user decided 2026-11-11)
- Rust as much as possible (user wants speed). Engine headless/separate; UI = viewer.
- Pops up as its own window on user's screen. NOT pretty. Black bg, green terminal font, ALL text.
- Candlesticks drawn from characters. Sections separated by |. Arranged intelligently, live updating.
- ONE green only, everywhere (for now). up/down candles distinguished by shape (hollow vs █), not colour.
- Separator = ASCII '|' (1 byte; user wants fastest; real diff negligible).
- User: "every single little time matters". -> UI must never block engine. engine publishes snapshots; UI renders latest at capped fps.
- Proposed (not yet agreed): ratatui. v1 runs in dedicated terminal window w/ black/green profile; later maybe own native window via ratatui GPU backend.
- MAIN screen: candle chart, scanner list, strategies, open positions. Future: live news panel, AI predicted outcomes panel.
- Tabs along top (each opens own screen): settings, data sources, execution, STORAGE (computer disk usage + our data size), BACKTEST (edit params, run fast many calcs in Rust).
- UX AGREED 2026-11-11: F-key tabs | ':' command bar | bottom status bar (source, mode, session, ET clock, delay, health) | '?' help overlay | Tab = focus panel (focused = double border / '>') | Space = maximise panel | alerts flash inverse on status line. "we will add more changes later".
- Next to command bar: scrollable list of stocks/commodities, mouse wheel scroll + click to pick.
- Mouse tracked everywhere (hover + click), not keyboard-only.
- "Data flows all around the page" = ALL THREE: scrolling ticker tapes (top/bottom edges), numbers flash inverse on change, chart draws newest candle live. Each toggleable in settings; global STATIC vs LIVELY switch.
- Picker list = EVERYTHING (all stocks + commodities etc), wheel-scroll + click. (proposed: type to filter; symbol master list from Nasdaq's free symbol directory files - unverified)
- Notifications: status-line flash AND Windows notification.
- AI tab (F4, user 2026-11-11): same 5-panel grid as MAIN (live chart + companies, shared symbol/zoom) but STRATEGIES/POSITIONS/LOG replaced by AI MODEL (pick from list, all [N/A], none installed - user said don't install yet), CANDLESTICK PREDICTION MODEL (on/off + toggleable settings: input tf, lookback, horizon, chart overlay, confidence, min conf, background, scope, notify, CPU/GPU; stored in settings.json "ai"), PREDICTION OUTPUT (placeholder). Only this one AI model "for now". Future model runtime should be Rust (e.g. ONNX/candle) per language rule.
- AI MODEL WORKING (2026-11-11): Kronos-small runs natively in Rust (src/kronos.rs, no ML framework), loop in src/ai.rs inside the engine.
  - Files: %LOCALAPPDATA%\WetIronTrading\models\Kronos-small\ + \Kronos-Tokenizer-base\ (model.safetensors, config.json, README.md model card, MIT). User downloaded from HF; dup gitattributes deleted.
  - Kronos-base ADDED 2026-11-11: models\Kronos-base\ (102.3M params, 409 MB, d_model 832, 16 heads, 12 layers; same Tokenizer-base, context 512). Pick in AI MODEL panel. Verified vs oracle (KRONOS_MODEL=Kronos-base): tokens 96/96 identical, logits within 1e-5. Live: ~6-9 s per prediction at lookback ~507 (small ~1-2 s), so a 30-stock leaderboard pass takes ~4 min on base.
  - Verified vs independent uncached PyTorch re-implementation (scratchpad oracle, not in repo): all tokens identical, logits/decoder within 2e-5. Running the downloaded official Python code was blocked by the safety classifier, so the oracle was written from reading the source instead.
  - 16 samples batched (shared context KV cache): ~1 s per prediction at lookback 256 on 8 cores. lookback capped at max_context - horizon (512 for small/base).
  - Stats per prediction: mean OHLC path, sampled close range, P(up) per step, p10/p90 final, sigma (std of 1-candle log returns * sqrt(h)), z = ln(mean final / last)/sigma.
  - UNUSUAL RISE = z >= 2 (ai::UNUSUAL_Z) AND P(up) >= "Min confidence" setting -> F4 AI tab blinks; Windows notification once when it turns unusual (if "Notify" ON).
  - AI chart overlay: '┆' now-line, predicted wick '┊', up '▒', down '▓', '·' = highest/lowest sampled close.
  - AI settings changes go live to the engine (ToEngine::Ai), no restart.
- TAB RENAMED (user 2026-11-11): F4 AI -> F4 FORECAST (holds AI models AND simulations). Panel AI MODEL -> MODEL. Settings key still "ai", consts still T_AI.
- BLACK-SCHOLES MONTE CARLO (user 2026-11-11): model id "bs_mc" (shared::MC_MODEL) in the MODEL list, no files, always READY. ai::monte_carlo: 1000 GBM paths, drift+vol = mean/std of lookback 1-candle log returns (historical drift, NOT risk-free: forecasting, not option pricing), 4 sub-steps/candle for high/low. Same Prediction output -> chart overlay, PREDICTION OUTPUT, LEADERBOARD, unusual flash all work unchanged. ~1 ms each; no lookback cap. Logbook entry in maths.json says where it lives. User: it's a prediction model / baseline for Kronos, not a strategy.
  - Caveat: leaderboard ranks by MEAN % move; under GBM mean > median for volatile symbols (e.g. ^VIX tops with P(up) 50%). Ranking by P(up) or median is an open choice.
- COMPARE MODE (user 2026-11-11): big 3-row [COMPARE ON/OFF] button top-left of the leaderboard panel (Click::Ctl(5)); ON turns the leaderboard into model A | model B for the 30 tracked stocks (settings ai.compare/compare_a/compare_b, A/B cycle through READY models, Ctl 6/7). Uses the leaderboard timeframe/horizon.
  - Engine runs board jobs for BOTH models (symbol-major so a stock gets both back to back), board keyed (model, symbol), shared 30 s candle cache, Kronos models loaded on demand + freed when unused. Single-stock panel still uses the picked model.
  - Highlights (inverse rows), user chose BOTH meanings of "pair success":
    - A1-A3 AGREE NOW: order by the WORSE of the two models' ranks (by exp%), tie by sum; needs both exp>0 and both P(up)>=50%. Header also shows Spearman rank correlation of the two models.
    - R1-R3 BOTH RIGHT MOST: track record. Every leaderboard prediction (any mode) saved as pending in data_dir/track.json (one per model/sym/tf/h/last candle); once a newer candle exists after its target candle, direction checked (close at target vs start close). Score = worse of the two hit rates, needs >= 5 checked per model per stock (ui::MIN_CHECKED), scored per tf+horizon.
    - Track record only fills as target candles close: 1 MIN x 5 = minutes; 1 MONTH x 20 = 20 months. Pending older than the downloaded range is dropped unchecked.
- LEADERBOARD (user 2026-11-11): AI tab = 6 panels. Top: chart 60% | LEADERBOARD 40% (user: more important than companies/model list here). Bottom: CANDLESTICK PREDICTION MODEL 20 | PREDICTION OUTPUT (one stock, kept) 34 | SCANNER 20 | AI MODEL 26. MAIN keeps its 5-panel layout.
  - Prediction span: Kronos predicts N candles of the timeframe it's fed, so "how far ahead" = timeframe x N; each panel's top line shows it, e.g. "(= 5 hours of trading ahead)" (ui::ahead).
  - Leaderboard predicts every watchlist symbol (30) on its OWN timeframe + horizon (settings ai.board/board_tf/board_horizon, default ON, 1 HOUR, 5 candles), ranked by expected % rise. Click row = chart it. Each symbol re-run ~every 60 s (ai::BOARD_REFRESH); ~1 s each, so the full board fills in ~35 s. Charted stock always gets priority.
  - Timeframe/horizon controls = clickable [values] on each panel's top line (click = next, wheel = step; Click::Ctl). One-stock timeframe can FOLLOW CHART or be fixed. Removed from settings rows; "whole watchlist" setting replaced by the board.
  - Leaderboard unusual rises also blink F4 + notify. Changing board tf/horizon clears the board (ranks not comparable).
- MATHS tab: logbook of equations/calculations user uses/adds. Entry = name, equation (text), short bullets on what it contributes to, what it's part of (strategy), which stock sections it applies to, notes, date added. More fields later.
- STORAGE tab: disk total/used/free, our data by source, growth/day, days-until-full.
- BACKTEST tab: pick strategy/symbols/dates, edit params, sweep across all cores, results (text equity curve, drawdown, win rate, trades, table of all combos tried). Phase 1 = layout only.

## RUNTIME (user, 2026-11-11)
- Engine keeps running in background when window closed; runs "in its own section" of the computer, uses cache when needed. "smooth Rust typa thing".
- => 2 programs: background engine (always on) + viewer window (opens/closes freely, attaches to engine via local IPC).
- Started MANUALLY by user (no auto-start at Windows boot, no tray icon).

## SOURCES (user decided 2026-11-11)
- App has a SETTINGS section where user picks data source + execution mode from a list.
- Data: Yahoo Finance ACTIVE (just to get working; unofficial, rate-limited, can't truly stream whole market). Dummy (listed, not implemented, marked [N/A] since 1 colour): Alpaca (IEX free / SIP $99), Databento, Massive (ex-Polygon), IBKR, + other reasonable ones.
- Execution: built-in paper simulator ACTIVE (same sim as backtest). Dummy: Alpaca paper, Alpaca live, IBKR paper, IBKR live.
- Every source/broker sits behind one shared interface so swapping = config, not rewrite.
- Live-money modes must require explicit confirm even when implemented later.
- Research notes: Alpaca Algo Trader Plus ~$99/mo SIP; Databento live US eq needs Standard/Plus/Unlimited sub (price unverified); Massive real-time = Advanced plan (~$199 per 3rd party, unverified). Avoid IBKR as data (line limits).

## INVARIANTS (apply once code exists)
- SAME strategy code path for backtest + live. no divergence. strategy = pure fn of (state, event) -> signals/orders.
- NO lookahead bias. event-time ordering strict. point-in-time data only (survivorship-free universe, as-reported fundamentals, split/div adjust correctly).
- backtest must model costs: commission, spread, slippage, latency, partial fills, borrow for shorts.
- money = fixed-point/integer (cents or rust_decimal), never f64 for P&L/order qty/price in accounting. f64 ok for signal math.
- timestamps UTC, ns precision, exchange tz conversion only at edges. handle DST, holidays, half-days, halts.
- risk layer between strategy and broker: max position, max loss/day, kill switch. can't be bypassed.
- secrets (API keys) never in repo. .env + .gitignore.
- deterministic backtests (seeded, reproducible).
- overfitting guard: walk-forward / out-of-sample split, record params tried.

## LIKELY SHAPE (tentative, not agreed)
cargo workspace. crates roughly: data-ingest (adapters per source) -> normalized event stream -> store -> strategy engine -> risk -> execution(broker adapter | sim) ; backtest = replay store through same engine w/ sim execution. tokio for async IO.
Keep it small; add crates only when needed.

## LOG
- 2026-11-11: README created. intro from user.
- 2026-11-11: UI direction set (green-on-black all-text window).
- 2026-11-11: data/exec sources, mode, horizons, universe, hours, paper defaults, prediction-source vision, tabs (storage, backtest) decided.
- 2026-11-11: phase 1 built. verified headless: engine pulls 12,896 symbols, quotes, 1m candles, session times; select + stop work. viewer verified only via TestBackend renders, NOT yet run interactively by user.
- known gaps: maths logbook + account limits edited via JSON files (no in-app forms); scanner = watchlist only (Yahoo can't stream all stocks); detached engine surviving window close untested; ':stop' has no confirm prompt.

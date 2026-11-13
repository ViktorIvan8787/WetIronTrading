//! Everything drawn: one green on black, all text, sections split by '|'.
use crate::ai::UNUSUAL_Z;
use crate::shared::*;
use crate::paper::{self, SHARE};
use crate::viewer::*;
use std::collections::HashMap;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::symbols::border;
use ratatui::text::Line;
use ratatui::widgets::{Block, Clear, Paragraph};
use ratatui::Frame;
use std::time::Duration;

pub const GREEN: (u8, u8, u8) = (38, 204, 38); // #26CC26, softer phosphor green
const G: Color = Color::Rgb(GREEN.0, GREEN.1, GREEN.2);
const PIPE: border::Set = border::Set {
    top_left: "+", top_right: "+", bottom_left: "+", bottom_right: "+",
    vertical_left: "|", vertical_right: "|", horizontal_top: "-", horizontal_bottom: "-",
};
const FOCUS: border::Set = border::DOUBLE;

fn base() -> Style {
    Style::new().fg(G).bg(Color::Black)
}
fn inv() -> Style {
    Style::new().fg(Color::Black).bg(G)
}
fn panel(title: &str, focused: bool) -> Block<'static> {
    Block::bordered().border_set(if focused { FOCUS } else { PIPE }).title(format!(" {title} ")).style(base())
}
fn inside(r: Rect, (x, y): (u16, u16)) -> bool {
    x >= r.x && x < r.right() && y >= r.y && y < r.bottom()
}

struct Ui<'a> {
    a: &'a App,
    hits: Vec<(Rect, Click)>,
}

pub fn draw(f: &mut Frame, a: &App) -> Vec<(Rect, Click)> {
    let mut u = Ui { a, hits: vec![] };
    let area = f.area();
    f.render_widget(Block::new().style(base()), area);
    if area.width < 40 || area.height < 10 {
        f.render_widget(Paragraph::new("window too small").style(base()), area);
        return u.hits;
    }
    if a.strat.is_some() {
        u.strat_window(f, area);
        return u.hits;
    }
    let t = a.settings.tapes;
    let mut c = vec![Constraint::Length(1), Constraint::Length(1)];
    if t {
        c.push(Constraint::Length(1));
    }
    c.push(Constraint::Min(0));
    if t {
        c.push(Constraint::Length(1));
    }
    c.push(Constraint::Length(1));
    let rows = Layout::vertical(c).split(area);
    u.tabs(f, rows[0]);
    u.cmdbar(f, rows[1]);
    let body = rows[if t { 3 } else { 2 }];
    if t {
        u.tape(f, rows[2], true);
        u.tape(f, rows[4], false);
    }
    u.status(f, rows[rows.len() - 1]);
    match a.tab {
        T_MAIN => u.grid(f, body, false),
        T_SHARES => u.click_list(f, body, "CURRENT SHARES - running strategies, paper or live (click one = open its window, wheel = scroll)", shares_lines(a)),
        T_HISTORY => u.click_list(f, body, "HISTORY - terminated runs, newest first (click one = open its window, wheel = scroll)", history_lines(a)),
        T_SCAN => u.scanner(f, body, true, true),
        T_STRAT => u.text(f, body, "STRATEGIES", false, strat_lines()),
        T_AI | T_MATHF => u.grid(f, body, true),
        T_BACKTEST => u.backtest(f, body),
        T_DATA => u.data(f, body),
        T_STORAGE => u.storage(f, body),
        T_MATHS => u.maths(f, body),
        _ => u.settings(f, body),
    }
    if a.help {
        u.help(f, area);
    }
    u.hits
}

impl Ui<'_> {
    fn hover(&self, r: Rect) -> bool {
        inside(r, self.a.mouse)
    }

    fn text(&mut self, f: &mut Frame, r: Rect, title: &str, focused: bool, lines: Vec<String>) {
        f.render_widget(Paragraph::new(lines.into_iter().map(Line::from).collect::<Vec<_>>()).block(panel(title, focused)).style(base()), r);
    }

    fn tabs(&mut self, f: &mut Frame, r: Rect) {
        let mut x = r.x;
        // widest style that fits: " F1 CURRENT SHARES " .. "SHARES" (no key numbers) at 80 columns
        let label = |(pad, key, short, num): (&str, &str, bool, bool), i: usize| {
            let name = if short { TABS_SHORT[i] } else { TABS[i] };
            if num { format!("{pad}{key}{} {name}{pad}", i + 1) } else { name.to_string() }
        };
        let tiers = [(" ", "F", false, true), ("", "F", false, true), ("", "", false, true), ("", "", true, true), ("", "", true, false)];
        let fits = |t: (&str, &str, bool, bool)| (0..TABS.len()).map(|i| label(t, i).len() + 1).sum::<usize>() <= r.width as usize;
        let tier = tiers.into_iter().find(|t| fits(*t)).unwrap_or(tiers[4]);
        for i in 0..TABS.len() {
            let s = label(tier, i);
            let w = s.len() as u16;
            if x + w > r.right() {
                break;
            }
            let rr = Rect::new(x, r.y, w, 1);
            let mut st = if self.a.tab == i || self.hover(rr) { inv() } else { base() };
            if (i == T_AI || i == T_MATHF) && self.a.ai_alert(i == T_MATHF) {
                // unusual predicted rise: blink twice a second
                st = if (self.a.started.elapsed().as_millis() / 500) % 2 == 0 { inv() } else { base() };
            }
            f.buffer_mut().set_string(x, r.y, &s, st);
            self.hits.push((rr, Click::Tab(i)));
            x += w;
            if x < r.right() {
                f.buffer_mut().set_string(x, r.y, "|", base());
                x += 1;
            }
        }
        let h = " ? help ";
        if x + h.len() as u16 <= r.right() {
            f.buffer_mut().set_string(r.right() - h.len() as u16, r.y, h, base());
        }
    }

    /// Command bar + horizontally scrolling picker of every symbol (wheel/click; typing filters it).
    fn cmdbar(&mut self, f: &mut Frame, r: Rect) {
        let a = self.a;
        let [left, right] = Layout::horizontal([Constraint::Percentage(30), Constraint::Min(12)]).areas(r);
        let txt = match &a.cmd {
            Some(b) => format!(":{b}_"),
            None => ": press : to type a command or symbol".into(),
        };
        let st = if a.cmd.is_some() { inv() } else { base() };
        f.buffer_mut().set_stringn(left.x, left.y, format!("{txt:<w$}", w = left.width as usize), left.width as usize, st);

        let items = a.picker_items();
        let y = right.y;
        self.hits.push((right, Click::PickScroll(0)));
        let buf = f.buffer_mut();
        buf.set_string(right.x, y, "| <", base());
        self.hits.push((Rect::new(right.x + 2, y, 1, 1), Click::PickScroll(-10)));
        let end = right.right().saturating_sub(2);
        buf.set_string(end, y, " >", base());
        self.hits.push((Rect::new(end + 1, y, 1, 1), Click::PickScroll(10)));
        let mut x = right.x + 4;
        if items.is_empty() {
            buf.set_string(x, y, if a.symbols.is_empty() { "(loading symbols)" } else { "(no match)" }, base());
        }
        for s in items.iter().skip(a.picker_off) {
            let w = s.len() as u16;
            if x + w >= end {
                break;
            }
            let rr = Rect::new(x, y, w, 1);
            let st = if inside(rr, a.mouse) { inv() } else { base() };
            buf.set_string(x, y, s.as_str(), st);
            self.hits.push((rr, Click::Pick(s.to_string())));
            x += w + 1;
        }
    }

    /// Scrolling ticker tape: top = stocks (scrolls left), bottom = futures/FX/crypto/indices (scrolls right).
    fn tape(&mut self, f: &mut Frame, r: Rect, top: bool) {
        let a = self.a;
        let items: Vec<String> = a.snap.quotes.iter().filter(|q| is_stock(&q.sym) == top).map(|q| format!("{} {} {:+.2}%", q.sym, px(q.price), q.change_pct)).collect();
        if items.is_empty() {
            f.buffer_mut().set_string(r.x, r.y, " (waiting for data)", base());
            return;
        }
        let s = items.join("  |  ") + "  |  ";
        let n = s.chars().count();
        let step = if a.settings.lively { (a.started.elapsed().as_millis() / 120) as usize % n } else { 0 };
        let off = if top { step } else { (n - step) % n };
        let line: String = s.chars().cycle().skip(off).take(r.width as usize).collect();
        f.buffer_mut().set_string(r.x, r.y, line, base());
    }

    fn status(&mut self, f: &mut Frame, r: Rect) {
        let a = self.a;
        let w = r.width as usize;
        if let Some((m, _)) = a.status_flash.as_ref().filter(|(_, t)| t.elapsed() < Duration::from_secs(4)) {
            f.buffer_mut().set_stringn(r.x, r.y, format!(" !! {m:<w$}"), w, inv());
            return;
        }
        let now = chrono::Utc::now().timestamp();
        let off = a.snap.session.map_or(-18_000, |s| s.gmtoffset);
        let src = DATA_SOURCES.iter().find(|d| d.0 == a.settings.data_source).map_or("?", |d| d.1);
        let upd = if a.snap.last_ok_ms > 0 { format!("updated {:.1}s ago", (now_ms() - a.snap.last_ok_ms) as f64 / 1000.0) } else { "no data yet".into() };
        let eng = match a.conn {
            Conn::Up if a.snap.healthy || a.snap.last_ok_ms == 0 => "engine OK",
            Conn::Up => "engine OK, DATA ERROR",
            Conn::Connecting => "engine STARTING",
            Conn::Stopped => "engine STOPPED",
        };
        let s = format!(" {} | {} | {} | {} ET | {} | {}", src.to_uppercase(), a.settings.mode, session_label(a, now), hms(now + off), upd, eng);
        f.buffer_mut().set_stringn(r.x, r.y, s, w, base());
    }

    /// Grid shared by MAIN (5 panels) and AI (6, + LEADERBOARD): chart + companies top-left, the rest per tab.
    fn grid(&mut self, f: &mut Frame, r: Rect, ai: bool) {
        let a = self.a;
        let n = a.panel_count();
        let focus = a.focus.min(n - 1); // focus may come from the other tab
        let rects: Vec<Rect> = if a.maxed {
            vec![r; n]
        } else {
            let [top, bot] = Layout::vertical([Constraint::Percentage(70), Constraint::Percentage(30)]).areas(r);
            // AI: leaderboard takes the companies + model slots top-right; those move to the bottom row
            let bp: &[u16] = if ai { &[20, 34, 20, 26] } else { &[50, 50] };
            let b = Layout::horizontal(bp.iter().map(|p| Constraint::Percentage(*p))).split(bot);
            if ai {
                // chart | TEST button | leaderboard
                let [c, test, lb] = Layout::horizontal([Constraint::Percentage(56), Constraint::Length(15), Constraint::Min(0)]).areas(top);
                self.test_button(f, test);
                [&[c, lb][..], &b[..]].concat()
            } else {
                let t = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(20), Constraint::Percentage(20)]).split(top);
                [&t[..], &b[..]].concat()
            }
        };
        for (i, rr) in rects.into_iter().enumerate() {
            if a.maxed && i != focus {
                continue;
            }
            let foc = focus == i;
            self.hits.push((rr, Click::Focus(i)));
            match (ai, i) {
                (_, 0) => self.chart(f, rr, foc, ai),
                (true, 1) => self.leaderboard(f, rr, foc),
                (true, 2) => self.ai_settings(f, rr, foc),
                (true, 3) => self.prediction(f, rr, foc),
                (true, 4) => self.scanner(f, rr, foc, false),
                (true, _) => self.ai_models(f, rr, foc),
                (_, 1) => self.scanner(f, rr, foc, false),
                (_, 2) => self.text(f, rr, "STRATEGIES", foc, strat_lines()),
                (_, 3) => self.text(f, rr, "POSITIONS", foc, position_lines(a)),
                _ => {
                    let blk = panel("LOG", foc);
                    let h = blk.inner(rr).height as usize;
                    let log = &a.snap.log;
                    self.text(f, rr, "LOG", foc, log[log.len().saturating_sub(h)..].to_vec());
                }
            }
        }
    }

    /// One candle per column. Wick '│', up body '░' (hollow), down body '█', flat '─'.
    /// Mouse wheel / + - over it steps the timeframe.
    fn chart(&mut self, f: &mut Frame, r: Rect, focused: bool, ai: bool) {
        let a = self.a;
        let tf = TIMEFRAMES.get(a.snap.tf).map_or("?", |t| t.2);
        let mut title = match a.snap.quotes.iter().find(|q| q.sym == a.snap.selected) {
            Some(q) => format!("{} {}  {}  {:+.2}%  | {}  | wheel/+- = zoom", q.sym, tf, px(q.price), q.change_pct, q.name),
            None => format!("{} {tf}", a.snap.selected),
        };
        if ai {
            let (d, k) = (a.desk(), if a.math() { "MATH" } else { "AI" });
            title += &match a.single_pred() {
                _ if !d.enabled => format!("  | {k} PREDICTION OFF"),
                Some(p) if p.samples == 0 && !p.signal.is_empty() => format!("  | {k}: NO SIGNAL"),
                Some(p) if d.show_conf => format!("  | {k} {:+.2}% P(up) {:.0}%{}", p.exp_pct, p.p_up * 100.0, if p.unusual { " !! UNUSUAL" } else { "" }),
                Some(p) => format!("  | {k} {:+.2}%{}", p.exp_pct, if p.unusual { " !! UNUSUAL" } else { "" }),
                None => format!("  | {k} PREDICTION ON"),
            };
        }
        let blk = panel(&title, focused);
        let inner = blk.inner(r);
        f.render_widget(blk, r);
        self.hits.push((r, Click::Chart));
        if a.tf != a.snap.tf {
            let want = TIMEFRAMES.get(a.tf).map_or("?", |t| t.2);
            f.buffer_mut().set_stringn(inner.x, inner.y, format!("(loading {want} candles)"), inner.width as usize, base());
            return;
        }
        let mut cs: &[Candle] = &a.snap.candles;
        if !(a.settings.lively && a.settings.live_candle) && cs.len() > 1 {
            cs = &cs[..cs.len() - 1]; // hide the still-forming candle
        }
        const AXIS: u16 = 11;
        if inner.width <= AXIS + 2 || inner.height < 4 {
            return;
        }
        if cs.is_empty() {
            f.buffer_mut().set_string(inner.x, inner.y, "(waiting for candles)", base());
            return;
        }
        let w = (inner.width - AXIS) as usize;
        let h = inner.height - 1; // last row = time axis
        // AI overlay: predicted candles to the right of a '┆' "now" line
        let pred = a.single_pred().filter(|p| ai && a.desk().enabled && a.desk().overlay && p.samples > 0 && p.tf == a.snap.tf && p.steps.len() + 9 <= w);
        let pw = pred.map_or(0, |p| p.steps.len() + 1);
        let cs = &cs[cs.len().saturating_sub(w - pw)..];
        let mut hi = cs.iter().map(|c| c.h).fold(f64::MIN, f64::max);
        let mut lo = cs.iter().map(|c| c.l).fold(f64::MAX, f64::min);
        for s in pred.map_or(&[][..], |p| &p.steps) {
            hi = hi.max(s.h).max(s.c_hi);
            lo = lo.min(s.l).min(s.c_lo);
        }
        let span = (hi - lo).max(1e-9);
        let row = |p: f64| inner.y + (((hi - p) / span) * (h - 1) as f64).round() as u16;
        let buf = f.buffer_mut();
        for (i, c) in cs.iter().enumerate() {
            let x = inner.x + i as u16;
            for y in row(c.h)..=row(c.l) {
                buf.set_string(x, y, "│", base());
            }
            let body = if c.c > c.o { "░" } else if c.c < c.o { "█" } else { "─" };
            for y in row(c.o.max(c.c))..=row(c.o.min(c.c)) {
                buf.set_string(x, y, body, base());
            }
        }
        if let Some(p) = pred {
            // predicted: wick '┊', up '▒', down '▓', '·' = highest/lowest sampled close
            let x0 = inner.x + cs.len() as u16;
            for y in inner.y..inner.y + h {
                buf.set_string(x0, y, "┆", base());
            }
            for (i, s) in p.steps.iter().enumerate() {
                let x = x0 + 1 + i as u16;
                buf.set_string(x, row(s.c_lo), "·", base());
                buf.set_string(x, row(s.c_hi), "·", base());
                for y in row(s.h)..=row(s.l) {
                    buf.set_string(x, y, "┊", base());
                }
                let body = if s.c > s.o { "▒" } else if s.c < s.o { "▓" } else { "─" };
                for y in row(s.o.max(s.c))..=row(s.o.min(s.c)) {
                    buf.set_string(x, y, body, base());
                }
            }
        }
        let ax = inner.x + w as u16 + 1;
        for (y, p) in [(inner.y, hi), (inner.y + (h - 1) / 2, (hi + lo) / 2.0), (inner.y + h - 1, lo)] {
            buf.set_stringn(ax, y, px(p), (AXIS - 1) as usize, base());
        }
        // time axis: label every few columns, format fits the timeframe; timeframe name under the price axis
        let off = a.snap.session.map_or(-18_000, |s| s.gmtoffset);
        let label = |t: i64| time_label(t + off, a.snap.tf);
        let lw = label(cs[0].t).chars().count();
        let y = inner.y + h;
        let mut x = 0;
        while x + lw <= cs.len() {
            buf.set_string(inner.x + x as u16, y, label(cs[x].t), base());
            x += lw + 4;
        }
        buf.set_stringn(ax, y, format!("[{tf}]"), (AXIS - 1) as usize, base());
    }

    fn scanner(&mut self, f: &mut Frame, r: Rect, focused: bool, full: bool) {
        let a = self.a;
        let blk = panel(if full { "SCANNER - watchlist (Enter/click = chart)" } else { "SCANNER" }, focused);
        let inner = blk.inner(r);
        f.render_widget(blk, r);
        let w = inner.width as usize;
        let hdr = if full {
            format!("  {:<10}{:>12}{:>9}{:>12}{:>15}  {:<10}{}", "SYM", "PRICE", "CHG%", "PREV", "VOLUME", "TIME ET", "NAME")
        } else {
            format!("  {:<9}{:>10}{:>8}  {}", "SYM", "PRICE", "CHG%", "NAME")
        };
        f.buffer_mut().set_stringn(inner.x, inner.y, hdr, w, base());
        let vis = inner.height.saturating_sub(1) as usize;
        let scroll = if full { a.sel_row.saturating_sub(vis.saturating_sub(1)) } else { 0 };
        let off = a.snap.session.map_or(-18_000, |s| s.gmtoffset);
        for (i, q) in a.snap.quotes.iter().enumerate().skip(scroll).take(vis) {
            let y = inner.y + 1 + (i - scroll) as u16;
            let rr = Rect::new(inner.x, y, inner.width, 1);
            let mark = if self.hover(rr) || (full && a.sel_row == i) { ">" } else { " " };
            let line = if full {
                format!("{mark} {:<10}{:>12}{:>+8.2}%{:>12}{:>15}  {:<10}{}", q.sym, px(q.price), q.change_pct, px(q.prev_close), q.volume, hms(q.time + off), q.name)
            } else {
                format!("{mark} {:<9}{:>10}{:>+7.2}%  {}", q.sym, px(q.price), q.change_pct, q.name)
            };
            let flashing = a.settings.lively && a.settings.flash && a.flashes.get(&q.sym).and_then(|x| x.1).is_some_and(|t| t.elapsed() < Duration::from_millis(600));
            f.buffer_mut().set_stringn(inner.x, y, format!("{line:<w$}"), w, if flashing { inv() } else { base() });
            self.hits.push((rr, Click::Sym(q.sym.clone())));
        }
        if a.snap.quotes.is_empty() {
            f.buffer_mut().set_string(inner.x, inner.y + 1, "  (waiting for data)", base());
        }
    }

    /// Selectable rows starting at `first` (shared sel_row numbering within a tab).
    fn rows(&mut self, f: &mut Frame, inner: Rect, first: usize, lines: Vec<String>) {
        for (i, line) in lines.into_iter().enumerate() {
            let y = inner.y + i as u16;
            if y >= inner.bottom() {
                break;
            }
            let rr = Rect::new(inner.x, y, inner.width, 1);
            let mark = if self.a.sel_row == first + i || self.hover(rr) { ">" } else { " " };
            f.buffer_mut().set_stringn(inner.x, y, format!("{mark} {line}"), inner.width as usize, base());
            self.hits.push((rr, Click::Row(first + i)));
        }
    }

    fn ai_models(&mut self, f: &mut Frame, r: Rect, focused: bool) {
        let math = self.a.math();
        let blk = panel(if math { "STRATEGY - maths (click to pick, details in MATHS tab)" } else { "MODEL - AI or simulation (click to pick)" }, focused);
        let inner = blk.inner(r);
        f.render_widget(blk, r);
        let cur = &self.a.desk().model;
        let lines = desk_models(math).into_iter().map(|(id, label)| format!("{} {label:<36}[{}]", if id == cur { "[*]" } else { "[ ]" }, model_status(id))).collect();
        self.rows(f, inner, 0, lines);
    }

    fn ai_settings(&mut self, f: &mut Frame, r: Rect, focused: bool) {
        let math = self.a.math();
        let blk = panel(if math { "MATH FORECAST SETTINGS (Enter/click toggles)" } else { "CANDLESTICK PREDICTION MODEL (Enter/click toggles)" }, focused);
        let inner = blk.inner(r);
        f.render_widget(blk, r);
        let lines = (0..AI_ROWS).map(|i| ai_row(self.a.desk(), i)).map(|(k, v)| format!("{k:<22}{v}")).collect();
        self.rows(f, inner, desk_models(math).len(), lines);
    }

    /// One line of plain text with clickable [values] (click = next, wheel = step). Returns the space below.
    fn controls(&mut self, f: &mut Frame, r: Rect, parts: &[(String, Option<u8>)]) -> Rect {
        let mut x = r.x;
        for (s, id) in parts {
            let s = if id.is_some() { format!("[{s}]") } else { s.clone() };
            let w = (s.chars().count() as u16).min(r.right().saturating_sub(x));
            if w == 0 {
                break;
            }
            let rr = Rect::new(x, r.y, w, 1);
            f.buffer_mut().set_stringn(x, r.y, &s, w as usize, if id.is_some() && self.hover(rr) { inv() } else { base() });
            if let Some(id) = id {
                self.hits.push((rr, Click::Ctl(*id)));
            }
            x += w;
        }
        Rect { y: r.y + 1, height: r.height.saturating_sub(1), ..r }
    }

    /// PREDICTION OUTPUT (one stock): timeframe + horizon controls, then the forecast.
    fn prediction(&mut self, f: &mut Frame, r: Rect, focused: bool) {
        let a = self.a;
        let blk = panel("PREDICTION OUTPUT - charted stock", focused);
        let inner = blk.inner(r);
        f.render_widget(blk, r);
        let ai = a.desk();
        let tf = match ai.input_tf {
            0 => format!("FOLLOW CHART ({})", TIMEFRAMES.get(a.tf).map_or("?", |t| t.2)),
            i => TIMEFRAMES[(i - 1).min(TIMEFRAMES.len() - 1)].2.to_string(),
        };
        let rest = self.controls(f, inner, &[(" timeframe ".into(), None), (tf, Some(0)), ("  next ".into(), None), (ai.horizon.to_string(), Some(1)), (format!(" candles  {}", ahead(ai.input_tf.checked_sub(1).unwrap_or(a.tf).min(TIMEFRAMES.len() - 1), ai.horizon)), None)]);
        f.render_widget(Paragraph::new(prediction_lines(a).into_iter().map(Line::from).collect::<Vec<_>>()).style(base()), rest);
    }

    /// LEADERBOARD: every tracked symbol predicted on its own timeframe/horizon, highest expected rise first.
    /// The big COMPARE button turns it into model A vs model B side by side.
    fn leaderboard(&mut self, f: &mut Frame, r: Rect, focused: bool) {
        let a = self.a;
        let ai = a.desk();
        let title = if ai.compare { "COMPARE - two models on the tracked stocks (click row = chart)" } else { "LEADERBOARD - tracked stocks, highest predicted rise first (click = chart)" };
        let blk = panel(title, focused);
        let inner = blk.inner(r);
        f.render_widget(blk, r);
        if inner.height < 4 || inner.width < 20 {
            return;
        }
        // big 3-row COMPARE button: boxed when OFF, solid when ON / hovered
        let bw = 17u16;
        let btn = Rect::new(inner.x, inner.y, bw, 3);
        let solid = ai.compare || self.hover(btn);
        let label = format!("COMPARE {}", if ai.compare { "ON" } else { "OFF" });
        let rows = if solid {
            [" ".repeat(17), format!("{label:^17}"), " ".repeat(17)]
        } else {
            [format!("+{}+", "-".repeat(15)), format!("|{label:^15}|"), format!("+{}+", "-".repeat(15))]
        };
        for (i, row) in rows.iter().enumerate() {
            f.buffer_mut().set_stringn(btn.x, btn.y + i as u16, row, bw as usize, if solid { inv() } else { base() });
        }
        self.hits.push((btn, Click::Ctl(5)));

        let side = |y: u16| Rect::new(inner.x + bw + 1, inner.y + y, inner.width.saturating_sub(bw + 1), 1);
        let btf = ai.board_tf.min(TIMEFRAMES.len() - 1);
        let tfline = [("timeframe ".into(), None), (TIMEFRAMES[btf].2.to_string(), Some(3)), ("  next ".into(), None), (ai.board_horizon.to_string(), Some(4)), (format!(" candles  {}", ahead(btf, ai.board_horizon)), None)];
        let on = |p: &&Prediction, m: &str| p.model == m && p.tf == btf && p.steps.len() == ai.board_horizon as usize;
        let of = |m: &str| a.snap.board.iter().filter(|p| on(p, m)).count();
        let n = a.settings.watchlist.len();
        if ai.compare {
            self.controls(f, side(0), &[("A ".into(), None), (short(&ai.compare_a), Some(6)), ("  vs  B ".into(), None), (short(&ai.compare_b), Some(7))]);
            self.controls(f, side(1), &tfline);
            let pa: Vec<&Prediction> = a.snap.board.iter().filter(|p| on(p, &ai.compare_a)).collect();
            let pb: Vec<&Prediction> = a.snap.board.iter().filter(|p| on(p, &ai.compare_b)).collect();
            let hits = |m: &str| -> Vec<(&str, u32, u32)> { a.snap.track.iter().filter(|t| t.model == m && t.tf == btf && t.h == ai.board_horizon as usize).map(|t| (t.sym.as_str(), t.hits, t.total)).collect() };
            let c = compare(&pa, &pb, &hits(&ai.compare_a), &hits(&ai.compare_b));
            let rho = c.rho.map_or("-".into(), |r| format!("{r:+.2}"));
            self.controls(f, side(2), &[(format!("ranked A {}/{n}  B {}/{n}  | rank agreement (Spearman) {rho}", pa.len(), pb.len()), None)]);
            let rest = Rect { y: inner.y + 3, height: inner.height - 3, ..inner };
            self.compare_table(f, rest, &c, &hits(&ai.compare_a), &hits(&ai.compare_b));
            return;
        }
        self.controls(f, side(0), &[("leaderboard ".into(), None), (if ai.board { "ON" } else { "OFF" }.into(), Some(2)), (format!("  model {}", short(&ai.model)), None)]);
        self.controls(f, side(1), &tfline);
        self.controls(f, side(2), &[(format!("{}/{n} ranked", of(&ai.model)), None)]);
        let mut rest = Rect { y: inner.y + 3, height: inner.height - 3, ..inner };
        let w = rest.width as usize;
        let mut put = |rest: &mut Rect, s: String| {
            if rest.height > 0 {
                f.buffer_mut().set_stringn(rest.x, rest.y, s, w, base());
                *rest = Rect { y: rest.y + 1, height: rest.height - 1, ..*rest };
            }
        };
        let mut b: Vec<&Prediction> = a.snap.board.iter().filter(|p| on(p, &ai.model)).collect();
        let why = if !ai.enabled {
            Some(" (prediction model is OFF - switch it on in CANDLESTICK PREDICTION MODEL)")
        } else if !ai.board {
            Some(" (leaderboard OFF - click [OFF] above to predict all tracked stocks)")
        } else if b.is_empty() && a.math() {
            Some(" (strategies run in milliseconds; stocks fill in as their candles download)")
        } else if b.is_empty() {
            Some(" (predictions arrive one stock at a time: ~1-2 s each on Kronos-small, ~6-9 s on Kronos-base)")
        } else {
            None
        };
        if let Some(why) = why {
            put(&mut rest, String::new());
            put(&mut rest, why.into());
            return;
        }
        put(&mut rest, format!("  {:<3}{:<8}{:>8}{:>7}{:>17}{:>7}{:>6}  {}", "#", "SYM", "EXP%", "P(UP)", "p10 .. p90", "Z", "AGE", "NAME"));
        // MATH strategies with no signal right now go last
        b.sort_by(|x, y| (x.samples == 0).cmp(&(y.samples == 0)).then(y.exp_pct.total_cmp(&x.exp_pct)));
        let now = now_ms();
        for (i, p) in b.into_iter().enumerate() {
            if rest.height == 0 {
                break;
            }
            let rr = Rect { height: 1, ..rest };
            let mark = if self.hover(rr) || p.sym == a.snap.selected { ">" } else { " " };
            let age = now.saturating_sub(p.made_ms) / 1000;
            let age = match age { 0..120 => format!("{age}s"), 120..7200 => format!("{}m", age / 60), _ => format!("{}h", age / 3600) };
            let s = if p.samples == 0 {
                format!("{mark} {:<3}{:<8}  -- {}", i + 1, p.sym, p.signal)
            } else {
                format!("{mark} {:<3}{:<8}{:>+7.2}%{:>6.0}%{:>+8.1}..{:<+7.1}{:>+7.2}{:>6}  {}{}", i + 1, p.sym, p.exp_pct, p.p_up * 100.0, p.p10_pct, p.p90_pct, p.z, age, if p.unusual { "!! " } else { "" }, name_of(a, &p.sym))
            };
            self.hits.push((rr, Click::Sym(p.sym.clone())));
            f.buffer_mut().set_stringn(rr.x, rr.y, s, w, base());
            rest = Rect { y: rest.y + 1, height: rest.height - 1, ..rest };
        }
    }

    /// COMPARE body: highlight summary, then one row per stock with model A | model B.
    fn compare_table(&mut self, f: &mut Frame, r: Rect, c: &Cmp, ha: &[(&str, u32, u32)], hb: &[(&str, u32, u32)]) {
        let a = self.a;
        let w = r.width as usize;
        let list = |v: &[&str]| if v.is_empty() { "-".to_string() } else { v.join(", ") };
        let checked = |h: &[(&str, u32, u32)]| h.iter().map(|x| x.2).sum::<u32>();
        let lines = [
            format!(" A1-A3 AGREE NOW (both rank it high, both expect a rise): {}", list(&c.agree)),
            if c.both_right.is_empty() {
                format!(" R1-R3 BOTH RIGHT MOST: needs {MIN_CHECKED}+ checked predictions per model per stock (checked so far: A {}, B {})", checked(ha), checked(hb))
            } else {
                format!(" R1-R3 BOTH RIGHT MOST (direction right, worse of the two hit rates): {}", list(&c.both_right))
            },
            format!("  {:<5}{:<9}| A {:>7}{:>6}{:>10} | B {:>7}{:>6}{:>10} | {}", "MARK", "SYM", "EXP%", "P(UP)", "HIT", "EXP%", "P(UP)", "HIT", "NAME"),
        ];
        for (i, l) in lines.iter().enumerate() {
            if (i as u16) < r.height {
                f.buffer_mut().set_stringn(r.x, r.y + i as u16, l, w, base());
            }
        }
        let hit = |h: &[(&str, u32, u32)], s: &str| h.iter().find(|x| x.0 == s).map_or("-".into(), |x| format!("{}/{} {:.0}%", x.1, x.2, x.1 as f64 * 100.0 / x.2 as f64));
        let cell = |p: Option<&&Prediction>| match p {
            Some(p) if p.samples > 0 => format!("{:>+6.2}%{:>5.0}%", p.exp_pct, p.p_up * 100.0),
            Some(_) => format!("{:>13}", "no signal"),
            None => format!("{:>7}{:>6}", "-", "-"),
        };
        for (i, (sym, pa, pb)) in c.rows.iter().enumerate() {
            let y = r.y + 3 + i as u16;
            if y >= r.bottom() {
                break;
            }
            let rr = Rect::new(r.x, y, r.width, 1);
            let ag = c.agree.iter().position(|s| s == sym).map_or(String::new(), |k| format!("A{}", k + 1));
            let br = c.both_right.iter().position(|s| s == sym).map_or(String::new(), |k| format!("R{}", k + 1));
            let mark = if self.hover(rr) || *sym == a.snap.selected { ">" } else { " " };
            let line = format!("{mark} {:<5}{:<9}| A {} {:>9} | B {} {:>9} | {}", format!("{ag}{br}"), sym, cell(pa.as_ref()), hit(ha, sym), cell(pb.as_ref()), hit(hb, sym), name_of(a, sym));
            let lit = !ag.is_empty() || !br.is_empty();
            f.buffer_mut().set_stringn(r.x, y, format!("{line:<w$}"), w, if lit { inv() } else { base() });
            self.hits.push((rr, Click::Sym(sym.to_string())));
        }
    }

    /// Big TEST button next to the chart: paper-trade compared models A, B and the pair.
    fn test_button(&mut self, f: &mut Frame, r: Rect) {
        if r.width < 8 || r.height < 6 {
            return;
        }
        let btn = Rect::new(r.x + 1, r.y, r.width - 2, 5);
        self.big_button(f, btn, "TEST", Click::Test, false);
        let n = self.a.snap.strategies.iter().filter(|s| s.kind.ends_with("PAIR")).count();
        let lines = ["paper-trade", "model A, B", "and A+B with", "fake money", "(own window)", "", &format!("{n} test(s)"), "see F2", "CURRENT SHARES"];
        for (i, l) in lines.iter().enumerate() {
            let y = btn.bottom() + 1 + i as u16;
            if y < r.bottom() {
                f.buffer_mut().set_stringn(r.x + 1, y, l, (r.width - 1) as usize, base());
            }
        }
    }

    /// Boxed button (solid when hovered or `on`), label centred.
    fn big_button(&mut self, f: &mut Frame, r: Rect, label: &str, click: Click, on: bool) {
        let (w, h) = (r.width as usize, r.height);
        if w < 4 || h < 1 {
            return;
        }
        let solid = on || self.hover(r);
        for i in 0..h {
            let s = if solid {
                if i == h / 2 { format!("{label:^w$}") } else { " ".repeat(w) }
            } else if h >= 3 && (i == 0 || i == h - 1) {
                format!("+{}+", "-".repeat(w - 2))
            } else if i == h / 2 {
                format!("|{label:^iw$}|", iw = w - 2)
            } else {
                format!("|{}|", " ".repeat(w - 2))
            };
            f.buffer_mut().set_stringn(r.x, r.y + i, s, w, if solid { inv() } else { base() });
        }
        self.hits.push((r, click));
    }

    /// Scrolling list of lines, some clickable (CURRENT SHARES, HISTORY).
    fn click_list(&mut self, f: &mut Frame, r: Rect, title: &str, lines: Vec<(String, Option<Click>)>) {
        let a = self.a;
        let blk = panel(title, false);
        let inner = blk.inner(r);
        f.render_widget(blk, r);
        self.hits.push((inner, Click::Focus(0))); // wheel anywhere scrolls
        let top = a.sel_row.min(lines.len().saturating_sub(1));
        for (i, (l, id)) in lines.into_iter().skip(top).enumerate() {
            let y = inner.y + i as u16;
            if y >= inner.bottom() {
                break;
            }
            let rr = Rect::new(inner.x, y, inner.width, 1);
            let mark = if id.is_some() && self.hover(rr) { ">" } else { " " };
            let armed = matches!(id, Some(Click::End(x) | Click::Del(x)) if a.armed(x));
            f.buffer_mut().set_stringn(inner.x, y, format!("{mark}{l}"), inner.width as usize, if armed { inv() } else { base() });
            if let Some(c) = id {
                self.hits.push((rr, c));
            }
        }
    }

    /// A strategy's own window: F1 BUY/SELL, F2 HISTORY.
    fn strat_window(&mut self, f: &mut Frame, area: Rect) {
        let a = self.a;
        let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)]).split(area);
        let mut x = rows[0].x;
        for (i, name) in STRAT_TABS.iter().enumerate() {
            let s = format!(" F{} {name} ", i + 1);
            let rr = Rect::new(x, rows[0].y, s.len() as u16, 1);
            f.buffer_mut().set_string(x, rr.y, &s, if a.tab == i || self.hover(rr) { inv() } else { base() });
            self.hits.push((rr, Click::Tab(i)));
            x += rr.width;
            f.buffer_mut().set_string(x, rr.y, "|", base());
            x += 1;
        }
        self.status(f, rows[5]);
        let Some(s) = a.strategy() else {
            f.buffer_mut().set_string(rows[1].x, rows[1].y, format!(" waiting for strategy {} from the engine...", a.strat.unwrap_or(0)), base());
            return;
        };
        let right = format!(" {} | {} | q = close ", s.name, s.kind);
        let rw = right.chars().count() as u16;
        if x + rw < rows[0].right() {
            f.buffer_mut().set_string(rows[0].right() - rw, rows[0].y, right, base());
        }
        let q = quote_map(a);
        // accounts of the same TEST: pair + each model
        let mut accts: Vec<&Strategy> = a.snap.strategies.iter().filter(|x| x.group == s.group).collect();
        accts.sort_by_key(|x| x.id);
        let head = " accounts in this test: ";
        f.buffer_mut().set_string(rows[1].x, rows[1].y, head, base());
        let mut xx = rows[1].x + head.len() as u16;
        for acc in accts {
            let v = paper::value(acc, &q);
            let who = if acc.models.len() > 1 { "A+B PAIR".to_string() } else { short(&acc.models[0]) };
            let lbl = format!("[{who} {:+.2}%]", (v - acc.start) as f64 * 100.0 / acc.start.max(1) as f64);
            let rr = Rect::new(xx, rows[1].y, (lbl.len() as u16).min(rows[1].right().saturating_sub(xx)), 1);
            if rr.width == 0 {
                break;
            }
            f.buffer_mut().set_stringn(xx, rr.y, &lbl, rr.width as usize, if acc.id == s.id || self.hover(rr) { inv() } else { base() });
            self.hits.push((rr, Click::Strat(acc.id)));
            xx += rr.width + 1;
        }
        let v = paper::value(s, &q);
        let auto = match s.auto {
            Auto::Off => "OFF",
            Auto::TopPicks => "TOP PICKS (hold top 3 predicted rises, sell drop-outs)",
            Auto::Unusual => "UNUSUAL (buy unusual rises, sell at predicted horizon)",
            Auto::Signal => "SIGNAL (buy every stock the model signals, sell after the predicted horizon)",
        };
        let preds = format!("  | predictions {} x{} {}", TIMEFRAMES[s.tf.min(TIMEFRAMES.len() - 1)].2, s.h, ahead(s.tf.min(TIMEFRAMES.len() - 1), s.h as u32));
        if s.ended_ms > 0 {
            self.controls(f, rows[2], &[(format!(" ENDED {} | final value {} | P/L {} | ran {}{preds}", when(s.ended_ms), money(v), pl(v - s.start, s.start), age(s.ended_ms - s.created_ms)), None)]);
        } else {
            let end = if a.armed(s.id) { "CLICK AGAIN TO TERMINATE" } else { "TERMINATE" };
            self.controls(f, rows[2], &[
                (format!(" {} | value {} | P/L {} | cash {} | in shares {} | AUTO ", s.mode, money(v), pl(v - s.start, s.start), money(s.cash), money(v - s.cash)), None),
                (auto.into(), Some(10)),
                (preds, None),
                ("  | SIZE ".into(), None),
                (if s.vol_target { "VOL TARGET" } else { "EQUAL $" }.into(), Some(17)),
                ("  ".into(), None),
                (end.into(), Some(16)),
            ]);
            let pct = |v: u32| if v == 0 { "OFF".to_string() } else { format!("{v}%") };
            let cooling: Vec<String> = s.cooldown.iter().map(|c| format!("{} {}", c.0, age(c.1 - now_ms()))).collect();
            self.controls(f, rows[3], &[
                (" RISK (any AUTO mode, also OFF): STOP-LOSS ".into(), None),
                (pct(s.stop_pct), Some(18)),
                ("  TRAILING STOP ".into(), None),
                (pct(s.trail_pct), Some(19)),
                ("  MAX HOLD ".into(), None),
                (if s.max_hold { format!("ON = sell after {} candles", s.h) } else { "OFF".into() }, Some(20)),
                (if cooling.is_empty() { String::new() } else { format!("  | no AUTO re-buy until cooled: {}", cooling.join(", ")) }, None),
            ]);
        }
        if a.tab == 0 { self.trade_tab(f, rows[4], s, &q) } else { self.history_tab(f, rows[4], s) }
    }

    fn trade_tab(&mut self, f: &mut Frame, r: Rect, s: &Strategy, q: &HashMap<String, Quote>) {
        let a = self.a;
        let [l, rt] = Layout::horizontal([Constraint::Percentage(55), Constraint::Min(0)]).areas(r);
        // STOCKS: tracked stocks with this strategy's model predictions
        let blk = panel("STOCKS - click to pick, wheel/up/down to move", false);
        let inner = blk.inner(l);
        f.render_widget(blk, l);
        let w = inner.width as usize;
        let models: String = s.models.iter().map(|m| format!("| {:<19}", format!("{} EXP% P(UP)", short(m).chars().take(9).collect::<String>()))).collect();
        f.buffer_mut().set_stringn(inner.x, inner.y, format!("  {:<9}{:>11}  {models}| {:>12}  NAME", "SYM", "PRICE", "HELD"), w, base());
        for (i, sym) in a.strat_syms().iter().enumerate() {
            let y = inner.y + 1 + i as u16;
            if y >= inner.bottom() {
                break;
            }
            let rr = Rect::new(inner.x, y, inner.width, 1);
            let price = q.get(sym).map_or("-".into(), |x| px(x.price));
            let cols: String = s.models.iter().map(|m| match pred_of(a, s, m, sym) {
                Some(p) => format!("| {:>+7.2}% {:>5.0}%{:<4}", p.exp_pct, p.p_up * 100.0, if p.unusual { " !!" } else { "" }),
                None => format!("| {:<19}", "  (predicting...)"),
            }).collect();
            let held = s.positions.iter().find(|p| &p.sym == sym).map_or(String::new(), |p| money(q.get(sym).map_or(p.cost, |x| p.micro * paper::cents(x) / SHARE)));
            let mark = if self.hover(rr) { ">" } else { " " };
            let line = format!("{mark} {sym:<9}{price:>11}  {cols}| {held:>12}  {}", name_of(a, sym));
            f.buffer_mut().set_stringn(inner.x, y, format!("{line:<w$}"), w, if *sym == a.sym_sel { inv() } else { base() });
            self.hits.push((rr, Click::Sym(sym.clone())));
        }
        let [order, pos, trades] = Layout::vertical([Constraint::Length(12), Constraint::Min(4), Constraint::Length(9)]).areas(rt);
        // ORDER
        let blk = panel("ORDER (paper money)", false);
        let oi = blk.inner(order);
        f.render_widget(blk, order);
        let put = |f: &mut Frame, y: u16, s: String| {
            if y < oi.bottom() {
                f.buffer_mut().set_stringn(oi.x, y, s, oi.width as usize, base());
            }
        };
        let sym = &a.sym_sel;
        let price = q.get(sym).map(paper::cents);
        put(f, oi.y, if sym.is_empty() { " stock   (click one on the left)".into() } else { format!(" stock   {sym}  {}  price {}", name_of(a, sym), price.map_or("-".into(), money)) });
        let v = paper::value(s, q);
        let cap = v * a.settings.max_position_pct as i64 / 100;
        if oi.height > 2 {
            self.controls(f, Rect { y: oi.y + 1, height: 1, ..oi }, &[(" amount  ".into(), None), (" - ".into(), Some(11)), (format!(" {} ", money(a.amount)), None), (" + ".into(), Some(12)), (format!("   max {}% per stock = {} | daily loss stop -{}%", a.settings.max_position_pct, money(cap), a.settings.daily_loss_stop_pct), None)]);
        }
        if oi.height >= 6 && oi.width >= 50 {
            let bw = 14;
            for (k, (lbl, id)) in [("BUY", 13u8), ("SELL", 14), ("SELL ALL", 15)].into_iter().enumerate() {
                self.big_button(f, Rect::new(oi.x + 1 + k as u16 * (bw + 2), oi.y + 2, bw, 3), lbl, Click::Ctl(id), false);
            }
        }
        if let Some(p) = s.positions.iter().find(|p| &p.sym == sym) {
            let mv = price.map_or(p.cost, |c| p.micro * c / SHARE);
            put(f, oi.y + 6, format!(" held    {:.4} sh  cost {}  value {}  P/L {}", p.micro as f64 / SHARE as f64, money(p.cost), money(mv), pl(mv - p.cost, p.cost)));
        }
        put(f, oi.y + 7, format!(" engine  {}", if s.note.is_empty() { "-" } else { &s.note }));
        put(f, oi.y + 9, " keys    b buy | s sell | + - amount | up/down stock | F2 history | q close".into());
        // POSITIONS
        let blk = panel("POSITIONS (click = pick)", false);
        let pi = blk.inner(pos);
        f.render_widget(blk, pos);
        let pw = pi.width as usize;
        f.buffer_mut().set_stringn(pi.x, pi.y, format!("  {:<9}{:>12}{:>13}{:>11}{:>13}{:>22}  {:<14}{:<14}{}", "SYM", "SHARES", "COST", "NOW", "VALUE", "P/L", "PRED AT BUY", "PRED NOW", "HELD"), pw, base());
        let now = now_ms();
        for (i, p) in s.positions.iter().enumerate() {
            let y = pi.y + 1 + i as u16;
            if y >= pi.bottom() {
                break;
            }
            let rr = Rect::new(pi.x, y, pi.width, 1);
            let c = q.get(&p.sym).map(paper::cents);
            let mv = c.map_or(p.cost, |c| p.micro * c / SHARE);
            let pn = s.models.first().and_then(|m| pred_of(a, s, m, &p.sym)).map_or("-".into(), |x| format!("{:+.2}% {:.0}%", x.exp_pct, x.p_up * 100.0));
            let line = format!("{} {:<9}{:>12.4}{:>13}{:>11}{:>13}{:>22}  {:<14}{:<14}{}", if self.hover(rr) { ">" } else { " " }, p.sym, p.micro as f64 / SHARE as f64, money(p.cost), c.map_or("-".into(), money), money(mv), pl(mv - p.cost, p.cost), format!("{:+.2}% {:.0}%", p.exp_pct, p.p_up * 100.0), pn, age(now - p.opened_ms));
            f.buffer_mut().set_stringn(pi.x, y, line, pw, base());
            self.hits.push((rr, Click::Sym(p.sym.clone())));
        }
        if s.positions.is_empty() {
            f.buffer_mut().set_stringn(pi.x, pi.y + 1, "  (nothing held)", pw, base());
        }
        // TRADES, newest first
        let lines: Vec<String> = s.trades.iter().rev().map(|t| {
            let when = chrono::DateTime::from_timestamp_millis(t.t_ms).map_or(String::new(), |d| d.with_timezone(&chrono::Local).format("%m-%d %H:%M:%S").to_string());
            format!(" {when}  {:<4} {:<9}{:>12.4} sh @ {:>10} = {:>12}  pred {:+.2}% {:.0}%  {}", if t.buy { "BUY" } else { "SELL" }, t.sym, t.micro as f64 / SHARE as f64, money(t.price), money(t.cents), t.exp_pct, t.p_up * 100.0, t.why)
        }).collect();
        self.text(f, trades, "TRADES (newest first)", false, if lines.is_empty() { vec![" (no trades yet)".into()] } else { lines });
    }

    fn history_tab(&mut self, f: &mut Frame, r: Rect, s: &Strategy) {
        let a = self.a;
        let pts = &s.history;
        let n = pts.len();
        if n == 0 {
            self.text(f, r, "HISTORY", false, vec![" (no history yet - points start with the first trade)".into()]);
            return;
        }
        let sel = a.hist_sel.min(n - 1); // 0 = newest
        let idx = n - 1 - sel;
        let [chart, table] = Layout::vertical([Constraint::Percentage(45), Constraint::Min(4)]).areas(r);
        // chart: one '•' per point; x = point number (time is log-spaced), '-' = starting money
        let blk = panel("P/L OVER TIME  (x = recorded points: dense early, sparser as the trade ages | '-' = start | '┆' = picked point)", false);
        let ci = blk.inner(chart);
        f.render_widget(blk, chart);
        self.hits.push((ci, Click::Hist(sel)));
        const AXIS: u16 = 15;
        if ci.width > AXIS + 4 && ci.height >= 3 {
            let w = (ci.width - AXIS) as usize;
            let hi = pts.iter().map(|p| p.value).max().unwrap().max(s.start);
            let lo = pts.iter().map(|p| p.value).min().unwrap().min(s.start);
            let span = (hi - lo).max(1) as f64;
            let row = |v: i64| ci.y + (((hi - v) as f64 / span) * (ci.height - 1) as f64).round() as u16;
            let col = |i: usize| if n <= 1 { 0 } else { i * (w - 1) / (n - 1) };
            let buf = f.buffer_mut();
            for x in 0..w {
                buf.set_string(ci.x + x as u16, row(s.start), "-", base());
            }
            for y in ci.y..ci.bottom() {
                buf.set_string(ci.x + col(idx) as u16, y, "┆", base());
            }
            for (i, p) in pts.iter().enumerate() {
                buf.set_string(ci.x + col(i) as u16, row(p.value), "•", base());
            }
            let ax = ci.x + w as u16 + 1;
            for (y, v) in [(ci.y, hi), (row(s.start), s.start), (ci.bottom() - 1, lo)] {
                buf.set_stringn(ax, y, money(v), (AXIS - 1) as usize, base());
            }
        }
        // table, newest first
        let blk = panel("HISTORY - wheel/up/down/click to move | a point every clamp(age / 60, 10 s, 12 h) since the first trade, + one per trade", false);
        let ti = blk.inner(table);
        f.render_widget(blk, table);
        self.hits.push((ti, Click::Hist(sel)));
        let tw = ti.width as usize;
        f.buffer_mut().set_stringn(ti.x, ti.y, format!("  {:<6}{:<21}{:>10}{:>15}{:>24}{:>14}{:>15}", "#", "TIME", "AGE", "VALUE", "P/L", "CHANGE", "CASH"), tw, base());
        let vis = ti.height.saturating_sub(1) as usize;
        let top = sel.saturating_sub(vis.saturating_sub(1));
        let first_t = s.trades.first().map_or(pts[0].t_ms, |t| t.t_ms);
        for k in top..n {
            let y = ti.y + 1 + (k - top) as u16;
            if y >= ti.bottom() {
                break;
            }
            let p = pts[n - 1 - k];
            let prev = if n - 1 - k > 0 { pts[n - 2 - k].value } else { s.start };
            let when = chrono::DateTime::from_timestamp_millis(p.t_ms).map_or(String::new(), |d| d.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M:%S").to_string());
            let line = format!("  {:<6}{:<21}{:>10}{:>15}{:>24}{:>14}{:>15}", n - k, when, age(p.t_ms - first_t), money(p.value), pl(p.value - s.start, s.start), signed(p.value - prev), money(p.cash));
            let rr = Rect::new(ti.x, y, ti.width, 1);
            f.buffer_mut().set_stringn(ti.x, y, format!("{line:<tw$}"), tw, if k == sel || self.hover(rr) { inv() } else { base() });
            self.hits.push((rr, Click::Hist(k)));
        }
    }

    fn backtest(&mut self, f: &mut Frame, r: Rect) {
        let [l, rt] = Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)]).areas(r);
        let [eq, res] = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(rt);
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let p = |k: &str, v: &str| format!(" {k:<14}[{v}]");
        self.text(f, l, "BACKTEST - PARAMETERS", false, vec![
            p("Strategy", "none"),
            p("Symbols", &self.a.snap.selected),
            p("From", "----------"),
            p("To", "----------"),
            p("Bars", "1m | 1d"),
            p("Sweep", "none"),
            p("Commission", "-"),
            p("Slippage", "-"),
            format!(" {:<14}{cores}", "CPU cores"),
            String::new(),
            " [ RUN ]  disabled - phase 1 is display only".into(),
        ]);
        self.text(f, eq, "EQUITY CURVE", false, vec![" (no runs yet)".into()]);
        self.text(f, res, "RESULTS - every combination tried", false, vec![
            format!(" {:<24}{:>10}{:>10}{:>8}{:>8}", "PARAMS", "RETURN", "MAX DD", "WIN%", "TRADES"),
            " (no runs yet)".into(),
        ]);
    }

    fn data(&mut self, f: &mut Frame, r: Rect) {
        let [l, rt] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(r);
        let a = self.a;
        self.choices(f, l, "DATA SOURCE (change = engine restart)", DATA_SOURCES, &a.settings.data_source, 0);
        self.choices(f, rt, "EXECUTION (change = engine restart)", EXECUTION, &a.settings.execution, DATA_SOURCES.len());
    }

    fn choices(&mut self, f: &mut Frame, r: Rect, title: &str, list: &[(&str, &str, bool)], cur: &str, first: usize) {
        let a = self.a;
        let blk = panel(title, false);
        let inner = blk.inner(r);
        f.render_widget(blk, r);
        for (i, (id, label, built)) in list.iter().enumerate() {
            let y = inner.y + i as u16;
            if y >= inner.bottom() {
                break;
            }
            let rr = Rect::new(inner.x, y, inner.width, 1);
            let mark = if a.sel_row == first + i || self.hover(rr) { ">" } else { " " };
            let on = if *id == cur { "[*]" } else { "[ ]" };
            let na = if *built { "" } else { "[N/A]" };
            f.buffer_mut().set_stringn(inner.x, y, format!("{mark} {on} {label:<26}{na}"), inner.width as usize, base());
            self.hits.push((rr, Click::Row(first + i)));
        }
        let y = inner.y + list.len() as u16 + 1;
        if y < inner.bottom() {
            f.buffer_mut().set_stringn(inner.x, y, "  Enter/click to choose. [N/A] = not built yet.", inner.width as usize, base());
        }
    }

    fn storage(&mut self, f: &mut Frame, r: Rect) {
        let s = &self.a.storage;
        let mut v = vec![format!(" {:<24}{:>12}{:>12}{:>12}{:>7}", "DISK", "TOTAL", "USED", "FREE", "USED%")];
        for (m, total, avail) in &s.disks {
            let used = total.saturating_sub(*avail);
            v.push(format!(" {:<24}{:>12}{:>12}{:>12}{:>6}%", m, bytes(*total), bytes(used), bytes(*avail), used * 100 / (*total).max(1)));
        }
        let dir = data_dir();
        v.push(String::new());
        v.push(format!(" OUR DATA  {}", dir.display()));
        for (name, size) in &s.data {
            v.push(format!("   {name:<30}{:>12}", bytes(*size)));
        }
        v.push(format!("   {:<30}{:>12}", "TOTAL", bytes(s.total)));
        v.push(String::new());
        // free space on the disk holding our data = longest mount point that prefixes the data dir
        let free = s.disks.iter().filter(|d| dir.starts_with(&d.0)).max_by_key(|d| d.0.len()).map(|d| d.2);
        match s.first {
            Some((t0, b0)) if t0.elapsed().as_secs() >= 60 && s.total > b0 => {
                let per_day = (s.total - b0) as f64 / t0.elapsed().as_secs_f64() * 86_400.0;
                v.push(format!(" growth      {}/day (measured since this tab first opened)", bytes(per_day as u64)));
                if let Some(free) = free {
                    v.push(format!(" disk full in ~{:.0} days at this rate", free as f64 / per_day));
                }
            }
            _ => v.push(" growth      measuring... (needs a minute of data coming in)".into()),
        }
        self.text(f, r, "STORAGE", false, v);
    }

    fn maths(&mut self, f: &mut Frame, r: Rect) {
        let lines = maths_lines(self.a);
        let p = Paragraph::new(lines.into_iter().map(Line::from).collect::<Vec<_>>())
            .block(panel("MATHS LOGBOOK (Up/Down/wheel scroll)", false))
            .style(base())
            .scroll((self.a.sel_row as u16, 0));
        f.render_widget(p, r);
        self.hits.push((r, Click::Focus(self.a.focus))); // so the wheel scrolls; clicks change nothing
    }

    fn settings(&mut self, f: &mut Frame, r: Rect) {
        let a = self.a;
        let [l, rt] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(r);
        let blk = panel("SETTINGS (Enter/click toggles)", false);
        let inner = blk.inner(l);
        f.render_widget(blk, l);
        for i in 0..SETTING_ROWS {
            let y = inner.y + i as u16;
            if y >= inner.bottom() {
                break;
            }
            let rr = Rect::new(inner.x, y, inner.width, 1);
            let (k, v) = setting_row(&a.settings, i);
            let mark = if a.sel_row == i || self.hover(rr) { ">" } else { " " };
            f.buffer_mut().set_stringn(inner.x, y, format!("{mark} {k:<22}{v}"), inner.width as usize, base());
            self.hits.push((rr, Click::Row(i)));
        }
        let s = &a.settings;
        self.text(f, rt, "ACCOUNT & FILES (edit settings.json)", false, vec![
            format!(" Paper start cash   {}", money(s.start_cash_cents)),
            format!(" Max per position   {}%", s.max_position_pct),
            format!(" Daily loss stop    -{}%", s.daily_loss_stop_pct),
            format!(" Poll interval      {} ms", s.poll_ms),
            format!(" Watchlist          {} symbols", s.watchlist.len()),
            String::new(),
            format!(" {}", data_dir().join("settings.json").display()),
            " Mode / source / execution changes restart the engine.".into(),
        ]);
    }

    fn help(&mut self, f: &mut Frame, area: Rect) {
        let w = 66.min(area.width);
        let h = 24.min(area.height);
        let r = Rect::new(area.x + (area.width - w) / 2, area.y + (area.height - h) / 2, w, h);
        f.render_widget(Clear, r);
        let l = [
            "F1-F9          switch tab (or click the tab)",
            ":              command bar (typing also filters the picker)",
            "?              this help",
            "Tab/Shift-Tab  move focus between MAIN panels",
            "Space          maximise focused panel / restore",
            "Wheel / + -    over chart: zoom timeframe 1 MIN..1 MONTH",
            "Esc            restore / cancel command",
            "Up/Down Enter  move + choose in lists",
            "Mouse          click tabs, symbols, rows; wheel scrolls",
            "               the picker and lists",
            "",
            "COMMANDS",
            ":AAPL          chart a symbol",
            ":mode signals  signals only    :mode paper  auto paper trades",
            ":static        no motion       :lively      motion on",
            ":alert         test notification",
            ":stop          stop the engine :start       start it",
            ":q             close window (engine keeps running)",
        ];
        self.text(f, r, "HELP - any key closes", true, l.iter().map(|s| format!(" {s}")).collect());
    }
}

/// Built-in strategy library (quant.rs): every listed strategy, explained, and where it runs in the app.
fn library_lines() -> Vec<String> {
    let mut v = vec![
        " ==================== STRATEGY LIBRARY (built in) ====================".to_string(),
        " RUNS = pick it in F7 MATH FORECAST (chart overlay, leaderboard, COMPARE, TEST paper trading).".into(),
        " PATTERN rules forecast with what happened after every earlier time the rule fired on that stock".into(),
        " (no lookahead); MODEL strategies fit their equation and simulate 500 paths. N/A = data this app lacks.".into(),
    ];
    let mut cat = "";
    for q in crate::quant::STRATS {
        if q.cat != cat {
            cat = q.cat;
            v.extend([String::new(), format!(" ---- {cat} ----")]);
        }
        let kind = match q.run {
            crate::quant::Run::Pattern(_) => "RUNS (pattern rule)".to_string(),
            crate::quant::Run::Model(_) => "RUNS (fitted model, simulated)".to_string(),
            crate::quant::Run::Rule(_) => "RUNS (rule using other stocks / accounts / learning)".to_string(),
            crate::quant::Run::CtxModel(_) => "RUNS (fitted model on factor ETFs, simulated)".to_string(),
            crate::quant::Run::Sizing => "RUNS as a SIZE rule in strategy windows (not a forecast)".to_string(),
            crate::quant::Run::Na(why) => format!("N/A - {why}"),
        };
        v.push(format!(" == {} ==   [{kind}]", q.name));
        v.push(format!("   formula     {}", q.formula));
        v.push(format!("   what        {}", q.what));
        v.push(format!("   example     {}", q.example));
        v.push(format!("   used for    {}", q.used));
        if q.runnable() {
            v.push("   in the app  F7 MATH FORECAST > STRATEGY list".into());
        } else if matches!(q.run, crate::quant::Run::Sizing) {
            v.push("   in the app  strategy window (TEST) > SIZE [VOL TARGET]".into());
        }
        v.push(String::new());
    }
    v
}

pub fn maths_lines(a: &App) -> Vec<String> {
    let mut v = vec![format!(" file: {}  (edit it to add entries; reopen tab to reload)", data_dir().join("maths.json").display()), String::new()];
    match &a.maths {
        Err(e) => v.push(format!(" {e}")),
        Ok(list) if list.is_empty() => v.push(" (empty)".into()),
        Ok(list) => {
            for m in list {
                v.push(format!(" == {} ==   added {}", m.name, m.added));
                v.push(format!("   equation    {}", m.equation));
                v.push(format!("   part of     {}", m.part_of));
                v.push(format!("   applies to  {}", m.applies_to));
                v.push("   contributes".into());
                v.extend(m.contributes.iter().map(|c| format!("     - {c}")));
                if !m.notes.is_empty() {
                    v.push(format!("   notes       {}", m.notes));
                }
                v.push(String::new());
            }
        }
    }
    v.push(String::new());
    v.extend(library_lines());
    v
}

/// Date/time format that suits the timeframe (intraday shows time, longer shows dates).
fn time_label(t: i64, tf: usize) -> String {
    let fmt = match tf {
        0 => "%H:%M",
        1..=3 => "%b%d %H:%M",
        4 => "%Y-%m-%d",
        _ => "%Y-%m",
    };
    chrono::DateTime::from_timestamp(t, 0).map_or(String::new(), |d| d.format(fmt).to_string())
}

fn quote_map(a: &App) -> HashMap<String, Quote> {
    a.snap.quotes.iter().map(|q| (q.sym.clone(), q.clone())).collect()
}

/// A strategy's model prediction for a stock, at the strategy's own timeframe/horizon.
fn pred_of<'a>(a: &'a App, s: &Strategy, m: &str, sym: &str) -> Option<&'a Prediction> {
    a.snap.board.iter().find(|p| p.model == m && p.sym == sym && p.tf == s.tf && p.steps.len() == s.h)
}

fn signed(c: i64) -> String {
    format!("{}{}", if c >= 0 { "+" } else { "" }, money(c))
}

/// "+$12.34 (+1.23%)" against `base` cents.
fn pl(c: i64, base: i64) -> String {
    format!("{} ({:+.2}%)", signed(c), c as f64 * 100.0 / base.max(1) as f64)
}

/// "45s", "12m", "3h 05m", "4d 06h"
fn age(ms: i64) -> String {
    let s = ms.max(0) / 1000;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m", s / 60),
        3600..86_400 => format!("{}h {:02}m", s / 3600, s / 60 % 60),
        _ => format!("{}d {:02}h", s / 86_400, s / 3600 % 24),
    }
}

/// CURRENT SHARES lines; Some(id) = clicking opens that strategy's window.
/// Local date-time of a ms timestamp.
fn when(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms).map_or("?".into(), |d| d.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M").to_string())
}

/// F3 HISTORY: terminated runs, newest first, with their final results.
pub fn history_lines(a: &App) -> Vec<(String, Option<Click>)> {
    let q = quote_map(a);
    let mut ss: Vec<&Strategy> = a.snap.strategies.iter().filter(|s| s.ended_ms > 0).collect();
    if ss.is_empty() {
        return vec![(" (no finished runs yet - TERMINATE a strategy in F2 CURRENT SHARES or its window)".into(), None)];
    }
    ss.sort_by_key(|s| -s.ended_ms);
    let (tv, ts): (i64, i64) = ss.iter().map(|s| (paper::value(s, &q), s.start)).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let mut v = vec![(format!(" {} finished runs | total final value {} | total P/L {}", ss.len(), money(tv), pl(tv - ts, ts)), None), (String::new(), None)];
    for s in ss {
        let val = paper::value(s, &q);
        let id = Some(Click::Strat(s.id));
        let tf = s.tf.min(TIMEFRAMES.len() - 1);
        let models: Vec<String> = s.models.iter().map(|m| short(m)).collect();
        let buys = s.trades.iter().filter(|t| t.buy).count();
        let peak = s.history.iter().map(|h| h.value).max().unwrap_or(s.start);
        let low = s.history.iter().map(|h| h.value).min().unwrap_or(s.start);
        v.push((format!(" {:<44} {:<11} {:<6} start {:>14}  final {:>14}  P/L {:>24}", s.name, s.kind, s.mode, money(s.start), money(val), pl(val - s.start, s.start)), id.clone()));
        v.push((format!("   {} -> {} (ran {}) | models {} | {} x{} {}", when(s.created_ms), when(s.ended_ms), age(s.ended_ms - s.created_ms), models.join(" + "), TIMEFRAMES[tf].2, s.h, ahead(tf, s.h as u32)), id.clone()));
        v.push((format!("   {} trades ({buys} buys, {} sells) | best value {} | worst value {}", s.trades.len(), s.trades.len() - buys, money(peak), money(low)), id));
        v.push((format!("   [{}]", if a.armed(s.id) { "CLICK AGAIN TO DELETE FOREVER" } else { "DELETE THIS RUN" }), Some(Click::Del(s.id))));
        v.push((String::new(), None));
    }
    v
}

pub fn shares_lines(a: &App) -> Vec<(String, Option<Click>)> {
    let q = quote_map(a);
    let ss: Vec<&Strategy> = a.snap.strategies.iter().filter(|s| s.ended_ms == 0).collect();
    if ss.is_empty() {
        return vec![(" (no running strategies - press TEST next to the chart in F6 AI FORECAST or F7 MATH FORECAST; finished runs are in F3 HISTORY)".into(), None)];
    }
    let (tv, ts): (i64, i64) = ss.iter().map(|s| (paper::value(s, &q), s.start)).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let mut v = vec![(format!(" {} strategies | total value {} | total P/L {}", ss.len(), money(tv), pl(tv - ts, ts)), None), (String::new(), None)];
    let now = now_ms();
    for s in ss {
        let val = paper::value(s, &q);
        let auto = match s.auto {
            Auto::Off => "OFF",
            Auto::TopPicks => "TOP PICKS",
            Auto::Unusual => "UNUSUAL",
            Auto::Signal => "SIGNAL",
        };
        let id = Some(Click::Strat(s.id));
        v.push((format!(" {:<44} {:<8} {:<6} AUTO {:<10} value {:>14}  P/L {:>24}  cash {:>14}  {} stock(s)  running {}", s.name, s.kind, s.mode, auto, money(val), pl(val - s.start, s.start), money(s.cash), s.positions.len(), age(now - s.created_ms)), id.clone()));
        let tf = s.tf.min(TIMEFRAMES.len() - 1);
        let models: Vec<String> = s.models.iter().map(|m| short(m)).collect();
        v.push((format!("   models {} | predictions {} x{} {} | {} trades{}", models.join(" + "), TIMEFRAMES[tf].2, s.h, ahead(tf, s.h as u32), s.trades.len(), if s.note.is_empty() { String::new() } else { format!(" | last: {}", s.note) }), id.clone()));
        for p in &s.positions {
            let c = q.get(&p.sym).map(paper::cents);
            let mv = c.map_or(p.cost, |c| p.micro * c / SHARE);
            let pn = s.models.first().and_then(|m| pred_of(a, s, m, &p.sym)).map_or("-".into(), |x| format!("{:+.2}% P(up) {:.0}%", x.exp_pct, x.p_up * 100.0));
            v.push((format!("     {:<9}{:>12.4} sh  bought {:>13} @ {:>11}  now {:>11}  value {:>13}  P/L {:>24}  pred at buy {:+.2}% P(up) {:.0}%  now {pn}  held {}", p.sym, p.micro as f64 / SHARE as f64, money(p.cost), money(p.cost * SHARE / p.micro.max(1)), c.map_or("-".into(), money), money(mv), pl(mv - p.cost, p.cost), p.exp_pct, p.p_up * 100.0, age(now - p.opened_ms)), id.clone()));
        }
        v.push((format!("   [{}]", if a.armed(s.id) { "CLICK AGAIN TO TERMINATE (sells everything at market)" } else { "TERMINATE" }), Some(Click::End(s.id))));
        v.push((String::new(), None));
    }
    v
}

fn name_of<'a>(a: &'a App, sym: &str) -> &'a str {
    a.snap.quotes.iter().find(|q| q.sym == sym).map_or("", |q| q.name.as_str())
}

/// Short model name for tight lines: "Kronos-base", "Black-Scholes Monte Carlo".
fn short(id: &str) -> String {
    model_label(id)
}

/// "Both right most" needs at least this many checked predictions per model per stock.
pub const MIN_CHECKED: u32 = 5;

/// COMPARE view of two models' leaderboards.
pub struct Cmp<'a> {
    /// every stock, best agreement first: (symbol, model A, model B)
    pub rows: Vec<(&'a str, Option<&'a Prediction>, Option<&'a Prediction>)>,
    /// top 3 both models rank high right now, both expecting a rise with P(up) >= 50%
    pub agree: Vec<&'a str>,
    /// top 3 by the worse of the two models' direction hit rates (MIN_CHECKED+ each)
    pub both_right: Vec<&'a str>,
    /// Spearman rank correlation of the two models' expected moves (3+ shared stocks)
    pub rho: Option<f64>,
}

/// Rows in ai::agreement order, plus each model's leftovers; highlight picks.
pub fn compare<'a>(pa: &[&'a Prediction], pb: &[&'a Prediction], ha: &[(&str, u32, u32)], hb: &[(&str, u32, u32)]) -> Cmp<'a> {
    let find = |v: &[&'a Prediction], s: &str| v.iter().find(|p| p.sym == s).copied();
    let (pairs, rho) = crate::ai::agreement(pa, pb);
    let agree = pairs.iter().filter(|(x, y)| x.exp_pct > 0.0 && y.exp_pct > 0.0 && x.p_up >= 0.5 && y.p_up >= 0.5).take(3).map(|x| x.0.sym.as_str()).collect();
    let mut rows: Vec<(&'a str, Option<&'a Prediction>, Option<&'a Prediction>)> = pairs.iter().map(|(x, y)| (x.sym.as_str(), Some(*x), Some(*y))).collect();
    rows.extend(pa.iter().filter(|x| find(pb, &x.sym).is_none()).map(|x| (x.sym.as_str(), Some(*x), None)));
    rows.extend(pb.iter().filter(|y| find(pa, &y.sym).is_none()).map(|y| (y.sym.as_str(), None, Some(*y))));
    let rate = |h: &[(&str, u32, u32)], s: &str| h.iter().find(|x| x.0 == s && x.2 >= MIN_CHECKED).map(|x| (x.1 as f64 / x.2 as f64, x.2));
    let mut br: Vec<(&'a str, f64, u32)> = rows.iter().filter_map(|r| Some((r.0, rate(ha, r.0)?, rate(hb, r.0)?))).map(|(s, x, y)| (s, x.0.min(y.0), x.1.min(y.1))).collect();
    br.sort_by(|x, y| y.1.total_cmp(&x.1).then(y.2.cmp(&x.2)));
    Cmp { rows, agree, both_right: br.into_iter().take(3).map(|x| x.0).collect(), rho }
}

/// How far ahead `h` candles of timeframe `tf` reach. Intraday candles only exist while the market trades.
fn ahead(tf: usize, h: u32) -> String {
    let (n, unit) = match tf {
        0 => (h, "min"),
        1 => (5 * h, "min"),
        2 => (15 * h, "min"),
        3 => (h, "hour"),
        4 => (h, "trading day"),
        5 => (h, "week"),
        _ => (h, "month"),
    };
    let (n, unit) = if unit == "min" && n >= 60 && n % 60 == 0 { (n / 60, "hour") } else { (n, unit) };
    let s = if n == 1 { "" } else { "s" };
    let trading = if tf <= 3 { " of trading" } else { "" };
    format!("(= {n} {unit}{s}{trading} ahead)")
}

/// PREDICTION OUTPUT panel: the forecast for the charted symbol and how unusual it is.
fn prediction_lines(a: &App) -> Vec<String> {
    let ai = a.desk();
    let label = if ai.model.is_empty() { "none picked".to_string() } else { model_label(&ai.model) };
    let status = if a.math() { &a.snap.math_status } else { &a.snap.ai_status };
    let mut v = vec![
        format!(" model   {label} [{}]", model_status(&ai.model)),
        format!(" status  {}", if ai.enabled { status.as_str() } else { "OFF" }),
    ];
    if let Some(q) = crate::quant::find(&ai.model) {
        v.push(format!(" rule    {}", q.formula));
    }
    let off = a.snap.session.map_or(-18_000, |s| s.gmtoffset);
    let Some(p) = a.single_pred() else {
        v.extend([String::new(), " (no prediction for this symbol yet)".into()]);
        return v;
    };
    if !p.signal.is_empty() {
        v.push(format!(" signal  {}", p.signal));
    }
    if p.samples == 0 {
        v.extend([String::new(), format!(" no forecast for {} right now - the rule only forecasts on candles where it fires.", p.sym), " the LEADERBOARD shows which tracked stocks have a signal now.".into()]);
        return v;
    }
    let h = p.steps.len();
    let conf = ai.show_conf;
    v.push(format!(" input   {} {} | last {} candles | {} samples | made {} ET in {} ms", p.sym, TIMEFRAMES[p.tf].2, p.lookback, p.samples, hms(p.made_ms / 1000 + off), p.ms));
    v.push(format!(" now     {}   ->   in {h} candles mean close {}  ({:+.2}%)", px(p.last_close), px(p.steps[h - 1].c), p.exp_pct));
    if conf {
        v.push(format!(" P(up)   {:.0}%   middle 80% of samples: {:+.2}% .. {:+.2}%", p.p_up * 100.0, p.p10_pct, p.p90_pct));
    }
    v.push(format!(" vs usual moves: typical {h}-candle move +-{:.2}%  ->  z {:+.2}", p.sigma_pct, p.z));
    v.push(if p.unusual {
        format!(" !! UNUSUAL RISE (z >= {UNUSUAL_Z} and P(up) >= {}%)", ai.min_conf)
    } else {
        format!(" normal (unusual rise = z >= {UNUSUAL_Z} and P(up) >= {}%)", ai.min_conf)
    });
    v.push(String::new());
    v.push(format!(" {:<5}{:<12}{:>10}{:>10}{:>10}{:>10}   {:<21}{}", "STEP", "TIME ET", "OPEN", "HIGH", "LOW", "CLOSE", "CLOSE RANGE", if conf { "P(UP)" } else { "" }));
    for (i, s) in p.steps.iter().enumerate() {
        let pu = if conf { format!("{:.0}%", s.p_up * 100.0) } else { String::new() };
        v.push(format!(" +{:<4}{:<12}{:>10}{:>10}{:>10}{:>10}   {:>9} .. {:<9}{pu}", i + 1, time_label(s.t + off, p.tf), px(s.o), px(s.h), px(s.l), px(s.c), px(s.c_lo), px(s.c_hi)));
    }
    v
}

fn strat_lines() -> Vec<String> {
    vec![" (none yet)".into(), String::new(), " phase 1 = display only.".into(), " strategies get added later.".into()]
}

fn position_lines(a: &App) -> Vec<String> {
    let s = &a.settings;
    let exec = EXECUTION.iter().find(|e| e.0 == s.execution).map_or("?", |e| e.1);
    vec![
        format!(" cash        {}", money(s.start_cash_cents)),
        " positions   (none)".into(),
        String::new(),
        format!(" mode        {}", s.mode),
        format!(" execution   {exec}"),
        format!(" limits      max {}% per position | stop day at -{}%", s.max_position_pct, s.daily_loss_stop_pct),
    ]
}

fn session_label(a: &App, now: i64) -> String {
    let Some(s) = a.snap.session else { return "SESSION ?".into() };
    let inr = |w: (i64, i64)| now >= w.0 && now < w.1;
    let st = &a.settings;
    let (name, traded) = if inr(s.pre) {
        ("PRE", st.trade_pre)
    } else if inr(s.regular) {
        ("OPEN", st.trade_regular)
    } else if inr(s.post) {
        ("AFTER", st.trade_post)
    } else {
        return "CLOSED".into();
    };
    if traded { name.into() } else { format!("{name} (not traded)") }
}

fn is_stock(sym: &str) -> bool {
    !sym.contains(['=', '^']) && !sym.ends_with("-USD")
}

fn px(p: f64) -> String {
    if p.abs() < 10.0 { format!("{p:.4}") } else { format!("{p:.2}") }
}

fn hms(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0).map_or("--:--:--".into(), |d| d.format("%H:%M:%S").to_string())
}

fn money(cents: i64) -> String {
    paper::fmt_cents(cents)
}

fn bytes(n: u64) -> String {
    let u = ["B", "KB", "MB", "GB", "TB"];
    let mut f = n as f64;
    let mut i = 0;
    while f >= 1024.0 && i < u.len() - 1 {
        f /= 1024.0;
        i += 1;
    }
    format!("{f:.1} {}", u[i])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn formats() {
        assert_eq!(money(100_000_00), "$100,000.00");
        assert_eq!(money(1_234_56), "$1,234.56");
        assert_eq!(money(5), "$0.05");
        assert_eq!(bytes(1536), "1.5 KB");
        assert_eq!(hms(3661), "01:01:01");
        assert!(is_stock("AAPL") && !is_stock("CL=F") && !is_stock("^VIX") && !is_stock("BTC-USD"));
    }
}

#[cfg(test)]
mod tests_ahead {
    #[test]
    fn spans() {
        assert_eq!(super::ahead(1, 12), "(= 1 hour of trading ahead)");
        assert_eq!(super::ahead(2, 3), "(= 45 mins of trading ahead)");
        assert_eq!(super::ahead(4, 5), "(= 5 trading days ahead)");
    }
}

#[cfg(test)]
mod tests_compare {
    use super::*;

    fn p(sym: &str, exp: f64, up: f64) -> Prediction {
        Prediction { sym: sym.into(), exp_pct: exp, p_up: up, ..Default::default() }
    }

    #[test]
    fn agreement_and_both_right() {
        let a = [p("W", 3.0, 0.9), p("X", 2.0, 0.8), p("Y", 1.0, 0.7), p("Z", -1.0, 0.2), p("ONLY_A", 9.0, 0.9)];
        let b = [p("W", 1.0, 0.6), p("X", 3.0, 0.9), p("Y", 2.0, 0.8), p("Z", -2.0, 0.1)];
        let (ra, rb): (Vec<&Prediction>, Vec<&Prediction>) = (a.iter().collect(), b.iter().collect());
        let hit = [("W", 9, 10), ("X", 4, 10), ("Y", 8, 10), ("Z", 9, 9)];
        let few = [("W", 9, 10), ("X", 9, 10), ("Y", 7, 10), ("Z", 1, 3)];
        let c = compare(&ra, &rb, &hit, &few);
        // ranks A: W0 X1 Y2 Z3, B: X0 Y1 W2 Z3 -> worse rank: X1, W2, Y2(sum 3), Z3
        assert_eq!(c.rows.iter().map(|r| r.0).collect::<Vec<_>>(), ["X", "W", "Y", "Z", "ONLY_A"]);
        assert_eq!(c.agree, ["X", "W", "Y"]); // Z falls, ONLY_A has no B
        // min hit rate: W .9, Y .7, X .4; Z has < MIN_CHECKED for B
        assert_eq!(c.both_right, ["W", "Y", "X"]);
        let rho = c.rho.unwrap(); // d = 2,1,1,0 -> 1 - 6*6/(4*15) = 0.4
        assert!((rho - 0.4).abs() < 1e-9, "{rho}");
    }
}

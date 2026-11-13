//! Yahoo Finance adapter. Unofficial endpoints: can break or rate-limit without notice.
use crate::shared::{Candle, Quote, Session};
use serde_json::Value;
use std::time::Duration;

const BASE: &str = "https://query1.finance.yahoo.com";
pub const MAX_BATCH: usize = 20; // Yahoo rejects spark calls with more symbols

/// Futures, FX, crypto and indices added on top of the Nasdaq stock lists.
pub const EXTRA: &[&str] = &[
    "CL=F", "BZ=F", "NG=F", "GC=F", "SI=F", "HG=F", "PL=F", "ZC=F", "ZW=F", "ZS=F", "KC=F", "CC=F", "SB=F", "CT=F",
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "BTC-USD", "ETH-USD",
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^TNX",
];

pub struct Yahoo {
    agent: ureq::Agent,
}

impl Yahoo {
    pub fn new() -> Self {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(10)).user_agent("Mozilla/5.0").build();
        Self { agent }
    }

    fn json(&self, url: &str) -> Result<Value, String> {
        self.agent.get(url).call().map_err(|e| e.to_string())?.into_json().map_err(|e| e.to_string())
    }

    /// Latest quote for up to MAX_BATCH symbols in one call.
    pub fn quotes(&self, syms: &[String]) -> Result<(Vec<Quote>, Option<Session>), String> {
        let url = format!("{BASE}/v7/finance/spark?symbols={}&range=1d&interval=1d", enc(&syms.join(",")));
        let v = self.json(&url)?;
        let res = v["spark"]["result"].as_array().ok_or_else(|| err(&v["spark"]))?;
        let metas: Vec<&Value> = res.iter().map(|r| &r["response"][0]["meta"]).collect();
        Ok((metas.iter().filter_map(|m| quote(m)).collect(), metas.iter().find_map(|m| session(m))))
    }

    /// Candles at one Yahoo interval/range (intraday incl. pre/after hours).
    /// No quote from here: its previous close depends on the range.
    pub fn chart(&self, sym: &str, interval: &str, range: &str) -> Result<(Vec<Candle>, Option<Session>), String> {
        let url = format!("{BASE}/v8/finance/chart/{}?interval={interval}&range={range}&includePrePost=true", enc(sym));
        let v = self.json(&url)?;
        let r = &v["chart"]["result"][0];
        if r.is_null() {
            return Err(err(&v["chart"]));
        }
        let q = &r["indicators"]["quote"][0];
        let f = |k: &str, i: usize| q[k][i].as_f64();
        let candles = r["timestamp"].as_array().map_or(vec![], |ts| {
            ts.iter()
                .enumerate()
                .filter_map(|(i, t)| {
                    // Yahoo leaves nulls for minutes with no trades
                    Some(Candle { t: t.as_i64()?, o: f("open", i)?, h: f("high", i)?, l: f("low", i)?, c: f("close", i)?, v: q["volume"][i].as_u64().unwrap_or(0) })
                })
                .collect()
        });
        Ok((candles, session(&r["meta"])))
    }

    /// Last ~5 fiscal quarters of company accounts (no login needed). Empty for non-companies.
    pub fn fundamentals(&self, sym: &str) -> Result<Vec<crate::quant::FundQ>, String> {
        let now = chrono::Utc::now().timestamp();
        let types = "quarterlyDilutedEPS,quarterlyStockholdersEquity,quarterlyNetIncome,quarterlyDilutedAverageShares";
        let url = format!("{BASE}/ws/fundamentals-timeseries/v1/finance/timeseries/{}?type={types}&period1={}&period2={}", enc(sym), now - 4 * 365 * 86_400, now + 86_400);
        let v = self.json(&url)?;
        let res = v["timeseries"]["result"].as_array().ok_or_else(|| err(&v["timeseries"]))?;
        let mut q: std::collections::BTreeMap<i64, crate::quant::FundQ> = Default::default();
        for r in res {
            let Some(k) = r["meta"]["type"][0].as_str() else { continue };
            for x in r[k].as_array().into_iter().flatten() {
                let (Some(d), Some(val)) = (x["asOfDate"].as_str(), x["reportedValue"]["raw"].as_f64()) else { continue };
                let Ok(d) = chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d") else { continue };
                let t = d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp();
                let e = q.entry(t).or_insert(crate::quant::FundQ { end_t: t, eps: f64::NAN, equity: f64::NAN, net_income: f64::NAN, shares: f64::NAN });
                match k {
                    "quarterlyDilutedEPS" => e.eps = val,
                    "quarterlyStockholdersEquity" => e.equity = val,
                    "quarterlyNetIncome" => e.net_income = val,
                    _ => e.shares = val,
                }
            }
        }
        Ok(q.into_values().collect())
    }

    /// Every US-listed symbol from Nasdaq's public symbol directory, Yahoo-style tickers.
    pub fn symbol_list(&self) -> Result<Vec<String>, String> {
        let mut out = vec![];
        // (file, column of the "Test Issue" flag)
        for (file, test_col) in [("nasdaqlisted.txt", 3), ("otherlisted.txt", 6)] {
            let url = format!("https://www.nasdaqtrader.com/dynamic/SymDir/{file}");
            let txt = self.agent.get(&url).call().map_err(|e| e.to_string())?.into_string().map_err(|e| e.to_string())?;
            for line in txt.lines().skip(1) {
                let f: Vec<&str> = line.split('|').collect();
                if f.len() <= test_col || f[test_col] == "Y" || f[0].contains('$') || line.starts_with("File Creation") {
                    continue;
                }
                out.push(f[0].replace('.', "-")); // BRK.B -> BRK-B
            }
        }
        out.sort();
        out.dedup();
        Ok(out)
    }
}

fn quote(m: &Value) -> Option<Quote> {
    Some(Quote {
        sym: m["symbol"].as_str()?.into(),
        name: m["longName"].as_str().or(m["shortName"].as_str()).unwrap_or("").into(),
        price: m["regularMarketPrice"].as_f64()?,
        change_pct: m["regularMarketChangePercent"].as_f64().unwrap_or(0.0),
        prev_close: m["chartPreviousClose"].as_f64().unwrap_or(0.0),
        volume: m["regularMarketVolume"].as_u64().unwrap_or(0),
        time: m["regularMarketTime"].as_i64().unwrap_or(0),
    })
}

fn session(m: &Value) -> Option<Session> {
    let p = &m["currentTradingPeriod"];
    let w = |k: &str| Some((p[k]["start"].as_i64()?, p[k]["end"].as_i64()?));
    Some(Session { pre: w("pre")?, regular: w("regular")?, post: w("post")?, gmtoffset: m["gmtoffset"].as_i64()? })
}

fn err(v: &Value) -> String {
    v["error"]["description"].as_str().unwrap_or("no data").to_string()
}

fn enc(s: &str) -> String {
    s.replace('^', "%5E").replace('=', "%3D")
}

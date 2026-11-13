//! Strategies that use other symbols, company accounts or event clustering. Long-only versions:
//! a pairs/basket/stat-arb signal fires when THIS stock is the cheap leg (the one you'd buy).
use super::*;

// ---------------------------------------------------------------- small linear algebra

type Mat = Vec<Vec<f64>>;

fn matmul(a: &Mat, b: &Mat) -> Mat {
    (0..a.len()).map(|i| (0..b[0].len()).map(|j| (0..b.len()).map(|k| a[i][k] * b[k][j]).sum()).collect()).collect()
}

fn transpose(a: &Mat) -> Mat {
    (0..a[0].len()).map(|j| (0..a.len()).map(|i| a[i][j]).collect()).collect()
}

fn inv(a: &Mat) -> Option<Mat> {
    let n = a.len();
    let cols: Option<Vec<Vec<f64>>> = (0..n).map(|j| solve(a.clone(), (0..n).map(|i| (i == j) as u8 as f64).collect())).collect();
    Some(transpose(&cols?))
}

/// lower-triangular L with L L' = a; None if not positive definite
fn chol(a: &Mat) -> Option<Mat> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Eigen-decomposition of a symmetric matrix (cyclic Jacobi): (values, vectors), largest value first.
pub(super) fn eigen_sym(mut a: Mat) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Mat = (0..n).map(|i| (0..n).map(|j| (i == j) as u8 as f64).collect()).collect();
    for _ in 0..100 {
        let off: f64 = (0..n).map(|i| (0..n).filter(|&j| j != i).map(|j| a[i][j] * a[i][j]).sum::<f64>()).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (a[p][k], a[q][k]);
                    a[p][k] = c * x - s * y;
                    a[q][k] = s * x + c * y;
                }
                for row in v.iter_mut() {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    (idx.iter().map(|&i| a[i][i]).collect(), idx.iter().map(|&i| (0..n).map(|r| v[r][i]).collect()).collect())
}

/// least squares: rows of regressors (include a 1.0 for a constant) -> coefficients
fn ols(rows: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let k = rows.first()?.len();
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![0.0; k];
    for (r, yy) in rows.iter().zip(y) {
        for i in 0..k {
            xty[i] += r[i] * yy;
            for j in 0..k {
                xtx[i][j] += r[i] * r[j];
            }
        }
    }
    solve(xtx, xty)
}

/// ADF t-statistic of a (mean ~0) residual series: d e(t) = g e(t-1) + u, no constant.
fn adf_t(e: &[f64]) -> f64 {
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for w in e.windows(2) {
        sxx += w[0] * w[0];
        sxy += w[0] * (w[1] - w[0]);
    }
    if sxx <= 0.0 {
        return 0.0;
    }
    let g = sxy / sxx;
    let ssr: f64 = e.windows(2).map(|w| (w[1] - w[0] - g * w[0]).powi(2)).sum();
    g / (ssr / (e.len() as f64 - 2.0) / sxx).sqrt()
}

fn lret(c: &[f64]) -> Vec<f64> {
    (0..c.len()).map(|i| if i == 0 || !(c[i - 1] > 0.0 && c[i] > 0.0) { NAN } else { (c[i] / c[i - 1]).ln() }).collect()
}

fn corr(a: &[f64], b: &[f64]) -> Option<f64> {
    let both: Vec<(f64, f64)> = a.iter().zip(b).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (*x, *y)).collect();
    if both.len() < 60 {
        return None;
    }
    let n = both.len() as f64;
    let (mx, my) = (both.iter().map(|p| p.0).sum::<f64>() / n, both.iter().map(|p| p.1).sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in &both {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    (sxx > 0.0 && syy > 0.0).then(|| sxy / (sxx * syy).sqrt())
}

/// The k tracked symbols whose returns move most like this stock's (correlation >= 0.3).
/// shortcut: picked on the whole lookback (mild selection bias in the history); pick walk-forward if it matters.
fn partners<'a>(b: &Bars, cx: &'a Ctx, k: usize) -> Result<Vec<(&'a str, &'a [f64])>, String> {
    let r = lret(&b.c);
    let mut v: Vec<(f64, &str, &[f64])> = cx.uni.iter().filter_map(|(s, c)| Some((corr(&r, &lret(c))?, *s, &c[..]))).filter(|x| x.0 >= 0.3).collect();
    v.sort_by(|x, y| y.0.total_cmp(&x.0));
    if v.len() < k {
        return Err(format!("needs {k} tracked stock(s) moving with {} (correlation >= 0.3); found {}", cx.sym, v.len()));
    }
    Ok(v.into_iter().take(k).map(|x| (x.1, x.2)).collect())
}

fn logs(c: &[f64]) -> Vec<f64> {
    c.iter().map(|x| if *x > 0.0 { x.ln() } else { NAN }).collect()
}

// ---------------------------------------------------------------- pairs / baskets

/// Rolling Engle-Granger: 60-candle OLS hedge ratio, residual z, ADF on the residuals.
pub(super) fn pairs(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    const W: usize = 60;
    let (x, y) = (logs(&b.c), logs(partners(b, cx, 1)?[0].1));
    let n = b.n();
    let (mut z, mut ok) = (vec![NAN; n], vec![false; n]);
    for i in W..n {
        let (xs, ys) = (&x[i + 1 - W..=i], &y[i + 1 - W..=i]);
        if ys.iter().chain(xs).any(|v| !v.is_finite()) {
            continue;
        }
        let rows: Vec<Vec<f64>> = ys.iter().map(|y| vec![1.0, *y]).collect();
        let Some(k) = ols(&rows, xs) else { continue };
        let res: Vec<f64> = xs.iter().zip(ys).map(|(x, y)| x - k[0] - k[1] * y).collect();
        let sd = mean_var(&res).1.sqrt();
        if sd > 0.0 {
            z[i] = res[W - 1] / sd;
            ok[i] = adf_t(&res) < -3.04; // Engle-Granger 10% critical value, 2 series
        }
    }
    Ok(each(b, W + 1, |i| ok[i] && down_through(&z, -2.0, i)))
}

/// Kalman filter on (alpha, beta) of log A = alpha + beta log B; signal = standardised innovation.
pub(super) fn kalman_pairs(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    const W: usize = 60;
    let (x, y) = (logs(&b.c), logs(partners(b, cx, 1)?[0].1));
    let n = b.n();
    let start = (0..n).find(|&i| i >= W && (i + 1 - W..=i).all(|j| x[j].is_finite() && y[j].is_finite())).ok_or("partner history too short")?;
    // initial state + measurement noise from the first W candles only (causal)
    let rows: Vec<Vec<f64>> = (start + 1 - W..=start).map(|j| vec![1.0, y[j]]).collect();
    let mut th = ols(&rows, &x[start + 1 - W..=start]).ok_or("fit failed")?;
    // measurement noise = the fast part of the static fit's residual: var(d res) = var(d state) + 2 Ve
    let res: Vec<f64> = (start + 1 - W..=start).map(|j| x[j] - th[0] - th[1] * y[j]).collect();
    let ve = (mean_var(&res.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>()).1 / 2.0).max(1e-12);
    let vw = 1e-4 / (1.0 - 1e-4) * ve; // state drift, relative to measurement noise
    let mut p = [[ve, 0.0], [0.0, ve]];
    let mut z = vec![NAN; n];
    for t in start + 1..n {
        if !(x[t].is_finite() && y[t].is_finite()) {
            continue;
        }
        p[0][0] += vw;
        p[1][1] += vw;
        let f = [1.0, y[t]];
        let e = x[t] - th[0] - th[1] * y[t];
        let pf = [p[0][0] * f[0] + p[0][1] * f[1], p[1][0] * f[0] + p[1][1] * f[1]];
        let q = f[0] * pf[0] + f[1] * pf[1] + ve;
        let k = [pf[0] / q, pf[1] / q];
        th[0] += k[0] * e;
        th[1] += k[1] * e;
        p = [[p[0][0] - k[0] * pf[0], p[0][1] - k[0] * pf[1]], [p[1][0] - k[1] * pf[0], p[1][1] - k[1] * pf[1]]];
        z[t] = e / q.sqrt();
    }
    Ok(each(b, start + 2, |i| down_through(&z, -2.0, i)))
}

/// Johansen trace test (1 lag, constant) on log prices; weights of the strongest cointegrating vector.
/// Returns (trace statistic for r = 0, weights normalised so series 0 has weight 1).
pub(super) fn johansen(x: &[Vec<f64>]) -> Option<(f64, Vec<f64>)> {
    let (k, len) = (x.len(), x[0].len());
    let t = len - 1;
    let mean = |f: &dyn Fn(usize, usize) -> f64, j: usize| (1..len).map(|s| f(s, j)).sum::<f64>() / t as f64;
    let d = |s: usize, j: usize| x[j][s] - x[j][s - 1];
    let l = |s: usize, j: usize| x[j][s - 1];
    let (md, ml): (Vec<f64>, Vec<f64>) = ((0..k).map(|j| mean(&d, j)).collect(), (0..k).map(|j| mean(&l, j)).collect());
    let cov = |a: &dyn Fn(usize, usize) -> f64, ma: &[f64], b: &dyn Fn(usize, usize) -> f64, mb: &[f64]| -> Mat {
        (0..k).map(|i| (0..k).map(|j| (1..len).map(|s| (a(s, i) - ma[i]) * (b(s, j) - mb[j])).sum::<f64>() / t as f64).collect()).collect()
    };
    let (s00, s11, s01) = (cov(&d, &md, &d, &md), cov(&l, &ml, &l, &ml), cov(&d, &md, &l, &ml));
    let lc = chol(&s11)?;
    let li = inv(&lc)?;
    // symmetric form: L^-1 S10 S00^-1 S01 L^-T
    let m = matmul(&matmul(&li, &matmul(&matmul(&transpose(&s01), &inv(&s00)?), &s01)), &transpose(&li));
    let (vals, vecs) = eigen_sym(m);
    if vals.iter().any(|v| !(0.0..1.0).contains(v)) {
        return None;
    }
    let trace = -(t as f64) * vals.iter().map(|v| (1.0 - v).ln()).sum::<f64>();
    let beta: Vec<f64> = (0..k).map(|i| (0..k).map(|r| li[r][i] * vecs[0][r]).sum()).collect(); // L^-T u
    (beta[0].abs() > 1e-12).then(|| (trace, beta.iter().map(|w| w / beta[0]).collect()))
}

/// Basket of this stock + its 2 closest partners, rolling 120-candle Johansen.
pub(super) fn johansen_basket(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    const W: usize = 120;
    let ps = partners(b, cx, 2)?;
    let series: Vec<Vec<f64>> = std::iter::once(logs(&b.c)).chain(ps.iter().map(|p| logs(p.1))).collect();
    let n = b.n();
    let (mut z, mut ok) = (vec![NAN; n], vec![false; n]);
    for i in W..n {
        let win: Vec<Vec<f64>> = series.iter().map(|s| s[i + 1 - W..=i].to_vec()).collect();
        if win.iter().flatten().any(|v| !v.is_finite()) {
            continue;
        }
        let Some((trace, w)) = johansen(&win) else { continue };
        let basket: Vec<f64> = (0..W).map(|s| (0..3).map(|j| w[j] * win[j][s]).sum()).collect();
        let (m, v) = mean_var(&basket);
        if v > 0.0 {
            z[i] = (basket[W - 1] - m) / v.sqrt();
            ok[i] = trace > 29.80; // r = 0 rejected at 5% (3 series, constant)
        }
    }
    Ok(each(b, W + 1, |i| ok[i] && down_through(&z, -2.0, i)))
}

// ---------------------------------------------------------------- cross-section

/// Fires when this stock enters the top 3 of the tracked stocks by 63-candle return (skipping the last 5).
pub(super) fn xs_momentum(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    const L: usize = 63;
    const S: usize = 5;
    if cx.uni.len() < 7 {
        return Err("needs 7+ other tracked symbols".into());
    }
    let score = |c: &[f64], i: usize| if i >= L && c[i - L] > 0.0 { c[i - S] / c[i - L] - 1.0 } else { NAN };
    let top3: Vec<bool> = (0..b.n())
        .map(|i| {
            let me = score(&b.c, i);
            let others: Vec<f64> = cx.uni.iter().map(|u| score(&u.1, i)).filter(|x| x.is_finite()).collect();
            me.is_finite() && others.len() >= 6 && others.iter().filter(|x| **x > me).count() < 3
        })
        .collect();
    Ok(each(b, L + 1, |i| top3[i] && !top3[i - 1]))
}

/// Avellaneda-Lee style: residual of this stock's returns after the top 3 principal components of the
/// other tracked stocks (60-candle window, refreshed every 5 candles); fires when its cumulative
/// residual drops 1.5 SD below its window average (s-score).
pub(super) fn pca(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    const W: usize = 60;
    const K: usize = 3;
    let n = b.n();
    let r0 = lret(&b.c);
    let rs: Vec<Vec<f64>> = cx.uni.iter().map(|u| lret(&u.1)).collect();
    if rs.len() < 8 {
        return Err("needs 8+ other tracked symbols".into());
    }
    let mut s = vec![NAN; n];
    let mut cache: Option<(Vec<usize>, Vec<f64>, Vec<Vec<f64>>)> = None; // (columns, their sd, eigenvectors)
    for i in W + 1..n {
        let lo = i + 1 - W;
        if r0[lo..=i].iter().any(|x| !x.is_finite()) {
            continue;
        }
        let cols: Vec<usize> = (0..rs.len()).filter(|&j| rs[j][lo..=i].iter().all(|x| x.is_finite())).collect();
        if cols.len() < 8 {
            continue;
        }
        let stale = cache.as_ref().is_none_or(|c| c.0 != cols || i % 5 == 0);
        if stale {
            let z: Vec<(f64, f64)> = cols.iter().map(|&j| mean_var(&rs[j][lo..=i])).collect();
            let sd: Vec<f64> = z.iter().map(|x| x.1.sqrt().max(1e-12)).collect();
            let m = cols.len();
            let corr: Mat = (0..m)
                .map(|a| (0..m).map(|c| (lo..=i).map(|t| (rs[cols[a]][t] - z[a].0) * (rs[cols[c]][t] - z[c].0)).sum::<f64>() / W as f64 / (sd[a] * sd[c])).collect())
                .collect();
            let (_, vecs) = eigen_sym(corr);
            cache = Some((cols.clone(), sd, vecs.into_iter().take(K).collect()));
        }
        let (cols, sd, vecs) = cache.as_ref().unwrap();
        let rows: Vec<Vec<f64>> = (lo..=i).map(|t| std::iter::once(1.0).chain(vecs.iter().map(|v| cols.iter().enumerate().map(|(a, &j)| v[a] * rs[j][t] / sd[a]).sum())).collect()).collect();
        let Some(k) = ols(&rows, &r0[lo..=i]) else { continue };
        let mut cum = 0.0;
        let x: Vec<f64> = rows.iter().zip(&r0[lo..=i]).map(|(row, y)| {
            cum += y - row.iter().zip(&k).map(|(a, b)| a * b).sum::<f64>();
            cum
        }).collect();
        let (m, v) = mean_var(&x);
        if v > 0.0 {
            s[i] = (x[W - 1] - m) / v.sqrt();
        }
    }
    Ok(each(b, W + 2, |i| down_through(&s, -1.5, i)))
}

/// Latest quarter whose accounts were public at time t (reported ~45 days after quarter end).
fn as_of(f: &[FundQ], t: i64) -> Option<&FundQ> {
    f.iter().filter(|q| q.end_t + 45 * 86_400 <= t).max_by_key(|q| q.end_t)
}

/// Multi-factor score across the tracked stocks: value (book/price), quality (ROE), size (small = better),
/// momentum (63-5 candle return), low volatility. Fires when this stock enters the top 3.
pub(super) fn factor(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    let mine = cx.funds.get(cx.sym).filter(|f| !f.is_empty()).ok_or("factor investing ranks companies: no company accounts for this symbol (futures/FX/crypto/ETF)")?;
    let mut syms: Vec<(&[f64], &[FundQ])> = vec![(&b.c[..], &mine[..])];
    syms.extend(cx.uni.iter().filter_map(|u| Some((&u.1[..], &cx.funds.get(u.0).filter(|f| !f.is_empty())?[..]))));
    if syms.len() < 6 {
        return Err(format!("needs 6+ tracked companies with accounts, has {}", syms.len()));
    }
    let n = b.n();
    let top3: Vec<bool> = (0..n)
        .map(|i| {
            if i < 64 {
                return false;
            }
            // raw factor values per stock (NaN = unknown)
            let raw: Vec<[f64; 5]> = syms
                .iter()
                .map(|(c, f)| {
                    let p = c[i];
                    let q = as_of(f, b.t[i]);
                    let cap = q.map_or(NAN, |q| q.shares * p);
                    let rets: Vec<f64> = (i - 59..=i).map(|j| (c[j] / c[j - 1]).ln()).collect();
                    [
                        q.map_or(NAN, |q| q.equity / cap),
                        q.map_or(NAN, |q| 4.0 * q.net_income / q.equity),
                        -cap.ln(),
                        if c[i - 63] > 0.0 { c[i - 5] / c[i - 63] - 1.0 } else { NAN },
                        -mean_var(&rets).1.sqrt(),
                    ]
                })
                .collect();
            let mut score = vec![(0.0, 0usize); raw.len()];
            for k in 0..5 {
                let vals: Vec<f64> = raw.iter().map(|r| r[k]).filter(|x| x.is_finite()).collect();
                if vals.len() < 5 {
                    continue;
                }
                let (m, v) = mean_var(&vals);
                if v <= 0.0 {
                    continue;
                }
                for (s, r) in score.iter_mut().zip(&raw) {
                    if r[k].is_finite() {
                        *s = (s.0 + (r[k] - m) / v.sqrt(), s.1 + 1);
                    }
                }
            }
            let comp: Vec<f64> = score.iter().map(|s| if s.1 > 0 { s.0 / s.1 as f64 } else { NAN }).collect();
            comp[0].is_finite() && comp.iter().filter(|x| x.is_finite()).count() >= 6 && comp[1..].iter().filter(|x| **x > comp[0]).count() < 3
        })
        .collect();
    Ok(each(b, 65, |i| top3[i] && !top3[i - 1]))
}

/// Fama-French style factor model via ETF proxies: MKT = SPY, SMB = IWM - SPY, HML = IWD - IWF,
/// RMW = QUAL - SPY, UMD = MTUM - SPY. Simulated as betas x resampled factor days + resampled residuals
/// (alpha set to 0: past alpha is mostly noise).
pub(super) fn ff5(b: &Bars, cx: &Ctx, rng: &mut Rng) -> Result<(Samples, String), String> {
    // the stock may itself be one of the factor ETFs
    let get = |s: &str| if s == cx.sym { Ok(lret(&b.c)) } else { cx.get(s).map(lret).ok_or(format!("factor ETF {s} not downloaded yet")) };
    let (spy, iwm, iwd, iwf, qual, mtum) = (get("SPY")?, get("IWM")?, get("IWD")?, get("IWF")?, get("QUAL")?, get("MTUM")?);
    let r = lret(&b.c);
    let (mut rows, mut y) = (vec![], vec![]);
    for t in 1..b.n() {
        let f = [spy[t], iwm[t] - spy[t], iwd[t] - iwf[t], qual[t] - spy[t], mtum[t] - spy[t]];
        if r[t].is_finite() && f.iter().all(|x| x.is_finite()) {
            rows.push(std::iter::once(1.0).chain(f).collect::<Vec<f64>>());
            y.push(r[t]);
        }
    }
    if rows.len() < 60 {
        return Err(format!("only {} candles overlap with the factor ETFs (needs 60)", rows.len()));
    }
    let k = ols(&rows, &y).ok_or("factor fit failed")?;
    let resid: Vec<f64> = rows.iter().zip(&y).map(|(row, y)| y - row.iter().zip(&k).map(|(a, b)| a * b).sum::<f64>()).collect();
    let last = b.c[b.n() - 1];
    let s = (0..SIM_PATHS)
        .map(|_| {
            let mut price = last;
            (0..cx.h)
                .map(|_| {
                    let f = &rows[pick(rng, rows.len())];
                    let x = (1..6).map(|j| k[j] * f[j]).sum::<f64>() + resid[pick(rng, resid.len())];
                    step(&mut price, x)
                })
                .collect()
        })
        .collect();
    Ok((s, format!("MODEL betas MKT {:.2} SMB {:+.2} HML {:+.2} RMW {:+.2} UMD {:+.2} on {} candles", k[1], k[2], k[3], k[4], k[5], rows.len())))
}

/// Post-earnings drift: the earnings day of each fiscal quarter = first day 10-75 days after quarter end
/// with volume > 1.8x its 20-day average; fires there if the stock rose that day (and EPS beat the
/// same quarter a year earlier, when Yahoo has both).
pub(super) fn pead(b: &Bars, cx: &Ctx) -> Result<Vec<bool>, String> {
    let n = b.n();
    let mut gaps: Vec<i64> = b.t.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort();
    let g = gaps.get(gaps.len() / 2).copied().unwrap_or(0);
    if !(57_600..=345_600).contains(&g) {
        return Err("needs 1 DAY candles (earnings come once a quarter)".into());
    }
    if !b.has_volume() {
        return Err("needs volume data".into());
    }
    let f = cx.funds.get(cx.sym).filter(|f| !f.is_empty()).ok_or("needs company accounts (none for this symbol)")?;
    // quarter ends: Yahoo's (last ~5) extended back and one ahead every 91.3 days
    const Q: i64 = 7_889_000;
    let mut ends: Vec<i64> = f.iter().map(|q| q.end_t).collect();
    let (first, last) = (*ends.iter().min().unwrap(), *ends.iter().max().unwrap());
    let mut e = first - Q;
    while e > b.t[0] - 90 * 86_400 {
        ends.push(e);
        e -= Q;
    }
    ends.push(last + Q);
    let vavg = sma(&b.v, 20);
    let mut fires = vec![false; n];
    for qe in ends {
        let (lo, hi) = (qe + 10 * 86_400, qe + 75 * 86_400);
        let Some(i) = (21..n).find(|&i| b.t[i] >= lo && b.t[i] <= hi && vavg[i - 1] > 0.0 && b.v[i] > 1.8 * vavg[i - 1]) else { continue };
        let eps_ok = match (f.iter().find(|q| (q.end_t - qe).abs() < 20 * 86_400), f.iter().find(|q| (q.end_t - (qe - 4 * Q)).abs() < 20 * 86_400)) {
            (Some(now), Some(ago)) if now.eps.is_finite() && ago.eps.is_finite() => now.eps > ago.eps,
            _ => true,
        };
        fires[i] = b.c[i] > b.c[i - 1] && eps_ok;
    }
    Ok(fires)
}

// ---------------------------------------------------------------- Hawkes

/// Self-exciting (Hawkes) model of big moves: events = candles moving > 2 SD; intensity
/// mu + alpha sum e^(-beta (t - t_i)) fitted by maximum likelihood (grid), simulated forward with
/// resampled jump and normal returns.
pub(super) fn hawkes(b: &Bars, h: usize, rng: &mut Rng) -> Result<Samples, String> {
    let r = clean_rets(b);
    if r.len() < 200 {
        return Err("needs 200+ candles".into());
    }
    let sd = mean_var(&r).1.sqrt();
    let ev: Vec<f64> = (0..r.len()).filter(|&i| r[i].abs() > 2.0 * sd).map(|i| i as f64).collect();
    let (jumps, calm): (Vec<f64>, Vec<f64>) = r.iter().partition(|x| x.abs() > 2.0 * sd);
    if ev.len() < 10 {
        return Err(format!("only {} big moves in the lookback (needs 10)", ev.len()));
    }
    let t_end = r.len() as f64;
    let ll = |mu: f64, a: f64, be: f64| {
        let (mut acc, mut prev, mut l) = (0.0, 0.0, 0.0);
        for (k, &t) in ev.iter().enumerate() {
            if k > 0 {
                acc = (-be * (t - prev)).exp() * (1.0 + acc);
            }
            l += (mu + a * acc).ln();
            prev = t;
        }
        l - mu * t_end - a / be * ev.iter().map(|t| 1.0 - (-be * (t_end - t)).exp()).sum::<f64>()
    };
    let mut best = (f64::MIN, 0.0, 0.0, 1.0);
    for be in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2] {
        for eta in 0..=9 {
            let a = eta as f64 * 0.1 * be;
            let mu0 = ev.len() as f64 * (1.0 - eta as f64 * 0.1) / t_end;
            for m in [0.6, 0.8, 1.0, 1.25, 1.5] {
                let l = ll(mu0 * m, a, be);
                if l > best.0 {
                    best = (l, mu0 * m, a, be);
                }
            }
        }
    }
    let (_, mu, a, be) = best;
    let g0: f64 = ev.iter().map(|t| (-be * (t_end - 1.0 - t)).exp()).sum();
    let last = b.c[b.n() - 1];
    Ok((0..SIM_PATHS)
        .map(|_| {
            let (mut g, mut price) = (g0, last);
            (0..h)
                .map(|_| {
                    g *= (-be).exp();
                    let p = 1.0 - (-(mu + a * g)).exp();
                    let x = if (rng.unit() as f64) < p {
                        g += 1.0;
                        jumps[pick(rng, jumps.len())]
                    } else {
                        calm[pick(rng, calm.len())]
                    };
                    step(&mut price, x)
                })
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_and_johansen() {
        let (vals, vecs) = eigen_sym(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        assert!((vals[0] - 3.0).abs() < 1e-9 && (vals[1] - 1.0).abs() < 1e-9);
        assert!((vecs[0][0].abs() - 0.5f64.sqrt()).abs() < 1e-9 && (vecs[0][0] - vecs[0][1]).abs() < 1e-9);
        // x1 = random walk, x2 = x1 + noise, x3 independent walk: cointegrated with weights ~ (1, -1, 0)
        let mut rng = Rng::new(5);
        let (mut a, mut c) = (0.0, 0.0);
        let mut x = vec![vec![]; 3];
        for _ in 0..400 {
            a += 0.01 * normal(&mut rng);
            c += 0.01 * normal(&mut rng);
            x[0].push(a + 0.002 * normal(&mut rng));
            x[1].push(a);
            x[2].push(c);
        }
        let (trace, w) = johansen(&x).unwrap();
        assert!(trace > 29.8, "{trace}");
        assert!((w[1] + 1.0).abs() < 0.05 && w[2].abs() < 0.05, "{w:?}");
        assert!(adf_t(&x[0].iter().zip(&x[1]).map(|(p, q)| p - q).collect::<Vec<_>>()) < -5.0);
    }
}

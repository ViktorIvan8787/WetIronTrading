//! Kronos candlestick foundation model, inference only, pure Rust (no ML framework).
//! Faithful port of shiyu-coder/Kronos model/module.py + model/kronos.py (MIT licence),
//! weights loaded straight from the official NeoQuasar safetensors files.
//!
//! Speed: the context is encoded once and cached (KV cache), then every sample only runs the
//! new tokens. This equals the reference's full recompute as long as context + horizon <= max_context,
//! which `predict` enforces.
use chrono::{Datelike, Timelike};
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// safetensors = 8-byte header length, JSON header, raw little-endian tensors. Only F32 is used.
fn load_safetensors(path: &Path) -> Result<Tensors, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let n = u64::from_le_bytes(bytes.get(..8).ok_or("file too short")?.try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + n).ok_or("bad header")?).map_err(|e| e.to_string())?;
    let data = &bytes[8 + n..];
    let mut out = HashMap::new();
    for (k, v) in hdr.as_object().ok_or("bad header")? {
        if v["dtype"] != "F32" {
            continue; // __metadata__ and the tokenizer's unused I64 buffers
        }
        let shape: Vec<usize> = v["shape"].as_array().ok_or("bad shape")?.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect();
        let off = &v["data_offsets"];
        let (a, b) = (off[0].as_u64().ok_or("bad offsets")? as usize, off[1].as_u64().ok_or("bad offsets")? as usize);
        let raw = data.get(a..b).ok_or("offsets out of range")?;
        let f: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        if f.len() != shape.iter().product::<usize>() {
            return Err(format!("{k}: size mismatch"));
        }
        out.insert(k.clone(), (shape, f));
    }
    Ok(out)
}

fn take(t: &mut Tensors, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    t.remove(name).ok_or_else(|| format!("missing tensor {name}"))
}

fn count_blocks(t: &Tensors, prefix: &str) -> usize {
    (0..).take_while(|i| t.contains_key(&format!("{prefix}.{i}.norm1.weight"))).count()
}

fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// Runs `f(item_index, item)` over `y` split into items of `unit` floats, across all cores when `heavy`.
fn par_for(y: &mut [f32], unit: usize, heavy: bool, f: impl Fn(usize, &mut [f32]) + Sync) {
    let items = y.len() / unit;
    let th = if heavy { threads().min(items) } else { 1 };
    if th <= 1 {
        y.chunks_mut(unit).enumerate().for_each(|(i, c)| f(i, c));
        return;
    }
    let per = items.div_ceil(th);
    std::thread::scope(|s| {
        for (ci, chunk) in y.chunks_mut(per * unit).enumerate() {
            let f = &f;
            s.spawn(move || chunk.chunks_mut(unit).enumerate().for_each(|(k, c)| f(ci * per + k, c)));
        }
    });
}

/// 8 independent accumulators so the compiler vectorises it.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}

struct Linear {
    w: Vec<f32>, // [out, in] row-major, as PyTorch stores it
    b: Option<Vec<f32>>,
    i: usize,
    o: usize,
}

impl Linear {
    fn load(t: &mut Tensors, p: &str, bias: bool) -> Result<Self, String> {
        let (s, w) = take(t, &format!("{p}.weight"))?;
        let b = if bias { Some(take(t, &format!("{p}.bias"))?.1) } else { None };
        Ok(Self { w, b, o: s[0], i: s[1] })
    }
    /// x: n rows of `i` -> n rows of `o`
    fn fwd(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() / self.i;
        let mut y = vec![0f32; n * self.o];
        par_for(&mut y, 1, n * self.o * self.i > 1 << 18, |e, out| {
            let (r, j) = (e / self.o, e % self.o);
            out[0] = dot(&x[r * self.i..(r + 1) * self.i], &self.w[j * self.i..(j + 1) * self.i]) + self.b.as_ref().map_or(0.0, |b| b[j]);
        });
        y
    }
}

/// RMSNorm, eps 1e-5, per row of len(w).
fn rms(x: &[f32], w: &[f32]) -> Vec<f32> {
    let d = w.len();
    let mut y = x.to_vec();
    for row in y.chunks_mut(d) {
        let r = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-5).sqrt();
        row.iter_mut().zip(w).for_each(|(v, g)| *v = *v * r * g);
    }
    y
}

fn softmax(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    v.iter_mut().for_each(|x| {
        *x = (*x - m).exp();
        s += *x;
    });
    v.iter_mut().for_each(|x| *x /= s);
}

/// Attention cache rows of one sequence: keys (already rotated) and values, one row of d per position.
#[derive(Clone, Default)]
struct Kv {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Per-layer cache for a batch of sequences with a common prefix (the real candles):
/// the prefix is stored once and each sampled future only stores its own newer rows.
#[derive(Clone)]
struct Cache {
    shared: Kv,
    seqs: Vec<Kv>,
}

impl Cache {
    fn new() -> Self {
        Self { shared: Kv::default(), seqs: vec![Kv::default()] }
    }
    /// One sequence so far -> `n` sequences that all continue from it.
    fn fork(&self, n: usize) -> Self {
        let mut shared = self.shared.clone();
        shared.k.extend_from_slice(&self.seqs[0].k);
        shared.v.extend_from_slice(&self.seqs[0].v);
        Self { shared, seqs: vec![Kv::default(); n] }
    }
    fn len(&self, d: usize) -> usize {
        (self.shared.k.len() + self.seqs[0].k.len()) / d
    }
    fn row<'a>(&'a self, s: usize, j: usize, d: usize, pick: fn(&Kv) -> &Vec<f32>) -> &'a [f32] {
        let sl = self.shared.k.len() / d;
        if j < sl { &pick(&self.shared)[j * d..(j + 1) * d] } else { &pick(&self.seqs[s])[(j - sl) * d..(j - sl + 1) * d] }
    }
    fn k(&self, s: usize, j: usize, d: usize) -> &[f32] {
        self.row(s, j, d, |kv| &kv.k)
    }
    fn v(&self, s: usize, j: usize, d: usize) -> &[f32] {
        self.row(s, j, d, |kv| &kv.v)
    }
    fn append(&mut self, k: &[f32], v: &[f32]) {
        let per = k.len() / self.seqs.len();
        for (s, kv) in self.seqs.iter_mut().enumerate() {
            kv.k.extend_from_slice(&k[s * per..(s + 1) * per]);
            kv.v.extend_from_slice(&v[s * per..(s + 1) * per]);
        }
    }
}

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    inv_freq: Vec<f32>,
}

impl Attn {
    fn load(t: &mut Tensors, p: &str, heads: usize) -> Result<Self, String> {
        Ok(Self {
            q: Linear::load(t, &format!("{p}.q_proj"), true)?,
            k: Linear::load(t, &format!("{p}.k_proj"), true)?,
            v: Linear::load(t, &format!("{p}.v_proj"), true)?,
            o: Linear::load(t, &format!("{p}.out_proj"), true)?,
            heads,
            inv_freq: take(t, &format!("{p}.rotary.inv_freq"))?.1,
        })
    }

    /// RoPE on every head of one row at position `pos` (rotate_half convention).
    fn rope(&self, x: &mut [f32], pos: usize) {
        let hd = x.len() / self.heads;
        let half = hd / 2;
        for h in 0..self.heads {
            let b = h * hd;
            for t in 0..half {
                let (s, c) = (pos as f32 * self.inv_freq[t]).sin_cos();
                let (x1, x2) = (x[b + t], x[b + t + half]);
                x[b + t] = x1 * c - x2 * s;
                x[b + t + half] = x2 * c + x1 * s;
            }
        }
    }

    /// Causal self-attention. `x` holds one equal-length run of new rows per sequence in `c`
    /// (back to back), at positions pos0..; their keys/values are appended per sequence.
    /// Batching the sequences means the weights are read once per step instead of once per sample.
    fn fwd(&self, x: &[f32], pos0: usize, c: &mut Cache) -> Vec<f32> {
        let d = self.q.o;
        let n = x.len() / d;
        let per = n / c.seqs.len();
        debug_assert_eq!(c.len(d), pos0);
        let (mut q, mut k, v) = (self.q.fwd(x), self.k.fwd(x), self.v.fwd(x));
        for i in 0..n {
            self.rope(&mut q[i * d..(i + 1) * d], pos0 + i % per);
            self.rope(&mut k[i * d..(i + 1) * d], pos0 + i % per);
        }
        c.append(&k, &v);
        let (hd, c) = (d / self.heads, &*c);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0f32; n * d];
        par_for(&mut out, d, n * (pos0 + per) * d > 1 << 18, |i, row| {
            let (s, lim) = (i / per, pos0 + i % per + 1); // causal: sees itself and everything before
            let mut sc = vec![0f32; lim];
            for h in 0..self.heads {
                let qh = &q[i * d + h * hd..][..hd];
                for (j, sj) in sc.iter_mut().enumerate() {
                    *sj = dot(qh, &c.k(s, j, d)[h * hd..][..hd]) * scale;
                }
                softmax(&mut sc);
                let o = &mut row[h * hd..(h + 1) * hd];
                for (j, p) in sc.iter().enumerate() {
                    o.iter_mut().zip(&c.v(s, j, d)[h * hd..][..hd]).for_each(|(a, b)| *a += p * b);
                }
            }
        });
        self.o.fwd(&out)
    }
}

/// Pre-norm transformer block: x += attn(rms(x)); x += w2(silu(w1 h) * w3 h)
struct Block {
    n1: Vec<f32>,
    att: Attn,
    n2: Vec<f32>,
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Block {
    fn load(t: &mut Tensors, p: &str, heads: usize) -> Result<Self, String> {
        Ok(Self {
            n1: take(t, &format!("{p}.norm1.weight"))?.1,
            att: Attn::load(t, &format!("{p}.self_attn"), heads)?,
            n2: take(t, &format!("{p}.norm2.weight"))?.1,
            w1: Linear::load(t, &format!("{p}.ffn.w1"), false)?,
            w2: Linear::load(t, &format!("{p}.ffn.w2"), false)?,
            w3: Linear::load(t, &format!("{p}.ffn.w3"), false)?,
        })
    }
    fn fwd(&self, x: &mut [f32], pos0: usize, kv: &mut Cache) {
        let a = self.att.fwd(&rms(x, &self.n1), pos0, kv);
        x.iter_mut().zip(&a).for_each(|(v, a)| *v += a);
        let h = rms(x, &self.n2);
        let mut g = self.w1.fwd(&h);
        let u = self.w3.fwd(&h);
        g.iter_mut().zip(&u).for_each(|(g, u)| *g = *g / (1.0 + (-*g).exp()) * u);
        let o = self.w2.fwd(&g);
        x.iter_mut().zip(&o).for_each(|(v, o)| *v += o);
    }
}

fn run_blocks(blocks: &[Block], x: &mut [f32], pos0: usize, kv: &mut [Cache]) {
    for (b, c) in blocks.iter().zip(kv.iter_mut()) {
        b.fwd(x, pos0, c);
    }
}

pub type Token = (u32, u32); // (s1 coarse, s2 fine), 10 bits each

/// Turns normalised OHLCV+amount rows into (s1, s2) tokens and back.
struct Tokenizer {
    embed: Linear,
    enc: Vec<Block>,
    quant: Linear,
    post: Linear,
    dec: Vec<Block>,
    head: Linear,
    s1_bits: usize,
}

impl Tokenizer {
    fn load(dir: &Path) -> Result<Self, String> {
        let cfg = config(dir)?;
        let mut t = load_safetensors(&dir.join("model.safetensors"))?;
        let heads = cfg["n_heads"].as_u64().ok_or("config: n_heads")? as usize;
        let (ne, nd) = (count_blocks(&t, "encoder"), count_blocks(&t, "decoder"));
        Ok(Self {
            embed: Linear::load(&mut t, "embed", true)?,
            enc: (0..ne).map(|i| Block::load(&mut t, &format!("encoder.{i}"), heads)).collect::<Result<_, _>>()?,
            quant: Linear::load(&mut t, "quant_embed", true)?,
            post: Linear::load(&mut t, "post_quant_embed", true)?,
            dec: (0..nd).map(|i| Block::load(&mut t, &format!("decoder.{i}"), heads)).collect::<Result<_, _>>()?,
            head: Linear::load(&mut t, "head", true)?,
            s1_bits: cfg["s1_bits"].as_u64().ok_or("config: s1_bits")? as usize,
        })
    }

    /// Binary spherical quantisation: each code dimension's sign is one bit (LSB first per half).
    fn encode(&self, x: &[f32]) -> Vec<Token> {
        let mut z = self.embed.fwd(x);
        run_blocks(&self.enc, &mut z, 0, &mut vec![Cache::new(); self.enc.len()]);
        let q = self.quant.fwd(&z);
        let s1 = self.s1_bits;
        q.chunks(self.quant.o)
            .map(|r| {
                let bits = |c: &[f32]| c.iter().enumerate().map(|(k, v)| ((*v > 0.0) as u32) << k).sum::<u32>();
                (bits(&r[..s1]), bits(&r[s1..]))
            })
            .collect()
    }

    /// Decoder over tokens at positions pos0.. (causal, cached; one equal run per sequence) -> rows of 6 normalised values.
    fn decode(&self, toks: &[Token], pos0: usize, kv: &mut [Cache]) -> Vec<f32> {
        let dim = self.post.i;
        let scale = 1.0 / (dim as f32).sqrt();
        let s1 = self.s1_bits;
        let x: Vec<f32> = toks
            .iter()
            .flat_map(|&(a, b)| (0..dim).map(move |k| if (if k < s1 { a >> k } else { b >> (k - s1) }) & 1 == 1 { scale } else { -scale }))
            .collect();
        let mut z = self.post.fwd(&x);
        run_blocks(&self.dec, &mut z, pos0, kv);
        self.head.fwd(&z)
    }
}

/// The autoregressive model over tokens, plus the dependency layer that picks s2 given s1.
struct Predictor {
    d: usize,
    emb_s1: Vec<f32>,
    emb_s2: Vec<f32>,
    fusion: Linear,
    time: [Vec<f32>; 5], // minute, hour, weekday, day, month tables
    blocks: Vec<Block>,
    norm: Vec<f32>,
    dep: Attn, // cross-attention, 4 heads; its RoPE is a no-op at inference (query length 1)
    dep_norm: Vec<f32>,
    proj_s1: Linear,
    proj_s2: Linear,
}

/// Running state of a batch of sequences (1 during prefill, one per sample afterwards).
struct State {
    kv: Vec<Cache>, // per transformer layer
    dep: Cache,     // dep-layer keys/values of every position's normed hidden (no RoPE)
    last: Vec<f32>, // normed hidden of the newest position, one row per sequence
    len: usize,
}

impl State {
    fn new(layers: usize) -> Self {
        Self { kv: vec![Cache::new(); layers], dep: Cache::new(), last: vec![], len: 0 }
    }
    fn fork(&self, n: usize) -> Self {
        Self { kv: self.kv.iter().map(|c| c.fork(n)).collect(), dep: self.dep.fork(n), last: self.last.repeat(n), len: self.len }
    }
}

impl Predictor {
    fn load(dir: &Path) -> Result<Self, String> {
        let cfg = config(dir)?;
        let mut t = load_safetensors(&dir.join("model.safetensors"))?;
        let heads = cfg["n_heads"].as_u64().ok_or("config: n_heads")? as usize;
        let n = count_blocks(&t, "transformer");
        let mut tab = |k: &str| take(&mut t, &format!("time_emb.{k}_embed.weight")).map(|x| x.1);
        let time = [tab("minute")?, tab("hour")?, tab("weekday")?, tab("day")?, tab("month")?];
        let (s, emb_s1) = take(&mut t, "embedding.emb_s1.weight")?;
        Ok(Self {
            d: s[1],
            emb_s1,
            emb_s2: take(&mut t, "embedding.emb_s2.weight")?.1,
            fusion: Linear::load(&mut t, "embedding.fusion_proj", true)?,
            time,
            blocks: (0..n).map(|i| Block::load(&mut t, &format!("transformer.{i}"), heads)).collect::<Result<_, _>>()?,
            norm: take(&mut t, "norm.weight")?.1,
            dep: Attn::load(&mut t, "dep_layer.cross_attn", 4)?,
            dep_norm: take(&mut t, "dep_layer.norm.weight")?.1,
            proj_s1: Linear::load(&mut t, "head.proj_s1", true)?,
            proj_s2: Linear::load(&mut t, "head.proj_s2", true)?,
        })
    }

    /// Feeds one equal run of tokens per sequence (sharing `stamps`) at the end of `st`.
    fn feed(&self, st: &mut State, toks: &[Token], stamps: &[[usize; 5]]) {
        let d = self.d;
        let per = stamps.len();
        let sq = (d as f32).sqrt();
        let cat: Vec<f32> = toks
            .iter()
            .flat_map(|&(a, b)| {
                let e1 = self.emb_s1[a as usize * d..][..d].iter().map(move |v| v * sq);
                let e2 = self.emb_s2[b as usize * d..][..d].iter().map(move |v| v * sq);
                e1.chain(e2)
            })
            .collect();
        let mut x = self.fusion.fwd(&cat);
        for (r, row) in x.chunks_mut(d).enumerate() {
            for (tab, &i) in self.time.iter().zip(&stamps[r % per]) {
                row.iter_mut().zip(&tab[i * d..][..d]).for_each(|(v, e)| *v += e);
            }
        }
        run_blocks(&self.blocks, &mut x, st.len, &mut st.kv);
        let h = rms(&x, &self.norm);
        st.dep.append(&self.dep.k.fwd(&h), &self.dep.v.fwd(&h));
        st.last = h.chunks(d * per).flat_map(|seq| seq[(per - 1) * d..].iter().copied()).collect();
        st.len += per;
    }

    fn s1_logits(&self, st: &State) -> Vec<f32> {
        self.proj_s1.fwd(&st.last)
    }

    /// s2 logits for each sequence's newest position given its chosen s1 (dependency-aware layer:
    /// the s1 embedding attends over every position, no mask).
    fn s2_logits(&self, st: &State, s1: &[u32]) -> Vec<f32> {
        let d = self.d;
        let emb: Vec<f32> = s1.iter().flat_map(|&a| self.emb_s1[a as usize * d..][..d].iter().copied()).collect();
        let q = self.dep.q.fwd(&emb);
        let hd = d / self.dep.heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let total = st.dep.len(d);
        let mut att = vec![0f32; s1.len() * d];
        par_for(&mut att, d, s1.len() * total * d > 1 << 18, |s, row| {
            for h in 0..self.dep.heads {
                let qh = &q[s * d + h * hd..][..hd];
                let mut sc: Vec<f32> = (0..total).map(|j| dot(qh, &st.dep.k(s, j, d)[h * hd..][..hd]) * scale).collect();
                softmax(&mut sc);
                for (j, p) in sc.iter().enumerate() {
                    row[h * hd..][..hd].iter_mut().zip(&st.dep.v(s, j, d)[h * hd..][..hd]).for_each(|(a, b)| *a += p * b);
                }
            }
        });
        let a = self.dep.o.fwd(&att);
        let x2: Vec<f32> = st.last.iter().zip(&a).map(|(l, a)| l + a).collect();
        self.proj_s2.fwd(&rms(&x2, &self.dep_norm))
    }
}

fn config(dir: &Path) -> Result<serde_json::Value, String> {
    let p = dir.join("config.json");
    serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| e.to_string())
}

/// Small seeded RNG (xorshift64*) so a prediction for the same candles is reproducible.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn unit(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Temperature 1, nucleus (top-p) sampling exactly like the reference's top_k_top_p_filtering.
fn sample(logits: &[f32], top_p: f32, rng: &mut Rng) -> u32 {
    let mut p = logits.to_vec();
    softmax(&mut p);
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
    // keep tokens until the cumulative probability before them exceeds top_p (always keep the first)
    let (mut cum, mut keep) = (0f32, 0);
    for &i in &idx {
        if keep > 0 && cum > top_p {
            break;
        }
        cum += p[i];
        keep += 1;
    }
    let mut u = rng.unit() * idx[..keep].iter().map(|&i| p[i]).sum::<f32>();
    for &i in &idx[..keep] {
        u -= p[i];
        if u <= 0.0 {
            return i as u32;
        }
    }
    idx[keep - 1] as u32
}

/// (minute, hour, weekday Mon=0, day, month) of a unix time shifted to exchange local time.
pub fn stamp(t: i64) -> [usize; 5] {
    let d = chrono::DateTime::from_timestamp(t, 0).unwrap_or_default().naive_utc();
    [d.minute() as usize, d.hour() as usize, d.weekday().num_days_from_monday() as usize, d.day() as usize, d.month() as usize]
}

pub struct Kronos {
    tok: Tokenizer,
    pred: Predictor,
    pub max_context: usize,
}

pub const CLIP: f32 = 5.0;
pub const TOP_P: f32 = 0.9;

impl Kronos {
    pub fn load(predictor_dir: &Path, tokenizer_dir: &Path, max_context: usize) -> Result<Self, String> {
        Ok(Self { tok: Tokenizer::load(tokenizer_dir)?, pred: Predictor::load(predictor_dir)?, max_context })
    }

    /// x: raw rows [open, high, low, close, volume, amount]; stamps for x and for each future step.
    /// Returns one path per sample: `horizon` rows of the same 6 columns, in raw units.
    pub fn predict(&self, x: &[[f64; 6]], x_stamp: &[[usize; 5]], y_stamp: &[[usize; 5]], samples: usize, seed: u64) -> Result<Vec<Vec<[f64; 6]>>, String> {
        let (n, h) = (x.len(), y_stamp.len());
        if n == 0 || h == 0 || n != x_stamp.len() {
            return Err("empty input".into());
        }
        if n + h > self.max_context {
            return Err(format!("lookback {n} + horizon {h} exceeds model context {}", self.max_context));
        }
        // per-column z-score over the window (population std), clipped, as KronosPredictor.predict
        let mut mean = [0f64; 6];
        let mut std = [0f64; 6];
        for c in 0..6 {
            mean[c] = x.iter().map(|r| r[c]).sum::<f64>() / n as f64;
            std[c] = (x.iter().map(|r| (r[c] - mean[c]).powi(2)).sum::<f64>() / n as f64).sqrt();
        }
        let norm: Vec<f32> = x.iter().flat_map(|r| (0..6).map(move |c| (((r[c] - mean[c]) / (std[c] + 1e-5)) as f32).clamp(-CLIP, CLIP))).collect();
        let samples = samples.max(1);
        let (base, dec_base, recon) = self.prefill(&norm, x_stamp);
        // The tokenizer decodes an extreme last candle (far from the window mean) back toward the mean,
        // up to ~0.5% off. Samples continue from that decoded level, so every path would "move" by that
        // error before the model predicts anything. Re-anchor prices to the real last close.
        let decoded_last = recon[(n - 1) * 6 + 3] as f64 * (std[3] + 1e-5) + mean[3];
        let anchor = if decoded_last > 0.0 && x[n - 1][3] > 0.0 { x[n - 1][3] / decoded_last } else { 1.0 };
        let mut st = base.fork(samples);
        let toks = self.generate(&mut st, y_stamp, &mut Rng::new(seed));
        let mut dkv: Vec<Cache> = dec_base.iter().map(|c| c.fork(samples)).collect();
        let out = self.tok.decode(&toks.concat(), n, &mut dkv);
        let raw = |r: &[f32], c: usize| r[c] as f64 * (std[c] + 1e-5) + mean[c];
        // columns 0..4 = o, h, l, c and 5 = amount scale with price; 4 = volume does not
        Ok(out.chunks(h * 6).map(|seq| seq.chunks(6).map(|r| std::array::from_fn(|c| if c == 4 { raw(r, c) } else { raw(r, c) * anchor })).collect()).collect())
    }

    /// Context work shared by every sample: tokenise, run the predictor and the tokenizer decoder over it once.
    /// Also returns the context as the tokenizer decodes it (normalised units).
    fn prefill(&self, norm: &[f32], x_stamp: &[[usize; 5]]) -> (State, Vec<Cache>, Vec<f32>) {
        let toks = self.tok.encode(norm);
        let mut st = State::new(self.pred.blocks.len());
        self.pred.feed(&mut st, &toks, x_stamp);
        let mut dkv = vec![Cache::new(); self.tok.dec.len()];
        let recon = self.tok.decode(&toks, 0, &mut dkv);
        (st, dkv, recon)
    }

    /// All sampled futures advance together, one step at a time. Returns tokens per sample.
    fn generate(&self, st: &mut State, y_stamp: &[[usize; 5]], rng: &mut Rng) -> Vec<Vec<Token>> {
        let ns = st.kv[0].seqs.len();
        let mut out = vec![Vec::with_capacity(y_stamp.len()); ns];
        for (i, ys) in y_stamp.iter().enumerate() {
            let s1: Vec<u32> = self.pred.s1_logits(st).chunks(self.pred.proj_s1.o).map(|l| sample(l, TOP_P, rng)).collect();
            let s2: Vec<u32> = self.pred.s2_logits(st, &s1).chunks(self.pred.proj_s2.o).map(|l| sample(l, TOP_P, rng)).collect();
            let step: Vec<Token> = s1.into_iter().zip(s2).collect();
            out.iter_mut().zip(&step).for_each(|(o, t)| o.push(*t));
            if i + 1 < y_stamp.len() {
                self.pred.feed(st, &step, &[*ys]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_respects_top_p() {
        // one dominant token above top_p: nucleus keeps only it
        let mut l = vec![0f32; 10];
        l[3] = 10.0;
        let mut r = Rng::new(7);
        assert!((0..50).all(|_| sample(&l, 0.9, &mut r) == 3));
        // uniform: everything is reachable
        let l = vec![1f32; 4];
        let seen: std::collections::HashSet<u32> = (0..400).map(|_| sample(&l, 1.0, &mut r)).collect();
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn stamp_fields() {
        // 2026-11-11 14:30 UTC is a Wednesday
        assert_eq!(stamp(1_794_407_400), [30, 14, 2, 11, 11]);
    }

    /// Dumps internals for scratch/kronos_oracle.py to compare against the official PyTorch code.
    /// KRONOS_DUMP=<out.json> [KRONOS_MODEL=Kronos-base] cargo test --release kronos_dump -- --ignored
    #[test]
    #[ignore]
    fn kronos_dump() {
        let out = std::env::var("KRONOS_DUMP").expect("KRONOS_DUMP");
        let m = crate::shared::data_dir().join("models");
        let model = std::env::var("KRONOS_MODEL").unwrap_or("Kronos-small".into());
        let k = Kronos::load(&m.join(model), &m.join("Kronos-Tokenizer-base"), 512).unwrap();
        let n = 96;
        // deterministic random walk input, already normalised-ish
        let mut r = Rng::new(42);
        let mut px = 0f32;
        let norm: Vec<f32> = (0..n)
            .flat_map(|_| {
                px += r.unit() - 0.5;
                let (o, c) = (px, px + r.unit() - 0.5);
                let (hi, lo) = (o.max(c) + r.unit() * 0.3, o.min(c) - r.unit() * 0.3);
                let v = r.unit() * 2.0 - 1.0;
                [o, hi, lo, c, v, v * 0.5]
            })
            .collect();
        let t0 = 1_794_407_400i64;
        let xs: Vec<[usize; 5]> = (0..n as i64).map(|i| stamp(t0 + i * 60)).collect();
        let ys = [stamp(t0 + n as i64 * 60), stamp(t0 + (n as i64 + 1) * 60)];
        let toks = k.tok.encode(&norm);
        let (st0, dkv, _) = k.prefill(&norm, &xs);
        let mut st = st0.fork(1);
        let s1l = k.pred.s1_logits(&st);
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        let a1 = argmax(&s1l);
        let s2l = k.pred.s2_logits(&st, &[a1]);
        let a2 = argmax(&s2l);
        k.pred.feed(&mut st, &[(a1, a2)], &[ys[0]]);
        let s1l_next = k.pred.s1_logits(&st);
        let b1 = argmax(&s1l_next);
        let s2l_next = k.pred.s2_logits(&st, &[b1]);
        // batched: the middle of 3 sequences gets the same token as the single path -> same numbers
        let mut b = st0.fork(3);
        k.pred.feed(&mut b, &[(1, 2), (a1, a2), (900, 7)], &[ys[0]]);
        let v = k.pred.proj_s1.o;
        let bl = k.pred.s1_logits(&b);
        let b2l = k.pred.s2_logits(&b, &[5, b1, 6]);
        let md = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(md(&bl[v..2 * v], &s1l_next) < 1e-4, "batched s1 differs");
        assert!(md(&b2l[v..2 * v], &s2l_next) < 1e-4, "batched s2 differs");
        assert!(md(&bl[..v], &s1l_next) > 1e-3, "other sequences must differ");
        let mut d = dkv.clone();
        let dec = k.tok.decode(&[(a1, a2), (b1, argmax(&s2l_next))], n, &mut d);
        let mut full = vec![Cache::new(); k.tok.dec.len()];
        let dec_full = k.tok.decode(&toks, 0, &mut full);
        let j = serde_json::json!({
            "x": norm, "xs": xs, "ys": ys, "tokens": toks,
            "s1_logits": s1l, "a1": a1, "s2_logits": s2l, "a2": a2,
            "s1_logits_next": s1l_next, "b1": b1, "s2_logits_next": s2l_next,
            "dec_new": dec, "dec_ctx": dec_full,
        });
        std::fs::write(out, j.to_string()).unwrap();
    }
}

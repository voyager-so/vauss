//! Baseline JPEG decoding that reproduces libjpeg-turbo's default output bit for bit (what PIL,
//! and so DA3's reference preprocessing, decodes with): the "islow" integer IDCT, "fancy"
//! (triangle) chroma upsampling with edge-replicated context rows, and the fixed-point
//! YCbCr->RGB tables. Other decoders differ by a few levels on a few percent of pixels, which
//! moves DA3 poses measurably on ill-conditioned views.
//!
//! Only 8-bit baseline/extended Huffman files with 1 or 3 components and chroma subsampling
//! 4:4:4, 4:2:2, 4:4:0 or 4:2:0 are handled; anything else returns `None` and the caller falls
//! back to the `image` crate.
//!
//! The algorithms follow libjpeg-turbo's jidctint.c, jdsample.c and jdcolor.c (IJG license); upstream
//! copyright notices, changes and the IJG README are in ../LICENSE-IJG.txt.

use image::RgbImage;

struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// Blocks per line / column in the MCU-padded plane.
    bw: usize,
    bh: usize,
    plane: Vec<u8>,
}

struct Huffman {
    /// Per code length 1..=16: (first code, last code + 1, index of the first value).
    ranges: [(i32, i32, usize); 17],
    values: Vec<u8>,
}

impl Huffman {
    fn new(counts: &[u8; 16], values: Vec<u8>) -> Self {
        let mut ranges = [(0, 0, 0); 17];
        let (mut code, mut k) = (0i32, 0usize);
        for len in 1..=16 {
            let n = counts[len - 1] as usize;
            ranges[len] = (code, code + n as i32, k);
            code = (code + n as i32) << 1;
            k += n;
        }
        Self { ranges, values }
    }
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u32,
    n: u32,
}

impl Bits<'_> {
    fn bit(&mut self) -> u32 {
        if self.n == 0 {
            // Past a marker or the end of data libjpeg feeds zeros.
            let mut byte = 0;
            if self.pos < self.data.len() && self.data[self.pos] != 0xFF {
                byte = self.data[self.pos];
                self.pos += 1;
            } else if self.pos + 1 < self.data.len() && self.data[self.pos + 1] == 0 {
                byte = 0xFF;
                self.pos += 2;
            }
            self.acc = u32::from(byte);
            self.n = 8;
        }
        self.n -= 1;
        (self.acc >> self.n) & 1
    }

    fn bits(&mut self, count: u32) -> i32 {
        (0..count).fold(0, |v, _| (v << 1) | self.bit() as i32)
    }

    fn decode(&mut self, table: &Huffman) -> Option<u8> {
        let mut code = 0i32;
        for len in 1..=16 {
            code = (code << 1) | self.bit() as i32;
            let (first, end, k) = table.ranges[len];
            if code < end {
                return table.values.get(k + (code - first) as usize).copied();
            }
        }
        None
    }

    fn receive_extend(&mut self, s: u32) -> i32 {
        if s == 0 {
            return 0;
        }
        let v = self.bits(s);
        if v < 1 << (s - 1) { v - (1 << s) + 1 } else { v }
    }

    /// Skip to the next restart marker (byte-aligned).
    fn restart(&mut self) {
        self.n = 0;
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF && (0xD0..=0xD7).contains(&self.data[self.pos + 1]) {
                self.pos += 2;
                return;
            }
            self.pos += 1;
        }
    }
}

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

pub fn decode(data: &[u8]) -> Option<RgbImage> {
    if !data.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut qt = [[0i32; 64]; 4];
    let mut dc: [Option<Huffman>; 4] = Default::default();
    let mut ac: [Option<Huffman>; 4] = Default::default();
    let (mut width, mut height) = (0usize, 0usize);
    let mut comps: Vec<Component> = Vec::new();
    let mut restart_interval = 0usize;
    let mut pos = 2;
    let (mut hmax, mut vmax) = (1, 1);
    loop {
        while pos < data.len() && data[pos] != 0xFF {
            pos += 1;
        }
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }
        let marker = *data.get(pos)?;
        pos += 1;
        if marker == 0xD9 {
            break;
        }
        if (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            continue;
        }
        let len = usize::from(u16::from_be_bytes([*data.get(pos)?, *data.get(pos + 1)?]));
        let seg = data.get(pos + 2..pos + len)?;
        pos += len;
        match marker {
            0xDB => {
                let mut s = seg;
                while !s.is_empty() {
                    let (precision, id) = (s[0] >> 4, usize::from(s[0] & 15));
                    if precision != 0 || id > 3 {
                        return None;
                    }
                    for k in 0..64 {
                        qt[id][ZIGZAG[k]] = i32::from(*s.get(1 + k)?);
                    }
                    s = &s[65..];
                }
            }
            0xC4 => {
                let mut s = seg;
                while !s.is_empty() {
                    let (class, id) = (s[0] >> 4, usize::from(s[0] & 15));
                    let counts: [u8; 16] = s.get(1..17)?.try_into().ok()?;
                    let n: usize = counts.iter().map(|&c| usize::from(c)).sum();
                    let table = Huffman::new(&counts, s.get(17..17 + n)?.to_vec());
                    match (class, id) {
                        (0, 0..=3) => dc[id] = Some(table),
                        (1, 0..=3) => ac[id] = Some(table),
                        _ => return None,
                    }
                    s = &s[17 + n..];
                }
            }
            0xDD => restart_interval = usize::from(u16::from_be_bytes([*seg.first()?, *seg.get(1)?])),
            // Baseline and extended sequential Huffman only.
            0xC0 | 0xC1 => {
                if seg[0] != 8 {
                    return None;
                }
                height = usize::from(u16::from_be_bytes([seg[1], seg[2]]));
                width = usize::from(u16::from_be_bytes([seg[3], seg[4]]));
                let n = usize::from(seg[5]);
                if width == 0 || height == 0 || !(n == 1 || n == 3) {
                    return None;
                }
                for c in 0..n {
                    let s = seg.get(6 + 3 * c..9 + 3 * c)?;
                    comps.push(Component {
                        id: s[0],
                        h: usize::from(s[1] >> 4),
                        v: usize::from(s[1] & 15),
                        tq: usize::from(s[2] & 3),
                        bw: 0,
                        bh: 0,
                        plane: Vec::new(),
                    });
                }
                if n == 1 {
                    comps[0].h = 1;
                    comps[0].v = 1;
                }
                hmax = comps.iter().map(|c| c.h).max()?;
                vmax = comps.iter().map(|c| c.v).max()?;
                let (mcux, mcuy) = (width.div_ceil(8 * hmax), height.div_ceil(8 * vmax));
                for c in &mut comps {
                    if !(1..=2).contains(&c.h) || !(1..=2).contains(&c.v) || hmax % c.h != 0 || vmax % c.v != 0 {
                        return None;
                    }
                    c.bw = mcux * c.h;
                    c.bh = mcuy * c.v;
                    c.plane = vec![0; c.bw * 8 * c.bh * 8];
                }
            }
            0xC2 | 0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => return None,
            0xDA => {
                if comps.is_empty() {
                    return None;
                }
                let ns = usize::from(seg[0]);
                let mut scan = Vec::with_capacity(ns);
                for k in 0..ns {
                    let (id, tables) = (seg[1 + 2 * k], seg[2 + 2 * k]);
                    let ci = comps.iter().position(|c| c.id == id)?;
                    scan.push((ci, usize::from(tables >> 4), usize::from(tables & 15)));
                }
                // Spectral selection / approximation must be the full sequential range.
                let tail = seg.get(1 + 2 * ns..1 + 2 * ns + 3)?;
                if tail != [0, 63, 0] {
                    return None;
                }
                let end = scan_end(data, pos);
                decode_scan(&data[pos..end], &mut comps, &scan, &qt, &dc, &ac, restart_interval, hmax, vmax, width, height)?;
                pos = end;
            }
            _ => {}
        }
    }
    if comps.is_empty() || comps.iter().any(|c| c.plane.is_empty()) {
        return None;
    }
    to_rgb(&comps, width, height, hmax, vmax)
}

/// The first marker after entropy-coded data that is not a stuffed byte or a restart marker.
fn scan_end(data: &[u8], mut pos: usize) -> usize {
    while pos + 1 < data.len() {
        if data[pos] == 0xFF && data[pos + 1] != 0 && !(0xD0..=0xD7).contains(&data[pos + 1]) {
            return pos;
        }
        pos += 1;
    }
    data.len()
}

#[allow(clippy::too_many_arguments)]
fn decode_scan(
    data: &[u8],
    comps: &mut [Component],
    scan: &[(usize, usize, usize)],
    qt: &[[i32; 64]; 4],
    dc: &[Option<Huffman>; 4],
    ac: &[Option<Huffman>; 4],
    restart_interval: usize,
    hmax: usize,
    vmax: usize,
    width: usize,
    height: usize,
) -> Option<()> {
    let mut bits = Bits { data, pos: 0, acc: 0, n: 0 };
    let mut pred = vec![0i32; scan.len()];
    // A single-component scan walks that component's own (unpadded) block grid.
    let single = scan.len() == 1;
    let (units_x, units_y) = if single {
        let c = &comps[scan[0].0];
        ((width * c.h).div_ceil(hmax).div_ceil(8), (height * c.v).div_ceil(vmax).div_ceil(8))
    } else {
        (width.div_ceil(8 * hmax), height.div_ceil(8 * vmax))
    };
    let mut block = [0i32; 64];
    for unit in 0..units_x * units_y {
        if restart_interval > 0 && unit > 0 && unit % restart_interval == 0 {
            bits.restart();
            pred.iter_mut().for_each(|p| *p = 0);
        }
        let (ux, uy) = (unit % units_x, unit / units_x);
        for (si, &(ci, td, ta)) in scan.iter().enumerate() {
            let (h, v) = if single { (1, 1) } else { (comps[ci].h, comps[ci].v) };
            for by in 0..v {
                for bx in 0..h {
                    block.fill(0);
                    let s = bits.decode(dc[td].as_ref()?)?;
                    pred[si] += bits.receive_extend(u32::from(s));
                    block[0] = pred[si];
                    let mut k = 1;
                    while k < 64 {
                        let rs = bits.decode(ac[ta].as_ref()?)?;
                        let (r, s) = (usize::from(rs >> 4), u32::from(rs & 15));
                        if s == 0 {
                            if r != 15 {
                                break;
                            }
                            k += 16;
                            continue;
                        }
                        k += r;
                        if k > 63 {
                            break;
                        }
                        block[ZIGZAG[k]] = bits.receive_extend(s);
                        k += 1;
                    }
                    let c = &mut comps[ci];
                    let (bxi, byi) = if single { (ux, uy) } else { (ux * c.h + bx, uy * c.v + by) };
                    let stride = c.bw * 8;
                    idct_islow(&block, &qt[c.tq], &mut c.plane[byi * 8 * stride + bxi * 8..], stride);
                }
            }
        }
    }
    Some(())
}

/// jidctint.c `jpeg_idct_islow`: CONST_BITS 13, PASS1_BITS 2.
fn idct_islow(coef: &[i32; 64], q: &[i32; 64], out: &mut [u8], stride: usize) {
    const CB: i64 = 13;
    const P1: i64 = 2;
    #[allow(clippy::too_many_arguments)]
    fn butterfly(s0: i64, s1: i64, s2: i64, s3: i64, s4: i64, s5: i64, s6: i64, s7: i64) -> [i64; 8] {
        let z1 = (s2 + s6) * 4433;
        let tmp2 = z1 + s6 * -15137;
        let tmp3 = z1 + s2 * 6270;
        let tmp0 = (s0 + s4) << CB;
        let tmp1 = (s0 - s4) << CB;
        let (t10, t13, t11, t12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
        let (o0, o1, o2, o3) = (s7, s5, s3, s1);
        let (z1, z2, z3, z4) = (o0 + o3, o1 + o2, o0 + o2, o1 + o3);
        let z5 = (z3 + z4) * 9633;
        let (o0, o1, o2, o3) = (o0 * 2446, o1 * 16819, o2 * 25172, o3 * 12299);
        let (z1, z2) = (z1 * -7373, z2 * -20995);
        let (z3, z4) = (z3 * -16069 + z5, z4 * -3196 + z5);
        let (o0, o1, o2, o3) = (o0 + z1 + z3, o1 + z2 + z4, o2 + z2 + z3, o3 + z1 + z4);
        [t10 + o3, t11 + o2, t12 + o1, t13 + o0, t13 - o0, t12 - o1, t11 - o2, t10 - o3]
    }
    let descale = |x: i64, n: i64| (x + (1 << (n - 1))) >> n;
    let mut ws = [0i64; 64];
    for c in 0..8 {
        let s: [i64; 8] = std::array::from_fn(|r| i64::from(coef[r * 8 + c] * q[r * 8 + c]));
        let col = butterfly(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
        for r in 0..8 {
            ws[r * 8 + c] = descale(col[r], CB - P1);
        }
    }
    for r in 0..8 {
        let s = &ws[r * 8..r * 8 + 8];
        let row = butterfly(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
        for c in 0..8 {
            out[r * stride + c] = (descale(row[c], CB + P1 + 3) + 128).clamp(0, 255) as u8;
        }
    }
}

/// Upsample to full resolution (libjpeg-turbo "fancy" filters) and convert to RGB.
fn to_rgb(comps: &[Component], width: usize, height: usize, hmax: usize, vmax: usize) -> Option<RgbImage> {
    let full: Vec<Vec<u8>> = comps
        .iter()
        .map(|c| {
            // The component's real (unpadded) size; context beyond it replicates the edge.
            let (w, h) = ((width * c.h).div_ceil(hmax), (height * c.v).div_ceil(vmax));
            let stride = c.bw * 8;
            let px = |x: usize, y: usize| i32::from(c.plane[y.min(h - 1) * stride + x.min(w - 1)]);
            let mut out = vec![0u8; width * height];
            match (hmax / c.h, vmax / c.v) {
                (1, 1) => {
                    for y in 0..height {
                        for x in 0..width {
                            out[y * width + x] = px(x, y) as u8;
                        }
                    }
                }
                (2, 1) => {
                    for y in 0..height {
                        for x in 0..width {
                            out[y * width + x] = h2v1(&px, x, y, w) as u8;
                        }
                    }
                }
                (1, 2) => {
                    for y in 0..height {
                        let (iy, below) = (y / 2, y % 2 == 1);
                        let other = if below { iy + 1 } else { iy.saturating_sub(1) };
                        let bias = if below { 2 } else { 1 };
                        for x in 0..width {
                            out[y * width + x] = ((px(x, iy) * 3 + px(x, other) + bias) >> 2) as u8;
                        }
                    }
                }
                _ => {
                    for y in 0..height {
                        let (iy, below) = (y / 2, y % 2 == 1);
                        let other = if below { iy + 1 } else { iy.saturating_sub(1) };
                        let col = |x: usize| px(x, iy) * 3 + px(x, other);
                        for x in 0..width {
                            let ix = x / 2;
                            let this = col(ix);
                            out[y * width + x] = if w == 1 {
                                ((this * 4 + 8) >> 4) as u8
                            } else if x % 2 == 0 {
                                let last = if ix == 0 { this } else { col(ix - 1) };
                                ((this * 3 + last + 8) >> 4) as u8
                            } else if ix + 1 >= w {
                                ((this * 4 + 7) >> 4) as u8
                            } else {
                                ((this * 3 + col(ix + 1) + 7) >> 4) as u8
                            };
                        }
                    }
                }
            }
            out
        })
        .collect();
    let mut rgb = vec![0u8; width * height * 3];
    if full.len() == 1 {
        for (p, &y) in rgb.chunks_exact_mut(3).zip(&full[0]) {
            p.fill(y);
        }
    } else {
        // jdcolor.c tables, SCALEBITS 16.
        let fix = |x: f64| (x * 65536.0 + 0.5) as i64;
        let half = 1i64 << 15;
        for (i, p) in rgb.chunks_exact_mut(3).enumerate() {
            let y = i64::from(full[0][i]);
            let cb = i64::from(full[1][i]) - 128;
            let cr = i64::from(full[2][i]) - 128;
            let r = y + ((fix(1.40200) * cr + half) >> 16);
            let g = y + ((-fix(0.34414) * cb + half - fix(0.71414) * cr) >> 16);
            let b = y + ((fix(1.77200) * cb + half) >> 16);
            p[0] = r.clamp(0, 255) as u8;
            p[1] = g.clamp(0, 255) as u8;
            p[2] = b.clamp(0, 255) as u8;
        }
    }
    RgbImage::from_raw(width as u32, height as u32, rgb)
}

/// jdsample.c `h2v1_fancy_upsample` for output column `x` of row `y`.
fn h2v1(px: &impl Fn(usize, usize) -> i32, x: usize, y: usize, w: usize) -> i32 {
    let ix = x / 2;
    let this = px(ix, y);
    if w == 1 {
        this
    } else if x % 2 == 0 {
        if ix == 0 { this } else { (this * 3 + px(ix - 1, y) + 1) >> 2 }
    } else if ix + 1 >= w {
        this
    } else {
        (this * 3 + px(ix + 1, y) + 2) >> 2
    }
}

//! Canvas access for kernels (stage 0 of the port, Derek's review point 1).
//!
//! A kernel sees the wet paste-paint state only through [`WetCanvas`]:
//! `px(y, x)` hands out every field of one pixel at once ([`WetPx`]), and
//! `px2` hands out two distinct pixels (ploughing moves paint between them).
//! Read-only planes (`tooth`, `mask`) are plain lookups.
//!
//! Two backends:
//! * [`DenseWet`] wraps the full-size arrays the Python engine owns today
//!   (numpy, row-major, `(H, W)` and `(H, W, NB)`). Zero copies.
//! * [`TiledWet`] stores 64x64 tiles that are allocated on first touch.
//!   This is the stage B layout: untouched canvas costs nothing, and an idle
//!   tile can later be packed to f16 and unpacked to f32 inside `px` while it
//!   is wet or being written (never accumulate in f16).
//!
//! Number types follow the Python engine: canvas fields are f32, the
//! kernels compute in f64 and round on store, exactly as numba does when it
//! mixes float32 arrays with float64 scalars.

use crate::NB;

/// Mutable view of every wet-paint field at one pixel.
pub struct WetPx<'a> {
    /// Optical depth K*x of the top wet film, per band.
    pub film_k: &'a mut [f32],
    /// Optical depth S*x of the top wet film, per band.
    pub film_s: &'a mut [f32],
    /// Older wet paint underneath the latest stroke, per band.
    pub base_k: &'a mut [f32],
    pub base_s: &'a mut [f32],
    pub film_vol: &'a mut f32,
    pub base_vol: &'a mut f32,
    pub wet: &'a mut f32,
    pub gloss: &'a mut f32,
    pub body: &'a mut f32,
    pub drate: &'a mut f32,
    /// Id of the stroke that owns the top film.
    pub top_id: &'a mut i32,
}

/// What a paste-paint kernel may do to the canvas.
pub trait WetCanvas {
    fn height(&self) -> usize;
    fn width(&self) -> usize;
    /// Normalised local relief, about -1.6..1.6.
    fn tooth(&self, y: usize, x: usize) -> f32;
    /// Protection 0..1 (1 = fully masked).
    fn mask(&self, y: usize, x: usize) -> f32;
    /// All wet fields of pixel (y, x).
    fn px(&mut self, y: usize, x: usize) -> WetPx<'_>;
    /// Two distinct pixels at once. Panics if `a == b`.
    fn px2(&mut self, a: (usize, usize), b: (usize, usize)) -> (WetPx<'_>, WetPx<'_>);
}

// --------------------------------------------------------------------------
// Shared raw-pointer plumbing. Each backend finds the base pointers of the
// storage that holds a pixel and its index inside that storage; `px_at`
// turns them into a WetPx. Two WetPx built from the same pointers are
// disjoint as long as their indices differ, which px2 asserts.
// --------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct Ptrs {
    film_k: *mut f32,
    film_s: *mut f32,
    base_k: *mut f32,
    base_s: *mut f32,
    film_vol: *mut f32,
    base_vol: *mut f32,
    wet: *mut f32,
    gloss: *mut f32,
    body: *mut f32,
    drate: *mut f32,
    top_id: *mut i32,
}

/// # Safety
/// `p` must point into live storage with at least `i + 1` pixels, and no
/// other live reference may alias pixel `i` for the returned lifetime.
unsafe fn px_at<'b>(p: Ptrs, i: usize) -> WetPx<'b> {
    WetPx {
        film_k: std::slice::from_raw_parts_mut(p.film_k.add(i * NB), NB),
        film_s: std::slice::from_raw_parts_mut(p.film_s.add(i * NB), NB),
        base_k: std::slice::from_raw_parts_mut(p.base_k.add(i * NB), NB),
        base_s: std::slice::from_raw_parts_mut(p.base_s.add(i * NB), NB),
        film_vol: &mut *p.film_vol.add(i),
        base_vol: &mut *p.base_vol.add(i),
        wet: &mut *p.wet.add(i),
        gloss: &mut *p.gloss.add(i),
        body: &mut *p.body.add(i),
        drate: &mut *p.drate.add(i),
        top_id: &mut *p.top_id.add(i),
    }
}

// --------------------------------------------------------------------------
// Dense backend: the Python engine's full-size arrays
// --------------------------------------------------------------------------

/// The wet-paint arrays of a canvas, row-major. Spectral arrays have
/// `H * W * NB` elements, scalar planes `H * W`.
pub struct WetArrays<'a> {
    pub film_k: &'a mut [f32],
    pub film_s: &'a mut [f32],
    pub base_k: &'a mut [f32],
    pub base_s: &'a mut [f32],
    pub film_vol: &'a mut [f32],
    pub base_vol: &'a mut [f32],
    pub wet: &'a mut [f32],
    pub gloss: &'a mut [f32],
    pub body: &'a mut [f32],
    pub drate: &'a mut [f32],
    pub top_id: &'a mut [i32],
}

pub struct DenseWet<'a> {
    h: usize,
    w: usize,
    a: WetArrays<'a>,
    tooth: &'a [f32],
    mask: &'a [f32],
}

impl<'a> DenseWet<'a> {
    pub fn new(h: usize, w: usize, a: WetArrays<'a>, tooth: &'a [f32], mask: &'a [f32]) -> Self {
        let n = h * w;
        for (name, len) in [
            ("film_K", a.film_k.len()),
            ("film_S", a.film_s.len()),
            ("base_K", a.base_k.len()),
            ("base_S", a.base_s.len()),
        ] {
            assert_eq!(len, n * NB, "{name}: expected H*W*NB elements");
        }
        for (name, len) in [
            ("film_vol", a.film_vol.len()),
            ("base_vol", a.base_vol.len()),
            ("wet", a.wet.len()),
            ("gloss", a.gloss.len()),
            ("body", a.body.len()),
            ("drate", a.drate.len()),
            ("top_id", a.top_id.len()),
            ("tooth", tooth.len()),
            ("mask", mask.len()),
        ] {
            assert_eq!(len, n, "{name}: expected H*W elements");
        }
        DenseWet { h, w, a, tooth, mask }
    }

    fn ptrs(&mut self) -> Ptrs {
        Ptrs {
            film_k: self.a.film_k.as_mut_ptr(),
            film_s: self.a.film_s.as_mut_ptr(),
            base_k: self.a.base_k.as_mut_ptr(),
            base_s: self.a.base_s.as_mut_ptr(),
            film_vol: self.a.film_vol.as_mut_ptr(),
            base_vol: self.a.base_vol.as_mut_ptr(),
            wet: self.a.wet.as_mut_ptr(),
            gloss: self.a.gloss.as_mut_ptr(),
            body: self.a.body.as_mut_ptr(),
            drate: self.a.drate.as_mut_ptr(),
            top_id: self.a.top_id.as_mut_ptr(),
        }
    }
}

impl WetCanvas for DenseWet<'_> {
    #[inline]
    fn height(&self) -> usize {
        self.h
    }
    #[inline]
    fn width(&self) -> usize {
        self.w
    }
    #[inline]
    fn tooth(&self, y: usize, x: usize) -> f32 {
        self.tooth[y * self.w + x]
    }
    #[inline]
    fn mask(&self, y: usize, x: usize) -> f32 {
        self.mask[y * self.w + x]
    }
    #[inline]
    fn px(&mut self, y: usize, x: usize) -> WetPx<'_> {
        assert!(y < self.h && x < self.w);
        let p = self.ptrs();
        // SAFETY: in bounds (checked), single pixel, tied to &mut self.
        unsafe { px_at(p, y * self.w + x) }
    }
    fn px2(&mut self, a: (usize, usize), b: (usize, usize)) -> (WetPx<'_>, WetPx<'_>) {
        assert!(a != b, "px2 needs two distinct pixels");
        assert!(a.0 < self.h && a.1 < self.w && b.0 < self.h && b.1 < self.w);
        let p = self.ptrs();
        // SAFETY: both in bounds, distinct indices -> disjoint memory.
        unsafe { (px_at(p, a.0 * self.w + a.1), px_at(p, b.0 * self.w + b.1)) }
    }
}

// --------------------------------------------------------------------------
// Tiled backend: the stage B layout
// --------------------------------------------------------------------------

/// Tile edge in pixels.
pub const TILE: usize = 64;
const TPX: usize = TILE * TILE;

/// All wet fields of one 64x64 tile, zero-initialised (a fresh canvas).
pub struct WetTile {
    film_k: Box<[f32]>,
    film_s: Box<[f32]>,
    base_k: Box<[f32]>,
    base_s: Box<[f32]>,
    film_vol: Box<[f32]>,
    base_vol: Box<[f32]>,
    wet: Box<[f32]>,
    gloss: Box<[f32]>,
    body: Box<[f32]>,
    drate: Box<[f32]>,
    top_id: Box<[i32]>,
}

impl WetTile {
    fn new() -> Self {
        let z = |n: usize| vec![0f32; n].into_boxed_slice();
        WetTile {
            film_k: z(TPX * NB),
            film_s: z(TPX * NB),
            base_k: z(TPX * NB),
            base_s: z(TPX * NB),
            film_vol: z(TPX),
            base_vol: z(TPX),
            wet: z(TPX),
            gloss: z(TPX),
            body: z(TPX),
            drate: z(TPX),
            top_id: vec![0i32; TPX].into_boxed_slice(),
        }
    }

    fn ptrs(&mut self) -> Ptrs {
        Ptrs {
            film_k: self.film_k.as_mut_ptr(),
            film_s: self.film_s.as_mut_ptr(),
            base_k: self.base_k.as_mut_ptr(),
            base_s: self.base_s.as_mut_ptr(),
            film_vol: self.film_vol.as_mut_ptr(),
            base_vol: self.base_vol.as_mut_ptr(),
            wet: self.wet.as_mut_ptr(),
            gloss: self.gloss.as_mut_ptr(),
            body: self.body.as_mut_ptr(),
            drate: self.drate.as_mut_ptr(),
            top_id: self.top_id.as_mut_ptr(),
        }
    }

    /// Approximate heap bytes of one tile.
    pub const BYTES: usize = TPX * (4 * NB * 4 + 7 * 4);
}

/// Wet state stored in lazily allocated 64x64 tiles.
pub struct TiledWet {
    h: usize,
    w: usize,
    tiles_x: usize,
    tiles: Vec<Option<Box<WetTile>>>,
    tooth: Vec<f32>,
    mask: Vec<f32>,
}

impl TiledWet {
    /// An all-dry canvas: no tile is allocated until paint arrives.
    pub fn new(h: usize, w: usize, tooth: Vec<f32>, mask: Vec<f32>) -> Self {
        assert_eq!(tooth.len(), h * w);
        assert_eq!(mask.len(), h * w);
        let tiles_x = w.div_ceil(TILE);
        let tiles_y = h.div_ceil(TILE);
        let mut tiles = Vec::with_capacity(tiles_x * tiles_y);
        tiles.resize_with(tiles_x * tiles_y, || None);
        TiledWet { h, w, tiles_x, tiles, tooth, mask }
    }

    /// How many tiles exist (memory actually used ~ this x WetTile::BYTES).
    pub fn allocated_tiles(&self) -> usize {
        self.tiles.iter().filter(|t| t.is_some()).count()
    }

    /// Base pointers of the tile holding (y, x), allocating it if needed,
    /// and the pixel's index inside that tile.
    fn tile_ptrs(&mut self, y: usize, x: usize) -> (Ptrs, usize) {
        assert!(y < self.h && x < self.w);
        let t = (y / TILE) * self.tiles_x + x / TILE;
        let tile = self.tiles[t].get_or_insert_with(|| Box::new(WetTile::new()));
        (tile.ptrs(), (y % TILE) * TILE + x % TILE)
    }

    /// Copy dense arrays (e.g. a numpy canvas) into tiles. Tiles whose
    /// fields are all zero stay unallocated.
    pub fn from_dense(h: usize, w: usize, a: &WetArrays<'_>, tooth: &[f32], mask: &[f32]) -> Self {
        let mut t = TiledWet::new(h, w, tooth.to_vec(), mask.to_vec());
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let spec = |s: &[f32]| s[i * NB..(i + 1) * NB].iter().any(|v| *v != 0.0);
                let any = spec(a.film_k)
                    || spec(a.film_s)
                    || spec(a.base_k)
                    || spec(a.base_s)
                    || [a.film_vol[i], a.base_vol[i], a.wet[i], a.gloss[i], a.body[i], a.drate[i]]
                        .iter()
                        .any(|v| *v != 0.0)
                    || a.top_id[i] != 0;
                if !any {
                    continue;
                }
                let p = t.px(y, x);
                p.film_k.copy_from_slice(&a.film_k[i * NB..(i + 1) * NB]);
                p.film_s.copy_from_slice(&a.film_s[i * NB..(i + 1) * NB]);
                p.base_k.copy_from_slice(&a.base_k[i * NB..(i + 1) * NB]);
                p.base_s.copy_from_slice(&a.base_s[i * NB..(i + 1) * NB]);
                *p.film_vol = a.film_vol[i];
                *p.base_vol = a.base_vol[i];
                *p.wet = a.wet[i];
                *p.gloss = a.gloss[i];
                *p.body = a.body[i];
                *p.drate = a.drate[i];
                *p.top_id = a.top_id[i];
            }
        }
        t
    }

    /// Write the tiles back into dense arrays (absent tiles become zeros).
    pub fn to_dense(&mut self, a: &mut WetArrays<'_>) {
        let (h, w) = (self.h, self.w);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let t = (y / TILE) * self.tiles_x + x / TILE;
                if self.tiles[t].is_none() {
                    for s in [&mut *a.film_k, &mut *a.film_s, &mut *a.base_k, &mut *a.base_s] {
                        s[i * NB..(i + 1) * NB].fill(0.0);
                    }
                    for s in [
                        &mut *a.film_vol,
                        &mut *a.base_vol,
                        &mut *a.wet,
                        &mut *a.gloss,
                        &mut *a.body,
                        &mut *a.drate,
                    ] {
                        s[i] = 0.0;
                    }
                    a.top_id[i] = 0;
                    continue;
                }
                let p = self.px(y, x);
                a.film_k[i * NB..(i + 1) * NB].copy_from_slice(p.film_k);
                a.film_s[i * NB..(i + 1) * NB].copy_from_slice(p.film_s);
                a.base_k[i * NB..(i + 1) * NB].copy_from_slice(p.base_k);
                a.base_s[i * NB..(i + 1) * NB].copy_from_slice(p.base_s);
                a.film_vol[i] = *p.film_vol;
                a.base_vol[i] = *p.base_vol;
                a.wet[i] = *p.wet;
                a.gloss[i] = *p.gloss;
                a.body[i] = *p.body;
                a.drate[i] = *p.drate;
                a.top_id[i] = *p.top_id;
            }
        }
    }
}

impl WetCanvas for TiledWet {
    #[inline]
    fn height(&self) -> usize {
        self.h
    }
    #[inline]
    fn width(&self) -> usize {
        self.w
    }
    #[inline]
    fn tooth(&self, y: usize, x: usize) -> f32 {
        self.tooth[y * self.w + x]
    }
    #[inline]
    fn mask(&self, y: usize, x: usize) -> f32 {
        self.mask[y * self.w + x]
    }
    #[inline]
    fn px(&mut self, y: usize, x: usize) -> WetPx<'_> {
        let (p, i) = self.tile_ptrs(y, x);
        // SAFETY: the tile's boxes live as long as self; one pixel.
        unsafe { px_at(p, i) }
    }
    fn px2(&mut self, a: (usize, usize), b: (usize, usize)) -> (WetPx<'_>, WetPx<'_>) {
        assert!(a != b, "px2 needs two distinct pixels");
        // Allocating tile b never moves tile a's boxed buffers.
        let (pa, ia) = self.tile_ptrs(a.0, a.1);
        let (pb, ib) = self.tile_ptrs(b.0, b.1);
        // SAFETY: distinct pixels -> distinct (tile, index) -> disjoint memory.
        unsafe { (px_at(pa, ia), px_at(pb, ib)) }
    }
}

// --------------------------------------------------------------------------
// General views for arrays outside the wet paste film (stage A, for tools,
// watercolour and pastel kernels). Kernels index them with absolute (y, x)
// through these methods only, never with raw `y * w + x`, so stage B can
// back them with tiles or a bbox window without touching the kernels.
// --------------------------------------------------------------------------

/// Read-only H x W plane.
#[derive(Clone, Copy)]
pub struct Plane<'a, T> {
    h: usize,
    w: usize,
    data: &'a [T],
}

impl<'a, T: Copy> Plane<'a, T> {
    pub fn new(h: usize, w: usize, data: &'a [T]) -> Self {
        assert_eq!(data.len(), h * w, "Plane: expected H*W elements");
        Plane { h, w, data }
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }
    #[inline]
    pub fn at(&self, y: usize, x: usize) -> T {
        self.data[y * self.w + x]
    }
}

/// Mutable H x W plane.
pub struct PlaneMut<'a, T> {
    h: usize,
    w: usize,
    data: &'a mut [T],
}

impl<'a, T: Copy> PlaneMut<'a, T> {
    pub fn new(h: usize, w: usize, data: &'a mut [T]) -> Self {
        assert_eq!(data.len(), h * w, "PlaneMut: expected H*W elements");
        PlaneMut { h, w, data }
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }
    #[inline]
    pub fn at(&self, y: usize, x: usize) -> T {
        self.data[y * self.w + x]
    }
    #[inline]
    pub fn set(&mut self, y: usize, x: usize, v: T) {
        self.data[y * self.w + x] = v;
    }
    #[inline]
    pub fn get_mut(&mut self, y: usize, x: usize) -> &mut T {
        &mut self.data[y * self.w + x]
    }
}

/// Read-only H x W x C array (C = NB for spectral arrays, 3 for RGB, ...).
#[derive(Clone, Copy)]
pub struct Spec<'a, T> {
    h: usize,
    w: usize,
    c: usize,
    data: &'a [T],
}

impl<'a, T: Copy> Spec<'a, T> {
    pub fn new(h: usize, w: usize, c: usize, data: &'a [T]) -> Self {
        assert_eq!(data.len(), h * w * c, "Spec: expected H*W*C elements");
        Spec { h, w, c, data }
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }
    #[inline]
    pub fn channels(&self) -> usize {
        self.c
    }
    #[inline]
    pub fn px(&self, y: usize, x: usize) -> &[T] {
        let i = (y * self.w + x) * self.c;
        &self.data[i..i + self.c]
    }
}

/// Mutable H x W x C array.
pub struct SpecMut<'a, T> {
    h: usize,
    w: usize,
    c: usize,
    data: &'a mut [T],
}

impl<'a, T: Copy> SpecMut<'a, T> {
    pub fn new(h: usize, w: usize, c: usize, data: &'a mut [T]) -> Self {
        assert_eq!(data.len(), h * w * c, "SpecMut: expected H*W*C elements");
        SpecMut { h, w, c, data }
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }
    #[inline]
    pub fn channels(&self) -> usize {
        self.c
    }
    #[inline]
    pub fn px(&self, y: usize, x: usize) -> &[T] {
        let i = (y * self.w + x) * self.c;
        &self.data[i..i + self.c]
    }
    #[inline]
    pub fn px_mut(&mut self, y: usize, x: usize) -> &mut [T] {
        let i = (y * self.w + x) * self.c;
        &mut self.data[i..i + self.c]
    }
}

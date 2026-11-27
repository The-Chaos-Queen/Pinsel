//! pinsel-core: the Rust core of the pinsel painting engine.
//!
//! Kernels never touch canvas memory directly. They go through the access
//! types in [`grid`], which today wrap the numpy arrays of the Python engine
//! ([`grid::DenseWet`]) and in stage B are backed by lazily allocated tiles
//! ([`grid::TiledWet`]). A kernel is written once and runs on both.

pub mod grid;
pub mod paste;

/// Number of spectral bands (380-740 nm), as `pinsel.spectral.NB`.
pub const NB: usize = 12;

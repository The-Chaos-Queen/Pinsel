//! Bindings for pinsel_core::tools_film. Register each function in `register`.
//! Python names carry the module prefix (see docs/RUST_PORTING_GUIDE.md).

#[allow(unused_imports)]
use crate::util::{hw, hw_mut, ro, rw};
use pyo3::prelude::*;

pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

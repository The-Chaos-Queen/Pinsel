//! Shared helpers for the bindings: numpy arrays -> slices with clear errors.

use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Read-only C-contiguous slice of a numpy array.
pub fn ro<'a, T: numpy::Element>(a: &'a PyReadonlyArrayDyn<'_, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("pinsel_core: {name} must be a C-contiguous array")))
}

/// Mutable C-contiguous slice of a numpy array.
pub fn rw<'a, T: numpy::Element>(a: &'a mut PyReadwriteArrayDyn<'_, T>, name: &str) -> PyResult<&'a mut [T]> {
    a.as_slice_mut()
        .map_err(|_| PyValueError::new_err(format!("pinsel_core: {name} must be a C-contiguous array")))
}

/// (H, W) of a 2-D or 3-D array.
pub fn hw<T: numpy::Element>(a: &PyReadonlyArrayDyn<'_, T>, name: &str) -> PyResult<(usize, usize)> {
    let s = a.shape();
    if s.len() < 2 {
        return Err(PyValueError::new_err(format!("pinsel_core: {name} must be at least 2-D")));
    }
    Ok((s[0], s[1]))
}

/// (H, W) of a mutable 2-D or 3-D array.
pub fn hw_mut<T: numpy::Element>(a: &PyReadwriteArrayDyn<'_, T>, name: &str) -> PyResult<(usize, usize)> {
    let s = a.shape();
    if s.len() < 2 {
        return Err(PyValueError::new_err(format!("pinsel_core: {name} must be at least 2-D")));
    }
    Ok((s[0], s[1]))
}

/// Read-only data of a numpy array that may be a non-contiguous view (e.g. a
/// region slice of the canvas): borrowed when contiguous, copied otherwise.
pub fn ro_cow<'a, T: numpy::Element + Copy>(a: &'a PyReadonlyArrayDyn<'_, T>) -> std::borrow::Cow<'a, [T]> {
    match a.as_slice() {
        Ok(s) => std::borrow::Cow::Borrowed(s),
        Err(_) => std::borrow::Cow::Owned(a.as_array().iter().copied().collect()),
    }
}

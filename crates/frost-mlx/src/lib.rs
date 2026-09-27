//! Safe RAII wrapper over the mlx-c API: the ops the encoder and the generator need.
//!
//! Arrays are reference-counted handles; `Arr` frees its handle on drop. Ops borrow
//! their inputs, so passing `a.0` never transfers ownership. MLX errors are routed
//! through an installed handler into a thread-local so they surface as Rust panics
//! (with the MLX message) instead of mlx-c's default `exit(-1)`; callers that must
//! not unwind across an FFI boundary wrap work in `catch_unwind`.
#![allow(non_camel_case_types, clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::sync::Once;

#[repr(C)] #[derive(Clone, Copy)] pub struct mlx_array { ctx: *mut c_void }
#[repr(C)] #[derive(Clone, Copy)] pub struct mlx_stream { ctx: *mut c_void }
#[repr(C)] #[derive(Clone, Copy)] pub struct mlx_vector_array { ctx: *mut c_void }
#[repr(C)] #[derive(Clone, Copy)] pub struct mlx_optional_int { pub value: i32, pub has_value: bool }
#[repr(C)] #[derive(Clone, Copy)] pub struct mlx_optional_float { pub value: f32, pub has_value: bool }
#[repr(C)] #[derive(Clone, Copy)] pub struct mlx_optional_dtype { pub value: u32, pub has_value: bool }

/// mlx_dtype enum values (verified against the installed header with clang).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype { Bool = 0, U8 = 1, U16 = 2, U32 = 3, U64 = 4, I8 = 5, I16 = 6, I32 = 7, I64 = 8, F16 = 9, F32 = 10, F64 = 11, BF16 = 12 }

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
            Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        }
    }
    fn from_raw(v: u32) -> Dtype {
        match v { 0 => Dtype::Bool, 1 => Dtype::U8, 2 => Dtype::U16, 3 => Dtype::U32, 4 => Dtype::U64, 5 => Dtype::I8,
            6 => Dtype::I16, 7 => Dtype::I32, 8 => Dtype::I64, 9 => Dtype::F16, 10 => Dtype::F32, 11 => Dtype::F64, _ => Dtype::BF16 }
    }
}

extern "C" {
    fn mlx_set_error_handler(h: extern "C" fn(*const c_char, *mut c_void), data: *mut c_void, dtor: Option<extern "C" fn(*mut c_void)>);

    fn mlx_array_new() -> mlx_array;
    fn mlx_array_new_data(data: *const c_void, shape: *const i32, dim: i32, dtype: u32) -> mlx_array;
    fn mlx_array_new_float32(v: f32) -> mlx_array;
    fn mlx_array_new_int(v: i32) -> mlx_array;
    fn mlx_array_free(a: mlx_array) -> i32;
    fn mlx_array_eval(a: mlx_array) -> i32;
    fn mlx_array_data_float32(a: mlx_array) -> *const f32;
    fn mlx_array_data_uint32(a: mlx_array) -> *const u32;
    fn mlx_array_data_int32(a: mlx_array) -> *const i32;
    fn mlx_array_size(a: mlx_array) -> usize;
    fn mlx_array_nbytes(a: mlx_array) -> usize;
    fn mlx_array_ndim(a: mlx_array) -> usize;
    fn mlx_array_shape(a: mlx_array) -> *const i32;
    fn mlx_array_dtype(a: mlx_array) -> u32;

    fn mlx_vector_array_new_data(data: *const mlx_array, size: usize) -> mlx_vector_array;
    fn mlx_vector_array_free(v: mlx_vector_array) -> i32;
    fn mlx_eval(outputs: mlx_vector_array) -> i32;
    fn mlx_async_eval(outputs: mlx_vector_array) -> i32;

    fn mlx_default_gpu_stream_new() -> mlx_stream;
    fn mlx_default_cpu_stream_new() -> mlx_stream;
    fn mlx_stream_free(s: mlx_stream) -> i32;
    fn mlx_synchronize(s: mlx_stream) -> i32;
    fn mlx_metal_is_available(res: *mut bool) -> i32;

    fn mlx_get_active_memory(res: *mut usize) -> i32;
    fn mlx_get_peak_memory(res: *mut usize) -> i32;
    fn mlx_get_cache_memory(res: *mut usize) -> i32;
    fn mlx_reset_peak_memory() -> i32;
    fn mlx_clear_cache() -> i32;
    fn mlx_set_cache_limit(res: *mut usize, limit: usize) -> i32;
    fn mlx_set_memory_limit(res: *mut usize, limit: usize) -> i32;
    fn mlx_set_wired_limit(res: *mut usize, limit: usize) -> i32;

    fn mlx_matmul(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_add(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_subtract(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_multiply(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_divide(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_sigmoid(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_exp(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_softmax_axis(res: *mut mlx_array, a: mlx_array, axis: i32, precise: bool, s: mlx_stream) -> i32;
    fn mlx_reshape(res: *mut mlx_array, a: mlx_array, shape: *const i32, n: usize, s: mlx_stream) -> i32;
    fn mlx_transpose_axes(res: *mut mlx_array, a: mlx_array, axes: *const i32, n: usize, s: mlx_stream) -> i32;
    fn mlx_mean_axis(res: *mut mlx_array, a: mlx_array, axis: i32, keepdims: bool, s: mlx_stream) -> i32;
    fn mlx_argmax_axis(res: *mut mlx_array, a: mlx_array, axis: i32, keepdims: bool, s: mlx_stream) -> i32;
    fn mlx_astype(res: *mut mlx_array, a: mlx_array, dtype: u32, s: mlx_stream) -> i32;
    fn mlx_take_axis(res: *mut mlx_array, a: mlx_array, idx: mlx_array, axis: i32, s: mlx_stream) -> i32;
    fn mlx_slice(res: *mut mlx_array, a: mlx_array, start: *const i32, ns: usize, stop: *const i32, ne: usize, strides: *const i32, nst: usize, s: mlx_stream) -> i32;
    fn mlx_slice_update(res: *mut mlx_array, src: mlx_array, upd: mlx_array, start: *const i32, ns: usize, stop: *const i32, ne: usize, strides: *const i32, nst: usize, s: mlx_stream) -> i32;
    fn mlx_concatenate_axis(res: *mut mlx_array, arrays: mlx_vector_array, axis: i32, s: mlx_stream) -> i32;
    fn mlx_zeros(res: *mut mlx_array, shape: *const i32, n: usize, dtype: u32, s: mlx_stream) -> i32;
    fn mlx_quantized_matmul(res: *mut mlx_array, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, transpose: bool,
                            group_size: mlx_optional_int, bits: mlx_optional_int, mode: *const c_char, s: mlx_stream) -> i32;
    fn mlx_dequantize(res: *mut mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int,
                      mode: *const c_char, global_scale: mlx_array, dtype: mlx_optional_dtype, s: mlx_stream) -> i32;
    fn mlx_fast_layer_norm(res: *mut mlx_array, x: mlx_array, w: mlx_array, b: mlx_array, eps: f32, s: mlx_stream) -> i32;
    fn mlx_fast_rms_norm(res: *mut mlx_array, x: mlx_array, w: mlx_array, eps: f32, s: mlx_stream) -> i32;
    fn mlx_fast_rope(res: *mut mlx_array, x: mlx_array, dims: i32, traditional: bool, base: mlx_optional_float, scale: f32, offset: i32, freqs: mlx_array, s: mlx_stream) -> i32;
    fn mlx_fast_scaled_dot_product_attention(res: *mut mlx_array, q: mlx_array, k: mlx_array, v: mlx_array, scale: f32,
                                             mask_mode: *const c_char, mask: mlx_array, sinks: mlx_array, s: mlx_stream) -> i32;
}

thread_local! { static LAST_ERR: RefCell<Option<String>> = const { RefCell::new(None) }; }

extern "C" fn error_handler(msg: *const c_char, _data: *mut c_void) {
    let m = if msg.is_null() { "unknown".to_string() } else { unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned() };
    LAST_ERR.with(|e| *e.borrow_mut() = Some(m));
}

/// Install the error handler once. Called by every constructor; cheap after the first call.
pub fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe { mlx_set_error_handler(error_handler, std::ptr::null_mut(), None) });
}

/// The message of the most recent MLX error on this thread, if any.
pub fn take_last_error() -> Option<String> { LAST_ERR.with(|e| e.borrow_mut().take()) }

#[inline]
fn ck(status: i32, what: &str) {
    if status != 0 {
        let msg = take_last_error().unwrap_or_else(|| "no message".into());
        panic!("mlx {what} failed: {msg}");
    }
}

const NONE_INT: mlx_optional_int = mlx_optional_int { value: 0, has_value: false };
fn some_int(v: i32) -> mlx_optional_int { mlx_optional_int { value: v, has_value: true } }
fn null_arr() -> mlx_array { mlx_array { ctx: std::ptr::null_mut() } }

/// Owned MLX array handle.
pub struct Arr(pub mlx_array);
impl Drop for Arr {
    fn drop(&mut self) { if !self.0.ctx.is_null() { unsafe { mlx_array_free(self.0); } } }
}
// SAFETY: mlx arrays are shared_ptr-backed; moving a handle between threads is sound
// as long as it is used from one thread at a time (enforced by Rust ownership).
unsafe impl Send for Arr {}

/// A compute stream. One per engine; the generator uses the GPU stream.
pub struct Stream(pub mlx_stream);
unsafe impl Send for Stream {}
impl Stream {
    pub fn gpu() -> Self { init(); Stream(unsafe { mlx_default_gpu_stream_new() }) }
    pub fn cpu() -> Self { init(); Stream(unsafe { mlx_default_cpu_stream_new() }) }
    pub fn synchronize(&self) { ck(unsafe { mlx_synchronize(self.0) }, "synchronize"); }
}
impl Drop for Stream {
    fn drop(&mut self) { unsafe { mlx_stream_free(self.0); } }
}

pub fn metal_available() -> bool { init(); let mut b = false; unsafe { mlx_metal_is_available(&mut b); } b }

/// MLX allocator statistics (bytes). These cover MLX-owned buffers only, not the
/// process RSS, the Zig index mmap or the UI.
#[derive(Debug, Clone, Copy, Default)]
pub struct Memory { pub active: usize, pub peak: usize, pub cache: usize }
pub fn memory() -> Memory {
    init();
    let mut m = Memory::default();
    unsafe { mlx_get_active_memory(&mut m.active); mlx_get_peak_memory(&mut m.peak); mlx_get_cache_memory(&mut m.cache); }
    m
}
pub fn reset_peak_memory() { unsafe { mlx_reset_peak_memory(); } }
pub fn clear_cache() { unsafe { mlx_clear_cache(); } }
/// Returns the previous limit.
pub fn set_cache_limit(bytes: usize) -> usize { init(); let mut p = 0; unsafe { mlx_set_cache_limit(&mut p, bytes); } p }
pub fn set_memory_limit(bytes: usize) -> usize { init(); let mut p = 0; unsafe { mlx_set_memory_limit(&mut p, bytes); } p }
pub fn set_wired_limit(bytes: usize) -> usize { init(); let mut p = 0; unsafe { mlx_set_wired_limit(&mut p, bytes); } p }

/// Evaluate several arrays in one graph pass.
pub fn eval_all(arrs: &[&Arr]) {
    let raw: Vec<mlx_array> = arrs.iter().map(|a| a.0).collect();
    let v = unsafe { mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
    let st = unsafe { mlx_eval(v) };
    unsafe { mlx_vector_array_free(v); }
    ck(st, "eval");
}
/// Schedule evaluation without blocking (the next `eval` or data read waits for it).
pub fn async_eval_all(arrs: &[&Arr]) {
    let raw: Vec<mlx_array> = arrs.iter().map(|a| a.0).collect();
    let v = unsafe { mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
    let st = unsafe { mlx_async_eval(v) };
    unsafe { mlx_vector_array_free(v); }
    ck(st, "async_eval");
}

fn numel(shape: &[i32]) -> usize { shape.iter().map(|&d| d.max(0) as usize).product() }

impl Arr {
    fn empty() -> mlx_array { unsafe { mlx_array_new() } }
    fn op(what: &str, f: impl FnOnce(*mut mlx_array) -> i32) -> Arr {
        let mut r = Self::empty();
        ck(f(&mut r), what);
        Arr(r)
    }

    /// Copy raw little-endian bytes into a new array of `dtype` with `shape`.
    pub fn from_bytes(bytes: &[u8], shape: &[i32], dtype: Dtype) -> Arr {
        init();
        assert_eq!(bytes.len(), numel(shape) * dtype.size(), "byte length vs shape {shape:?} {dtype:?}");
        // mlx copies the buffer; alignment of `bytes` is irrelevant (memcpy semantics).
        Arr(unsafe { mlx_array_new_data(bytes.as_ptr() as *const c_void, shape.as_ptr(), shape.len() as i32, dtype as u32) })
    }
    pub fn from_f32(data: &[f32], shape: &[i32]) -> Arr {
        init();
        assert_eq!(numel(shape), data.len(), "shape/data mismatch {shape:?} vs {}", data.len());
        Arr(unsafe { mlx_array_new_data(data.as_ptr() as *const c_void, shape.as_ptr(), shape.len() as i32, Dtype::F32 as u32) })
    }
    pub fn from_u32(data: &[u32], shape: &[i32]) -> Arr {
        init();
        assert_eq!(numel(shape), data.len(), "shape/data mismatch");
        Arr(unsafe { mlx_array_new_data(data.as_ptr() as *const c_void, shape.as_ptr(), shape.len() as i32, Dtype::U32 as u32) })
    }
    pub fn from_i32(data: &[i32], shape: &[i32]) -> Arr {
        init();
        assert_eq!(numel(shape), data.len(), "shape/data mismatch");
        Arr(unsafe { mlx_array_new_data(data.as_ptr() as *const c_void, shape.as_ptr(), shape.len() as i32, Dtype::I32 as u32) })
    }
    pub fn scalar(v: f32) -> Arr { init(); Arr(unsafe { mlx_array_new_float32(v) }) }
    pub fn scalar_i32(v: i32) -> Arr { init(); Arr(unsafe { mlx_array_new_int(v) }) }
    pub fn zeros(shape: &[i32], dtype: Dtype, s: &Stream) -> Arr {
        Self::op("zeros", |r| unsafe { mlx_zeros(r, shape.as_ptr(), shape.len(), dtype as u32, s.0) })
    }

    pub fn eval(&self) { ck(unsafe { mlx_array_eval(self.0) }, "eval"); }
    pub fn dtype(&self) -> Dtype { Dtype::from_raw(unsafe { mlx_array_dtype(self.0) }) }
    pub fn size(&self) -> usize { unsafe { mlx_array_size(self.0) } }
    pub fn nbytes(&self) -> usize { unsafe { mlx_array_nbytes(self.0) } }
    pub fn shape(&self) -> Vec<i32> {
        let nd = unsafe { mlx_array_ndim(self.0) };
        let p = unsafe { mlx_array_shape(self.0) };
        (0..nd).map(|i| unsafe { *p.add(i) }).collect()
    }

    /// Copy the (evaluated) contents out as f32, converting other dtypes first.
    pub fn to_vec(&self) -> Vec<f32> {
        if self.dtype() != Dtype::F32 {
            let s = Stream::gpu();
            return self.astype(Dtype::F32, &s).to_vec();
        }
        self.eval();
        let n = self.size();
        let p = unsafe { mlx_array_data_float32(self.0) };
        assert!(!p.is_null(), "null f32 data pointer");
        unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
    }
    pub fn to_vec_u32(&self) -> Vec<u32> {
        assert_eq!(self.dtype(), Dtype::U32);
        self.eval();
        let p = unsafe { mlx_array_data_uint32(self.0) };
        assert!(!p.is_null());
        unsafe { std::slice::from_raw_parts(p, self.size()) }.to_vec()
    }
    pub fn to_vec_i32(&self) -> Vec<i32> {
        assert_eq!(self.dtype(), Dtype::I32);
        self.eval();
        let p = unsafe { mlx_array_data_int32(self.0) };
        assert!(!p.is_null());
        unsafe { std::slice::from_raw_parts(p, self.size()) }.to_vec()
    }

    pub fn matmul(&self, b: &Arr, s: &Stream) -> Arr { Self::op("matmul", |r| unsafe { mlx_matmul(r, self.0, b.0, s.0) }) }
    pub fn add(&self, b: &Arr, s: &Stream) -> Arr { Self::op("add", |r| unsafe { mlx_add(r, self.0, b.0, s.0) }) }
    pub fn sub(&self, b: &Arr, s: &Stream) -> Arr { Self::op("subtract", |r| unsafe { mlx_subtract(r, self.0, b.0, s.0) }) }
    pub fn mul(&self, b: &Arr, s: &Stream) -> Arr { Self::op("multiply", |r| unsafe { mlx_multiply(r, self.0, b.0, s.0) }) }
    pub fn div(&self, b: &Arr, s: &Stream) -> Arr { Self::op("divide", |r| unsafe { mlx_divide(r, self.0, b.0, s.0) }) }
    pub fn sigmoid(&self, s: &Stream) -> Arr { Self::op("sigmoid", |r| unsafe { mlx_sigmoid(r, self.0, s.0) }) }
    pub fn exp(&self, s: &Stream) -> Arr { Self::op("exp", |r| unsafe { mlx_exp(r, self.0, s.0) }) }
    /// SiLU / swish: x * sigmoid(x).
    pub fn silu(&self, s: &Stream) -> Arr { self.mul(&self.sigmoid(s), s) }
    pub fn softmax_last(&self, s: &Stream) -> Arr { Self::op("softmax", |r| unsafe { mlx_softmax_axis(r, self.0, -1, true, s.0) }) }
    pub fn reshape(&self, shape: &[i32], s: &Stream) -> Arr { Self::op("reshape", |r| unsafe { mlx_reshape(r, self.0, shape.as_ptr(), shape.len(), s.0) }) }
    pub fn transpose(&self, axes: &[i32], s: &Stream) -> Arr { Self::op("transpose", |r| unsafe { mlx_transpose_axes(r, self.0, axes.as_ptr(), axes.len(), s.0) }) }
    pub fn mean_axis(&self, axis: i32, keepdims: bool, s: &Stream) -> Arr { Self::op("mean", |r| unsafe { mlx_mean_axis(r, self.0, axis, keepdims, s.0) }) }
    pub fn argmax_axis(&self, axis: i32, keepdims: bool, s: &Stream) -> Arr { Self::op("argmax", |r| unsafe { mlx_argmax_axis(r, self.0, axis, keepdims, s.0) }) }
    pub fn astype(&self, dtype: Dtype, s: &Stream) -> Arr { Self::op("astype", |r| unsafe { mlx_astype(r, self.0, dtype as u32, s.0) }) }
    /// Gather along `axis` with integer indices (e.g. embedding rows).
    pub fn take_axis(&self, idx: &Arr, axis: i32, s: &Stream) -> Arr { Self::op("take", |r| unsafe { mlx_take_axis(r, self.0, idx.0, axis, s.0) }) }
    /// `a[start:stop]` along every axis (stride 1).
    pub fn slice(&self, start: &[i32], stop: &[i32], s: &Stream) -> Arr {
        assert_eq!(start.len(), stop.len());
        let strides = vec![1i32; start.len()];
        Self::op("slice", |r| unsafe { mlx_slice(r, self.0, start.as_ptr(), start.len(), stop.as_ptr(), stop.len(), strides.as_ptr(), strides.len(), s.0) })
    }
    /// Functional `a[start:stop] = update` (returns the new array).
    pub fn slice_update(&self, update: &Arr, start: &[i32], stop: &[i32], s: &Stream) -> Arr {
        assert_eq!(start.len(), stop.len());
        let strides = vec![1i32; start.len()];
        Self::op("slice_update", |r| unsafe { mlx_slice_update(r, self.0, update.0, start.as_ptr(), start.len(), stop.as_ptr(), stop.len(), strides.as_ptr(), strides.len(), s.0) })
    }
    pub fn concat(arrs: &[&Arr], axis: i32, s: &Stream) -> Arr {
        let raw: Vec<mlx_array> = arrs.iter().map(|a| a.0).collect();
        let v = unsafe { mlx_vector_array_new_data(raw.as_ptr(), raw.len()) };
        let out = Self::op("concatenate", |r| unsafe { mlx_concatenate_axis(r, v, axis, s.0) });
        unsafe { mlx_vector_array_free(v); }
        out
    }
    pub fn layer_norm(&self, w: &Arr, b: &Arr, eps: f32, s: &Stream) -> Arr {
        Self::op("layer_norm", |r| unsafe { mlx_fast_layer_norm(r, self.0, w.0, b.0, eps, s.0) })
    }
    pub fn rms_norm(&self, w: &Arr, eps: f32, s: &Stream) -> Arr {
        Self::op("rms_norm", |r| unsafe { mlx_fast_rms_norm(r, self.0, w.0, eps, s.0) })
    }
    /// Rotary embedding with a fixed base (non-interleaved when traditional=false).
    pub fn rope(&self, dims: i32, traditional: bool, base: f32, s: &Stream) -> Arr {
        let b = mlx_optional_float { value: base, has_value: true };
        Self::op("rope", |r| unsafe { mlx_fast_rope(r, self.0, dims, traditional, b, 1.0, 0, null_arr(), s.0) })
    }
    /// Rotary embedding with explicit per-pair frequencies (YaRN/llama3 scaling) at `offset`.
    pub fn rope_freqs(&self, dims: i32, offset: i32, freqs: &Arr, s: &Stream) -> Arr {
        let b = mlx_optional_float { value: 0.0, has_value: false };
        Self::op("rope(freqs)", |r| unsafe { mlx_fast_rope(r, self.0, dims, false, b, 1.0, offset, freqs.0, s.0) })
    }
    /// Fused scaled-dot-product attention over [B, heads, L, D]; GQA handled by MLX when
    /// the kv head count divides the query head count.
    pub fn sdpa(q: &Arr, k: &Arr, v: &Arr, scale: f32, causal: bool, s: &Stream) -> Arr {
        let mode = CString::new(if causal { "causal" } else { "" }).expect("static");
        Self::op("sdpa", |r| unsafe {
            mlx_fast_scaled_dot_product_attention(r, q.0, k.0, v.0, scale, mode.as_ptr(), null_arr(), null_arr(), s.0)
        })
    }
    /// `x @ dequant(w)^T` for MLX affine-quantized weights (`w` U32 packed, scales/biases per group).
    pub fn quantized_matmul(&self, w: &Arr, scales: &Arr, biases: &Arr, group_size: i32, bits: i32, s: &Stream) -> Arr {
        let mode = CString::new("affine").expect("static");
        Self::op("quantized_matmul", |r| unsafe {
            mlx_quantized_matmul(r, self.0, w.0, scales.0, biases.0, true, some_int(group_size), some_int(bits), mode.as_ptr(), s.0)
        })
    }
    /// Dequantize affine-quantized weights to `dtype`.
    pub fn dequantize(w: &Arr, scales: &Arr, biases: &Arr, group_size: i32, bits: i32, dtype: Dtype, s: &Stream) -> Arr {
        let mode = CString::new("affine").expect("static");
        let dt = mlx_optional_dtype { value: dtype as u32, has_value: true };
        Self::op("dequantize", |r| unsafe {
            mlx_dequantize(r, w.0, scales.0, biases.0, some_int(group_size), some_int(bits), mode.as_ptr(), null_arr(), dt, s.0)
        })
    }
}

#[allow(dead_code)]
const _KEEP: mlx_optional_int = NONE_INT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_to_mlx_gpu_matmul() {
        let s = Stream::gpu();
        let a = Arr::from_f32(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = Arr::from_f32(&[1., 0., 0., 1., 1., 1.], &[3, 2]);
        assert_eq!(a.matmul(&b, &s).to_vec(), vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn layer_norm_matches_manual() {
        let s = Stream::gpu();
        let x = Arr::from_f32(&[1., 2., 3., 4.], &[1, 4]);
        let w = Arr::from_f32(&[1., 1., 1., 1.], &[4]);
        let b = Arr::from_f32(&[0., 0., 0., 0.], &[4]);
        let y = x.layer_norm(&w, &b, 1e-5, &s).to_vec();
        let inv = 1.0 / (1.25f32 + 1e-5).sqrt();
        for (i, xi) in [1., 2., 3., 4.].iter().enumerate() {
            assert!((y[i] - (xi - 2.5) * inv).abs() < 1e-3);
        }
    }

    #[test]
    fn rms_norm_matches_manual() {
        let s = Stream::gpu();
        let x = Arr::from_f32(&[1., 2., 3., 4.], &[1, 4]);
        let w = Arr::from_f32(&[1., 2., 1., 2.], &[4]);
        let y = x.rms_norm(&w, 1e-5, &s).to_vec();
        let rms = ((1. + 4. + 9. + 16.) / 4.0f32 + 1e-5).sqrt();
        let exp = [1. / rms, 2. * 2. / rms, 3. / rms, 4. * 2. / rms];
        for i in 0..4 { assert!((y[i] - exp[i]).abs() < 1e-3, "rms[{i}] {} vs {}", y[i], exp[i]); }
    }

    #[test]
    fn silu_correct() {
        let s = Stream::gpu();
        let y = Arr::from_f32(&[0.0, 1.0, -1.0], &[3]).silu(&s).to_vec();
        assert!(y[0].abs() < 1e-4 && (y[1] - 0.7310586).abs() < 1e-4 && (y[2] + 0.2689414).abs() < 1e-4);
    }

    #[test]
    fn quantized_matmul_matches_dequantized_matmul() {
        // 4-bit affine, group 64: build a [4 rows, 64 cols] weight from nibbles, then compare
        // x @ W^T via the fused kernel against x @ dequantize(W)^T.
        let s = Stream::gpu();
        let rows = 4; let cols = 64; let packed = cols / 8;
        let mut w = Vec::with_capacity(rows * packed);
        let mut q = 0u32;
        for _ in 0..rows * packed { let mut word = 0u32; for j in 0..8 { word |= ((q + j) % 16) << (4 * j); } q = (q + 3) % 16; w.push(word); }
        let scales: Vec<u16> = (0..rows).map(|r| ((0.5f32 + r as f32 * 0.25).to_bits() >> 16) as u16).collect(); // 1 group per row
        let biases: Vec<u16> = (0..rows).map(|r| ((-1.0f32 + r as f32 * 0.5).to_bits() >> 16) as u16).collect();
        let wa = Arr::from_u32(&w, &[rows as i32, packed as i32]);
        let sb = |v: &[u16]| Arr::from_bytes(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(), &[rows as i32, 1], Dtype::BF16);
        let sa = sb(&scales); let ba = sb(&biases);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) - 0.3).collect();
        let xa = Arr::from_f32(&x, &[1, cols as i32]).astype(Dtype::BF16, &s);
        let fused = xa.quantized_matmul(&wa, &sa, &ba, 64, 4, &s).to_vec();
        let deq = Arr::dequantize(&wa, &sa, &ba, 64, 4, Dtype::F32, &s); // [rows, cols]
        let refv = xa.astype(Dtype::F32, &s).matmul(&deq.transpose(&[1, 0], &s), &s).to_vec();
        // scalar reference for dequantize: low nibble first, value = scale*q + bias
        let deqv = deq.to_vec();
        for r in 0..rows { for c in 0..cols {
            let word = w[r * packed + c / 8]; let nib = ((word >> (4 * (c % 8))) & 0xF) as f32;
            let sc = f32::from_bits((scales[r] as u32) << 16); let bi = f32::from_bits((biases[r] as u32) << 16);
            assert!((deqv[r * cols + c] - (sc * nib + bi)).abs() < 1e-5, "dequant[{r},{c}]");
        }}
        for r in 0..rows { assert!((fused[r] - refv[r]).abs() < 2e-2 * (1.0 + refv[r].abs()), "row {r}: fused {} vs ref {}", fused[r], refv[r]); }
    }

    #[test]
    fn sdpa_causal_matches_manual_softmax() {
        let s = Stream::gpu();
        // B=1, H=1, L=3, D=2 ; causal attention where each row attends to <= its index
        let q = Arr::from_f32(&[1., 0., 0., 1., 1., 1.], &[1, 1, 3, 2]);
        let k = Arr::from_f32(&[1., 0., 0., 1., 1., 1.], &[1, 1, 3, 2]);
        let v = Arr::from_f32(&[1., 2., 3., 4., 5., 6.], &[1, 1, 3, 2]);
        let out = Arr::sdpa(&q, &k, &v, 1.0, true, &s).to_vec();
        // manual
        let qv = [[1., 0.], [0., 1.], [1., 1.]]; let kv = qv; let vv = [[1., 2.], [3., 4.], [5., 6.]];
        for i in 0..3 {
            let mut sc: Vec<f32> = (0..=i).map(|j| qv[i][0] * kv[j][0] + qv[i][1] * kv[j][1]).collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let z: f32 = sc.iter().map(|x| (x - m).exp()).sum();
            for x in &mut sc { *x = (*x - m).exp() / z; }
            for d in 0..2 { let e: f32 = (0..=i).map(|j| sc[j] * vv[j][d]).sum(); assert!((out[i * 2 + d] - e).abs() < 1e-4, "sdpa[{i},{d}] {} vs {e}", out[i * 2 + d]); }
        }
    }

    #[test]
    fn rope_with_explicit_freqs_matches_base_rope() {
        let s = Stream::gpu();
        let dims = 8;
        let x = Arr::from_f32(&(0..2 * dims).map(|i| i as f32 * 0.1).collect::<Vec<_>>(), &[1, 1, 2, dims as i32]);
        let base = 1000.0f32;
        let freqs: Vec<f32> = (0..dims / 2).map(|i| base.powf(2.0 * i as f32 / dims as f32)).collect();
        let fa = Arr::from_f32(&freqs, &[dims as i32 / 2]);
        let a = x.rope(dims as i32, false, base, &s).to_vec();
        let b = x.rope_freqs(dims as i32, 0, &fa, &s).to_vec();
        for i in 0..a.len() { assert!((a[i] - b[i]).abs() < 1e-4, "rope[{i}] {} vs {}", a[i], b[i]); }
    }

    #[test]
    fn slice_update_and_take_and_concat() {
        let s = Stream::gpu();
        let z = Arr::zeros(&[1, 4], Dtype::F32, &s);
        let u = Arr::from_f32(&[7., 8.], &[1, 2]);
        let y = z.slice_update(&u, &[0, 1], &[1, 3], &s);
        assert_eq!(y.to_vec(), vec![0., 7., 8., 0.]);
        assert_eq!(y.slice(&[0, 1], &[1, 3], &s).to_vec(), vec![7., 8.]);
        let table = Arr::from_f32(&[10., 11., 20., 21., 30., 31.], &[3, 2]);
        let idx = Arr::from_i32(&[2, 0], &[2]);
        assert_eq!(table.take_axis(&idx, 0, &s).to_vec(), vec![30., 31., 10., 11.]);
        let c = Arr::concat(&[&u, &u], 1, &s);
        assert_eq!(c.shape(), vec![1, 4]);
        assert_eq!(Arr::from_f32(&[0.1, 5.0, 2.0], &[1, 3]).argmax_axis(-1, false, &s).to_vec_u32(), vec![1]);
    }

    #[test]
    fn mlx_error_becomes_rust_panic_not_exit() {
        let s = Stream::gpu();
        let a = Arr::from_f32(&[1., 2.], &[2]);
        let b = Arr::from_f32(&[1., 2., 3.], &[3]);
        let r = std::panic::catch_unwind(|| a.matmul(&b, &s));
        assert!(r.is_err(), "shape mismatch must surface as a Rust panic, not a process exit");
        assert!(memory().active < (1 << 30));
    }
}

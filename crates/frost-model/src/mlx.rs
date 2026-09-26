//! Minimal safe wrapper over the mlx-c API — only the ops the encoder needs.
//! Arrays are reference-counted handles; `Arr` frees its handle on drop. Inputs
//! are borrowed (const) by ops, so passing `a.0` never transfers ownership.
#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::os::raw::c_char;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct mlx_array { ctx: *mut c_void }
#[repr(C)]
#[derive(Clone, Copy)]
pub struct mlx_stream { ctx: *mut c_void }
#[repr(C)]
#[derive(Clone, Copy)]
pub struct mlx_vector_array { ctx: *mut c_void }
#[repr(C)]
#[derive(Clone, Copy)]
pub struct mlx_optional_float { pub value: f32, pub has_value: bool }

const MLX_FLOAT32: u32 = 10; // verified via clang: enum index of FLOAT32

extern "C" {
    fn mlx_array_new() -> mlx_array;
    fn mlx_array_new_data(data: *const c_void, shape: *const i32, dim: i32, dtype: u32) -> mlx_array;
    fn mlx_array_new_float32(v: f32) -> mlx_array;
    fn mlx_array_free(a: mlx_array) -> i32;
    fn mlx_array_eval(a: mlx_array) -> i32;
    fn mlx_array_data_float32(a: mlx_array) -> *const f32;
    fn mlx_array_size(a: mlx_array) -> usize;
    fn mlx_array_ndim(a: mlx_array) -> usize;
    fn mlx_array_shape(a: mlx_array) -> *const i32;

    fn mlx_default_gpu_stream_new() -> mlx_stream;
    fn mlx_stream_free(s: mlx_stream) -> i32;

    fn mlx_matmul(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_add(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_multiply(res: *mut mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) -> i32;
    fn mlx_sigmoid(res: *mut mlx_array, a: mlx_array, s: mlx_stream) -> i32;
    fn mlx_softmax_axis(res: *mut mlx_array, a: mlx_array, axis: i32, precise: bool, s: mlx_stream) -> i32;
    fn mlx_reshape(res: *mut mlx_array, a: mlx_array, shape: *const i32, n: usize, s: mlx_stream) -> i32;
    fn mlx_transpose_axes(res: *mut mlx_array, a: mlx_array, axes: *const i32, n: usize, s: mlx_stream) -> i32;
    fn mlx_mean_axis(res: *mut mlx_array, a: mlx_array, axis: i32, keepdims: bool, s: mlx_stream) -> i32;
    fn mlx_fast_layer_norm(res: *mut mlx_array, x: mlx_array, w: mlx_array, b: mlx_array, eps: f32, s: mlx_stream) -> i32;
    fn mlx_fast_rope(res: *mut mlx_array, x: mlx_array, dims: i32, traditional: bool,
                     base: mlx_optional_float, scale: f32, offset: i32, freqs: mlx_array, s: mlx_stream) -> i32;
}

/// Owned MLX array handle.
pub struct Arr(pub mlx_array);
impl Drop for Arr {
    fn drop(&mut self) { unsafe { mlx_array_free(self.0); } }
}

/// The GPU stream (default device). One per engine.
pub struct Stream(pub mlx_stream);
impl Stream {
    pub fn gpu() -> Self { Stream(unsafe { mlx_default_gpu_stream_new() }) }
}
impl Drop for Stream {
    fn drop(&mut self) { unsafe { mlx_stream_free(self.0); } }
}

#[inline]
fn ck(status: i32, what: &str) {
    assert_eq!(status, 0, "mlx op failed: {what}");
}

impl Arr {
    /// New f32 array from a row-major slice with the given shape.
    pub fn from_f32(data: &[f32], shape: &[i32]) -> Arr {
        let n: i64 = shape.iter().map(|&d| d as i64).product();
        assert_eq!(n as usize, data.len(), "shape/data mismatch {shape:?} vs {}", data.len());
        let a = unsafe {
            mlx_array_new_data(data.as_ptr() as *const c_void, shape.as_ptr(), shape.len() as i32, MLX_FLOAT32)
        };
        Arr(a)
    }
    pub fn scalar(v: f32) -> Arr { Arr(unsafe { mlx_array_new_float32(v) }) }
    fn empty() -> mlx_array { unsafe { mlx_array_new() } }

    pub fn eval(&self) { ck(unsafe { mlx_array_eval(self.0) }, "eval"); }

    pub fn shape(&self) -> Vec<i32> {
        let nd = unsafe { mlx_array_ndim(self.0) };
        let p = unsafe { mlx_array_shape(self.0) };
        (0..nd).map(|i| unsafe { *p.add(i) }).collect()
    }

    /// Copy the (evaluated) contents out as a Vec<f32>.
    pub fn to_vec(&self) -> Vec<f32> {
        self.eval();
        let n = unsafe { mlx_array_size(self.0) };
        let p = unsafe { mlx_array_data_float32(self.0) };
        assert!(!p.is_null(), "null data ptr (dtype not f32?)");
        unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
    }

    pub fn matmul(&self, b: &Arr, s: &Stream) -> Arr {
        let mut r = Self::empty();
        ck(unsafe { mlx_matmul(&mut r, self.0, b.0, s.0) }, "matmul");
        Arr(r)
    }
    pub fn add(&self, b: &Arr, s: &Stream) -> Arr {
        let mut r = Self::empty();
        ck(unsafe { mlx_add(&mut r, self.0, b.0, s.0) }, "add");
        Arr(r)
    }
    pub fn mul(&self, b: &Arr, s: &Stream) -> Arr {
        let mut r = Self::empty();
        ck(unsafe { mlx_multiply(&mut r, self.0, b.0, s.0) }, "mul");
        Arr(r)
    }
    pub fn sigmoid(&self, s: &Stream) -> Arr {
        let mut r = Self::empty();
        ck(unsafe { mlx_sigmoid(&mut r, self.0, s.0) }, "sigmoid");
        Arr(r)
    }
    /// SiLU / swish: x * sigmoid(x).
    pub fn silu(&self, s: &Stream) -> Arr { self.mul(&self.sigmoid(s), s) }

    pub fn softmax_last(&self, s: &Stream) -> Arr {
        let mut r = Self::empty();
        ck(unsafe { mlx_softmax_axis(&mut r, self.0, -1, true, s.0) }, "softmax");
        Arr(r)
    }
    pub fn reshape(&self, shape: &[i32], s: &Stream) -> Arr {
        let mut r = Self::empty();
        ck(unsafe { mlx_reshape(&mut r, self.0, shape.as_ptr(), shape.len(), s.0) }, "reshape");
        Arr(r)
    }
    pub fn transpose(&self, axes: &[i32], s: &Stream) -> Arr {
        let mut r = Self::empty();
        ck(unsafe { mlx_transpose_axes(&mut r, self.0, axes.as_ptr(), axes.len(), s.0) }, "transpose");
        Arr(r)
    }
    pub fn mean_axis(&self, axis: i32, keepdims: bool, s: &Stream) -> Arr {
        let mut r = Self::empty();
        ck(unsafe { mlx_mean_axis(&mut r, self.0, axis, keepdims, s.0) }, "mean");
        Arr(r)
    }
    pub fn layer_norm(&self, w: &Arr, b: &Arr, eps: f32, s: &Stream) -> Arr {
        let mut r = Self::empty();
        ck(unsafe { mlx_fast_layer_norm(&mut r, self.0, w.0, b.0, eps, s.0) }, "layer_norm");
        Arr(r)
    }
    /// Rotary position embedding (NeoX/non-interleaved when traditional=false).
    pub fn rope(&self, dims: i32, traditional: bool, base: f32, s: &Stream) -> Arr {
        let mut r = Self::empty();
        let freqs = Arr(Self::empty());
        let b = mlx_optional_float { value: base, has_value: true };
        ck(unsafe { mlx_fast_rope(&mut r, self.0, dims, traditional, b, 1.0, 0, freqs.0, s.0) }, "rope");
        Arr(r)
    }
}

// silence unused import warning if c_char ends up unused in some builds
#[allow(dead_code)]
type _Keep = *const c_char;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_to_mlx_gpu_matmul() {
        let s = Stream::gpu();
        // A[2x3] @ B[3x2] = [[4,5],[10,11]]
        let a = Arr::from_f32(&[1.,2.,3., 4.,5.,6.], &[2, 3]);
        let b = Arr::from_f32(&[1.,0., 0.,1., 1.,1.], &[3, 2]);
        let c = a.matmul(&b, &s);
        let v = c.to_vec();
        assert_eq!(v, vec![4.0, 5.0, 10.0, 11.0], "Rust->MLX GPU matmul");
    }

    #[test]
    fn layer_norm_matches_manual() {
        let s = Stream::gpu();
        let x = Arr::from_f32(&[1., 2., 3., 4.], &[1, 4]);
        let w = Arr::from_f32(&[1., 1., 1., 1.], &[4]);
        let b = Arr::from_f32(&[0., 0., 0., 0.], &[4]);
        let y = x.layer_norm(&w, &b, 1e-5, &s).to_vec();
        // manual: mean=2.5, var=1.25, std=sqrt(1.25)=1.1180; normalized
        let mean = 2.5f32;
        let var = 1.25f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (i, xi) in [1., 2., 3., 4.].iter().enumerate() {
            let expect = (xi - mean) * inv;
            assert!((y[i] - expect).abs() < 1e-3, "ln[{i}]={} exp {}", y[i], expect);
        }
    }

    #[test]
    fn matmul_3d_2d() {
        let s = Stream::gpu();
        // A[1,2,3] @ B[3,2] should give [1,2,2] = per-row (A[0]) @ B
        let a = Arr::from_f32(&[1.,2.,3., 4.,5.,6.], &[1,2,3]);
        let b = Arr::from_f32(&[1.,0., 0.,1., 1.,1.], &[3,2]);
        let c = a.matmul(&b, &s);
        eprintln!("shape={:?} data={:?}", c.shape(), c.to_vec());
        // expect [[4,5],[10,11]] flattened
        assert_eq!(c.to_vec(), vec![4.,5.,10.,11.], "3D@2D broadcast matmul");
    }
    #[test]
    fn ln_axis_probe() {
        let s = Stream::gpu();
        // two DIFFERENT rows; per-row LN must normalize each row independently
        let x = Arr::from_f32(&[1.,2.,3.,  10.,20.,30.], &[2,3]);
        let w = Arr::from_f32(&[1.,1.,1.], &[3]);
        let b = Arr::from_f32(&[0.,0.,0.], &[3]);
        let y = x.layer_norm(&w,&b,1e-5,&s).to_vec();
        eprintln!("LN[2,3] rows -> {:?}", y);
        // per-row: both rows have same pattern (mean-centered, unit var): ~[-1.2247,0,1.2247]
        assert!((y[0]-y[3]).abs()<1e-3 && (y[1]-y[4]).abs()<1e-3,
            "rows must normalize identically if per-row; got {:?}", y);
    }
    #[test]
    fn silu_correct() {
        let s = Stream::gpu();
        let x = Arr::from_f32(&[0.0, 1.0, -1.0], &[3]);
        let y = x.silu(&s).to_vec();
        // silu(0)=0, silu(1)=1/(1+e^-1)=0.7311, silu(-1)=-0.2689
        assert!(y[0].abs() < 1e-4);
        assert!((y[1] - 0.7310586).abs() < 1e-4);
        assert!((y[2] + 0.2689414).abs() < 1e-4);
    }
}

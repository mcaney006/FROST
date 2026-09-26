# FROST Mojo numerical kernels, exported with a C ABI for use from Rust/C.
# Substantive kernel: fused L2-normalize of a float32 vector (in place).
# This runs on CPU in compiled Mojo; no Python interpreter at build or run time.
from memory import UnsafePointer
from math import sqrt

@export(ABI="C")
fn frost_l2_normalize(data: UnsafePointer[Float32], n: Int32):
    var ss: Float32 = 0.0
    var i: Int32 = 0
    while i < n:
        var v = data[Int(i)]
        ss += v * v
        i += 1
    if ss <= 0.0:
        return
    var inv = 1.0 / sqrt(ss)
    i = 0
    while i < n:
        data[Int(i)] = data[Int(i)] * inv
        i += 1

@export(ABI="C")
fn frost_dot(a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int32) -> Float32:
    var acc: Float32 = 0.0
    var i: Int32 = 0
    while i < n:
        acc += a[Int(i)] * b[Int(i)]
        i += 1
    return acc

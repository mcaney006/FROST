// FROST Zig action-index kernels, C ABI. Retrieval cascade primitives:
//  1) binary sketch coarse search (XOR + popcount Hamming distance)
//  2) INT8 shortlist scoring (integer dot with explicit caller-side scale)
//  3) FP32 exact rerank (dot product)
const std = @import("std");

export fn frost_hamming(a: [*]const u8, b: [*]const u8, nbytes: usize) u32 {
    var acc: u32 = 0;
    var i: usize = 0;
    while (i < nbytes) : (i += 1) acc += @popCount(a[i] ^ b[i]);
    return acc;
}

export fn frost_dot_i8(a: [*]const i8, b: [*]const i8, n: usize) i32 {
    var acc: i32 = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) acc += @as(i32, a[i]) * @as(i32, b[i]);
    return acc;
}

export fn frost_dot_f32(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    var acc: f32 = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) acc += a[i] * b[i];
    return acc;
}

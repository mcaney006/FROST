//! frost-index: safe Rust wrapper over the Zig retrieval-cascade kernels, with
//! a scalar Rust reference used for parity testing. The cascade is:
//!   1) binary sketch coarse search  (Hamming distance over packed bits)
//!   2) INT8 shortlist scoring        (integer dot, caller applies the scale)
//!   3) FP32 exact rerank             (dot product)

extern "C" {
    fn frost_hamming(a: *const u8, b: *const u8, nbytes: usize) -> u32;
    fn frost_dot_i8(a: *const i8, b: *const i8, n: usize) -> i32;
    fn frost_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;
}

/// Hamming distance between two equal-length packed-bit sketches.
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "sketch length mismatch");
    // SAFETY: pointers valid for `len` bytes; lengths checked equal.
    unsafe { frost_hamming(a.as_ptr(), b.as_ptr(), a.len()) }
}

/// Integer dot product of two INT8 vectors (shortlist scoring).
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    unsafe { frost_dot_i8(a.as_ptr(), b.as_ptr(), a.len()) }
}

/// FP32 dot product (exact rerank).
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    unsafe { frost_dot_f32(a.as_ptr(), b.as_ptr(), a.len()) }
}

/// Scalar Rust reference implementations for numeric/index parity tests.
pub mod reference {
    pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
        a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
    }
    pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
        a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
    }
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zig_matches_scalar_reference_random() {
        // deterministic pseudo-random vectors, parity Zig vs scalar reference
        let mut s: u64 = 0x1234_5678;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for n in [1usize, 3, 8, 64, 257] {
            let ba: Vec<u8> = (0..n).map(|_| (next() & 0xff) as u8).collect();
            let bb: Vec<u8> = (0..n).map(|_| (next() & 0xff) as u8).collect();
            assert_eq!(hamming(&ba, &bb), reference::hamming(&ba, &bb), "hamming n={n}");

            let ia: Vec<i8> = (0..n).map(|_| (next() as i8)).collect();
            let ib: Vec<i8> = (0..n).map(|_| (next() as i8)).collect();
            assert_eq!(dot_i8(&ia, &ib), reference::dot_i8(&ia, &ib), "dot_i8 n={n}");

            let fa: Vec<f32> = (0..n).map(|_| (next() % 1000) as f32 / 1000.0).collect();
            let fb: Vec<f32> = (0..n).map(|_| (next() % 1000) as f32 / 1000.0).collect();
            let d = dot_f32(&fa, &fb);
            let r = reference::dot_f32(&fa, &fb);
            assert!((d - r).abs() < 1e-3, "dot_f32 n={n}: {d} vs {r}");
        }
    }

    #[test]
    fn hamming_edge_cases() {
        assert_eq!(hamming(&[], &[]), 0);
        assert_eq!(hamming(&[0xFF], &[0x00]), 8);
        assert_eq!(hamming(&[0xAA, 0x55], &[0x55, 0xAA]), 16);
    }
}

// FROST Zig native data operations, C ABI.
//
// Retrieval cascade primitives:
//  1) binary sketch coarse search (XOR + popcount Hamming distance)
//  2) INT8 shortlist scoring (integer dot with explicit caller-side scale)
//  3) FP32 exact rerank (dot product)
// Checked data operations:
//  4) MLX 4-bit affine quantized-weight validation + row dequant
//  5) FROSTIDX1 memory-mapped exact vector index (write / open / search)
//
// Error codes (i32; 0 = OK). Mirrored by `IndexError` in crates/frost-index.
//   -1  E_UNSUPPORTED_BITS  quantization bits != 4
//   -2  E_DIM_MISMATCH      shape/dimension disagreement (q4 cols vs groups,
//                           query dim vs index dim)
//   -3  E_OVERFLOW          size arithmetic overflowed usize/u64
//   -4  E_NONFINITE         NaN/Inf in q4 scales/biases or in a search query
//   -5  E_INVALID_ARG       zero dim / zero group_size / bad argument
//   -6  E_IO                open/fstat/mmap/write/fsync/rename failed
//   -7  E_BAD_MAGIC         file does not start with "FROSTIDX"
//   -8  E_BAD_HEADER        version != 1, dim == 0, or nonzero pad
//   -9  E_BAD_HASH          header FNV-1a 64 does not match
//   -10 E_BAD_LENGTH        file length disagrees with header arithmetic
//                           (truncated, or trailing bytes)
//   -11 E_NOMEM             handle allocation failed
const std = @import("std");
const builtin = @import("builtin");
const c = std.c;

const OK: i32 = 0;
const E_UNSUPPORTED_BITS: i32 = -1;
const E_DIM_MISMATCH: i32 = -2;
const E_OVERFLOW: i32 = -3;
const E_NONFINITE: i32 = -4;
const E_INVALID_ARG: i32 = -5;
const E_IO: i32 = -6;
const E_BAD_MAGIC: i32 = -7;
const E_BAD_HEADER: i32 = -8;
const E_BAD_HASH: i32 = -9;
const E_BAD_LENGTH: i32 = -10;
const E_NOMEM: i32 = -11;

// ponytail: ids/vectors are written as raw host bytes; header fields are explicit LE.
// Big-endian hosts would need per-element byte swaps.
comptime {
    std.debug.assert(builtin.cpu.arch.endian() == .little);
}

// ---------------------------------------------------------------- cascade kernels

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
    return dot8(a, b, n);
}

const V8 = @Vector(8, f32);

/// 8-lane f32 dot product with scalar tail. Four independent accumulators
/// break the fadd latency chain (10k x 768 search: 510 -> 380 us, now near
/// single-core memory bandwidth).
fn dot8(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    var acc4 = [_]V8{@splat(0)} ** 4;
    var i: usize = 0;
    while (i + 32 <= n) : (i += 32) {
        inline for (0..4) |u| {
            const va: V8 = a[i + u * 8 ..][0..8].*;
            const vb: V8 = b[i + u * 8 ..][0..8].*;
            acc4[u] += va * vb;
        }
    }
    var acc = (acc4[0] + acc4[1]) + (acc4[2] + acc4[3]);
    while (i + 8 <= n) : (i += 8) {
        const va: V8 = a[i..][0..8].*;
        const vb: V8 = b[i..][0..8].*;
        acc += va * vb;
    }
    var s = @reduce(.Add, acc);
    while (i < n) : (i += 1) s += a[i] * b[i];
    return s;
}

// ---------------------------------------------------------------- MLX q4 affine

pub const FrostQ4Stats = extern struct {
    rows: u64,
    cols: u64,
    nonfinite: u64,
    zero_scales: u64,
    min_scale: f32,
    max_scale: f32,
};

inline fn bf16(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

inline fn bf16Finite(x: u16) bool {
    return (x & 0x7F80) != 0x7F80;
}

/// Validates an MLX affine quantized weight layout: w is rows x cols_packed u32,
/// scales/biases are rows x groups bf16 bit patterns. `w` itself is not read:
/// every nibble pattern is a valid 4-bit code.
export fn frost_q4_validate(
    w: [*]const u32,
    rows: usize,
    cols_packed: usize,
    scales: [*]const u16,
    biases: [*]const u16,
    groups: usize,
    group_size: u32,
    bits: u32,
    out: *FrostQ4Stats,
) i32 {
    _ = w;
    out.* = .{ .rows = rows, .cols = 0, .nonfinite = 0, .zero_scales = 0, .min_scale = 0, .max_scale = 0 };
    if (bits != 4) return E_UNSUPPORTED_BITS;
    if (group_size == 0) return E_INVALID_ARG;
    const cols = std.math.mul(usize, cols_packed, 32 / bits) catch return E_OVERFLOW;
    const covered = std.math.mul(usize, groups, group_size) catch return E_OVERFLOW;
    if (cols != covered) return E_DIM_MISMATCH;
    const total = std.math.mul(usize, rows, groups) catch return E_OVERFLOW;
    out.cols = cols;

    var nonfinite: u64 = 0;
    var zeros: u64 = 0;
    var lo = std.math.inf(f32);
    var hi = -std.math.inf(f32);
    for (0..total) |i| {
        const s = scales[i];
        if (!bf16Finite(biases[i])) nonfinite += 1;
        if (!bf16Finite(s)) {
            nonfinite += 1;
            continue;
        }
        if (s & 0x7FFF == 0) zeros += 1;
        const f = bf16(s);
        lo = @min(lo, f);
        hi = @max(hi, f);
    }
    out.nonfinite = nonfinite;
    out.zero_scales = zeros;
    if (lo <= hi) {
        out.min_scale = lo;
        out.max_scale = hi;
    }
    return if (nonfinite != 0) E_NONFINITE else OK;
}

/// Dequantizes one row: element j = nibble (j % 8) of word (j / 8), low nibble
/// first; value = scale[j / group_size] * q + bias[j / group_size].
/// Writes cols_packed * 8 floats. Caller guarantees scales/biases hold
/// cols / group_size entries.
export fn frost_q4_dequant_row(
    w: [*]const u32,
    scales: [*]const u16,
    biases: [*]const u16,
    cols_packed: usize,
    group_size: u32,
    out: [*]f32,
) i32 {
    if (group_size == 0) return E_INVALID_ARG;
    const cols = std.math.mul(usize, cols_packed, 8) catch return E_OVERFLOW;
    if (cols % group_size != 0) return E_DIM_MISMATCH;

    if (group_size % 8 == 0) {
        // Every word lies inside one group: unpack 8 nibbles as one vector.
        const shifts: @Vector(8, u5) = .{ 0, 4, 8, 12, 16, 20, 24, 28 };
        const mask: @Vector(8, u32) = @splat(0xF);
        const words_per_group = group_size / 8;
        for (0..cols / group_size) |g| {
            const s: V8 = @splat(bf16(scales[g]));
            const b: V8 = @splat(bf16(biases[g]));
            for (g * words_per_group..(g + 1) * words_per_group) |wi| {
                const q = (@as(@Vector(8, u32), @splat(w[wi])) >> shifts) & mask;
                const qf: V8 = @floatFromInt(q);
                out[wi * 8 ..][0..8].* = qf * s + b;
            }
        }
    } else {
        for (0..cols) |j| {
            const q: u32 = (w[j / 8] >> @intCast((j % 8) * 4)) & 0xF;
            const g = j / group_size;
            out[j] = bf16(scales[g]) * @as(f32, @floatFromInt(q)) + bf16(biases[g]);
        }
    }
    return OK;
}

// ---------------------------------------------------------------- FROSTIDX1 index
//
// Layout (little-endian):
//   [0..48)    header: magic[8]="FROSTIDX", version u32=1, dim u32, count u64,
//              generation u64, flags u32, pad u32, header_hash u64
//              (FNV-1a 64 over bytes [0..40))
//   [48..)     count u64 chunk ids
//   [vec_off..) count*dim f32, vec_off = alignForward(48 + 8*count, 64)
// File length must equal vec_off + 4*count*dim exactly.

const MAGIC = "FROSTIDX";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 48;
const HASHED_LEN: usize = 40;

/// Opaque to C. Holds the read-only mapping and pointers into it.
pub const FrostIndex = extern struct {
    map_ptr: [*]align(std.heap.page_size_min) const u8,
    map_len: usize,
    ids: [*]const u64,
    vecs: [*]const f32,
    count: u64,
    generation: u64,
    dim: u32,
};

const Layout = struct { vec_off: usize, total: usize };

fn layout(count: u64, dim: u32) error{Overflow}!Layout {
    const n = std.math.cast(usize, count) orelse return error.Overflow;
    const ids_end = try std.math.add(usize, HEADER_LEN, try std.math.mul(usize, n, 8));
    const vec_off = std.mem.alignForward(usize, ids_end, 64);
    if (vec_off < ids_end) return error.Overflow;
    const floats = try std.math.mul(usize, n, dim);
    const total = try std.math.add(usize, vec_off, try std.math.mul(usize, floats, 4));
    return .{ .vec_off = vec_off, .total = total };
}

fn writeAll(fd: c.fd_t, bytes: []const u8) bool {
    var rest = bytes;
    while (rest.len > 0) {
        // Cap per call: some libcs reject writes > INT_MAX.
        const n = c.write(fd, rest.ptr, @min(rest.len, 1 << 30));
        if (n <= 0) return false;
        rest = rest[@intCast(n)..];
    }
    return true;
}

/// Writes `<path>.tmp`, fsyncs, then renames over `path` (atomic generation swap).
export fn frost_index_write(
    path: [*:0]const u8,
    ids: [*]const u64,
    vectors: [*]const f32,
    count: u64,
    dim: u32,
    generation: u64,
) i32 {
    if (dim == 0) return E_INVALID_ARG;
    const lay = layout(count, dim) catch return E_OVERFLOW;
    const n: usize = @intCast(count);

    var hdr = [_]u8{0} ** HEADER_LEN;
    @memcpy(hdr[0..8], MAGIC);
    std.mem.writeInt(u32, hdr[8..12], VERSION, .little);
    std.mem.writeInt(u32, hdr[12..16], dim, .little);
    std.mem.writeInt(u64, hdr[16..24], count, .little);
    std.mem.writeInt(u64, hdr[24..32], generation, .little);
    // flags [32..36) and pad [36..40) stay zero.
    std.mem.writeInt(u64, hdr[40..48], std.hash.Fnv1a_64.hash(hdr[0..HASHED_LEN]), .little);

    const plen = std.mem.len(path);
    const tmp_z = std.heap.c_allocator.allocSentinel(u8, plen + 4, 0) catch return E_NOMEM;
    defer std.heap.c_allocator.free(tmp_z);
    @memcpy(tmp_z[0..plen], path[0..plen]);
    @memcpy(tmp_z[plen..], ".tmp");

    const fd = c.open(tmp_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c_uint, 0o644));
    if (fd < 0) return E_IO;

    const pad = [_]u8{0} ** 64;
    const ids_end = HEADER_LEN + n * 8;
    const ok = writeAll(fd, &hdr) and
        writeAll(fd, std.mem.sliceAsBytes(ids[0..n])) and
        writeAll(fd, pad[0 .. lay.vec_off - ids_end]) and
        writeAll(fd, std.mem.sliceAsBytes(vectors[0 .. n * dim])) and
        c.fsync(fd) == 0;
    const closed = c.close(fd) == 0;
    if (!ok or !closed or c.rename(tmp_z.ptr, path) != 0) {
        _ = c.unlink(tmp_z.ptr);
        return E_IO;
    }
    return OK;
}

export fn frost_index_open(path: [*:0]const u8, out: **FrostIndex) i32 {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return E_IO;
    defer _ = c.close(fd); // the mapping outlives the descriptor

    var st: c.Stat = undefined;
    if (c.fstat(fd, &st) != 0) return E_IO;
    if (st.size < HEADER_LEN) return E_BAD_LENGTH;
    const len = std.math.cast(usize, st.size) orelse return E_OVERFLOW;

    const raw = c.mmap(null, len, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    if (raw == c.MAP_FAILED) return E_IO;
    const map: [*]align(std.heap.page_size_min) const u8 = @ptrCast(@alignCast(raw));
    const rc = validate(map, len, out);
    if (rc != OK) _ = c.munmap(@ptrCast(map), len);
    return rc;
}

fn validate(map: [*]align(std.heap.page_size_min) const u8, len: usize, out: **FrostIndex) i32 {
    const hdr = map[0..HEADER_LEN];
    if (!std.mem.eql(u8, hdr[0..8], MAGIC)) return E_BAD_MAGIC;
    if (std.mem.readInt(u64, hdr[40..48], .little) != std.hash.Fnv1a_64.hash(hdr[0..HASHED_LEN])) return E_BAD_HASH;
    const version = std.mem.readInt(u32, hdr[8..12], .little);
    const dim = std.mem.readInt(u32, hdr[12..16], .little);
    const count = std.mem.readInt(u64, hdr[16..24], .little);
    const pad = std.mem.readInt(u32, hdr[36..40], .little);
    if (version != VERSION or dim == 0 or pad != 0) return E_BAD_HEADER;
    const lay = layout(count, dim) catch return E_OVERFLOW;
    if (lay.total != len) return E_BAD_LENGTH;

    const idx = std.heap.c_allocator.create(FrostIndex) catch return E_NOMEM;
    idx.* = .{
        .map_ptr = map,
        .map_len = len,
        .ids = @ptrCast(@alignCast(map + HEADER_LEN)),
        .vecs = @ptrCast(@alignCast(map + lay.vec_off)),
        .count = count,
        .generation = std.mem.readInt(u64, hdr[24..32], .little),
        .dim = dim,
    };
    out.* = idx;
    return OK;
}

export fn frost_index_close(idx: *FrostIndex) void {
    _ = c.munmap(@ptrCast(idx.map_ptr), idx.map_len);
    std.heap.c_allocator.destroy(idx);
}

export fn frost_index_meta(idx: *const FrostIndex, out_dim: *u32, out_count: *u64, out_generation: *u64) void {
    out_dim.* = idx.dim;
    out_count.* = idx.count;
    out_generation.* = idx.generation;
}

/// Exact brute-force top-k by dot product. Results are written in descending
/// score order, ties broken by lower id. Vectors scoring NaN are skipped.
/// out_ids/out_scores must hold at least min(k, count) entries.
export fn frost_index_search(
    idx: *const FrostIndex,
    query: [*]const f32,
    dim: u32,
    k: u32,
    out_ids: [*]u64,
    out_scores: [*]f32,
    out_n: *u32,
) i32 {
    out_n.* = 0;
    if (dim != idx.dim) return E_DIM_MISMATCH;
    for (query[0..dim]) |x| if (!std.math.isFinite(x)) return E_NONFINITE;
    if (k == 0 or idx.count == 0) return OK;

    const kk: usize = @intCast(@min(@as(u64, k), idx.count));
    const d: usize = dim;
    // ponytail: sorted k-buffer with insertion, O(k) per accepted candidate;
    // swap for a binary heap if k ever grows into the thousands.
    var n: usize = 0;
    var i: usize = 0;
    while (i < idx.count) : (i += 1) {
        const s = dot8(query, idx.vecs + i * d, d);
        if (std.math.isNan(s)) continue;
        const id = idx.ids[i];
        if (n == kk and !better(s, id, out_scores[kk - 1], out_ids[kk - 1])) continue;
        var pos = if (n < kk) n else kk - 1;
        while (pos > 0 and better(s, id, out_scores[pos - 1], out_ids[pos - 1])) : (pos -= 1) {
            out_scores[pos] = out_scores[pos - 1];
            out_ids[pos] = out_ids[pos - 1];
        }
        out_scores[pos] = s;
        out_ids[pos] = id;
        if (n < kk) n += 1;
    }
    out_n.* = @intCast(n);
    return OK;
}

inline fn better(s: f32, id: u64, s2: f32, id2: u64) bool {
    return s > s2 or (s == s2 and id < id2);
}

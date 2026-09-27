# FROST sampling kernels (top-k, softmax + top-p), exported with a C ABI.
#
# Built by crates/frost-kernels/build.rs with the *native* Mojo driver
# (<sdk>/modular/bin/mojo), never the Python entrypoint; see verification/mojo_audit.md.
#
# Nightly rules verified on Mojo 1.1.0.dev2026082005 (c72288dd):
#   - `@export` decorator + `abi("C")` after the signature ( `@export("n", abi("C"))` is rejected )
#   - `UnsafePointer` is deprecated -> `Pointer[T, Origin]`; rebuild from an address with
#     `Pointer[T, MutAnyOrigin](unsafe_from_address=addr)` and index with `p[unsafe_offset=i]`
#   - `sort` is builtin: `sort(list, cmp_fn)`; `cmp_fn(a, b)` returns True when a sorts before b
#
# Every entry point returns 0 on success or one of these negative codes (mirrored in Rust):
#   -1 n == 0        -2 k == 0        -3 k > n         -4 NaN in logits
#   -5 temperature is NaN or <= 0     -6 top_p is NaN or not in (0, 1]
#   -7 degenerate distribution (max logit not finite: any +inf, or all -inf)
from std.math import exp, isnan, isinf
from std.memory import Pointer
from std.collections import List


@fieldwise_init
struct Item(Copyable, Movable):
    var idx: Int
    var val: Float64


def _desc(a: Item, b: Item) -> Bool:
    # Descending by value; equal values keep the lower index first (deterministic).
    if a.val > b.val:
        return True
    if a.val < b.val:
        return False
    return a.idx < b.idx


@fieldwise_init
struct TopItem(Copyable, ImplicitlyCopyable, Movable):
    var idx: Int
    var val: Float32


def _top_desc(a: TopItem, b: TopItem) -> Bool:
    if a.val > b.val:
        return True
    if a.val < b.val:
        return False
    return a.idx < b.idx


def _worse(a: TopItem, b: TopItem) -> Bool:
    # True when `a` must leave the top-k before `b`: smaller value, or equal value with a
    # higher index (ties keep the lower index).
    if a.val < b.val:
        return True
    if a.val > b.val:
        return False
    return a.idx > b.idx


def _sift_down(mut h: List[TopItem], start: Int):
    # Min-heap on `_worse`: the element to evict next sits at h[0].
    var size = len(h)
    var i = start
    while True:
        var m = 2 * i + 1
        if m >= size:
            return
        if m + 1 < size and _worse(h[m + 1], h[m]):
            m += 1
        if not _worse(h[m], h[i]):
            return
        var t = h[i]
        h[i] = h[m]
        h[m] = t
        i = m


@export
def frost_kernels_abi_version() abi("C") -> Int32:
    return 1


@export
def frost_topk_f32(
    logits_addr: Int, n: Int, k: Int, out_idx_addr: Int, out_val_addr: Int
) abi("C") -> Int32:
    if n <= 0:
        return -1
    if k <= 0:
        return -2
    if k > n:
        return -3
    var logits = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=logits_addr)
    var out_idx = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=out_idx_addr)
    var out_val = Pointer[Float32, MutAnyOrigin](unsafe_from_address=out_val_addr)
    # Bounded min-heap of the k best seen so far: O(n log k), no sort of the input.
    var heap = List[TopItem](capacity=k)
    for i in range(k):
        var v = logits[unsafe_offset=i]
        if isnan(v):
            return -4
        heap.append(TopItem(i, v))
    var j = k // 2 - 1
    while j >= 0:
        _sift_down(heap, j)
        j -= 1
    for i in range(k, n):
        var v = logits[unsafe_offset=i]
        if isnan(v):
            return -4
        # Later indices lose ties, so only a strictly larger value displaces the current worst.
        if v > heap[0].val:
            heap[0] = TopItem(i, v)
            _sift_down(heap, 0)
    sort(heap, _top_desc)
    for i in range(k):
        out_idx[unsafe_offset=i] = UInt32(heap[i].idx)
        out_val[unsafe_offset=i] = heap[i].val
    return 0


@export
def frost_softmax_topp_f32(
    logits_addr: Int,
    n: Int,
    temperature: Float32,
    top_p: Float32,
    out_prob_addr: Int,
    out_idx_addr: Int,
    out_n_addr: Int,
) abi("C") -> Int32:
    if n <= 0:
        return -1
    if isnan(temperature) or temperature <= 0.0:
        return -5
    if isnan(top_p) or top_p <= 0.0 or top_p > 1.0:
        return -6
    var logits = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=logits_addr)
    var out_prob = Pointer[Float32, MutAnyOrigin](unsafe_from_address=out_prob_addr)
    var out_idx = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=out_idx_addr)
    var out_n = Pointer[Int, MutAnyOrigin](unsafe_from_address=out_n_addr)

    # Scale in Float64, track the max for a stable softmax.
    var inv_t = 1.0 / Float64(temperature)
    var items = List[Item](capacity=n)
    var mx: Float64 = 0.0
    for i in range(n):
        var v = logits[unsafe_offset=i]
        if isnan(v):
            return -4
        var z = Float64(v) * inv_t
        if i == 0 or z > mx:
            mx = z
        items.append(Item(i, z))
    if isinf(mx):
        return -7

    # exp(z - max) accumulated in Float64, then normalize.
    var total: Float64 = 0.0
    for i in range(n):
        var p = exp(items[i].val - mx)
        items[i].val = p
        total += p
    for i in range(n):
        items[i].val = items[i].val / total

    # Smallest descending prefix whose cumulative mass reaches top_p (all n if rounding never gets there).
    sort(items, _desc)
    var count = n
    var cum: Float64 = 0.0
    var target = Float64(top_p)
    for i in range(n):
        cum += items[i].val
        if cum >= target:
            count = i + 1
            break

    var kept: Float64 = 0.0
    for i in range(count):
        kept += items[i].val
    for i in range(count):
        out_prob[unsafe_offset=i] = Float32(items[i].val / kept)
        out_idx[unsafe_offset=i] = UInt32(items[i].idx)
    out_n[unsafe_offset=0] = count
    return 0

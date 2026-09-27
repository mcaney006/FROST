# zig-rle

`rle.encode("aaabcc", ...)` produces `3 a 1 b 1 c 1 c` instead of `3 a 1 b 2 c`: the run that reaches the end of the input is split, with its final byte emitted as a separate run of length 1. A 300-byte run of `z` likewise comes out as three pairs instead of `255 z 45 z`. Each maximal run of identical bytes should become one (count, byte) pair, split only when it exceeds 255. Fix `rle.zig`; do not modify `held_out_test.zig`.

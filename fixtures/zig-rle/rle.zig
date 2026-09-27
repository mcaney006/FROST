//! Run-length encoder: each run of identical bytes becomes a (count, byte)
//! pair. Runs longer than 255 are split across several pairs.

pub const Error = error{OutputTooSmall};

/// Encodes `input` into `out` and returns the number of bytes written.
pub fn encode(input: []const u8, out: []u8) Error!usize {
    var written: usize = 0;
    var i: usize = 0;
    while (i < input.len) {
        const byte = input[i];
        var run: usize = 1;
        while (i + run < input.len - 1 and input[i + run] == byte and run < 255) : (run += 1) {}
        if (written + 2 > out.len) return error.OutputTooSmall;
        out[written] = @intCast(run);
        out[written + 1] = byte;
        written += 2;
        i += run;
    }
    return written;
}

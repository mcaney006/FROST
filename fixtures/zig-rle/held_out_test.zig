// HELD-OUT acceptance oracle. The assistant must never edit this file.
const std = @import("std");
const rle = @import("rle.zig");

fn expectEncoded(input: []const u8, expected: []const u8) !void {
    var buf: [1024]u8 = undefined;
    const n = try rle.encode(input, &buf);
    try std.testing.expectEqualSlices(u8, expected, buf[0..n]);
}

test "held out: run-length encoding" {
    try expectEncoded("", "");
    try expectEncoded("x", &.{ 1, 'x' });
    try expectEncoded("aaabcc", &.{ 3, 'a', 1, 'b', 2, 'c' });
    var long: [300]u8 = undefined;
    @memset(&long, 'z');
    try expectEncoded(&long, &.{ 255, 'z', 45, 'z' });
}

# FROST Mojo kernel helper (persistent coprocessor, bounded line-framed IPC).
#
# Compiled Mojo doing real numerical work on the decision path. This nightly's
# Pointer/origin unification blocks a clean buffer C-ABI export, so per the
# build contract we use the sanctioned fallback: a persistent native helper
# over bounded IPC (one long-lived process, never spawned per request).
#
# Protocol (one request per line on stdin, one response line on stdout):
#   NORM <n> <v0..v[n-1]>            -> L2-normalized vector
#   SCORE <d> <k> <q0..q[d-1]> <c...> -> k cosine scores of q vs each candidate
#   PING                             -> PONG   (liveness)
# Blank line or EOF terminates the loop.
from std.math import sqrt

def norm_inplace(mut v: List[Float32]):
    var ss: Float32 = 0.0
    for i in range(len(v)):
        ss += v[i] * v[i]
    if ss > 0.0:
        var inv = Float32(1.0) / sqrt(ss)
        for i in range(len(v)):
            v[i] = v[i] * inv

def parse_floats(parts: List[String], start: Int, count: Int) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(count):
        out.append(Float32(Float64(parts[start + i])))
    return out^

def main() raises:
    while True:
        var line: String
        try:
            line = input()
        except:
            break
        if line.byte_length() == 0:
            break
        var raw = line.split(" ")
        var parts = List[String]()
        for p in raw:
            if p.byte_length() > 0:
                parts.append(String(p))
        if len(parts) == 0:
            print("")
            continue
        var op = parts[0]
        var out = String("")
        if op == "PING":
            print("PONG")
            continue
        elif op == "NORM":
            var n = Int(parts[1])
            var v = parse_floats(parts, 2, n)
            norm_inplace(v)
            for i in range(n):
                if i > 0: out += " "
                out += String(v[i])
        elif op == "SCORE":
            var d = Int(parts[1])
            var k = Int(parts[2])
            var q = parse_floats(parts, 3, d)
            norm_inplace(q)
            for j in range(k):
                var c = parse_floats(parts, 3 + d + j * d, d)
                norm_inplace(c)
                var s: Float32 = 0.0
                for t in range(d):
                    s += q[t] * c[t]
                if j > 0: out += " "
                out += String(s)
        else:
            out = String("ERR unknown_op")
        print(out)

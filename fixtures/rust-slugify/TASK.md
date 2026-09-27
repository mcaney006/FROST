# rust-slugify

`slugify("Hello, World!")` returns `"hello-worl"` instead of `"hello-world"`: the last character of every slug goes missing, and a single-character input such as `"a"` comes back as an empty string. Slugs should be the input's ASCII letters and digits, lowercased, with each run of other characters collapsed into one `-` and no leading or trailing `-`. Fix `src/lib.rs`; do not modify `tests/held_out.rs`.

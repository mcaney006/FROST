// HELD-OUT acceptance oracle. The assistant must never edit this file.
use slugify::slugify;

#[test]
fn held_out() {
    assert_eq!(slugify("Hello, World!"), "hello-world");
    assert_eq!(slugify("  Rust   is -- fun  "), "rust-is-fun");
    assert_eq!(slugify("a"), "a");
    assert_eq!(slugify("---"), "");
}

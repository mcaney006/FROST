/// Converts arbitrary text into a URL slug: ASCII letters and digits are kept
/// (lowercased), every run of other characters becomes a single `-`, and the
/// slug never starts or ends with `-`.
pub fn slugify(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    // Trim the separator runs at both ends.
    let start = out.find(|c| c != '-').unwrap_or(out.len());
    let end = out.rfind(|c| c != '-').unwrap_or(0);
    if start >= end {
        return String::new();
    }
    out[start..end].to_string()
}

# c-parse-kv

`kv_parse_line` accepts a line whose value is exactly 64 characters long (`KV_VALUE_MAX`) instead of rejecting it, and the sanitizer build of the test aborts with an AddressSanitizer `heap-buffer-overflow` WRITE inside `kv_parse_line` on that input. Values up to 63 characters parse correctly. Per `kv.h`, both fields must stay NUL-terminated inside their fixed-size buffers. Fix `kv.c`; do not modify `held_out_test_kv.c`.

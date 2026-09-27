#ifndef KV_H
#define KV_H

#define KV_KEY_MAX 16
#define KV_VALUE_MAX 64

struct kv_pair {
    char key[KV_KEY_MAX];
    char value[KV_VALUE_MAX];
};

/*
 * Parses one "key=value" line; a trailing "\n" or "\r\n" is not part of the
 * value. Both fields are NUL-terminated, so a key holds at most KV_KEY_MAX - 1
 * characters and a value at most KV_VALUE_MAX - 1.
 *
 * Returns 0 on success, or -1 (leaving *out unspecified) if the line has no
 * '=', the key is empty, or the key or value does not fit.
 */
int kv_parse_line(const char *line, struct kv_pair *out);

#endif

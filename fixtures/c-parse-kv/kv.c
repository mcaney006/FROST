#include "kv.h"

#include <string.h>

int kv_parse_line(const char *line, struct kv_pair *out) {
    const char *eq = strchr(line, '=');
    if (eq == NULL || eq == line) {
        return -1;
    }
    size_t key_len = (size_t)(eq - line);
    const char *value = eq + 1;
    size_t value_len = strcspn(value, "\r\n");
    if (key_len >= sizeof out->key || value_len > sizeof out->value) {
        return -1;
    }
    strncpy(out->key, line, key_len);
    out->key[key_len] = '\0';
    strncpy(out->value, value, value_len);
    out->value[value_len] = '\0';
    return 0;
}

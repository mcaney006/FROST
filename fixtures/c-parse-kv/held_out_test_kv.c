/* HELD-OUT acceptance oracle. The assistant must never edit this file. */
#include "kv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            failures++;                                                        \
        }                                                                      \
    } while (0)

/* Fills buf with "k=" followed by value_len 'v' characters. */
static void make_line(char *buf, size_t value_len) {
    buf[0] = 'k';
    buf[1] = '=';
    memset(buf + 2, 'v', value_len);
    buf[2 + value_len] = '\0';
}

int main(void) {
    /* Heap-allocated so AddressSanitizer sees the exact object bounds. */
    struct kv_pair *p = malloc(sizeof *p);
    char line[KV_VALUE_MAX + 8];
    if (p == NULL) {
        return 2;
    }

    CHECK(kv_parse_line("port=8080\n", p) == 0);
    CHECK(strcmp(p->key, "port") == 0);
    CHECK(strcmp(p->value, "8080") == 0);
    CHECK(kv_parse_line("novalue", p) == -1);
    CHECK(kv_parse_line("=x", p) == -1);

    make_line(line, KV_VALUE_MAX - 1); /* longest value that fits */
    CHECK(kv_parse_line(line, p) == 0);
    CHECK(strlen(p->value) == KV_VALUE_MAX - 1);

    make_line(line, KV_VALUE_MAX); /* one more character must be rejected */
    CHECK(kv_parse_line(line, p) == -1);

    free(p);
    if (failures != 0) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    puts("all kv checks passed");
    return 0;
}

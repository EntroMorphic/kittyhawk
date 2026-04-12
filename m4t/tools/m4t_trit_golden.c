/*
 * m4t_trit_golden.c — enumerate all trit-pair combinations and print
 * expected outputs for each TBL-based binary trit opcode.
 *
 * This is a host-side tool. It uses printf (not m4t runtime). The output
 * is used to generate exhaustive test vectors for the 6 TBL opcodes.
 *
 * Build: cc -o m4t_trit_golden m4t/tools/m4t_trit_golden.c
 * Run:   ./m4t_trit_golden
 */

#include <stdio.h>
#include <stdlib.h>

static int trit_mul(int a, int b) {
    return a * b;  /* F_3: {-1,0,+1} × {-1,0,+1} → {-1,0,+1} */
}

static int trit_sat_add(int a, int b) {
    int s = a + b;
    if (s > 1) s = 1;
    if (s < -1) s = -1;
    return s;
}

static int trit_max(int a, int b) {
    return (a > b) ? a : b;
}

static int trit_min(int a, int b) {
    return (a < b) ? a : b;
}

static int trit_eq(int a, int b) {
    return (a == b) ? 1 : 0;
}

static int trit_neg(int a, int b) {
    (void)b;
    return -a;
}

typedef int (*trit_op_fn)(int a, int b);

static const char* op_names[] = {
    "mul", "sat_add", "max", "min", "eq", "neg"
};
static trit_op_fn op_fns[] = {
    trit_mul, trit_sat_add, trit_max, trit_min, trit_eq, trit_neg
};

/* 2-bit trit encoding: 0→0b00, +1→0b01, -1→0b10 */
static int trit_vals[3] = { -1, 0, 1 };
static int trit_codes[3] = { 2, 0, 1 };  /* -1→2, 0→0, +1→1 */

int main(void) {
    int n_ops = (int)(sizeof(op_fns) / sizeof(op_fns[0]));

    for (int op = 0; op < n_ops; op++) {
        printf("/* %s LUT — 16-byte table indexed by (a_code << 2) | b_code */\n", op_names[op]);
        printf("static const int8_t LUT_%s[16] = {\n    ", op_names[op]);

        for (int idx = 0; idx < 16; idx++) {
            int a_code = (idx >> 2) & 0x3;
            int b_code = idx & 0x3;

            /* Find trit values for these codes, or 0 for reserved (code=3) */
            int a_val = 0, b_val = 0;
            for (int i = 0; i < 3; i++) {
                if (trit_codes[i] == a_code) a_val = trit_vals[i];
                if (trit_codes[i] == b_code) b_val = trit_vals[i];
            }

            int result = op_fns[op](a_val, b_val);

            /* Convert result trit to code */
            int result_code = 0;
            for (int i = 0; i < 3; i++) {
                if (trit_vals[i] == result) { result_code = trit_codes[i]; break; }
            }

            /* For the decode LUT we want signed int8, not codes */
            printf("%2d", result);
            if (idx < 15) printf(", ");
            if (idx == 7) printf("\n    ");
        }
        printf("\n};\n\n");

        /* Also print the truth table for human verification */
        printf("/* %s truth table:\n", op_names[op]);
        printf(" *   a\\b | -1 |  0 | +1\n");
        printf(" *   ----+----+----+----\n");
        for (int ai = 0; ai < 3; ai++) {
            int a = trit_vals[ai];
            printf(" *   %+d  |", a);
            for (int bi = 0; bi < 3; bi++) {
                int b = trit_vals[bi];
                int r = op_fns[op](a, b);
                printf(" %+d |", r);
            }
            printf("\n");
        }
        printf(" */\n\n");
    }

    return 0;
}

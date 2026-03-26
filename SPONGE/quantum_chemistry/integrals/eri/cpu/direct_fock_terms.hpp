#pragma once

struct QC_Angular_Term_CPU
{
    unsigned short hr_offset;
    float coeff;
};

// Keep in sync with MAX_CART_SHELL in quantum_chemistry.h.
constexpr int QC_MAX_CART_SHELL_CPU = 15;
constexpr int QC_MAX_PAIR_TERM_COUNT_CPU = 9 * 9 * 9;
constexpr int QC_MAX_CART_PAIR_COUNT_CPU =
    QC_MAX_CART_SHELL_CPU * QC_MAX_CART_SHELL_CPU;

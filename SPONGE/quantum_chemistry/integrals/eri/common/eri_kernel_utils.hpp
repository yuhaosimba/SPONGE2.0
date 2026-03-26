#pragma once

#include "../../one_e.hpp"

static __device__ __forceinline__ int QC_Shell_Pair_Index(const int a,
                                                          const int b)
{
    return (a >= b) ? (a * (a + 1) / 2 + b) : (b * (b + 1) / 2 + a);
}

static __device__ __forceinline__ int QC_AO_Pair_Index(const int a, const int b)
{
    return (a >= b) ? (a * (a + 1) / 2 + b) : (b * (b + 1) / 2 + a);
}

static __device__ __forceinline__ int QC_Shell_Buffer_Index(
    int a, int b, int c, int d, const int dim1, const int dim2, const int dim3)
{
    return ((a * dim1 + b) * dim2 + c) * dim3 + d;
}

static __device__ __forceinline__ int QC_Shell_Dim(const int l,
                                                   const int is_spherical)
{
    return is_spherical ? (2 * l + 1) : ((l + 1) * (l + 2) / 2);
}

static __device__ __forceinline__ float QC_Max4(const float a, const float b,
                                                const float c, const float d)
{
    return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

static __device__ __forceinline__ void QC_Fock_Add(float* x, const float value)
{
#ifdef GPU_ARCH_NAME
    atomicAdd(x, value);
#else
    *x += value;
#endif
}

static __device__ __forceinline__ bool QC_Same_Ordered_Fock_Term(
    const int lhs[4], const int rhs[4])
{
    return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] &&
           lhs[3] == rhs[3];
}

static __device__ __forceinline__ void QC_Accumulate_Fock_Unique_Quartet(
    const int p, const int q, const int r, const int s, const float value,
    const int nao, const float* P_coul, const float* P_exx_a,
    const float* P_exx_b, const float exx_scale_a, const float exx_scale_b,
    float* F_a, float* F_b)
{
    const int j_terms[8][4] = {{p, q, r, s}, {q, p, r, s}, {p, q, s, r},
                               {q, p, s, r}, {r, s, p, q}, {s, r, p, q},
                               {r, s, q, p}, {s, r, q, p}};
    for (int n = 0; n < 8; n++)
    {
        bool duplicate = false;
        for (int prev = 0; prev < n; prev++)
            if (QC_Same_Ordered_Fock_Term(j_terms[n], j_terms[prev]))
            {
                duplicate = true;
                break;
            }
        if (duplicate) continue;
        const int i = j_terms[n][0], j = j_terms[n][1];
        const int k = j_terms[n][2], l = j_terms[n][3];
        const float j_val = P_coul[k * nao + l] * value;
        QC_Fock_Add(&F_a[i * nao + j], j_val);
        if (F_b != NULL) QC_Fock_Add(&F_b[i * nao + j], j_val);
    }
    const int k_terms[8][4] = {{p, r, q, s}, {p, s, q, r}, {q, r, p, s},
                               {q, s, p, r}, {r, p, s, q}, {r, q, s, p},
                               {s, p, r, q}, {s, q, r, p}};
    for (int n = 0; n < 8; n++)
    {
        bool duplicate = false;
        for (int prev = 0; prev < n; prev++)
            if (QC_Same_Ordered_Fock_Term(k_terms[n], k_terms[prev]))
            {
                duplicate = true;
                break;
            }
        if (duplicate) continue;
        const int i = k_terms[n][0], j = k_terms[n][1];
        const int k = k_terms[n][2], l = k_terms[n][3];
        if (exx_scale_a != 0.0f)
        {
            const float exx_a = -exx_scale_a * P_exx_a[k * nao + l] * value;
            QC_Fock_Add(&F_a[i * nao + j], exx_a);
        }
        if (F_b != NULL && P_exx_b != NULL && exx_scale_b != 0.0f)
        {
            const float exx_b = -exx_scale_b * P_exx_b[k * nao + l] * value;
            QC_Fock_Add(&F_b[i * nao + j], exx_b);
        }
    }
}

static __device__ __forceinline__ void QC_Accumulate_Fock_General_Quartet(
    const int p, const int q, const int r, const int s, const float value,
    const int nao, const float* P_coul, const float* P_exx_a,
    const float* P_exx_b, const float exx_scale_a, const float exx_scale_b,
    float* F_a, float* F_b)
{
    const int pn = p * nao, qn = q * nao, rn = r * nao, sn = s * nao;
    const float Ppq_sym = P_coul[pn + q] + P_coul[qn + p];
    const float Prs_sym = P_coul[rn + s] + P_coul[sn + r];
    const float j_pq = Prs_sym * value, j_rs = Ppq_sym * value;
    QC_Fock_Add(&F_a[pn + q], j_pq);
    QC_Fock_Add(&F_a[qn + p], j_pq);
    QC_Fock_Add(&F_a[rn + s], j_rs);
    QC_Fock_Add(&F_a[sn + r], j_rs);
    if (F_b != NULL)
    {
        QC_Fock_Add(&F_b[pn + q], j_pq);
        QC_Fock_Add(&F_b[qn + p], j_pq);
        QC_Fock_Add(&F_b[rn + s], j_rs);
        QC_Fock_Add(&F_b[sn + r], j_rs);
    }
    if (exx_scale_a != 0.0f)
    {
        const float nsv = -exx_scale_a * value;
        QC_Fock_Add(&F_a[pn + r], nsv * P_exx_a[qn + s]);
        QC_Fock_Add(&F_a[rn + p], nsv * P_exx_a[qn + s]);
        QC_Fock_Add(&F_a[pn + s], nsv * P_exx_a[qn + r]);
        QC_Fock_Add(&F_a[sn + p], nsv * P_exx_a[qn + r]);
        QC_Fock_Add(&F_a[qn + r], nsv * P_exx_a[pn + s]);
        QC_Fock_Add(&F_a[rn + q], nsv * P_exx_a[pn + s]);
        QC_Fock_Add(&F_a[qn + s], nsv * P_exx_a[pn + r]);
        QC_Fock_Add(&F_a[sn + q], nsv * P_exx_a[pn + r]);
    }
    if (F_b != NULL && P_exx_b != NULL && exx_scale_b != 0.0f)
    {
        const float nsv = -exx_scale_b * value;
        QC_Fock_Add(&F_b[pn + r], nsv * P_exx_b[qn + s]);
        QC_Fock_Add(&F_b[rn + p], nsv * P_exx_b[qn + s]);
        QC_Fock_Add(&F_b[pn + s], nsv * P_exx_b[qn + r]);
        QC_Fock_Add(&F_b[sn + p], nsv * P_exx_b[qn + r]);
        QC_Fock_Add(&F_b[qn + r], nsv * P_exx_b[pn + s]);
        QC_Fock_Add(&F_b[rn + q], nsv * P_exx_b[pn + s]);
        QC_Fock_Add(&F_b[qn + s], nsv * P_exx_b[pn + r]);
        QC_Fock_Add(&F_b[sn + q], nsv * P_exx_b[pn + r]);
    }
}

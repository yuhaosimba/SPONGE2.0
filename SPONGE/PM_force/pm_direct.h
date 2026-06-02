#pragma once

// Internal PM direct-space helpers.
// This header expects PM_Direct_Parameters to be defined by PM_force.h.

__host__ __device__ __forceinline__ float PM_Eval_Direct_Table(
    const float* table, int table_points, float x)
{
    if (table == NULL || table_points <= 0) return 0.0f;
    if (x <= 0.0f) return table[0];
    if (x >= 1.0f) return table[table_points - 1];
    float scaled = x * (table_points - 1);
    int lower = (int)scaled;
    int upper = lower + 1;
    if (upper >= table_points) upper = table_points - 1;
    float t = scaled - lower;
    return (1.0f - t) * table[lower] + t * table[upper];
}

__host__ __device__ __forceinline__ float PM_Eval_Direct_Poly(
    const float* coeff, int coeff_count, float x)
{
    if (coeff == NULL || coeff_count <= 0) return 0.0f;
    float value = 0.0f;
    for (int i = coeff_count - 1; i >= 0; i--)
    {
        value = value * x + coeff[i];
    }
    return value;
}

__host__ __device__ __forceinline__ int PM_Float_Is_Finite(float value)
{
#ifdef GPU_ARCH_NAME
    return isfinite(value);
#else
    union FloatBits
    {
        float f;
        unsigned int u;
    } bits;
    bits.f = value;
    return (bits.u & 0x7f800000u) != 0x7f800000u;
#endif
}

__host__ __device__ __forceinline__ int PM_Float_Is_Bounded(float value,
                                                            float limit)
{
    return PM_Float_Is_Finite(value) && value <= limit && value >= -limit;
}

__host__ __device__ __forceinline__ int PM_Direct_Uses_ESP(
    const PM_Direct_Parameters& pm_direct)
{
    return pm_direct.backend == ParticleMeshBackend::ESP;
}

__host__ __device__ __forceinline__ float PM_Eval_Direct_Scalar(
    const float* table, const float* coeff, int table_points, int coeff_count,
    int use_polynomial_tables, float x)
{
    if (use_polynomial_tables)
    {
        return PM_Eval_Direct_Poly(coeff, coeff_count, x);
    }
    return PM_Eval_Direct_Table(table, table_points, x);
}

__host__ __device__ __forceinline__ float PME_Get_Direct_Coulomb_Energy(
    float charge_product, float dr_abs, float pme_beta)
{
    return charge_product * erfcf(pme_beta * dr_abs) / dr_abs;
}

__host__ __device__ __forceinline__ float PME_Get_Direct_Coulomb_Force(
    float charge_product, float dr_abs, float pme_beta)
{
    float beta_dr = pme_beta * dr_abs;
    return charge_product * powf(dr_abs, -3.0f) *
           (beta_dr * 1.1283791670218446f * expf(-beta_dr * beta_dr) +
            erfcf(beta_dr));
}

__host__ __device__ __forceinline__ float PME_Get_Direct_Coulomb_Virial(
    float charge_product, float dr_abs, float pme_beta)
{
    float beta_dr = pme_beta * dr_abs;
    return charge_product / dr_abs *
           (beta_dr * 1.1283791670218446f * expf(-beta_dr * beta_dr) +
            erfcf(beta_dr));
}

__host__ __device__ __forceinline__ float PME_Get_Excluded_Coulomb_Energy(
    float charge_product, float dr_abs, float pme_beta)
{
    return -charge_product * erff(pme_beta * dr_abs) / dr_abs;
}

__host__ __device__ __forceinline__ float PME_Get_Excluded_Coulomb_Force(
    float charge_product, float dr_abs, float pme_beta)
{
    float dr2 = dr_abs * dr_abs;
    float beta_dr = pme_beta * dr_abs;
    float frc_abs = beta_dr * 1.1283791670218446f * expf(-beta_dr * beta_dr) +
                    erfcf(beta_dr);
    frc_abs = (frc_abs - 1.0f) / dr2 / dr_abs;
    return -charge_product * frc_abs;
}

__host__ __device__ __forceinline__ float ESP_Split_Long_Range_Factor(
    float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (!PM_Direct_Uses_ESP(pm_direct) || pm_direct.cutoff <= 0.0f)
        return 0.0f;
    float x = dr_abs / pm_direct.cutoff;
    if (x >= 1.0f) return 1.0f;
    if (x <= 0.0f) return 0.0f;
    if (pm_direct.table_points > 1)
    {
        return PM_Eval_Direct_Table(pm_direct.split_real_table,
                                    pm_direct.table_points, x);
    }
    return PM_Eval_Direct_Scalar(
        pm_direct.split_real_table, pm_direct.split_real_coeff,
        pm_direct.table_points, pm_direct.split_poly_order,
        pm_direct.use_polynomial_tables, x);
}

__host__ __device__ __forceinline__ float ESP_Split_Long_Range_Derivative(
    float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (!PM_Direct_Uses_ESP(pm_direct) || pm_direct.cutoff <= 0.0f)
        return 0.0f;
    float x = dr_abs / pm_direct.cutoff;
    if (x >= 1.0f) return 0.0f;
    if (x < 0.0f) x = 0.0f;
    if (pm_direct.table_points > 1)
    {
        return PM_Eval_Direct_Table(pm_direct.split_real_derivative_table,
                                    pm_direct.table_points, x) /
               pm_direct.cutoff;
    }
    float d_split_dx = PM_Eval_Direct_Scalar(
        pm_direct.split_real_derivative_table,
        pm_direct.split_real_derivative_coeff, pm_direct.table_points,
        pm_direct.split_poly_order, pm_direct.use_polynomial_tables, x);
    return d_split_dx / pm_direct.cutoff;
}

__host__ __device__ __forceinline__ float ESP_Get_Direct_Coulomb_Energy(
    float charge_product, float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (charge_product == 0.0f || !PM_Float_Is_Finite(dr_abs) ||
        dr_abs <= 1.0e-12f)
        return 0.0f;
    float split = ESP_Split_Long_Range_Factor(dr_abs, pm_direct);
    return charge_product * (1.0f - split) / dr_abs;
}

__host__ __device__ __forceinline__ float ESP_Get_Direct_Coulomb_Force(
    float charge_product, float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (charge_product == 0.0f || !PM_Float_Is_Finite(dr_abs) ||
        dr_abs <= 1.0e-12f)
        return 0.0f;
    float inv_r = 1.0f / dr_abs;
    float inv_r2 = inv_r * inv_r;
    float inv_r3 = inv_r2 * inv_r;
    float split = ESP_Split_Long_Range_Factor(dr_abs, pm_direct);
    float d_split_dr = ESP_Split_Long_Range_Derivative(dr_abs, pm_direct);
    return charge_product * ((1.0f - split) * inv_r3 + d_split_dr * inv_r2);
}

__host__ __device__ __forceinline__ float ESP_Get_Excluded_Coulomb_Energy(
    float charge_product, float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (charge_product == 0.0f || !PM_Float_Is_Finite(dr_abs) ||
        dr_abs <= 1.0e-12f)
        return 0.0f;
    float split = ESP_Split_Long_Range_Factor(dr_abs, pm_direct);
    return -charge_product * split / dr_abs;
}

__host__ __device__ __forceinline__ float ESP_Get_Excluded_Coulomb_Force(
    float charge_product, float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (charge_product == 0.0f || !PM_Float_Is_Finite(dr_abs) ||
        dr_abs <= 1.0e-12f)
        return 0.0f;
    float inv_r = 1.0f / dr_abs;
    float inv_r2 = inv_r * inv_r;
    float inv_r3 = inv_r2 * inv_r;
    float split = ESP_Split_Long_Range_Factor(dr_abs, pm_direct);
    float d_split_dr = ESP_Split_Long_Range_Derivative(dr_abs, pm_direct);
    return charge_product * (split * inv_r3 - d_split_dr * inv_r2);
}

__host__ __device__ __forceinline__ float PM_Get_Direct_Coulomb_Energy(
    float charge_product, float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (PM_Direct_Uses_ESP(pm_direct))
    {
        return ESP_Get_Direct_Coulomb_Energy(charge_product, dr_abs,
                                             pm_direct);
    }
    return PME_Get_Direct_Coulomb_Energy(charge_product, dr_abs,
                                         pm_direct.pme_beta);
}

__host__ __device__ __forceinline__ float PM_Get_Direct_Coulomb_Force(
    float charge_product, float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (PM_Direct_Uses_ESP(pm_direct))
    {
        return ESP_Get_Direct_Coulomb_Force(charge_product, dr_abs,
                                            pm_direct);
    }
    return PME_Get_Direct_Coulomb_Force(charge_product, dr_abs,
                                        pm_direct.pme_beta);
}

__host__ __device__ __forceinline__ float PM_Get_Excluded_Coulomb_Energy(
    float charge_product, float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (PM_Direct_Uses_ESP(pm_direct))
    {
        return ESP_Get_Excluded_Coulomb_Energy(charge_product, dr_abs,
                                               pm_direct);
    }
    return PME_Get_Excluded_Coulomb_Energy(charge_product, dr_abs,
                                           pm_direct.pme_beta);
}

__host__ __device__ __forceinline__ float PM_Get_Excluded_Coulomb_Force(
    float charge_product, float dr_abs, const PM_Direct_Parameters& pm_direct)
{
    if (PM_Direct_Uses_ESP(pm_direct))
    {
        return ESP_Get_Excluded_Coulomb_Force(charge_product, dr_abs,
                                              pm_direct);
    }
    return PME_Get_Excluded_Coulomb_Force(charge_product, dr_abs,
                                          pm_direct.pme_beta);
}

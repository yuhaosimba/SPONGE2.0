#pragma once

// Internal ESP direct-space helpers.
// This header expects ESP_Direct_Parameters to be defined by PM_force.h.

__host__ __device__ __forceinline__ float ESP_Eval_Direct_Table(
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

__host__ __device__ __forceinline__ float ESP_Eval_Direct_Poly(
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

__host__ __device__ __forceinline__ int ESP_Float_Is_Finite(float value)
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

__host__ __device__ __forceinline__ int ESP_Float_Is_Bounded(float value,
                                                             float limit)
{
    return ESP_Float_Is_Finite(value) && value <= limit && value >= -limit;
}

__host__ __device__ __forceinline__ float ESP_Eval_Direct_Scalar(
    const float* table, const float* coeff, int table_points, int coeff_count,
    int use_polynomial_tables, float x)
{
    if (use_polynomial_tables)
    {
        return ESP_Eval_Direct_Poly(coeff, coeff_count, x);
    }
    return ESP_Eval_Direct_Table(table, table_points, x);
}

__host__ __device__ __forceinline__ float ESP_Split_Long_Range_Factor(
    float dr_abs, const ESP_Direct_Parameters& esp_direct)
{
    if (!esp_direct.enabled || esp_direct.cutoff <= 0.0f) return 0.0f;
    float x = dr_abs / esp_direct.cutoff;
    if (x >= 1.0f) return 1.0f;
    if (x <= 0.0f) return 0.0f;
    if (esp_direct.table_points > 1)
    {
        return ESP_Eval_Direct_Table(esp_direct.split_real_table,
                                     esp_direct.table_points, x);
    }
    return ESP_Eval_Direct_Scalar(
        esp_direct.split_real_table, esp_direct.split_real_coeff,
        esp_direct.table_points, esp_direct.split_poly_order,
        esp_direct.use_polynomial_tables, x);
}

__host__ __device__ __forceinline__ float ESP_Split_Long_Range_Derivative(
    float dr_abs, const ESP_Direct_Parameters& esp_direct)
{
    if (!esp_direct.enabled || esp_direct.cutoff <= 0.0f) return 0.0f;
    float x = dr_abs / esp_direct.cutoff;
    if (x >= 1.0f) return 0.0f;
    if (x < 0.0f) x = 0.0f;
    if (esp_direct.table_points > 1)
    {
        return ESP_Eval_Direct_Table(esp_direct.split_real_derivative_table,
                                     esp_direct.table_points, x) /
               esp_direct.cutoff;
    }
    float d_split_dx = ESP_Eval_Direct_Scalar(
        esp_direct.split_real_derivative_table,
        esp_direct.split_real_derivative_coeff, esp_direct.table_points,
        esp_direct.split_poly_order, esp_direct.use_polynomial_tables, x);
    return d_split_dx / esp_direct.cutoff;
}

__host__ __device__ __forceinline__ float ESP_Get_Direct_Coulomb_Energy(
    float charge_product, float dr_abs, const ESP_Direct_Parameters& esp_direct)
{
    if (charge_product == 0.0f || !ESP_Float_Is_Finite(dr_abs) ||
        dr_abs <= 1.0e-12f)
        return 0.0f;
    float split = ESP_Split_Long_Range_Factor(dr_abs, esp_direct);
    return charge_product * (1.0f - split) / dr_abs;
}

__host__ __device__ __forceinline__ float ESP_Get_Direct_Coulomb_Force(
    float charge_product, float dr_abs, const ESP_Direct_Parameters& esp_direct)
{
    if (charge_product == 0.0f || !ESP_Float_Is_Finite(dr_abs) ||
        dr_abs <= 1.0e-12f)
        return 0.0f;
    float inv_r = 1.0f / dr_abs;
    float inv_r2 = inv_r * inv_r;
    float inv_r3 = inv_r2 * inv_r;
    float split = ESP_Split_Long_Range_Factor(dr_abs, esp_direct);
    float d_split_dr = ESP_Split_Long_Range_Derivative(dr_abs, esp_direct);
    return charge_product * ((1.0f - split) * inv_r3 + d_split_dr * inv_r2);
}

__host__ __device__ __forceinline__ float ESP_Get_Excluded_Coulomb_Energy(
    float charge_product, float dr_abs, const ESP_Direct_Parameters& esp_direct)
{
    if (charge_product == 0.0f || !ESP_Float_Is_Finite(dr_abs) ||
        dr_abs <= 1.0e-12f)
        return 0.0f;
    float split = ESP_Split_Long_Range_Factor(dr_abs, esp_direct);
    return -charge_product * split / dr_abs;
}

__host__ __device__ __forceinline__ float ESP_Get_Excluded_Coulomb_Force(
    float charge_product, float dr_abs, const ESP_Direct_Parameters& esp_direct)
{
    if (charge_product == 0.0f || !ESP_Float_Is_Finite(dr_abs) ||
        dr_abs <= 1.0e-12f)
        return 0.0f;
    float inv_r = 1.0f / dr_abs;
    float inv_r2 = inv_r * inv_r;
    float inv_r3 = inv_r2 * inv_r;
    float split = ESP_Split_Long_Range_Factor(dr_abs, esp_direct);
    float d_split_dr = ESP_Split_Long_Range_Derivative(dr_abs, esp_direct);
    return charge_product * (split * inv_r3 - d_split_dr * inv_r2);
}

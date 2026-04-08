#pragma once

#include "nbnxm_frozen_compat.h"

namespace sponge::nbnxm::frozen
{

static __forceinline__ __device__ float pmeCorrF(const float z2)
{
    constexpr float FN6 = -1.7357322914161492954e-8F;
    constexpr float FN5 = 1.4703624142580877519e-6F;
    constexpr float FN4 = -0.000053401640219807709149F;
    constexpr float FN3 = 0.0010054721316683106153F;
    constexpr float FN2 = -0.019278317264888380590F;
    constexpr float FN1 = 0.069670166153766424023F;
    constexpr float FN0 = -0.75225204789749321333F;

    constexpr float FD4 = 0.0011193462567257629232F;
    constexpr float FD3 = 0.014866955030185295499F;
    constexpr float FD2 = 0.11583842382862377919F;
    constexpr float FD1 = 0.50736591960530292870F;
    constexpr float FD0 = 1.0F;

    const float z4 = z2 * z2;

    float       polyFD0 = FD4 * z4 + FD2;
    const float polyFD1 = FD3 * z4 + FD1;
    polyFD0             = polyFD0 * z4 + FD0;
    polyFD0             = polyFD1 * z2 + polyFD0;

    polyFD0 = 1.0F / polyFD0;

    float polyFN0 = FN6 * z4 + FN4;
    float polyFN1 = FN5 * z4 + FN3;
    polyFN0       = polyFN0 * z4 + FN2;
    polyFN1       = polyFN1 * z4 + FN1;
    polyFN0       = polyFN0 * z4 + FN0;
    polyFN0       = polyFN1 * z2 + polyFN0;

    return polyFN0 * polyFD0;
}

static __forceinline__ __device__ float calculate_lj_ewald_c6grid(const NBParamGpu nbparam, int typei, int typej)
{
    const float c6_i = LDG(&nbparam.nbfp_comb[typei].x);
    const float c6_j = LDG(&nbparam.nbfp_comb[typej].x);
    return c6_i * c6_j;
}

static __forceinline__ __device__ void calculate_lj_ewald_comb_geom_F(const NBParamGpu nbparam,
                                                                       int              typei,
                                                                       int              typej,
                                                                       float            r2,
                                                                       float            inv_r2,
                                                                       float            lje_coeff2,
                                                                       float            lje_coeff6_6,
                                                                       float*           F_invr)
{
    float c6grid, inv_r6_nm, cr2, expmcr2, poly;

    c6grid = calculate_lj_ewald_c6grid(nbparam, typei, typej);

    inv_r6_nm = inv_r2 * inv_r2 * inv_r2;
    cr2       = lje_coeff2 * r2;
    expmcr2   = expf(-cr2);
    poly      = 1.0F + cr2 + 0.5F * cr2 * cr2;

    *F_invr += c6grid * (inv_r6_nm - expmcr2 * (inv_r6_nm * poly + lje_coeff6_6)) * inv_r2;
}

static __forceinline__ __device__ void fetch_nbfp_c6_c12(float& c6, float& c12, const NBParamGpu nbparam, int baseIndex)
{
    const float2 c6c12 = nbparam.nbfp[baseIndex];
    c6 = c6c12.x;
    c12 = c6c12.y;
}

static __forceinline__ __device__ void
reduce_force_j_warp_shfl(float3 f, float3* fout, int tidxi, int aidx, const unsigned int activemask)
{
    f.x += __shfl_down_sync(activemask, f.x, 1);
    f.y += __shfl_up_sync(activemask, f.y, 1);
    f.z += __shfl_down_sync(activemask, f.z, 1);

    if (tidxi & 1)
    {
        f.x = f.y;
    }

    f.x += __shfl_down_sync(activemask, f.x, 2);
    f.z += __shfl_up_sync(activemask, f.z, 2);

    if (tidxi & 2)
    {
        f.x = f.z;
    }

    f.x += __shfl_down_sync(activemask, f.x, 4);

    if (tidxi < 3)
    {
        atomicAdd((&fout[aidx].x) + tidxi, f.x);
    }
}

static __forceinline__ __device__ void reduce_force_i_warp_shfl(float3             fin,
                                                                 float3*            fout,
                                                                 float*             fshift_buf,
                                                                 bool               bCalcFshift,
                                                                 int                tidxj,
                                                                 int                aidx,
                                                                 const unsigned int activemask)
{
    fin.x += __shfl_down_sync(activemask, fin.x, c_clusterSize);
    fin.y += __shfl_up_sync(activemask, fin.y, c_clusterSize);
    fin.z += __shfl_down_sync(activemask, fin.z, c_clusterSize);

    if (tidxj & 1)
    {
        fin.x = fin.y;
    }

    fin.x += __shfl_down_sync(activemask, fin.x, 2 * c_clusterSize);
    fin.z += __shfl_up_sync(activemask, fin.z, 2 * c_clusterSize);

    if (tidxj & 2)
    {
        fin.x = fin.z;
    }

    if ((tidxj & 3) < 3)
    {
        atomicAdd(&fout[aidx].x + (tidxj & 3), fin.x);

        if (bCalcFshift)
        {
            *fshift_buf += fin.x;
        }
    }
}

} // namespace sponge::nbnxm::frozen

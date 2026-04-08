#include "ljewald_kernel_frozen_port.h"

#include "nbnxm_frozen_helpers.cuh"

#include <cassert>

namespace sponge::nbnxm::frozen
{
namespace
{

static constexpr int c_numThreadsZFrozen = 1;
static constexpr int c_minBlocksPerMpFrozen = 16;
static constexpr int c_threadsPerBlockFrozen = c_clusterSize * c_clusterSize * c_numThreadsZFrozen;
#if GMX_PTX_ARCH == 800
static constexpr int c_jmLoopUnrollFactorFrozen = 4;
#else
static constexpr int c_jmLoopUnrollFactorFrozen = 2;
#endif

__launch_bounds__(c_threadsPerBlockFrozen, c_minBlocksPerMpFrozen)
__global__ void nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_frozen_cuda(const NBAtomDataGpu atdat,
                                                                  const NBParamGpu    nbparam,
                                                                  const GpuPairlist   plist,
                                                                  bool                bCalcFshift)
{
    const nbnxn_sci_t*       pl_sci      = plist.sorting.sciSorted;
    const nbnxn_cj_packed_t* pl_cjPacked = plist.cjPacked;
    const nbnxn_excl_t*      excl        = plist.excl;
    const int*               atom_types  = atdat.atomTypes;
    const int                ntypes      = atdat.numTypes;
    const float4*            xq          = atdat.xq;
    float3*                  f           = asFloat3(atdat.f);
    const float3*            shift_vec   = asFloat3(atdat.shiftVec);
    const float              rcoulomb_sq = nbparam.rcoulomb_sq;
    const float              lje_coeff2  = nbparam.ewaldcoeff_lj * nbparam.ewaldcoeff_lj;
    const float              lje_coeff6_6 = lje_coeff2 * lje_coeff2 * lje_coeff2 * c_oneSixth;
    const float              beta2       = nbparam.ewald_beta * nbparam.ewald_beta;
    const float              beta3       = nbparam.ewald_beta * nbparam.ewald_beta * nbparam.ewald_beta;

    const unsigned int tidxi = threadIdx.x;
    const unsigned int tidxj = threadIdx.y;
    const unsigned int tidx  = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int tidxz = 0;
    const unsigned int bidx  = blockIdx.x;
    const unsigned int widx  = __shfl_sync(c_fullWarpMask, tidx / warp_size, 0, 32);

    int          sci, ci, cj, ai, aj, cijPackedBegin, cijPackedEnd;
    int          typei, typej;
    int          i, jm, jPacked, wexcl_idx;
    float        qi, qj_f;
    float        r2, inv_r, inv_r2;
    float        inv_r6, c6, c12;
    float        int_bit, F_invr;
    unsigned int wexcl, imask, mask_ji;
    float4       xqbuf;
    float3       xi, xj, rv, f_ij, fcj_buf;
    float3       fci_buf[c_superClusterSize];
    nbnxn_sci_t  nb_sci;

    constexpr bool     c_loadUsingAllXYThreads = (c_clusterSize == c_superClusterSize);
    constexpr bool     c_preloadCj = (GMX_PTX_ARCH < 700 || GMX_PTX_ARCH == 750);
    constexpr unsigned superClInteractionMask = ((1U << c_superClusterSize) - 1U);

    extern __shared__ char sm_dynamicShmem[];
    char*                  sm_nextSlotPtr = sm_dynamicShmem;

    float4* xqib = reinterpret_cast<float4*>(sm_nextSlotPtr);
    sm_nextSlotPtr += (c_superClusterSize * c_clusterSize * sizeof(*xqib));

    int* cjs = reinterpret_cast<int*>(sm_nextSlotPtr);
    if (c_preloadCj)
    {
        cjs += tidxz * c_clusterPairSplit * c_jGroupSize;
        sm_nextSlotPtr += (c_numThreadsZFrozen * c_clusterPairSplit * c_jGroupSize * sizeof(*cjs));
    }

    int* atib = reinterpret_cast<int*>(sm_nextSlotPtr);

    nb_sci         = pl_sci[bidx];
    sci            = nb_sci.sci;
    cijPackedBegin = nb_sci.cjPackedBegin;
    cijPackedEnd   = nb_sci.cjPackedEnd;

    if (tidxz == 0 && (c_loadUsingAllXYThreads || tidxj < c_superClusterSize))
    {
        ci = sci * c_superClusterSize + tidxj;
        ai = ci * c_clusterSize + tidxi;

        const float* shiftptr = reinterpret_cast<const float*>(&shift_vec[nb_sci.shift]);
        xqbuf = xq[ai] + make_float4(LDG(shiftptr), LDG(shiftptr + 1), LDG(shiftptr + 2), 0.0F);
        xqbuf.w *= nbparam.epsfac;
        xqib[tidxj * c_clusterSize + tidxi] = xqbuf;
        atib[tidxj * c_clusterSize + tidxi] = atom_types[ai];
    }

    __syncthreads();

#pragma unroll
    for (i = 0; i < c_superClusterSize; i++)
    {
        fci_buf[i] = make_float3(0.0F, 0.0F, 0.0F);
    }

    const int nonSelfInteraction = !(nb_sci.shift == c_centralShiftIndex & tidxj <= tidxi);

    for (jPacked = cijPackedBegin + tidxz; jPacked < cijPackedEnd; jPacked += c_numThreadsZFrozen)
    {
        wexcl_idx = pl_cjPacked[jPacked].imei[widx].excl_ind;
        imask     = pl_cjPacked[jPacked].imei[widx].imask;
        wexcl     = excl[wexcl_idx].pair[(tidx) & (warp_size - 1)];

        if (imask)
        {
            if (c_preloadCj)
            {
                if ((tidxj == 0 || tidxj == 4) && (tidxi < c_jGroupSize))
                {
                    cjs[tidxi + tidxj * c_jGroupSize / c_splitJClusterSize] = pl_cjPacked[jPacked].cj[tidxi];
                }
                __syncwarp(c_fullWarpMask);
            }

#pragma unroll c_jmLoopUnrollFactorFrozen
            for (jm = 0; jm < c_jGroupSize; jm++)
            {
                if (imask & (superClInteractionMask << (jm * c_superClusterSize)))
                {
                    mask_ji = (1U << (jm * c_superClusterSize));
                    cj = c_preloadCj ? cjs[jm + (tidxj & 4) * c_jGroupSize / c_splitJClusterSize]
                                     : pl_cjPacked[jPacked].cj[jm];
                    aj = cj * c_clusterSize + tidxj;

                    xqbuf = xq[aj];
                    xj    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);
                    qj_f  = xqbuf.w;
                    typej = atom_types[aj];
                    fcj_buf = make_float3(0.0F, 0.0F, 0.0F);

#pragma unroll
                    for (i = 0; i < c_superClusterSize; i++)
                    {
                        if (imask & mask_ji)
                        {
                            ci    = sci * c_superClusterSize + i;
                            xqbuf = xqib[i * c_clusterSize + tidxi];
                            xi    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);

                            rv      = xi - xj;
                            r2      = gmxDeviceNorm2(rv);
                            int_bit = (wexcl & mask_ji) ? 1.0F : 0.0F;

                            if ((r2 < rcoulomb_sq) * (nonSelfInteraction || (ci != cj)))
                            {
                                qi = xqbuf.w;
                                typei = atib[i * c_clusterSize + tidxi];
                                fetch_nbfp_c6_c12(c6, c12, nbparam, ntypes * typei + typej);

                                r2 = fmaxf(r2, c_nbnxnMinDistanceSquared);
                                inv_r = rsqrtf(r2);
                                inv_r2 = inv_r * inv_r;
                                inv_r6 = inv_r2 * inv_r2 * inv_r2;
                                inv_r6 *= int_bit;

                                F_invr = inv_r6 * (c12 * inv_r6 - c6) * inv_r2;
                                calculate_lj_ewald_comb_geom_F(
                                        nbparam, typei, typej, r2, inv_r2, lje_coeff2, lje_coeff6_6, &F_invr);
                                F_invr += qi * qj_f * (int_bit * inv_r2 * inv_r + pmeCorrF(beta2 * r2) * beta3);

                                f_ij = rv * F_invr;
                                fcj_buf -= f_ij;
                                fci_buf[i] += f_ij;
                            }
                        }

                        mask_ji += mask_ji;
                    }

                    reduce_force_j_warp_shfl(fcj_buf, f, tidxi, aj, c_fullWarpMask);
                }
            }

            if (c_preloadCj)
            {
                __syncwarp(c_fullWarpMask);
            }
        }
    }

    if (nb_sci.shift == c_centralShiftIndex)
    {
        bCalcFshift = false;
    }

    float fshift_buf = 0.0F;

    for (i = 0; i < c_superClusterSize; i++)
    {
        ai = (sci * c_superClusterSize + i) * c_clusterSize + tidxi;
        reduce_force_i_warp_shfl(fci_buf[i], f, &fshift_buf, bCalcFshift, tidxj, ai, c_fullWarpMask);
    }

    if (bCalcFshift && (tidxj & 3) < 3)
    {
        float3* fShift = asFloat3(atdat.fShift);
        atomicAdd(&(fShift[nb_sci.shift].x) + (tidxj & 3), fshift_buf);
    }
}

int calcNbKernelNblockFrozen(const int nworkUnits, const FrozenKernelContext& context)
{
    assert(nworkUnits > 0);
    const int maxGridXSize = 2147483647;
    if (nworkUnits > maxGridXSize)
    {
        return maxGridXSize;
    }
    (void)context;
    return nworkUnits;
}

int calcShmemRequiredFrozen(const int numThreadsZ, const NBParamGpu* nbp)
{
    int shmem = c_superClusterSize * c_clusterSize * sizeof(float4);
    shmem += numThreadsZ * c_clusterSplitSize * c_jGroupSize * sizeof(int);

    if (nbp->vdwType == VdwType::CutCombGeom || nbp->vdwType == VdwType::CutCombLB)
    {
        shmem += c_superClusterSize * c_clusterSize * sizeof(float2);
    }
    else
    {
        shmem += c_superClusterSize * c_clusterSize * sizeof(int);
    }

    shmem += sizeof(int);
    return shmem;
}

} // namespace

void launchLJEwaldFrozenKernel(const FrozenKernelContext& context, const bool calcFshift)
{
    static bool cacheConfigSet = false;
    if (!cacheConfigSet)
    {
        cudaFuncSetCacheConfig(nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_frozen_cuda, cudaFuncCachePreferEqual);
        cacheConfigSet = true;
    }

    if (context.plist.numSci == 0)
    {
        return;
    }

    int numThreadsZ = 1;
    if (context.deviceSmMajor == 3 && context.deviceSmMinor == 7)
    {
        numThreadsZ = 2;
    }

    KernelLaunchConfig config;
    config.blockSize[0] = c_clusterSize;
    config.blockSize[1] = c_clusterSize;
    config.blockSize[2] = numThreadsZ;
    config.gridSize[0]  = calcNbKernelNblockFrozen(context.plist.numSci, context);
    config.sharedMemorySize = calcShmemRequiredFrozen(numThreadsZ, &context.nbparam);

    nbnxn_kernel_ElecEw_VdwLJEwCombGeom_F_frozen_cuda
            <<<dim3(config.gridSize[0], config.gridSize[1], config.gridSize[2]),
               dim3(config.blockSize[0], config.blockSize[1], config.blockSize[2]),
               config.sharedMemorySize,
               context.stream>>>(context.atdat, context.nbparam, context.plist, calcFshift);
}

} // namespace sponge::nbnxm::frozen

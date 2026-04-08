#pragma once

#include "../nbnxm_pairlist_layout.h"
#include "../nbnxm_pairlist_types.h"

#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

namespace sponge::nbnxm::frozen
{

#ifndef GMX_PTX_ARCH
#    ifndef __CUDA_ARCH__
#        define GMX_PTX_ARCH 0
#    else
#        define GMX_PTX_ARCH __CUDA_ARCH__
#    endif
#endif

static constexpr int warp_size = 32;
static constexpr unsigned int c_fullWarpMask = 0xFFFFFFFFU;
static constexpr float c_oneSixth = 0.16666667F;
static constexpr float c_nbnxnMinDistanceSquared = 3.82e-07F;
static constexpr int c_centralShiftIndex = 22;
static constexpr int c_clusterSizeSq = c_clusterSize * c_clusterSize;
static constexpr int c_clusterSplitSize = c_clusterPairSplit;

// Force-only path with geometric LJ-Ewald.
enum class VdwType : int
{
    Cut = 0,
    CutCombGeom = 1,
    CutCombLB = 2,
    FSwitch = 3,
    PSwitch = 4,
    EwaldGeom = 5,
    EwaldLB = 6
};

struct NBAtomDataGpu
{
    int numTypes = 0;
    const float4* xq = nullptr;
    const int* atomTypes = nullptr;
    float3* f = nullptr;
    const float3* shiftVec = nullptr;
    float3* fShift = nullptr;
};

struct NBParamGpu
{
    VdwType vdwType = VdwType::EwaldGeom;
    float epsfac = 0.0F;
    float ewald_beta = 0.0F;
    float ewaldcoeff_lj = 0.0F;
    float rcoulomb_sq = 0.0F;
    float sh_lj_ewald = 0.0F;

    const float2* nbfp = nullptr;
    const float2* nbfp_comb = nullptr;
};

struct GpuPairlistSorting
{
    const nbnxn_sci_t* sciSorted = nullptr;
};

struct GpuPairlist
{
    int numSci = 0;
    GpuPairlistSorting sorting;
    const nbnxn_cj_packed_t* cjPacked = nullptr;
    const nbnxn_excl_t* excl = nullptr;
};

struct KernelLaunchConfig
{
    unsigned int blockSize[3] = {1, 1, 1};
    unsigned int gridSize[3] = {1, 1, 1};
    unsigned int sharedMemorySize = 0;
};

struct FrozenKernelContext
{
    NBAtomDataGpu atdat;
    NBParamGpu nbparam;
    GpuPairlist plist;
    cudaStream_t stream = nullptr;
    int deviceSmMajor = 0;
    int deviceSmMinor = 0;
};

static __host__ __device__ __forceinline__ float3 operator+(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __host__ __device__ __forceinline__ float3 operator-(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __host__ __device__ __forceinline__ float3 operator*(const float3 a, const float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __host__ __device__ __forceinline__ float3& operator+=(float3& a, const float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

static __host__ __device__ __forceinline__ float3& operator-=(float3& a, const float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

static __host__ __device__ __forceinline__ float4 operator+(const float4 a, const float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static __device__ __forceinline__ float3* asFloat3(float3* ptr) { return ptr; }
static __device__ __forceinline__ const float3* asFloat3(const float3* ptr) { return ptr; }

static __device__ __forceinline__ float gmxDeviceNorm2(const float3& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

static __device__ __forceinline__ float LDG(const float* ptr)
{
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

} // namespace sponge::nbnxm::frozen

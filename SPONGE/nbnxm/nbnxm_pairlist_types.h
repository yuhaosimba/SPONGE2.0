#pragma once

#include "nbnxm_pairlist_layout.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sponge::nbnxm
{

struct Float2
{
    float x;
    float y;

    bool operator==(const Float2& other) const
    {
        return x == other.x && y == other.y;
    }
};

struct Float3
{
    float x;
    float y;
    float z;

    bool operator==(const Float3& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct Float4
{
    float x;
    float y;
    float z;
    float w;

    bool operator==(const Float4& other) const
    {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }
};

struct nbnxn_sci_t
{
    int sci;
    int shift;
    int cjPackedBegin;
    int cjPackedEnd;

    bool operator==(const nbnxn_sci_t& other) const
    {
        return sci == other.sci && shift == other.shift && cjPackedBegin == other.cjPackedBegin
               && cjPackedEnd == other.cjPackedEnd;
    }
};

struct nbnxn_im_ei_t
{
    std::uint32_t imask = 0U;
    int excl_ind = 0;

    bool operator==(const nbnxn_im_ei_t& other) const
    {
        return imask == other.imask && excl_ind == other.excl_ind;
    }
};

struct nbnxn_cj_packed_t
{
    int cj[c_jGroupSize];
    nbnxn_im_ei_t imei[c_clusterPairSplit];

    bool operator==(const nbnxn_cj_packed_t& other) const
    {
        return std::equal(std::begin(cj), std::end(cj), std::begin(other.cj), std::end(other.cj))
               && std::equal(std::begin(imei), std::end(imei), std::begin(other.imei), std::end(other.imei));
    }
};

struct nbnxn_excl_t
{
    std::uint32_t pair[c_exclSize];

    bool operator==(const nbnxn_excl_t& other) const
    {
        return std::equal(std::begin(pair), std::end(pair), std::begin(other.pair), std::end(other.pair));
    }
};

static_assert(sizeof(nbnxn_sci_t) == 16, "nbnxn_sci_t must match GROMACS layout");
static_assert(sizeof(nbnxn_im_ei_t) == 8, "nbnxn_im_ei_t must match GROMACS layout");
static_assert(sizeof(nbnxn_cj_packed_t) == 32, "nbnxn_cj_packed_t must match GROMACS layout");
static_assert(sizeof(nbnxn_excl_t) == 128, "nbnxn_excl_t must match GROMACS layout");

static_assert(offsetof(nbnxn_sci_t, sci) == 0, "sci offset mismatch");
static_assert(offsetof(nbnxn_sci_t, shift) == 4, "shift offset mismatch");
static_assert(offsetof(nbnxn_sci_t, cjPackedBegin) == 8, "cjPackedBegin offset mismatch");
static_assert(offsetof(nbnxn_sci_t, cjPackedEnd) == 12, "cjPackedEnd offset mismatch");

struct NbnxmPairlistHost
{
    int na_ci = c_clusterSize;
    int na_cj = c_clusterSize;
    int na_sc = c_superClusterSize;
    int nci_tot = 0;
    float rlist = 0.0f;

    std::vector<nbnxn_sci_t> sci;
    std::vector<nbnxn_cj_packed_t> cjPacked;
    std::vector<nbnxn_excl_t> excl;
};

struct NbnxmPairlistDevice
{
    int na_ci = c_clusterSize;
    int na_cj = c_clusterSize;
    int na_sc = c_superClusterSize;
    int nci_tot = 0;
    float rlist = 0.0f;

    int numSci = 0;
    int numPackedJClusters = 0;
    int numExcl = 0;

    nbnxn_sci_t* d_sci = nullptr;
    nbnxn_cj_packed_t* d_cjPacked = nullptr;
    nbnxn_excl_t* d_excl = nullptr;
};

void clearDeviceMirror(NbnxmPairlistDevice* device);
void uploadToDeviceMirror(const NbnxmPairlistHost& host, NbnxmPairlistDevice* device);

} // namespace sponge::nbnxm

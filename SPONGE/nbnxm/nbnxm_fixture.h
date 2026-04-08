#pragma once

#include "nbnxm_pairlist_types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace sponge::nbnxm
{

inline constexpr std::uint64_t c_fixtureMagic = 0x4C4A4557414C4431ULL;
inline constexpr std::uint32_t c_fixtureVersion = 2U;
inline constexpr std::uint32_t c_fixtureEndian = 0x01020304U;

struct LJEwaldForceSummary
{
    double sumX = 0.0;
    double sumY = 0.0;
    double sumZ = 0.0;
    double l1Norm = 0.0;
    double l2Norm = 0.0;
    double maxAbs = 0.0;
};

struct FixtureHeader
{
    std::uint64_t magic = c_fixtureMagic;
    std::uint32_t version = c_fixtureVersion;
    std::uint32_t endian = c_fixtureEndian;

    std::uint32_t layoutType = static_cast<std::uint32_t>(c_layoutType);
    std::uint32_t kernelElecType = 4U;
    std::uint32_t kernelVdwType = 5U;
    std::uint32_t forceOnly = 1U;
    std::uint32_t usePruning = 0U;
    std::uint32_t useTwinCut = 0U;
    std::uint32_t reserved0 = 0U;
    std::uint32_t reserved1 = 0U;

    std::uint32_t numWaters = 0U;
    std::uint32_t numAtoms = 0U;
    std::uint32_t numTypes = 0U;
    std::uint32_t numSci = 0U;
    std::uint32_t numPackedJClusters = 0U;
    std::uint32_t numExcl = 0U;
    std::uint32_t numShiftVec = 0U;
    std::uint32_t numAtomsPerCluster = 0U;

    std::uint64_t effectiveClusterPairs = 0U;
    std::uint64_t effectiveAtomPairs = 0U;
    std::uint64_t forceChecksum = 0U;
    LJEwaldForceSummary referenceForceSummary;

    float epsfac = 0.0F;
    float ewaldBeta = 0.0F;
    float ewaldCoeffLJ = 0.0F;
    float rcoulombSq = 0.0F;
    float rvdwSq = 0.0F;
    float shEwald = 0.0F;
    float shLjEwald = 0.0F;
    float boxXX = 0.0F;
    float boxYY = 0.0F;
    float boxZZ = 0.0F;

    char fixtureName[64] = {};
    char waterModel[16] = {};
    char variantName[64] = {};
};

struct FixtureData
{
    FixtureHeader header;
    std::vector<Float4> xq;
    std::vector<int> atomTypes;
    std::vector<Float3> shiftVec;
    std::vector<Float2> nbfp;
    std::vector<Float2> nbfpComb;

    NbnxmPairlistHost pairlist;
};

FixtureData loadFixture(const std::filesystem::path& path);
void saveFixture(const std::filesystem::path& path, const FixtureData& fixture);

} // namespace sponge::nbnxm

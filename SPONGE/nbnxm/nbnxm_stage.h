#pragma once

#include "nbnxm_pairlist_types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace sponge::nbnxm
{

inline constexpr std::uint64_t c_stageMagic = 0x4E424E584D535447ULL;
inline constexpr std::uint32_t c_stageVersion = 2U;
inline constexpr std::uint32_t c_stageEndian = 0x01020304U;

enum class StageKind : std::uint32_t
{
    Params = 1U,
    Grid = 2U,
    Pairlist = 3U
};

struct StageFileHeader
{
    std::uint64_t magic = c_stageMagic;
    std::uint32_t version = c_stageVersion;
    std::uint32_t endian = c_stageEndian;
    StageKind kind = StageKind::Params;
    std::uint32_t reserved = 0U;
};

static_assert(sizeof(StageFileHeader) == 24, "StageFileHeader layout mismatch");
static_assert(offsetof(StageFileHeader, magic) == 0, "StageFileHeader::magic offset mismatch");
static_assert(offsetof(StageFileHeader, version) == 8, "StageFileHeader::version offset mismatch");
static_assert(offsetof(StageFileHeader, endian) == 12, "StageFileHeader::endian offset mismatch");
static_assert(offsetof(StageFileHeader, kind) == 16, "StageFileHeader::kind offset mismatch");
static_assert(offsetof(StageFileHeader, reserved) == 20, "StageFileHeader::reserved offset mismatch");

struct StageParamsHeader
{
    std::uint32_t layoutType = static_cast<std::uint32_t>(c_layoutType);
    std::uint32_t numAtomsPerCluster = c_clusterSize;
    std::uint32_t numClustersPerSuperCluster = c_superClusterSize;
    std::uint32_t usePruning = 0U;
    std::uint32_t useTwinCut = 0U;
    std::uint32_t hasFep = 0U;
    float cutoff = 0.0F;
    float rlist = 0.0F;
    char label[64] = {};
};

static_assert(sizeof(StageParamsHeader) == 96, "StageParamsHeader layout mismatch");
static_assert(offsetof(StageParamsHeader, cutoff) == 24, "StageParamsHeader::cutoff offset mismatch");
static_assert(offsetof(StageParamsHeader, rlist) == 28, "StageParamsHeader::rlist offset mismatch");
static_assert(offsetof(StageParamsHeader, label) == 32, "StageParamsHeader::label offset mismatch");

struct StageParamsDump
{
    StageParamsHeader header;
};

struct StageGridHeader
{
    float box[3] = {};
    std::uint32_t numCells = 0U;
    std::uint32_t numBins = 0U;
    std::uint32_t numAtomsPerBin = 0U;
    std::uint32_t numAtoms = 0U;
    std::uint32_t numShiftVec = 0U;
    char label[64] = {};
};

static_assert(sizeof(StageGridHeader) == 96, "StageGridHeader layout mismatch");
static_assert(offsetof(StageGridHeader, box) == 0, "StageGridHeader::box offset mismatch");
static_assert(offsetof(StageGridHeader, numCells) == 12, "StageGridHeader::numCells offset mismatch");
static_assert(offsetof(StageGridHeader, numShiftVec) == 28, "StageGridHeader::numShiftVec offset mismatch");
static_assert(offsetof(StageGridHeader, label) == 32, "StageGridHeader::label offset mismatch");

struct StageGridDump
{
    StageGridHeader header;
    std::vector<Float3> shiftVec;
    std::vector<int> atomIndices;
    std::vector<int> bins;
    std::vector<int> numAtomsPerCell;
    std::vector<int> cellToBin;
    std::vector<int> numClustersPerBin;
    std::vector<float> packedBoundingBoxes;
};

struct StagePairlistHeader
{
    float rlist = 0.0F;
    int nciTot = 0;
    std::uint32_t numSci = 0U;
    std::uint32_t numPackedJClusters = 0U;
    std::uint32_t numExcl = 0U;
    char label[64] = {};
};

static_assert(sizeof(StagePairlistHeader) == 84, "StagePairlistHeader layout mismatch");
static_assert(offsetof(StagePairlistHeader, rlist) == 0, "StagePairlistHeader::rlist offset mismatch");
static_assert(offsetof(StagePairlistHeader, nciTot) == 4, "StagePairlistHeader::nciTot offset mismatch");
static_assert(offsetof(StagePairlistHeader, label) == 20, "StagePairlistHeader::label offset mismatch");

struct StagePairlistDump
{
    StagePairlistHeader header;
    NbnxmPairlistHost pairlist;
};

void saveStageParams(const std::filesystem::path& path, const StageParamsDump& dump);
StageParamsDump loadStageParams(const std::filesystem::path& path);

void saveStageGrid(const std::filesystem::path& path, const StageGridDump& dump);
StageGridDump loadStageGrid(const std::filesystem::path& path);

void saveStagePairlist(const std::filesystem::path& path, const StagePairlistDump& dump);
StagePairlistDump loadStagePairlist(const std::filesystem::path& path);

} // namespace sponge::nbnxm

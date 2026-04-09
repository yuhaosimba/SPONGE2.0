#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace sponge::nbnxm
{

inline constexpr std::uint64_t c_exclusionsMagic = 0x4E424E584D455843ULL;
inline constexpr std::uint32_t c_exclusionsVersion = 1U;
inline constexpr std::uint32_t c_exclusionsEndian = 0x01020304U;

struct ExclusionsFileHeader
{
    std::uint64_t magic = c_exclusionsMagic;
    std::uint32_t version = c_exclusionsVersion;
    std::uint32_t endian = c_exclusionsEndian;
    std::uint32_t numAtoms = 0U;
    std::uint32_t numElements = 0U;
    char label[64] = {};
};

struct ExclusionsDump
{
    ExclusionsFileHeader header;
    std::vector<int> exclusionListStart;
    std::vector<int> exclusionList;
    std::vector<int> exclusionCounts;
};

void saveExclusions(const std::filesystem::path& path, const ExclusionsDump& dump);
ExclusionsDump loadExclusions(const std::filesystem::path& path);

} // namespace sponge::nbnxm

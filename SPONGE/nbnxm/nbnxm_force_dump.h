#pragma once

#include "nbnxm_fixture.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace sponge::nbnxm
{

inline constexpr std::uint64_t c_forceDumpMagic = 0x4E424E584D465243ULL;
inline constexpr std::uint32_t c_forceDumpVersion = 1U;
inline constexpr std::uint32_t c_forceDumpEndian = 0x01020304U;

struct ForceDumpHeader
{
    std::uint64_t magic = c_forceDumpMagic;
    std::uint32_t version = c_forceDumpVersion;
    std::uint32_t endian = c_forceDumpEndian;
    std::uint32_t numAtoms = 0U;
    char label[64] = {};
};

struct ForceDump
{
    ForceDumpHeader header;
    std::vector<Float3> forces;
};

void saveForces(const std::filesystem::path& path, const ForceDump& dump);
ForceDump loadForces(const std::filesystem::path& path);

} // namespace sponge::nbnxm

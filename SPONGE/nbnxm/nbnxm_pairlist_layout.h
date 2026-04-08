#pragma once

#include <cstdint>

namespace sponge::nbnxm
{

enum class PairlistType : std::uint32_t
{
    Hierarchical8x8x8 = 3U
};

static constexpr PairlistType c_layoutType = PairlistType::Hierarchical8x8x8;

static constexpr int c_clusterSize       = 8;
static constexpr int c_clusterPairSplit  = 2;
static constexpr int c_superClusterSize  = 8;
static constexpr int c_jGroupSize        = 4;
static constexpr int c_splitJClusterSize = c_clusterSize / c_clusterPairSplit;
static constexpr int c_exclSize          = c_clusterSize * c_splitJClusterSize;
static constexpr std::uint32_t c_fullInteractionMask = 0xFFFFFFFFU;

static_assert(c_clusterSize * c_splitJClusterSize == 32,
              "GPU warp execution width must be 32 for this layout");

} // namespace sponge::nbnxm

#pragma once

#include "nbnxm_pairlist_types.h"
#include "nbnxm_stage.h"

#include <array>
#include <string>
#include <vector>

namespace sponge::nbnxm
{

struct OrthorhombicNoPruneInput
{
    std::vector<Float4> xq;
    std::vector<int> atomTypes;
    std::array<float, 3> box = { 0.0F, 0.0F, 0.0F };
    float cutoff = 0.0F;
    float rlist = 0.0F;

    // CSR-like exclusion storage aligned with the legacy SPONGE neighbor-list inputs.
    std::vector<int> exclusionListStart;
    std::vector<int> exclusionList;
    std::vector<int> exclusionCounts;
};

void validateOrthorhombicNoPruneInput(const OrthorhombicNoPruneInput& input);

StageParamsDump buildPairlistParamsOrthorhombicNoPrune(const OrthorhombicNoPruneInput& input,
                                                       const std::string&               label);

StageGridDump buildGridOrthorhombicNoPrune(const OrthorhombicNoPruneInput& input,
                                           const std::string&               label);

NbnxmPairlistHost buildPairlistOrthorhombicNoPrune(const OrthorhombicNoPruneInput& input);

} // namespace sponge::nbnxm

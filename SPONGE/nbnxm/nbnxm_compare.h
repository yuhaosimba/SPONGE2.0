#pragma once

#include "nbnxm_stage.h"

#include <string>

namespace sponge::nbnxm
{

struct CompareResult
{
    bool equal = false;
    std::string message;
};

CompareResult compareParams(const StageParamsDump& lhs, const StageParamsDump& rhs);
CompareResult compareGrid(const StageGridDump& lhs, const StageGridDump& rhs);
CompareResult comparePairlistBytes(const StagePairlistDump& lhs, const StagePairlistDump& rhs);

} // namespace sponge::nbnxm

#pragma once

#include "nbnxm_builder.h"
#include "nbnxm_exclusions.h"
#include "nbnxm_fixture.h"
#include "nbnxm_stage.h"

namespace sponge::nbnxm
{

OrthorhombicNoPruneInput makeOrthorhombicNoPruneInput(const FixtureData&     fixture,
                                                      const StageGridDump&   grid,
                                                      const ExclusionsDump*  exclusions = nullptr);

} // namespace sponge::nbnxm

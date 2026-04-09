#include "nbnxm_live_builder_input.h"

#include <cmath>
#include <stdexcept>

namespace sponge::nbnxm
{

OrthorhombicNoPruneInput makeOrthorhombicNoPruneInput(const FixtureData&    fixture,
                                                      const StageGridDump&  grid,
                                                      const ExclusionsDump* exclusions)
{
    const std::size_t numRealAtoms = grid.bins.size();
    if (fixture.xq.size() != grid.atomIndices.size() || fixture.atomTypes.size() != grid.atomIndices.size())
    {
        throw std::runtime_error("fixture/grid atom-order size mismatch");
    }

    OrthorhombicNoPruneInput input;
    input.xq.resize(numRealAtoms);
    input.atomTypes.resize(numRealAtoms);
    input.box = { fixture.header.boxXX, fixture.header.boxYY, fixture.header.boxZZ };
    input.cutoff = std::sqrt(fixture.header.rcoulombSq);
    input.rlist = std::sqrt(fixture.header.rcoulombSq);

    for (std::size_t slot = 0; slot < grid.atomIndices.size(); ++slot)
    {
        const int atomIndex = grid.atomIndices[slot];
        if (atomIndex < 0)
        {
            continue;
        }
        if (static_cast<std::size_t>(atomIndex) >= numRealAtoms)
        {
            throw std::runtime_error("grid atom index out of range");
        }

        input.xq[atomIndex] = fixture.xq[slot];
        input.atomTypes[atomIndex] = fixture.atomTypes[slot];
    }

    if (exclusions != nullptr)
    {
        input.exclusionListStart = exclusions->exclusionListStart;
        input.exclusionList = exclusions->exclusionList;
        input.exclusionCounts = exclusions->exclusionCounts;
    }

    return input;
}

} // namespace sponge::nbnxm

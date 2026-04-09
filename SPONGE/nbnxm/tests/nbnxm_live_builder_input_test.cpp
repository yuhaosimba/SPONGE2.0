#include "../nbnxm_compare.h"
#include "../nbnxm_exclusions.h"
#include "../nbnxm_live_builder_input.h"

#include <iostream>
#include <stdexcept>

namespace
{

void expect(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    namespace nbnxm = sponge::nbnxm;

    nbnxm::OrthorhombicNoPruneInput original;
    original.box = { 3.0F, 3.0F, 3.0F };
    original.cutoff = 0.9F;
    original.rlist = 0.9F;
    original.xq = {
        { 0.1F, 0.1F, 0.1F, -0.8F },
        { 0.2F, 0.1F, 0.1F, 0.4F },
        { 0.3F, 0.1F, 0.1F, 0.4F },
    };
    original.atomTypes = { 0, 1, 1 };
    original.exclusionListStart = { 0, 1, 2 };
    original.exclusionList = { 1, 0 };
    original.exclusionCounts = { 1, 1, 0 };

    const auto grid = nbnxm::buildGridOrthorhombicNoPrune(original, "live-input-test");
    const auto referencePairlist = nbnxm::buildPairlistOrthorhombicNoPrune(original);

    nbnxm::FixtureData fixture;
    fixture.header.boxXX = original.box[0];
    fixture.header.boxYY = original.box[1];
    fixture.header.boxZZ = original.box[2];
    fixture.header.rcoulombSq = original.cutoff * original.cutoff;
    fixture.xq.resize(grid.atomIndices.size());
    fixture.atomTypes.resize(grid.atomIndices.size(), 0);

    for (std::size_t slot = 0; slot < grid.atomIndices.size(); ++slot)
    {
        const int atomIndex = grid.atomIndices[slot];
        if (atomIndex >= 0)
        {
            fixture.xq[slot] = original.xq[atomIndex];
            fixture.atomTypes[slot] = original.atomTypes[atomIndex];
        }
    }

    nbnxm::ExclusionsDump exclusions;
    exclusions.header.numAtoms = static_cast<std::uint32_t>(original.xq.size());
    exclusions.header.numElements = static_cast<std::uint32_t>(original.exclusionList.size());
    exclusions.exclusionListStart = original.exclusionListStart;
    exclusions.exclusionList = original.exclusionList;
    exclusions.exclusionCounts = original.exclusionCounts;

    const auto rebuilt = nbnxm::makeOrthorhombicNoPruneInput(fixture, grid, &exclusions);
    expect(rebuilt.box == original.box, "box mismatch");
    expect(rebuilt.cutoff == original.cutoff, "cutoff mismatch");
    expect(rebuilt.rlist == original.rlist, "rlist mismatch");
    expect(rebuilt.xq == original.xq, "xq mismatch");
    expect(rebuilt.atomTypes == original.atomTypes, "atom type mismatch");
    expect(rebuilt.exclusionListStart == original.exclusionListStart, "exclusionListStart mismatch");
    expect(rebuilt.exclusionList == original.exclusionList, "exclusionList mismatch");
    expect(rebuilt.exclusionCounts == original.exclusionCounts, "exclusionCounts mismatch");

    nbnxm::StagePairlistDump lhs;
    lhs.pairlist = referencePairlist;
    lhs.header.rlist = referencePairlist.rlist;
    lhs.header.nciTot = referencePairlist.nci_tot;
    lhs.header.numSci = static_cast<std::uint32_t>(referencePairlist.sci.size());
    lhs.header.numPackedJClusters = static_cast<std::uint32_t>(referencePairlist.cjPacked.size());
    lhs.header.numExcl = static_cast<std::uint32_t>(referencePairlist.excl.size());

    const auto rebuiltPairlist = nbnxm::buildPairlistOrthorhombicNoPrune(rebuilt);
    nbnxm::StagePairlistDump rhs;
    rhs.pairlist = rebuiltPairlist;
    rhs.header.rlist = rebuiltPairlist.rlist;
    rhs.header.nciTot = rebuiltPairlist.nci_tot;
    rhs.header.numSci = static_cast<std::uint32_t>(rebuiltPairlist.sci.size());
    rhs.header.numPackedJClusters = static_cast<std::uint32_t>(rebuiltPairlist.cjPacked.size());
    rhs.header.numExcl = static_cast<std::uint32_t>(rebuiltPairlist.excl.size());

    const auto compare = nbnxm::comparePairlistBytes(lhs, rhs);
    expect(compare.equal, "rebuilt pairlist mismatch");

    std::cout << "nbnxm_live_builder_input_test: PASS\n";
    return 0;
}

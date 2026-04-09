#include "../nbnxm_compare.h"
#include "../nbnxm_stage.h"

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

    nbnxm::StageParamsDump lhsParams;
    lhsParams.header.cutoff = 1.2F;
    lhsParams.header.rlist = 1.2F;
    nbnxm::StageParamsDump rhsParams = lhsParams;
    expect(nbnxm::compareParams(lhsParams, rhsParams).equal, "equal params should compare equal");
    rhsParams.header.rlist = 1.3F;
    expect(!nbnxm::compareParams(lhsParams, rhsParams).equal, "different params should compare different");

    nbnxm::StageGridDump lhsGrid;
    lhsGrid.header.numAtoms = 64U;
    lhsGrid.atomIndices = { 0, 1, 2, 3 };
    lhsGrid.bins = { 0, 1, 2, 3 };
    lhsGrid.numAtomsPerCell = { 32, 32 };
    lhsGrid.cellToBin = { 0, 2, 4 };
    lhsGrid.numClustersPerBin = { 2, 1, 0 };
    lhsGrid.packedBoundingBoxes = { 0.0F, 1.0F, 2.0F, 3.0F };
    nbnxm::StageGridDump rhsGrid = lhsGrid;
    expect(nbnxm::compareGrid(lhsGrid, rhsGrid).equal, "equal grids should compare equal");
    rhsGrid.packedBoundingBoxes[2] = 99.0F;
    expect(!nbnxm::compareGrid(lhsGrid, rhsGrid).equal, "different grids should compare different");

    nbnxm::StagePairlistDump lhsPairlist;
    lhsPairlist.pairlist.sci.push_back({ 0, 0, 0, 1 });
    lhsPairlist.pairlist.cjPacked.push_back({});
    lhsPairlist.pairlist.excl.push_back({});
    nbnxm::StagePairlistDump rhsPairlist = lhsPairlist;
    expect(nbnxm::comparePairlistBytes(lhsPairlist, rhsPairlist).equal,
           "equal pairlists should compare equal");
    rhsPairlist.pairlist.sci[0].cjPackedEnd = 2;
    expect(!nbnxm::comparePairlistBytes(lhsPairlist, rhsPairlist).equal,
           "different pairlists should compare different");

    std::cout << "nbnxm_compare_test: PASS\n";
    return 0;
}

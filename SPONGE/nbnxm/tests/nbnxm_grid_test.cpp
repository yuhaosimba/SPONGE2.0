#include "../nbnxm_builder.h"

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

    nbnxm::OrthorhombicNoPruneInput input;
    input.box = { 3.0F, 4.0F, 5.0F };
    input.cutoff = 0.9F;
    input.rlist = 0.9F;
    input.xq.push_back({ 0.2F, 0.3F, 0.4F, -0.8F });
    input.xq.push_back({ 0.8F, 0.3F, 0.1F, 0.4F });
    input.xq.push_back({ 0.1F, 0.4F, 0.2F, 0.4F });
    input.atomTypes = { 1, 2, 2 };

    const auto grid = nbnxm::buildGridOrthorhombicNoPrune(input, "grid-test");
    expect(grid.header.numCells == 1U, "expected one cell");
    expect(grid.header.numBins == 1U, "expected one bin");
    expect(grid.header.numAtomsPerBin == 64U, "expected 64 atoms per bin");
    expect(grid.header.numAtoms == 64U, "expected one padded bin");
    expect(grid.shiftVec.size() == 45U, "expected 45 shift vectors");
    expect(grid.shiftVec[22] == nbnxm::Float3{ 0.0F, 0.0F, 0.0F }, "central shift mismatch");
    expect(grid.numAtomsPerCell.size() == 2U, "numAtomsPerCell size mismatch");
    expect(grid.cellToBin.size() == 3U, "cellToBin size mismatch");
    expect(grid.atomIndices[0] >= 0 && grid.atomIndices[0] < 3, "first atom index out of range");
    expect(grid.atomIndices[1] >= 0 && grid.atomIndices[1] < 3, "second atom index out of range");
    expect(grid.atomIndices[2] >= 0 && grid.atomIndices[2] < 3, "third atom index out of range");
    expect(grid.atomIndices[3] == -1, "padding should start after the third atom");
    expect(grid.bins[grid.atomIndices[0]] == 0, "inverse bin mapping mismatch at 0");
    expect(grid.bins[grid.atomIndices[1]] == 1, "inverse bin mapping mismatch at 1");
    expect(grid.bins[grid.atomIndices[2]] == 2, "inverse bin mapping mismatch at 2");

    std::cout << "nbnxm_grid_test: PASS\n";
    return 0;
}

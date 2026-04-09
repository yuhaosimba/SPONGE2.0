#include "../nbnxm_stage.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

    const std::filesystem::path tmpDir = std::filesystem::temp_directory_path() / "spg_nbnxm_stage_io_test";
    std::filesystem::create_directories(tmpDir);

    nbnxm::StageParamsDump params;
    params.header.layoutType = static_cast<std::uint32_t>(nbnxm::c_layoutType);
    params.header.numAtomsPerCluster = nbnxm::c_clusterSize;
    params.header.numClustersPerSuperCluster = nbnxm::c_superClusterSize;
    params.header.cutoff = 1.25F;
    params.header.rlist = 1.25F;
    params.header.usePruning = 0U;
    params.header.useTwinCut = 0U;
    params.header.hasFep = 0U;
    std::snprintf(params.header.label, sizeof(params.header.label), "%s", "params-roundtrip");

    nbnxm::StageGridDump grid;
    grid.header.box[0] = 3.0F;
    grid.header.box[1] = 4.0F;
    grid.header.box[2] = 5.0F;
    grid.header.numCells = 2U;
    grid.header.numBins = 4U;
    grid.header.numAtomsPerBin = 64U;
    grid.header.numAtoms = 128U;
    grid.header.numShiftVec = 2U;
    std::snprintf(grid.header.label, sizeof(grid.header.label), "%s", "grid-roundtrip");
    grid.shiftVec = { { 0.0F, 0.0F, 0.0F }, { 3.0F, 0.0F, 0.0F } };
    grid.atomIndices = { 0, 1, 2, 3, 4, 5 };
    grid.bins = { 0, 1, 2, 3, 4, 5 };
    grid.numAtomsPerCell = { 64, 64 };
    grid.cellToBin = { 0, 2, 4 };
    grid.numClustersPerBin = { 2, 1, 0, 3, 0 };
    grid.packedBoundingBoxes = { 0.0F, 0.1F, 0.2F, 0.3F, 1.0F, 1.1F, 1.2F, 1.3F };

    nbnxm::StagePairlistDump pairlist;
    pairlist.header.rlist = 1.25F;
    pairlist.header.nciTot = 7;
    pairlist.header.numSci = 1U;
    pairlist.header.numPackedJClusters = 1U;
    pairlist.header.numExcl = 1U;
    std::snprintf(pairlist.header.label, sizeof(pairlist.header.label), "%s", "pairlist-roundtrip");
    pairlist.pairlist.na_ci = nbnxm::c_clusterSize;
    pairlist.pairlist.na_cj = nbnxm::c_clusterSize;
    pairlist.pairlist.na_sc = nbnxm::c_superClusterSize;
    pairlist.pairlist.nci_tot = 7;
    pairlist.pairlist.rlist = 1.25F;
    pairlist.pairlist.sci.push_back({ 0, 0, 0, 1 });
    nbnxm::nbnxn_cj_packed_t packed{};
    packed.cj[0] = 0;
    packed.cj[1] = 1;
    packed.cj[2] = 2;
    packed.cj[3] = 3;
    packed.imei[0].imask = 0xFFFFFFFFU;
    packed.imei[1].imask = 0xFFFFFFFFU;
    packed.imei[0].excl_ind = 0;
    packed.imei[1].excl_ind = 0;
    pairlist.pairlist.cjPacked.push_back(packed);
    nbnxm::nbnxn_excl_t excl{};
    for (auto& value : excl.pair)
    {
        value = nbnxm::c_fullInteractionMask;
    }
    pairlist.pairlist.excl.push_back(excl);

    const auto paramsPath = tmpDir / "params.bin";
    const auto gridPath = tmpDir / "grid.bin";
    const auto pairlistPath = tmpDir / "pairlist.bin";

    nbnxm::saveStageParams(paramsPath, params);
    nbnxm::saveStageGrid(gridPath, grid);
    nbnxm::saveStagePairlist(pairlistPath, pairlist);

    const auto loadedParams = nbnxm::loadStageParams(paramsPath);
    const auto loadedGrid = nbnxm::loadStageGrid(gridPath);
    const auto loadedPairlist = nbnxm::loadStagePairlist(pairlistPath);

    expect(loadedParams.header.cutoff == params.header.cutoff, "params cutoff mismatch");
    expect(loadedParams.header.rlist == params.header.rlist, "params rlist mismatch");
    expect(loadedGrid.header.numAtoms == grid.header.numAtoms, "grid atom count mismatch");
    expect(loadedGrid.shiftVec == grid.shiftVec, "grid shift vectors mismatch");
    expect(loadedGrid.cellToBin == grid.cellToBin, "grid cellToBin mismatch");
    expect(loadedGrid.numClustersPerBin == grid.numClustersPerBin, "grid numClustersPerBin mismatch");
    expect(loadedGrid.packedBoundingBoxes == grid.packedBoundingBoxes, "grid packedBoundingBoxes mismatch");
    expect(loadedPairlist.pairlist.sci == pairlist.pairlist.sci, "pairlist sci mismatch");
    expect(loadedPairlist.pairlist.cjPacked == pairlist.pairlist.cjPacked, "pairlist cjPacked mismatch");
    expect(loadedPairlist.pairlist.excl == pairlist.pairlist.excl, "pairlist excl mismatch");

    std::filesystem::remove_all(tmpDir);
    std::cout << "nbnxm_stage_io_test: PASS\n";
    return 0;
}

#include "nbnxm_compare.h"

#include <sstream>
#include <cstring>

namespace sponge::nbnxm
{
namespace
{

template <typename T>
CompareResult compareScalar(const T& lhs, const T& rhs, const char* label)
{
    if (lhs == rhs)
    {
        return { true, {} };
    }
    return { false, label };
}

template <typename T>
CompareResult compareVector(const std::vector<T>& lhs, const std::vector<T>& rhs, const char* label)
{
    if (lhs == rhs)
    {
        return { true, {} };
    }
    return { false, label };
}

CompareResult compareFloatVectorDetailed(const std::vector<float>& lhs,
                                         const std::vector<float>& rhs,
                                         const char*               label)
{
    if (lhs == rhs)
    {
        return { true, {} };
    }

    std::ostringstream message;
    message << label;
    if (lhs.size() != rhs.size())
    {
        message << ": size lhs=" << lhs.size() << " rhs=" << rhs.size();
        return { false, message.str() };
    }

    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        if (lhs[i] != rhs[i])
        {
            message << ": first mismatch at " << i << " lhs=" << lhs[i] << " rhs=" << rhs[i];
            return { false, message.str() };
        }
    }

    return { false, label };
}

} // namespace

CompareResult compareParams(const StageParamsDump& lhs, const StageParamsDump& rhs)
{
    const auto& a = lhs.header;
    const auto& b = rhs.header;
    if (a.layoutType == b.layoutType && a.numAtomsPerCluster == b.numAtomsPerCluster
        && a.numClustersPerSuperCluster == b.numClustersPerSuperCluster && a.usePruning == b.usePruning
        && a.useTwinCut == b.useTwinCut && a.hasFep == b.hasFep && a.cutoff == b.cutoff && a.rlist == b.rlist)
    {
        return { true, {} };
    }
    return { false, "StageParamsHeader mismatch" };
}

CompareResult compareGrid(const StageGridDump& lhs, const StageGridDump& rhs)
{
    const auto& a = lhs.header;
    const auto& b = rhs.header;
    if (a.box[0] != b.box[0] || a.box[1] != b.box[1] || a.box[2] != b.box[2] || a.numCells != b.numCells
        || a.numBins != b.numBins || a.numAtomsPerBin != b.numAtomsPerBin || a.numAtoms != b.numAtoms
        || a.numShiftVec != b.numShiftVec)
    {
        return { false, "StageGridHeader mismatch" };
    }
    if (auto result = compareVector(lhs.shiftVec, rhs.shiftVec, "grid shiftVec mismatch"); !result.equal)
    {
        return result;
    }
    if (auto result = compareVector(lhs.atomIndices, rhs.atomIndices, "grid atomIndices mismatch"); !result.equal)
    {
        return result;
    }
    if (auto result = compareVector(lhs.bins, rhs.bins, "grid bins mismatch"); !result.equal)
    {
        return result;
    }
    if (auto result = compareVector(lhs.numAtomsPerCell, rhs.numAtomsPerCell, "grid numAtomsPerCell mismatch");
        !result.equal)
    {
        return result;
    }
    if (auto result = compareVector(lhs.cellToBin, rhs.cellToBin, "grid cellToBin mismatch"); !result.equal)
    {
        return result;
    }
    if (auto result = compareVector(lhs.numClustersPerBin, rhs.numClustersPerBin, "grid numClustersPerBin mismatch");
        !result.equal)
    {
        return result;
    }
    if (auto result = compareFloatVectorDetailed(
                lhs.packedBoundingBoxes, rhs.packedBoundingBoxes, "grid packedBoundingBoxes mismatch");
        !result.equal)
    {
        return result;
    }
    return { true, {} };
}

CompareResult comparePairlistBytes(const StagePairlistDump& lhs, const StagePairlistDump& rhs)
{
    const auto& a = lhs.header;
    const auto& b = rhs.header;
    if (a.rlist != b.rlist)
    {
        return { false, "StagePairlistHeader rlist mismatch" };
    }
    if (a.nciTot != b.nciTot)
    {
        std::ostringstream message;
        message << "StagePairlistHeader nciTot mismatch: lhs=" << a.nciTot << " rhs=" << b.nciTot;
        return { false, message.str() };
    }
    if (a.numSci != b.numSci)
    {
        std::ostringstream message;
        message << "StagePairlistHeader numSci mismatch: lhs=" << a.numSci << " rhs=" << b.numSci;
        return { false, message.str() };
    }
    if (a.numPackedJClusters != b.numPackedJClusters)
    {
        std::ostringstream message;
        message << "StagePairlistHeader numPackedJClusters mismatch: lhs=" << a.numPackedJClusters
                << " rhs=" << b.numPackedJClusters;
        return { false, message.str() };
    }
    if (a.numExcl != b.numExcl)
    {
        std::ostringstream message;
        message << "StagePairlistHeader numExcl mismatch: lhs=" << a.numExcl << " rhs=" << b.numExcl;
        return { false, message.str() };
    }
    if (auto result = compareScalar(lhs.pairlist.na_ci, rhs.pairlist.na_ci, "pairlist na_ci mismatch");
        !result.equal)
    {
        return result;
    }
    if (auto result = compareScalar(lhs.pairlist.na_cj, rhs.pairlist.na_cj, "pairlist na_cj mismatch");
        !result.equal)
    {
        return result;
    }
    if (auto result = compareScalar(lhs.pairlist.na_sc, rhs.pairlist.na_sc, "pairlist na_sc mismatch");
        !result.equal)
    {
        return result;
    }
    if (auto result = compareScalar(lhs.pairlist.nci_tot, rhs.pairlist.nci_tot, "pairlist nci_tot mismatch");
        !result.equal)
    {
        return result;
    }
    if (auto result = compareScalar(lhs.pairlist.rlist, rhs.pairlist.rlist, "pairlist rlist mismatch");
        !result.equal)
    {
        return result;
    }
    if (auto result = compareVector(lhs.pairlist.sci, rhs.pairlist.sci, "pairlist sci mismatch"); !result.equal)
    {
        return result;
    }
    if (auto result = compareVector(lhs.pairlist.cjPacked, rhs.pairlist.cjPacked, "pairlist cjPacked mismatch");
        !result.equal)
    {
        return result;
    }
    if (auto result = compareVector(lhs.pairlist.excl, rhs.pairlist.excl, "pairlist excl mismatch"); !result.equal)
    {
        return result;
    }
    return { true, {} };
}

} // namespace sponge::nbnxm

#include "nbnxm_builder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace sponge::nbnxm
{
namespace
{

constexpr int c_gpuNumClusterPerBinX = 2;
constexpr int c_gpuNumClusterPerBinY = 2;
constexpr int c_gpuNumClusterPerBinZ = 2;
constexpr int c_sortGridRatio = 4;
constexpr int c_centralShiftIndex = 22;
constexpr int c_packedBoundingBoxesDimSize = 4;
constexpr int c_packedBoundingBoxesSize = c_packedBoundingBoxesDimSize * 3 * 2;
constexpr float c_farAwayCoordinate = -1000000.0F;

struct ClusterBounds
{
    Float3 lower = { 0.0F, 0.0F, 0.0F };
    Float3 upper = { 0.0F, 0.0F, 0.0F };
};

void validateFinitePositive(float value, const char* label)
{
    if (!std::isfinite(value) || value <= 0.0F)
    {
        throw std::runtime_error(std::string(label) + " must be finite and positive");
    }
}

int divideRoundUp(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

float coordinateComponent(const Float4& value, int dim)
{
    switch (dim)
    {
        case 0: return value.x;
        case 1: return value.y;
        default: return value.z;
    }
}

std::vector<Float3> makeOrthorhombicShiftVectors(const std::array<float, 3>& box)
{
    std::vector<Float3> shiftVec;
    shiftVec.reserve(45);
    for (int iz = -1; iz <= 1; ++iz)
    {
        for (int iy = -1; iy <= 1; ++iy)
        {
            for (int ix = -2; ix <= 2; ++ix)
            {
                shiftVec.push_back(Float3{ ix * box[0], iy * box[1], iz * box[2] });
            }
        }
    }
    return shiftVec;
}

nbnxn_excl_t makeFullExclusionMask()
{
    nbnxn_excl_t excl = {};
    for (auto& pair : excl.pair)
    {
        pair = c_fullInteractionMask;
    }
    return excl;
}

float clusterBoundingBoxDistance2(const ClusterBounds& ibb, const ClusterBounds& jbb, const Float3& shift)
{
    float d2 = 0.0F;

    const float iLower[3] = { ibb.lower.x + shift.x, ibb.lower.y + shift.y, ibb.lower.z + shift.z };
    const float iUpper[3] = { ibb.upper.x + shift.x, ibb.upper.y + shift.y, ibb.upper.z + shift.z };
    const float jLower[3] = { jbb.lower.x, jbb.lower.y, jbb.lower.z };
    const float jUpper[3] = { jbb.upper.x, jbb.upper.y, jbb.upper.z };

    for (int dim = 0; dim < 3; ++dim)
    {
        if (jLower[dim] > iUpper[dim])
        {
            const float d = jLower[dim] - iUpper[dim];
            d2 += d * d;
        }
        else if (jUpper[dim] < iLower[dim])
        {
            const float d = iLower[dim] - jUpper[dim];
            d2 += d * d;
        }
    }

    return d2;
}

float atomDistance2(const Float4& a, const Float4& b, const Float3& shift)
{
    const float dx = (a.x + shift.x) - b.x;
    const float dy = (a.y + shift.y) - b.y;
    const float dz = (a.z + shift.z) - b.z;
    return dx * dx + dy * dy + dz * dz;
}

std::vector<int> computeNumClustersPerBin(const StageGridDump& grid)
{
    std::vector<int> numClustersPerBin(static_cast<std::size_t>(grid.header.numBins), 0);
    for (std::size_t bin = 0; bin < numClustersPerBin.size(); ++bin)
    {
        int realAtoms = 0;
        const std::size_t atomOffset = bin * c_superClusterSize * c_clusterSize;
        for (int atom = 0; atom < c_superClusterSize * c_clusterSize; ++atom)
        {
            if (grid.atomIndices[atomOffset + atom] >= 0)
            {
                ++realAtoms;
            }
        }
        numClustersPerBin[bin] = divideRoundUp(realAtoms, c_clusterSize);
    }
    return numClustersPerBin;
}

std::vector<ClusterBounds> computeClusterBounds(const OrthorhombicNoPruneInput& input, const StageGridDump& grid)
{
    std::vector<ClusterBounds> bounds(static_cast<std::size_t>(grid.header.numBins) * c_superClusterSize);

    for (std::size_t bin = 0; bin < static_cast<std::size_t>(grid.header.numBins); ++bin)
    {
        for (int cluster = 0; cluster < c_superClusterSize; ++cluster)
        {
            const std::size_t clusterOffset =
                    bin * c_superClusterSize * c_clusterSize + static_cast<std::size_t>(cluster) * c_clusterSize;

            int firstAtom = -1;
            for (int atom = 0; atom < c_clusterSize; ++atom)
            {
                const int atomIndex = grid.atomIndices[clusterOffset + atom];
                if (atomIndex >= 0)
                {
                    firstAtom = atomIndex;
                    break;
                }
            }
            if (firstAtom < 0)
            {
                continue;
            }

            ClusterBounds clusterBounds;
            clusterBounds.lower = { input.xq[firstAtom].x, input.xq[firstAtom].y, input.xq[firstAtom].z };
            clusterBounds.upper = clusterBounds.lower;

            for (int atom = 1; atom < c_clusterSize; ++atom)
            {
                const int atomIndex = grid.atomIndices[clusterOffset + atom];
                if (atomIndex < 0)
                {
                    continue;
                }

                const auto& coordinate = input.xq[atomIndex];
                clusterBounds.lower.x = std::min(clusterBounds.lower.x, coordinate.x);
                clusterBounds.lower.y = std::min(clusterBounds.lower.y, coordinate.y);
                clusterBounds.lower.z = std::min(clusterBounds.lower.z, coordinate.z);
                clusterBounds.upper.x = std::max(clusterBounds.upper.x, coordinate.x);
                clusterBounds.upper.y = std::max(clusterBounds.upper.y, coordinate.y);
                clusterBounds.upper.z = std::max(clusterBounds.upper.z, coordinate.z);
            }

            bounds[bin * c_superClusterSize + cluster] = clusterBounds;
        }
    }

    return bounds;
}

std::vector<float> packClusterBounds(const std::vector<ClusterBounds>& clusterBounds,
                                     const std::vector<int>&          numClustersPerBin,
                                     int                              numBins)
{
    const int totalClusters = static_cast<int>(numClustersPerBin.size()) * c_superClusterSize;
    std::vector<float> packed(static_cast<std::size_t>((totalClusters + c_packedBoundingBoxesDimSize - 1)
                                                       / c_packedBoundingBoxesDimSize)
                                      * c_packedBoundingBoxesSize,
                              0.0F);

    for (int bin = 0; bin < numBins; ++bin)
    {
        for (int clusterInBin = 0; clusterInBin < c_superClusterSize; ++clusterInBin)
        {
            const int clusterIndex = bin * c_superClusterSize + clusterInBin;
            const int packedBase = (clusterIndex / c_packedBoundingBoxesDimSize) * c_packedBoundingBoxesSize
                                   + (clusterIndex & (c_packedBoundingBoxesDimSize - 1));

            Float3 lower = { c_farAwayCoordinate, c_farAwayCoordinate, c_farAwayCoordinate };
            Float3 upper = lower;
            if (clusterInBin < numClustersPerBin[bin])
            {
                lower = clusterBounds[clusterIndex].lower;
                upper = clusterBounds[clusterIndex].upper;
            }

            packed[packedBase + 0 * c_packedBoundingBoxesDimSize] = lower.x;
            packed[packedBase + 1 * c_packedBoundingBoxesDimSize] = lower.y;
            packed[packedBase + 2 * c_packedBoundingBoxesDimSize] = lower.z;
            packed[packedBase + 3 * c_packedBoundingBoxesDimSize] = upper.x;
            packed[packedBase + 4 * c_packedBoundingBoxesDimSize] = upper.y;
            packed[packedBase + 5 * c_packedBoundingBoxesDimSize] = upper.z;
        }
    }

    return packed;
}

bool clusterPairInRange(const OrthorhombicNoPruneInput& input,
                        const StageGridDump&            grid,
                        int                             iBin,
                        int                             iCluster,
                        int                             jBin,
                        int                             jCluster,
                        const Float3&                   shift,
                        float                           rlist2)
{
    const std::size_t iClusterOffset = static_cast<std::size_t>(iBin) * c_superClusterSize * c_clusterSize
                                       + static_cast<std::size_t>(iCluster) * c_clusterSize;
    const std::size_t jClusterOffset = static_cast<std::size_t>(jBin) * c_superClusterSize * c_clusterSize
                                       + static_cast<std::size_t>(jCluster) * c_clusterSize;

    for (int i = 0; i < c_clusterSize; ++i)
    {
        const int iAtom = grid.atomIndices[iClusterOffset + i];
        if (iAtom < 0)
        {
            continue;
        }
        for (int j = 0; j < c_clusterSize; ++j)
        {
            const int jAtom = grid.atomIndices[jClusterOffset + j];
            if (jAtom < 0)
            {
                continue;
            }
            if (atomDistance2(input.xq[iAtom], input.xq[jAtom], shift) < rlist2)
            {
                return true;
            }
        }
    }

    return false;
}

float boundingBoxOnlyDistance2(const OrthorhombicNoPruneInput& input)
{
    const float boxVolume = input.box[0] * input.box[1] * input.box[2];
    const float atomDensity = static_cast<float>(input.xq.size()) / boxVolume;
    const float targetSubcellLength = std::cbrt(static_cast<float>(c_clusterSize) / atomDensity);
    const float targetCellLengthX = targetSubcellLength * c_gpuNumClusterPerBinX;
    const float targetCellLengthY = targetSubcellLength * c_gpuNumClusterPerBinY;
    const int numCellsX = std::max(1, static_cast<int>(input.box[0] / targetCellLengthX));
    const int numCellsY = std::max(1, static_cast<int>(input.box[1] / targetCellLengthY));
    const float cellSizeX = input.box[0] / static_cast<float>(numCellsX);
    const float cellSizeY = input.box[1] / static_cast<float>(numCellsY);
    const float bbx = cellSizeX / static_cast<float>(c_gpuNumClusterPerBinX);
    const float bby = cellSizeY / static_cast<float>(c_gpuNumClusterPerBinY);
    const float rbb = std::max(0.0F, input.rlist - 0.5F * std::sqrt(bbx * bbx + bby * bby));
    return rbb * rbb;
}

nbnxn_excl_t& getExclusionMask(NbnxmPairlistHost* pairlist, int cjPackedIndex, int warp)
{
    if (pairlist->cjPacked[cjPackedIndex].imei[warp].excl_ind == 0)
    {
        pairlist->excl.push_back(makeFullExclusionMask());
        pairlist->cjPacked[cjPackedIndex].imei[warp].excl_ind = static_cast<int>(pairlist->excl.size()) - 1;
    }

    return pairlist->excl[pairlist->cjPacked[cjPackedIndex].imei[warp].excl_ind];
}

void setSelfAndNewtonExclusionsGpu(NbnxmPairlistHost* pairlist,
                                   int                cjPackedIndex,
                                   int                jOffsetInGroup,
                                   int                iClusterInBin)
{
    for (int part = 0; part < c_clusterPairSplit; ++part)
    {
        auto& excl = getExclusionMask(pairlist, cjPackedIndex, part);
        const int jOffset = part * c_splitJClusterSize;
        for (int jIndexInPart = 0; jIndexInPart < c_splitJClusterSize; ++jIndexInPart)
        {
            for (int i = jOffset + jIndexInPart; i < c_clusterSize; ++i)
            {
                excl.pair[jIndexInPart * c_clusterSize + i] &=
                        ~(1U << (jOffsetInGroup * c_superClusterSize + iClusterInBin));
            }
        }
    }
}

int findJClusterIndexInCurrentSci(const NbnxmPairlistHost& pairlist, const nbnxn_sci_t& sci, int currentCjCount, int jCluster)
{
    const int start = sci.cjPackedBegin * c_jGroupSize;
    for (int index = start; index < currentCjCount; ++index)
    {
        const auto& packed = pairlist.cjPacked[index / c_jGroupSize];
        if (packed.cj[index & (c_jGroupSize - 1)] == jCluster)
        {
            return index;
        }
    }
    return -1;
}

void applyTopologyExclusionsForSci(const OrthorhombicNoPruneInput& input,
                                   const StageGridDump&            grid,
                                   NbnxmPairlistHost*              pairlist,
                                   int                             currentCjCount,
                                   bool                            diagRemoved)
{
    if (pairlist->sci.empty())
    {
        return;
    }
    const auto& currentSci = pairlist->sci.back();
    if (currentSci.cjPackedBegin == currentSci.cjPackedEnd)
    {
        return;
    }

    const int iSuperCluster = currentSci.sci;
    for (int i = 0; i < c_superClusterSize * c_clusterSize; ++i)
    {
        const int iIndex = iSuperCluster * c_superClusterSize * c_clusterSize + i;
        const int iAtom = grid.atomIndices[iIndex];
        if (iAtom < 0)
        {
            continue;
        }

        const int iCluster = i / c_clusterSize;
        const int exclCount = input.exclusionCounts.empty() ? 0 : input.exclusionCounts[iAtom];
        const int exclStart = (exclCount == 0 || input.exclusionListStart.empty()) ? 0 : input.exclusionListStart[iAtom];
        for (int idx = exclStart; idx < exclStart + exclCount; ++idx)
        {
            const int jAtom = input.exclusionList[idx];
            if (jAtom == iAtom)
            {
                continue;
            }

            const int jIndex = grid.bins[jAtom];
            if (jIndex < 0)
            {
                continue;
            }
            if (diagRemoved && jIndex <= iSuperCluster * c_superClusterSize * c_clusterSize + i)
            {
                continue;
            }

            const int jCluster = jIndex / c_clusterSize;
            const int listIndex = findJClusterIndexInCurrentSci(*pairlist, currentSci, currentCjCount, jCluster);
            if (listIndex < 0)
            {
                continue;
            }

            const std::uint32_t pairMask =
                    (1U << ((listIndex & (c_jGroupSize - 1)) * c_superClusterSize + iCluster));
            auto& packed = pairlist->cjPacked[listIndex / c_jGroupSize];
            if ((packed.imei[0].imask & pairMask) == 0U)
            {
                continue;
            }

            const int innerI = i & (c_clusterSize - 1);
            const int innerJ = jIndex & (c_clusterSize - 1);
            const int jHalf = innerJ / c_splitJClusterSize;
            auto& interactionMask = getExclusionMask(pairlist, listIndex / c_jGroupSize, jHalf);
            interactionMask.pair[(innerJ & (c_splitJClusterSize - 1)) * c_clusterSize + innerI] &= ~pairMask;
        }
    }
}

void sortSciEntries(NbnxmPairlistHost* pairlist)
{
    if (pairlist->cjPacked.size() <= pairlist->sci.size())
    {
        return;
    }

    const int m = static_cast<int>((2 * pairlist->cjPacked.size()) / pairlist->sci.size());
    std::vector<nbnxn_sci_t> sorted(pairlist->sci.size());
    std::vector<int> offsets(static_cast<std::size_t>(m) + 1, 0);

    for (const auto& sci : pairlist->sci)
    {
        const int groups = std::min(m, sci.cjPackedEnd - sci.cjPackedBegin);
        offsets[groups]++;
    }

    int s0 = offsets[m];
    offsets[m] = 0;
    for (int i = m - 1; i >= 0; --i)
    {
        const int s1 = offsets[i];
        offsets[i] = offsets[i + 1] + s0;
        s0 = s1;
    }

    for (const auto& sci : pairlist->sci)
    {
        const int groups = std::min(m, sci.cjPackedEnd - sci.cjPackedBegin);
        sorted[offsets[groups]++] = sci;
    }

    pairlist->sci.swap(sorted);
}

void sortAtomIndicesInPlace(const OrthorhombicNoPruneInput& input,
                            int                             dim,
                            bool                            backwards,
                            int*                            atomIndices,
                            int                             count,
                            float                           lowerBound,
                            float                           inverseRange,
                            int                             expectedParticlesPerHole,
                            std::vector<int>*               sortBuffer)
{
    if (count <= 1)
    {
        return;
    }
    if (count > expectedParticlesPerHole)
    {
        throw std::runtime_error("internal sort buffer contract violated");
    }

    const float holeInverseRange = inverseRange * expectedParticlesPerHole * c_sortGridRatio;
    const int nsort = expectedParticlesPerHole * c_sortGridRatio + count;
    sortBuffer->assign(nsort, -1);

    int ziMin = std::numeric_limits<int>::max();
    int ziMax = -1;

    for (int i = 0; i < count; ++i)
    {
        const int atomIndex = atomIndices[i];
        int zi = static_cast<int>((coordinateComponent(input.xq[atomIndex], dim) - lowerBound) * holeInverseRange);
        if (zi < 0)
        {
            zi = 0;
        }
        const int ziClampMax = expectedParticlesPerHole * c_sortGridRatio;
        if (zi > ziClampMax)
        {
            zi = ziClampMax;
        }

        if ((*sortBuffer)[zi] < 0)
        {
            (*sortBuffer)[zi] = atomIndex;
            ziMin = std::min(ziMin, zi);
            ziMax = std::max(ziMax, zi);
            continue;
        }

        while ((*sortBuffer)[zi] >= 0
               && (coordinateComponent(input.xq[atomIndex], dim)
                               > coordinateComponent(input.xq[(*sortBuffer)[zi]], dim)
                   || (coordinateComponent(input.xq[atomIndex], dim)
                               == coordinateComponent(input.xq[(*sortBuffer)[zi]], dim)
                       && atomIndex > (*sortBuffer)[zi])))
        {
            ++zi;
        }

        if ((*sortBuffer)[zi] >= 0)
        {
            int carry = (*sortBuffer)[zi];
            int zim = zi + 1;
            while ((*sortBuffer)[zim] >= 0)
            {
                const int tmp = (*sortBuffer)[zim];
                (*sortBuffer)[zim] = carry;
                carry = tmp;
                ++zim;
            }
            (*sortBuffer)[zim] = carry;
            ziMax = std::max(ziMax, zim);
        }

        (*sortBuffer)[zi] = atomIndex;
        ziMin = std::min(ziMin, zi);
        ziMax = std::max(ziMax, zi);
    }

    int out = 0;
    if (!backwards)
    {
        for (int zi = 0; zi < nsort; ++zi)
        {
            if ((*sortBuffer)[zi] >= 0)
            {
                atomIndices[out++] = (*sortBuffer)[zi];
            }
        }
    }
    else
    {
        for (int zi = ziMax; zi >= ziMin; --zi)
        {
            if ((*sortBuffer)[zi] >= 0)
            {
                atomIndices[out++] = (*sortBuffer)[zi];
            }
        }
    }
}

} // namespace

void validateOrthorhombicNoPruneInput(const OrthorhombicNoPruneInput& input)
{
    if (input.xq.size() != input.atomTypes.size())
    {
        throw std::runtime_error("xq/atomTypes size mismatch");
    }

    validateFinitePositive(input.cutoff, "cutoff");
    validateFinitePositive(input.rlist, "rlist");

    if (input.rlist < input.cutoff)
    {
        throw std::runtime_error("rlist must be greater than or equal to cutoff");
    }

    validateFinitePositive(input.box[0], "box[0]");
    validateFinitePositive(input.box[1], "box[1]");
    validateFinitePositive(input.box[2], "box[2]");

    const bool hasExclusions = !input.exclusionCounts.empty() || !input.exclusionListStart.empty()
                               || !input.exclusionList.empty();
    if (!hasExclusions)
    {
        return;
    }

    if (input.exclusionCounts.size() != input.xq.size())
    {
        throw std::runtime_error("exclusionCounts size mismatch");
    }
    if (input.exclusionListStart.size() != input.xq.size())
    {
        throw std::runtime_error("exclusionListStart size mismatch");
    }

    for (std::size_t atom = 0; atom < input.xq.size(); ++atom)
    {
        const int count = input.exclusionCounts[atom];
        if (count < 0)
        {
            throw std::runtime_error("negative exclusion count");
        }

        const int start = (count == 0 ? 0 : input.exclusionListStart[atom]);
        if (start < 0 || start > static_cast<int>(input.exclusionList.size()))
        {
            throw std::runtime_error("exclusion start out of range");
        }
        if (start + count > static_cast<int>(input.exclusionList.size()))
        {
            throw std::runtime_error("exclusion span out of range");
        }

        for (int idx = start; idx < start + count; ++idx)
        {
            const int excludedAtom = input.exclusionList[idx];
            if (excludedAtom < 0 || excludedAtom >= static_cast<int>(input.xq.size()))
            {
                throw std::runtime_error("excluded atom index out of range");
            }
            if (idx > start && input.exclusionList[idx - 1] > excludedAtom)
            {
                throw std::runtime_error("exclusion list must be sorted per atom");
            }
        }
    }
}

StageParamsDump buildPairlistParamsOrthorhombicNoPrune(const OrthorhombicNoPruneInput& input,
                                                       const std::string&               label)
{
    validateOrthorhombicNoPruneInput(input);

    StageParamsDump params;
    params.header.layoutType = static_cast<std::uint32_t>(c_layoutType);
    params.header.numAtomsPerCluster = c_clusterSize;
    params.header.numClustersPerSuperCluster = c_superClusterSize;
    params.header.usePruning = 0U;
    params.header.useTwinCut = 0U;
    params.header.hasFep = 0U;
    params.header.cutoff = input.cutoff;
    params.header.rlist = input.rlist;
    std::snprintf(params.header.label, sizeof(params.header.label), "%s", label.c_str());
    return params;
}

StageGridDump buildGridOrthorhombicNoPrune(const OrthorhombicNoPruneInput& input,
                                           const std::string&               label)
{
    validateOrthorhombicNoPruneInput(input);

    const float boxVolume = input.box[0] * input.box[1] * input.box[2];
    const float atomDensity = static_cast<float>(input.xq.size()) / boxVolume;
    const float targetSubcellLength = std::cbrt(static_cast<float>(c_clusterSize) / atomDensity);
    const float targetCellLengthX = targetSubcellLength * c_gpuNumClusterPerBinX;
    const float targetCellLengthY = targetSubcellLength * c_gpuNumClusterPerBinY;

    const int numCellsX = std::max(1, static_cast<int>(input.box[0] / targetCellLengthX));
    const int numCellsY = std::max(1, static_cast<int>(input.box[1] / targetCellLengthY));
    const int numCells = numCellsX * numCellsY;
    const float cellSizeX = input.box[0] / numCellsX;
    const float cellSizeY = input.box[1] / numCellsY;
    const float inverseCellSizeX = 1.0F / cellSizeX;
    const float inverseCellSizeY = 1.0F / cellSizeY;

    std::vector<std::vector<int>> atomsPerCell(numCells);
    for (int atom = 0; atom < static_cast<int>(input.xq.size()); ++atom)
    {
        const auto& coordinate = input.xq[atom];
        int cx = static_cast<int>(coordinate.x * inverseCellSizeX);
        int cy = static_cast<int>(coordinate.y * inverseCellSizeY);
        cx = std::max(0, std::min(cx, numCellsX - 1));
        cy = std::max(0, std::min(cy, numCellsY - 1));
        atomsPerCell[cx * numCellsY + cy].push_back(atom);
    }

    StageGridDump grid;
    grid.header.box[0] = input.box[0];
    grid.header.box[1] = input.box[1];
    grid.header.box[2] = input.box[2];
    grid.header.numCells = static_cast<std::uint32_t>(numCells);
    grid.header.numBins = 0U;
    grid.header.numAtomsPerBin = static_cast<std::uint32_t>(c_superClusterSize * c_clusterSize);
    grid.header.numAtoms = 0U;
    grid.header.numShiftVec = 45U;
    std::snprintf(grid.header.label, sizeof(grid.header.label), "%s", label.c_str());
    grid.shiftVec = makeOrthorhombicShiftVectors(input.box);

    grid.numAtomsPerCell.resize(numCells + 1, 0);
    grid.cellToBin.resize(numCells + 2, 0);
    for (int cell = 0; cell < numCells; ++cell)
    {
        grid.numAtomsPerCell[cell] = static_cast<int>(atomsPerCell[cell].size());
        const int numBinsInCell = divideRoundUp(grid.numAtomsPerCell[cell], c_superClusterSize * c_clusterSize);
        grid.cellToBin[cell + 1] = grid.cellToBin[cell] + numBinsInCell;
    }
    grid.cellToBin[numCells + 1] = grid.cellToBin[numCells];
    grid.header.numBins = static_cast<std::uint32_t>(grid.cellToBin[numCells]);

    grid.atomIndices.assign(static_cast<std::size_t>(grid.header.numBins) * c_superClusterSize * c_clusterSize, -1);
    grid.bins.assign(input.xq.size(), -1);
    grid.numClustersPerBin.assign(static_cast<std::size_t>(grid.header.numBins) + 1U, 0);
    grid.header.numAtoms = static_cast<std::uint32_t>(grid.atomIndices.size());

    std::vector<int> sortBuffer;
    constexpr int subdivX = c_clusterSize;
    constexpr int subdivY = c_gpuNumClusterPerBinX * subdivX;
    constexpr int subdivZ = c_gpuNumClusterPerBinY * subdivY;

    for (int cell = 0; cell < numCells; ++cell)
    {
        auto& atomOrder = atomsPerCell[cell];
        const int numAtomsInCell = static_cast<int>(atomOrder.size());
        const int numBinsInCell = grid.cellToBin[cell + 1] - grid.cellToBin[cell];
        const int atomOffset = grid.cellToBin[cell] * c_superClusterSize * c_clusterSize;
        const int gridX = cell / numCellsY;
        const int gridY = cell - gridX * numCellsY;

        sortAtomIndicesInPlace(
                input, 2, false, atomOrder.data(), numAtomsInCell, 0.0F, 1.0F / input.box[2], numBinsInCell * 64, &sortBuffer);

        for (int subZ = 0; subZ < numBinsInCell * c_gpuNumClusterPerBinZ; ++subZ)
        {
            const int offsetZ = subZ * subdivZ;
            if (offsetZ >= numAtomsInCell)
            {
                break;
            }
            const int numAtomsZ = std::min(subdivZ, numAtomsInCell - offsetZ);
            sortAtomIndicesInPlace(input,
                                   1,
                                   (subZ & 1) != 0,
                                   atomOrder.data() + offsetZ,
                                   numAtomsZ,
                                   gridY * cellSizeY,
                                   inverseCellSizeY,
                                   subdivZ,
                                   &sortBuffer);

            const int cz = subZ / c_gpuNumClusterPerBinZ;
            for (int subY = 0; subY < c_gpuNumClusterPerBinY; ++subY)
            {
                const int offsetY = offsetZ + subY * subdivY;
                if (offsetY >= numAtomsInCell)
                {
                    break;
                }
                const int numAtomsY = std::min(subdivY, numAtomsInCell - offsetY);
                sortAtomIndicesInPlace(input,
                                       0,
                                       ((cz * c_gpuNumClusterPerBinY + subY) & 1) != 0,
                                       atomOrder.data() + offsetY,
                                       numAtomsY,
                                       gridX * cellSizeX,
                                       inverseCellSizeX,
                                       subdivY,
                                       &sortBuffer);
            }
        }

        for (int localIndex = 0; localIndex < numAtomsInCell; ++localIndex)
        {
            const int atomIndex = atomOrder[localIndex];
            const int slotIndex = atomOffset + localIndex;
            grid.atomIndices[slotIndex] = atomIndex;
            grid.bins[atomIndex] = slotIndex;
        }

        for (int bin = grid.cellToBin[cell]; bin < grid.cellToBin[cell + 1]; ++bin)
        {
            const int atomsBeforeBin = (bin - grid.cellToBin[cell]) * c_superClusterSize * c_clusterSize;
            const int numAtomsInBin = std::max(0, std::min(c_superClusterSize * c_clusterSize, numAtomsInCell - atomsBeforeBin));
            grid.numClustersPerBin[bin] = std::min(c_superClusterSize, divideRoundUp(numAtomsInBin, c_clusterSize));
        }
    }

    const auto clusterBounds = computeClusterBounds(input, grid);
    grid.packedBoundingBoxes = packClusterBounds(
            clusterBounds, grid.numClustersPerBin, static_cast<int>(grid.header.numBins));

    return grid;
}

NbnxmPairlistHost buildPairlistOrthorhombicNoPrune(const OrthorhombicNoPruneInput& input)
{
    validateOrthorhombicNoPruneInput(input);

    const StageGridDump grid = buildGridOrthorhombicNoPrune(input, "pairlist-grid");
    const auto numClustersPerBin = computeNumClustersPerBin(grid);
    const auto clusterBounds = computeClusterBounds(input, grid);
    const auto shiftVec = makeOrthorhombicShiftVectors(input.box);
    const float rlist2 = input.rlist * input.rlist;
    const float rbb2 = boundingBoxOnlyDistance2(input);
    NbnxmPairlistHost pairlist;
    pairlist.na_ci = c_clusterSize;
    pairlist.na_cj = c_clusterSize;
    pairlist.na_sc = c_superClusterSize;
    pairlist.rlist = input.rlist;
    pairlist.excl.push_back(makeFullExclusionMask());

    for (int sci = 0; sci < static_cast<int>(grid.header.numBins); ++sci)
    {
        const int numIClusters = numClustersPerBin[sci];
        if (numIClusters == 0)
        {
            continue;
        }

        for (int shift = 0; shift <= c_centralShiftIndex; ++shift)
        {
            const bool excludeSubDiagonal = (shift == c_centralShiftIndex);
            const int cjPackedBegin = static_cast<int>(pairlist.cjPacked.size());
            int currentCjCount = cjPackedBegin * c_jGroupSize;

            pairlist.sci.push_back({ sci, shift, cjPackedBegin, cjPackedBegin });

            for (int scj = (excludeSubDiagonal ? sci : 0); scj < static_cast<int>(grid.header.numBins); ++scj)
            {
                const int numJClusters = numClustersPerBin[scj];
                for (int subc = 0; subc < numJClusters; ++subc)
                {
                    const int cj_gl = scj * c_superClusterSize + subc;
                    const int ciLimit = (excludeSubDiagonal && sci == scj) ? (subc + 1) : numIClusters;

                    int npair = 0;
                    int ciLast = -1;
                    float d2Last = 0.0F;
                    std::uint32_t imask = 0U;

                    for (int ci = 0; ci < ciLimit; ++ci)
                    {
                        const float d2 = clusterBoundingBoxDistance2(clusterBounds[sci * c_superClusterSize + ci],
                                                                     clusterBounds[scj * c_superClusterSize + subc],
                                                                     shiftVec[shift]);
                        if (d2 < rlist2)
                        {
                            imask |= (1U << ((currentCjCount & (c_jGroupSize - 1)) * c_superClusterSize + ci));
                            ciLast = ci;
                            d2Last = d2;
                            ++npair;
                        }
                    }

                    if (npair == 1 && d2Last >= rbb2
                        && !clusterPairInRange(input, grid, sci, ciLast, scj, subc, shiftVec[shift], rlist2))
                    {
                        imask &= ~(1U << ((currentCjCount & (c_jGroupSize - 1)) * c_superClusterSize + ciLast));
                        --npair;
                    }

                    if (npair == 0)
                    {
                        continue;
                    }

                    const int cjPackedIndex = currentCjCount / c_jGroupSize;
                    const int cjOffset = currentCjCount & (c_jGroupSize - 1);
                    if (cjOffset == 0)
                    {
                        pairlist.cjPacked.push_back(nbnxn_cj_packed_t{});
                    }

                    auto& packed = pairlist.cjPacked[cjPackedIndex];
                    packed.cj[cjOffset] = cj_gl;
                    for (int warp = 0; warp < c_clusterPairSplit; ++warp)
                    {
                        packed.imei[warp].imask |= imask;
                    }

                    if (excludeSubDiagonal && sci == scj)
                    {
                        setSelfAndNewtonExclusionsGpu(&pairlist, cjPackedIndex, cjOffset, subc);
                    }

                    ++currentCjCount;
                    pairlist.nci_tot += npair;
                    pairlist.sci.back().cjPackedEnd = divideRoundUp(currentCjCount, c_jGroupSize);
                }
            }

            if (pairlist.sci.back().cjPackedBegin == pairlist.sci.back().cjPackedEnd)
            {
                pairlist.sci.pop_back();
                continue;
            }

            applyTopologyExclusionsForSci(input, grid, &pairlist, currentCjCount, excludeSubDiagonal);
        }
    }

    sortSciEntries(&pairlist);
    return pairlist;
}

} // namespace sponge::nbnxm

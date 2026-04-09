#include "nbnxm_builder.h"
#include "nbnxm_compare.h"
#include "nbnxm_exclusions.h"
#include "nbnxm_fixture.h"
#include "nbnxm_live_builder_input.h"

#include <cstdio>
#include <cmath>
#include <set>
#include <tuple>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace
{

struct CliOptions
{
    std::filesystem::path fixturePath;
    std::filesystem::path paramsPath;
    std::filesystem::path gridPath;
    std::filesystem::path pairlistPath;
    std::filesystem::path exclusionsPath;
};

CliOptions parseCli(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--fixture" && i + 1 < argc)
        {
            options.fixturePath = argv[++i];
        }
        else if (arg == "--params" && i + 1 < argc)
        {
            options.paramsPath = argv[++i];
        }
        else if (arg == "--grid" && i + 1 < argc)
        {
            options.gridPath = argv[++i];
        }
        else if (arg == "--pairlist" && i + 1 < argc)
        {
            options.pairlistPath = argv[++i];
        }
        else if (arg == "--exclusions" && i + 1 < argc)
        {
            options.exclusionsPath = argv[++i];
        }
        else
        {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }

    if (options.fixturePath.empty())
    {
        throw std::runtime_error("missing --fixture");
    }
    if (options.paramsPath.empty() && options.gridPath.empty() && options.pairlistPath.empty())
    {
        throw std::runtime_error("specify at least one of --params, --grid or --pairlist");
    }
    return options;
}

void printResult(const char* name, const sponge::nbnxm::CompareResult& result)
{
    std::cout << name << ": " << (result.equal ? "PASS" : "FAIL") << "\n";
    if (!result.equal)
    {
        std::cout << name << "_message: " << result.message << "\n";
    }
}

void printShiftHistogram(const char* name, const sponge::nbnxm::NbnxmPairlistHost& pairlist)
{
    std::map<int, int> histogram;
    for (const auto& sci : pairlist.sci)
    {
        histogram[sci.shift]++;
    }

    std::cout << name << ":";
    for (const auto& [shift, count] : histogram)
    {
        std::cout << " " << shift << "=" << count;
    }
    std::cout << "\n";
}

void printSciShiftDifferences(const sponge::nbnxm::NbnxmPairlistHost& built,
                              const sponge::nbnxm::NbnxmPairlistHost& reference)
{
    std::set<std::pair<int, int>> builtPairs;
    std::set<std::pair<int, int>> referencePairs;

    for (const auto& sci : built.sci)
    {
        builtPairs.emplace(sci.sci, sci.shift);
    }
    for (const auto& sci : reference.sci)
    {
        referencePairs.emplace(sci.sci, sci.shift);
    }

    std::cout << "extra_sci_shift_pairs:";
    for (const auto& entry : builtPairs)
    {
        if (referencePairs.count(entry) == 0U)
        {
            std::cout << " (" << entry.first << "," << entry.second << ")";
        }
    }
    std::cout << "\n";

    std::cout << "missing_sci_shift_pairs:";
    for (const auto& entry : referencePairs)
    {
        if (builtPairs.count(entry) == 0U)
        {
            std::cout << " (" << entry.first << "," << entry.second << ")";
        }
    }
    std::cout << "\n";
}

void printSciEntryDetails(const char* name,
                          const sponge::nbnxm::NbnxmPairlistHost& pairlist,
                          int sciIndex,
                          int shift)
{
    for (const auto& sci : pairlist.sci)
    {
        if (sci.sci != sciIndex || sci.shift != shift)
        {
            continue;
        }

        std::cout << name << "_entry (" << sciIndex << "," << shift << "):";
        for (int packedIndex = sci.cjPackedBegin; packedIndex < sci.cjPackedEnd; ++packedIndex)
        {
            const auto& packed = pairlist.cjPacked[packedIndex];
            for (int offset = 0; offset < 4; ++offset)
            {
                std::cout << " " << packed.cj[offset];
            }
        }
        std::cout << "\n";
        return;
    }
}

std::vector<int> collectJClustersForEntry(const sponge::nbnxm::NbnxmPairlistHost& pairlist, const sponge::nbnxm::nbnxn_sci_t& sci)
{
    std::vector<int> values;
    for (int packedIndex = sci.cjPackedBegin; packedIndex < sci.cjPackedEnd; ++packedIndex)
    {
        const auto& packed = pairlist.cjPacked[packedIndex];
        for (int offset = 0; offset < 4; ++offset)
        {
            values.push_back(packed.cj[offset]);
        }
    }
    return values;
}

void printFirstEntryContentMismatch(const sponge::nbnxm::NbnxmPairlistHost& built,
                                    const sponge::nbnxm::NbnxmPairlistHost& reference)
{
    for (const auto& referenceSci : reference.sci)
    {
        for (const auto& builtSci : built.sci)
        {
            if (builtSci.sci != referenceSci.sci || builtSci.shift != referenceSci.shift)
            {
                continue;
            }

            const auto builtJ = collectJClustersForEntry(built, builtSci);
            const auto referenceJ = collectJClustersForEntry(reference, referenceSci);
            if (builtJ != referenceJ)
            {
                std::cout << "first_entry_content_mismatch: (" << referenceSci.sci << "," << referenceSci.shift << ")\n";
                printSciEntryDetails("built", built, referenceSci.sci, referenceSci.shift);
                printSciEntryDetails("reference", reference, referenceSci.sci, referenceSci.shift);
                return;
            }
            break;
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    namespace nbnxm = sponge::nbnxm;

    try
    {
        const CliOptions options = parseCli(argc, argv);
        const auto fixture = nbnxm::loadFixture(options.fixturePath);

        if (!options.paramsPath.empty())
        {
            const auto referenceParams = nbnxm::loadStageParams(options.paramsPath);
            nbnxm::OrthorhombicNoPruneInput paramsInput;
            paramsInput.xq.resize(fixture.header.numWaters * 3);
            paramsInput.atomTypes.resize(fixture.header.numWaters * 3);
            paramsInput.box = { fixture.header.boxXX, fixture.header.boxYY, fixture.header.boxZZ };
            paramsInput.cutoff = referenceParams.header.cutoff;
            paramsInput.rlist = referenceParams.header.rlist;
            const auto builtParams =
                    nbnxm::buildPairlistParamsOrthorhombicNoPrune(paramsInput, referenceParams.header.label);
            printResult("params_compare", nbnxm::compareParams(builtParams, referenceParams));
        }

        if (!options.gridPath.empty())
        {
            const auto referenceGrid = nbnxm::loadStageGrid(options.gridPath);
            const auto input = nbnxm::makeOrthorhombicNoPruneInput(fixture, referenceGrid);
            const auto builtGrid = nbnxm::buildGridOrthorhombicNoPrune(input, referenceGrid.header.label);
            printResult("grid_compare", nbnxm::compareGrid(builtGrid, referenceGrid));
        }

        if (!options.pairlistPath.empty())
        {
            if (options.gridPath.empty())
            {
                throw std::runtime_error("--pairlist requires --grid");
            }

            const auto referenceGrid = nbnxm::loadStageGrid(options.gridPath);
            auto input = nbnxm::makeOrthorhombicNoPruneInput(fixture, referenceGrid);
            if (!options.exclusionsPath.empty())
            {
                const auto exclusions = nbnxm::loadExclusions(options.exclusionsPath);
                input = nbnxm::makeOrthorhombicNoPruneInput(fixture, referenceGrid, &exclusions);
            }

            const auto referencePairlist = nbnxm::loadStagePairlist(options.pairlistPath);
            nbnxm::StagePairlistDump builtPairlist;
            builtPairlist.pairlist = nbnxm::buildPairlistOrthorhombicNoPrune(input);
            builtPairlist.header.rlist = builtPairlist.pairlist.rlist;
            builtPairlist.header.nciTot = builtPairlist.pairlist.nci_tot;
            builtPairlist.header.numSci = static_cast<std::uint32_t>(builtPairlist.pairlist.sci.size());
            builtPairlist.header.numPackedJClusters =
                    static_cast<std::uint32_t>(builtPairlist.pairlist.cjPacked.size());
            builtPairlist.header.numExcl = static_cast<std::uint32_t>(builtPairlist.pairlist.excl.size());
            std::snprintf(builtPairlist.header.label,
                          sizeof(builtPairlist.header.label),
                          "%s",
                          referencePairlist.header.label);
            const auto result = nbnxm::comparePairlistBytes(builtPairlist, referencePairlist);
            printResult("pairlist_compare", result);
            if (!result.equal)
            {
                std::cout << "built_pairlist_counts: nciTot=" << builtPairlist.header.nciTot
                          << " numSci=" << builtPairlist.header.numSci
                          << " numPackedJClusters=" << builtPairlist.header.numPackedJClusters
                          << " numExcl=" << builtPairlist.header.numExcl << "\n";
                std::cout << "reference_pairlist_counts: nciTot=" << referencePairlist.header.nciTot
                          << " numSci=" << referencePairlist.header.numSci
                          << " numPackedJClusters=" << referencePairlist.header.numPackedJClusters
                          << " numExcl=" << referencePairlist.header.numExcl << "\n";
                printShiftHistogram("built_shift_histogram", builtPairlist.pairlist);
                printShiftHistogram("reference_shift_histogram", referencePairlist.pairlist);
                printSciShiftDifferences(builtPairlist.pairlist, referencePairlist.pairlist);
                printFirstEntryContentMismatch(builtPairlist.pairlist, referencePairlist.pairlist);
            }
        }

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "nbnxm_builder_compare error: " << ex.what() << "\n";
        return 1;
    }
}

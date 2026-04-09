#include "nbnxm_compare.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

struct CliOptions
{
    std::string kind;
    std::filesystem::path lhs;
    std::filesystem::path rhs;
};

CliOptions parseCli(int argc, char** argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--kind" && i + 1 < argc)
        {
            options.kind = argv[++i];
        }
        else if (arg == "--lhs" && i + 1 < argc)
        {
            options.lhs = argv[++i];
        }
        else if (arg == "--rhs" && i + 1 < argc)
        {
            options.rhs = argv[++i];
        }
        else
        {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }

    if (options.kind.empty())
    {
        throw std::runtime_error("missing --kind");
    }
    if (options.lhs.empty())
    {
        throw std::runtime_error("missing --lhs");
    }
    if (options.rhs.empty())
    {
        throw std::runtime_error("missing --rhs");
    }

    return options;
}

void printResult(const std::string& kind,
                 const std::filesystem::path& lhs,
                 const std::filesystem::path& rhs,
                 const sponge::nbnxm::CompareResult& result)
{
    std::cout << "kind: " << kind << "\n";
    std::cout << "lhs: " << lhs << "\n";
    std::cout << "rhs: " << rhs << "\n";
    std::cout << "stage_compare: " << (result.equal ? "PASS" : "FAIL") << "\n";
    if (!result.equal)
    {
        std::cout << "message: " << result.message << "\n";
    }
}

} // namespace

int main(int argc, char** argv)
{
    namespace nbnxm = sponge::nbnxm;

    try
    {
        const CliOptions options = parseCli(argc, argv);
        if (options.kind == "params")
        {
            const auto lhs = nbnxm::loadStageParams(options.lhs);
            const auto rhs = nbnxm::loadStageParams(options.rhs);
            const auto result = nbnxm::compareParams(lhs, rhs);
            printResult(options.kind, options.lhs, options.rhs, result);
            return result.equal ? 0 : 2;
        }
        if (options.kind == "grid")
        {
            const auto lhs = nbnxm::loadStageGrid(options.lhs);
            const auto rhs = nbnxm::loadStageGrid(options.rhs);
            const auto result = nbnxm::compareGrid(lhs, rhs);
            printResult(options.kind, options.lhs, options.rhs, result);
            return result.equal ? 0 : 2;
        }
        if (options.kind == "pairlist")
        {
            const auto lhs = nbnxm::loadStagePairlist(options.lhs);
            const auto rhs = nbnxm::loadStagePairlist(options.rhs);
            const auto result = nbnxm::comparePairlistBytes(lhs, rhs);
            printResult(options.kind, options.lhs, options.rhs, result);
            return result.equal ? 0 : 2;
        }

        throw std::runtime_error("unsupported --kind value: " + options.kind);
    }
    catch (const std::exception& ex)
    {
        std::cerr << "nbnxm_stage_compare error: " << ex.what() << "\n";
        return 1;
    }
}

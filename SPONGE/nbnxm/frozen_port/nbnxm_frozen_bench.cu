#include "../nbnxm_fixture.h"
#include "../nbnxm_compare.h"
#include "../nbnxm_exclusions.h"
#include "../nbnxm_force_dump.h"
#include "../nbnxm_live_builder_input.h"
#include "../nbnxm_stage.h"
#include "../nbnxm_builder.h"
#include "ljewald_kernel_frozen_port.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace sponge::nbnxm::frozen
{
namespace
{

struct CliOptions
{
    std::filesystem::path fixturePath;
    std::filesystem::path gridDumpPath;
    std::filesystem::path exclusionsDumpPath;
    std::filesystem::path pairlistDumpPath;
    std::filesystem::path referenceForceDumpPath;
    int iterations = 50;
    int warmup = 10;
    int repeats = 3;
    float thresholdPct = 5.0F;
    std::string mode = "pairlist+kernel";
    std::string pairlistSource = "fixture";
    std::filesystem::path gromacsBenchBin = "/home/ylj/应用/gromacs-2026.1/build-cuda/bin/nbnxm-ljewald-bench";
};

struct Runtime
{
    FixtureData fixture;
    NbnxmPairlistHost pairlist;

    float4* d_xq = nullptr;
    int* d_atomTypes = nullptr;
    float3* d_shiftVec = nullptr;
    float2* d_nbfp = nullptr;
    float2* d_nbfpComb = nullptr;
    nbnxn_sci_t* d_sci = nullptr;
    nbnxn_cj_packed_t* d_cjPacked = nullptr;
    nbnxn_excl_t* d_excl = nullptr;
    float3* d_f = nullptr;
    float3* d_fShift = nullptr;

    int smMajor = 0;
    int smMinor = 0;
    std::string gpuName;
    cudaStream_t stream = nullptr;

    ~Runtime()
    {
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
        cudaFree(d_xq);
        cudaFree(d_atomTypes);
        cudaFree(d_shiftVec);
        cudaFree(d_nbfp);
        cudaFree(d_nbfpComb);
        cudaFree(d_sci);
        cudaFree(d_cjPacked);
        cudaFree(d_excl);
        cudaFree(d_f);
        cudaFree(d_fShift);
    }
};

struct Stats
{
    double minUs = std::numeric_limits<double>::infinity();
    double maxUs = 0.0;
    double avgUs = 0.0;
    std::vector<double> samples;
};

void checkCuda(cudaError_t err, const char* where)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

CliOptions parseCli(int argc, char** argv)
{
    CliOptions opt;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--fixture" && i + 1 < argc)
        {
            opt.fixturePath = argv[++i];
        }
        else if (arg == "--iters" && i + 1 < argc)
        {
            opt.iterations = std::atoi(argv[++i]);
        }
        else if (arg == "--grid-dump" && i + 1 < argc)
        {
            opt.gridDumpPath = argv[++i];
        }
        else if (arg == "--exclusions-dump" && i + 1 < argc)
        {
            opt.exclusionsDumpPath = argv[++i];
        }
        else if (arg == "--warmup" && i + 1 < argc)
        {
            opt.warmup = std::atoi(argv[++i]);
        }
        else if (arg == "--repeats" && i + 1 < argc)
        {
            opt.repeats = std::atoi(argv[++i]);
        }
        else if (arg == "--mode" && i + 1 < argc)
        {
            opt.mode = argv[++i];
        }
        else if (arg == "--pairlist-source" && i + 1 < argc)
        {
            opt.pairlistSource = argv[++i];
        }
        else if (arg == "--pairlist-dump" && i + 1 < argc)
        {
            opt.pairlistDumpPath = argv[++i];
        }
        else if (arg == "--reference-force-dump" && i + 1 < argc)
        {
            opt.referenceForceDumpPath = argv[++i];
        }
        else if (arg == "--threshold-pct" && i + 1 < argc)
        {
            opt.thresholdPct = std::atof(argv[++i]);
        }
        else if (arg == "--gromacs-bench-bin" && i + 1 < argc)
        {
            opt.gromacsBenchBin = argv[++i];
        }
        else
        {
            throw std::runtime_error("Usage: nbnxm_frozen_bench --fixture <path> [--pairlist-source fixture|live] [--grid-dump <path>] [--exclusions-dump <path>] [--pairlist-dump <path>] [--reference-force-dump <path>] [--iters N] [--warmup N] [--repeats N] [--mode pairlist-only|pairlist+kernel|compare] [--gromacs-bench-bin <path>] [--threshold-pct P]");
        }
    }
    if (opt.fixturePath.empty())
    {
        throw std::runtime_error("missing --fixture");
    }
    if (opt.mode != "pairlist-only" && opt.mode != "pairlist+kernel" && opt.mode != "compare")
    {
        throw std::runtime_error("--mode must be pairlist-only, pairlist+kernel or compare");
    }
    if (opt.pairlistSource != "fixture" && opt.pairlistSource != "live")
    {
        throw std::runtime_error("--pairlist-source must be fixture or live");
    }
    if (opt.pairlistSource == "live" && (opt.gridDumpPath.empty() || opt.exclusionsDumpPath.empty()))
    {
        throw std::runtime_error("live pairlist source requires --grid-dump and --exclusions-dump");
    }
    if (opt.iterations <= 0 || opt.warmup < 0 || opt.repeats <= 0)
    {
        throw std::runtime_error("invalid iters/warmup/repeats");
    }
    return opt;
}

template <typename TFrom, typename TTo>
std::vector<TTo> convertVec(const std::vector<TFrom>& in)
{
    std::vector<TTo> out(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        out[i].x = in[i].x;
        out[i].y = in[i].y;
        if constexpr (sizeof(TTo) >= sizeof(float) * 3)
        {
            out[i].z = in[i].z;
        }
        if constexpr (sizeof(TTo) >= sizeof(float) * 4)
        {
            out[i].w = in[i].w;
        }
    }
    return out;
}

Runtime makeRuntime(const FixtureData& fixture, const NbnxmPairlistHost& pairlist)
{
    Runtime rt;
    rt.fixture = fixture;
    rt.pairlist = pairlist;

    cudaDeviceProp prop{};
    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    rt.smMajor = prop.major;
    rt.smMinor = prop.minor;
    rt.gpuName = prop.name;

    checkCuda(cudaStreamCreate(&rt.stream), "cudaStreamCreate");

    const auto xq = convertVec<Float4, float4>(fixture.xq);
    const auto shiftVec = convertVec<Float3, float3>(fixture.shiftVec);
    const auto nbfp = convertVec<Float2, float2>(fixture.nbfp);
    const auto nbfpComb = convertVec<Float2, float2>(fixture.nbfpComb);

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_xq), sizeof(float4) * xq.size()), "cudaMalloc d_xq");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_atomTypes), sizeof(int) * fixture.atomTypes.size()), "cudaMalloc d_atomTypes");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_shiftVec), sizeof(float3) * shiftVec.size()), "cudaMalloc d_shiftVec");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_nbfp), sizeof(float2) * nbfp.size()), "cudaMalloc d_nbfp");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_nbfpComb), sizeof(float2) * nbfpComb.size()), "cudaMalloc d_nbfpComb");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_sci), sizeof(nbnxn_sci_t) * pairlist.sci.size()), "cudaMalloc d_sci");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_cjPacked), sizeof(nbnxn_cj_packed_t) * pairlist.cjPacked.size()), "cudaMalloc d_cjPacked");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_excl), sizeof(nbnxn_excl_t) * pairlist.excl.size()), "cudaMalloc d_excl");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_f), sizeof(float3) * fixture.xq.size()), "cudaMalloc d_f");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&rt.d_fShift), sizeof(float3) * shiftVec.size()), "cudaMalloc d_fShift");

    checkCuda(cudaMemcpy(rt.d_xq, xq.data(), sizeof(float4) * xq.size(), cudaMemcpyHostToDevice), "copy xq");
    checkCuda(cudaMemcpy(rt.d_atomTypes,
                         fixture.atomTypes.data(),
                         sizeof(int) * fixture.atomTypes.size(),
                         cudaMemcpyHostToDevice),
              "copy atomTypes");
    checkCuda(cudaMemcpy(rt.d_shiftVec,
                         shiftVec.data(),
                         sizeof(float3) * shiftVec.size(),
                         cudaMemcpyHostToDevice),
              "copy shiftVec");
    checkCuda(cudaMemcpy(rt.d_nbfp, nbfp.data(), sizeof(float2) * nbfp.size(), cudaMemcpyHostToDevice), "copy nbfp");
    checkCuda(cudaMemcpy(rt.d_nbfpComb,
                         nbfpComb.data(),
                         sizeof(float2) * nbfpComb.size(),
                         cudaMemcpyHostToDevice),
              "copy nbfpComb");
    checkCuda(cudaMemcpy(rt.d_sci,
                         pairlist.sci.data(),
                         sizeof(nbnxn_sci_t) * pairlist.sci.size(),
                         cudaMemcpyHostToDevice),
              "copy sci");
    checkCuda(cudaMemcpy(rt.d_cjPacked,
                         pairlist.cjPacked.data(),
                         sizeof(nbnxn_cj_packed_t) * pairlist.cjPacked.size(),
                         cudaMemcpyHostToDevice),
              "copy cjPacked");
    checkCuda(cudaMemcpy(rt.d_excl,
                         pairlist.excl.data(),
                         sizeof(nbnxn_excl_t) * pairlist.excl.size(),
                         cudaMemcpyHostToDevice),
              "copy excl");

    checkCuda(cudaMemset(rt.d_f, 0, sizeof(float3) * fixture.xq.size()), "memset f");
    checkCuda(cudaMemset(rt.d_fShift, 0, sizeof(float3) * shiftVec.size()), "memset fShift");

    return rt;
}

FrozenKernelContext makeKernelContext(Runtime& rt)
{
    FrozenKernelContext ctx;
    ctx.atdat.numTypes = static_cast<int>(rt.fixture.header.numTypes);
    ctx.atdat.xq = rt.d_xq;
    ctx.atdat.atomTypes = rt.d_atomTypes;
    ctx.atdat.f = rt.d_f;
    ctx.atdat.shiftVec = rt.d_shiftVec;
    ctx.atdat.fShift = rt.d_fShift;

    ctx.nbparam.vdwType = VdwType::EwaldGeom;
    ctx.nbparam.epsfac = rt.fixture.header.epsfac;
    ctx.nbparam.ewald_beta = rt.fixture.header.ewaldBeta;
    ctx.nbparam.ewaldcoeff_lj = rt.fixture.header.ewaldCoeffLJ;
    ctx.nbparam.rcoulomb_sq = rt.fixture.header.rcoulombSq;
    ctx.nbparam.sh_lj_ewald = rt.fixture.header.shLjEwald;
    ctx.nbparam.nbfp = rt.d_nbfp;
    ctx.nbparam.nbfp_comb = rt.d_nbfpComb;

    ctx.plist.numSci = static_cast<int>(rt.pairlist.sci.size());
    ctx.plist.sorting.sciSorted = rt.d_sci;
    ctx.plist.cjPacked = rt.d_cjPacked;
    ctx.plist.excl = rt.d_excl;
    ctx.stream = rt.stream;
    ctx.deviceSmMajor = rt.smMajor;
    ctx.deviceSmMinor = rt.smMinor;
    return ctx;
}

double runSpongeOnce(Runtime& rt, int warmup, int iterations)
{
    FrozenKernelContext ctx = makeKernelContext(rt);

    for (int i = 0; i < warmup; ++i)
    {
        checkCuda(cudaMemsetAsync(rt.d_f, 0, sizeof(float3) * rt.fixture.xq.size(), rt.stream), "warmup memset f");
        checkCuda(cudaMemsetAsync(rt.d_fShift,
                                  0,
                                  sizeof(float3) * rt.fixture.shiftVec.size(),
                                  rt.stream),
                  "warmup memset fshift");
        launchLJEwaldFrozenKernel(ctx, false);
        checkCuda(cudaStreamSynchronize(rt.stream), "warmup sync");
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    double totalMs = 0.0;
    for (int i = 0; i < iterations; ++i)
    {
        checkCuda(cudaMemsetAsync(rt.d_f, 0, sizeof(float3) * rt.fixture.xq.size(), rt.stream), "timed memset f");
        checkCuda(cudaMemsetAsync(rt.d_fShift,
                                  0,
                                  sizeof(float3) * rt.fixture.shiftVec.size(),
                                  rt.stream),
                  "timed memset fshift");
        checkCuda(cudaStreamSynchronize(rt.stream), "pre record sync");
        checkCuda(cudaEventRecord(start, rt.stream), "record start");
        launchLJEwaldFrozenKernel(ctx, false);
        checkCuda(cudaEventRecord(stop, rt.stream), "record stop");
        checkCuda(cudaEventSynchronize(stop), "stop sync");
        float ms = 0.0F;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        totalMs += ms;
    }

    checkCuda(cudaEventDestroy(start), "destroy start");
    checkCuda(cudaEventDestroy(stop), "destroy stop");

    return totalMs * 1000.0 / static_cast<double>(iterations);
}

Stats aggregate(const std::vector<double>& samples)
{
    Stats s;
    s.samples = samples;
    s.minUs = *std::min_element(samples.begin(), samples.end());
    s.maxUs = *std::max_element(samples.begin(), samples.end());
    double sum = 0.0;
    for (double v : samples)
    {
        sum += v;
    }
    s.avgUs = sum / static_cast<double>(samples.size());
    return s;
}

double runGromacsOnce(const CliOptions& opt)
{
    std::ostringstream cmd;
    cmd << '"' << opt.gromacsBenchBin.string() << '"'
        << " --fixture " << '"' << opt.fixturePath.string() << '"'
        << " --impl frozen"
        << " --iters " << opt.iterations
        << " --warmup " << opt.warmup;

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (pipe == nullptr)
    {
        throw std::runtime_error("failed to run gromacs bench");
    }

    std::array<char, 1024> buffer{};
    std::string output;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
    {
        output += buffer.data();
    }
    const int rc = pclose(pipe);
    if (rc != 0)
    {
        throw std::runtime_error("gromacs bench failed: " + output);
    }

    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line))
    {
        const std::string key = "avg_kernel_us:";
        auto pos = line.find(key);
        if (pos != std::string::npos)
        {
            const std::string value = line.substr(pos + key.size());
            return std::atof(value.c_str());
        }
    }

    throw std::runtime_error("could not parse avg_kernel_us from gromacs output: " + output);
}

Stats runGromacsRepeated(const CliOptions& opt)
{
    std::vector<double> samples;
    samples.reserve(opt.repeats);
    for (int i = 0; i < opt.repeats; ++i)
    {
        samples.push_back(runGromacsOnce(opt));
    }
    return aggregate(samples);
}

Stats runSpongeRepeated(Runtime& rt, const CliOptions& opt)
{
    std::vector<double> samples;
    samples.reserve(opt.repeats);
    for (int i = 0; i < opt.repeats; ++i)
    {
        samples.push_back(runSpongeOnce(rt, opt.warmup, opt.iterations));
    }
    return aggregate(samples);
}

LJEwaldForceSummary summarizeForces(const std::vector<float3>& forces)
{
    LJEwaldForceSummary summary;
    for (const auto& force : forces)
    {
        summary.sumX += force.x;
        summary.sumY += force.y;
        summary.sumZ += force.z;
        summary.l1Norm += std::abs(force.x) + std::abs(force.y) + std::abs(force.z);
        summary.l2Norm += static_cast<double>(force.x) * force.x + static_cast<double>(force.y) * force.y
                          + static_cast<double>(force.z) * force.z;
        summary.maxAbs = std::max(summary.maxAbs,
                                  std::max(std::abs(static_cast<double>(force.x)),
                                           std::max(std::abs(static_cast<double>(force.y)),
                                                    std::abs(static_cast<double>(force.z)))));
    }
    summary.l2Norm = std::sqrt(summary.l2Norm);
    return summary;
}

std::uint64_t checksumForces(const std::vector<float3>& forces)
{
    std::uint64_t hash = 1469598103934665603ULL;
    auto mix = [&hash](float value)
    {
        std::uint32_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    };

    for (const auto& force : forces)
    {
        mix(force.x);
        mix(force.y);
        mix(force.z);
    }
    return hash;
}

std::vector<float3> computeForces(Runtime& rt)
{
    FrozenKernelContext ctx = makeKernelContext(rt);

    checkCuda(cudaMemsetAsync(rt.d_f, 0, sizeof(float3) * rt.fixture.xq.size(), rt.stream), "force memset f");
    checkCuda(cudaMemsetAsync(rt.d_fShift, 0, sizeof(float3) * rt.fixture.shiftVec.size(), rt.stream),
              "force memset fshift");
    launchLJEwaldFrozenKernel(ctx, false);
    checkCuda(cudaStreamSynchronize(rt.stream), "force sync");

    std::vector<float3> forces(rt.fixture.xq.size());
    checkCuda(cudaMemcpy(forces.data(),
                         rt.d_f,
                         sizeof(float3) * forces.size(),
                         cudaMemcpyDeviceToHost),
              "copy forces");
    return forces;
}

bool forceReferenceAvailable(const FixtureHeader& header)
{
    return header.forceChecksum != 0U || header.referenceForceSummary.l1Norm != 0.0
           || header.referenceForceSummary.l2Norm != 0.0 || header.referenceForceSummary.maxAbs != 0.0;
}

bool equalForceSummary(const LJEwaldForceSummary& lhs, const LJEwaldForceSummary& rhs)
{
    return lhs.sumX == rhs.sumX && lhs.sumY == rhs.sumY && lhs.sumZ == rhs.sumZ && lhs.l1Norm == rhs.l1Norm
           && lhs.l2Norm == rhs.l2Norm && lhs.maxAbs == rhs.maxAbs;
}

void printStats(const char* label, const Stats& s)
{
    std::cout << label << "_us_min: " << s.minUs << "\n";
    std::cout << label << "_us_avg: " << s.avgUs << "\n";
    std::cout << label << "_us_max: " << s.maxUs << "\n";
}

struct ForceDiffStats
{
    double rms = 0.0;
    double maxAbs = 0.0;
    std::size_t worstAtom = 0;
    int worstComponent = 0;
    double worstObserved = 0.0;
    double worstReference = 0.0;
};

ForceDiffStats computeForceDiffStats(const std::vector<float3>& observed, const std::vector<Float3>& reference)
{
    if (observed.size() != reference.size())
    {
        throw std::runtime_error("force array size mismatch");
    }

    ForceDiffStats stats;
    double sumSq = 0.0;
    std::size_t componentCount = 0;
    for (std::size_t atom = 0; atom < observed.size(); ++atom)
    {
        const std::array<double, 3> obs = { observed[atom].x, observed[atom].y, observed[atom].z };
        const std::array<double, 3> ref = { reference[atom].x, reference[atom].y, reference[atom].z };
        for (int component = 0; component < 3; ++component)
        {
            const double diff = obs[component] - ref[component];
            const double absDiff = std::abs(diff);
            sumSq += diff * diff;
            ++componentCount;
            if (absDiff > stats.maxAbs)
            {
                stats.maxAbs = absDiff;
                stats.worstAtom = atom;
                stats.worstComponent = component;
                stats.worstObserved = obs[component];
                stats.worstReference = ref[component];
            }
        }
    }

    if (componentCount > 0)
    {
        stats.rms = std::sqrt(sumSq / static_cast<double>(componentCount));
    }
    return stats;
}

NbnxmPairlistHost makePairlist(const CliOptions& options, const FixtureData& fixture)
{
    if (options.pairlistSource == "fixture")
    {
        return fixture.pairlist;
    }

    const StageGridDump grid = loadStageGrid(options.gridDumpPath);
    const ExclusionsDump exclusions = loadExclusions(options.exclusionsDumpPath);
    const auto input = makeOrthorhombicNoPruneInput(fixture, grid, &exclusions);
    return buildPairlistOrthorhombicNoPrune(input);
}

} // namespace
} // namespace sponge::nbnxm::frozen

int main(int argc, char** argv)
{
    using namespace sponge::nbnxm::frozen;
    using namespace sponge::nbnxm;

    try
    {
        const CliOptions options = parseCli(argc, argv);
        const sponge::nbnxm::FixtureData fixture = sponge::nbnxm::loadFixture(options.fixturePath);

        std::cout << "fixture: " << options.fixturePath << "\n";
        std::cout << "variant: " << fixture.header.variantName << "\n";
        std::cout << "waters: " << fixture.header.numWaters << "\n";
        std::cout << "atoms: " << fixture.header.numAtoms << "\n";
        std::cout << "iters: " << options.iterations << " warmup: " << options.warmup
                  << " repeats: " << options.repeats << "\n";
        const NbnxmPairlistHost pairlist = makePairlist(options, fixture);
        std::cout << "pairlist_source: " << options.pairlistSource << "\n";
        std::cout << "pairlist_sci: " << pairlist.sci.size()
                  << " pairlist_cjPacked: " << pairlist.cjPacked.size()
                  << " pairlist_excl: " << pairlist.excl.size() << "\n";

        if (options.mode == "pairlist-only")
        {
            if (options.pairlistDumpPath.empty())
            {
                std::cerr << "pairlist-only mode requires --pairlist-dump\n";
                return 3;
            }

            const StagePairlistDump pairlistDump = loadStagePairlist(options.pairlistDumpPath);
            StagePairlistDump fixtureDump;
            fixtureDump.header.rlist = pairlist.rlist;
            fixtureDump.header.nciTot = pairlist.nci_tot;
            fixtureDump.header.numSci = static_cast<std::uint32_t>(pairlist.sci.size());
            fixtureDump.header.numPackedJClusters = static_cast<std::uint32_t>(pairlist.cjPacked.size());
            fixtureDump.header.numExcl = static_cast<std::uint32_t>(pairlist.excl.size());
            fixtureDump.pairlist = pairlist;

            const CompareResult result = comparePairlistBytes(fixtureDump, pairlistDump);
            std::cout << "pairlist_compare: " << (result.equal ? "PASS" : "FAIL") << "\n";
            if (!result.equal)
            {
                std::cout << "pairlist_compare_message: " << result.message << "\n";
                return 2;
            }
            return 0;
        }

        Runtime rt                  = makeRuntime(fixture, pairlist);
        std::cout << "gpu: " << rt.gpuName << "\n";
        std::cout << "sm: " << rt.smMajor << "." << rt.smMinor << "\n";

        if (options.mode == "pairlist+kernel")
        {
            const auto forces = computeForces(rt);
            const auto summary = summarizeForces(forces);
            const auto checksum = checksumForces(forces);
            std::cout << "force_checksum: " << checksum << "\n";
            std::cout << "force_sum_x: " << summary.sumX << "\n";
            std::cout << "force_sum_y: " << summary.sumY << "\n";
            std::cout << "force_sum_z: " << summary.sumZ << "\n";
            std::cout << "force_l1: " << summary.l1Norm << "\n";
            std::cout << "force_l2: " << summary.l2Norm << "\n";
            std::cout << "force_max_abs: " << summary.maxAbs << "\n";

            if (!forceReferenceAvailable(fixture.header))
            {
                std::cout << "kernel_compare: NO_REFERENCE\n";
                return 3;
            }

            std::cout << "reference_force_checksum: " << fixture.header.forceChecksum << "\n";
            std::cout << "reference_force_sum_x: " << fixture.header.referenceForceSummary.sumX << "\n";
            std::cout << "reference_force_sum_y: " << fixture.header.referenceForceSummary.sumY << "\n";
            std::cout << "reference_force_sum_z: " << fixture.header.referenceForceSummary.sumZ << "\n";
            std::cout << "reference_force_l1: " << fixture.header.referenceForceSummary.l1Norm << "\n";
            std::cout << "reference_force_l2: " << fixture.header.referenceForceSummary.l2Norm << "\n";
            std::cout << "reference_force_max_abs: " << fixture.header.referenceForceSummary.maxAbs << "\n";

            const bool checksumMatch = (checksum == fixture.header.forceChecksum);
            const bool summaryMatch = equalForceSummary(summary, fixture.header.referenceForceSummary);
            std::cout << "kernel_compare: " << ((checksumMatch && summaryMatch) ? "PASS" : "FAIL") << "\n";
            if (!checksumMatch)
            {
                std::cout << "kernel_compare_message: force checksum mismatch\n";
            }
            else if (!summaryMatch)
            {
                std::cout << "kernel_compare_message: force summary mismatch\n";
            }
            if (!options.referenceForceDumpPath.empty())
            {
                const auto referenceForces = loadForces(options.referenceForceDumpPath);
                const auto diffStats = computeForceDiffStats(forces, referenceForces.forces);
                static constexpr const char* componentNames[3] = { "x", "y", "z" };
                std::cout << "force_array_compare: AVAILABLE\n";
                std::cout << "force_array_rms: " << diffStats.rms << "\n";
                std::cout << "force_array_max_abs: " << diffStats.maxAbs << "\n";
                std::cout << "force_array_worst_atom: " << diffStats.worstAtom << "\n";
                std::cout << "force_array_worst_component: " << componentNames[diffStats.worstComponent] << "\n";
                std::cout << "force_array_worst_observed: " << diffStats.worstObserved << "\n";
                std::cout << "force_array_worst_reference: " << diffStats.worstReference << "\n";
            }
            return (checksumMatch && summaryMatch) ? 0 : 2;
        }

        const Stats spongeStats = runSpongeRepeated(rt, options);
        printStats("sponge_frozen", spongeStats);

        if (options.mode == "compare")
        {
            const Stats gmxStats = runGromacsRepeated(options);
            printStats("gromacs_frozen", gmxStats);

            const double diffPct = std::abs(spongeStats.avgUs - gmxStats.avgUs) * 100.0 / gmxStats.avgUs;
            std::cout << "diff_pct_avg: " << diffPct << "\n";
            std::cout << "threshold_pct: " << options.thresholdPct << "\n";
            if (diffPct <= options.thresholdPct)
            {
                std::cout << "result: PASS\n";
                return 0;
            }
            std::cout << "result: FAIL\n";
            return 2;
        }
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "nbnxm_frozen_bench error: " << ex.what() << "\n";
        return 1;
    }
}

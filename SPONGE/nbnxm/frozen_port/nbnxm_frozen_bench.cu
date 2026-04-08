#include "../nbnxm_fixture.h"
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
    int iterations = 50;
    int warmup = 10;
    int repeats = 3;
    float thresholdPct = 5.0F;
    std::string mode = "compare";
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
            throw std::runtime_error("Usage: nbnxm_frozen_bench --fixture <path> [--iters N] [--warmup N] [--repeats N] [--mode sponge|compare] [--gromacs-bench-bin <path>] [--threshold-pct P]");
        }
    }
    if (opt.fixturePath.empty())
    {
        throw std::runtime_error("missing --fixture");
    }
    if (opt.mode != "sponge" && opt.mode != "compare")
    {
        throw std::runtime_error("--mode must be sponge or compare");
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

void printStats(const char* label, const Stats& s)
{
    std::cout << label << "_us_min: " << s.minUs << "\n";
    std::cout << label << "_us_avg: " << s.avgUs << "\n";
    std::cout << label << "_us_max: " << s.maxUs << "\n";
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

        NbnxmPairlistHost pairlist = fixture.pairlist;

        Runtime rt = makeRuntime(fixture, pairlist);
        const Stats spongeStats = runSpongeRepeated(rt, options);

        std::cout << "fixture: " << options.fixturePath << "\n";
        std::cout << "variant: " << fixture.header.variantName << "\n";
        std::cout << "gpu: " << rt.gpuName << "\n";
        std::cout << "sm: " << rt.smMajor << "." << rt.smMinor << "\n";
        std::cout << "waters: " << fixture.header.numWaters << "\n";
        std::cout << "atoms: " << fixture.header.numAtoms << "\n";
        std::cout << "iters: " << options.iterations << " warmup: " << options.warmup
                  << " repeats: " << options.repeats << "\n";
        std::cout << "pairlist_source: fixture\n";
        std::cout << "pairlist_sci: " << pairlist.sci.size()
                  << " pairlist_cjPacked: " << pairlist.cjPacked.size()
                  << " pairlist_excl: " << pairlist.excl.size() << "\n";
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

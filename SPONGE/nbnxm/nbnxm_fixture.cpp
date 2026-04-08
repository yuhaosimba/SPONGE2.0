#include "nbnxm_fixture.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace sponge::nbnxm
{
namespace
{

template <typename T>
void writeScalar(std::ofstream* stream, const T& value)
{
    stream->write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void readScalar(std::ifstream* stream, T* value)
{
    stream->read(reinterpret_cast<char*>(value), sizeof(T));
}

template <typename T>
void writeVector(std::ofstream* stream, const std::vector<T>& values)
{
    const std::uint64_t size = static_cast<std::uint64_t>(values.size());
    writeScalar(stream, size);
    if (!values.empty())
    {
        stream->write(reinterpret_cast<const char*>(values.data()),
                      static_cast<std::streamsize>(sizeof(T) * values.size()));
    }
}

template <typename T>
void readVector(std::ifstream* stream, std::vector<T>* values)
{
    std::uint64_t size = 0;
    readScalar(stream, &size);
    values->resize(static_cast<std::size_t>(size));
    if (!values->empty())
    {
        stream->read(reinterpret_cast<char*>(values->data()),
                     static_cast<std::streamsize>(sizeof(T) * values->size()));
    }
}

int computeNciTotFromImask(const NbnxmPairlistHost& pairlist)
{
    int total = 0;
    for (const auto& packed : pairlist.cjPacked)
    {
        const std::uint32_t imask = packed.imei[0].imask;
        for (int jm = 0; jm < c_jGroupSize; ++jm)
        {
            const std::uint32_t iMask8 = (imask >> (jm * c_superClusterSize)) & ((1U << c_superClusterSize) - 1U);
            total += __builtin_popcount(iMask8);
        }
    }
    return total;
}

} // namespace

FixtureData loadFixture(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to open fixture for reading: " + path.string());
    }

    FixtureData fixture;
    readScalar(&stream, &fixture.header);

    if (fixture.header.magic != c_fixtureMagic)
    {
        throw std::runtime_error("fixture magic mismatch");
    }
    if (fixture.header.version != c_fixtureVersion)
    {
        throw std::runtime_error("unsupported fixture version");
    }
    if (fixture.header.endian != c_fixtureEndian)
    {
        throw std::runtime_error("fixture endian mismatch");
    }

    readVector(&stream, &fixture.xq);
    readVector(&stream, &fixture.atomTypes);
    readVector(&stream, &fixture.shiftVec);
    readVector(&stream, &fixture.nbfp);
    readVector(&stream, &fixture.nbfpComb);

    readVector(&stream, &fixture.pairlist.sci);
    readVector(&stream, &fixture.pairlist.cjPacked);
    readVector(&stream, &fixture.pairlist.excl);

    fixture.pairlist.na_ci = static_cast<int>(fixture.header.numAtomsPerCluster);
    fixture.pairlist.na_cj = static_cast<int>(fixture.header.numAtomsPerCluster);
    fixture.pairlist.na_sc = c_superClusterSize;
    fixture.pairlist.nci_tot = computeNciTotFromImask(fixture.pairlist);
    fixture.pairlist.rlist = std::sqrt(fixture.header.rcoulombSq);

    if (fixture.xq.size() != fixture.header.numAtoms ||
        fixture.atomTypes.size() != fixture.header.numAtoms ||
        fixture.pairlist.sci.size() != fixture.header.numSci ||
        fixture.pairlist.cjPacked.size() != fixture.header.numPackedJClusters ||
        fixture.pairlist.excl.size() != fixture.header.numExcl)
    {
        throw std::runtime_error("fixture header/payload mismatch");
    }

    return fixture;
}

void saveFixture(const std::filesystem::path& path, const FixtureData& fixture)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error("failed to open fixture for writing: " + path.string());
    }

    writeScalar(&stream, fixture.header);
    writeVector(&stream, fixture.xq);
    writeVector(&stream, fixture.atomTypes);
    writeVector(&stream, fixture.shiftVec);
    writeVector(&stream, fixture.nbfp);
    writeVector(&stream, fixture.nbfpComb);

    writeVector(&stream, fixture.pairlist.sci);
    writeVector(&stream, fixture.pairlist.cjPacked);
    writeVector(&stream, fixture.pairlist.excl);
}

} // namespace sponge::nbnxm

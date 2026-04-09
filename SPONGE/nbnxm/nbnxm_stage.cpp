#include "nbnxm_stage.h"

#include <fstream>
#include <stdexcept>
#include <string>

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
    std::uint64_t size = 0U;
    readScalar(stream, &size);
    values->resize(static_cast<std::size_t>(size));
    if (!values->empty())
    {
        stream->read(reinterpret_cast<char*>(values->data()),
                     static_cast<std::streamsize>(sizeof(T) * values->size()));
    }
}

std::ofstream openOutput(const std::filesystem::path& path)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error("failed to open stage file for writing: " + path.string());
    }
    return stream;
}

std::ifstream openInput(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to open stage file for reading: " + path.string());
    }
    return stream;
}

void validateStageHeader(const StageFileHeader& header, StageKind expectedKind)
{
    if (header.magic != c_stageMagic)
    {
        throw std::runtime_error("stage magic mismatch");
    }
    if (header.version != c_stageVersion)
    {
        throw std::runtime_error("unsupported stage version");
    }
    if (header.endian != c_stageEndian)
    {
        throw std::runtime_error("stage endian mismatch");
    }
    if (header.kind != expectedKind)
    {
        throw std::runtime_error("stage kind mismatch");
    }
}

void writeStageHeader(std::ofstream* stream, StageKind kind)
{
    StageFileHeader fileHeader;
    fileHeader.kind = kind;
    writeScalar(stream, fileHeader);
}

StageFileHeader readStageHeader(std::ifstream* stream, StageKind expectedKind)
{
    StageFileHeader fileHeader;
    readScalar(stream, &fileHeader);
    validateStageHeader(fileHeader, expectedKind);
    return fileHeader;
}

void validateGridDump(const StageGridDump& dump)
{
    if (dump.header.numShiftVec != dump.shiftVec.size())
    {
        throw std::runtime_error("grid dump shift vector count mismatch");
    }
    if (!dump.numClustersPerBin.empty()
        && dump.numClustersPerBin.size() != dump.header.numBins
        && dump.numClustersPerBin.size() != dump.header.numBins + 1U)
    {
        throw std::runtime_error("grid dump numClustersPerBin size mismatch");
    }
}

void validatePairlistDump(const StagePairlistDump& dump)
{
    if (dump.header.numSci != dump.pairlist.sci.size()
        || dump.header.numPackedJClusters != dump.pairlist.cjPacked.size()
        || dump.header.numExcl != dump.pairlist.excl.size())
    {
        throw std::runtime_error("pairlist dump header/payload mismatch");
    }
}

} // namespace

void saveStageParams(const std::filesystem::path& path, const StageParamsDump& dump)
{
    auto stream = openOutput(path);
    writeStageHeader(&stream, StageKind::Params);
    writeScalar(&stream, dump.header);
}

StageParamsDump loadStageParams(const std::filesystem::path& path)
{
    auto stream = openInput(path);
    readStageHeader(&stream, StageKind::Params);
    StageParamsDump dump;
    readScalar(&stream, &dump.header);
    return dump;
}

void saveStageGrid(const std::filesystem::path& path, const StageGridDump& dump)
{
    validateGridDump(dump);

    auto stream = openOutput(path);
    writeStageHeader(&stream, StageKind::Grid);
    writeScalar(&stream, dump.header);
    writeVector(&stream, dump.shiftVec);
    writeVector(&stream, dump.atomIndices);
    writeVector(&stream, dump.bins);
    writeVector(&stream, dump.numAtomsPerCell);
    writeVector(&stream, dump.cellToBin);
    writeVector(&stream, dump.numClustersPerBin);
    writeVector(&stream, dump.packedBoundingBoxes);
}

StageGridDump loadStageGrid(const std::filesystem::path& path)
{
    auto stream = openInput(path);
    readStageHeader(&stream, StageKind::Grid);
    StageGridDump dump;
    readScalar(&stream, &dump.header);
    readVector(&stream, &dump.shiftVec);
    readVector(&stream, &dump.atomIndices);
    readVector(&stream, &dump.bins);
    readVector(&stream, &dump.numAtomsPerCell);
    readVector(&stream, &dump.cellToBin);
    readVector(&stream, &dump.numClustersPerBin);
    readVector(&stream, &dump.packedBoundingBoxes);
    validateGridDump(dump);
    return dump;
}

void saveStagePairlist(const std::filesystem::path& path, const StagePairlistDump& dump)
{
    validatePairlistDump(dump);

    auto stream = openOutput(path);
    writeStageHeader(&stream, StageKind::Pairlist);
    writeScalar(&stream, dump.header);
    writeVector(&stream, dump.pairlist.sci);
    writeVector(&stream, dump.pairlist.cjPacked);
    writeVector(&stream, dump.pairlist.excl);
}

StagePairlistDump loadStagePairlist(const std::filesystem::path& path)
{
    auto stream = openInput(path);
    readStageHeader(&stream, StageKind::Pairlist);
    StagePairlistDump dump;
    readScalar(&stream, &dump.header);
    readVector(&stream, &dump.pairlist.sci);
    readVector(&stream, &dump.pairlist.cjPacked);
    readVector(&stream, &dump.pairlist.excl);
    dump.pairlist.na_ci = c_clusterSize;
    dump.pairlist.na_cj = c_clusterSize;
    dump.pairlist.na_sc = c_superClusterSize;
    dump.pairlist.nci_tot = dump.header.nciTot;
    dump.pairlist.rlist = dump.header.rlist;
    validatePairlistDump(dump);
    return dump;
}

} // namespace sponge::nbnxm

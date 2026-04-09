#include "nbnxm_exclusions.h"

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

} // namespace

void saveExclusions(const std::filesystem::path& path, const ExclusionsDump& dump)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error("failed to open exclusions dump for writing: " + path.string());
    }

    writeScalar(&stream, dump.header);
    writeVector(&stream, dump.exclusionListStart);
    writeVector(&stream, dump.exclusionList);
    writeVector(&stream, dump.exclusionCounts);
}

ExclusionsDump loadExclusions(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to open exclusions dump for reading: " + path.string());
    }

    ExclusionsDump dump;
    readScalar(&stream, &dump.header);
    if (dump.header.magic != c_exclusionsMagic)
    {
        throw std::runtime_error("exclusions magic mismatch");
    }
    if (dump.header.version != c_exclusionsVersion)
    {
        throw std::runtime_error("unsupported exclusions version");
    }
    if (dump.header.endian != c_exclusionsEndian)
    {
        throw std::runtime_error("exclusions endian mismatch");
    }

    readVector(&stream, &dump.exclusionListStart);
    readVector(&stream, &dump.exclusionList);
    readVector(&stream, &dump.exclusionCounts);

    if (dump.exclusionListStart.size() != dump.header.numAtoms || dump.exclusionCounts.size() != dump.header.numAtoms
        || dump.exclusionList.size() != dump.header.numElements)
    {
        throw std::runtime_error("exclusions header/payload mismatch");
    }

    return dump;
}

} // namespace sponge::nbnxm

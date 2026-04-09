#include "nbnxm_force_dump.h"

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

void saveForces(const std::filesystem::path& path, const ForceDump& dump)
{
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error("failed to open force dump for writing: " + path.string());
    }

    writeScalar(&stream, dump.header);
    writeVector(&stream, dump.forces);
}

ForceDump loadForces(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("failed to open force dump for reading: " + path.string());
    }

    ForceDump dump;
    readScalar(&stream, &dump.header);
    if (dump.header.magic != c_forceDumpMagic)
    {
        throw std::runtime_error("force dump magic mismatch");
    }
    if (dump.header.version != c_forceDumpVersion)
    {
        throw std::runtime_error("unsupported force dump version");
    }
    if (dump.header.endian != c_forceDumpEndian)
    {
        throw std::runtime_error("force dump endian mismatch");
    }

    readVector(&stream, &dump.forces);
    if (dump.forces.size() != dump.header.numAtoms)
    {
        throw std::runtime_error("force dump header/payload mismatch");
    }
    return dump;
}

} // namespace sponge::nbnxm

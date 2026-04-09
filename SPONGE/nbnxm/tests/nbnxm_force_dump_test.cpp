#include "../nbnxm_force_dump.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace
{

void expect(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

} // namespace

int main()
{
    namespace nbnxm = sponge::nbnxm;

    const std::filesystem::path tmpDir = std::filesystem::temp_directory_path() / "spg_nbnxm_force_dump_test";
    std::filesystem::create_directories(tmpDir);
    const auto path = tmpDir / "reference.force.bin";

    nbnxm::ForceDump dump;
    dump.header.numAtoms = 3U;
    std::snprintf(dump.header.label, sizeof(dump.header.label), "%s", "force-roundtrip");
    dump.forces = {
        { 1.0F, 2.0F, 3.0F },
        { -4.0F, 5.5F, -6.5F },
        { 0.0F, 0.25F, -0.75F },
    };

    nbnxm::saveForces(path, dump);
    const auto loaded = nbnxm::loadForces(path);

    expect(loaded.header.numAtoms == dump.header.numAtoms, "force header numAtoms mismatch");
    expect(std::string(loaded.header.label) == std::string(dump.header.label), "force header label mismatch");
    expect(loaded.forces == dump.forces, "force payload mismatch");

    std::filesystem::remove_all(tmpDir);
    std::cout << "nbnxm_force_dump_test: PASS\n";
    return 0;
}

#include "../nbnxm_fixture.h"
#include "../nbnxm_stage.h"

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

    expect(sizeof(nbnxm::StageFileHeader) == 24, "StageFileHeader size mismatch");
    expect(sizeof(nbnxm::StageParamsHeader) == 96, "StageParamsHeader size mismatch");
    expect(sizeof(nbnxm::StageGridHeader) == 96, "StageGridHeader size mismatch");
    expect(sizeof(nbnxm::StagePairlistHeader) == 84, "StagePairlistHeader size mismatch");
    expect(sizeof(nbnxm::LJEwaldForceSummary) == 48, "LJEwaldForceSummary size mismatch");
    expect(sizeof(nbnxm::FixtureHeader) == 336, "FixtureHeader size mismatch");

    std::cout << "nbnxm_layout_test: PASS\n";
    return 0;
}

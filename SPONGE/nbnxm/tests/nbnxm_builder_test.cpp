#include "../nbnxm_builder.h"

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

template <typename Fn>
void expectThrows(const char* message, Fn&& fn)
{
    try
    {
        fn();
    }
    catch (const std::runtime_error&)
    {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main()
{
    namespace nbnxm = sponge::nbnxm;

    nbnxm::OrthorhombicNoPruneInput input;
    input.xq = { { 0.0F, 0.1F, 0.2F, -0.8F }, { 0.3F, 0.4F, 0.5F, 0.4F } };
    input.atomTypes = { 1, 2 };
    input.box = { 3.0F, 4.0F, 5.0F };
    input.cutoff = 0.9F;
    input.rlist = 0.9F;
    input.exclusionListStart = { 0, 1 };
    input.exclusionList = { 1 };
    input.exclusionCounts = { 1, 0 };

    nbnxm::validateOrthorhombicNoPruneInput(input);
    const auto params = nbnxm::buildPairlistParamsOrthorhombicNoPrune(input, "tip3p-v1");
    expect(params.header.cutoff == 0.9F, "cutoff mismatch");
    expect(params.header.rlist == 0.9F, "rlist mismatch");
    expect(params.header.usePruning == 0U, "usePruning mismatch");
    expect(params.header.useTwinCut == 0U, "useTwinCut mismatch");

    auto badInput = input;
    badInput.rlist = 0.8F;
    expectThrows("rlist<cutoff should fail",
                 [&]() { nbnxm::validateOrthorhombicNoPruneInput(badInput); });

    badInput = input;
    badInput.box[2] = 0.0F;
    expectThrows("non-positive box should fail",
                 [&]() { nbnxm::validateOrthorhombicNoPruneInput(badInput); });

    badInput = input;
    badInput.atomTypes.pop_back();
    expectThrows("xq/type mismatch should fail",
                 [&]() { nbnxm::validateOrthorhombicNoPruneInput(badInput); });

    badInput = input;
    badInput.exclusionListStart = { 0 };
    expectThrows("bad exclusionListStart size should fail",
                 [&]() { nbnxm::validateOrthorhombicNoPruneInput(badInput); });

    std::cout << "nbnxm_builder_test: PASS\n";
    return 0;
}

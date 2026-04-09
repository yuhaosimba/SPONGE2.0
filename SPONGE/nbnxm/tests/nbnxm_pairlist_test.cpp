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

} // namespace

int main()
{
    namespace nbnxm = sponge::nbnxm;

    nbnxm::OrthorhombicNoPruneInput input;
    input.box = { 3.0F, 3.0F, 3.0F };
    input.cutoff = 0.9F;
    input.rlist = 0.9F;
    input.xq.push_back({ 0.1F, 0.1F, 0.1F, -0.8F });
    input.xq.push_back({ 0.2F, 0.1F, 0.1F, 0.4F });
    input.xq.push_back({ 0.3F, 0.1F, 0.1F, 0.4F });
    input.atomTypes = { 0, 1, 1 };

    const auto pairlist = nbnxm::buildPairlistOrthorhombicNoPrune(input);

    expect(pairlist.na_ci == 8, "na_ci mismatch");
    expect(pairlist.na_cj == 8, "na_cj mismatch");
    expect(pairlist.na_sc == 8, "na_sc mismatch");
    expect(pairlist.rlist == 0.9F, "rlist mismatch");
    expect(pairlist.nci_tot == 1, "expected one cluster pair");
    expect(pairlist.sci.size() == 1U, "expected one sci entry");
    expect(pairlist.cjPacked.size() == 1U, "expected one packed j entry");
    expect(pairlist.excl.size() == 3U, "expected default and two split exclusion masks");
    expect(pairlist.sci[0].sci == 0, "unexpected sci index");
    expect(pairlist.sci[0].shift == 22, "unexpected shift index");
    expect(pairlist.sci[0].cjPackedBegin == 0, "unexpected cjPackedBegin");
    expect(pairlist.sci[0].cjPackedEnd == 1, "unexpected cjPackedEnd");
    expect(pairlist.cjPacked[0].cj[0] == 0, "unexpected first j-cluster");
    expect(pairlist.cjPacked[0].imei[0].imask == 1U, "unexpected warp0 imask");
    expect(pairlist.cjPacked[0].imei[1].imask == 1U, "unexpected warp1 imask");
    expect(pairlist.cjPacked[0].imei[0].excl_ind == 1, "unexpected warp0 exclusion index");
    expect(pairlist.cjPacked[0].imei[1].excl_ind == 2, "unexpected warp1 exclusion index");
    expect((pairlist.excl[1].pair[0] & 1U) == 0U, "self interaction should be excluded");
    expect((pairlist.excl[1].pair[8] & 1U) != 0U, "upper triangle interaction should remain");

    std::cout << "nbnxm_pairlist_test: PASS\n";
    return 0;
}

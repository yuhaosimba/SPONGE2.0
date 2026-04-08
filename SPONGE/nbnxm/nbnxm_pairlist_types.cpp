#include "nbnxm_pairlist_types.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace sponge::nbnxm
{

namespace
{

template <typename T>
T* cloneToMirror(const std::vector<T>& source)
{
    if (source.empty())
    {
        return nullptr;
    }
    auto* ptr = static_cast<T*>(std::malloc(sizeof(T) * source.size()));
    if (ptr == nullptr)
    {
        throw std::bad_alloc();
    }
    std::memcpy(ptr, source.data(), sizeof(T) * source.size());
    return ptr;
}

} // namespace

void clearDeviceMirror(NbnxmPairlistDevice* device)
{
    if (device == nullptr)
    {
        return;
    }

    std::free(device->d_sci);
    std::free(device->d_cjPacked);
    std::free(device->d_excl);

    device->d_sci = nullptr;
    device->d_cjPacked = nullptr;
    device->d_excl = nullptr;
    device->numSci = 0;
    device->numPackedJClusters = 0;
    device->numExcl = 0;
    device->nci_tot = 0;
    device->rlist = 0.0f;
}

void uploadToDeviceMirror(const NbnxmPairlistHost& host, NbnxmPairlistDevice* device)
{
    if (device == nullptr)
    {
        return;
    }

    clearDeviceMirror(device);

    device->na_ci = host.na_ci;
    device->na_cj = host.na_cj;
    device->na_sc = host.na_sc;
    device->nci_tot = host.nci_tot;
    device->rlist = host.rlist;
    device->numSci = static_cast<int>(host.sci.size());
    device->numPackedJClusters = static_cast<int>(host.cjPacked.size());
    device->numExcl = static_cast<int>(host.excl.size());

    device->d_sci = cloneToMirror(host.sci);
    device->d_cjPacked = cloneToMirror(host.cjPacked);
    device->d_excl = cloneToMirror(host.excl);
}

} // namespace sponge::nbnxm

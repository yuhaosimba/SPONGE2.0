#pragma once

#include "nbnxm_frozen_compat.h"

namespace sponge::nbnxm::frozen
{

void launchLJEwaldFrozenKernel(const FrozenKernelContext& context, bool calcFshift);

} // namespace sponge::nbnxm::frozen

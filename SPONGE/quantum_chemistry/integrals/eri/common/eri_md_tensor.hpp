#pragma once

#include "../../one_e.hpp"

static __device__ void compute_hr_tensor(float* HR, float alpha, float PQ[3],
                                         int L_tot, int hr_base, float t_arg)
{
    // Use double Boys + double seeding to avoid (-2α)^n amplification
    double F_d[17];
    compute_boys_double(F_d, t_arg, L_tot);
    double m2a = -2.0 * (double)alpha;
    double fac = 1.0;
    for (int n = 0; n <= L_tot; n++)
    {
        HR[HR_IDX_RUNTIME(0, 0, 0, n, hr_base)] = (float)(fac * F_d[n]);
        fac *= m2a;
    }
    for (int N = 1; N <= L_tot; N++)
    {
        for (int t = 0; t <= N; t++)
        {
            for (int u = 0; u <= N - t; u++)
            {
                int v = N - t - u;
                int max_n = L_tot - N;
                for (int n = 0; n <= max_n; n++)
                {
                    float val = 0.0f;
                    if (t > 0)
                    {
                        val = PQ[0] *
                              HR[HR_IDX_RUNTIME(t - 1, u, v, n + 1, hr_base)];
                        if (t > 1)
                            val +=
                                (float)(t - 1) *
                                HR[HR_IDX_RUNTIME(t - 2, u, v, n + 1, hr_base)];
                    }
                    else if (u > 0)
                    {
                        val = PQ[1] *
                              HR[HR_IDX_RUNTIME(t, u - 1, v, n + 1, hr_base)];
                        if (u > 1)
                            val +=
                                (float)(u - 1) *
                                HR[HR_IDX_RUNTIME(t, u - 2, v, n + 1, hr_base)];
                    }
                    else if (v > 0)
                    {
                        val = PQ[2] *
                              HR[HR_IDX_RUNTIME(t, u, v - 1, n + 1, hr_base)];
                        if (v > 1)
                            val +=
                                (float)(v - 1) *
                                HR[HR_IDX_RUNTIME(t, u, v - 2, n + 1, hr_base)];
                    }
                    HR[HR_IDX_RUNTIME(t, u, v, n, hr_base)] = val;
                }
            }
        }
    }
}

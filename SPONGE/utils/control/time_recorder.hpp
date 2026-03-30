#pragma once

#include <chrono>

namespace Chrono = std::chrono;
using Clock = Chrono::high_resolution_clock;
using time_recorder_t = Chrono::time_point<Clock>;

struct TIME_RECORDER
{
   private:
    time_recorder_t start_timestamp;
    time_recorder_t end_timestamp;
    inline static bool strict_sync = false;

   public:
    double time = 0;

    static void Set_Strict_Sync(bool strict)
    {
        strict_sync = strict;
    }

    void Start()
    {
#ifdef GPU_ARCH_NAME
        if (strict_sync) hostDeviceSynchronize();
#endif
        start_timestamp = Clock::now();
    }

    void Stop()
    {
#ifdef GPU_ARCH_NAME
        if (strict_sync) hostDeviceSynchronize();
#endif
        end_timestamp = Clock::now();
        time += Chrono::duration_cast<Chrono::duration<double>>(end_timestamp -
                                                                start_timestamp)
                    .count();
    }

    void Clear()
    {
        time = 0;
        start_timestamp = time_recorder_t();
        end_timestamp = time_recorder_t();
    }
};

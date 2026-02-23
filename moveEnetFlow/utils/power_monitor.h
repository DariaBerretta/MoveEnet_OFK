#ifndef MOVEENETFLOW_POWER_MONITOR_H
#define MOVEENETFLOW_POWER_MONITOR_H

#include <string>
#include <unistd.h>

struct PowerMonitorConfig
{
    std::string powerjoular_file;

    std::string gpu_file;
    int gpu_period_ms{200};
    int gpu_index{0};

    pid_t target_pid{0};
};

class PowerMonitor
{
public:
    bool start(const PowerMonitorConfig &cfg);
    void stop();

    ~PowerMonitor();

private:
    bool startPowerJoular(const PowerMonitorConfig &cfg);
    bool startGpuMonitor(const PowerMonitorConfig &cfg);

    pid_t gpu_monitor_pid_{-1};
};

#endif

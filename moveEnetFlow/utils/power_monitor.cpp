#include "power_monitor.h"

#include <yarp/os/LogStream.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <signal.h>
#include <sstream>
#include <system_error>

namespace
{
bool commandExists(const char *command)
{
    std::ostringstream check_cmd;
    check_cmd << "command -v " << command << " >/dev/null 2>&1";
    return std::system(check_cmd.str().c_str()) == 0;
}
}

bool PowerMonitor::start(const PowerMonitorConfig &cfg)
{
    if (cfg.target_pid <= 0) {
        yError() << "Invalid target PID for monitor start:" << cfg.target_pid;
        return false;
    }

    if (!cfg.powerjoular_file.empty() && !startPowerJoular(cfg)) {
        return false;
    }

    if (!cfg.gpu_file.empty() && !startGpuMonitor(cfg)) {
        stop();
        return false;
    }

    return true;
}

void PowerMonitor::stop()
{
    if (gpu_monitor_pid_ > 0) {
        kill(gpu_monitor_pid_, SIGTERM);
        gpu_monitor_pid_ = -1;
    }
}

PowerMonitor::~PowerMonitor()
{
    stop();
}

bool PowerMonitor::startPowerJoular(const PowerMonitorConfig &cfg)
{
    if (!commandExists("powerjoular")) {
        yError() << "PowerJoular was requested but not found in PATH.";
        yError() << "Install it and retry, or run without --pwrjlr_file.";
        return false;
    }

    std::filesystem::path pj_path(cfg.powerjoular_file);
    if (!pj_path.parent_path().empty()) {
        std::filesystem::create_directories(pj_path.parent_path());
    }

    std::ostringstream pj_cmd;
    // PowerJoular sampling note:
    // v1.1.1 samples power about once per second (no configurable period option).
    // - `-l`: use linear regression model
    // - `-c`: write millisecond-resolution timestamps in CSV
    // - `-p`: bind monitor lifecycle/target to this PID
    // Output is saved by PowerJoular to files derived from `cfg.powerjoular_file`.
    pj_cmd << "powerjoular -l -c -p " << cfg.target_pid
           << " -f \"" << cfg.powerjoular_file << "\""
           << " >/dev/null 2>&1 &";
    const int pj_status = std::system(pj_cmd.str().c_str());
    if (pj_status != 0) {
        yError() << "Failed to start PowerJoular monitor.";
        return false;
    }

    yInfo() << "PowerJoular enabled -> base output:" << cfg.powerjoular_file
            << "(sampling period is fixed by PowerJoular, typically ~1 s)";
    return true;
}

bool PowerMonitor::startGpuMonitor(const PowerMonitorConfig &cfg)
{
    if (cfg.gpu_period_ms <= 0) {
        yError() << "gpu_period_ms must be > 0. Provided:" << cfg.gpu_period_ms;
        return false;
    }

    if (!commandExists("nvidia-smi")) {
        yError() << "GPU monitor was requested but nvidia-smi was not found in PATH.";
        yError() << "Run without --gpu_file or install NVIDIA utilities.";
        return false;
    }

    std::filesystem::path gpu_path(cfg.gpu_file);
    if (!gpu_path.parent_path().empty()) {
        std::filesystem::create_directories(gpu_path.parent_path());
    }

    std::filesystem::path gpu_pid_path = std::filesystem::temp_directory_path() /
                                         ("moveEnet_gpu_mon_" + std::to_string(cfg.target_pid) + ".pid");
    std::ostringstream gpu_cmd;
    // GPU telemetry sampled via nvidia-smi and saved as CSV (nounits):
    // timestamp,index,name,utilization.gpu,utilization.memory,power.draw,temperature.gpu
    // - utilization.gpu: GPU core utilization (%)
    // - utilization.memory: memory controller utilization (%)
    // - power.draw: instantaneous board power draw (W)
    // - temperature.gpu: GPU temperature (C)
    gpu_cmd << "sh -c 'nvidia-smi "
            << "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,power.draw,temperature.gpu "
            << "-i " << cfg.gpu_index << " "
            << "--format=csv,nounits -lms " << cfg.gpu_period_ms
            << " > \"" << cfg.gpu_file << "\" 2>/dev/null & echo $! > \"" << gpu_pid_path.string() << "\"'";

    const int gpu_status = std::system(gpu_cmd.str().c_str());
    if (gpu_status != 0) {
        yError() << "Failed to start nvidia-smi GPU monitor.";
        return false;
    }

    std::ifstream pid_in(gpu_pid_path);
    if (!(pid_in >> gpu_monitor_pid_) || gpu_monitor_pid_ <= 0) {
        yError() << "Failed to read nvidia-smi monitor PID.";
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(gpu_pid_path, ec);

    yInfo() << "GPU monitor enabled ->" << cfg.gpu_file
            << "(gpu index:" << cfg.gpu_index
            << ", period ms:" << cfg.gpu_period_ms << ")";
    return true;
}

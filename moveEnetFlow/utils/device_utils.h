#pragma once

#include <string>

struct DeviceConfig {
    bool use_gpu{false};
    int gpu_id{0};
};

DeviceConfig parseDeviceConfig(const std::string &device);
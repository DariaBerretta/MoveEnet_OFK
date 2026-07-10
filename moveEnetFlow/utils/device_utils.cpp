#include "device_utils.h"

#include <algorithm>
#include <cctype>

DeviceConfig parseDeviceConfig(const std::string &device)
{
    DeviceConfig cfg;

    if (device.empty()) {
        return cfg;
    }

    std::string dev_l = device;
    std::transform(dev_l.begin(), dev_l.end(), dev_l.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });

    if (dev_l.rfind("cpu", 0) == 0) {
        cfg.use_gpu = false;
        return cfg;
    }

    cfg.use_gpu = true;

    size_t colon = dev_l.find(':');
    if (colon != std::string::npos) {
        try {
            cfg.gpu_id = std::stoi(dev_l.substr(colon + 1));
        } catch (...) {
            cfg.gpu_id = 0;
        }
    } else if (std::all_of(dev_l.begin(), dev_l.end(),
                          [](unsigned char c) {
                              return std::isdigit(c);
                          })) {
        try {
            cfg.gpu_id = std::stoi(dev_l);
        } catch (...) {
            cfg.gpu_id = 0;
        }
    }

    return cfg;
}
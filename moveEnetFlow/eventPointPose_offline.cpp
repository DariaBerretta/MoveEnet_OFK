// EventPointPose offline runner - PointNet/RasEPC integration.
//
// This file fills the YARP-side integration that was missing in the uploaded
// eventPointPose_offline.cpp. The Python sidecar performs the actual PyTorch
// inference and rasterized event-point-cloud preprocessing. C++ keeps the same
// offline-pipeline responsibility: read event windows, send them to the model,
// receive hpe-core skeleton13 predictions, write CSV, and optionally visualize.

#include <yarp/os/all.h>
#include <yarp/os/Bottle.h>
#include <event-driven/core.h>
#include <event-driven/algs.h>
#include <event-driven/vis.h>
#include <hpe-core/utility.h>
#include <opencv2/opencv.hpp>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "utils/power_monitor.h"
#include "utils/visualization_utils.h"

using namespace yarp::os;
using std::string;
using yarp::os::Value;

namespace {

struct EventPoint {
    double x{0.0};
    double y{0.0};
    double t{0.0};
    double p{0.0};
};

// CSV order: match moveEnetOFK_offline (joint index order 0..12)
static const std::array<int, 13> CSV_HPE_ORDER = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
};

static const std::array<const char*, 13> CSV_JOINT_NAMES = {
    "joint0",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "joint8",
    "joint9",
    "joint10",
    "joint11",
    "joint12"
};

static std::string shellQuote(const std::string &s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static cv::Mat eventsToFrame(const std::vector<EventPoint> &events, const cv::Size &res) {
    // Accumulate per-pixel event counts then normalize using a 3-sigma rule
    cv::Mat counts(res, CV_64F, cv::Scalar(0));
    for (const auto &e : events) {
        int x = static_cast<int>(std::lround(e.x));
        int y = static_cast<int>(std::lround(e.y));
        x = std::max(0, std::min(res.width - 1, x));
        y = std::max(0, std::min(res.height - 1, y));
        counts.at<double>(y, x) += 1.0;
    }

    // If no events, return empty (black) frame
    double sum = 0.0;
    int nz = 0;
    for (int r = 0; r < counts.rows; ++r) {
        const double *row = counts.ptr<double>(r);
        for (int c = 0; c < counts.cols; ++c) {
            double v = row[c];
            if (v > 0.0) {
                sum += v;
                ++nz;
            }
        }
    }
    if (nz == 0) {
        return cv::Mat(res, CV_8UC1, cv::Scalar(0));
    }

    double mean = sum / nz;
    double var_sum = 0.0;
    for (int r = 0; r < counts.rows; ++r) {
        const double *row = counts.ptr<double>(r);
        for (int c = 0; c < counts.cols; ++c) {
            double v = row[c];
            if (v > 0.0) {
                double d = v - mean;
                var_sum += d * d;
            }
        }
    }
    double var = (nz > 1) ? (var_sum / (nz - 1)) : var_sum;
    double sig = std::sqrt(var);
    if (sig < (0.1 / 255.0)) {
        sig = 0.1 / 255.0;
    }

    const double numSDevs = 3.0;
    const double range_old = numSDevs * sig; // scaling factor
    const double range_new = 255.0;

    cv::Mat out(res, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < counts.rows; ++r) {
        const double *inrow = counts.ptr<double>(r);
        unsigned char *outrow = out.ptr<unsigned char>(r);
        for (int c = 0; c < counts.cols; ++c) {
            double v = inrow[c];
            if (v <= 0.0) {
                outrow[c] = 0;
            } else {
                double f = (v * range_new) / range_old;
                if (f > 255.0) f = 255.0;
                if (f < 0.0) f = 0.0;
                outrow[c] = static_cast<unsigned char>(std::floor(f));
            }
        }
    }
    return out;
}

} // namespace

class eventPointPose_Detector
{
private:
    BufferedPort<Bottle> output_port;
    BufferedPort<Bottle> input_port;

    double period{0.05};
    double tic{0.0};
    bool waiting{false};
    bool launched_python{false};

    std::string local_events_out;
    std::string local_sklt_in;
    std::string epp_events_in{ "/eventPointPose/events:i" };
    std::string epp_sklt_out{ "/eventPointPose/sklt:o" };
    std::string pid_file;

public:
    bool init(const std::string &output_name,
              const std::string &input_name,
              const std::string &checkpoint_path,
              const std::string &script_path,
              double rate,
              const cv::Size &sensor_size,
              int num_points,
              const std::string &device,
              bool swap_lr)
    {
        local_events_out = output_name;
        local_sklt_in = input_name;
        period = (rate > 0.0) ? 1.0 / rate : 0.05;
        pid_file = "/tmp/eventpointpose_offline_python_" + std::to_string(::getpid()) + ".pid";

        if (!yarp::os::Network::checkNetwork(2.0)) {
            yError() << "Could not connect to YARP. Start yarpserver first.";
            return false;
        }

        // Clean stale local registrations from a previous crashed run.
        yarp::os::Network::unregisterName(local_events_out);
        yarp::os::Network::unregisterName(local_sklt_in);

        if (!output_port.open(local_events_out)) {
            yError() << "Could not open EventPointPose event output port:" << local_events_out;
            return false;
        }
        if (!input_port.open(local_sklt_in)) {
            yError() << "Could not open EventPointPose skeleton input port:" << local_sklt_in;
            output_port.close();
            return false;
        }

        if (!script_path.empty()) {
            std::ostringstream cmd;
            cmd << "python3 " << shellQuote(script_path)
                << " --checkpoint " << shellQuote(checkpoint_path)
                << " --events_port " << epp_events_in
                << " --sklt_port " << epp_sklt_out
                << " --sensor_w " << sensor_size.width
                << " --sensor_h " << sensor_size.height
                << " --num_points " << num_points
                << " --input_coord_base zero";
            if (swap_lr) {
                cmd << " --swap_lr";
            }
            if (!device.empty()) {
                cmd << " --device " << shellQuote(device);
            }
            cmd << " & echo $! > " << pid_file;

            int r = std::system(cmd.str().c_str());
            if (r != 0) {
                yError() << "Could not launch EventPointPose sidecar:" << cmd.str();
                close();
                return false;
            }
            launched_python = true;
        }

        bool ports_ready = false;
        for (int i = 0; i < 100; ++i) {
            if (yarp::os::NetworkBase::exists(epp_events_in) &&
                yarp::os::NetworkBase::exists(epp_sklt_out)) {
                ports_ready = true;
                break;
            }
            yarp::os::Time::delay(0.1);
        }
        if (!ports_ready) {
            yError() << "EventPointPose sidecar ports not found. Expected" << epp_events_in << "and" << epp_sklt_out;
            close();
            return false;
        }

        if (!yarp::os::Network::connect(local_events_out, epp_events_in, "fast_tcp")) {
            yError() << "Could not connect" << local_events_out << "->" << epp_events_in;
            close();
            return false;
        }
        if (!yarp::os::Network::connect(epp_sklt_out, local_sklt_in, "fast_tcp")) {
            yError() << "Could not connect" << epp_sklt_out << "->" << local_sklt_in;
            close();
            return false;
        }

        yInfo() << "EventPointPose PointNet sidecar connected.";
        return true;
    }

    void close()
    {
        yarp::os::Network::disconnect(local_events_out, epp_events_in);
        yarp::os::Network::disconnect(epp_sklt_out, local_sklt_in);
        output_port.close();
        input_port.close();

        if (launched_python) {
            std::ostringstream cmd;
            cmd << "if [ -f " << pid_file << " ]; then "
                << "kill $(cat " << pid_file << ") 2>/dev/null; "
                << "rm -f " << pid_file << "; fi";
            std::system(cmd.str().c_str());
            launched_python = false;
        }
    }

    bool update(const std::vector<EventPoint> &latest_events, double latest_ts, hpecore::stampedPose &previous_skeleton)
    {
        if (latest_events.empty()) {
            return false;
        }

        if (latest_ts < tic) {
            tic = latest_ts - 2.0;
            waiting = false;
        }

        if ((!waiting && latest_ts - tic >= period) || (latest_ts - tic > 2.0)) {
            Bottle &b = output_port.prepare();
            b.clear();
            b.addFloat64(latest_ts);
            Bottle &payload = b.addList();
            for (const auto &e : latest_events) {
                payload.addFloat64(e.x);
                payload.addFloat64(e.y);
                payload.addFloat64(e.t);
                payload.addFloat64(e.p);
            }
            output_port.write();
            tic = latest_ts;
            waiting = true;
        }

        Bottle *mn_container = input_port.read(false);
        if (mn_container) {
            previous_skeleton.pose = hpecore::extractSkeletonFromYARP<Bottle>(*mn_container);
            previous_skeleton.conf = hpecore::extractConfidenceFromYARP<Bottle>(*mn_container);
            previous_skeleton.timestamp = tic;
            previous_skeleton.delay = latest_ts - tic;
            waiting = false;
        }

        return mn_container != nullptr;
    }
};

// -----------------------------------------------------------------------------
// Event window adapter for EventPointPose / RasEPC
// -----------------------------------------------------------------------------
// EventPointPose PointNet is trained on a fixed-size random sample taken AFTER
// rasterization, not on a fixed number of raw events. For online single-camera
// inference, the closest setting to the paper's LastLabel setup is a rolling
// window of 7500 raw events, then RasEPC rasterization in Python, then random
// sampling to 2048 rasterized points.
//
// This local event-driven loader exposes packet-level timestamps only through
// iterator::timestamp(). If ev::AE has no per-event timestamp, packet timestamp
// collapses many events into the same temporal slice. To avoid that, we send a
// monotonic event-order timestamp. The Python sidecar normalizes timestamps
// within each window, so only their relative order is needed for the 4 RasEPC
// temporal slices.
static bool readNextEventWindow(ev::offlineLoader<ev::AE> &eloader,
                                double dt,
                                std::size_t raw_events_per_window,
                                std::vector<EventPoint> &events,
                                double &tnow)
{
    static bool first_call = true;
    static double next_ts = 0.0;
    static std::deque<EventPoint> rolling_events;
    static std::uint64_t event_order = 0;

    if (raw_events_per_window == 0) {
        raw_events_per_window = 7500;
    }

    if (first_call) {
        first_call = false;
        next_ts = dt;
    } else {
        next_ts += dt;
    }

    events.clear();

    if (!eloader.incrementReadTill(next_ts)) {
        return false;
    }

    std::size_t appended = 0;
    for (auto it = eloader.begin(); it != eloader.end(); ++it) {
        const ev::AE &ae = *it;

        EventPoint e;
        // Send zero-based sensor coordinates. The Python sidecar is called with
        // --input_coord_base zero and will not subtract 1. This removes the
        // unsafe min(x)>=1 auto-detection used previously.
        e.x = static_cast<double>(ae.x);
        e.y = static_cast<double>(ae.y);

        // Prefer event order over packet timestamp because the installed loader
        // exposes only packet time here. RasEPC only needs monotonic relative
        // time before normalization to [0,1].
        e.t = static_cast<double>(event_order++);

        // Keep polarity as 0/1. Python converts it to -1/+1 and accumulates
        // pacc exactly as in the EventPointPose rasterizer.
        e.p = (ae.p > 0) ? 1.0 : 0.0;

        rolling_events.push_back(e);
        ++appended;
        // discarding the oldest events keeps the window size fixed and ensures that the most recent
        while (rolling_events.size() > raw_events_per_window) {
            rolling_events.pop_front();
        }
    }

    tnow = next_ts;

    // Warm up until the rolling LastLabel-style window is full. Returning true
    // keeps the main loop advancing without sending an under-populated cloud.
    if (rolling_events.size() < raw_events_per_window) {
        return true;
    }

    events.assign(rolling_events.begin(), rolling_events.end());
    return appended > 0 || !events.empty();
}

int main(int argc, char *argv[])
{
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    if (rf.check("help")) {
        std::stringstream ss;
        ss << "Usage: eventPointPose_offline [options]\n\n";
        ss << std::left << std::setw(24) << "--data_file"       << std::setw(12) << "<string>" << ": path to event data.log\n";
        ss << std::left << std::setw(24) << "--checkpoint_path" << std::setw(12) << "<string>" << ": path to EventPointPose PointNet model.pth\n";
        ss << std::left << std::setw(24) << "--epp_script"      << std::setw(12) << "<string>" << ": path to eventPointPose_yarp_server.py\n";
        ss << std::left << std::setw(24) << "--num_points"      << std::setw(12) << "<int>"    << ": sampled RasEPC points, default 2048\n";
        ss << std::left << std::setw(24) << "--raw_events_per_window" << std::setw(12) << "<int>" << ": raw events before RasEPC, default 7500\n";
        ss << std::left << std::setw(24) << "--net_period"      << std::setw(12) << "<double>" << ": model update period, default 0.05\n";
        ss << std::left << std::setw(24) << "--output_period"   << std::setw(12) << "<double>" << ": CSV/vis period, default 0.005\n";
        ss << std::left << std::setw(24) << "--w"               << std::setw(12) << "<int>"    << ": DHP19 sensor width, default 346\n";
        ss << std::left << std::setw(24) << "--h"               << std::setw(12) << "<int>"    << ": DHP19 sensor height, default 260\n";
        ss << std::left << std::setw(24) << "--device"          << std::setw(12) << "<string>" << ": PyTorch device, e.g. cpu/cuda:0\n";
        ss << std::left << std::setw(24) << "--vis"             << std::setw(12) << ""         << ": enable visualization\n";
        ss << std::left << std::setw(24) << "--output_csv"      << std::setw(12) << "<string>" << ": output CSV path\n";
        ss << std::left << std::setw(24) << "--no_csv"          << std::setw(12) << ""         << ": skip CSV logging\n";
        ss << std::left << std::setw(24) << "--output_video"    << std::setw(12) << "<string>" << ": output .mp4 path\n";
        ss << std::left << std::setw(24) << "--no_video"        << std::setw(12) << ""         << ": disable video output\n";
        ss << std::left << std::setw(24) << "--swap_lr"         << std::setw(12) << ""         << ": swap left/right joints\n";
        yInfo() << ss.str();
        return 0;
    }

    const std::string datapath_file = rf.check("data_file", Value("/data/dhp19_testing_set_S13toS17/S13_1_1/ch3dvs/data.log")).asString();
    const std::string checkpoint_path = rf.check("checkpoint_path", Value("/home/model_mounts/eventpointpose/PointNet/models/model.pth")).asString();
    const std::string epp_script = rf.check("epp_script", Value("/home/model_mounts/eventpointpose/PointNet/models/eventPointPose_yarp_server.py")).asString();
    const std::string device = rf.check("device", Value("")).asString();
    const int num_points = rf.check("num_points", Value(2048)).asInt32();
    const int raw_events_per_window = rf.check("raw_events_per_window", Value(7500)).asInt32();
    const bool swap_lr = rf.check("swap_lr");

    const double output_period = rf.check("output_period", Value(0.005)).asFloat64();
    const double net_period = rf.check("net_period", Value(0.02)).asFloat64();
    const cv::Size res(rf.check("w", Value(346)).asInt32(), rf.check("h", Value(260)).asInt32());
    const bool is_visualize = rf.check("vis");
    const std::string output_csv = rf.check("output_csv", Value("/tmp/eventpointpose_output.csv")).asString();
    const bool no_csv = rf.check("no_csv");
    std::string output_video = rf.check("output_video", Value("")).asString();
    const bool no_video = rf.check("no_video");
    
    // For debugging: limit the number of processed event windows to avoid long runs on large datasets. Set max_packets to -1 to disable.
    const int max_packets = rf.check("max_packets", Value(-1)).asInt32();
    yInfo() << "max_packets set to" << max_packets;

    PowerMonitor power_monitor;
    PowerMonitorConfig power_cfg;
    power_cfg.powerjoular_file = rf.check("pwrjlr_file", Value("")).asString();
    power_cfg.gpu_file = rf.check("gpu_file", Value("/tmp/eventpointpose_gpu.csv")).asString();
    power_cfg.gpu_period_ms = rf.check("gpu_period_ms", Value(5)).asInt32();
    int gpu_monitor_index = rf.check("gpu_index", Value(0)).asInt32();

    // If a device was specified (e.g. cuda:0), align the power monitor index to it
    if (!device.empty()) {
        std::string dev = device;
        std::string dev_l = dev;
        std::transform(dev_l.begin(), dev_l.end(), dev_l.begin(), ::tolower);
        if (dev_l.rfind("cpu", 0) != 0) {
            int parsed_gpu = 0;
            size_t colon = dev.find(':');
            if (colon != std::string::npos) {
                std::string tail = dev.substr(colon + 1);
                try { parsed_gpu = std::stoi(tail); } catch (...) { parsed_gpu = 0; }
            } else if (!dev.empty() && std::all_of(dev.begin(), dev.end(), ::isdigit)) {
                try { parsed_gpu = std::stoi(dev); } catch (...) { parsed_gpu = 0; }
            }
            gpu_monitor_index = parsed_gpu;
        }
    }

    power_cfg.gpu_index = gpu_monitor_index;
    power_cfg.target_pid = ::getpid();
    if (!power_monitor.start(power_cfg)) {
        return -1;
    }

    std::ofstream csv_file;
    std::vector<std::string> csv_buffer;
    VisualizationContext vis_ctx;

    if (!no_csv) {
        csv_file.open(output_csv);
        if (!csv_file.is_open()) {
            yError() << "Could not open CSV file for writing:" << output_csv;
            return -1;
        }
        csv_file << "timestamp,latency";
        for (int j = 0; j < 13; ++j) {
            csv_file << "," << CSV_JOINT_NAMES[j] << "_x," << CSV_JOINT_NAMES[j] << "_y";
        }
        csv_file << "\n";
        csv_file.flush();
    }

    if (!initialiseVisualization(vis_ctx, res, is_visualize, no_video, output_video, datapath_file, output_period)) {
        return -1;
    }

    ev::offlineLoader<ev::AE> eloader;
    yInfo() << "Loading event data:" << datapath_file;
    if (!eloader.load(datapath_file)) {
        yError() << "Could not open event data file";
        return -1;
    }
    yInfo() << eloader.getinfo();

    if (!yarp::os::Network::checkNetwork()) {
        yError() << "Could not connect to YARP";
        return -1;
    }

    const std::string pointclouds_out = "/eventPointPose_offline/events:o_" + std::to_string(::getpid());
    const std::string eventpointpose_in = "/eventPointPose_offline/sklt:i_" + std::to_string(::getpid());

    eventPointPose_Detector epp_handler;
    const double detF = 1.0 / net_period;
    if (!epp_handler.init(pointclouds_out, eventpointpose_in, checkpoint_path, epp_script, detF, res, num_points, device, swap_lr)) {
        yError() << "EventPointPose detector initialization failed";
        return -1;
    }

    hpecore::stampedPose detected_pose;
    hpecore::skeleton13 network_pose{};
    bool pose_initialised = false;

    double tnow = 0.0;
    double next_csv_upd = output_period;
    double next_vis_upd = output_period;
    std::vector<EventPoint> event_window;
    int packet_count = 0;

    while (readNextEventWindow(eloader, net_period, static_cast<std::size_t>(std::max(1, raw_events_per_window)), event_window, tnow)) {
        packet_count++;

        // For debugging: stop after a certain number of packets to avoid long runs on large datasets. Set max_packets to -1 to disable.
        if (max_packets > 0 && packet_count > max_packets) {
            yInfo() << "Reached max_packets limit of" << max_packets << ", stopping early.";
            break;
        }

        if (event_window.empty()) {
            continue;
        }

        const bool was_detected = epp_handler.update(event_window, tnow, detected_pose);
        if (was_detected && hpecore::poseNonZero(detected_pose.pose)) {
            network_pose = detected_pose.pose;
            pose_initialised = true;
        }

        if (tnow >= next_csv_upd) {
            next_csv_upd += output_period;
            if (pose_initialised && csv_file.is_open()) {
                std::ostringstream row;
                row << std::fixed << std::setprecision(6) << tnow;
                // detected_pose.timestamp is the skeleton timestamp (tic), while tnow is the event window timestamp (toc). Latency is toc-tic, i.e. how old the skeleton prediction is at the current event window time.
                const double lat = (detected_pose.timestamp > 0.0) ? detected_pose.delay : 0.0;
                row << "," << lat;
                for (int j = 0; j < 13; ++j) {
                    const int hpe_idx = CSV_HPE_ORDER[j];
                    row << "," << network_pose[hpe_idx].u << "," << network_pose[hpe_idx].v;
                }
                csv_buffer.push_back(row.str());
            }
        }

        if ((is_visualize || (!output_video.empty() && !no_video)) && tnow >= next_vis_upd) {
            next_vis_upd += output_period;

            cv::Mat frame = eventsToFrame(event_window, res);

            renderVisualizationFrameEPP(vis_ctx, frame, pose_initialised, network_pose, detected_pose, tnow);
            writeVisualizationFrame(vis_ctx, pose_initialised);
            if (showVisualizationFrame(vis_ctx)) {
                yInfo() << "User requested stop";
                break;
            }
        }
    }

    yInfo() << "End of event stream after" << packet_count << "event windows.";

    if (csv_file.is_open()) {
        for (const auto &line : csv_buffer) {
            csv_file << line << "\n";
        }
        csv_file.close();
        yInfo() << "CSV rows written:" << csv_buffer.size() << "->" << output_csv;
    }

    power_monitor.stop();
    epp_handler.close();
    closeVisualization(vis_ctx, output_video);
    return 0;
}

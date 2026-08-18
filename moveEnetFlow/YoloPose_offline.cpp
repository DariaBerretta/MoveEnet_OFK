// YoloPose offline runner.
//
// YOLO inference rate and output rate are independent:
//   - --net_period controls how often a video frame is sent to YOLO.
//   - --output_period controls how often the latest valid pose is written.
//
// Between two YOLO inferences, the latest valid pose is held using a
// zero-order hold. The CSV format is intentionally unchanged.

#include <yarp/os/all.h>
#include <yarp/cv/Cv.h>
#include <yarp/sig/Image.h>

#include <opencv2/opencv.hpp>

#include <hpe-core/utility.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <tuple>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <chrono>
#include <limits>

#include "utils/power_monitor.h"
#include "utils/visualization_utils.h"

using namespace yarp::os;
using namespace yarp::sig;
using std::string;
using yarp::os::Value;

namespace {

// CSV order: match moveEnetOFK_offline (joint index order 0..12).
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

static std::string shellQuote(const std::string &s)
{
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

} // namespace

class offlineDetector
{
private:
    cv::Size sensor_size;

    // Port used to send one selected image to the Python YOLO process.
    BufferedPort<ImageOf<PixelMono>> output_port;

    // Port used to receive the pose produced for that image.
    BufferedPort<Bottle> input_port;

    bool launched_python{false};

    std::string local_img_out{"/YoloPose_offline/img:o"};
    std::string local_sklt_in{"/YoloPose_offline/sklt:i"};
    std::string yolo_img_in{"/YoloPose/img:i"};
    std::string yolo_sklt_out{"/YoloPose/sklt:o"};
    std::string pid_file{"/tmp/yolopose_offline_python.pid"};

public:
    bool init(const std::string &model_path,
              const std::string &script_path,
              const cv::Size &image_size,
              const std::string &device = std::string())
    {
        sensor_size = image_size;

        if (!yarp::os::Network::checkNetwork(2.0)) {
            yError() << "Could not connect to YARP. Start yarpserver first.";
            return false;
        }

        if (!output_port.open(local_img_out)) {
            yError() << "Could not open image output port:" << local_img_out;
            return false;
        }

        if (!input_port.open(local_sklt_in)) {
            yError() << "Could not open skeleton input port:" << local_sklt_in;
            output_port.close();
            return false;
        }

        // Start the Python Ultralytics/YARP sidecar. If script_path is empty,
        // assume that the sidecar is already running in another terminal.
        if (!script_path.empty()) {
            std::ostringstream cmd;
            cmd << "python3 " << shellQuote(script_path)
                << " --model " << shellQuote(model_path)
                << " --w " << sensor_size.width
                << " --h " << sensor_size.height
                << " --img_port " << yolo_img_in
                << " --sklt_port " << yolo_sklt_out;

            if (!device.empty()) {
                std::string dev = device;
                dev.erase(dev.begin(), std::find_if(dev.begin(), dev.end(),
                    [](unsigned char ch) { return !std::isspace(ch); }));
                dev.erase(std::find_if(dev.rbegin(), dev.rend(),
                    [](unsigned char ch) { return !std::isspace(ch); }).base(), dev.end());

                if (dev == "cpu" || dev.rfind("cpu", 0) == 0) {
                    cmd << " --device cpu";
                } else if (dev.rfind("cuda:", 0) == 0) {
                    const std::string gpu_id = dev.substr(5);
                    cmd << " --device " << shellQuote(gpu_id);
                } else {
                    cmd << " --device " << shellQuote(dev);
                }
            }

            cmd << " & echo $! > " << pid_file;

            const int result = std::system(cmd.str().c_str());
            if (result != 0) {
                yError() << "Could not launch YoloPose Python sidecar with command:"
                         << cmd.str();
                close();
                return false;
            }

            launched_python = true;
        }

        // Wait until the Python sidecar ports are visible.
        bool ports_ready = false;
        for (int i = 0; i < 50; ++i) {
            if (yarp::os::NetworkBase::exists(yolo_img_in) &&
                yarp::os::NetworkBase::exists(yolo_sklt_out)) {
                ports_ready = true;
                break;
            }
            yarp::os::Time::delay(0.1);
        }

        if (!ports_ready) {
            yError() << "YoloPose sidecar ports not found. Expected"
                     << yolo_img_in << "and" << yolo_sklt_out;
            close();
            return false;
        }

        if (!yarp::os::Network::connect(local_img_out, yolo_img_in, "fast_tcp")) {
            yError() << "Could not connect" << local_img_out << "->" << yolo_img_in;
            close();
            return false;
        }

        if (!yarp::os::Network::connect(yolo_sklt_out, local_sklt_in, "fast_tcp")) {
            yError() << "Could not connect" << yolo_sklt_out << "->" << local_sklt_in;
            close();
            return false;
        }

        yInfo() << "YoloPose sidecar connected.";
        return true;
    }

    void close()
    {
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

    // Execute exactly one blocking YOLO inference for latest_image.
    // Rate limiting is deliberately handled by main(), using video timestamps.
    bool infer(const cv::Mat &latest_image,
           double frame_ts,
           hpecore::stampedPose &detected_skeleton,
           bool &response_received)
    {
        response_received = false;
        detected_skeleton.delay =
            std::numeric_limits<double>::quiet_NaN();

        if (latest_image.empty()) {
            return false;
        }

        // ===== PREPROCESSING: excluded from latency =====
        cv::Mat gray;

        if (latest_image.channels() == 3) {
            cv::cvtColor(latest_image, gray, cv::COLOR_BGR2GRAY);
        } else if (latest_image.channels() == 4) {
            cv::cvtColor(latest_image, gray, cv::COLOR_BGRA2GRAY);
        } else {
            latest_image.convertTo(gray, CV_8U);
        }

        if (gray.size() != sensor_size) {
            cv::resize(gray, gray, sensor_size);
        }

        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0, 0);

        ImageOf<PixelMono> &out_img = output_port.prepare();
        out_img.copy(yarp::cv::fromCvMat<PixelMono>(gray));

        // ===== LATENCY START =====
        const auto request_start =
            std::chrono::steady_clock::now();

        output_port.write();

        Bottle *mn_container = input_port.read(true);

        if (!mn_container) {
            const auto request_end =
                std::chrono::steady_clock::now();

            detected_skeleton.delay =
                std::chrono::duration<double>(
                    request_end - request_start
                ).count();

            detected_skeleton.timestamp = frame_ts;

            return false;
        }

        response_received = true;

        detected_skeleton.pose =
            hpecore::extractSkeletonFromYARP<Bottle>(
                *mn_container
            );

        detected_skeleton.conf =
            hpecore::extractConfidenceFromYARP<Bottle>(
                *mn_container
            );

        detected_skeleton.timestamp = frame_ts;

        // ===== LATENCY END =====
        const auto request_end =
            std::chrono::steady_clock::now();

        detected_skeleton.delay =
            std::chrono::duration<double>(
                request_end - request_start
            ).count();

        return hpecore::poseNonZero(
            detected_skeleton.pose
        );
    }
};

int main(int argc, char *argv[])
{
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    if (rf.check("help")) {
        std::stringstream ss;
        ss << "Usage: YoloPose_offline [options]\n\n";
        ss << "Options:\n";
        ss << std::left << std::setw(24) << "--data_file"       << std::setw(12) << "<string>" << ": path to input video file (.mp4)\n";
        ss << std::left << std::setw(24) << "--output_period"   << std::setw(12) << "<double>" << ": held-pose CSV/video output period (s), default 0.02\n";
        ss << std::left << std::setw(24) << "--net_period"      << std::setw(12) << "<double>" << ": interval between YOLO inferences (s), default 0.05\n";
        ss << std::left << std::setw(24) << "--h"               << std::setw(12) << "<int>"    << ": image height (default 480)\n";
        ss << std::left << std::setw(24) << "--w"               << std::setw(12) << "<int>"    << ": image width (default 640)\n";
        ss << std::left << std::setw(24) << "--vis"             << std::setw(12) << ""         << ": enable on-screen visualization\n";
        ss << std::left << std::setw(24) << "--output_csv"      << std::setw(12) << "<string>" << ": path to output CSV file\n";
        ss << std::left << std::setw(24) << "--no_csv"          << std::setw(12) << ""         << ": skip CSV logging\n";
        ss << std::left << std::setw(24) << "--output_video"    << std::setw(12) << "<string>" << ": path to output video file (.mp4)\n";
        ss << std::left << std::setw(24) << "--no_video"        << std::setw(12) << ""         << ": disable video output\n";
        ss << std::left << std::setw(24) << "--yolo_model_path" << std::setw(12) << "<string>" << ": path to yolo26n-pose.pt\n";
        ss << std::left << std::setw(24) << "--YoloPose_script" << std::setw(12) << "<string>" << ": path to YoloPose_yarp_server.py\n";
        ss << std::left << std::setw(24) << "--latency_csv"     << std::setw(12) << "<string>" << ": per-inference latency CSV\n";
        ss << std::left << std::setw(24) << "--pwrjlr_file"     << std::setw(12) << "<string>" << ": base path for PowerJoular output\n";
        ss << std::left << std::setw(24) << "--gpu_file"        << std::setw(12) << "<string>" << ": output CSV for GPU telemetry\n";
        ss << std::left << std::setw(24) << "--gpu_period_ms"   << std::setw(12) << "<int>"    << ": nvidia-smi sampling period in ms\n";
        ss << std::left << std::setw(24) << "--gpu_index"       << std::setw(12) << "<int>"    << ": NVIDIA GPU index to monitor\n";
        ss << std::left << std::setw(24) << "--device"          << std::setw(12) << "<string>" << ": 'cpu', 'cuda:N', or GPU index\n";
        yInfo() << ss.str();
        return 0;
    }

    // ===== READ PARAMETERS =====
    const std::string datapath_file = rf.check(
        "data_file",
        Value("/data/eh36m_testing_set_S9S11/rgb/cam2_S9_Directions_1.mp4")
    ).asString();

    const double output_period = rf.check(
        "output_period",
        Value(0.02)
    ).asFloat64();

    const double net_period = rf.check(
        "net_period",
        Value(0.05)
    ).asFloat64();

    if (output_period <= 0.0) {
        yError() << "--output_period must be greater than zero. Received:"
                 << output_period;
        return -1;
    }

    if (net_period <= 0.0) {
        yError() << "--net_period must be greater than zero. Received:"
                 << net_period;
        return -1;
    }

    const cv::Size res(
        rf.check("w", Value(640)).asInt32(),
        rf.check("h", Value(480)).asInt32()
    );

    const bool is_visualize = rf.check("vis");
    const std::string output_csv = rf.check(
        "output_csv",
        Value("/tmp/YoloPose_output.csv")
    ).asString();
    const bool no_csv = rf.check("no_csv");
    std::string output_video = rf.check(
        "output_video",
        Value("")
    ).asString();
    const bool no_video = rf.check("no_video");
    const std::string yolo_model_path = rf.check(
        "yolo_model_path",
        Value("/workspace/model_mounts/YoloPose/yolo26n-pose.pt")
    ).asString();
    const std::string yolo_script_path = rf.check(
        "YoloPose_script",
        Value("/workspace/model_mounts/YoloPose/YoloPose_yarp_server.py")
    ).asString();
    const std::string latency_csv_path = rf.check(
        "latency_csv",
        Value("")
    ).asString();
    const std::string powerjoular_file = rf.check(
        "pwrjlr_file",
        Value("")
    ).asString();
    const std::string gpu_monitor_file = rf.check(
        "gpu_file",
        Value("")
    ).asString();
    const int gpu_monitor_period_ms = rf.check(
        "gpu_period_ms",
        Value(5)
    ).asInt32();
    int gpu_monitor_index = rf.check(
        "gpu_index",
        Value(0)
    ).asInt32();
    const std::string device = rf.check(
        "device",
        Value("")
    ).asString();

    // Align GPU telemetry index with the requested YOLO device.
    if (!device.empty()) {
        std::string dev = device;
        std::string dev_l = dev;
        std::transform(dev_l.begin(), dev_l.end(), dev_l.begin(), ::tolower);

        if (dev_l.rfind("cpu", 0) != 0) {
            int parsed_gpu = 0;
            const size_t colon = dev.find(':');

            if (colon != std::string::npos) {
                const std::string tail = dev.substr(colon + 1);
                try {
                    parsed_gpu = std::stoi(tail);
                } catch (...) {
                    parsed_gpu = 0;
                }
            } else if (!dev.empty() &&
                       std::all_of(dev.begin(), dev.end(), ::isdigit)) {
                try {
                    parsed_gpu = std::stoi(dev);
                } catch (...) {
                    parsed_gpu = 0;
                }
            }

            gpu_monitor_index = parsed_gpu;
        }
    }

    yInfo() << "Requested YOLO inference period:" << net_period << "s";
    yInfo() << "Requested YOLO inference rate:" << 1.0 / net_period << "Hz";
    yInfo() << "Requested held-pose output period:" << output_period << "s";
    yInfo() << "Requested held-pose output rate:" << 1.0 / output_period << "Hz";

    // ===== POWER MONITORING =====
    PowerMonitor power_monitor;
    PowerMonitorConfig power_cfg;
    power_cfg.powerjoular_file = powerjoular_file;
    power_cfg.gpu_file = gpu_monitor_file;
    power_cfg.gpu_period_ms = gpu_monitor_period_ms;
    power_cfg.gpu_index = gpu_monitor_index;
    power_cfg.target_pid = ::getpid();

    if (!power_monitor.start(power_cfg)) {
        return -1;
    }

    // ===== PREPARE CSV, VIDEO, AND VISUALIZATION RESOURCES =====
    std::ofstream csv_file;
    std::vector<std::string> csv_buffer;
    std::ofstream latency_file;
    std::vector<std::string> latency_buffer;
    VisualizationContext vis_ctx;

    // CSV format intentionally unchanged.
    if (!no_csv) {
        csv_file.open(output_csv);
        if (!csv_file.is_open()) {
            yError() << "Could not open CSV file for writing:" << output_csv;
            power_monitor.stop();
            return -1;
        }

        yInfo() << "CSV logging enabled ->" << output_csv;
        csv_file << "timestamp,latency";
        for (int j = 0; j < 13; ++j) {
            csv_file << "," << CSV_JOINT_NAMES[j] << "_x,"
                     << CSV_JOINT_NAMES[j] << "_y";
        }
        csv_file << "\n";
        csv_file.flush();
    }

    if (!latency_csv_path.empty()) {
        latency_file.open(latency_csv_path);

        if (!latency_file.is_open()) {
            yError() << "Could not open latency CSV:"
                    << latency_csv_path;
            return -1;
        }

        latency_file
            << "request_id,"
            << "dataset_timestamp,"
            << "scheduled_timestamp,"
            << "latency_ms,"
            << "response_received,"
            << "valid_pose\n";

        yInfo()
            << "Per-inference latency logging enabled ->"
            << latency_csv_path;
    }

    if (!initialiseVisualization(
            vis_ctx,
            res,
            is_visualize,
            no_video,
            output_video,
            datapath_file,
            output_period)) {
        power_monitor.stop();
        return -1;
    }

    // ===== OPEN VIDEO FILE =====
    yInfo() << "Loading video:" << datapath_file;
    cv::VideoCapture cap(datapath_file);
    if (!cap.isOpened()) {
        yError() << "Could not open video file:" << datapath_file;
        power_monitor.stop();
        closeVisualization(vis_ctx, output_video);
        return -1;
    }

    // ===== INITIALIZE YOLO SIDECAR =====
    offlineDetector yolo_handler;
    if (!yolo_handler.init(
            yolo_model_path,
            yolo_script_path,
            res,
            device)) {
        yError() << "YoloPose init failed";
        power_monitor.stop();
        closeVisualization(vis_ctx, output_video);
        return -1;
    }

    // ===== ALGORITHMIC STATE =====
    hpecore::stampedPose detected_pose;

    // Last valid YOLO pose. This is held between inferences and also when an
    // inference is executed but returns no valid person.
    hpecore::stampedPose held_pose;
    hpecore::skeleton13 filtered_pose{};
    bool pose_initialised = false;

    // Independent video-time schedules.
    double next_inference_ts = 0.0;
    double next_output_ts = 0.0;
    bool schedules_initialised = false;

    constexpr double time_epsilon = 1e-9;

    std::size_t inference_count = 0;
    std::size_t valid_inference_count = 0;
    double first_inference_ts = -1.0;
    double last_inference_ts = -1.0;

    int frame_count = 0;
    bool stop_requested = false;

    // Previous RGB frame is used when generating held outputs whose timestamp
    // lies between the previous frame and the current frame. This prevents
    // using a future RGB image for an earlier visualization timestamp.
    cv::Mat previous_frame;

    // Append one output sample. The CSV schema remains exactly the same.
    // For held samples, latency is the latency of the inference that produced
    // the currently held pose.
    auto emit_output = [&](double output_ts, const cv::Mat &visual_frame) -> bool {
        // Preserve the old behavior before the first valid detection: no pose
        // is written until a valid pose exists to hold.
        if (!pose_initialised) {
            return false;
        }

        if (csv_file.is_open()) {
            std::ostringstream row;
            row << std::fixed << std::setprecision(6) << output_ts;
            row << "," << held_pose.delay;

            for (int j = 0; j < 13; ++j) {
                const int hpe_idx = CSV_HPE_ORDER[j];
                row << "," << filtered_pose[hpe_idx].u
                    << "," << filtered_pose[hpe_idx].v;
            }

            csv_buffer.push_back(row.str());
        }

        if ((is_visualize || (!output_video.empty() && !no_video)) &&
            !visual_frame.empty()) {
            renderVisualizationFrameOP(
                vis_ctx,
                visual_frame,
                pose_initialised,
                filtered_pose,
                held_pose,
                output_ts
            );
            writeVisualizationFrame(vis_ctx, pose_initialised);

            if (showVisualizationFrame(vis_ctx)) {
                yInfo() << "User requested stop";
                return true;
            }
        }

        return false;
    };

    // ===== MAIN PROCESSING LOOP =====
    while (!stop_requested) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            yInfo() << "End of video reached after" << frame_count << "frames.";
            break;
        }

        ++frame_count;
        const double tnow = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;

        if (!schedules_initialised) {
            // Anchor both schedules to the timestamp of the first video frame.
            next_inference_ts = tnow;
            next_output_ts = tnow;
            schedules_initialised = true;
        }

        // 1. Produce every scheduled output strictly before the current frame.
        // These samples must use the pose and RGB frame available before tnow.
        const cv::Mat &held_visual_frame =
            previous_frame.empty() ? frame : previous_frame;

        while (next_output_ts < tnow - time_epsilon) {
            if (emit_output(next_output_ts, held_visual_frame)) {
                stop_requested = true;
                break;
            }
            next_output_ts += output_period;
        }

        if (stop_requested) {
            break;
        }

        // 2. Execute at most one YOLO inference for the current video frame,
        // only when the next net_period deadline has been reached.
        if (tnow + time_epsilon >= next_inference_ts) {

            const std::size_t request_id = inference_count;

            // Important: this is the deadline that caused
            // this inference to be executed.
            const double scheduled_timestamp = next_inference_ts;

            if (inference_count == 0) {
                first_inference_ts = tnow;
            }

            last_inference_ts = tnow;
            bool response_received = false;
            

            const bool valid_pose = yolo_handler.infer(
                frame,
                tnow,
                detected_pose,
                response_received
            );

            ++inference_count;

            if (valid_pose) {
                held_pose = detected_pose;
                filtered_pose = detected_pose.pose;
                pose_initialised = true;
                ++valid_inference_count;
            }

            // One row per actual network request.
            if (latency_file.is_open()) {
                std::ostringstream row;

                row << request_id
                    << ","
                    << std::fixed
                    << std::setprecision(9)
                    << tnow
                    << ","
                    << scheduled_timestamp
                    << ",";

                if (std::isnan(detected_pose.delay)) {
                    row << "nan";
                } else {
                    row << std::setprecision(6)
                        << detected_pose.delay * 1000.0;
                }

                row << ","
                    << (response_received ? 1 : 0)
                    << ","
                    << (valid_pose ? 1 : 0);

                latency_buffer.push_back(row.str());
            }
           
            // If YOLO returns an invalid pose, keep the previous valid pose.

            // Advance the absolute inference schedule. If the requested rate is
            // higher than the video frame rate, multiple deadlines may have
            // elapsed; only one inference can be performed per available frame.
            do {
                next_inference_ts += net_period;
            } while (next_inference_ts <= tnow + time_epsilon);
        }

        // 3. Produce outputs coincident with the current frame timestamp after
        // the possible inference. Therefore a newly computed pose is available
        // starting from its own frame timestamp, never before it.
        while (next_output_ts <= tnow + time_epsilon) {
            if (emit_output(next_output_ts, frame)) {
                stop_requested = true;
                break;
            }
            next_output_ts += output_period;
        }

        previous_frame = frame.clone();
    }

    // ===== CLEANUP =====
    if (csv_file.is_open()) {
        for (const auto &line : csv_buffer) {
            csv_file << line << "\n";
        }
        csv_file.close();
        yInfo() << "CSV rows written:" << csv_buffer.size() << "->" << output_csv;
    }

    if (latency_file.is_open()) {
        for (const auto &line : latency_buffer) {
            latency_file << line << "\n";
        }

        latency_file.close();

        yInfo()
            << "Latency rows written:"
            << latency_buffer.size()
            << "->"
            << latency_csv_path;
    }

    yInfo() << "YOLO inferences completed:" << inference_count;
    yInfo() << "YOLO valid inferences:" << valid_inference_count;

    if (inference_count > 1 && last_inference_ts > first_inference_ts) {
        const double observed_inference_rate =
            static_cast<double>(inference_count - 1) /
            (last_inference_ts - first_inference_ts);

        yInfo() << "Requested inference rate:" << 1.0 / net_period << "Hz";
        yInfo() << "Observed inference rate:" << observed_inference_rate << "Hz";
    }

    power_monitor.stop();
    yolo_handler.close();
    closeVisualization(vis_ctx, output_video);

    return 0;
}
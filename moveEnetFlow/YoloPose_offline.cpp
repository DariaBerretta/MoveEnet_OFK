// YoloPose offline runner.
// Output CSV bounded at 50Hz, .mp4 frame rate

#include <yarp/os/all.h>
#include <yarp/cv/Cv.h>                     // needed for yarp::cv::fromCvMat
#include <yarp/sig/Image.h>                 // needed for BufferedPort<ImageOf<PixelMono>>

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

#include "utils/power_monitor.h"
#include "utils/visualization_utils.h"

using namespace yarp::os;
using namespace yarp::sig;
using std::string;
using yarp::os::Value;

namespace {

// CSV order requested by the user:
//   Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist,
//   LHip, RHip, LKnee, RKnee, LAnkle, RAnkle.
// Values are indices inside hpe-core::skeleton13.
static const std::array<int, 13> CSV_HPE_ORDER = {
    hpecore::head,
    hpecore::shoulderL,
    hpecore::shoulderR,
    hpecore::elbowL,
    hpecore::elbowR,
    hpecore::handL,
    hpecore::handR,
    hpecore::hipL,
    hpecore::hipR,
    hpecore::kneeL,
    hpecore::kneeR,
    hpecore::footL,
    hpecore::footR
};

static const std::array<const char*, 13> CSV_JOINT_NAMES = {
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
};

static std::string shellQuote(const std::string &s) {
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
    cv::Size sensor_size;  ///< Logical image size used by the C++ pipeline
    BufferedPort<ImageOf<PixelMono>> output_port;  ///< Port used to send images to YoloPose
    BufferedPort<Bottle> input_port;               ///< Port used to receive skeleton

    double period{0.1};
    double tic{0.0};
    bool waiting{false};
    bool launched_python{false};

    std::string local_img_out{"/YoloPose_offline/img:o"};
    std::string local_sklt_in{"/YoloPose_offline/sklt:i"};
    std::string yolo_img_in{"/YoloPose/img:i"};
    std::string yolo_sklt_out{"/YoloPose/sklt:o"};
    std::string pid_file{"/tmp/yolopose_offline_python.pid"};

public:
    bool init(const std::string &model_path,
              const std::string &script_path,
              double rate,
              const cv::Size &image_size)
    {
        sensor_size = image_size;
        period = (rate > 0.0) ? 1.0 / rate : 0.1;

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
        // this class assumes the sidecar is already running in another terminal.
        if (!script_path.empty()) {
            std::ostringstream cmd;
            cmd << "python3 " << shellQuote(script_path)
                << " --model " << shellQuote(model_path)
                << " --w " << sensor_size.width
                << " --h " << sensor_size.height
                << " --img_port " << yolo_img_in
                << " --sklt_port " << yolo_sklt_out
                << " & echo $! > " << pid_file;

            int r = std::system(cmd.str().c_str());
            if (r != 0) {
                yError() << "Could not launch YoloPose Python sidecar with command:" << cmd.str();
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
            yError() << "YoloPose sidecar ports not found. Expected" << yolo_img_in << "and" << yolo_sklt_out;
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

    bool update(const cv::Mat &latest_image,
                double latest_ts,
                hpecore::stampedPose &previous_skeleton)
    {
        if (latest_image.empty()) {
            return false;
        }

        // Handle video timestamp reset/loop.
        if (latest_ts < tic) {
            tic = latest_ts - 2.0;
            waiting = false;
        }

        // Send a new image if the detector is not already processing one, or
        // force a resend if no reply has arrived for 2 seconds.
        if ((!waiting && latest_ts - tic > period) || (latest_ts - tic > 2.0)) {
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
            output_port.write();

            tic = latest_ts;
            waiting = true;
        }

        // Read a ready skeleton without blocking the video loop.
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

int main(int argc, char *argv[]){

    // Prepare and configure the resource finder
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    if(rf.check("help")) {
        std::stringstream ss;
        ss << "Usage: YoloPose_offline [options]\n\n";
        ss << "Options:\n";
        ss << std::left << std::setw(24) << "--data_file"       << std::setw(12) << "<string>" << ": path to input video file (.mp4)\n";
        ss << std::left << std::setw(24) << "--output_period"   << std::setw(12) << "<double>" << ": CSV/vis output period (s), default 0.005\n";
        ss << std::left << std::setw(24) << "--net_period"      << std::setw(12) << "<double>" << ": YoloPose update period (s), default 0.05\n";
        ss << std::left << std::setw(24) << "--h"               << std::setw(12) << "<int>"    << ": image height (default 480)\n";
        ss << std::left << std::setw(24) << "--w"               << std::setw(12) << "<int>"    << ": image width (default 640)\n";
        ss << std::left << std::setw(24) << "--vis"             << std::setw(12) << ""         << ": enable on-screen visualization\n";
        ss << std::left << std::setw(24) << "--output_csv"      << std::setw(12) << "<string>" << ": path to output CSV file\n";
        ss << std::left << std::setw(24) << "--no_csv"          << std::setw(12) << ""         << ": skip CSV logging\n";
        ss << std::left << std::setw(24) << "--output_video"    << std::setw(12) << "<string>" << ": path to output video file (.mp4)\n";
        ss << std::left << std::setw(24) << "--no_video"        << std::setw(12) << ""         << ": disable video output\n";
        ss << std::left << std::setw(24) << "--yolo_model_path" << std::setw(12) << "<string>" << ": path to yolo26n-pose.pt\n";
        ss << std::left << std::setw(24) << "--YoloPose_script" << std::setw(12) << "<string>" << ": path to YoloPose_yarp_server.py\n";
        ss << std::left << std::setw(24) << "--pwrjlr_file"     << std::setw(12) << "<string>" << ": base path for PowerJoular output\n";
        ss << std::left << std::setw(24) << "--gpu_file"        << std::setw(12) << "<string>" << ": output CSV for GPU telemetry\n";
        ss << std::left << std::setw(24) << "--gpu_period_ms"   << std::setw(12) << "<int>"    << ": nvidia-smi sampling period in ms\n";
        ss << std::left << std::setw(24) << "--gpu_index"       << std::setw(12) << "<int>"    << ": NVIDIA GPU index to monitor\n";
        yInfo() << ss.str();
        // exit after printing help
        return 0;
    }

    // Read parameters from command line with default values
    std::string datapath_file = rf.check("data_file", Value("/data/moveEnet_test/mp4/cam2_S11_Directions_1.mp4")).asString();
    double output_period = rf.check("output_period", Value(0.005)).asFloat64();                     // CSV write period
    double net_period = rf.check("net_period", Value(0.05)).asFloat64();                            // Range from 5ms to 100ms -> 200 Hz to 10 Hz
    cv::Size res(rf.check("w", Value(640)).asInt32(), rf.check("h", Value(480)).asInt32());
    bool is_visualize = rf.check("vis");                                                            // Visualization flag
    std::string output_csv = rf.check("output_csv", Value("/home/moveEnetFlow/csv_file/YoloPose/test_YoloPose.csv")).asString();
    bool no_csv = rf.check("no_csv");
    std::string output_video = rf.check("output_video", Value("")).asString();
    bool no_video = rf.check("no_video");
    std::string yolo_model_path = rf.check("yolo_model_path", Value("/home/model_mounts/YoloPose/yolo26n-pose.pt")).asString();
    std::string yolo_script_path = rf.check("YoloPose_script", Value("/home/model_mounts/YoloPose/YoloPose_yarp_server.py")).asString();
    std::string powerjoular_file = rf.check("pwrjlr_file", Value("")).asString();
    std::string gpu_monitor_file = rf.check("gpu_file", Value("/home/moveEnetFlow/pwr_gpu_file/single_test/YoloPose_gpu_pwr")).asString();
    int gpu_monitor_period_ms = rf.check("gpu_period_ms", Value(5)).asInt32();
    int gpu_monitor_index = rf.check("gpu_index", Value(0)).asInt32();


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
    std::ofstream csv_file;                             // CSV file stream for logging results
    std::vector<std::string> csv_buffer;                // store rows for deferred write
    VisualizationContext vis_ctx;                      // Visualization and video writer resources

    // CSV writer setup
    if (!no_csv) {
        csv_file.open(output_csv);
        if (!csv_file.is_open()) {
            yError() << "Could not open CSV file for writing:" << output_csv;
            return -1;
        }
        yInfo() << "CSV logging enabled ->" << output_csv;
        csv_file << "timestamp,latency";
        for (int j = 0; j < 13; j++) {
            csv_file << "," << CSV_JOINT_NAMES[j] << "_x," << CSV_JOINT_NAMES[j] << "_y";
        }
        csv_file << "\n";
        csv_file.flush();
    }

    // Visualization and video setup
    if (!initialiseVisualization(vis_ctx, res, is_visualize, no_video, output_video, datapath_file, output_period)) {
        return -1;
    }


    // ===== OPEN VIDEO FILE =====
    yInfo() << "Loading video:" << datapath_file;
    cv::VideoCapture cap(datapath_file);
    if (!cap.isOpened()) {
        yError() << "Could not open video file:" << datapath_file;
        return -1;
    }

    // Init detector ports
    double detF = 1.0 / net_period;                     // Detection frequency in Hz
    offlineDetector yolo_handler;
    if (!yolo_handler.init(yolo_model_path, yolo_script_path, detF, res)) {
        yError() << "YoloPose init failed";
        return -1;
    }


    // ===== INITIALIZE ALGORITHMIC COMPONENTS =====
    hpecore::stampedPose detected_pose;                 // Detected pose from YoloPose
    hpecore::skeleton13 filtered_pose{};                // Last known filtered pose
    bool pose_initialised = false;                      // True after first valid detection

    double tnow = 0.0;                                  // Current simulation time
    double next_csv_upd = output_period;               // time threshold for next CSV row
    double next_vis_upd = output_period;               // time threshold for next visualization frame


    // ===== MAIN PROCESSING LOOP (frame-by-frame) =====

    int frame_count = 0;

    while (true) {

        // Read next frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            yInfo() << "End of video reached after" << frame_count << "frames.";
            break;
        }
        frame_count++;

        tnow = cap.get(cv::CAP_PROP_POS_MSEC) / 100.0; // current timestamp in seconds

        // Send frames to YoloPose at net_period intervals and read returned skeletons.
        // The rate limiting is implemented inside offlineDetector::update().
        bool was_detected = yolo_handler.update(frame, tnow, detected_pose);
        if (was_detected && hpecore::poseNonZero(detected_pose.pose)) {
            filtered_pose = detected_pose.pose;
            pose_initialised = true;
        }

        // CSV logging at output_period rate
        if (tnow >= next_csv_upd) {
            next_csv_upd += output_period;
            if (pose_initialised && csv_file.is_open()) {
                std::ostringstream row;
                row << std::fixed << std::setprecision(6) << tnow;
                double lat = (detected_pose.timestamp > 0) ? detected_pose.delay : 0.0;
                row << "," << lat;
                for (int j = 0; j < 13; j++) {
                    int hpe_idx = CSV_HPE_ORDER[j];
                    row << "," << filtered_pose[hpe_idx].u << "," << filtered_pose[hpe_idx].v;
                }
                csv_buffer.push_back(row.str());
            }
        }

        // Visualization
        if ((is_visualize || (!output_video.empty() && !no_video)) && tnow >= next_vis_upd) {
            next_vis_upd += output_period;
            renderVisualizationFrameOP(vis_ctx, frame, pose_initialised, filtered_pose, detected_pose, tnow);
            writeVisualizationFrame(vis_ctx, pose_initialised);
            if (showVisualizationFrame(vis_ctx)) {
                yInfo() << "User requested stop";
                break;
            }
        }


    }


    // Cleanup
    if (csv_file.is_open()) {
        for (const auto &line : csv_buffer) {
            csv_file << line << "\n";
        }
        csv_file.close();
        yInfo() << "CSV rows written:" << csv_buffer.size() << "->" << output_csv;
    }
    power_monitor.stop();
    yolo_handler.close();

    closeVisualization(vis_ctx, output_video);

    return 0;
}

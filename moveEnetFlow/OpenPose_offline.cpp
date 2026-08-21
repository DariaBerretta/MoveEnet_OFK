// OpenPose offline runner.
// Outout CSV bounded at 50Hz, .mp4 frame rate

#include <yarp/os/all.h>
#include <yarp/cv/Cv.h>                     // needed for yarp::cv::fromCvMat
#include <yarp/sig/Image.h>                 // needed for BufferedPort<ImageOf<PixelMono>>

#include <opencv2/opencv.hpp>

#include <hpe-core/utility.h>

#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <unistd.h>
#include <cstdlib>
#include <openpose/headers.hpp>
#include <tuple>
#include <chrono>
#include <limits>
#include <cmath>

#include "utils/power_monitor.h"
#include "utils/visualization_utils.h"

using namespace yarp::os;
using namespace yarp::sig;
using std::string;
using yarp::os::Value;

class offlineDetector
{
private:
    // double period{0.001};

    // OpenPose
    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
    bool op_ready{false};
    op::PoseModel poseModel{op::PoseModel::BODY_25};
    //op::PoseModel poseModel{op::PoseModel::COCO_18};
    op::String modelFolder;

public:

    bool init(const std::string& model_folder, double rate, const std::string& device = std::string())
    {
        if (rate <= 0.0) return false;
        // period = 1.0 / rate;

        try {
            op::WrapperStructPose poseConfig{};
            poseConfig.modelFolder = op::String{model_folder};
            poseConfig.poseModel = poseModel;
            poseConfig.netInputSize = op::Point<int>{-1, 320};
            poseConfig.outputSize = op::Point<int>{-1, -1};
            poseConfig.keypointScaleMode = op::ScaleMode::InputResolution;
            poseConfig.renderMode = op::RenderMode::None;
            //poseConfig.renderPose = -1;
            poseConfig.blendOriginalFrame = false;
            poseConfig.numberPeopleMax = 1;

            // Device handling: allow CPU or select GPU index via `device` string.
            if (!device.empty()) {
                std::string dev = device;
                std::string dev_l = dev;
                std::transform(dev_l.begin(), dev_l.end(), dev_l.begin(), ::tolower);
                if (dev_l.rfind("cpu", 0) == 0) {
                    // Force CPU-only OpenPose
                    poseConfig.gpuNumber = 0;
                } else {
                    // GPU requested: extract GPU index if present (cuda:N or N)
                    int gpu_id = 0;
                    size_t colon = dev.find(':');
                    if (colon != std::string::npos) {
                        std::string tail = dev.substr(colon + 1);
                        try { gpu_id = std::stoi(tail); } catch (...) { gpu_id = 0; }
                    } else if (!dev.empty() && std::all_of(dev.begin(), dev.end(), ::isdigit)) {
                        try { gpu_id = std::stoi(dev); } catch (...) { gpu_id = 0; }
                    }
                    poseConfig.gpuNumber = 1;
                    poseConfig.gpuNumberStart = gpu_id;
                }
            }

            opWrapper.configure(poseConfig);
            // opWrapper.disableMultiThreading();
            opWrapper.start();
            op_ready = true;
            return true;
        }
        catch (const std::exception& e) {
            yError() << "OpenPose init failed:" << e.what();
            return false;
        }
    }

    void close()
    {
        if (op_ready)
            opWrapper.stop();
    }

    bool update(const cv::Mat &latest_image,
            double latest_ts,
            hpecore::stampedPose &previous_skeleton,
            bool &response_received)
    {
        response_received = false;

        previous_skeleton.timestamp = latest_ts;
        previous_skeleton.delay =
            std::numeric_limits<double>::quiet_NaN();

        if (!op_ready || latest_image.empty())
            return false;

        // ============================================================
        // Caller-side preprocessing:
        // excluded from service latency measured inside update(),
        // included in complete method latency measured by main().
        // ============================================================

        // OpenPose expects BGR
        cv::Mat input_bgr;
        if (latest_image.channels() == 3) {
            input_bgr = latest_image;
        }
        else if (latest_image.channels() == 4) {
            cv::cvtColor(latest_image, input_bgr, cv::COLOR_BGRA2BGR);
        }
        else {
            cv::cvtColor(latest_image, input_bgr, cv::COLOR_GRAY2BGR);
        }

        // Convert/wrap input in OpenPose format
        const op::Matrix imageToProcess =
            OP_CV2OPCONSTMAT(input_bgr);

        // ============================================================
        // SERVICE LATENCY START
        // ============================================================

        const auto request_start =
            std::chrono::steady_clock::now();

        auto datumProcessed =
            opWrapper.emplaceAndPop(imageToProcess);

        // OpenPose call returned: analogous to receiving the response
        // from the Python sidecar for MoveNet / YOLOPose.
        if (datumProcessed && !datumProcessed->empty())
            response_received = true;

        // We still measure failed/empty responses.
        if (!response_received) {
            const auto request_end =
                std::chrono::steady_clock::now();

            previous_skeleton.delay =
                std::chrono::duration<double>(
                    request_end - request_start
                ).count();

            return false;
        }

        const auto& keypoints =
            datumProcessed->at(0)->poseKeypoints;

        if (keypoints.getSize(0) == 0 ||
            keypoints.getSize(1) < 15)
        {
            const auto request_end =
                std::chrono::steady_clock::now();

            previous_skeleton.delay =
                std::chrono::duration<double>(
                    request_end - request_start
                ).count();

            return false;
        }

        // ============================================================
        // BODY_25 -> skeleton13
        // Included in latency, like pose decoding for MoveNet/YOLO
        // ============================================================

        hpecore::skeleton13 pose13{};
        hpecore::confidence13 conf13{};

        pose13.fill({0.f, 0.f});
        conf13.fill(0.f);

        constexpr int person = 0;

        auto getXYC = [&](int part)
        {
            const float x = keypoints[{person, part, 0}];
            const float y = keypoints[{person, part, 1}];
            const float c = keypoints[{person, part, 2}];

            return std::tuple<float, float, float>(x, y, c);
        };

        auto setJoint = [&](int j13, int op_idx)
        {
            auto [x, y, c] = getXYC(op_idx);
            pose13[j13] = {x, y};
            conf13[j13] = c;
        };

        setJoint(hpecore::head,       0);
        setJoint(hpecore::shoulderR,  2);
        setJoint(hpecore::shoulderL,  5);
        setJoint(hpecore::elbowR,     3);
        setJoint(hpecore::elbowL,     6);
        setJoint(hpecore::hipL,      12);
        setJoint(hpecore::hipR,       9);
        setJoint(hpecore::handR,      4);
        setJoint(hpecore::handL,      7);
        setJoint(hpecore::kneeR,     10);
        setJoint(hpecore::kneeL,     13);
        setJoint(hpecore::footR,     11);
        setJoint(hpecore::footL,     14);

        previous_skeleton.pose = pose13;
        previous_skeleton.conf = conf13;
        previous_skeleton.timestamp = latest_ts;

        // ============================================================
        // SERVICE LATENCY END: usable skeleton available
        // ============================================================

        const auto request_end =
            std::chrono::steady_clock::now();

        previous_skeleton.delay =
            std::chrono::duration<double>(
                request_end - request_start
            ).count();

        return hpecore::poseNonZero(previous_skeleton.pose);
    }
};

int main(int argc, char *argv[]){
    
    // Prepare and configure the resource finder
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    if(rf.check("help")) {
        std::stringstream ss;
        ss << "Usage: OpenPose_offline [options]\n\n";
        ss << "Options:\n";
        ss << std::left << std::setw(24) << "--data_file"       << std::setw(12) << "<string>" << ": path to input video file (.mp4)\n";
        ss << std::left << std::setw(24) << "--output_period"   << std::setw(12) << "<double>" << ": CSV/vis output period (s), default 0.005\n";
        ss << std::left << std::setw(24) << "--net_period"      << std::setw(12) << "<double>" << ": OpenPose update period (s), default 0.05\n";
        ss << std::left << std::setw(24) << "--h"               << std::setw(12) << "<int>"    << ": image height (default 480)\n";
        ss << std::left << std::setw(24) << "--w"               << std::setw(12) << "<int>"    << ": image width (default 640)\n";
        ss << std::left << std::setw(24) << "--vis"             << std::setw(12) << ""         << ": enable on-screen visualization\n";
        ss << std::left << std::setw(24) << "--output_csv"      << std::setw(12) << "<string>" << ": path to output CSV file\n";
        ss << std::left << std::setw(24) << "--no_csv"          << std::setw(12) << ""         << ": skip CSV logging\n";
        ss << std::left << std::setw(24) << "--output_video"    << std::setw(12) << "<string>" << ": path to output video file (.mp4)\n";
        ss << std::left << std::setw(24) << "--no_video"        << std::setw(12) << ""         << ": disable video output\n";
        ss << std::left << std::setw(24) << "--op_model_path" << std::setw(12) << "<string>" << ": path to OpenPose model weights\n";
        ss << std::left << std::setw(24) << "--openpose_script" << std::setw(12) << "<string>" << ": path to OpenPose Python script\n";
        ss << std::left << std::setw(24) << "--latency_csv" << std::setw(12) << "<string>" << ": per-inference latency CSV\n";
        ss << std::left << std::setw(24) << "--pwrjlr_file"     << std::setw(12) << "<string>" << ": base path for PowerJoular output\n";
        ss << std::left << std::setw(24) << "--gpu_file"        << std::setw(12) << "<string>" << ": output CSV for GPU telemetry\n";
        ss << std::left << std::setw(24) << "--device"          << std::setw(12) << "<string>" << ": device for OpenPose (cpu or cuda:N)\n";
        ss << std::left << std::setw(24) << "--gpu_period_ms"   << std::setw(12) << "<int>"    << ": nvidia-smi sampling period in ms\n";
        ss << std::left << std::setw(24) << "--gpu_index"       << std::setw(12) << "<int>"    << ": NVIDIA GPU index to monitor\n";
        yInfo() << ss.str();
        // exit after printing help
        return 0;
    }

    // Read parameters from command line with default values
    std::string datapath_file = rf.check("data_file", Value("/data/eh36m_testing_set_S9S11/rgb/cam2_S9_Directions_1.mp4")).asString();
    double output_period = rf.check("output_period", Value(0.005)).asFloat64();                     // CSV write period
    double net_period = rf.check("net_period", Value(0.05)).asFloat64();                            // Range from 5ms to 100ms -> 200 Hz to 10 Hz
    cv::Size res(rf.check("w", Value(640)).asInt32(), rf.check("h", Value(480)).asInt32());
    bool is_visualize = rf.check("vis");                                                            // Visualization flag
    std::string output_csv = rf.check("output_csv", Value("tmp/output.csv")).asString();
    bool no_csv = rf.check("no_csv");
    std::string output_video = rf.check("output_video", Value("tmp/output.mp4")).asString();
    bool no_video = rf.check("no_video");
    std::string op_model_path = rf.check("op_model_path", Value("/usr/local/src/openpose/models/")).asString();
    std::string latency_csv_path = rf.check("latency_csv", Value("")).asString();
    // std::string op_model_path = rf.check("op_model_path", Value("/model mounts/OpenPose/")).asString();
    std::string powerjoular_file = rf.check("pwrjlr_file", Value("")).asString();
    std::string gpu_monitor_file = rf.check("gpu_file", Value("")).asString();
    int gpu_monitor_period_ms = rf.check("gpu_period_ms", Value(5)).asInt32();
    int gpu_monitor_index = rf.check("gpu_index", Value(0)).asInt32();
    std::string device = rf.check("device", Value("")).asString();

    // If device indicates a CUDA GPU, prefer that GPU index for monitoring
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
    std::ofstream latency_file;
    std::vector<std::string> latency_buffer;
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
            csv_file << ",joint" << j << "_x,joint" << j << "_y";
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
            << "service_latency_ms,"
            << "method_latency_ms,"
            << "response_received,"
            << "valid_pose\n";

        yInfo()
            << "Per-inference latency logging enabled ->"
            << latency_csv_path;
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
    offlineDetector op_handler;

    if (!op_handler.init(op_model_path, detF, device)) {
        yError() << "OpenPose init failed";
        return -1;
    }


    // ===== INITIALIZE ALGORITHMIC COMPONENTS =====
    hpecore::stampedPose detected_pose;                 // Detected pose from OpenPose
    hpecore::skeleton13 filtered_pose{};                // Last known filtered pose 
    bool pose_initialised = false;                      // True after first valid detection

    double tnow = 0.0;                                  // Current simulation time
    double next_net_upd = 0.0;                         // time threshold for next OpenPose call
    double next_csv_upd = output_period;               // time threshold for next CSV row
    double next_vis_upd = output_period;               // time threshold for next visualization frame

    std::size_t inference_count = 0;
    std::size_t valid_inference_count = 0;

    double first_inference_ts = std::numeric_limits<double>::quiet_NaN();

    double last_inference_ts = std::numeric_limits<double>::quiet_NaN();

    constexpr double time_epsilon = 1e-9;


    // ===== MAIN PROCESSING LOOP (frame-by-frame) =====

    int frame_count = 0;

    while (true) {

        // Read next frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            yInfo() << "End of video reached. Frames processed:" << frame_count;
            break;
        }
        tnow = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0; // Convert ms to seconds
        frame_count++;

        // Send frame to OpenPose at net_period intervals
        bool was_detected = false;
        bool response_received = false;

        // if (tnow >= next_net_upd) {
        //     next_net_upd += net_period;
        //     was_detected = op_handler.update(frame, tnow, detected_pose, response_received);
        //     if (was_detected && hpecore::poseNonZero(detected_pose.pose)) {
        //         filtered_pose = detected_pose.pose;
        //         pose_initialised = true;
        //     }
        // }

        if (tnow + time_epsilon >= next_net_upd) {

            const std::size_t request_id = inference_count;
            const double scheduled_timestamp = next_net_upd;

            if (inference_count == 0) {
                first_inference_ts = tnow;
            }

            last_inference_ts = tnow;

            response_received = false;

            // Complete-method latency starts when the RGB frame is
            // already available and OpenPose processing begins.
            const auto method_start =
                std::chrono::steady_clock::now();

            was_detected = op_handler.update(
                frame,
                tnow,
                detected_pose,
                response_received
            );

            ++inference_count;

            double method_latency_s =
                std::numeric_limits<double>::quiet_NaN();

            if (was_detected) {

                filtered_pose = detected_pose.pose;
                pose_initialised = true;

                // Complete OpenPose method output is now usable.
                const auto method_end =
                    std::chrono::steady_clock::now();

                method_latency_s =
                    std::chrono::duration<double>(
                        method_end - method_start
                    ).count();

                ++valid_inference_count;
            }

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

                // Internal OpenPose service latency
                if (std::isfinite(detected_pose.delay)) {
                    row << std::setprecision(6)
                        << detected_pose.delay * 1000.0;
                } else {
                    row << "nan";
                }

                row << ",";

                // Complete OpenPose method latency
                if (std::isfinite(method_latency_s)) {
                    row << std::setprecision(6)
                        << method_latency_s * 1000.0;
                } else {
                    row << "nan";
                }

                row << ","
                    << (response_received ? 1 : 0)
                    << ","
                    << (was_detected ? 1 : 0);

                latency_buffer.push_back(
                    row.str()
                );
            }

            // Same scheduling policy as YOLOPose:
            // skip nominal deadlines for which no RGB frame existed.
            do {
                next_net_upd += net_period;
            } while (
                next_net_upd <= tnow + time_epsilon
            );
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
                    row << "," << filtered_pose[j].u << "," << filtered_pose[j].v;
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

    if (latency_file.is_open()) {

        for (const auto &line : latency_buffer)
            latency_file << line << "\n";

        latency_file.close();

        yInfo()
            << "Latency rows written:"
            << latency_buffer.size()
            << "->"
            << latency_csv_path;
    }

    yInfo()
        << "OpenPose inferences completed:"
        << inference_count;

    yInfo()
        << "OpenPose valid inferences:"
        << valid_inference_count;

    yInfo()
        << "Requested inference rate:"
        << (1.0 / net_period)
        << "Hz";

    if (inference_count > 1 &&
        last_inference_ts > first_inference_ts)
    {
        const double observed_rate =
            static_cast<double>(inference_count - 1) /
            (last_inference_ts - first_inference_ts);

        yInfo()
            << "Observed inference rate:"
            << observed_rate
            << "Hz";
    }
    power_monitor.stop();
    op_handler.close();

    closeVisualization(vis_ctx, output_video);
    
    return 0;
}



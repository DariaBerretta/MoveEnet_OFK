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
#include <unistd.h>
#include <cstdlib>
#include <openpose/headers.hpp>
#include <tuple>

#include "utils/power_monitor.h"
#include "utils/visualization_utils.h"

using namespace yarp::os;
using namespace yarp::sig;
using std::string;
using yarp::os::Value;

class offlineDetector
{
private:
    double period{0.001};

    // OpenPose
    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
    bool op_ready{false};
    op::PoseModel poseModel{op::PoseModel::BODY_25};
    //op::PoseModel poseModel{op::PoseModel::COCO_18};
    op::String modelFolder;

public:

    bool init(const std::string& model_folder, double rate)
    {
        if (rate <= 0.0) return false;
        period = 1.0 / rate;

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
                hpecore::stampedPose &previous_skeleton)
    {
        if (!op_ready)
            return false;

        double t0 = yarp::os::Time::now();
        previous_skeleton.timestamp = latest_ts;
        previous_skeleton.delay = 0.0;          // will be overwritten on success

        // OpenPose expects BGR
        cv::Mat input_bgr;
        if (latest_image.channels() == 3)
            input_bgr = latest_image;
        else
            cv::cvtColor(latest_image, input_bgr, cv::COLOR_GRAY2BGR);

        // Convert to OpenPose format
        const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(input_bgr);

        auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);

        if (!datumProcessed || datumProcessed->empty())
            return false;

        const auto& keypoints = datumProcessed->at(0)->poseKeypoints;

        if (keypoints.getSize(0) == 0)
            return false;
        if (keypoints.getSize(1) < 15)
            return false;

        // ===== Convert BODY_25 → skeleton13 =====

        hpecore::skeleton13 pose13{};
        hpecore::confidence13 conf13{};

        pose13.fill({0.f, 0.f});
        conf13.fill(0.f);

        int person = 0;

        auto getXYC = [&](int part)
        {
            float x = keypoints[{person, part, 0}];
            float y = keypoints[{person, part, 1}];
            float c = keypoints[{person, part, 2}];
            return std::tuple<float,float,float>(x,y,c);
        };

        auto setJoint = [&](int j13, int op_idx)
        {
            auto [x,y,c] = getXYC(op_idx);
            pose13[j13] = {x,y};
            conf13[j13] = c;
        };

        auto setHeadMidpoint = [&]()
        {
            auto [x0,y0,c0] = getXYC(0);
            auto [x1,y1,c1] = getXYC(1);
            pose13[hpecore::head] = {(x0 + x1) * 0.5f, (y0 + y1) * 0.5f};
            conf13[hpecore::head] = std::max(c0, c1);
        };

        // BODY_25 mapping
        setHeadMidpoint();
        setJoint(hpecore::shoulderR, 2);
        setJoint(hpecore::shoulderL, 5);
        setJoint(hpecore::elbowR,    3);
        setJoint(hpecore::elbowL,    6);
        setJoint(hpecore::hipL,      12);
        setJoint(hpecore::hipR,      9);
        setJoint(hpecore::handR,     4);
        setJoint(hpecore::handL,     7);
        setJoint(hpecore::kneeR,     10);
        setJoint(hpecore::kneeL,     13);
        setJoint(hpecore::footR,     11);
        setJoint(hpecore::footL,     14);

        previous_skeleton.pose = pose13;
        previous_skeleton.conf = conf13;
        previous_skeleton.timestamp = latest_ts;
        previous_skeleton.delay = yarp::os::Time::now() - t0;

        return true;
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
        ss << std::left << std::setw(24) << "--pwrjlr_file"     << std::setw(12) << "<string>" << ": base path for PowerJoular output\n";
        ss << std::left << std::setw(24) << "--gpu_file"        << std::setw(12) << "<string>" << ": output CSV for GPU telemetry\n";
        ss << std::left << std::setw(24) << "--gpu_period_ms"   << std::setw(12) << "<int>"    << ": nvidia-smi sampling period in ms\n";
        ss << std::left << std::setw(24) << "--gpu_index"       << std::setw(12) << "<int>"    << ": NVIDIA GPU index to monitor\n";
        yInfo() << ss.str();
        // exit after printing help
        return 0;
    }

    // Read parameters from command line with default values
    std::string datapath_file = rf.check("data_file", Value("/data/moveEnet_test/mp4/cam2_S1_Phoning.mp4")).asString();
    double output_period = rf.check("output_period", Value(0.005)).asFloat64();                     // CSV write period
    double net_period = rf.check("net_period", Value(0.05)).asFloat64();                            // Range from 5ms to 100ms -> 200 Hz to 10 Hz
    cv::Size res(rf.check("w", Value(640)).asInt32(), rf.check("h", Value(480)).asInt32());
    bool is_visualize = rf.check("vis");                                                            // Visualization flag
    std::string output_csv = rf.check("output_csv", Value("/home/moveEnetFlow/csv_file/openpose/test_openpose.csv")).asString();
    bool no_csv = rf.check("no_csv");
    std::string output_video = rf.check("output_video", Value("")).asString();
    bool no_video = rf.check("no_video");
    std::string op_model_path = rf.check("op_model_path", Value("/usr/local/src/openpose/models/")).asString();
    std::string powerjoular_file = rf.check("pwrjlr_file", Value("")).asString();
    std::string gpu_monitor_file = rf.check("gpu_file", Value("/home/moveEnetFlow/pwr_gpu_file/single_test/openpose_gpu_pwr")).asString();
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
            csv_file << ",joint" << j << "_x,joint" << j << "_y";
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
    offlineDetector op_handler;
    if (!op_handler.init(op_model_path, detF)) {
        yError() << "OpenPose init failed";
        return -1;
    }


    // ===== INITIALIZE ALGORITHMIC COMPONENTS =====
    hpecore::stampedPose detected_pose;                 // Detected pose from OpenPose
    hpecore::skeleton13 filtered_pose{};                // Last known filtered pose 
    bool pose_initialised = false;                      // True after first valid detection

    double tnow = 0.0;                                  // Current simulation time
    double next_net_upd = net_period;                   // time threshold for next OpenPose call
    double next_csv_upd = output_period;               // time threshold for next CSV row
    double next_vis_upd = output_period;               // time threshold for next visualization frame


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
        if (tnow >= next_net_upd) {
            next_net_upd += net_period;
            was_detected = op_handler.update(frame, tnow, detected_pose);
            if (was_detected && hpecore::poseNonZero(detected_pose.pose)) {
                filtered_pose = detected_pose.pose;
                pose_initialised = true;
            }
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
    power_monitor.stop();
    op_handler.close();

    closeVisualization(vis_ctx, output_video);
    
    return 0;
}



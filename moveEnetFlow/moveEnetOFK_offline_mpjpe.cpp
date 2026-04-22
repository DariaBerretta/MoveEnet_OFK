#include <yarp/os/all.h>
#include <yarp/cv/Cv.h>                     // needed for yarp::cv::fromCvMat
#include <yarp/sig/Image.h>                 // needed for BufferedPort<ImageOf<PixelMono>>
#include <event-driven/core.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/hdf.hpp>
#include <event-driven/algs.h>
#include <event-driven/vis.h>
#include <hpe-core/utility.h>
#include <hpe-core/motion.h>                // For hpecore::pwtripletvelocity
#include <hpe-core/fusion.h>                // For hpecore::multiJointLatComp
#include <hpe-core/representations.h>       // For hpecore::SAE
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <cstdlib>
#include "utils/power_monitor.h"
#include "utils/visualization_utils.h"

using namespace yarp::os;
using namespace yarp::sig;
using std::string;
using yarp::os::Value;

namespace {

constexpr int kNumJoints = 13;
constexpr int kNumCoords = kNumJoints * 2;
const std::regex kSkltRegex(R"(^\s*\d+\s+([0-9eE+\-\.]+)\s+SKLT\s+\(([^)]*)\).*$)");

struct GTSeries {
    std::vector<double> ts;
    std::vector<std::array<double, kNumCoords>> coords;
};

struct PredSample {
    double ts{0.0};
    std::array<double, kNumCoords> coords{};
};

bool deriveGTPathFromDataFile(const std::string &data_file, std::string &gt_file_out)
{
    const std::string suffix = "/ch0dvs/data.log";
    if (data_file.size() < suffix.size() ||
        data_file.compare(data_file.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }

    gt_file_out = data_file.substr(0, data_file.size() - suffix.size()) + "/ch0GT200Hzskeleton/data.log";
    return true;
}

bool loadGTSeries(const std::string &gt_file, GTSeries &out, std::string &error_msg)
{
    std::ifstream ifs(gt_file);
    if (!ifs.is_open()) {
        error_msg = "could not open GT file: " + gt_file;
        return false;
    }

    out.ts.clear();
    out.coords.clear();

    std::string line;
    size_t line_no = 0;
    while (std::getline(ifs, line)) {
        ++line_no;
        std::smatch match;
        if (!std::regex_match(line, match, kSkltRegex)) {
            continue;
        }

        double ts = 0.0;
        try {
            ts = std::stod(match[1].str());
        } catch (const std::exception &) {
            error_msg = "invalid GT timestamp at line " + std::to_string(line_no);
            return false;
        }

        std::istringstream coord_stream(match[2].str());
        std::array<double, kNumCoords> c{};
        for (int i = 0; i < kNumCoords; ++i) {
            if (!(coord_stream >> c[i])) {
                error_msg = "invalid GT coordinate count at line " + std::to_string(line_no);
                return false;
            }
        }
        double extra = 0.0;
        if (coord_stream >> extra) {
            error_msg = "too many GT coordinates at line " + std::to_string(line_no);
            return false;
        }

        if (!out.ts.empty() && ts <= out.ts.back()) {
            error_msg = "GT timestamps are not strictly increasing";
            return false;
        }

        out.ts.push_back(ts);
        out.coords.push_back(c);
    }

    if (out.ts.empty()) {
        error_msg = "no valid SKLT entries found in GT file";
        return false;
    }

    return true;
}

double interpWithPaddedEnds(const std::vector<double> &x, const std::vector<double> &y, double query)
{
    if (query <= x.front()) {
        return y.front();
    }
    if (query >= x.back()) {
        return y.back();
    }

    auto it_hi = std::upper_bound(x.begin(), x.end(), query);
    const size_t idx_hi = static_cast<size_t>(it_hi - x.begin());
    const size_t idx_lo = idx_hi - 1;

    const double x0 = x[idx_lo];
    const double x1 = x[idx_hi];
    const double y0 = y[idx_lo];
    const double y1 = y[idx_hi];

    if (x1 <= x0) {
        return y1;
    }

    const double alpha = (query - x0) / (x1 - x0);
    return y0 + alpha * (y1 - y0);
}

bool interpolateGTAtPredTimestamps(const GTSeries &gt,
                                   const std::vector<PredSample> &pred_samples,
                                   std::vector<std::array<double, kNumCoords>> &gt_interp,
                                   std::string &error_msg)
{
    if (pred_samples.empty()) {
        error_msg = "no prediction samples available";
        return false;
    }
    if (gt.ts.empty()) {
        error_msg = "GT is empty";
        return false;
    }

    std::vector<double> ts_pad;
    ts_pad.reserve(gt.ts.size() + 2);
    ts_pad.push_back(0.0);
    ts_pad.insert(ts_pad.end(), gt.ts.begin(), gt.ts.end());
    ts_pad.push_back(gt.ts.back() + 1.0);

    gt_interp.assign(pred_samples.size(), {});
    std::vector<double> axis_pad(ts_pad.size(), 0.0);
    for (int c = 0; c < kNumCoords; ++c) {
        axis_pad[0] = gt.coords.front()[c];
        for (size_t i = 0; i < gt.coords.size(); ++i) {
            axis_pad[i + 1] = gt.coords[i][c];
        }
        axis_pad.back() = gt.coords.back()[c];

        for (size_t i = 0; i < pred_samples.size(); ++i) {
            gt_interp[i][c] = interpWithPaddedEnds(ts_pad, axis_pad, pred_samples[i].ts);
        }
    }

    return true;
}

bool computeMPJPE(const std::vector<PredSample> &pred_samples,
                  const GTSeries &gt,
                  double &mpjpe_px,
                  std::string &error_msg)
{
    std::vector<std::array<double, kNumCoords>> gt_interp;
    if (!interpolateGTAtPredTimestamps(gt, pred_samples, gt_interp, error_msg)) {
        return false;
    }

    long double sum_error = 0.0L;
    size_t n_points = 0;
    for (size_t i = 0; i < pred_samples.size(); ++i) {
        for (int j = 0; j < kNumJoints; ++j) {
            const double dx = pred_samples[i].coords[2 * j] - gt_interp[i][2 * j];
            const double dy = pred_samples[i].coords[2 * j + 1] - gt_interp[i][2 * j + 1];
            sum_error += std::sqrt(dx * dx + dy * dy);
            ++n_points;
        }
    }

    if (n_points == 0) {
        error_msg = "no valid prediction points available for MPJPE";
        return false;
    }

    const double mpjpe = static_cast<double>(sum_error / static_cast<long double>(n_points));
    if (!std::isfinite(mpjpe)) {
        error_msg = "computed MPJPE is not finite";
        return false;
    }

    mpjpe_px = mpjpe;
    return true;
}

} // namespace

/* BEFORE TO OPEN THE DOCKER REMBER TO:
 * 1. xhost +local:docker
 * 2. docker compose up
 * New terminal:
 * 3. docker exec -it moveEnet_flow sh
 * 4. start yarp server: yarpserver &
 * Attach Vs to running container
 * 5. cd /home/moveEnetFlow/build
 * 6. cmake ..
 * 7. make
 */

/**
 * Utility wrapper for communicating with MoveNet using YARP ports.
 *
 * The class manages the request/response flow to MoveNet: it publishes
 * transformed frames to a YARP source port and reads the skeleton prediction
 * from a YARP sink port.
 */
class offlineDetector
{
private:
    double period{0.001};  ///< Minimum interval between requests (sec) and last send timestamp

    BufferedPort<ImageOf<PixelMono>> output_port;  ///< Port used to send images to MoveNet
    BufferedPort<Bottle> input_port;               ///< Port used to receive skeleton responses

public:
    /**
     * Opens YARP ports and configures the desired request rate.
     *
     * @param output_name YARP name for the port that sends frames to the model
     * @param input_name YARP name for the port that receives MoveNet skeletons
     * @param rate Requested number of model updates per second (must be > 0)
     * @return true if both ports opened successfully and rate was valid
     */
    bool init(std::string output_name, std::string input_name, double rate)
    {
        if (!output_port.open(output_name))
            return false;

        if (!input_port.open(input_name))
            return false;

        if (rate <= 0.0)
            return false;

        period = 1.0 / rate;
        return true;
    }

    /**
     * Closes the YARP ports to free resources when the detector is no longer needed.
     */
    void close()
    {
        output_port.close();
        input_port.close();
    }

    /**
     * Sends the latest processed image to MoveNet and blocks until a skeleton reply
     * arrives (or a timeout occurs internally in YARP). The method also enforces the
     * configured update rate by delaying between calls based on wall-clock time.
     *
     * @param latest_image Grayscale frame to send (already prepared image)
     * @param latest_ts Timestamp that should be copied into the returned skeleton
     * @param previous_skeleton Output reference populated with the prediction
     * @return true when a skeleton was received; false on timeout or error
     */
    bool update(const cv::Mat &latest_image, double latest_ts, hpecore::stampedPose &previous_skeleton)
    {
        // NOTE: wall-clock rate limiting removed for offline replay (runs as fast as possible)

        // transform latest_image into MoveNet-compatible monochrome frame
        static cv::Mat cv_image;
        latest_image.convertTo(cv_image, CV_8U);
        cv::GaussianBlur(cv_image, cv_image, cv::Size(5, 5), 0, 0);
        output_port.prepare().copy(yarp::cv::fromCvMat<PixelMono>(cv_image));
        output_port.write();

        const double req_wall_ts = yarp::os::Time::now();           // Timestamp when request was sent (for latency measurement)

        // wait (blocking) for MoveNet response, then populate the stamped pose
        Bottle *mn_container = input_port.read(true);
        if (mn_container)
        {
            previous_skeleton.pose = hpecore::extractSkeletonFromYARP<Bottle>(*mn_container);
            previous_skeleton.conf = hpecore::extractConfidenceFromYARP<Bottle>(*mn_container);
            previous_skeleton.timestamp = latest_ts;
            previous_skeleton.delay = yarp::os::Time::now() - req_wall_ts;
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
        ss << "Usage: moveEnetOFK_offline_mpjpe [options]\n\n";
        ss << "Options:\n";
        ss << std::left << std::setw(20) << "--data_file" << std::setw(12) << "<string>" << ": path to input dataset file\n";
        ss << std::left << std::setw(20) << "--gt_file" << std::setw(12) << "<string>" << ": optional GT path; defaults to inferred .../ch0GT200Hzskeleton/data.log\n";
        ss << std::left << std::setw(20) << "--output_period" << std::setw(12) << "<double>" << ": MPJPE sampling period (s), default 0.005\n";
        ss << std::left << std::setw(20) << "--net_period" << std::setw(12) << "<double>" << ": model update period\n";
        ss << std::left << std::setw(20) << "--flow_period" << std::setw(12) << "<double>" << ": optical flow update period\n";
        ss << std::left << std::setw(20) << "--moveenet_only" << std::setw(12) << "" << ": disable optical-flow/KF velocity update and use MoveNet detections only\n";
        ss << std::left << std::setw(20) << "--h" << std::setw(12) << "<int>" << ": height of image\n";
        ss << std::left << std::setw(20) << "--w" << std::setw(12) << "<int>" << ": width of image\n";
        ss << std::left << std::setw(20) << "--pu" << std::setw(12) << "<double>" << ": KF process uncertainty\n";
        ss << std::left << std::setw(20) << "--muD" << std::setw(12) << "<double>" << ": KF measurement uncertainty (position)\n";
        ss << std::left << std::setw(20) << "--muV" << std::setw(12) << "<double>" << ": KF measurement uncertainty (velocity)\n";
        ss << std::left << std::setw(20) << "--roi" << std::setw(12) << "<int>" << ": ROI size for velocity estimation\n";
        ss << std::left << std::setw(20) << "--use_lc" << std::setw(12) << "<bool>" << ": use latency compensation in KF\n";
        ss << std::left << std::setw(20) << "--vis" << std::setw(12) << "" << ": enable on-screen visualization\n";
        ss << std::left << std::setw(20) << "--output_video" << std::setw(12) << "<string>" << ": path to output video file (.mp4)\n";
        ss << std::left << std::setw(20) << "--no_video" << std::setw(12) << "" << ": disable video output\n";
        ss << std::left << std::setw(20) << "--pwrjlr_file" << std::setw(12) << "<string>" << ": base path for PowerJoular output (no extension)\n";
        ss << std::left << std::setw(20) << "--gpu_file" << std::setw(12) << "<string>" << ": output CSV file for NVIDIA GPU telemetry\n";
        ss << std::left << std::setw(20) << "--gpu_period_ms" << std::setw(12) << "<int>" << ": nvidia-smi sampling period in ms (default 5)\n";
        ss << std::left << std::setw(20) << "--gpu_index" << std::setw(12) << "<int>" << ": NVIDIA GPU index to monitor (default 0)\n";
        yInfo() << ss.str();
        // exit after printing help
        return 0;
    }

    // Read parameters from command line with default values
    std::string datapath_file = rf.check("data_file", Value("/data/moveEnet_test/raw/cam2_S1_Directions/ch0dvs/data.log")).asString();
    std::string gt_file = rf.check("gt_file", Value("")).asString();
    double output_period = rf.check("output_period", Value(0.005)).asFloat64();                    // MPJPE sampling period
    double net_period = rf.check("net_period", Value(0.05)).asFloat64();                            // Range from 5ms to 100ms -> 200 Hz to 10 Hz   
    double flow_period = rf.check("flow_period", Value(0.005)).asFloat64();                         // Range from 5ms to 100ms -> 200 Hz to 10 Hz
    bool moveenet_only = rf.check("moveenet_only");                                                  // If true, skip optical-flow/KF velocity update
    cv::Size res(rf.check("w", Value(640)).asInt32(), rf.check("h", Value(480)).asInt32());
    double procU = rf.check("pu", Value(1e-1)).asFloat64();                                         // Process uncertainty
    double measUD = rf.check("muD", Value(1e-4)).asFloat64();                                       // Measurement uncertainty (position)
    double measUV = rf.check("muV", Value(0.0)).asFloat64();                                        // Measurement uncertainty (velocity)
    int roiSize = rf.check("roi", Value(20)).asInt32();                                             // ROI size for velocity estimation
    bool latency_compensation = rf.check("use_lc", Value(false)).asBool();                          // Latency compensation flag
    bool is_visualize = rf.check("vis");                                                            // Visualization flag
    std::string output_video = rf.check("output_video", Value("")).asString();
    bool no_video = rf.check("no_video");
    // std::string powerjoular_file = rf.check("pwrjlr_file", Value("/home/moveEnetFlow/pwr_cpu_file/single_test/20260223_testcpu_pwr")).asString();
    std::string powerjoular_file = rf.check("pwrjlr_file", Value("")).asString();
    std::string gpu_monitor_file = rf.check("gpu_file", Value("/home/moveEnetFlow/pwr_gpu_file/single_test/20260223_testgpu_pwr")).asString();
    int gpu_monitor_period_ms = rf.check("gpu_period_ms", Value(5)).asInt32();
    int gpu_monitor_index = rf.check("gpu_index", Value(0)).asInt32();

    std::string resolved_gt_file = gt_file;
    if (resolved_gt_file.empty() && !deriveGTPathFromDataFile(datapath_file, resolved_gt_file)) {
        yError() << "Could not infer GT path from --data_file. Expected suffix /ch0dvs/data.log";
        return -1;
    }

    GTSeries gt_series;
    std::string gt_error;
    if (!loadGTSeries(resolved_gt_file, gt_series, gt_error)) {
        yError() << "GT load failed:" << gt_error;
        return -1;
    }

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


    // ===== PREPARE VIDEO AND VISUALIZATION RESOURCES =====
    VisualizationContext vis_ctx;                      // Visualization and video writer resources

    // Visualization and video setup
    if (!initialiseVisualization(vis_ctx, res, is_visualize, no_video, output_video, datapath_file, output_period)) {
        return -1;
    }

    
    // ===== INITIALIZE ALGORITHMIC COMPONENTS =====

    // Inizialize event handlers and variables
    ev::EROS eros;                                  // EROS event surface handler
    eros.init(res.width, res.height, 7, 0.3);       // Initialize EROS surface

    //Initialize event loader
    ev::offlineLoader<ev::AE> eloader;              // Offline event loader
    
    //Initialize MoveEnet handler
    offlineDetector mn_handler;                    // MoveEnet handler
    hpecore::stampedPose detected_pose;            // Detected pose from MoveEnet

    // Kalman filter components
    hpecore::multiJointLatComp state;               // Kalman filter for pose state fusion
    hpecore::pwtripletvelocity velocity_estimator;  // Velocity estimator
    
    // SAE surface handler
    hpecore::SAE sae_handler;                       // SAE event surface handler
    sae_handler.init(res.width, res.height);        // Initialize SAE surface

    double lc = latency_compensation ? 1.0 : 0.0;  // Latency compensation flag
    if (!state.initialise({procU, measUD, measUV, lc})) {
        yError() << "Kalman filter initialization failed";
        return -1;
    }

    // MoveEnet checkpoint path
    std::string checkpoint_path = rf.check("checkpoint_path", Value("/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth")).asString();

    // Detection frequency detF derived from moveEnet update period
    double detF = 1.0 / net_period;                     // Detection frequency in Hz
    double tnow = 0.0;                                  // Current simulation time
    double next_net_upd = net_period;                   // event-time threshold for next MoveNet call
    double next_flow_upd = flow_period;                 // event-time threshold for next OF+KF update
    double next_sample_upd = output_period;            // event-time threshold for next MPJPE sample
    double next_vis_upd = output_period;               // event-time threshold for next visualization frame

    // Load event data from file 
    yInfo() << "Loading data ... ";
    if(!eloader.load(datapath_file)) {
        yError() << "Could not open data events file";
        return -1;
    } else {
        yInfo() << eloader.getinfo();
    }


    // Open and connect to YARP
    if(!yarp::os::Network::checkNetwork()) {
        yError() << "Could not connect to YARP";
        return -1;
    }

    std::string eros_out = "/moveEnetOFK_offline_mpjpe/eros:o_" + std::to_string(::getpid()); 
    std::string movenet_in = "/moveEnetOFK_offline_mpjpe/movenet:i_" + std::to_string(::getpid());

    // Clear any lingering YARP port registrations
    system("pkill -f movenet_online.py >/dev/null 2>&1");
    //system("killall python3 >/dev/null 2>&1");
    sleep(2);
    system("yarp name unregister /movenet/img:i >/dev/null 2>&1");
    system("yarp name unregister /movenet/sklt:o >/dev/null 2>&1");

    // MoveEnet process launch
    std::string command = "python3 /usr/local/src/hpe-core/example/movenet/movenet_online.py --checkpoint_path " + checkpoint_path + " &";
    system(command.c_str());

     // check if moveEnet process started
    while (!yarp::os::NetworkBase::exists("/movenet/sklt:o"))
        sleep(1);
    yInfo() << "MoveEnet started correctly";

    // Init detector ports
    mn_handler.init(eros_out, movenet_in, detF);
    yarp::os::Network::connect("/movenet/sklt:o", movenet_in, "fast_tcp");
    yarp::os::Network::connect(eros_out, "/movenet/img:i", "fast_tcp");

    

    // ===== MAIN PROCESSING LOOP (packet-by-packet) =====

    const double packet_eps = 1e-6;             // minimal increment to advance loader cursor
    double next_packet_ts = 0.0;                // start from t=0; loader will advance on first increment
    int batch_count = 0;
    bool pending_detection = false;             // true when a new MoveNet result is waiting to correct the KF
    hpecore::skeleton13 jvs;                    // last known joint velocities (persistent across iterations)
    hpecore::skeleton13 filtered_pose;          // last known filtered pose (persistent across iterations)
    std::vector<PredSample> pred_samples;

    while (true) {
        
        if (!eloader.incrementReadTill(next_packet_ts)) {
            yInfo() << "Finished processing event file. Packets:" << batch_count;
            break;
        }

        if (eloader.begin() == eloader.end()) {
            // No packet yielded at this timestamp; try the next tick
            next_packet_ts += packet_eps;
            continue;
        }

        const double packet_ts = eloader.begin().timestamp();
        tnow = packet_ts;
        
        bool was_detected = false;
        bool did_flow_update = false;
        // NOTE: pending_detection is intentionally NOT reset here; it persists until the next flow update

        // Update EROS/SAE using exactly one packet of events
        int event_count = 0;
        for(ev::offlineLoader<ev::AE>::iterator v = eloader.begin(); v != eloader.end(); v++){
            eros.update(v->x, v->y);
            sae_handler.update(v->x, v->y, tnow);
            event_count++;
        }

        batch_count++;
        next_packet_ts = packet_ts + packet_eps;  // ask loader for strictly later packet

        // Skip empty packet payloads
        if (event_count == 0) {
            continue;
        }

        // Send EROS frame to MoveEnet every net_period
        if (tnow >= next_net_upd) {
            next_net_upd += net_period;
            // Pass EROS surface directly; update() handles the CV_8U conversion internally
            was_detected = mn_handler.update(eros.getSurface(), tnow, detected_pose);
            if (was_detected && hpecore::poseNonZero(detected_pose.pose))
                pending_detection = true;  // latch until the next flow update consumes it
        }

        // Optical flow update every flow_period
        if (tnow >= next_flow_upd) {
            next_flow_upd += flow_period;
            did_flow_update = true;
            // Kalman and velocity logic here

            if (pending_detection) {
                // Update Kalman filter with detected pose (correction step)
                if (state.poseIsInitialised())
                    state.updateFromPosition(detected_pose.pose, detected_pose.timestamp);
                else
                    state.set(detected_pose.pose, tnow);
                pending_detection = false;  // consumed
            }
            
            if (state.poseIsInitialised())
            {
                if (!moveenet_only) {
                    // Estimate velocities from SAE surface using current filtered pose
                    jvs = velocity_estimator.multi_area_velocity(sae_handler.getSurface(), tnow, state.query(), roiSize);

                    // Update Kalman filter with velocity (prediction step with optical flow)
                    state.setVelocity(jvs);
                    state.updateFromVelocity(jvs, tnow);
                }

                // Query current pose. In MoveNet-only mode this is detection-corrected KF state without OF update.
                filtered_pose = state.query();
            }
            else
            {
                // If Kalman filter not yet initialized, use detected pose as-is
                filtered_pose = detected_pose.pose;
            }
        }

        const bool snapshot_ready = did_flow_update && state.poseIsInitialised();
        // MPJPE sampling at output_period rate, independent of flow/MoveNet updates
        // Advance timer unconditionally so it doesn't stall before pose is initialised
        if (tnow >= next_sample_upd) {
            next_sample_upd += output_period;
            if (state.poseIsInitialised()) {
                PredSample sample;
                sample.ts = tnow;
                for (int j = 0; j < kNumJoints; ++j) {
                    sample.coords[2 * j] = filtered_pose[j].u;
                    sample.coords[2 * j + 1] = filtered_pose[j].v;
                }
                pred_samples.push_back(sample);
            }
        }

        // Visualization
        if ((is_visualize || (!output_video.empty() && !no_video)) && tnow >= next_vis_upd) {
            next_vis_upd += output_period;
            renderVisualizationFrame(vis_ctx, eros.getSurface(), state.poseIsInitialised(), filtered_pose, detected_pose, tnow);
            writeVisualizationFrame(vis_ctx, snapshot_ready);
            if (showVisualizationFrame(vis_ctx)) {
                yInfo() << "User requested stop";
                break;
            }
        }


    }


    int exit_code = 0;
    double mpjpe_px = std::numeric_limits<double>::quiet_NaN();
    std::string mpjpe_error;
    if (!computeMPJPE(pred_samples, gt_series, mpjpe_px, mpjpe_error)) {
        yError() << "Failed to compute MPJPE:" << mpjpe_error;
        exit_code = -1;
    } else {
        std::cout << std::fixed << std::setprecision(6) << "mpjpe_px=" << mpjpe_px << std::endl;
    }

    // Cleanup
    power_monitor.stop();
    mn_handler.close();
    yarp::os::Network::disconnect("/movenet/sklt:o", movenet_in, "fast_tcp");
    yarp::os::Network::disconnect(eros_out, "/movenet/img:i", "fast_tcp");

    closeVisualization(vis_ctx, output_video);

    // Kill only the MoveNet process (not all python3 processes, which would
    // also terminate any parent HPO/evaluation script running this binary)
    system("pkill -f movenet_online.py >/dev/null 2>&1");

    return exit_code;
}

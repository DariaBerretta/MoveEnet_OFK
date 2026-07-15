/*
    moveEnetOFK_offline.cpp

    Purpose
    - Offline replay pipeline that reads event data, generates EROS/SAE surfaces,
        sends periodic frames to MoveNet via YARP, fuses detections with an
        optical-flow + Kalman filter (OF+KF), logs CSV/video outputs, and optionally
        visualizes results.

    High-level flow
    1. Parse command-line parameters (ResourceFinder).
    2. Initialize telemetry, visualization, event loaders, surfaces, KF, and
         MoveNet YARP connections.
    3. Loop over event packets:
         - update EROS/SAE surfaces,
         - periodically send frames to MoveNet and read detections,
         - periodically run OF + KF updates,
         - log CSV/video and render visualization frames.
    4. Clean up resources and terminate MoveNet.


    BEFORE TO OPEN THE DOCKER REMBER TO:
    * 1.ls
    * 2. docker compose up
    * New terminal:
    * 3. docker exec -it moveEnet_flow sh
    * 4. start yarp server: yarpserver &
    * Attach Vs to running container
    * 5. cd /home/moveEnetFlow/build
    * 6. cmake ..
    * 7. make
*/

#// --- Includes -----------------------------------------------------------------
#include <yarp/os/all.h>
#include <yarp/cv/Cv.h>                     // needed for yarp::cv::fromCvMat
#include <yarp/sig/Image.h>                 // needed for BufferedPort<ImageOf<PixelMono>>
#include <event-driven/core.h>
#include <opencv2/opencv.hpp>
#include <event-driven/algs.h>
#include <event-driven/vis.h>
#include <hpe-core/utility.h>
#include <hpe-core/motion.h>                // For hpecore::pwtripletvelocity
#include <hpe-core/fusion.h>                // For hpecore::multiJointLatComp
#include <hpe-core/representations.h>       // For hpecore::SAE
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include "utils/power_monitor.h"

static std::string shellQuote(const std::string &s)
{
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\'\''";
        else out += c;
    }
    out += "'";
    return out;
}
#include "utils/device_utils.h"
#include "utils/visualization_utils.h"

// --- Usings -------------------------------------------------------------------
using namespace yarp::os;
using namespace yarp::sig;
using std::string;
using yarp::os::Value;


// --- Types / Helpers ----------------------------------------------------------
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
    double period{0.001};  // Minimum interval between requests (sec) and last send timestamp
    cv::Size sensor_size;  // Logical image size used by the C++ pipeline
    cv::Size movenet_frame_size;  // Image size sent through YARP to MoveNet

    BufferedPort<ImageOf<PixelMono>> output_port;  // Port used to send images to MoveNet
    BufferedPort<Bottle> input_port;               // Port used to receive skeleton responses

public:
    /**
     * Opens YARP ports and configures the desired request rate.
     *
     * @param output_name YARP name for the port that sends frames to the model
     * @param input_name YARP name for the port that receives MoveNet skeletons
     * @param rate Requested number of model updates per second (must be > 0)
     * @param logical_size Size of the event-camera canvas used by the C++ pipeline
     * @param transport_size Size of the image sent to MoveNet over YARP
     * @return true if both ports opened successfully and rate was valid
     */
    bool init(std::string output_name, std::string input_name, double rate,
              const cv::Size &logical_size, const cv::Size &transport_size)
    {
        if (!output_port.open(output_name))
            return false;

        if (!input_port.open(input_name))
            return false;

        if (rate <= 0.0)
            return false;

        period = 1.0 / rate;
        sensor_size = logical_size;
        movenet_frame_size = transport_size;
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
        static cv::Mat padded_image;
        latest_image.convertTo(cv_image, CV_8U);
        cv::GaussianBlur(cv_image, cv_image, cv::Size(5, 5), 0, 0);

        cv::Mat frame_to_send = cv_image;
        if (movenet_frame_size.area() > 0 && movenet_frame_size != cv_image.size()) {
            if (cv_image.cols <= movenet_frame_size.width && cv_image.rows <= movenet_frame_size.height) {
                padded_image = cv::Mat::zeros(movenet_frame_size, CV_8U);
                cv_image.copyTo(padded_image(cv::Rect(0, 0, cv_image.cols, cv_image.rows)));
                frame_to_send = padded_image;
            } else {
                cv::resize(cv_image, padded_image, movenet_frame_size, 0, 0, cv::INTER_AREA);
                frame_to_send = padded_image;
            }
        }

        output_port.prepare().copy(yarp::cv::fromCvMat<PixelMono>(frame_to_send));
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

            if (sensor_size.area() > 0 && movenet_frame_size.area() > 0 && sensor_size != movenet_frame_size) {
                for (auto &joint : previous_skeleton.pose) {
                    joint.u = std::clamp(joint.u, 0.0f, static_cast<float>(sensor_size.width - 1));
                    joint.v = std::clamp(joint.v, 0.0f, static_cast<float>(sensor_size.height - 1));
                }
            }
        }

        return mn_container != nullptr;
    }
};

// --- Main ---------------------------------------------------------------------
int main(int argc, char *argv[]){
    
    // Prepare and configure the resource finder
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    if(rf.check("help")) {
        std::stringstream ss;
        ss << "Usage: moveEnetOFK_offline [options]\n\n";
        ss << "Options:\n";
        ss << std::left << std::setw(20) << "--data_file" << std::setw(12) << "<string>" << ": path to input dataset file\n";
        ss << std::left << std::setw(20) << "--output_file" << std::setw(12) << "<string>" << ": output path file\n";
        ss << std::left << std::setw(20) << "--output_period" << std::setw(12) << "<double>" << ": CSV writing output period (s), default 0.005\n";
        ss << std::left << std::setw(20) << "--net_period" << std::setw(12) << "<double>" << ": Model detection period\n";
        ss << std::left << std::setw(20) << "--flow_period" << std::setw(12) << "<double>" << ": Optical Flow update period\n";
        ss << std::left << std::setw(20) << "--moveenet_only" << std::setw(12) << ""
            << ": disable optical-flow/KF velocity update and use MoveNet detections only\n";
        ss << std::left << std::setw(20) << "--h" << std::setw(12) << "<int>" << ": height of image\n";
        ss << std::left << std::setw(20) << "--w" << std::setw(12) << "<int>" << ": width of image\n";
        ss << std::left << std::setw(20) << "--dhp19" << std::setw(12) << ""
            << ": use DHP19 sensor size (346x260); pad MoveNet transport to 352x260\n";
        ss << std::left << std::setw(20) << "--pu" << std::setw(12) << "<double>" << ": KF process uncertainty\n";
        ss << std::left << std::setw(20) << "--muD" << std::setw(12) << "<double>" << ": KF measurement uncertainty (position)\n";
        ss << std::left << std::setw(20) << "--muV" << std::setw(12) << "<double>" << ": KF measurement uncertainty (velocity)\n";
        ss << std::left << std::setw(20) << "--roi" << std::setw(12) << "<int>" << ": ROI size for velocity estimation\n";
        ss << std::left << std::setw(20) << "--use_lc" << std::setw(12) << "<bool>" << ": use latency compensation in KF\n";
        ss << std::left << std::setw(20) << "--vis" << std::setw(12) << "" << ": enable on-screen visualization\n";
        ss << std::left << std::setw(20) << "--output_csv_f" << std::setw(12) << "<string>" << ": path to output csv file\n";
        ss << std::left << std::setw(20) << "--include_velocities" << std::setw(12) << "" << ": when eval_format is set, also log velocities\n";
        ss << std::left << std::setw(20) << "--no_csv" << std::setw(12) << "" << ": skip CSV logging\n";
        ss << std::left << std::setw(20) << "--output_video" << std::setw(12) << "<string>"
            << ": path to output video file (.mp4)\n";
        ss << std::left << std::setw(20) << "--no_video" << std::setw(12) << "" << ": disable video output\n";
        ss << std::left << std::setw(20) << "--pwrjlr_file" << std::setw(12) << "<string>"
            << ": base path for PowerJoular output (no extension)\n";
        ss << std::left << std::setw(20) << "--gpu_file" << std::setw(12) << "<string>"
            << ": output CSV file for NVIDIA GPU telemetry\n";
        ss << std::left << std::setw(20) << "--gpu_period_ms" << std::setw(12) << "<int>"
            << ": nvidia-smi sampling period in ms (default 5)\n";
        
        ss << std::left << std::setw(20) << "--gpu_index" << std::setw(12) << "<int>" << ": NVIDIA GPU index to monitor (default 0)\n";
        ss << std::left << std::setw(20) << "--device" << std::setw(12) << "<string>" << ": device for MoveNet sidecar (e.g. cpu or cuda:0)\n";
        yInfo() << ss.str();
        return 0;
    }

    // --- Params / Config ----------------------------------------------------
    // Read parameters from command line with default values
    // std::string datapath_file = rf.check("data_file", Value("/data/moveEnet_test/raw/cam2_S11_Directions_1/ch0dvs/data.log")).asString();
    std::string datapath_file = rf.check("data_file", Value("/data/dhp19_testing_set_S13toS17/S13_1_1/ch3dvs/data.log")).asString();
    std::string output_file = rf.check("output_file", Value("/tmp/output.csv")).asString();
    double output_period = rf.check("output_period", Value(0.005)).asFloat64();                    // CSV write period
    double net_period = rf.check("net_period", Value(0.05)).asFloat64();                            // Range from 5ms to 100ms -> 200 Hz to 10 Hz   
    double flow_period = rf.check("flow_period", Value(0.005)).asFloat64();                         // Range from 5ms to 100ms -> 200 Hz to 10 Hz
    bool moveenet_only = rf.check("moveenet_only");                                                  // If true, skip optical-flow/KF velocity update
    bool use_dhp19_size = rf.check("dhp19");                                    // Accept common typo as an alias
    
    cv::Size res;          // Actual sensor/event-surface resolution
    cv::Size movenet_res;  // Image resolution transported through YARP

    if (use_dhp19_size) {
        res = cv::Size(346, 260);

        // YARP-compatible transport width.
        // The six extra columns are zero padding on the right.
        movenet_res = cv::Size(352, 260);
    }
    else {
        res = cv::Size(
            rf.check("w", Value(640)).asInt32(),
            rf.check("h", Value(480)).asInt32()
        );

        movenet_res = res;
    }

    double procU = rf.check("pu", Value(0.77)).asFloat64();                                         // Process uncertainty
    double measUD = rf.check("muD", Value(0.06)).asFloat64();                                       // Measurement uncertainty (position)
    double measUV = rf.check("muV", Value(0.97)).asFloat64();                                       // Measurement uncertainty (velocity)
    int roiSize = rf.check("roi", Value(20)).asInt32();                                             // ROI size for velocity estimation

    bool latency_compensation = rf.check("use_lc", Value(true)).asBool();                          // Latency compensation flag
    bool is_visualize = rf.check("vis");                                                            // Visualization flag
    std::string output_csv_f = rf.check("output_csv_f", Value("/tmp/output.csv")).asString();
    bool include_velocities = rf.check("include_velocities");
    bool no_csv = rf.check("no_csv");
    std::string output_video = rf.check("output_video", Value("/tmp/output.mp4")).asString();
    bool no_video = rf.check("no_video");
    
    //std::string powerjoular_file = rf.check("pwrjlr_file", Value("/tmp/powerjoular.csv")).asString();
    std::string gpu_monitor_file = rf.check("gpu_file", Value("/tmp/gpu_monitor.csv")).asString();
    int gpu_monitor_period_ms = rf.check("gpu_period_ms", Value(5)).asInt32();
    int gpu_monitor_index = rf.check("gpu_index", Value(0)).asInt32();
    std::string device = rf.check("device", Value("cuda:0")).asString();

    DeviceConfig dev_cfg = parseDeviceConfig(device);
    if (dev_cfg.use_gpu) {
        gpu_monitor_index = dev_cfg.gpu_id;
    }
   

    // --- Power / Monitoring -------------------------------------------------
    PowerMonitor power_monitor;
    PowerMonitorConfig power_cfg;
    // power_cfg.powerjoular_file = powerjoular_file;
    power_cfg.gpu_file = gpu_monitor_file;
    power_cfg.gpu_period_ms = gpu_monitor_period_ms;
    power_cfg.gpu_index = gpu_monitor_index;
    power_cfg.target_pid = ::getpid();
    if (!power_monitor.start(power_cfg)) {
        return -1;
    }


    // --- CSV / Visualization ------------------------------------------------
    // Prepare CSV, video, and visualization resources
    std::ofstream csv_file;                             // CSV file stream for logging results
    std::vector<std::string> csv_buffer;                // store rows for deferred write
    VisualizationContext vis_ctx;                      // Visualization and video writer resources

    // CSV writer setup
    if (!no_csv) {
        csv_file.open(output_csv_f);
        if (!csv_file.is_open()) {
            yError() << "Could not open CSV file for writing:" << output_csv_f;
            return -1;
        }
        yInfo() << "CSV logging enabled ->" << output_csv_f;
        csv_file << "timestamp,latency";
        for (int j = 0; j < 13; j++) {
            csv_file << ",joint" << j << "_x,joint" << j << "_y";
        }
        if (include_velocities) {
            for (int j = 0; j < 13; j++) {
                csv_file << ",joint" << j << "_vx,joint" << j << "_vy";
            }
        }
        csv_file << "\n";
        csv_file.flush();
    }

    // Visualization and video setup, calls utils/visualization_utils.cpp
    if (!initialiseVisualization(vis_ctx, res, is_visualize, no_video, output_video, datapath_file, output_period, "MoveEnetOFK Visualisation")) {
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
    if(use_dhp19_size) {
        checkpoint_path = rf.check("checkpoint_path", Value("/usr/local/src/hpe-core/example/movenet/models/dhp19_allcams_e33_valacc0.87996.pth")).asString();
    }

    // Detection frequency detF derived from moveEnet update period
    double detF = 1.0 / net_period;                     // Detection frequency in Hz of MoveNet
    double tnow = 0.0;                                  // Current simulation time
    double next_net_upd = 0.0;                          // event-time threshold for next MoveNet call
    double next_flow_upd = 0.0;                         // event-time threshold for next OF+KF update
    double next_csv_upd = 0.0;                          // event-time threshold for next CSV row
    double next_vis_upd = 0.0;                          // event-time threshold for next visualization frame

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

    std::string eros_out = "/moveEnetOFK_offline/eros:o_" + std::to_string(::getpid()); 
    std::string movenet_in = "/moveEnetOFK_offline/movenet:i_" + std::to_string(::getpid());

    // Clear any lingering YARP port registrations
    system("pkill -f movenet_online.py >/dev/null 2>&1");
    sleep(2);
    system("yarp name unregister /movenet/img:i >/dev/null 2>&1");
    system("yarp name unregister /movenet/sklt:o >/dev/null 2>&1");

    // MoveEnet process launch, 
    // if (use_dhp19_size) {
    //     yInfo() << "DHP19 mode: C++ canvas" << res.width << "x" << res.height         
    //             << ", MoveNet YARP transport" << res.width << "x" << res.height;
    // }
    
    std::ostringstream cmd;
    cmd << "python3 /usr/local/src/hpe-core/example/movenet/movenet_online.py --checkpoint_path " << shellQuote(checkpoint_path)
        << " --w " << movenet_res.width << " --h " << movenet_res.height;

    if (dev_cfg.use_gpu) {
        cmd << " --gpu --GPU_ID " << dev_cfg.gpu_id;
    }

    cmd << " &";
    system(cmd.str().c_str());

     // check if moveEnet process started
    while (!yarp::os::NetworkBase::exists("/movenet/sklt:o"))
        sleep(1);
    yInfo() << "MoveEnet started correctly";

    // Init detector ports
    mn_handler.init(eros_out, movenet_in, detF, res, movenet_res);
    yarp::os::Network::connect("/movenet/sklt:o", movenet_in, "fast_tcp");
    yarp::os::Network::connect(eros_out, "/movenet/img:i", "fast_tcp");

    

    // ===== MAIN PROCESSING LOOP (packet-by-packet) =====

    const double packet_eps = 1e-6;             // minimal increment to advance loader cursor
    double next_packet_ts = 0.0;                // start from t=0; loader will advance on first increment
    int batch_count = 0;
    bool pending_detection = false;             // true when a new MoveNet result is waiting to correct the KF
    hpecore::skeleton13 jvs;                    // last known joint velocities (persistent across iterations)
    hpecore::skeleton13 filtered_pose;          // last known filtered pose (persistent across iterations)

    // For true movenet-only mode
    bool movenet_pose_available = false;
    hpecore::skeleton13 movenet_only_pose;
    double movenet_only_latency = 0.0;

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
        // if (tnow >= next_net_upd) {
        //     next_net_upd += net_period;
        //     // Pass EROS surface directly; update() handles the CV_8U conversion internally
        //     was_detected = mn_handler.update(eros.getSurface(), tnow, detected_pose);
        //     if (was_detected && hpecore::poseNonZero(detected_pose.pose))
        //         pending_detection = true;  // latch until the next flow update consumes it
        // }
        if (tnow >= next_net_upd) {
            next_net_upd = net_period + tnow;

            was_detected = mn_handler.update(eros.getSurface(), tnow, detected_pose);

            if (was_detected && hpecore::poseNonZero(detected_pose.pose)) {
                if (moveenet_only) {
                    movenet_only_pose = detected_pose.pose;
                    movenet_only_latency = detected_pose.delay;
                    movenet_pose_available = true;
                } else {
                    pending_detection = true;
                }
            }
        }

        // SECOND VERISON:

        if (!moveenet_only && tnow >= next_flow_upd) {
            next_flow_upd = flow_period + tnow;
            did_flow_update = true;

            // 1. Use latest valid MoveNet pose, if available, to initialize/correct KF
            if (pending_detection) {
                if (state.poseIsInitialised()) {
                    state.updateFromPosition(detected_pose.pose, detected_pose.timestamp);
                } else {
                    state.set(detected_pose.pose, detected_pose.timestamp);
                }

                pending_detection = false;
            }

            // If no KF state exists yet, no fusion step can be performed
            if (!state.poseIsInitialised()) {
                continue;
            }

            // 2. Query corrected state
            hpecore::skeleton13 corrected_pose = state.query();

            // 3. Use corrected state as ROI/reference for optical-flow velocity estimation
            jvs = velocity_estimator.multi_area_velocity(
                sae_handler.getSurface(),
                tnow,
                corrected_pose,
                roiSize
            );

            // 4. Feed estimated joint velocities back into KF
            state.setVelocity(jvs);
            state.updateFromVelocity(jvs, tnow);

            // 5. Query KF state for output
            filtered_pose = state.query();
        }


        // // FIRST VERSION:
        // // Optical flow update every flow_period
        // if (!moveenet_only && tnow >= next_flow_upd) {
        //     next_flow_upd = flow_period + tnow;
        //     did_flow_update = true;
            
        //     // Kalman and velocity logic here
        //     // 1. Use latest valid MoveNet pose, if available, to initialize/correct KF
        //     if (pending_detection) {
        //         // Update Kalman filter with detected pose (correction step)
        //         if (state.poseIsInitialised())
        //             state.updateFromPosition(detected_pose.pose, detected_pose.timestamp);
        //         else
        //             //state.set(detected_pose.pose, tnow);
        //             state.set(detected_pose.pose, detected_pose.timestamp);
                
        //         pending_detection = false;  // consumed
        //     }
            

        //     // 3. Use corrected state as ROI/reference for optical-flow velocity estimation
        //     if (state.poseIsInitialised())
        //     {
        //         // Estimate velocities from SAE surface using current filtered pose
        //         jvs = velocity_estimator.multi_area_velocity(sae_handler.getSurface(), tnow, state.query(), roiSize);

        //         // Update Kalman filter with velocity (prediction step with optical flow)
        //         state.setVelocity(jvs);
        //         state.updateFromVelocity(jvs, tnow);

        //         // Query current pose. In MoveNet-only mode this is detection-corrected KF state without OF update.
        //         filtered_pose = state.query();
        //     }
        //     else
        //     {
        //         // If Kalman filter not yet initialized, use detected pose as-is
        //         filtered_pose = detected_pose.pose;
        //     }
        // }

        const bool snapshot_ready = did_flow_update && state.poseIsInitialised();
        // CSV logging at output_period rate, independent of flow/MoveNet updates
        // Advance timer unconditionally so it doesn't stall before pose is initialised
        if (tnow >= next_csv_upd) {
            next_csv_upd = output_period + tnow;

            const bool ready =
                moveenet_only ? movenet_pose_available : state.poseIsInitialised();

            if (ready && csv_file.is_open()) {
                // const hpecore::skeleton13 &pose_to_write =
                //     moveenet_only ? movenet_only_pose : filtered_pose;
                const hpecore::skeleton13 &pose_to_write =
                    moveenet_only ? movenet_only_pose : state.query();

                const double lat =
                    moveenet_only ? movenet_only_latency :
                    ((detected_pose.timestamp > 0) ? detected_pose.delay : 0.0);

                std::ostringstream row;
                row << std::fixed << std::setprecision(6) << tnow;
                row << "," << lat;

                for (int j = 0; j < 13; j++) {
                    row << "," << pose_to_write[j].u << "," << pose_to_write[j].v;
                }

                if (include_velocities) {
                    for (int j = 0; j < 13; j++) {
                        if (moveenet_only)
                            row << ",0,0";
                        else
                            row << "," << jvs[j].u << "," << jvs[j].v;
                    }
                }

                csv_buffer.push_back(row.str());
            }
        }

        // Visualization
        if ((is_visualize || (!output_video.empty() && !no_video)) && tnow >= next_vis_upd) {
            next_vis_upd = output_period + tnow;
            // When running in MoveNet-only mode, prefer showing the raw MoveNet
            // prediction in the visualization instead of the KF/OF-corrected pose.
            hpecore::skeleton13 viz_pose = moveenet_only ? movenet_only_pose : filtered_pose;

            renderVisualizationFrame(
                vis_ctx,
                eros.getSurface(),
                moveenet_only ? movenet_pose_available : state.poseIsInitialised(),
                viz_pose,
                detected_pose,
                tnow
            );

            writeVisualizationFrame(
                vis_ctx,
                moveenet_only ? movenet_pose_available : snapshot_ready
            );
            
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
        yInfo() << "CSV rows written:" << csv_buffer.size() << "->" << output_csv_f;
    }
    power_monitor.stop();
    mn_handler.close();
    yarp::os::Network::disconnect("/movenet/sklt:o", movenet_in, "fast_tcp");
    yarp::os::Network::disconnect(eros_out, "/movenet/img:i", "fast_tcp");

    closeVisualization(vis_ctx, output_video);

    // Kill only the MoveNet process (not all python3 processes, which would
    // also terminate any parent HPO/evaluation script running this binary)
    system("pkill -f movenet_online.py >/dev/null 2>&1");

    return 0;
}

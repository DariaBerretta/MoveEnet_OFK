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
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <filesystem>
#include <ctime>

using namespace yarp::os;
using namespace yarp::sig;
using std::string;
using yarp::os::Value;

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
        ss << "Usage: moveEnetOFK_offline [options]\n\n";
        ss << "Options:\n";
        ss << std::left << std::setw(20) << "--data_file" << std::setw(12) << "<string>" << ": path to input dataset file\n";
        ss << std::left << std::setw(20) << "--output_file" << std::setw(12) << "<string>" << ": output path file\n";
        // ss << std::left << std::setw(20) << "--output_period" << std::setw(12) << "<double>" << ": interpolated GT rate\n";
        ss << std::left << std::setw(20) << "--net_period" << std::setw(12) << "<double>" << ": model update period\n";
        ss << std::left << std::setw(20) << "--flow_period" << std::setw(12) << "<double>" << ": optical flow update period\n";
        ss << std::left << std::setw(20) << "--h" << std::setw(12) << "<int>" << ": height of image\n";
        ss << std::left << std::setw(20) << "--w" << std::setw(12) << "<int>" << ": width of image\n";
        ss << std::left << std::setw(20) << "--pu" << std::setw(12) << "<double>" << ": KF process uncertainty\n";
        ss << std::left << std::setw(20) << "--muD" << std::setw(12) << "<double>" << ": KF measurement uncertainty (position)\n";
        ss << std::left << std::setw(20) << "--muV" << std::setw(12) << "<double>" << ": KF measurement uncertainty (velocity)\n";
        ss << std::left << std::setw(20) << "--roi" << std::setw(12) << "<int>" << ": ROI size for velocity estimation\n";
        ss << std::left << std::setw(20) << "--use_lc" << std::setw(12) << "<bool>" << ": use latency compensation in KF\n";
        ss << std::left << std::setw(20) << "--vis" << std::setw(12) << "" << ": enable on-screen visualization\n";
        ss << std::left << std::setw(20) << "--output_csv" << std::setw(12) << "<string>" << ": path to output csv file\n";
        ss << std::left << std::setw(20) << "--eval_format" << std::setw(12) << "" << ": output CSV in evaluate_hpe.py format\n";
        ss << std::left << std::setw(20) << "--include_velocities" << std::setw(12) << "" << ": when eval_format is set, also log velocities\n";
        ss << std::left << std::setw(20) << "--no_csv" << std::setw(12) << "" << ": skip CSV logging\n";
        ss << std::left << std::setw(20) << "--output_video" << std::setw(12) << "<string>" << ": path to output video file (.mp4)\n";
        ss << std::left << std::setw(20) << "--no_video" << std::setw(12) << "" << ": disable video output\n";
        yInfo() << ss.str();
        // exit after printing help
        return 0;
    }

    // Read parameters from command line with default values
    std::string datapath_file = rf.check("data_file", Value("/data/new_scarfGNN_full/raw/cam2_S8_Discussion/ch0dvs/data.log")).asString();
    std::string output_file = rf.check("output_file", Value("/home/scarf_images/")).asString();
    // double output_period = rf.check("output_period", Value(0.005)).asFloat64();                     // 5ms -> 200 Hz
    double net_period = rf.check("net_period", Value(0.05)).asFloat64();                            // Range from 5ms to 100ms -> 200 Hz to 10 Hz   
    double flow_period = rf.check("flow_period", Value(0.005)).asFloat64();                         // Range from 5ms to 100ms -> 200 Hz to 10 Hz
    cv::Size res(rf.check("w", Value(640)).asInt32(), rf.check("h", Value(480)).asInt32());
    double procU = rf.check("pu", Value(1e-1)).asFloat64();                                         // Process uncertainty
    double measUD = rf.check("muD", Value(1e-4)).asFloat64();                                       // Measurement uncertainty (position)
    double measUV = rf.check("muV", Value(0.0)).asFloat64();                                        // Measurement uncertainty (velocity)
    int roiSize = rf.check("roi", Value(20)).asInt32();                                             // ROI size for velocity estimation
    bool latency_compensation = rf.check("use_lc", Value(false)).asBool();                          // Latency compensation flag
    bool is_visualize = rf.check("vis");                                                            // Visualization flag
    std::string output_csv = rf.check("output_csv", Value("/home/moveEnetFlow/csv_file/test_moveEnetOFK_offline.csv")).asString();
    bool eval_format = rf.check("eval_format");
    bool include_velocities = rf.check("include_velocities");
    bool no_csv = rf.check("no_csv");
    std::string output_video = rf.check("output_video", Value("")).asString();
    bool no_video = rf.check("no_video");


    // ===== PREPARE CSV, VIDEO, AND VISUALIZATION RESOURCES =====
    std::ofstream csv_file;                             // CSV file stream for logging results
    std::vector<std::string> csv_buffer;                // store rows for deferred write
    cv::VideoWriter video_writer;                       // Video writer for output video file

    // CSV writer setup
    if (!no_csv) {
        csv_file.open(output_csv);
        if (!csv_file.is_open()) {
            yError() << "Could not open CSV file for writing:" << output_csv;
            return -1;
        }
        yInfo() << "CSV logging enabled ->" << output_csv;
        if (eval_format) {
            csv_file << "timestamp,latency";
            for (int j = 0; j < 13; j++) {
                csv_file << ",joint" << j << "_x,joint" << j << "_y";
            }
            if (include_velocities) {
                for (int j = 0; j < 13; j++) {
                    csv_file << ",joint" << j << "_vx,joint" << j << "_vy";
                }
            }
        } else {
            csv_file << "timestamp";
            for (int j = 0; j < 13; j++) {
                csv_file << ",joint" << j << "_x,joint" << j << "_y,joint" << j << "_vx,joint" << j << "_vy,confidence" << j;
            }
        }
        csv_file << "\n";
        csv_file.flush();
    }

    // Visualization setup
    cv::Mat canvas;
    if (is_visualize || (!output_video.empty() && !no_video)) {
        canvas = cv::Mat(res, CV_8UC3);
    }

    if (is_visualize) {
        cv::namedWindow("moveEnetOFK_offline", cv::WINDOW_NORMAL);
        cv::resizeWindow("moveEnetOFK_offline", res);
    }

    // Generate video filename if not specified and video is enabled
    if (output_video.empty() && !no_video) {
        
        // Get current date
        std::time_t now = std::time(nullptr);
        std::tm* tm = std::localtime(&now);
        char date_str[11];
        std::strftime(date_str, sizeof(date_str), "%Y-%m-%d", tm);

        // Get folder name from output_csv path
        std::filesystem::path csv_path(datapath_file);
        std::string folder_name = csv_path.parent_path().parent_path().filename().string();

        // Create video filename
        output_video = std::string("/home/moveEnetFlow/mp4_files/") + date_str + "_" + folder_name + ".mp4";
    }

    // Video writer setup
    if (!output_video.empty()) {
        // Create output directory if it doesn't exist
        std::filesystem::path video_path(output_video);
        std::filesystem::create_directories(video_path.parent_path());

        int fps = static_cast<int>(std::max(1.0, 1.0 / flow_period));
        video_writer.open(output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, res);
        if (!video_writer.isOpened()) {
            yError() << "Could not open video writer for:" << output_video;
            return -1;
        }
        yInfo() << "Video output enabled ->" << output_video << " at " << fps << " FPS";
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
    double net_accum = 0.0;                             // Accumulator for network (detection) period
    double flow_accum = 0.0;                            // Accumulator for flow (optical) period

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
    system("killall python3 >/dev/null 2>&1");
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

    double pts = 0.0;                           // previous packet timestamp
    const double packet_eps = 1e-6;             // minimal increment to advance loader cursor
    double next_packet_ts = 0.0;                // start from t=0; loader will advance on first increment
    int batch_count = 0;
    bool pending_detection = false;             // true when a new MoveNet result is waiting to correct the KF
    
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
        const double packet_dt = (batch_count == 0) ? 0.0 : std::max(0.0, packet_ts - pts);
        pts = packet_ts;
        tnow = packet_ts;
        
        bool was_detected = false;
        bool did_flow_update = false;
        // NOTE: pending_detection is intentionally NOT reset here; it persists until the next flow update
        hpecore::skeleton13 jvs;                        // Joint velocities
        hpecore::skeleton13 filtered_pose;              // Pose from Kalman filter

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
        net_accum += packet_dt;
        if (net_accum >= net_period) {
            net_accum = 0.0;  // Reset accumulator
            // Pass EROS surface directly; update() handles the CV_8U conversion internally
            was_detected = mn_handler.update(eros.getSurface(), tnow, detected_pose);
            if (was_detected && hpecore::poseNonZero(detected_pose.pose))
                pending_detection = true;  // latch until the next flow update consumes it
        }


        // Optical flow update every flow_period
        flow_accum += packet_dt;
        if (flow_accum >= flow_period) {
            flow_accum = 0.0;  // Reset accumulator
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
                // Estimate velocities from SAE surface using current filtered pose
                jvs = velocity_estimator.multi_area_velocity(sae_handler.getSurface(), tnow, state.query(), roiSize);
                
                // Update Kalman filter with velocity (prediction step with optical flow)
                state.setVelocity(jvs);
                state.updateFromVelocity(jvs, tnow);
                
                // Get the filtered pose from Kalman filter
                filtered_pose = state.query();
            }
            else
            {
                // If Kalman filter not yet initialized, use detected pose as-is
                filtered_pose = detected_pose.pose;
            }
        }

        const bool snapshot_ready = did_flow_update && state.poseIsInitialised();
        // CSV logging aligned with moveEnet_flowKalman
        if (snapshot_ready && csv_file.is_open()) {
            std::ostringstream row;
            row << std::fixed << std::setprecision(6) << tnow;
            if (eval_format) {
                double lat = (detected_pose.timestamp > 0) ? detected_pose.delay : 0.0;
                row << "," << lat;
                for (int j = 0; j < 13; j++) {
                    row << "," << filtered_pose[j].u << "," << filtered_pose[j].v;
                }
                if (include_velocities) {
                    for (int j = 0; j < 13; j++) {
                        row << "," << jvs[j].u << "," << jvs[j].v;
                    }
                }
            } else {
                for (int j = 0; j < 13; j++) {
                    row << "," << filtered_pose[j].u
                        << "," << filtered_pose[j].v
                        << "," << jvs[j].u
                        << "," << jvs[j].v
                        << "," << detected_pose.conf[j];
                }
            }
            csv_buffer.push_back(row.str());
        }

        // Visualization
        if (is_visualize || (!output_video.empty() && !no_video)) {
            cv::Mat eros_vis;
            eros.getSurface().convertTo(eros_vis, CV_8U);
            cv::GaussianBlur(eros_vis, eros_vis, {9, 9}, 0);
            cv::normalize(eros_vis, eros_vis, 0, 255, cv::NORM_MINMAX);
            cv::cvtColor(eros_vis, canvas, cv::COLOR_GRAY2BGR);

            if (state.poseIsInitialised()) {
                try {
                    hpecore::stampedPose pose_filtered;
                    pose_filtered.pose = state.query();
                    pose_filtered.timestamp = tnow;
                    pose_filtered.conf = detected_pose.conf; // ensure joints are drawn
                    hpecore::drawSkeleton(canvas, pose_filtered, {255, 0, 0}, 3, 0.0); // red = filtered
                } catch (const cv::Exception&) {
                    // skip drawing on error
                }
            }

            if (hpecore::poseNonZero(detected_pose.pose)) {
                try {
                    hpecore::stampedPose pose_raw = detected_pose;
                    hpecore::drawSkeleton(canvas, pose_raw, {0, 0, 255}, 2, 0.0); // blue = raw
                } catch (const cv::Exception&) {
                   // skip drawing on error
                }
            }

            if (video_writer.isOpened() && snapshot_ready) {
                video_writer.write(canvas);
            }

            if (is_visualize) {
                cv::imshow("moveEnetOFK_offline", canvas);
                char key_pressed = cv::waitKey(1);
                if (key_pressed == '\e' || key_pressed == 'q') {
                    yInfo() << "User requested stop";
                    break;
                }
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
    mn_handler.close();
    yarp::os::Network::disconnect("/movenet/sklt:o", movenet_in, "fast_tcp");
    yarp::os::Network::disconnect(eros_out, "/movenet/img:i", "fast_tcp");
    system("killall python3");  // Kill MoveNet process

    if (is_visualize) {
        cv::destroyAllWindows();
    }

    if (video_writer.isOpened()) {
        video_writer.release();
        yInfo() << "Video saved to:" << output_video;
    }
    
    return 0;
}
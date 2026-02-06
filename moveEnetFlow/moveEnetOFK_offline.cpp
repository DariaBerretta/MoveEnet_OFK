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

class externalDetector
{
private:
    double period{0.1}, tic{0.0};
    bool waiting{false};

    BufferedPort<ImageOf<PixelMono>> output_port;
    BufferedPort<Bottle> input_port;

public:
    bool init(std::string output_name, std::string input_name, double rate)
    {
        if (!output_port.open(output_name))
            return false;

        if (!input_port.open(input_name))
            return false;

        period = 1.0 / rate;
        return true;
    }
    void close()
    {
        output_port.close();
        input_port.close();
    }

    bool update(const cv::Mat &latest_image, double latest_ts, hpecore::stampedPose &previous_skeleton)
    {
        // send an update if the timer has elapsed
        if(latest_ts < tic) tic = latest_ts - 2.0;
        if ((!waiting && latest_ts - tic > period) || (latest_ts - tic > 2.0))
        {
            static cv::Mat cv_image;
            latest_image.convertTo(cv_image, CV_8U);
            cv::GaussianBlur(cv_image, cv_image, cv::Size(5, 5), 0, 0);
            output_port.prepare().copy(yarp::cv::fromCvMat<PixelMono>(cv_image));
            output_port.write();
            tic = latest_ts;
            waiting = true;
        }

        // read a ready data
        Bottle *mn_container = input_port.read(false);
        if (mn_container)
        {
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
    
    // prepare and configure the resource finder
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    if(rf.check("help")) {
        yInfo() << "--data_file\t<string>\t: path to input dataset file";
        yInfo() << "--output_file\t<string>\t: output path file";
        yInfo() << "--output_period\t<string>\t: interpolated GT rate";
        yInfo() << "--net_period\t<string>\t: model update period";
        yInfo() << "--flow_period\t<string>\t: optical flow update period";
        yInfo() << "--h\t<string>\t: height of image";
        yInfo() << "--w\t<string>\t: width of image";
        yInfo() << "--pu\t<string>\t: KF process uncertainty";
        yInfo() << "--muD\t<string>\t: KF measurement uncertainty (position)";
        yInfo() << "--muV\t<string>\t: KF measurement uncertainty (velocity)";
        yInfo() << "--roi\t<string>\t: ROI size for velocity estimation";
        yInfo() << "--use_lc\t<string>\t: use latency compensation in KF";
        yInfo() << "--vis\t\t: enable on-screen visualization";
        yInfo() << "--output_csv\t<string>\t: path to output csv file";
        yInfo() << "--eval_format\t\t: output CSV in evaluate_hpe.py format";
        yInfo() << "--include_velocities\t: when eval_format is set, also log velocities";
        yInfo() << "--no_csv\t\t: skip CSV logging";
    }

    // Read parameters from command line with default values
    std::string datapath_file = rf.check("data_file", Value("/data/new_scarfGNN_full/raw/cam2_S8_Discussion/ch0dvs/data.log")).asString();
    std::string output_file = rf.check("output_file", Value("/home/scarf_images/")).asString();
    double output_period = rf.check("output_period", Value(0.005)).asFloat64();                     // 5ms -> 200 Hz
    double net_period = rf.check("net_period", Value(0.005)).asFloat64();                           // Range from 5ms to 100ms -> 200 Hz to 10 Hz   
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

    std::ofstream csv_file;
    std::vector<std::string> csv_buffer;  // store rows for deferred write

    // Inizialize event handlers and variables
    ev::EROS eros;                                  // EROS event surface handler
    eros.init(res.width, res.height, 7, 0.3);       // Initialize EROS surface
    cv::Mat eros_frame;                             // EROS frame in cv::Mat format  to send to moveEnet

    //Initialize event loader
    ev::offlineLoader<ev::AE> eloader;              // Offline event loader
    
    //Initialize MoveEnet handler
    externalDetector mn_handler;                    // MoveEnet handler
    hpecore::stampedPose detected_pose;             // Detected pose from MoveEnet

    // Kalman filter components
    hpecore::multiJointLatComp state;               // Kalman filter for pose state fusion
    hpecore::pwtripletvelocity velocity_estimator;  // Velocity estimator
    
    // SAE surface handler
    hpecore::SAE sae_handler;                       // SAE event surface handler
    sae_handler.init(res.width, res.height);        // Initialize SAE surface
    // velocity_estimator.setParameters removed: multi_area_velocity accepts ROI per call

    double lc = latency_compensation ? 1.0 : 0.0;  // Latency compensation flag
    if (!state.initialise({procU, measUD, measUV, lc})) {
        yError() << "Kalman filter initialization failed";
        return -1;
    }

    // MoveEnet checkpoint path
    std::string checkpoint_path = rf.check("checkpoint_path", Value("/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth")).asString();

    // Detection frequency detF derived from output_period
    double detF = 1.0 / output_period;
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
    if (is_visualize) {
        canvas = cv::Mat(res, CV_8UC3);
        cv::namedWindow("moveEnetOFK_offline", cv::WINDOW_NORMAL);
        cv::resizeWindow("moveEnetOFK_offline", res);
    }

    // Main processing loop

    double pts = 0.0;  // previous timestamp guard
    int batch_count = 0;
    while (true) {
        tnow += output_period;
        
        bool was_detected = false;
        hpecore::skeleton13 jvs;                        // Joint velocities
        hpecore::skeleton13 filtered_pose;              // Pose from Kalman filter

        // Update EROS surface with events up to current time tnow
        eloader.incrementReadTill(tnow);
        int event_count = 0;
        for(ev::offlineLoader<ev::AE>::iterator v = eloader.begin(); v != eloader.end(); v++){
            eros.update(v->x, v->y);
            sae_handler.update(v->x, v->y, tnow);
            event_count++;
        }
        batch_count++;

        // Stop when no more events remain
        if (event_count == 0 && tnow > 0) {
            yInfo() << "Finished processing event file. Frames: " << batch_count;
            break;
        }

        // Send EROS frame to MoveEnet every net_period
        net_accum += output_period;
        if (net_accum >= net_period) {
            net_accum = 0.0;  // Reset accumulator
            // Convert EROS surface to cv::Mat and send to MoveEnet
            eros.getSurface().convertTo(eros_frame, CV_8U);
            was_detected = mn_handler.update(eros_frame, tnow, detected_pose);
        }


        // Optical flow update every flow_period
        flow_accum += output_period;
        if (flow_accum >= flow_period) {
            flow_accum = 0.0;  // Reset accumulator
            // Kalman and velocity logic here

            if (was_detected && hpecore::poseNonZero(detected_pose.pose)){
                // Update Kalman filter with detected pose (correction step)
                if (state.poseIsInitialised())
                    state.updateFromPosition(detected_pose.pose, detected_pose.timestamp);
                else
                    state.set(detected_pose.pose, tnow);
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

        // CSV logging aligned with moveEnet_flowKalman
        if (state.poseIsInitialised() && csv_file.is_open()) {
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
        if (is_visualize) {
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

            cv::imshow("moveEnetOFK_offline", canvas);
            char key_pressed = cv::waitKey(1);
            if (key_pressed == '\e' || key_pressed == 'q') {
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
    mn_handler.close();
    yarp::os::Network::disconnect("/movenet/sklt:o", movenet_in, "fast_tcp");
    yarp::os::Network::disconnect(eros_out, "/movenet/img:i", "fast_tcp");
    system("killall python3");  // Kill MoveNet process

    if (is_visualize) {
        cv::destroyAllWindows();
    }
    
    return 0;
}
/*
 * Author: Daria Berretta
 * 
 * This system processes event data from .log files offline to:
 * 1. Generate event representations (EROS, SAE, Binary)
 * 2. Detect human poses using MoveEnet
 * 3. Estimate joint velocities using Kalman filter fusion (like edpr-april.cpp)
 * 4. Save results to CSV and video
 *
 * Key differences from moveEnet_flow.cpp:
 * - Implements Kalman filter (multiJointLatComp state) for optical flow fusion
 * - Velocities are used for pose prediction and smoothing, not just logging
 * - More robust optical flow estimation through temporal filtering
 *
 * BEFORE TO OPEN THE DOCKER REMBER TO:
 * 1. xhost +local:docker
 * 2. docker compose up
 * New terminal:
 * 3. docker exec -it moveEnet_flow sh
 * 4. start yarp server: yarpserver &
 * Attach Vs to running container
 * 5. cd /home/moveEnetFlow/build
 * 6. cmake ..
 * 7. make
 * Run the application:
 * 8. ./moveEnet_flowKalman --f_det 10 --f_vis 100 --vis --use_lc --pu 0.2 --muD 0.3 --muV 0.0 --no_csv --no_video
 */

#include <yarp/cv/Cv.h>
#include <yarp/os/all.h>
#include <yarp/sig/Image.h>
#include <event-driven/core.h>
#include <event-driven/vis.h>
#include <event-driven/algs.h>
#include <hpe-core/utility.h>
#include <hpe-core/motion_estimation.h>
#include <hpe-core/motion.h>
#include <hpe-core/fusion.h>
#include <hpe-core/representations.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <unistd.h>

using namespace yarp::os;
using namespace yarp::sig;
using std::vector;


/**
 * @class externalDetector
 * @brief Manages asynchronous communication with external MoveNet pose detector via YARP ports.
 * 
 * This class handles:
 * - Rate-limited image transmission to MoveNet detector
 * - Non-blocking pose result reception
 * - Timing and latency tracking between request and response
 * 
 * Architecture:
 * - Uses buffered YARP ports for inter-process communication (IPC)
 * - Implements asynchronous request-response pattern with timeout protection
 * - Tracks detection latency and detection timing for downstream processing
 * 
 * Data flow:
 * Frame image → EROS surface → output_port → MoveNet (Python process)
 *                                                       ↓
 *                                            input_port ← Pose (skeleton + confidence)
 */
class externalDetector
{
private:
    double period{0.1};              ///< Minimum time interval between pose detection requests (1/rate)
    double tic{0.0};                 ///< Timestamp when last image was sent to MoveNet
    bool waiting{false};             ///< Flag: true if awaiting response from MoveNet

    BufferedPort<ImageOf<PixelMono>> output_port;  ///< YARP output port: sends EROS image to MoveNet
    BufferedPort<Bottle> input_port;               ///< YARP input port: receives pose skeleton from MoveNet

public:

    /**
     * @brief Initialize YARP ports and set detection rate.
     * 
     * @param output_name  YARP port name for sending images (e.g., "/module_name/eros:o")
     * @param input_name   YARP port name for receiving poses (e.g., "/module_name/movenet:i")
     * @param rate         Detection frequency in Hz (controls minimum time between requests)
     * 
     * @return true if both ports opened successfully, false otherwise
     * 
     * @note Creates ports but does NOT establish connections (done separately via YARP Network::connect)
     */
    bool init(std::string output_name, std::string input_name, double rate)
    {
        if (!output_port.open(output_name))
            return false;

        if (!input_port.open(input_name))
            return false;

        period = 1.0 / rate;
        return true;
    }

    /**
     * @brief Close YARP ports and cleanup resources.
     * 
     * Disconnects and closes both input and output ports. Should be called during
     * module shutdown or error handling.
     */
    void close()
    {
        output_port.close();
        input_port.close();
    }

    /**
     * @brief Send image to MoveNet and read pose response (non-blocking).
     * 
     * This method implements a rate-limited request-response pattern:
     * 
     * REQUEST PHASE:
     * - If detection period has elapsed since last request (tic), convert and blur image,
     *   then send via output_port
     * - Sets waiting=true to indicate MoveNet is processing
     * - Updates tic with current timestamp
     * 
     * RESPONSE PHASE:
     * - Attempts non-blocking read from input_port
     * - If MoveNet response available (Bottle received):
     *   * Extracts skeleton pose (13 joints × 2 coordinates)
     *   * Extracts confidence scores (13 values)
     *   * Records detection timestamp (tic) and latency (latest_ts - tic)
     *   * Sets waiting=false to allow next request
     * 
     * TIMEOUT PROTECTION:
     * - If latest_ts < tic (timeline reset), resets tic to protect against stale requests
     * - Forces new request if time gap exceeds 2.0 seconds
     * 
     * @param latest_image   Current event representation (EROS surface) as CV Mat
     * @param latest_ts      Current frame timestamp in seconds
     * @param previous_skeleton [OUT] Detected pose data (pose, confidence, timestamp, delay)
     * 
     * @return true if new pose was received, false if no pose available yet
     * 
     * @note Non-blocking: returns immediately without waiting for MoveNet
     * @note Pose only updated if complete response received; previous_skeleton unchanged otherwise
     */
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


/**
 * @class MOVEENET_FLOW_KALMAN
 * @brief Offline event-processing pipeline with MoveNet detection and Kalman fusion.
 *
 * Provides a complete loop to:
 * - Load events from a .log file and update EROS/SAE/Binary surfaces
 * - Send EROS frames to MoveNet asynchronously and receive 13-joint poses
 * - Fuse detections with joint-wise optical-flow velocities using a Kalman filter
 * - Output filtered poses (and optional velocities) to CSV and video, with live visualization
 *
 * Key properties:
 * - Non-blocking detection: the loop does not wait for MoveNet; KF maintains a continuous pose
 * - Latency-aware: stores detection time and delay for evaluation
 * - Configurable: frequencies, uncertainties, ROI, outputs controlled via CLI flags
 */
class MOVEENET_FLOW_KALMAN : public RFModule
{
    private:

        // Save output csv file and video
        std::ofstream csv_file;
        cv::VideoWriter video_writer;

        // Event loader from .log file
        ev::offlineLoader<ev::AE> eloader;

        //Surface handlers
        hpecore::EROS eros_handler;
        hpecore::SAE sae_handler;
        hpecore::BIN binary_handler;

        // Detection handler
        externalDetector mn_handler;

        // Detected pose
        hpecore::stampedPose detected_pose;

        // Velocity estimation
        hpecore::pwtripletvelocity velocity_estimator;

        // Kalman filter for pose state fusion
        hpecore::multiJointLatComp state;

        // standard parameters
        cv::Size image_size;
        int roiSize{20};
        int detF{10};
        double th_period{0.01}, thF{100.0};
        double c_thresh{0.4};
        bool is_visualize{false};
        
        // Kalman filter parameters
        double procU{1e-1};          // Process uncertainty (pu)
        double measUD{1e-4};         // Measurement uncertainty (position)
        double measUV{0.0};          // Measurement uncertainty (velocity)
        bool latency_compensation{false};   // If true, use latency compensation in Kalman filter
        
        // CSV output format flags
        bool eval_format{false};        // Output in evaluate_hpe.py compatible format
        bool include_velocities{false}; // Include velocities in eval_format
        bool no_csv{false};             // Skip CSV output
        bool no_video{false};           // Skip video output
        
        // cv::Scalar colors[13] = {{0, 0, 180}, {0, 180, 0}, {0, 0, 180},
        //                 {180, 180, 0}, {180, 0, 180}, {0, 180, 180},
        //                 {120, 0, 180}, {120, 180, 0}, {0, 120, 180},
        //                 {120, 120, 180}, {120, 180, 120}, {120, 120, 180}, {120, 120, 120}};

                    
    

    public:

        /**
         * @brief Configure module: load parameters, initialize handlers, setup YARP/files.
         * 
         * Execution flow:
         * 1. Parse command-line arguments and display help if requested
         * 2. Verify YARP network connectivity
         * 3. Read all parameters (paths, frequencies, Kalman filter settings, output options)
         * 4. Initialize event representation handlers (EROS, SAE, Binary)
         * 5. Initialize Kalman filter with process/measurement uncertainties
         * 6. Launch MoveNet detection process (Python subprocess)
         * 7. Initialize CSV output file with headers (if enabled)
         * 8. Initialize video writer for visualization output (if enabled)
         * 9. Open visualization window (if --vis flag set)
         * 10. Initialize YARP ports for MoveNet communication
         * 11. Connect MoveNet ports (/movenet/sklt:o ↔ /movenet:i)
         * 12. Load offline event log file (.log format)
         * 
         * @param rf ResourceFinder containing command-line arguments
         * @return true if all initialization successful, false on any error
         */
        bool configure(yarp::os::ResourceFinder &rf) override
        {
            // 1. If request print help for command line and exit
            if(rf.check("help")) 
            {
                yInfo() << "--help:";
                yInfo() << "--log_path <string>: path to input .log event data file";
                yInfo() << "--f_det <int>: detection frequency (default 10Hz)";
                yInfo() << "--f_vis <double>: visualization frequency (default 100Hz)";
                yInfo() << "--output_csv <string>: path to output csv file for joint positions and velocities";
                yInfo() << "--output_video <string>: path to output video file";
                yInfo() << "--vis: flag to enable visualization, if not set no display is shown";
                yInfo() << "--pu <float>: Kalman filter process uncertainty (default 1e-1)";
                yInfo() << "--muD <float>: Kalman filter measurement uncertainty position (default 1e-4)";
                yInfo() << "--muV <float>: Kalman filter measurement uncertainty velocity (default 0.0)";
                yInfo() << "--use_lc: enable latency compensation in Kalman filter (default false)";
                yInfo() << "--eval_format: output CSV in format compatible with evaluate_hpe.py (timestamp, latency, x0,y0,...,x12,y12)";
                yInfo() << "--include_velocities: when using --eval_format, also include velocities (vx0,vy0,...,vx12,vy12)";
                yInfo() << "--no_csv: skip CSV output";
                yInfo() << "--no_video: skip video output";
                return false;
            }

            // 2. Setup YARP connection
            if (!yarp::os::Network::checkNetwork(2.0))
            {
                std::cout << "Could not connect to YARP" << std::endl;
                return false;
            }

            // 3. Set up module name
            setName((rf.check("name", Value("/moveenet_flow_kalman")).asString()).c_str());

            // 4. Read parameters from command line
            detF = rf.check("f_det", Value(10)).asInt32();
            double visF = rf.check("f_vis", Value(100.0)).asFloat64();
            std::string log_path = rf.check("log_path", Value("/data/new_scarfGNN_full/raw/cam2_S8_Discussion/ch0dvs/data.log")).asString();
            std::string output_csv = rf.check("output_csv", Value("/home/moveEnetFlow/csv_file/260113_test_kalman5.csv")).asString();
            std::string output_video = rf.check("output_video", Value("/home/moveEnetFlow/avi_file/260113_test_kalman5.avi")).asString();
            is_visualize = rf.check("vis");
            
            // Read Kalman filter parameters
            procU = rf.check("pu", Value(0.0)).asFloat64();
            measUD = rf.check("muD", Value(0.0)).asFloat64();
            measUV = rf.check("muV", Value(0.0)).asFloat64();
            latency_compensation = rf.check("use_lc", Value(false)).asBool();
            
            // Read CSV format flags
            eval_format = rf.check("eval_format");
            include_velocities = rf.check("include_velocities");
            no_csv = rf.check("no_csv");
            no_video = rf.check("no_video");
            
            yInfo() << "Configuration:";
            yInfo() << "  - Log path: " << log_path;
            yInfo() << "  - Output CSV: " << output_csv;
            yInfo() << "  - Output video: " << output_video;
            yInfo() << "  - Visualization: " << (is_visualize ? "ENABLED" : "DISABLED");
            yInfo() << "  - Detection freq: " << detF << " Hz";
            yInfo() << "  - Visualization freq: " << visF << " Hz";
            yInfo() << "  - Kalman process uncertainty: " << procU;
            yInfo() << "  - Kalman measurement uncertainty (pos): " << measUD;
            yInfo() << "  - Kalman measurement uncertainty (vel): " << measUV;
            yInfo() << "  - Latency compensation: " << (latency_compensation ? "ENABLED" : "DISABLED");
            yInfo() << "  - Eval format: " << (eval_format ? "ENABLED" : "DISABLED");
            yInfo() << "  - Include velocities: " << (include_velocities ? "ENABLED" : "DISABLED");
            yInfo() << "  - Skip CSV: " << (no_csv ? "ENABLED" : "DISABLED");
            yInfo() << "  - Skip video: " << (no_video ? "ENABLED" : "DISABLED");

            // 5. Initialize internal parameters
            
            image_size = cv::Size(rf.check("w", Value(640)).asInt32(),
                                  rf.check("h", Value(480)).asInt32());
            roiSize = rf.check("roi", Value(20)).asInt32();

            std::string checkpoint_path = rf.check("checkpoint_path", Value("/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth")).asString();
            thF = visF;
            th_period = 1/thF;
            c_thresh = rf.check("confidence", Value(0.4)).asFloat64();


            // Initialize EROS, SAE, Binary handlers
            eros_handler.init(image_size.width, image_size.height, 7, 0.3);
            binary_handler.init(image_size.width, image_size.height);
            sae_handler.init(image_size.width, image_size.height);

            // Initialize Kalman filter state
            double lc = latency_compensation ? 1.0 : 0.0;
            if (!state.initialise({procU, measUD, measUV, lc}))
            {
                yError() << "Kalman filter (multiJointLatComp) initialization failed";
                return false;
            }

            // 6. Start of moveEnet flow process
            yInfo() << "Starting MoveEnet flow process with Kalman filter...";
            std::string command = "python3 /usr/local/src/hpe-core/example/movenet/movenet_online.py --checkpoint_path " + checkpoint_path + " &";
            system(command.c_str());

            // check if moveEnet process started
            while (!yarp::os::NetworkBase::exists("/movenet/sklt:o"))
                sleep(1);
            yInfo() << "MoveEnet started correctly";

            // 7. Initialise .csv output file if not skipped
            if (!no_csv) {
                csv_file.open(output_csv);
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
            }

            // 8. Initialize video writer if not skipped
            if (!no_video) {
                video_writer.open(output_video, cv::VideoWriter::fourcc('M','J','P','G'), thF, image_size);
                if (!video_writer.isOpened()) {
                    yError() << "Could not open the output video for write: " << output_video;
                    return false;
                }
                yInfo() << "Video writer initialized successfully: " << output_video;
            }

            // 9. Initialize visualization window if requested
            if (is_visualize) {
                cv::namedWindow("moveenet-flow-kalman", cv::WINDOW_NORMAL);
                cv::resizeWindow("moveenet-flow-kalman", image_size);
                yInfo() << "Visualization window created";
            }

            // 10. Initialize MoveEnet detection handler
            if (!mn_handler.init(getName("/eros:o"), getName("/movenet:i"), detF))
            {
                yError() << "Could not open movenet ports";
                return false;
            }

            // 11. Connect MoveEnet ports
            Network::connect("/movenet/sklt:o", getName("/movenet:i"), "fast_tcp");
            Network::connect(getName("/eros:o"), "/movenet/img:i", "fast_tcp");

            // 12. Initialize offline event loader from .log file
            if (!eloader.load(log_path)) {
                yError() << "Could not open event log file: " << log_path;
                return false;
            }
            yInfo() << "Successfully opened event log file: " << log_path;
            yInfo() << eloader.getinfo();


            return true;
        }

    double getPeriod() override
        {
            // run the processing loop at the specified frame rate
            return th_period;
        }

    /**
     * @brief Clean shutdown: close ports, files, processes, and windows.
     * 
     * Called when module receives interrupt signal (Ctrl+C).
     * Ensures all resources are properly released before termination.
     */
    bool interruptModule() override
        {
            // if the module is asked to stop, close ports and do other clean up
            yInfo() << "Interrupting module and closing resources...";
            
            mn_handler.close();
            
            // close files
            if (csv_file.is_open()) {
                csv_file.close();
                yInfo() << "CSV file closed";
            }
            
            if (video_writer.isOpened()) {
                video_writer.release();
                yInfo() << "Video file closed and finalized";
            }
            
            // Close visualization window if it exists
            if (is_visualize) {
                cv::destroyAllWindows();
            }
            
            // kill moveEnet process and clear resources
            yInfo() << "Stopping MoveEnet process...";
            system("killall python3");

            return true;
        }

    bool close() override
        {   
            //close python process
            system("killall python3");
            return true;
        }

    void drawEROS(cv::Mat &img)
    {
        cv::Mat eros8;
        eros_handler.getSurface().convertTo(eros8, CV_8U);
        cv::GaussianBlur(eros8, eros8, {9, 9}, 0);
        cv::normalize(eros8, eros8, 0, 255, cv::NORM_MINMAX);
        cv::cvtColor(eros8, img, cv::COLOR_GRAY2BGR);
    }

    /**
     * @brief Helper: Convert binary event surface to colored image.
     */
    void drawEVENTS(cv::Mat &img)
    {
        cv::Mat eventsmono;
        binary_handler.getSurface().convertTo(eventsmono, CV_8U);
        cv::cvtColor(eventsmono, img, CV_GRAY2BGR);
    }

    /**
     * @brief Helper: Convert SAE surface to colored image with temporal decay visualization.
     */
    void drawSAE(cv::Mat &img)
    {
        cv::Mat sae64, saemono;
        
        sae_handler.getSurface().copyTo(sae64);
        double maxval;
        cv::minMaxLoc(sae64, nullptr, &maxval);
        sae64 -= (maxval - 1.0);  //show 2 seconds of surface
        sae64.convertTo(saemono, CV_8U, 255.0);
        cv::cvtColor(saemono, img, CV_GRAY2BGR);
    }

    /**
     * @brief Main processing loop: Event→Graph→Pose→Kalman→Output pipeline.
     * 
     * **STEP 1: LOAD AND PROCESS EVENTS**
     * - Increment frame time by period (simulates fixed framerate)
     * - Load all events from log file up to current frame time
     * - Update EROS, SAE, and Binary surfaces with event data
     * - Detect stream resets and reset Kalman filter if detected
     * 
     * **STEP 2: POSE DETECTION WITH MOVENET**
     * - Send current EROS surface to MoveNet detector at specified frequency
     * - Receive detected skeleton (13 joints) and confidence scores
     * 
     * **STEP 3: KALMAN FILTER FUSION** (Correction step)
     * - If pose detected with non-zero confidence:
     *   * On first detection: Initialize Kalman filter with detected pose
     *   * On subsequent detections: Update filter with measurement (detected pose)
     * 
     * **STEP 4: VELOCITY ESTIMATION WITH OPTICAL FLOW** (Prediction step)
     * - Estimate joint velocities from SAE surface using pwtripletvelocity
     * - Update Kalman filter with velocity estimate for temporal smoothing
     * - Query filtered pose from Kalman state
     * - Compute and log max difference between raw and filtered poses
     * 
     * **STEP 5: SAVE RESULTS TO CSV**
     * - If Kalman filter initialized and CSV enabled:
     *   * Standard format: timestamp, joint_pos_x/y, joint_vel_x/y, confidence (×13 joints)
     *   * Eval format: timestamp, latency, joint_pos_x/y (×13 joints), [optional: velocities]
     * 
     * **STEP 6: VISUALIZATION AND VIDEO WRITING**
     * - Draw EROS surface as background (grayscale event density)
     * - Draw filtered skeleton (red) with velocity vectors (green)
     * - Optionally draw raw MoveNet skeleton (blue)
     * - Write frame to video file
     * - Display on screen if visualization enabled
     * - Handle user input (ESC/Q to stop)
     * 
     * **STEP 7: CLEANUP FOR NEXT FRAME**
     * - Clear binary surface for next frame (EROS and SAE are persistent)
     * 
     * @return false when event stream exhausted (no more events), true to continue
     */
    bool updateModule() override
    {
        // ===== STEP 1: LOAD AND PROCESS EVENTS =====
        static double tnow = 0.0;
        static double pts = 0.0;    // previous timestamp
        static int batch_count = 0;
        double period = getPeriod();
        
        // Increment event loader to read events up to the next time period
        tnow += period;
        eloader.incrementReadTill(tnow);
        
        // Check for reset (if timestamps go backwards)
        if(tnow < pts) {
            sae_handler.getSurface().setTo(0.0);
            binary_handler.getSurface().setTo(0.0);
            eros_handler.getSurface().setTo(0.0);
            state.reset();  // Reset Kalman filter state on timeline reset
            yInfo() << "Event stream reset detected - Kalman filter state reset";
        }
        
        // Process all events in this time window
        int event_count = 0;
        for (ev::offlineLoader<ev::AE>::iterator v = eloader.begin(); v != eloader.end(); v++) {
            eros_handler.update(v->x, v->y);
            binary_handler.update(v->x, v->y);
            sae_handler.update(v->x, v->y, tnow);
            event_count++;
        }
        
        pts = tnow;
        batch_count++;
            
        // Check if we've reached the end of the file (no more events)
        if (event_count == 0 && tnow > 0) {
            yInfo() << "Finished processing event file (no more events). Total frames: " << batch_count;
            return false; // Stop the module
        }
        
        // ===== STEP 2: POSE DETECTION WITH MOVENET =====
        bool was_detected = mn_handler.update(eros_handler.getSurface(), tnow, detected_pose);
        
        // ===== STEP 3: KALMAN FILTER FUSION =====
        if (was_detected && hpecore::poseNonZero(detected_pose.pose))
        {
            // Update Kalman filter with detected pose (correction step)
            if (state.poseIsInitialised())
                state.updateFromPosition(detected_pose.pose, detected_pose.timestamp);
                //state.updateFromPosition(detected_pose.pose, tnow);
            else
                state.set(detected_pose.pose, tnow);
        }
        
        // ===== STEP 4: VELOCITY ESTIMATION WITH OPTICAL FLOW =====
        hpecore::skeleton13 jvs;  // Joint velocities
        hpecore::skeleton13 filtered_pose;  // Pose from Kalman filter
        
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
        
        
        // // Debug: Compute maximum difference between detected and filtered pose
        // if (state.poseIsInitialised() && hpecore::poseNonZero(detected_pose.pose)) {
        //     double max_diff = 0.0;
        //     for (int j = 0; j < 13; j++) {
        //         double dx = detected_pose.pose[j].u - filtered_pose[j].u;
        //         double dy = detected_pose.pose[j].v - filtered_pose[j].v;
        //         double diff = sqrt(dx*dx + dy*dy);
        //         if (diff > max_diff) max_diff = diff;
        //     }
        //     yInfo() << "Max pose difference (detected vs filtered): " << max_diff << " pixels";
        //     usleep(10000);  // Small delay for readability
        // }
        
        // ===== STEP 5: SAVE RESULTS TO CSV =====
        if (state.poseIsInitialised() && csv_file.is_open())
        {
            csv_file << std::fixed << std::setprecision(6) << tnow;
            if (eval_format) {
                // latency: use detected_pose.delay if available, else 0.0
                double lat = (detected_pose.timestamp > 0) ? detected_pose.delay : 0.0;
                csv_file << "," << lat;
                for (int j = 0; j < 13; j++) {
                    csv_file << "," << filtered_pose[j].u << "," << filtered_pose[j].v;
                }
                if (include_velocities) {
                    for (int j = 0; j < 13; j++) {
                        csv_file << "," << jvs[j].u << "," << jvs[j].v;
                    }
                }
            } else {
                for (int j = 0; j < 13; j++) {
                    csv_file << "," << filtered_pose[j].u 
                            << "," << filtered_pose[j].v
                            << "," << jvs[j].u
                            << "," << jvs[j].v
                            << "," << detected_pose.conf[j];
                }
            }
            csv_file << "\n";
            csv_file.flush();  // Ensure data is written to disk
        }
        
        // ===== STEP 6: VISUALIZATION AND VIDEO WRITING =====
        static cv::Mat canvas = cv::Mat(image_size, CV_8UC3);
        canvas.setTo(cv::Vec3b(0, 0, 0));
        
        // Draw EROS background
        drawEROS(canvas);

        // Visualization thresholds (force draw everything) and a small offset to separate raw vs filtered visually
        const double det_draw_thresh = 0.0;
        const double filt_draw_thresh = 0.0;
        const double det_offset_px = 0.00;  // visual-only offset to avoid perfect overlap

        // Draw filtered skeleton from Kalman filter (red) on top
        if (state.poseIsInitialised()) {
            try {
                hpecore::stampedPose pose_filtered;
                pose_filtered.pose = state.query();
                pose_filtered.timestamp = tnow;
                hpecore::drawSkeleton(canvas, pose_filtered, {255, 0, 0}, 4, filt_draw_thresh);  // Red in display 
                // Optionally draw velocity vectors (green)
                hpecore::skeleton13 vel = state.queryVelocity();
                hpecore::stampedPose pose_with_vel;
                pose_with_vel.pose = state.query();
                pose_with_vel.timestamp = tnow;
                hpecore::drawVel(canvas, pose_with_vel, vel, {0, 255, 0}, 2, filt_draw_thresh);  // Green = velocity
            } catch (const cv::Exception& e) {
                // Silently skip drawing if OpenCV throws an error
            }
        }

        // Draw detected skeleton (MoveNet) last, thicker, to ensure visibility over red
        if (hpecore::poseNonZero(detected_pose.pose)) {
            try {
                hpecore::stampedPose pose_raw = detected_pose;
                for (int j = 0; j < 13; j++) {
                    pose_raw.pose[j].u += det_offset_px;
                    pose_raw.pose[j].v -= det_offset_px;
                }
                hpecore::drawSkeleton(canvas, pose_raw, {0, 0, 255}, 4, det_draw_thresh);  // Blue in display
            } catch (const cv::Exception& e) {
                // Silently skip drawing if OpenCV throws an error
            }
        }  
        
        // Always write frame to video file
        if (video_writer.isOpened()) {
            video_writer.write(canvas);
        }
        
        // Display frame only if visualization is enabled
        if (is_visualize) {
            cv::imshow("moveenet-flow-kalman", canvas);
            char key_pressed = cv::waitKey(1);
            if (key_pressed == '\e' || key_pressed == 'q') {
                yInfo() << "User requested stop";
                return false;
            }
        }
        
        // Clear binary events surface for next frame
        binary_handler.getSurface().setTo(0.0);
        
        return true;
    }

};


int main(int argc, char *argv[]) {
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    /* create the module */
    MOVEENET_FLOW_KALMAN instance;
    return instance.runModule(rf);          // This calls: updateModule() loop -> close()
}

/*
 * Offline Human Pose Estimation using Event-based Data
 * Author: Daria Berretta
 * 
 * This system processes event data from .log files offline to:
 * 1. Generate event representations (EROS, SAE, Binary)
 * 2. Detect human poses using MoveEnet
 * 3. Estimate joint velocities
 * 4. Save results to CSV and video
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

using namespace yarp::os;
using namespace yarp::sig;
using std::vector;


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




class MOVENET_FLOW : public RFModule
{
private:
    // Event data loader
    ev::offlineLoader<ev::AE> event_loader;
    ev::window<ev::AE> event_window;
    
    // Event representations handlers
    externalDetector mn_handler;
    // delayedGT gt_handler;
    hpecore::EROS eros_handler;
    hpecore::SAE sae_handler;
    hpecore::BIN binary_handler;
    
    // MoveEnet communication
    BufferedPort<ImageOf<PixelMono>> movenet_output_port;
    BufferedPort<Bottle> movenet_input_port;
    
    // Velocity estimation
    hpecore::pwtripletvelocity velocity_estimator;
    
    // Data structures
    hpecore::stampedPose current_pose;
    hpecore::skeleton13 current_velocity;
    
    // Parameters
    cv::Size image_size;
    int roiSize{20};
    int detF{10};  // Detection frequency in Hz
    double batch_frequency{100.0};  // Hz
    double batch_period;  // seconds
    std::string log_path;
    std::string output_csv_path;
    std::string output_video_path;
    bool save_csv{false};
    bool save_video{false};
    bool visualize{true};
    double c_thresh{0.4};  // Confidence threshold
    
    // Processing state
    double current_time{0.0};
    double end_time{0.0};
    bool movenet_waiting{false};
    double movenet_request_time{0.0};
    
    // Output file handles
    std::ofstream csv_file;
    cv::VideoWriter video_writer;
    
    // Visualization
    cv::Scalar colors[13] = {
        {0, 0, 180}, {0, 180, 0}, {0, 0, 180},
        {180, 180, 0}, {180, 0, 180}, {0, 180, 180},
        {120, 0, 180}, {120, 180, 0}, {0, 120, 180},
        {120, 120, 180}, {120, 180, 120}, {120, 120, 180}, 
        {120, 120, 120}
    };

public:
    bool configure(yarp::os::ResourceFinder &rf) override
    {
        if (rf.check("help")) {
            yInfo() << "MoveEnet_Flow System";
            yInfo() << "";
            yInfo() << "Options:";
            yInfo() << "--log_path <string>        : Path to .log event file (required)";
            yInfo() << "--f_det <double>           : detection rate in Hz [30]";
            yInfo() << "--confidence <double>      : Confidence threshold for visualization [0.4]";
            yInfo() << "--checkpoint_path <string> : Path to MoveEnet checkpoint";
            yInfo() << "--output_csv <string>      : Path to save CSV output";
            yInfo() << "--output_video <string>    : Path to save video output";
            yInfo() << "--no_viz                   : Disable real-time visualization";
            return false;
        }

        // =====SET UP YARP=====
        if (!yarp::os::Network::checkNetwork(2.0))
        {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }

        // Set module name FIRST (required before opening ports). The module name is used to name ports
        std::string module_name = rf.check("name", Value("/moveEnet_flow")).asString();
        setName(module_name.c_str());
        yInfo() << "Module name set to:" << module_name;
        

        // =====READ PARAMETERS=====
        // Path to event log file --> input event data
        log_path = rf.check("log_path", Value("")).asString();
        if (log_path.empty()) {
            yError() << "Please provide --log_path";
            return false;
        }
        // Image size for representations, visualization, and velocity estimation
        image_size = cv::Size(
            rf.check("w", Value(640)).asInt32(),
            rf.check("h", Value(480)).asInt32()
        );

        batch_frequency = rf.check("f_det", Value(30.0)).asFloat64();   //[Hz]
        batch_period = 1.0 / batch_frequency;                           //[s] --> periodo
        
        roiSize = rf.check("roi", Value(20)).asInt32();
        c_thresh = rf.check("confidence", Value(0.4)).asFloat64();
        
        // MoveEnet checkpoint path
        std::string checkpoint_path = rf.check("checkpoint_path", 
            Value("/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth")).asString();
        
        // Output options
        if (rf.check("output_csv")) {
            output_csv_path = rf.check("output_csv", Value("output.csv")).asString();
            save_csv = true;
        }
        
        if (rf.check("output_video")) {
            output_video_path = rf.check("output_video", Value("output.avi")).asString();
            save_video = true;
        }
        
        visualize = !(rf.check("no_viz") && rf.check("no_viz", Value(true)).asBool());

        // ==== START MOVENET NETWORK =====
        std::string command = "python3 /usr/local/src/hpe-core/example/movenet/movenet_online.py --checkpoint_path " + checkpoint_path + " &";
        int r = system(command.c_str());

        // Wait for MoveEnet to start and open its output port
        while (!yarp::os::NetworkBase::exists("/movenet/sklt:o"))
            sleep(1);
        yInfo() << "MoveEnet started correctly";

        // initialize MoveEnet handler
        if (!mn_handler.init(getName("/eros:o"), getName("/movenet:i"), batch_frequency))
        {
            yError() << "Could not open movenet ports";
            return false;
        }

        // ===== SET UP INTERNAL VARIABLE/DATA STRUCTURES =====

        // Initialize event representations
        eros_handler.init(image_size.width, image_size.height, 7, 0.3);
        binary_handler.init(image_size.width, image_size.height);
        sae_handler.init(image_size.width, image_size.height);

        // ==== TRY DEFAULT CONNECTIONS =====
        Network::connect("/file/ch0dvs:o", getName("/AE:i"), "fast_tcp");
        Network::connect("/file/ch2GT50Hzskeleton:o", getName("/gt:i"), "fast_tcp");
        Network::connect("/movenet/sklt:o", getName("/movenet:i"), "fast_tcp");
        Network::connect(getName("/eros:o"), "/movenet/img:i", "fast_tcp");

        // ==== LOAD EVENT DATA FROM LOG FILE =====
        yInfo() << "Loading data ... ";
        if (!event_loader.load(log_path, 60)) {
            yError() << "Could not open data file" << log_path;
            return false;
        }
        else {
            yInfo() << event_loader.getinfo();
        //    yInfo() << "Data time length [s]: " << event_loader.getLength();
        }  

        // ==== SEND EVENT DATA TO MOVENET =====
        // Open MoveEnet ports
        movenet_output_port.open(getName("/movenet/img:i").c_str());
        movenet_input_port.open(getName("/movenet/sklt:o").c_str());    
        yInfo() << "MoveEnet ports opened";

        // Start of the cicle where events are sent to MoveEnet,
        // detections are received, velocities computed, 
        // results saved and visualized
        // The loop runs until the end of the dataset is reached
        while (true) {
            if (!updateModule())
                break;
        }

        return true;
    }

    double getPeriod() override
    {
        return batch_period;
    }

    bool interruptModule() override
    {
        yInfo() << "Stopping module...";
        return true;
    }

    bool close() override
    {
        // Close ports
        movenet_output_port.close();
        movenet_input_port.close();
        
        // Close output files
        if (csv_file.is_open()) {
            csv_file.close();
            yInfo() << "CSV file closed";
        }
        
        if (video_writer.isOpened()) {
            video_writer.release();
            yInfo() << "Video file closed";
        }
        
        // Close visualization
        if (visualize) {
            cv::destroyAllWindows();
        }
        
        // Kill MoveEnet process
        yInfo() << "Stopping MoveEnet...";
        int r = system("killall python3");
        
        yInfo() << "Module closed successfully";
        return true;
    }

    bool updateModule() override
    {
        // Check if we've reached the end of the dataset
        if (current_time >= end_time) {
            yInfo() << "Processing complete!";
            return false;
        }

        // Update time
        current_time += batch_period;
        
        // Load events up to current time
        event_loader.incrementReadTill(current_time);
        
        // Update all representations with new events
        for (auto v = event_loader.begin(); v != event_loader.end(); v++) {
            eros_handler.update(v->x, v->y);
            sae_handler.update(v->x, v->y, current_time);
            binary_handler.update(v->x, v->y, v->p);
        }

        // Send EROS to MoveEnet at the detection rate (not every update)
        static double last_send_time = 0.0;
        double send_period = 1.0 / detF;  // detF is detection frequency (Hz)
        
        if (current_time - last_send_time >= send_period) {
            sendEROSToMoveEnet();
            last_send_time = current_time;
            movenet_request_time = current_time;
        }

        // Always try to read detection from MoveEnet (non-blocking)
        Bottle *detection = movenet_input_port.read(false);
        if (detection && detection->size() > 0) {
            try {
                current_pose.pose = hpecore::extractSkeletonFromYARP<Bottle>(*detection);
                current_pose.conf = hpecore::extractConfidenceFromYARP<Bottle>(*detection);
                current_pose.timestamp = movenet_request_time;
                current_pose.delay = current_time - movenet_request_time;
            } catch (const std::exception& e) {
                yWarning() << "Failed to parse detection:" << e.what();
            }
        }

        // Estimate velocities if we have a valid pose
        if (hpecore::poseNonZero(current_pose.pose)) {
            // The multi_area_velocity returns skeleton13 directly
            current_velocity = velocity_estimator.multi_area_velocity(
                sae_handler.getSurface(),
                current_time,
                current_pose.pose,
                roiSize
            );
        }

        // Save results
        saveResults();

        // Visualize
        if (visualize || save_video) {
            visualizeResults();
        }

        // Progress indicator
        static int last_progress = -1;
        int progress = (int)(100.0 * current_time / end_time);
        if (progress != last_progress && progress % 5 == 0) {
            yInfo() << "Progress:" << progress << "%";
            last_progress = progress;
        }

        return true;
    }

private:
    void sendEROSToMoveEnet()
    {
        cv::Mat eros_surface = eros_handler.getSurface();
        cv::Mat eros8;
        eros_surface.convertTo(eros8, CV_8U);
        cv::GaussianBlur(eros8, eros8, cv::Size(5, 5), 0, 0);
        
        ImageOf<PixelMono> &img = movenet_output_port.prepare();
        img.copy(yarp::cv::fromCvMat<PixelMono>(eros8));
        movenet_output_port.write();  // Use regular write() like edpr-april.cpp
    }

    void saveResults()
    {
        if (!save_csv || !csv_file.is_open())
            return;

        csv_file << std::fixed << std::setprecision(6) << current_time;
        csv_file << "," << std::scientific << current_pose.delay << std::fixed;
        
        for (int i = 0; i < 13; i++) {
            csv_file << "," << std::setprecision(2) 
                     << current_pose.pose[i].u << "," << current_pose.pose[i].v
                     << "," << std::setprecision(4) << current_pose.conf[i]
                     << "," << std::setprecision(2)
                     << current_velocity[i].u << "," << current_velocity[i].v;
        }
        csv_file << "\n";
    }

    void visualizeResults()
    {
        cv::Mat canvas = cv::Mat(image_size, CV_8UC3);
        canvas.setTo(cv::Vec3b(0, 0, 0));

        // Draw EROS background
        cv::Mat eros8;
        eros_handler.getSurface().convertTo(eros8, CV_8U);
        cv::GaussianBlur(eros8, eros8, {9, 9}, 0);
        cv::cvtColor(eros8, eros8, cv::COLOR_GRAY2BGR);
        cv::addWeighted(canvas, 0.5, eros8, 0.5, 0, canvas);

        // Draw skeleton if valid
        if (hpecore::poseNonZero(current_pose.pose)) {
            hpecore::drawSkeleton(canvas, current_pose, {0, 0, 255}, 3, c_thresh);
            
            // Draw velocity vectors
            hpecore::drawVel(canvas, current_pose, current_velocity, {0, 255, 255}, 2, c_thresh);
        }

        // Draw progress bar
        hpecore::drawProgressBar(canvas, current_time / end_time);

        // Add timestamp text
        std::string time_text = "Time: " + std::to_string(current_time) + "s";
        cv::putText(canvas, time_text, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        // Show or save
        if (visualize) {
            cv::imshow("Offline HPE", canvas);
            cv::waitKey(1);
        }

        if (save_video) {
            video_writer.write(canvas);
        }
    }
};

int main(int argc, char *argv[]) {
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    /* create the module */
    MOVENET_FLOW instance;
    return instance.runModule(rf);
}

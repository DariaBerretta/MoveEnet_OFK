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
#include <hpe-core/utility.h>
#include <hpe-core/motion_estimation.h>
#include <hpe-core/motion.h>
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

class OfflineHPE : public RFModule
{
private:
    // Event data loader
    ev::offlineLoader<ev::AE> event_loader;
    ev::window<ev::AE> event_window;
    
    // Event representations
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
            yInfo() << "Offline Human Pose Estimation System";
            yInfo() << "";
            yInfo() << "Options:";
            yInfo() << "--log_path <string>        : Path to .log event file (required)";
            yInfo() << "--w <int>                  : Image width [640]";
            yInfo() << "--h <int>                  : Image height [480]";
            yInfo() << "--frequency <double>       : Batch processing frequency in Hz [100.0]";
            yInfo() << "--roi <int>                : ROI size for velocity estimation [20]";
            yInfo() << "--confidence <double>      : Confidence threshold for visualization [0.4]";
            yInfo() << "--checkpoint_path <string> : Path to MoveEnet checkpoint";
            yInfo() << "--output_csv <string>      : Path to save CSV output";
            yInfo() << "--output_video <string>    : Path to save video output";
            yInfo() << "--no_viz                   : Disable real-time visualization";
            return false;
        }

        // =====SET UP YARP=====
        if (!yarp::os::Network::checkNetwork(2.0)) {
            yError() << "YARP network not available";
            return false;
        }

        // Set module name
        setName((rf.check("name", Value("/offline_hpe")).asString()).c_str());

        // =====READ PARAMETERS=====
        log_path = rf.check("log_path", Value("")).asString();
        if (log_path.empty()) {
            yError() << "Please provide --log_path";
            return false;
        }

        image_size = cv::Size(
            rf.check("w", Value(640)).asInt32(),
            rf.check("h", Value(480)).asInt32()
        );
        
        batch_frequency = rf.check("frequency", Value(100.0)).asFloat64();
        batch_period = 1.0 / batch_frequency;
        
        roiSize = rf.check("roi", Value(20)).asInt32();
        c_thresh = rf.check("confidence", Value(0.4)).asFloat64();
        
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

        // =====LOAD EVENT DATA=====
        yInfo() << "Loading event data from:" << log_path;
        if (!event_loader.load(log_path)) {
            yError() << "Could not load event data from" << log_path;
            return false;
        }
        
        yInfo() << event_loader.getinfo();
        end_time = event_loader.getLength();
        yInfo() << "Dataset duration:" << end_time << "seconds";

        // =====INITIALIZE REPRESENTATIONS=====
        eros_handler.init(image_size.width, image_size.height, 7, 0.3);
        binary_handler.init(image_size.width, image_size.height);
        sae_handler.init(image_size.width, image_size.height);

        // =====INITIALIZE VELOCITY ESTIMATOR=====
        velocity_estimator.setParameters(roiSize, 1, image_size);

        // Initialize pose structures
        current_pose.pose.fill({0.0, 0.0});
        current_pose.conf.fill(0.0);
        current_pose.timestamp = 0.0;
        current_pose.delay = 0.0;
        current_velocity.fill({0.0, 0.0});

        // =====START MOVENET=====
        std::string command = "python3 /usr/local/src/hpe-core/example/movenet/movenet_online.py --gpu --checkpoint_path " + checkpoint_path + " &";
        yInfo() << "Starting MoveEnet with command:" << command;
        int r = system(command.c_str());
        
        // Wait for MoveEnet to be ready
        yInfo() << "Waiting for MoveEnet to start...";
        int wait_count = 0;
        while (!yarp::os::NetworkBase::exists("/movenet/sklt:o")) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            wait_count++;
            if (wait_count > 20) {
                yError() << "MoveEnet failed to start";
                return false;
            }
        }
        yInfo() << "MoveEnet started successfully";

        // =====OPEN MOVENET PORTS=====
        if (!movenet_output_port.open(getName("/eros:o"))) {
            yError() << "Could not open output port";
            return false;
        }
        
        if (!movenet_input_port.open(getName("/movenet:i"))) {
            yError() << "Could not open input port";
            return false;
        }

        // Connect ports
        Network::connect(getName("/eros:o"), "/movenet/img:i", "fast_tcp");
        Network::connect("/movenet/sklt:o", getName("/movenet:i"), "fast_tcp");

        // =====SETUP OUTPUT FILES=====
        if (save_csv) {
            csv_file.open(output_csv_path);
            if (!csv_file.is_open()) {
                yError() << "Could not open CSV file:" << output_csv_path;
                return false;
            }
            // Write CSV header
            csv_file << "timestamp,delay";
            for (int i = 0; i < 13; i++) {
                csv_file << ",j" << i << "_u,j" << i << "_v,j" << i << "_conf";
                csv_file << ",j" << i << "_vel_u,j" << i << "_vel_v";
            }
            csv_file << "\n";
            yInfo() << "CSV output:" << output_csv_path;
        }

        if (save_video) {
            int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
            video_writer.open(output_video_path, fourcc, batch_frequency, image_size);
            if (!video_writer.isOpened()) {
                yError() << "Could not open video file:" << output_video_path;
                return false;
            }
            yInfo() << "Video output:" << output_video_path;
        }

        // =====SETUP VISUALIZATION=====
        if (visualize) {
            cv::namedWindow("Offline HPE", cv::WINDOW_NORMAL);
            cv::resizeWindow("Offline HPE", image_size);
        }

        yInfo() << "Configuration complete. Starting processing...";
        yInfo() << "Batch frequency:" << batch_frequency << "Hz (" << batch_period << "s period)";
        
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

        // Update SAE for velocity estimator
        velocity_estimator.updateSAE(event_loader.begin(), event_loader.end(), current_time);

        // Send EROS to MoveEnet if not waiting for previous result
        if (!movenet_waiting) {
            sendEROSToMoveEnet();
            movenet_waiting = true;
            movenet_request_time = current_time;
        }

        // Try to read detection from MoveEnet
        Bottle *detection = movenet_input_port.read(false);
        if (detection && detection->size() > 0) {
            try {
                current_pose.pose = hpecore::extractSkeletonFromYARP<Bottle>(*detection);
                current_pose.conf = hpecore::extractConfidenceFromYARP<Bottle>(*detection);
                current_pose.timestamp = movenet_request_time;
                current_pose.delay = current_time - movenet_request_time;
                movenet_waiting = false;
            } catch (const std::exception& e) {
                yWarning() << "Failed to parse detection:" << e.what();
            }
        }

        // Estimate velocities if we have a valid pose
        if (hpecore::poseNonZero(current_pose.pose)) {
            // The multi_area_velocity returns skeleton13 directly
            current_velocity = velocity_estimator.multi_area_velocity(
                velocity_estimator.querySAEP(),
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
        movenet_output_port.write();
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

int main(int argc, char *argv[])
{
    /* prepare and configure the resource finder */
    yarp::os::ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    /* create the module */
    OfflineHPE instance;
    return instance.runModule(rf);
}

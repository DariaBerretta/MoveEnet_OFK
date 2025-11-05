/*
 * Author: Daria Berretta
 * 
 * This system processes event data from .log files offline to:
 * 1. Generate event representations (EROS, SAE, Binary)
 * 2. Detect human poses using MoveEnet
 * 3. Estimate joint velocities
 * 4. Save results to CSV and video



    BEFORE TO OPEN THE DOCKER REMBER TO:
    1. xhost +local:docker
    2. start yarp server: yarpserver &
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

class MOVEENET_FLOW : public RFModule
{
    private:

        // Save output csv file and video
        std::ofstream csv_file;
        cv::VideoWriter video_writer;

        // Event loader from .log file
        ev::offlineLoader<ev::AE> eloader;
        double data_timelength{0.0};

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

        // standard parameters
        cv::Size image_size;
        int roiSize{20};
        int detF{10};
        double th_period{0.01}, thF{100.0};
        double c_thresh{0.4};
        bool is_visualize{false};
        cv::Scalar colors[13] = {{0, 0, 180}, {0, 180, 0}, {0, 0, 180},
                        {180, 180, 0}, {180, 0, 180}, {0, 180, 180},
                        {120, 0, 180}, {120, 180, 0}, {0, 120, 180},
                        {120, 120, 180}, {120, 180, 120}, {120, 120, 180}, {120, 120, 120}};

                    
    

    public:

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
                return false;
            }

            // 2. Setup YARP connection
            if (!yarp::os::Network::checkNetwork(2.0))
            {
                std::cout << "Could not connect to YARP" << std::endl;
                return false;
            }

            // 3. Set up module name
            setName((rf.check("name", Value("/moveenet_flow")).asString()).c_str());

            // 4. Read parameters from command line
            detF = rf.check("f_det", Value(10)).asInt32();
            double visF = rf.check("f_vis", Value(100.0)).asFloat64();
            std::string log_path = rf.check("log_path", Value("/data/new_scarfGNN_full/raw/cam2_S1_Directions/ch0dvs/data.log")).asString();
            std::string output_csv = rf.check("output_csv", Value("/home/moveEnetFlow/csv_file/251105_test3.csv")).asString();
            std::string output_video = rf.check("output_video", Value("/home/moveEnetFlow/avi_file/251105_test3.avi")).asString(); 
            is_visualize = rf.check("vis");
            
            yInfo() << "Configuration:";
            yInfo() << "  - Log path: " << log_path;
            yInfo() << "  - Output CSV: " << output_csv;
            yInfo() << "  - Output video: " << output_video;
            yInfo() << "  - Visualization: " << (is_visualize ? "ENABLED" : "DISABLED");
            yInfo() << "  - Detection freq: " << detF << " Hz";
            yInfo() << "  - Visualization freq: " << visF << " Hz"; 

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

            // 6. Start of moveEnet flow process
            yInfo() << "Starting MoveEnet flow process...";
            std::string command = "python3 /usr/local/src/hpe-core/example/movenet/movenet_online.py --checkpoint_path " + checkpoint_path + " &";
            system(command.c_str());

            // check if moveEnet process started
            while (!yarp::os::NetworkBase::exists("/movenet/sklt:o"))
                sleep(1);
            yInfo() << "MoveEnet started correctly";

            // 7. Initialise .csv output file
            csv_file.open(output_csv);
            csv_file << "timestamp";
            for (int j = 0; j < 13; j++) {
                csv_file << ",joint" << j << "_x,joint" << j << "_y,joint" << j << "_vx,joint" << j << "_vy,confidence" << j;
            }
            csv_file << "\n";

            // 8. Initialize video writer (always, independent of visualization)
            video_writer.open(output_video, cv::VideoWriter::fourcc('M','J','P','G'), thF, image_size);
            if (!video_writer.isOpened()) {
                yError() << "Could not open the output video for write: " << output_video;
                return false;
            }
            yInfo() << "Video writer initialized successfully: " << output_video;

            // 9. Initialize visualization window if requested
            if (is_visualize) {
                cv::namedWindow("moveenet-flow", cv::WINDOW_NORMAL);
                cv::resizeWindow("moveenet-flow", image_size);
                yInfo() << "Visualization window created";
            }

            // 9. Initialize MoveEnet detection handler
            if (!mn_handler.init(getName("/eros:o"), getName("/movenet:i"), detF))
            {
                yError() << "Could not open movenet ports";
                return false;
            }

            // 10. Connect MoveEnet ports
            Network::connect("/movenet/sklt:o", getName("/movenet:i"), "fast_tcp");
            Network::connect(getName("/eros:o"), "/movenet/img:i", "fast_tcp");

            // 11. Initialize velocity estimator
            // velocity_estimator doesn't need explicit initialization

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

    void drawEVENTS(cv::Mat &img)
    {
        cv::Mat eventsmono;
        binary_handler.getSurface().convertTo(eventsmono, CV_8U);
        cv::cvtColor(eventsmono, img, CV_GRAY2BGR);
    }

    void drawSAE(cv::Mat &img)
    {
        cv::Mat sae64, saemono;
        
        sae_handler.getSurface().copyTo(sae64);
        double maxval;
        cv::minMaxLoc(sae64, nullptr, &maxval);
        sae64 -= (maxval - 1.0);  //show 2 seconds of surface
        //cv::threshold(sae64, sae64, 0, 0, cv::THRESH_BINARY);
        sae64.convertTo(saemono, CV_8U, 255.0);
        cv::cvtColor(saemono, img, CV_GRAY2BGR);
    }

    bool updateModule() override
    {
        // ===== STEP 1: LOAD AND PROCESS EVENTS =====
        static double tnow = 0.0;
        static double pts = 0.0;
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
            yInfo() << "Event stream reset detected";
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
        
        // Log progress every 100 frames
        if (batch_count % 100 == 0) {
            yInfo() << "Processed frame " << batch_count << " at t=" << tnow << "s, events=" << event_count;
        }
        
        // Check if we've reached the end of the file (no more events)
        if (event_count == 0 && tnow > 0) {
            yInfo() << "Finished processing event file (no more events). Total frames: " << batch_count;
            return false; // Stop the module
        }
        
        // ===== STEP 2: POSE DETECTION WITH MOVENET =====
        bool was_detected = mn_handler.update(eros_handler.getSurface(), tnow, detected_pose);
        
        // ===== STEP 3: VELOCITY ESTIMATION =====
        if (was_detected && hpecore::poseNonZero(detected_pose.pose)) {
            // Estimate velocities using SAE surface
            auto jvs = velocity_estimator.multi_area_velocity(sae_handler.getSurface(), tnow, detected_pose.pose, roiSize);
            
            // ===== STEP 4: SAVE RESULTS TO CSV =====
            csv_file << std::fixed << std::setprecision(6) << tnow;
            for (int j = 0; j < 13; j++) {
                csv_file << "," << detected_pose.pose[j].u 
                        << "," << detected_pose.pose[j].v
                        << "," << jvs[j].u
                        << "," << jvs[j].v
                        << "," << detected_pose.conf[j];
            }
            csv_file << "\n";
            csv_file.flush();  // Ensure data is written to disk
        }
        
        // ===== STEP 5: VISUALIZATION AND VIDEO WRITING =====
        static cv::Mat canvas = cv::Mat(image_size, CV_8UC3);
        canvas.setTo(cv::Vec3b(0, 0, 0));
        
        // Draw EROS representation
        drawEROS(canvas);
        
        // Draw detected skeleton (with validation to prevent OpenCV errors)
        if (hpecore::poseNonZero(detected_pose.pose)) {
            try {
                hpecore::drawSkeleton(canvas, detected_pose, {0, 0, 255}, 3, c_thresh);
            } catch (const cv::Exception& e) {
                // Silently skip drawing if OpenCV throws an error (invalid pose coordinates)
                // This can happen if pose detection returns out-of-bounds or invalid values
            }
        }
        
        // Always write frame to video file
        if (video_writer.isOpened()) {
            video_writer.write(canvas);
        }
        
        // Display frame only if visualization is enabled
        if (is_visualize) {
            cv::imshow("moveenet-flow", canvas);
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
    MOVEENET_FLOW instance;
    return instance.runModule(rf);          // This calls: updateModule() loop -> close()
}

/*
 * Author: Daria Berretta
 *
 * This system processes event data from .log files offline to:
 * 1. Generate event representations (EROS, SAE, Binary)
 * 2. Detect human poses using MoveEnet
 * 3. Estimate joint velocities
 * 4. Save results to CSV files for each data.log

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
#include <filesystem>
#include <regex>

using namespace yarp::os;
using namespace yarp::sig;
using namespace std::filesystem;
std::vector<std::string> vector;


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

        // Save output csv file
        std::ofstream csv_file;

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
        double th_period{0.01};
        double c_thresh{0.4};

        // Data processing parameters
        std::string data_root_path;
        std::string output_csv_dir;
        std::vector<std::string> log_files_to_process;

                    
    

    public:

        bool configure(yarp::os::ResourceFinder &rf) override
        {
            // 1. If request print help for command line and exit
            if(rf.check("help"))
            {
                yInfo() << "--help:";
                yInfo() << "--data_root <string>: root directory containing subdirectories with data.log files";
                yInfo() << "--output_csv_dir <string>: directory to save output CSV files";
                yInfo() << "--f_det <int>: detection frequency (default 10Hz)";
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
            data_root_path = rf.check("data_root", Value("/data/new_scarfGNN_full/raw")).asString();
            output_csv_dir = rf.check("output_csv_dir", Value("/home/moveEnetFlow/csv_file")).asString();

            yInfo() << "Configuration:";
            yInfo() << "  - Data root path: " << data_root_path;
            yInfo() << "  - Output CSV directory: " << output_csv_dir;
            yInfo() << "  - Detection freq: " << detF << " Hz";

            // 5. Find all data.log files in subdirectories
            findLogFiles(data_root_path);
            if (log_files_to_process.empty()) {
                yError() << "No data.log files found in " << data_root_path;
                return false;
            }
            yInfo() << "Found " << log_files_to_process.size() << " data.log files to process";

            // 6. Initialize internal parameters
            image_size = cv::Size(rf.check("w", Value(640)).asInt32(),
                                  rf.check("h", Value(480)).asInt32());
            roiSize = rf.check("roi", Value(20)).asInt32();

            std::string checkpoint_path = rf.check("checkpoint_path", Value("/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth")).asString();
            c_thresh = rf.check("confidence", Value(0.4)).asFloat64();

            // Initialize EROS, SAE, Binary handlers
            eros_handler.init(image_size.width, image_size.height, 7, 0.3);
            binary_handler.init(image_size.width, image_size.height);
            sae_handler.init(image_size.width, image_size.height);

            // 7. Start of moveEnet flow process
            yInfo() << "Starting MoveEnet flow process...";
            std::string command = "python3 /usr/local/src/hpe-core/example/movenet/movenet_online.py --checkpoint_path " + checkpoint_path + " &";
            system(command.c_str());

            // check if moveEnet process started
            while (!yarp::os::NetworkBase::exists("/movenet/sklt:o"))
                sleep(1);
            yInfo() << "MoveEnet started correctly";

            // 8. Initialize MoveEnet detection handler
            if (!mn_handler.init(getName("/eros:o"), getName("/movenet:i"), detF))
            {
                yError() << "Could not open movenet ports";
                return false;
            }

            // 9. Connect MoveEnet ports
            Network::connect("/movenet/sklt:o", getName("/movenet:i"), "fast_tcp");
            Network::connect(getName("/eros:o"), "/movenet/img:i", "fast_tcp");

            return true;
        }

        void findLogFiles(const std::string& root_path) {
            log_files_to_process.clear();
            try {
                for (const auto& entry : recursive_directory_iterator(root_path)) {
                    if (entry.is_regular_file() && entry.path().filename() == "data.log") {
                        log_files_to_process.push_back(entry.path().string());
                    }
                }
            } catch (const filesystem_error& e) {
                yError() << "Error scanning directory: " << e.what();
            }

            // Sort files for consistent processing order
            std::sort(log_files_to_process.begin(), log_files_to_process.end());
        }

        std::string generateCsvFilename(const std::string& log_path) {
            // Extract the subdirectory name (e.g., "cam2_S11_Directions")
            path log_file_path(log_path);
            path parent_dir = log_file_path.parent_path().parent_path(); // Go up two levels from ch0dvs/data.log
            std::string dirname = parent_dir.filename().string();

            // Create CSV filename: dirname.csv
            return output_csv_dir + "/" + dirname + ".csv";
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

    bool updateModule() override
    {
        static size_t current_file_index = 0;
        static bool file_initialized = false;
        static double tnow = 0.0;
        static double pts = 0.0;
        static int batch_count = 0;

        // Check if we've processed all files
        if (current_file_index >= log_files_to_process.size()) {
            yInfo() << "All files processed successfully!";
            return false;
        }

        std::string current_log_path = log_files_to_process[current_file_index];

        // Initialize new file if not already done
        if (!file_initialized) {
            // Reset surfaces and time
            sae_handler.getSurface().setTo(0.0);
            binary_handler.getSurface().setTo(0.0);
            eros_handler.getSurface().setTo(0.0);
            tnow = 0.0;
            pts = 0.0;
            batch_count = 0;

            // Generate CSV filename and open file
            std::string csv_filename = generateCsvFilename(current_log_path);
            csv_file.open(csv_filename);
            if (!csv_file.is_open()) {
                yError() << "Could not open CSV file: " << csv_filename;
                current_file_index++;
                return true; // Skip this file and continue
            }

            // Write CSV header
            csv_file << "timestamp";
            for (int j = 0; j < 13; j++) {
                csv_file << ",joint" << j << "_x,joint" << j << "_y,joint" << j << "_vx,joint" << j << "_vy,confidence" << j;
            }
            csv_file << "\n";

            // Load the log file
            if (!eloader.load(current_log_path)) {
                yError() << "Could not open event log file: " << current_log_path;
                csv_file.close();
                current_file_index++;
                return true; // Skip this file and continue
            }

            yInfo() << "Processing file " << (current_file_index + 1) << "/" << log_files_to_process.size()
                    << ": " << path(current_log_path).parent_path().parent_path().filename().string();
            yInfo() << "Log file: " << current_log_path;
            yInfo() << "CSV output: " << csv_filename;

            file_initialized = true;
        }

        // ===== STEP 1: LOAD AND PROCESS EVENTS =====
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
            yInfo() << "Finished processing file. Total frames: " << batch_count;

            // Close current CSV file
            if (csv_file.is_open()) {
                csv_file.close();
                yInfo() << "CSV file closed";
            }

            // Move to next file
            current_file_index++;
            file_initialized = false;

            // If there are more files, continue processing
            if (current_file_index < log_files_to_process.size()) {
                return true;
            } else {
                yInfo() << "All files processed successfully!";
                return false;
            }
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

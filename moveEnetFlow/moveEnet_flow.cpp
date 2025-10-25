/*
Author: Daria Berretta
 */

#include <yarp/cv/Cv.h>
#include <yarp/os/all.h>
#include <yarp/sig/Image.h>
#include <event-driven/core.h>
#include <hpe-core/utility.h>
#include <hpe-core/motion_estimation.h>
#include <hpe-core/fusion.h>
#include <hpe-core/motion.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "april_msgs/yarp/rosmsg/april_msgs/NChumanPose.h"
#include <yarp/rosmsg/sensor_msgs/Image.h>

using namespace yarp::os;       // import all names from yarp::os, so i don't have to write yarp::os:: before every class
using namespace yarp::sig;      // same logic for yarp::sig
using std::vector;              // import only vector, so i can write vector instead of std::vector

class MoveEnetFlowModule : public yarp::os::RFModule {
private:
    // Event loader from .log file
    ev::offlineLoader<ev::AE> eloader

    // detection handlers
    externalDetector mn_handler;
    delayedGT gt_handler;
    hpecore::EROS eros_handler;
    hpecore::SAE sae_handler;
    hpecore::BIN binary_handler;

    // velocity and fusion
    hpecore::pwtripletvelocity velocity_estimator;
    hpecore::pwTripletVelocity pw_trip_velocity;
    hpecore::multiJointLatComp state;

    // internal data structures
    hpecore::stampedPose detected_pose;

    cv::Size image_size;
    cv::Mat edpr_logo;

    // parameters
    int detF{10}, roiSize{20};
    bool pltVel{false}, pltDet{false}, pltTra{false};
    int alt_view{0};
    bool latency_compensation{true};
    double scaler{1.0};
    double th_period{0.01}, thF{100.0};
    bool pltRoi{false};
    double c_thresh{0.4};
    cv::Scalar colors[13] = {{0, 0, 180}, {0, 180, 0}, {0, 0, 180},
                            {180, 180, 0}, {180, 0, 180}, {0, 180, 180},
                            {120, 0, 180}, {120, 180, 0}, {0, 120, 180},
                            {120, 120, 180}, {120, 180, 120}, {120, 120, 180}, {120, 120, 120}};
    bool started{false};
    double tnow;

    public:
    bool configure(yarp::os::ResourceFinder &rf) override
    {

        if(rf.check("help")) {
            yInfo() << " EDPR APRIL HPE ";
            yInfo() << "--name <string> : name of module for YARP ports";
            yInfo() << "--f_vis <float> : visualisation rate [20]";
            yInfo() << "--f_det <float> : HPE detection rate [5]";
            yInfo() << "--pu <float>    : KF process uncertainty [10.0]";
            yInfo() << "--muD <float>   : KF measurement uncertainty [1.0]";
            yInfo() << "--confidence <float> : threshold for skeleton confidence [0.4]";
            return false;
        }

        // =====SET UP YARP=====
        if (!yarp::os::Network::checkNetwork(2.0))
        {
            std::cout << "Could not connect to YARP" << std::endl;
            return false;
        }

        // set the module name used to name ports
        setName((rf.check("name", Value("/edpr_april")).asString()).c_str());

        if (!input_events.open(getName("/AE:i")))
        {
            yError() << "Could not open events input port";
            return false;
        }
        

        // =====READ PARAMETERS=====
        pltDet = rf.check("pltDet") && rf.check("pltDet", Value(true)).asBool();
        pltTra = rf.check("pltTra") && rf.check("pltTra", Value(true)).asBool();
        pltRoi = rf.check("pr") && rf.check("pr", Value(true)).asBool();
        detF = rf.check("f_det", Value(10)).asInt32();
        image_size = cv::Size(rf.check("w", Value(640)).asInt32(),
                              rf.check("h", Value(480)).asInt32());
        roiSize = rf.check("roi", Value(20)).asInt32();
        double procU = rf.check("pu", Value(1e-1)).asFloat64();
        double measUD = rf.check("muD", Value(1e-4)).asFloat64();
        double measUV = rf.check("muV", Value(0)).asFloat64();
        std::string checkpoint_path = rf.check("checkpoint_path", Value("/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth")).asString();
        latency_compensation = rf.check("use_lc") && rf.check("use_lc", Value(true)).asBool();
        double lc = latency_compensation ? 1.0 : 0.0;
        thF = rf.check("f_vis", Value(100.0)).asFloat64();
        th_period = 1/thF;
        c_thresh = rf.check("confidence", Value(0.4)).asFloat64();

        // pltDet = true;
        pltTra = true;
        
        // concatenate the checkpoint path
        std::string command = "python3 /usr/local/src/hpe-core/example/movenet/movenet_online.py --gpu --checkpoint_path " + checkpoint_path + " &";
        int r = system(command.c_str());

        while (!yarp::os::NetworkBase::exists("/movenet/sklt:o"))
            sleep(1);
        yInfo() << "MoveEnet started correctly";

        if (!mn_handler.init(getName("/eros:o"), getName("/movenet:i"), detF))
        {
            yError() << "Could not open movenet ports";
            return false;
        }

        // ===== SET UP INTERNAL VARIABLE/DATA STRUCTURES =====

        // shared images
        eros_handler.init(image_size.width, image_size.height, 7, 0.3);
        binary_handler.init(image_size.width, image_size.height);
        sae_handler.init(image_size.width, image_size.height);

        edpr_logo = cv::imread("/usr/local/src/EDPR-APRIL/edpr_logo.png");
        
        //velocity estimation
        pw_trip_velocity.setParameters(roiSize, 1, image_size);

        // fusion
        if (!state.initialise({procU, measUD, measUV, lc}))
        {
            yError() << "Not KF initialized";
            return false;
        }

        // ===== TRY DEFAULT CONNECTIONS =====
        Network::connect("/file/ch0dvs:o", getName("/AE:i"), "fast_tcp");
        Network::connect("/atis3/AE:o", getName("/AE:i"), "fast_tcp");
        Network::connect("/file/ch2GT50Hzskeleton:o", getName("/gt:i"), "fast_tcp");
        Network::connect("/movenet/sklt:o", getName("/movenet:i"), "fast_tcp");
        Network::connect("/zynqGrabber/AE:o", getName("/AE:i"), "fast_tcp");
        Network::connect(getName("/eros:o"), "/movenet/img:i", "fast_tcp");
        Network::connect("/file/atis/AE:o", getName("/AE:i"), "fast_tcp");

        cv::namedWindow("edpr-april", cv::WINDOW_NORMAL);
        cv::resizeWindow("edpr-april", image_size);

        // set-up ROS interface
        ros_node = new yarp::os::Node("/edpraprilhpe");
        if (!ros_publisher.topic("/pem/neuromorphic_camera/data"))
        {
            yError() << "Could not open ROS pose output publisher";
            return false;
        }

        if (!publisherPort_eros.topic("/isim/neuromorphic_camera/eros"))
        {
            yError() << "Could not open ROS EROS output publisher";
            return false;
        }

        if (!publisherPort_evs.topic("/isim/neuromorphic_camera/evs"))
        {
            yError() << "Could not open ROS EVS output publisher";
            return false;
        }
        
        camera_handler_thread = std::thread([this]{ this->run_camera_interface(); });
        hpe_thread = std::thread([this]{ this->run_hpe(); });

        return true;
    }


}



int main(int argc, char *argv[]){

    // Prepare the resource finder

    yarp::os::ResourceFinder rf;

    rf.setVerbose(true);
    rf.configure(argc, argv);

    // Display help if required
     if (rf.check("help")) {
        yInfo() << "--data\t<string>\t: path to input event dataset";
        yInfo() << "--input_rosbag\t<string>\t: path to input rosbag dataset";
        yInfo() << "--output_rosbag\t<string>\t: path to output EROS as rosbag file";
        yInfo() << "--kernel_size \t<int>\t: eros kernel size";
        yInfo() << "--alpha \t<double>\t: events decay factor [0.3]";
        yInfo() << "--ch \t<int>\t --cw \t<int>\t: height and width of camera resolution";
        yInfo() << "--vis: flag to visualize EROS on display";
        return false;
    }

    
    return 0;
}
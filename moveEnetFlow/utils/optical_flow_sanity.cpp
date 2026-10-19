/*
 * optical_flow_sanity.cpp
 *
 * Standalone, non-invasive visual sanity check for MoveEnetOFK.
 * Does not include or modify moveEnetOFK_offline.cpp.
 *
 * Panel 1: EROS, raw MoveEnet detections (red), fused OFK pose (cyan).
 * Panel 2: binary recent event activity, integrated OF pose (green),
 *          fused OFK pose (cyan), per-joint OF vectors (yellow), optional ROIs.
 *
 * The integrated-OF diagnostic reuses the SAME measured velocities as OFK,
 * whose ROIs remain centred on the KF state. It is NOT an independent
 * optical-flow tracker. In reset_on_detection mode (default) the green
 * integrated pose restarts from the corrected KF pose at each valid detection.
 * In continuous mode, it is initialised once and never corrected.
 *
 * Video timestamps come from the dataset. The saved MP4 plays at --fps;
 * its number of frames is ceil((last_packet_ts-first_packet_ts)*fps), so
 * output duration differs from the original packet span by < 1 frame.
 *
 * Model server uses its unmodified stock YARP names /movenet/*.
 * For safety, refuses to start if those names are already in use.
 */

#include <yarp/os/all.h>
#include <yarp/cv/Cv.h>
#include <yarp/sig/Image.h>

#include <event-driven/core.h>
#include <event-driven/algs.h>
#include <hpe-core/utility.h>
#include <hpe-core/motion.h>
#include <hpe-core/fusion.h>
#include <hpe-core/representations.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using yarp::os::Bottle;
using yarp::os::BufferedPort;
using yarp::os::Network;
using yarp::os::NetworkBase;
using yarp::os::ResourceFinder;
using yarp::os::Value;
using yarp::sig::ImageOf;
using yarp::sig::PixelMono;

namespace {
constexpr int kJoints = 13;
constexpr double kEps = 1e-9;
constexpr int kHeader = 66;
const cv::Scalar kRawColor(0, 0, 255);      // BGR: red
const cv::Scalar kFusedColor(255, 255, 0);  // BGR: cyan
const cv::Scalar kIntegratedColor(0, 255, 0); // BGR: green
const cv::Scalar kFlowColor(0, 220, 255);   // BGR: yellow
const cv::Scalar kRoiColor(200, 120, 200);

volatile std::sig_atomic_t gStop = 0;
void signalHandler(int) { gStop = 1; }

const std::array<std::pair<int, int>, 14> kBones = {{
    {0, 1}, {0, 2}, {1, 2}, {1, 3}, {3, 7}, {2, 4}, {4, 8},
    {1, 6}, {2, 5}, {5, 6}, {5, 10}, {10, 12}, {6, 9}, {9, 11}
}};

bool isJoint(const hpecore::joint &j, const cv::Size &sz) {
    return std::isfinite(j.u) && std::isfinite(j.v) &&
           (j.u > 0 || j.v > 0) &&
           j.u >= 0 && j.v >= 0 && j.u < sz.width && j.v < sz.height;
}
cv::Point asPoint(const hpecore::joint &j) {
    return {static_cast<int>(std::lround(j.u)), static_cast<int>(std::lround(j.v))};
}
std::array<bool, kJoints> selectJoints(const std::string &s) {
    std::array<bool, kJoints> result{};
    if (s == "all" || s.empty()) {
        result.fill(true);
        return result;
    }
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) throw std::runtime_error("Empty joint index in joint list");
        std::size_t parsed = 0;
        int index = std::stoi(token, &parsed);
        if (parsed != token.size() || index < 0 || index >= kJoints) {
            throw std::runtime_error("Joint indices must lie in [0,12]: " + token);
        }
        result[index] = true;
    }
    if (std::none_of(result.begin(), result.end(), [](bool v){ return v; })) {
        throw std::runtime_error("At least one joint must be selected");
    }
    return result;
}

void drawText(cv::Mat &canvas, const std::string &s, cv::Point at,
              double scale = 0.45, const cv::Scalar &color = cv::Scalar(255, 255, 255)) {
    cv::putText(canvas, s, at, cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(10, 10, 10), 3, cv::LINE_AA);
    cv::putText(canvas, s, at, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}
void drawPose(cv::Mat &img, const hpecore::skeleton13 &pose,
              const std::array<bool,kJoints> &selected,
              const cv::Scalar &color, int thickness = 2) {
    for (auto edge : kBones) {
        const int a = edge.first, b = edge.second;
        if (selected[a] && selected[b] &&
            isJoint(pose[a], img.size()) && isJoint(pose[b], img.size())) {
            cv::line(img, asPoint(pose[a]), asPoint(pose[b]), color, thickness, cv::LINE_AA);
        }
    }
    for (int i = 0; i < kJoints; ++i) {
        if (selected[i] && isJoint(pose[i], img.size())) {
            cv::circle(img, asPoint(pose[i]), 3, color, cv::FILLED, cv::LINE_AA);
        }
    }
}

struct Config {
    std::string dataFile;
    std::string outputVideo;
    std::string checkpoint;
    std::string pythonScript;
    std::string pythonBin;
    std::string device;
    cv::Size resolution;
    cv::Size transport;
    double netPeriod{0.2};
    double flowPeriod{0.005};
    double fps{50.0};
    double eventWindow{0.03};
    double arrowHorizon{0.05};
    double arrowScale{1.0};
    double arrowLength{18.0};
    double procU{0.77};
    double measUD{0.06};
    double measUV{0.97};
    int roi{20};
    bool latencyComp{true};
    bool showRois{true};
    bool showStatus{true};
    bool preview{false};
    bool normalizedArrows{false};
    bool resetIntegratedOnDetection{true};
    std::array<bool, kJoints> joints{};
    std::array<bool, kJoints> roiJoints{};
};

Config parseConfig(int argc, char **argv) {
    ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);
    if (rf.check("help")) {
        std::cout << R"HELP(
Usage: optical_flow_sanity --data_file <data.log> --output_video <result.mp4> [options]

Required:
  --data_file <path>        hpe-core event log (eH36M or DHP19)
  --output_video <path>     MP4 to create
Options:
  --dhp19                   346x260 input, pad MoveEnet transport to 352x260
  --w <int> --h <int>       Event camera resolution (default 640x480)
  --net_period <seconds>    MoveEnet detections, default 0.2 (5Hz)
  --flow_period <seconds>   OF and KF updates, default 0.005 (200Hz)
  --fps <number>            Video playback frame rate, default 50
  --event_window <seconds>  Recent event activity window, default 0.03
  --joints <all|0,3,7>     Skeleton/arrow joints to display, default all
  --roi_joints <all|0,3>   ROI overlay joints to display, default all
  --no_rois                 Hide all ROIs
  --no_status               Hide detection markers and elapsed time
  --arrow_mode <scaled|normalized>   default scaled
  --arrow_horizon <sec>     Scaled arrows show velocity * horizon, default 0.05
  --arrow_scale <number>    Additional scale multiplier, default 1
  --arrow_length <px>      Normalized arrow length, default 18
  --integration_mode <reset_on_detection|continuous>
                            Green OF-only pose: reset at each valid MoveEnet
                            detection (default), or integrate throughout video
  --roi <int>               OF ROI radius used in algorithm, default 20
  --pu <value> --muD <value> --muV <value>  KF uncertainties
  --use_lc <0|1>           Enable KF latency compensation, default 1
  --device <cpu|cuda:N>    default cuda:0
  --checkpoint_path <path>  MoveEnet checkpoint (dataset-dependent default)
  --movenet_script <path>  Stock hpe-core movenet_online.py
  --python <executable>     Python interpreter, default python3
  --vis                     Show optional interactive preview (q to stop)

The executable starts its own stock MoveEnet process ONLY if /movenet/*
ports are free. Does not kill, unregister, or reuse existing MoveEnet servers.
No CSV, no ground truth, no changes to the original benchmark executable.
)HELP";
        std::exit(0);
    }
    Config c;
    c.dataFile = rf.check("data_file", Value("")).asString();
    c.outputVideo = rf.check("output_video", Value("")).asString();
    const bool dhp19 = rf.check("dhp19");
    c.resolution = dhp19 ? cv::Size(346, 260) :
        cv::Size(rf.check("w", Value(640)).asInt32(), rf.check("h", Value(480)).asInt32());
    c.transport = dhp19 ? cv::Size(352, 260) : c.resolution;
    c.netPeriod = rf.check("net_period", Value(0.2)).asFloat64();
    c.flowPeriod = rf.check("flow_period", Value(0.005)).asFloat64();
    c.fps = rf.check("fps", Value(50.0)).asFloat64();
    c.eventWindow = rf.check("event_window", Value(0.03)).asFloat64();
    c.arrowHorizon = rf.check("arrow_horizon", Value(0.05)).asFloat64();
    c.arrowScale = rf.check("arrow_scale", Value(1.0)).asFloat64();
    c.arrowLength = rf.check("arrow_length", Value(18.0)).asFloat64();
    c.roi = rf.check("roi", Value(20)).asInt32();
    c.procU = rf.check("pu", Value(dhp19 ? 0.36 : 0.77)).asFloat64();
    c.measUD = rf.check("muD", Value(dhp19 ? 0.01 : 0.06)).asFloat64();
    c.measUV = rf.check("muV", Value(dhp19 ? 0.01 : 0.97)).asFloat64();
    c.latencyComp = rf.check("use_lc", Value(true)).asBool();
    c.showRois = !rf.check("no_rois");
    c.showStatus = !rf.check("no_status");
    c.preview = rf.check("vis");
    c.device = rf.check("device", Value("cuda:0")).asString();
    c.pythonScript = rf.check("movenet_script", Value("/usr/local/src/hpe-core/example/movenet/movenet_online.py")).asString();
    c.pythonBin = rf.check("python", Value("python3")).asString();
    c.checkpoint = rf.check("checkpoint_path", Value(dhp19 ?
        "/usr/local/src/hpe-core/example/movenet/models/dhp19_allcams_e33_valacc0.87996.pth" :
        "/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth")).asString();
    c.joints = selectJoints(rf.check("joints", Value("all")).asString());
    c.roiJoints = selectJoints(rf.check("roi_joints", Value("all")).asString());
    const std::string arrowMode = rf.check("arrow_mode", Value("scaled")).asString();
    if (arrowMode != "scaled" && arrowMode != "normalized")
        throw std::runtime_error("--arrow_mode must be 'scaled' or 'normalized'");
    c.normalizedArrows = arrowMode == "normalized";
    const std::string integrationMode = rf.check(
        "integration_mode", Value("reset_on_detection")).asString();
    if (integrationMode != "reset_on_detection" && integrationMode != "continuous")
        throw std::runtime_error("--integration_mode must be 'reset_on_detection' or 'continuous'");
    c.resetIntegratedOnDetection = (integrationMode == "reset_on_detection");
    if (c.dataFile.empty() || c.outputVideo.empty())
        throw std::runtime_error("--data_file and --output_video are required");
    if (!std::filesystem::exists(c.dataFile))
        throw std::runtime_error("Dataset file not found: " + c.dataFile);
    if (!std::filesystem::exists(c.pythonScript))
        throw std::runtime_error("MoveEnet script not found: " + c.pythonScript);
    if (!std::filesystem::exists(c.checkpoint))
        throw std::runtime_error("MoveEnet checkpoint not found: " + c.checkpoint);
    auto positive = [](double v) { return std::isfinite(v) && v > 0; };
    if (!positive(c.netPeriod) || !positive(c.flowPeriod) || !positive(c.fps) ||
        !positive(c.eventWindow) || !positive(c.arrowHorizon) ||
        !positive(c.arrowScale) || !positive(c.arrowLength) ||
        c.roi < 3 || c.resolution.width < 5 || c.resolution.height < 5 ||
        !positive(c.procU) || !positive(c.measUD) || !positive(c.measUV))
        throw std::runtime_error("Invalid periods, scale, ROI, resolution or KF uncertainties");
    if (c.fps > 240) throw std::runtime_error("--fps > 240 is not recommended/supported");
    return c;
}

// Owns one private stock Python process and only the local YARP client ports.
// No pkill, no yarp unregister, no implicit connection to someone else's server.
class MoveEnetClient {
    BufferedPort<ImageOf<PixelMono>> output_;
    BufferedPort<Bottle> input_;
    std::string outputName_, inputName_;
    pid_t child_{-1};
    cv::Size sensorSize_, transportSize_;
    void cleanup() noexcept {
        output_.interrupt();
        input_.interrupt();
        output_.close();
        input_.close();
        if (child_ > 0) {
            ::kill(child_, SIGTERM);
            int status = 0;
            for (int i = 0; i < 20; ++i) {
                if (::waitpid(child_, &status, WNOHANG) == child_) { child_ = -1; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (child_ > 0) {
                ::kill(child_, SIGKILL);
                ::waitpid(child_, &status, 0);
                child_ = -1;
            }
        }
    }
public:
    explicit MoveEnetClient(const Config &c) : sensorSize_(c.resolution), transportSize_(c.transport) {
        try {
        if (!Network::checkNetwork(2.0))
            throw std::runtime_error("YARP server unavailable. Run: yarpserver &");
        if (NetworkBase::exists("/movenet/img:i") || NetworkBase::exists("/movenet/sklt:o"))
            throw std::runtime_error("/movenet ports already in use. Stop the other MoveEnet job before this standalone run; no processes were touched.");

        const std::string suffix = std::to_string(::getpid());
        outputName_ = "/of_sanity/eros:o_" + suffix;
        inputName_ = "/of_sanity/pose:i_" + suffix;
        if (!output_.open(outputName_) || !input_.open(inputName_))
            throw std::runtime_error("Could not open sanity-check YARP client ports");

        const pid_t pid = ::fork();
        if (pid < 0) throw std::runtime_error("fork() failed");
        if (pid == 0) {
            std::vector<std::string> args = {
                c.pythonBin, c.pythonScript,
                "--checkpoint_path", c.checkpoint,
                "--w", std::to_string(c.transport.width),
                "--h", std::to_string(c.transport.height)
            };
            if (c.device != "cpu") {
                std::string index = c.device;
                if (index.rfind("cuda:", 0) == 0) index = index.substr(5);
                if (index.empty() || index.find_first_not_of("0123456789") != std::string::npos) {
                    std::cerr << "Unsupported --device: " << c.device << std::endl;
                    _exit(127);
                }
                args.push_back("--gpu");
                args.push_back("--GPU_ID");
                args.push_back(index);
            }
            std::vector<char *> argv;
            for (auto &arg : args) argv.push_back(const_cast<char *>(arg.c_str()));
            argv.push_back(nullptr);
            ::execvp(argv[0], argv.data());
            std::perror("execvp failed");
            _exit(127);
        }
        child_ = pid;

        bool ready = false;
        for (int tries = 0; tries < 600 && !gStop; ++tries) {
            if (NetworkBase::exists("/movenet/img:i") && NetworkBase::exists("/movenet/sklt:o")) {
                ready = true;
                break;
            }
            int status = 0;
            if (::waitpid(child_, &status, WNOHANG) == child_) {
                child_ = -1;
                throw std::runtime_error("MoveEnet Python process exited during startup; check its error above");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!ready) throw std::runtime_error("MoveEnet ports did not appear within 60 seconds");
        if (!Network::connect(outputName_, "/movenet/img:i", "fast_tcp") ||
            !Network::connect("/movenet/sklt:o", inputName_, "fast_tcp"))
            throw std::runtime_error("Could not connect local YARP ports to MoveEnet");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    MoveEnetClient(const MoveEnetClient &) = delete;
    MoveEnetClient &operator=(const MoveEnetClient &) = delete;

    bool infer(const cv::Mat &erosSurface, double packetTs, hpecore::stampedPose &out) {
        cv::Mat img;
        erosSurface.convertTo(img, CV_8U);
        cv::GaussianBlur(img, img, cv::Size(5, 5), 0, 0);
        if (img.size() != transportSize_) {
            cv::Mat padded = cv::Mat::zeros(transportSize_, CV_8U);
            if (img.cols <= padded.cols && img.rows <= padded.rows)
                img.copyTo(padded(cv::Rect(0, 0, img.cols, img.rows)));
            else
                cv::resize(img, padded, transportSize_, 0, 0, cv::INTER_AREA);
            img = padded;
        }
        ImageOf<PixelMono> &yarpImg = output_.prepare();
        yarpImg.copy(yarp::cv::fromCvMat<PixelMono>(img));
        const double startWall = yarp::os::Time::now();
        output_.write();
        Bottle *reply = input_.read(true);
        if (!reply) return false;
        out.pose = hpecore::extractSkeletonFromYARP<Bottle>(*reply);
        out.conf = hpecore::extractConfidenceFromYARP<Bottle>(*reply);
        out.timestamp = packetTs;
        out.delay = yarp::os::Time::now() - startWall;
        for (auto &j : out.pose) {
            j.u = std::clamp(j.u, 0.0f, static_cast<float>(sensorSize_.width - 1));
            j.v = std::clamp(j.v, 0.0f, static_cast<float>(sensorSize_.height - 1));
        }
        return hpecore::poseNonZero(out.pose);
    }
    ~MoveEnetClient() { cleanup(); }
};

// Rendering is deliberately separate from algorithmic updates.
class SanityVideo {
    cv::VideoWriter writer_;
    cv::Size sensor_;
    Config c_;
    cv::Mat lastEvent_; // latest packet timestamp per pixel, double
    std::size_t frames_{0};
    bool stopped_{false};
public:
    explicit SanityVideo(const Config &c) : sensor_(c.resolution), c_(c),
        lastEvent_(c.resolution, CV_64FC1, cv::Scalar(-1e30)) {
        const std::filesystem::path p(c.outputVideo);
        if (!p.parent_path().empty()) std::filesystem::create_directories(p.parent_path());
        const cv::Size outSize(2*c.resolution.width, c.resolution.height + kHeader);
        writer_.open(c.outputVideo, cv::VideoWriter::fourcc('m','p','4','v'), c.fps, outSize, true);
        if (!writer_.isOpened()) throw std::runtime_error("VideoWriter failed for " + c.outputVideo + "; check MP4 support");
        if (c.preview) cv::namedWindow("Optical flow sanity check", cv::WINDOW_NORMAL);
    }
    void markEvent(int x, int y, double t) { lastEvent_.at<double>(y,x) = t; }
    bool stopped() const { return stopped_; }
    std::size_t frames() const { return frames_; }

    void emit(double time, double startTime,
              const cv::Mat &erosSurface,
              bool poseReady, bool hasRaw,
              const hpecore::stampedPose &raw,
              const hpecore::skeleton13 &fused,
              const hpecore::skeleton13 &ofIntegrated,
              const hpecore::skeleton13 &velocity,
              const hpecore::skeleton13 &roiCentres,
              bool hasFlow, double lastDetectionTs) {
        if (stopped_) return;
        cv::Mat leftGray, left, right(sensor_, CV_8UC3, cv::Scalar(13, 13, 13));
        erosSurface.convertTo(leftGray, CV_8U);
        cv::GaussianBlur(leftGray, leftGray, cv::Size(9,9), 0);
        cv::normalize(leftGray, leftGray, 0, 255, cv::NORM_MINMAX);
        cv::cvtColor(leftGray, left, cv::COLOR_GRAY2BGR);

        // Binary accumulation of events in [video_time - event_window, video_time].
        // Uses packet timestamps, just like the pipeline's SAE/EROS updates.
        for (int y = 0; y < sensor_.height; ++y) {
            const double *src = lastEvent_.ptr<double>(y);
            cv::Vec3b *dst = right.ptr<cv::Vec3b>(y);
            for (int x = 0; x < sensor_.width; ++x) {
                if (src[x] >= time - c_.eventWindow && src[x] <= time + kEps)
                    dst[x] = cv::Vec3b(220, 220, 220);
            }
        }

        if (hasRaw) drawPose(left, raw.pose, c_.joints, kRawColor, 2);
        if (poseReady) {
            drawPose(left, fused, c_.joints, kFusedColor, 2);
            drawPose(right, ofIntegrated, c_.joints, kIntegratedColor, 2);
            drawPose(right, fused, c_.joints, kFusedColor, 1);
        }

        if (poseReady && hasFlow) {
            for (int i = 0; i < kJoints; ++i) {
                if (!isJoint(roiCentres[i], sensor_)) continue;
                const cv::Point origin = asPoint(roiCentres[i]);
                if (c_.showRois && c_.roiJoints[i]) {
                    const cv::Rect roi(origin.x-c_.roi, origin.y-c_.roi,
                                       c_.roi*2+1, c_.roi*2+1);
                    const cv::Rect clipped = roi & cv::Rect(cv::Point(0,0), sensor_);
                    cv::rectangle(right, clipped, kRoiColor, 1, cv::LINE_AA);
                }
                if (!c_.joints[i]) continue;
                const auto &v = velocity[i];
                if (!std::isfinite(v.u) || !std::isfinite(v.v)) continue;
                const double speed = std::hypot(v.u, v.v);
                if (speed < 1e-6) continue;
                double dx = v.u, dy = v.v;
                if (c_.normalizedArrows) {
                    dx *= c_.arrowLength/speed;
                    dy *= c_.arrowLength/speed;
                } else {
                    dx *= c_.arrowHorizon*c_.arrowScale;
                    dy *= c_.arrowHorizon*c_.arrowScale;
                }
                if (std::hypot(dx,dy) > 10000) continue;
                const cv::Point destination(origin.x + static_cast<int>(std::lround(dx)),
                                            origin.y + static_cast<int>(std::lround(dy)));
                cv::arrowedLine(right, origin, destination, kFlowColor, 2, cv::LINE_AA, 0, 0.25);
            }
        }

        cv::Mat canvas(sensor_.height + kHeader, sensor_.width*2, CV_8UC3, cv::Scalar(25,25,25));
        left.copyTo(canvas(cv::Rect(0,kHeader,sensor_.width,sensor_.height)));
        right.copyTo(canvas(cv::Rect(sensor_.width,kHeader,sensor_.width,sensor_.height)));
        cv::line(canvas, cv::Point(sensor_.width,0), cv::Point(sensor_.width,canvas.rows), cv::Scalar(70,70,70), 1);
        drawText(canvas, "EROS  |  net:red  OFK:cyan", cv::Point(8,20), 0.43);
        drawText(canvas, "Recent events | OF:green OFK:cyan",
                 cv::Point(sensor_.width+8,20), sensor_.width < 500 ? 0.38 : 0.43);
        if (c_.showStatus) {
            std::ostringstream t;
            t << std::fixed << std::setprecision(3) << "t=" << (time-startTime) << " s";
            if (hasRaw) {
                const double age = std::max(0.0, time-lastDetectionTs);
                t << " | net age=" << std::setprecision(3) << age << " s";
                if (age < 1.0/c_.fps + kEps) t << " [NEW DETECTION]";
            } else {
                t << " | waiting for first detection";
            }
            drawText(canvas, t.str(), cv::Point(8,47), 0.44);
            drawText(canvas, c_.normalizedArrows ? "OF arrows: direction only" :
                     "OF arrows: v * horizon * scale", cv::Point(sensor_.width+8,47), 0.43);
        }

        writer_.write(canvas);
        ++frames_;
        if (c_.preview) {
            cv::imshow("Optical flow sanity check", canvas);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) stopped_ = true;
        }
    }
    ~SanityVideo() {
        writer_.release();
        if (c_.preview) cv::destroyWindow("Optical flow sanity check");
    }
};

} // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    try {
        const Config c = parseConfig(argc, argv);
        ev::offlineLoader<ev::AE> loader;
        if (!loader.load(c.dataFile)) throw std::runtime_error("Could not load event log: " + c.dataFile);
        std::cout << "Loaded: " << loader.getinfo() << std::endl;

        // No interaction with the benchmark executable. The process we start
        // belongs exclusively to this tool and is terminated on normal exit.
        MoveEnetClient net(c);
        SanityVideo video(c);
        ev::EROS eros;
        eros.init(c.resolution.width, c.resolution.height, 7, 0.3);
        hpecore::SAE sae;
        sae.init(c.resolution.width, c.resolution.height);
        hpecore::multiJointLatComp state;
        if (!state.initialise({c.procU,c.measUD,c.measUV,c.latencyComp ? 1.0:0.0}))
            throw std::runtime_error("KF initialization failed");
        hpecore::pwtripletvelocity of;

        hpecore::stampedPose detection{};
        hpecore::skeleton13 flowVel{};
        hpecore::skeleton13 fusedPose{};
        hpecore::skeleton13 integratedPose{};
        hpecore::skeleton13 roiCentres{};
        bool pendingDetection = false;
        bool hasRaw = false;
        bool hasFlow = false;
        bool integratedReady = false;
        bool firstPacket = true;
        double firstTs = 0.0;
        double lastTs = 0.0;
        double lastDetectionTs = -1e20;
        double lastFlowTs = 0.0;
        double nextNet = 0.0;
        double nextFlow = 0.0;
        double nextPacket = 0.0;
        std::size_t frameIndex = 0;
        std::size_t packetCount = 0;
        std::size_t detections = 0;
        std::size_t flowUpdates = 0;

        const auto outputFrame = [&](double frameTime) {
            video.emit(frameTime, firstTs, eros.getSurface(), state.poseIsInitialised(),
                hasRaw, detection, fusedPose, integratedPose, flowVel, roiCentres,
                hasFlow, lastDetectionTs);
        };
        while (!gStop && !video.stopped() && loader.incrementReadTill(nextPacket)) {
            if (loader.begin() == loader.end()) {
                nextPacket += 1e-6;
                continue;
            }
            const double ts = loader.begin().timestamp();
            if (firstPacket) {
                firstTs = ts;
                nextNet = ts;
                nextFlow = ts;
                // hpe-core does not initialize this field in the class itself;
                // make the first OF query deterministic, without changing the algorithm.
                of.prev_update_ts = ts - 1e-6;
                firstPacket = false;
            } else {
                // Emit frames STRICTLY before this packet. No look-ahead:
                // each video timestamp sees only events with timestamps <= it.
                while (firstTs + static_cast<double>(frameIndex)/c.fps < ts - kEps) {
                    outputFrame(firstTs + static_cast<double>(frameIndex)/c.fps);
                    ++frameIndex;
                    if (gStop || video.stopped()) break;
                }
                if (gStop || video.stopped()) break;
            }
            lastTs = ts;
            std::size_t validEvents = 0;
            for (auto e = loader.begin(); e != loader.end(); ++e) {
                if (e->x < 0 || e->y < 0 || e->x >= c.resolution.width || e->y >= c.resolution.height)
                    continue;
                eros.update(e->x, e->y);
                sae.update(e->x, e->y, ts);
                video.markEvent(e->x, e->y, ts);
                ++validEvents;
            }
            nextPacket = ts + 1e-6;
            ++packetCount;
            if (!validEvents) continue;

            // Match original offline runner's relative dataset-time scheduling.
            if (ts >= nextNet) {
                nextNet = ts + c.netPeriod;
                if (net.infer(eros.getSurface(), ts, detection)) {
                    hasRaw = true;
                    pendingDetection = true;
                    lastDetectionTs = ts;
                    ++detections;
                }
            }
            if (ts >= nextFlow) {
                nextFlow = ts + c.flowPeriod;
                bool consumedDetection = false;
                if (pendingDetection) {
                    if (state.poseIsInitialised()) state.updateFromPosition(detection.pose, detection.timestamp);
                    else state.set(detection.pose, detection.timestamp);
                    pendingDetection = false;
                    consumedDetection = true;
                }
                if (state.poseIsInitialised()) {
                    roiCentres = state.query();
                    flowVel = of.multi_area_velocity(sae.getSurface(), ts, roiCentres, c.roi);
                    state.setVelocity(flowVel);
                    state.updateFromVelocity(flowVel, ts);
                    fusedPose = state.query();
                    // Only the green diagnostic resets; the OF/KF estimator is unchanged.
                    // Re-anchor at the KF-corrected pose at the same flow timestep
                    // that consumes a valid MoveEnet detection.
                    if (!integratedReady || (consumedDetection && c.resetIntegratedOnDetection)) {
                        integratedPose = roiCentres;
                        integratedReady = true;
                    } else {
                        // Integrate the same OF velocities used by OFK, without KF
                        // corrections until the next reset (or until the end in
                        // continuous mode). Units: pixels per second.
                        const double dt = std::max(0.0, ts-lastFlowTs);
                        for (int i = 0; i < kJoints; ++i) {
                            integratedPose[i].u += flowVel[i].u*dt;
                            integratedPose[i].v += flowVel[i].v*dt;
                        }
                    }
                    lastFlowTs = ts;
                    hasFlow = true;
                    ++flowUpdates;
                }
            }
        }
        if (firstPacket) throw std::runtime_error("Input event log contains no packets");
        // A frame rate F yields N=ceil(T*F) frames for original duration T.
        // The last input timestamp is an end bound, not an extra video frame.
        if (!gStop && !video.stopped()) {
            const double duration = std::max(0.0,lastTs-firstTs);
            const std::size_t target = std::max<std::size_t>(1,
                static_cast<std::size_t>(std::ceil(duration*c.fps-1e-9)));
            while (frameIndex < target) {
                outputFrame(firstTs + static_cast<double>(frameIndex)/c.fps);
                ++frameIndex;
            }
        }
        std::cout << "Event packets: " << packetCount
                  << ", MoveEnet detections: " << detections
                  << ", OF updates: " << flowUpdates
                  << ", frames: " << video.frames() << std::endl;
        std::cout << std::fixed << std::setprecision(3)
                  << "Dataset packet span: " << (lastTs-firstTs) << " s; "
                  << "MP4 nominal duration: " << video.frames()/c.fps << " s\n"
                  << "Saved: " << c.outputVideo << std::endl;
        return gStop || video.stopped() ? 130 : 0;
    } catch (const std::exception &e) {
        std::cerr << "[optical_flow_sanity ERROR] " << e.what() << std::endl;
        return 1;
    }
}

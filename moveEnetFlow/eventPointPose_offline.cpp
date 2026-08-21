// EventPointPose offline DHP19 runner.
//
// The runner preserves two independent dataset-time schedules:
//   --net_period     : event batches sent to the Python PointNet sidecar
//   --output_period  : zero-order-held pose rows written to CSV
//
// Input data.log files are read with ev::offlineLoader<ev::AE>. Every event
// carries the timestamp of its YARP packet, as required by this deployment.
// The Python process owns the persistent 7500-event FIFO, RasEPC, deterministic
// 2048-point sampling, PointNet inference, and the hpe-core joint remapping.

#include <yarp/os/all.h>
#include <event-driven/core.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <chrono>

using yarp::os::Bottle;
using yarp::os::BufferedPort;
using yarp::os::Network;
using yarp::os::NetworkBase;
using yarp::os::ResourceFinder;
using yarp::os::Value;

namespace {

constexpr int kJointCount = 13;
constexpr double kTimeEpsilon = 1.0e-9;

volatile std::sig_atomic_t gStopRequested = 0;

void handleTerminationSignal(int)
{
    gStopRequested = 1;
}

bool isReadableFile(const std::string &path)
{
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    return stream.good();
}

struct EventRecord
{
    int x{0};
    int y{0};
    int polarity{0};
    double packet_timestamp{0.0};
};

struct Joint
{
    double x{0.0};
    double y{0.0};
};

struct PoseSample
{
    std::array<Joint, kJointCount> joints{};
    std::array<double, kJointCount> confidence{};
    double timestamp{0.0};
    double latency{0.0};
};

struct ServerReply
{
    bool transport_ok{false};
    bool response_received{false};
    bool has_pose{false};

    std::string status{"ERROR"};
    std::string message;

    int fifo_size{0};
    int accepted_events{0};
    int rasepc_points{0};

    // Seconds in C++; converted from server diagnostic milliseconds.
    double server_processing_time{0.0};

    PoseSample pose{};
};

std::string shellQuote(const std::string &input)
{
    std::string output = "'";
    for (char c : input) {
        if (c == '\'') {
            output += "'\\''";
        } else {
            output += c;
        }
    }
    output += "'";
    return output;
}

std::string trim(std::string value)
{
    value.erase(value.begin(), std::find_if(value.begin(), value.end(),
        [](unsigned char c) { return !std::isspace(c); }));
    value.erase(std::find_if(value.rbegin(), value.rend(),
        [](unsigned char c) { return !std::isspace(c); }).base(), value.end());
    return value;
}

int inferCameraFromPath(const std::string &path)
{
    if (path.find("ch2dvs") != std::string::npos) {
        return 2;
    }
    if (path.find("ch3dvs") != std::string::npos) {
        return 3;
    }
    return -1;
}

std::string defaultServerLog()
{
    return "/tmp/eventpointpose_server_" + std::to_string(::getpid()) + ".log";
}

void writeCsvHeader(std::ofstream &csv)
{
    csv << "timestamp,latency";
    for (int joint = 0; joint < kJointCount; ++joint) {
        csv << ",joint" << joint << "_x,joint" << joint << "_y";
    }
    csv << "\n";
    csv.flush();
}

} // namespace

class EventPointPoseClient
{
private:
    BufferedPort<Bottle> request_port;
    BufferedPort<Bottle> response_port;

    bool launched_python{false};
    bool connected{false};
    double response_timeout_s{0.0};
    int camera_id{2};

    std::string local_request_port;
    std::string local_response_port;
    std::string server_request_port;
    std::string server_response_port;
    std::string pid_file;

    Bottle *readResponse()
    {
        if (response_timeout_s <= 0.0) {
            return response_port.read(true);
        }

        const double deadline = yarp::os::Time::now() + response_timeout_s;
        while (yarp::os::Time::now() < deadline) {
            Bottle *response = response_port.read(false);
            if (response != nullptr) {
                return response;
            }
            yarp::os::Time::delay(0.001);
        }
        return nullptr;
    }

public:
    bool init(const std::string &model_path,
              const std::string &script_path,
              const std::string &device,
              int camera,
              int width,
              int height,
              int fifo_size,
              int num_points,
              int seed,
              const std::string &input_preprocessing,
              const std::string &background_domain,
              double background_dt_us,
              const std::string &hotpixel_path,
              const std::string &dump_dir,
              int dump_first_n,
              bool server_verbose,
              double startup_timeout_s,
              double response_timeout,
              const std::string &server_log,
              const std::string &configured_server_request_port,
              const std::string &configured_server_response_port)
    {
        response_timeout_s = response_timeout;
        camera_id = camera;

        const std::string pid_suffix = std::to_string(::getpid());
        local_request_port = "/EventPointPose_offline/" + pid_suffix + "/events:o";
        local_response_port = "/EventPointPose_offline/" + pid_suffix + "/sklt:i";
        server_request_port = configured_server_request_port.empty()
            ? "/EventPointPose/" + pid_suffix + "/events:i"
            : configured_server_request_port;
        server_response_port = configured_server_response_port.empty()
            ? "/EventPointPose/" + pid_suffix + "/sklt:o"
            : configured_server_response_port;
        pid_file = "/tmp/eventpointpose_offline_python_" + pid_suffix + ".pid";

        if (!Network::checkNetwork(2.0)) {
            yError() << "Could not connect to YARP. Start yarpserver first.";
            return false;
        }
        if (!script_path.empty() && !isReadableFile(script_path)) {
            yError() << "EventPointPose server script is not readable:" << script_path;
            return false;
        }
        if (!script_path.empty() && !isReadableFile(model_path)) {
            yError() << "PointNet checkpoint is not readable:" << model_path;
            return false;
        }

        if (!request_port.open(local_request_port)) {
            yError() << "Could not open request port:" << local_request_port;
            return false;
        }
        if (!response_port.open(local_response_port)) {
            yError() << "Could not open response port:" << local_response_port;
            request_port.close();
            return false;
        }

        if (!script_path.empty()) {
            std::ostringstream command;
            command << "python3 " << shellQuote(script_path)
                    << " --model " << shellQuote(model_path)
                    << " --event_port " << shellQuote(server_request_port)
                    << " --sklt_port " << shellQuote(server_response_port)
                    << " --device " << shellQuote(trim(device))
                    << " --w " << width
                    << " --h " << height
                    << " --fifo_size " << fifo_size
                    << " --num_points " << num_points
                    << " --rasepc_channels 4"
                    << " --seed " << seed
                    << " --input_preprocessing " << shellQuote(input_preprocessing)
                    << " --background_domain " << shellQuote(background_domain)
                    << " --background_dt_us " << std::setprecision(12) << background_dt_us;

            if (!hotpixel_path.empty()) {
                command << " --hotpixel_path " << shellQuote(hotpixel_path);
            }
            if (!dump_dir.empty()) {
                command << " --dump_debug_dir " << shellQuote(dump_dir)
                        << " --dump_first_n " << dump_first_n;
            }
            if (server_verbose) {
                command << " --verbose";
            }
            if (!server_log.empty()) {
                command << " > " << shellQuote(server_log) << " 2>&1";
            }
            command << " & echo $! > " << shellQuote(pid_file);

            const int launch_result = std::system(command.str().c_str());
            if (launch_result != 0) {
                yError() << "Could not launch EventPointPose sidecar with command:"
                         << command.str();
                close();
                return false;
            }
            launched_python = true;
        }

        const double startup_deadline = yarp::os::Time::now() + startup_timeout_s;
        bool ports_ready = false;
        while (yarp::os::Time::now() < startup_deadline) {
            if (NetworkBase::exists(server_request_port) &&
                NetworkBase::exists(server_response_port)) {
                ports_ready = true;
                break;
            }
            yarp::os::Time::delay(0.1);
        }

        if (!ports_ready) {
            yError() << "EventPointPose sidecar ports not found. Expected"
                     << server_request_port << "and" << server_response_port;
            if (!server_log.empty()) {
                yError() << "Server log:" << server_log;
            }
            close();
            return false;
        }

        if (!Network::connect(local_request_port, server_request_port, "fast_tcp")) {
            yError() << "Could not connect" << local_request_port
                     << "->" << server_request_port;
            close();
            return false;
        }
        if (!Network::connect(server_response_port, local_response_port, "fast_tcp")) {
            yError() << "Could not connect" << server_response_port
                     << "->" << local_response_port;
            close();
            return false;
        }

        connected = true;
        yInfo() << "EventPointPose sidecar connected.";
        yInfo() << "Request port:" << local_request_port << "->" << server_request_port;
        yInfo() << "Response port:" << server_response_port << "->" << local_response_port;
        return true;
    }

    ServerReply infer(const std::vector<EventRecord> &events,
                      double inference_timestamp,
                      int camera,
                      int request_id)
    {
        ServerReply result;
        if (!connected) {
            result.message = "YARP client is not connected.";
            return result;
        }

        Bottle &request = request_port.prepare();
        request.clear();
        request.addInt32(request_id);
        request.addFloat64(inference_timestamp);
        request.addInt32(camera);
        Bottle &payload = request.addList();
        for (const EventRecord &event : events) {
            // Python expects [x, y, packet_timestamp, polarity].
            payload.addInt32(event.x);
            payload.addInt32(event.y);
            payload.addFloat64(event.packet_timestamp);
            payload.addInt32(event.polarity);
        }

        result.pose.timestamp = inference_timestamp;
        result.pose.latency =
            std::numeric_limits<double>::quiet_NaN();

        using Clock = std::chrono::steady_clock;

        // ============================================================
        // SERVICE LATENCY START
        // The request Bottle and event payload are already prepared.
        // Bottle construction is excluded from service latency but will
        // be included in complete method latency measured by main().
        // ============================================================
        const auto request_start = Clock::now();

        auto finishLatency = [&]() {
            const auto request_end = Clock::now();

            result.pose.latency =
                std::chrono::duration<double>(
                    request_end - request_start
                ).count();
        };

        request_port.write();

        Bottle *response = readResponse();

        if (response == nullptr) {
            finishLatency();

            result.status = "TIMEOUT";
            result.message =
                "Timed out waiting for EventPointPose response.";

            return result;
        }

        // A response was actually received from the service.
        result.response_received = true;
        result.transport_ok = true;

        if (response->size() < 3) {
            result.transport_ok = false;
            result.status = "PROTOCOL_ERROR";
            result.message =
                "Response contains fewer than three elements.";

            finishLatency();
            return result;
        }

        result.status = response->get(2).asString();

        const int response_id =
            response->size() > 3
                ? response->get(3).asInt32()
                : request_id;

        if (response_id != request_id) {
            result.transport_ok = false;
            result.status = "PROTOCOL_ERROR";
            result.message =
                "Response request id does not match the outstanding request.";

            finishLatency();
            return result;
        }

        // Server diagnostics:
        // [accepted_events, fifo_size, rasepc_points, inference_ms]
        if (response->size() > 4) {

            Bottle *diagnostics =
                response->get(4).asList();

            if (diagnostics != nullptr &&
                diagnostics->size() >= 4)
            {
                result.accepted_events =
                    static_cast<int>(
                        diagnostics->get(0).asFloat64()
                    );

                result.fifo_size =
                    static_cast<int>(
                        diagnostics->get(1).asFloat64()
                    );

                result.rasepc_points =
                    static_cast<int>(
                        diagnostics->get(2).asFloat64()
                    );

                // Python sends milliseconds.
                // Internally we keep seconds.
                result.server_processing_time =
                    diagnostics->get(3).asFloat64() * 0.001;
            }
        }

        if (response->size() > 5) {
            result.message =
                response->get(5).asString();
        }

        // WARMUP / NO_UPDATE / ERROR:
        // valid service responses, but no usable PointNet pose.
        if (result.status != "OK") {
            finishLatency();
            return result;
        }

        Bottle *pose_payload =
            response->get(1).asList();

        if (pose_payload == nullptr ||
            pose_payload->size() < 39)
        {
            result.transport_ok = false;
            result.status = "PROTOCOL_ERROR";
            result.message =
                "OK response does not contain the 39-value hpe-core payload.";

            finishLatency();
            return result;
        }

        // ============================================================
        // Decode response -> usable hpe-core pose.
        // INCLUDED in service latency.
        // ============================================================
        for (int joint = 0;
            joint < kJointCount;
            ++joint)
        {
            result.pose.joints[joint].x =
                pose_payload->get(
                    2 * joint
                ).asFloat64();

            result.pose.joints[joint].y =
                pose_payload->get(
                    2 * joint + 1
                ).asFloat64();

            result.pose.confidence[joint] =
                pose_payload->get(
                    26 + joint
                ).asFloat64();
        }

        result.has_pose = true;

        // ============================================================
        // SERVICE LATENCY END:
        // C++ client now has a decoded usable pose.
        // ============================================================
        finishLatency();

        return result;
    }

    void close()
    {
        if (connected) {
            Bottle &request = request_port.prepare();
            request.clear();
            request.addInt32(-1);
            request.addFloat64(0.0);
            request.addInt32(camera_id);
            request.addList();
            request_port.write();
            yarp::os::Time::delay(0.05);
        }

        request_port.close();
        response_port.close();
        connected = false;

        if (launched_python) {
            std::ostringstream command;
            command << "if [ -f " << shellQuote(pid_file) << " ]; then "
                    << "pid=$(cat " << shellQuote(pid_file) << "); "
                    << "kill $pid 2>/dev/null || true; "
                    << "i=0; while kill -0 $pid 2>/dev/null && [ $i -lt 20 ]; do "
                    << "sleep 0.05; i=$((i+1)); done; "
                    << "kill -9 $pid 2>/dev/null || true; "
                    << "rm -f " << shellQuote(pid_file) << "; fi";
            std::system(command.str().c_str());
            launched_python = false;
        }
    }

    ~EventPointPoseClient()
    {
        close();
    }
};

int main(int argc, char *argv[])
{
    ResourceFinder rf;
    rf.setVerbose(false);
    rf.configure(argc, argv);

    std::signal(SIGINT, handleTerminationSignal);
    std::signal(SIGTERM, handleTerminationSignal);

    if (rf.check("help")) {
        std::stringstream help;
        help << "Usage: EventPointPose_offline [options]\n\n";
        help << std::left
             << std::setw(30) << "--data_file <path>" << ": DHP19 ch2dvs/ch3dvs data.log\n"
             << std::setw(30) << "--net_period <seconds>" << ": PointNet request period, default 0.05\n"
             << std::setw(30) << "--output_period <seconds>" << ": held-pose CSV period, default 0.005\n"
             << std::setw(30) << "--output_csv <path>" << ": output CSV, default /tmp/EventPointPose_output.csv\n"
             << std::setw(30) << "--no_csv" << ": disable CSV output\n"
             << std::setw(30) << "--latency_csv <path>" << ": one row per EventPointPose service request\n"
             << std::setw(30) << "--camera <2|3>" << ": camera id; default inferred from data_file\n"
             << std::setw(30) << "--model_path <path>" << ": PointNet MeanLabel checkpoint\n"
             << std::setw(30) << "--EventPointPose_script <path>" << ": Python YARP sidecar (legacy: --eventPointPose_script)\n"
             << std::setw(30) << "--device <cpu|cuda:N|N>" << ": inference device, default cuda:0\n"
             << std::setw(30) << "--input_preprocessing <mode>" << ": already_filtered (default) or raw\n"
             << std::setw(30) << "--background_domain <mode>" << ": global (default) or local; raw mode only\n"
             << std::setw(30) << "--background_dt_us <value>" << ": raw background-filter dt, default 70000\n"
             << std::setw(30) << "--hotpixel_path <path>" << ": .npy/.csv mask or containing directory; raw only\n"
             << std::setw(30) << "--seed <int>" << ": deterministic sampling seed, default 1\n"
             << std::setw(30) << "--fifo_size <int>" << ": persistent event FIFO, default 7500\n"
             << std::setw(30) << "--num_points <int>" << ": post-RasEPC sample count, default 2048\n"
             << std::setw(30) << "--response_timeout <seconds>" << ": 0 means blocking forever\n"
             << std::setw(30) << "--startup_timeout <seconds>" << ": sidecar startup timeout, default 30\n"
             << std::setw(30) << "--server_log <path>" << ": Python stdout/stderr log\n"
             << std::setw(30) << "--server_verbose" << ": print every server request\n"
             << std::setw(30) << "--dump_dir <path>" << ": optional NumPy validation dumps\n"
             << std::setw(30) << "--dump_first_n <int>" << ": successful inferences to dump, default 1\n"
             << std::setw(30) << "--server_event_port <name>" << ": override generated Python input port\n"
             << std::setw(30) << "--server_skeleton_port <name>" << ": override generated Python output port\n";
        yInfo() << help.str();
        return 0;
    }

    const std::string data_file = rf.check(
        "data_file",
        Value("/data/dhp19_testing_set_S13toS17/S13_1_1/ch2dvs/data.log")
    ).asString();
    const double net_period = rf.check("net_period", Value(0.05)).asFloat64();
    const double output_period = rf.check("output_period", Value(0.005)).asFloat64();
    const std::string output_csv = rf.check(
        "output_csv",
        rf.check("output_file", Value("/tmp/EventPointPose_output.csv"))
    ).asString();
    const std::string latency_csv_path = rf.check(
        "latency_csv",
        Value("")
    ).asString();
    const bool no_csv = rf.check("no_csv");

    int camera = rf.check("camera", Value(-1)).asInt32();
    if (camera < 0) {
        camera = inferCameraFromPath(data_file);
    }

    const int width = rf.check("w", Value(346)).asInt32();
    const int height = rf.check("h", Value(260)).asInt32();
    const int fifo_size = rf.check("fifo_size", Value(7500)).asInt32();
    const int num_points = rf.check("num_points", Value(2048)).asInt32();
    const int seed = rf.check("seed", Value(1)).asInt32();

    const std::string model_path = rf.check(
        "model_path",
        Value("/workspace/model_mounts/eventpointpose/PointNet/models/model.pth")
    ).asString();
    const std::string script_path = rf.check(
        "EventPointPose_script",
        rf.check(
            "eventPointPose_script",
            Value("/workspace/model_mounts/eventpointpose/PointNet/models/eventPointPose_yarp_server.py")
        )
    ).asString();
    const std::string device = rf.check("device", Value("cuda:0")).asString();
    const std::string input_preprocessing = rf.check(
        "input_preprocessing", Value("already_filtered")
    ).asString();
    const std::string background_domain = rf.check(
        "background_domain", Value("global")
    ).asString();
    const double background_dt_us = rf.check(
        "background_dt_us", Value(70000.0)
    ).asFloat64();
    const std::string hotpixel_path = rf.check(
        "hotpixel_path",
        Value("/workspace/moveEnetFlow/experiments/ExpA/EPP_expA_accuracy_vs_network_period/dhp19_full_test/hotpixel_masks/S13_1_1/ch2dvs")
    ).asString();

    const double startup_timeout = rf.check(
        "startup_timeout", Value(30.0)
    ).asFloat64();
    const double response_timeout = rf.check(
        "response_timeout", Value(0.0)
    ).asFloat64();
    const std::string server_log = rf.check(
        "server_log", Value(defaultServerLog())
    ).asString();
    const bool server_verbose = rf.check("server_verbose");
    const std::string dump_dir = rf.check("dump_dir", Value("")).asString();
    const int dump_first_n = rf.check("dump_first_n", Value(1)).asInt32();
    const std::string server_event_port = rf.check(
        "server_event_port", Value("")
    ).asString();
    const std::string server_skeleton_port = rf.check(
        "server_skeleton_port", Value("")
    ).asString();

    if (!std::isfinite(net_period) || !std::isfinite(output_period) ||
        net_period <= 0.0 || output_period <= 0.0) {
        yError() << "--net_period and --output_period must be finite and positive.";
        return -1;
    }
    if (camera != 2 && camera != 3) {
        yError() << "Could not infer a supported camera. Use --camera 2 or --camera 3.";
        return -1;
    }
    if (width != 346 || height != 260) {
        yError() << "The PointNet MeanLabel checkpoint requires --w 346 --h 260.";
        return -1;
    }
    if (fifo_size != 7500) {
        yError() << "This online-style deployment requires --fifo_size 7500.";
        return -1;
    }
    if (num_points != 2048) {
        yError() << "The PointNet checkpoint requires --num_points 2048.";
        return -1;
    }
    if (input_preprocessing != "already_filtered" && input_preprocessing != "raw") {
        yError() << "--input_preprocessing must be already_filtered or raw.";
        return -1;
    }
    if (background_domain != "global" && background_domain != "local") {
        yError() << "--background_domain must be global or local.";
        return -1;
    }
    if (input_preprocessing == "raw" && hotpixel_path.empty()) {
        yError() << "Raw mode requires --hotpixel_path; an empty mask file is valid.";
        return -1;
    }
    if (!std::isfinite(background_dt_us) || background_dt_us <= 0.0) {
        yError() << "--background_dt_us must be finite and positive.";
        return -1;
    }
    if (!std::isfinite(startup_timeout) || !std::isfinite(response_timeout) ||
        startup_timeout <= 0.0 || response_timeout < 0.0) {
        yError() << "Startup timeout must be positive and response timeout non-negative.";
        return -1;
    }

    yInfo() << "Loading event log:" << data_file;
    ev::offlineLoader<ev::AE> event_loader;
    if (!event_loader.load(data_file)) {
        yError() << "Could not load event data file:" << data_file;
        return -1;
    }
    yInfo() << event_loader.getinfo();
    yInfo() << "Camera:" << camera;
    yInfo() << "PointNet period:" << net_period << "s (" << 1.0 / net_period << "Hz)";
    yInfo() << "CSV period:" << output_period << "s (" << 1.0 / output_period << "Hz)";
    yInfo() << "Input preprocessing:" << input_preprocessing;
    if (input_preprocessing == "already_filtered") {
        yInfo() << "Hot-pixel/background/IR filters will be skipped by the server.";
    } else {
        yInfo() << "Raw filter background domain:" << background_domain;
    }

    std::ofstream csv;
    std::ofstream latency_file;
    std::vector<std::string> latency_buffer;

    if (!no_csv) {
            csv.open(output_csv);
            if (!csv.is_open()) {
                yError() << "Could not open CSV file:" << output_csv;
                return -1;
            }
            writeCsvHeader(csv);
            yInfo() << "CSV output:" << output_csv;
        }
        if (!latency_csv_path.empty()) {

        latency_file.open(latency_csv_path);

        if (!latency_file.is_open()) {
            yError() << "Could not open latency CSV:"
                    << latency_csv_path;
            return -1;
        }

        latency_file
            << "request_id,"
            << "dataset_timestamp,"
            << "scheduled_timestamp,"
            << "service_latency_ms,"
            << "method_latency_ms,"
            << "response_received,"
            << "valid_pose,"
            << "status,"
            << "input_events,"
            << "accepted_events,"
            << "fifo_size,"
            << "rasepc_points,"
            << "server_processing_ms\n";

        yInfo()
            << "Per-request latency logging enabled ->"
            << latency_csv_path;
    }

    EventPointPoseClient client;
    if (!client.init(
            model_path,
            script_path,
            device,
            camera,
            width,
            height,
            fifo_size,
            num_points,
            seed,
            input_preprocessing,
            background_domain,
            background_dt_us,
            hotpixel_path,
            dump_dir,
            dump_first_n,
            server_verbose,
            startup_timeout,
            response_timeout,
            server_log,
            server_event_port,
            server_skeleton_port)) {
        if (csv.is_open()) {
            csv.close();
        }
        return -1;
    }

    std::vector<EventRecord> pending_batch;
    pending_batch.reserve(16384);

    PoseSample held_pose;
    bool pose_initialised = false;
    bool schedules_initialised = false;
    double next_inference_ts = 0.0;
    double next_output_ts = 0.0;

    std::size_t packet_groups = 0;
    std::size_t total_events = 0;
    std::size_t request_count = 0;
    std::size_t valid_inference_count = 0;
    std::size_t warmup_count = 0;
    std::size_t no_update_count = 0;
    std::size_t server_error_count = 0;
    std::size_t csv_rows = 0;
    double first_request_ts = -1.0;
    double last_request_ts = -1.0;

    double first_valid_inference_ts = -1.0;
    double last_valid_inference_ts = -1.0;

    auto emitOutput = [&](double output_timestamp) {
        if (!pose_initialised || !csv.is_open()) {
            return;
        }
        csv << std::fixed << std::setprecision(6)
            << output_timestamp << "," << held_pose.latency;
        for (int joint = 0; joint < kJointCount; ++joint) {
            csv << "," << held_pose.joints[joint].x
                << "," << held_pose.joints[joint].y;
        }
        csv << "\n";
        ++csv_rows;
    };

    int request_id = 0;
    bool fatal_error = false;

    auto processPacketGroup = [&](double current_timestamp,
                                  const std::vector<EventRecord> &current_events) -> bool {
        if (gStopRequested != 0) {
            return true;
        }
        if (current_events.empty() || !std::isfinite(current_timestamp)) {
            return true;
        }

        ++packet_groups;
        total_events += current_events.size();

        if (!schedules_initialised) {
            next_inference_ts = current_timestamp;
            next_output_ts = current_timestamp;
            schedules_initialised = true;
        }

        // Like YoloPose_offline: outputs strictly before the current packet use
        // only the pose that was available before this packet/inference.
        while (next_output_ts < current_timestamp - kTimeEpsilon) {
            emitOutput(next_output_ts);
            next_output_ts += output_period;
        }

        pending_batch.insert(
            pending_batch.end(), current_events.begin(), current_events.end()
        );

        // At most one request is executed for each available packet timestamp.
        // Missed absolute deadlines are skipped by advancing the schedule after
        // the request, exactly as in the agreed YOLO-style offline semantics.
        if (current_timestamp + kTimeEpsilon >= next_inference_ts) {

            // Save scheduling information BEFORE advancing next_inference_ts.
            const double scheduled_timestamp = next_inference_ts;
            const int this_request_id = request_id++;
            const std::size_t input_event_count = pending_batch.size();

            if (request_count == 0) {
                first_request_ts = current_timestamp;
            }

            last_request_ts = current_timestamp;
            ++request_count;

            // Complete-method latency starts with the pending event batch
            // already available, immediately before EventPointPose processing.
            const auto method_start =
                std::chrono::steady_clock::now();

            ServerReply reply = client.infer(
                pending_batch,
                current_timestamp,
                camera,
                this_request_id
            );

            const bool valid_pose =
                reply.transport_ok &&
                reply.response_received &&
                reply.status == "OK" &&
                reply.has_pose;

            double method_latency_s =
                std::numeric_limits<double>::quiet_NaN();

            // A complete method latency exists only when this request
            // actually produces a new usable pose.
            if (valid_pose) {

                held_pose = reply.pose;
                pose_initialised = true;

                const auto method_end =
                    std::chrono::steady_clock::now();

                method_latency_s =
                    std::chrono::duration<double>(
                        method_end - method_start
                    ).count();
            }

            // ============================================================
            // ONE latency row per real service request.
            // ============================================================
            if (latency_file.is_open()) {

                std::ostringstream row;

                row << this_request_id
                    << ","
                    << std::fixed
                    << std::setprecision(9)
                    << current_timestamp
                    << ","
                    << scheduled_timestamp
                    << ",";

                if (std::isfinite(reply.pose.latency)) {
                    row << std::setprecision(6)
                        << reply.pose.latency * 1000.0;
                } else {
                    row << "nan";
                }

                row << ",";

                // Complete EventPointPose method latency
                if (std::isfinite(method_latency_s)) {
                    row << std::setprecision(6)
                        << method_latency_s * 1000.0;
                } else {
                    row << "nan";
                }

                row << ","
                    << (reply.response_received ? 1 : 0)
                    << ","
                    << (valid_pose ? 1 : 0)
                    << ","
                    << reply.status
                    << ","
                    << input_event_count
                    << ","
                    << reply.accepted_events
                    << ","
                    << reply.fifo_size
                    << ","
                    << reply.rasepc_points
                    << ",";

                if (reply.server_processing_time > 0.0) {
                    row << std::setprecision(6)
                        << reply.server_processing_time * 1000.0;
                } else {
                    row << "0.000000";
                }

                latency_buffer.push_back(row.str());
            }

            // Log the failed request before aborting.
            if (!reply.transport_ok) {
                yError()
                    << "EventPointPose request failed:"
                    << reply.status
                    << reply.message;

                return false;
            }

            // The server consumed the batch for every valid protocol response:
            // OK, WARMUP, NO_UPDATE, ERROR.
            pending_batch.clear();

            if (valid_pose) {

                if (valid_inference_count == 0) {
                    first_valid_inference_ts = current_timestamp;
                }

                last_valid_inference_ts = current_timestamp;
                ++valid_inference_count;

            }
            else if (reply.status == "WARMUP") {

                ++warmup_count;

                yInfo()
                    << "EventPointPose warm-up: FIFO"
                    << reply.fifo_size
                    << "/"
                    << fifo_size;

            }
            else if (reply.status == "NO_UPDATE") {

                ++no_update_count;

            }
            else if (reply.status == "ERROR") {

                ++server_error_count;

                yWarning()
                    << "EventPointPose server returned ERROR:"
                    << reply.message;

            }
            else {

                yWarning()
                    << "EventPointPose returned status"
                    << reply.status
                    << reply.message;
            }

            // Keep the existing absolute YOLO-style schedule unchanged.
            do {
                next_inference_ts += net_period;
            }
            while (
                next_inference_ts <=
                current_timestamp + kTimeEpsilon
            );
        }

        // Outputs coincident with the packet timestamp are generated after the
        // possible inference, so a new prediction is valid from its own time.
        while (next_output_ts <= current_timestamp + kTimeEpsilon) {
            emitOutput(next_output_ts);
            next_output_ts += output_period;
        }

        return true;
    };

    // Load one iterator window spanning the complete file. The iterator keeps
    // the packet envelope timestamp and updates it when crossing packet
    // boundaries, so this avoids guessing the next packet time or stepping in
    // artificial microsecond increments.
    if (!event_loader.incrementReadTill(std::numeric_limits<double>::max())) {
        yError() << "Could not expose events from the loaded data.log.";
        fatal_error = true;
    } else if (event_loader.begin() == event_loader.end()) {
        yError() << "The loaded data.log contains no events.";
        fatal_error = true;
    } else {
        std::vector<EventRecord> current_events;
        current_events.reserve(4096);
        double current_timestamp = std::numeric_limits<double>::quiet_NaN();

        for (ev::offlineLoader<ev::AE>::iterator event = event_loader.begin();
             event != event_loader.end() && gStopRequested == 0;
             event++) {
            const double packet_timestamp = event.timestamp();

            if (!std::isfinite(current_timestamp)) {
                current_timestamp = packet_timestamp;
            } else if (packet_timestamp != current_timestamp) {
                if (!processPacketGroup(current_timestamp, current_events)) {
                    fatal_error = true;
                    break;
                }
                current_events.clear();
                current_timestamp = packet_timestamp;
            }

            EventRecord record;
            record.x = static_cast<int>(event->x);
            record.y = static_cast<int>(event->y);
            record.polarity = static_cast<int>(event->p);
            record.packet_timestamp = packet_timestamp;
            current_events.push_back(record);
        }

        if (!fatal_error && gStopRequested == 0 && !current_events.empty()) {
            if (!processPacketGroup(current_timestamp, current_events)) {
                fatal_error = true;
            }
        }
    }

    if (csv.is_open()) {
        csv.flush();
        csv.close();
    }

    if (latency_file.is_open()) {

        for (const auto &line : latency_buffer) {
            latency_file << line << "\n";
        }

        latency_file.close();

        yInfo()
            << "Latency rows written:"
            << latency_buffer.size()
            << "->"
            << latency_csv_path;
    }

    client.close();

    yInfo() << "Packet groups processed:" << packet_groups;
    yInfo() << "Events read:" << total_events;
    yInfo() << "PointNet requests:" << request_count;
    yInfo() << "Valid PointNet inferences:" << valid_inference_count;
    yInfo() << "Warm-up responses:" << warmup_count;
    yInfo() << "NO_UPDATE responses:" << no_update_count;
    yInfo() << "Server ERROR responses:" << server_error_count;
    yInfo() << "CSV rows written:" << csv_rows;
    yInfo() << "Python server log:" << server_log;

  
    yInfo()
        << "Requested service rate:"
        << 1.0 / net_period
        << "Hz";

    if (request_count > 1 &&
        last_request_ts > first_request_ts)
    {
        const double observed_request_rate =
            static_cast<double>(request_count - 1) /
            (last_request_ts - first_request_ts);

        yInfo()
            << "Observed service request rate:"
            << observed_request_rate
            << "Hz";
    }

    if (valid_inference_count > 1 &&
        last_valid_inference_ts > first_valid_inference_ts)
    {
        const double observed_valid_rate =
            static_cast<double>(valid_inference_count - 1) /
            (
                last_valid_inference_ts -
                first_valid_inference_ts
            );

        yInfo()
            << "Observed valid PointNet inference rate:"
            << observed_valid_rate
            << "Hz";
    }

    if (gStopRequested != 0) {
        yInfo() << "Termination signal received; EventPointPose replay stopped.";
        return 130;
    }
    return fatal_error ? -1 : 0;
}

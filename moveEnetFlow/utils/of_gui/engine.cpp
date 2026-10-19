/*
 * Optical-flow live web viewer engine.
 * Independent of moveEnetOFK_offline.cpp and the original experimental CMake.
 * Protocol: one JSON object per line on stdout, prefixed by META/FRAME/PROGRESS/DONE/ERROR.
 * Control: PAUSE 0|1, SPEED multiplier, QUIT (one per stdin line).
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

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using yarp::os::Network;
using yarp::os::NetworkBase;
using yarp::os::ResourceFinder;
using yarp::os::Value;
using yarp::os::Bottle;
using yarp::os::BufferedPort;
using yarp::sig::ImageOf;
using yarp::sig::PixelMono;
namespace {
constexpr int NJ=13;
constexpr double EPS=1e-8;
std::atomic<bool> stopRequested{false};
std::atomic<bool> paused{false};
std::atomic<double> speedFactor{1.0};
std::mutex waitMutex;
std::condition_variable waitCv;
void sigHandler(int) {stopRequested=true; waitCv.notify_all();}

struct Config {
    std::string input;
    std::string python="python3";
    std::string script="/usr/local/src/hpe-core/example/movenet/movenet_online.py";
    std::string checkpoint;
    std::string device="cuda:0";
    bool dhp19=false;
    int width=640, height=480;
    double netPeriod=0.2, flowPeriod=0.005, frameFps=30;
    double gainA=1.4, gainB=1.0;
    double pu=0.77, muD=0.06, muV=0.97;
    int roi=20;
    bool useLc=true, compare=true, resetOnDetection=true;
    double startAt=0, durationHint=-1;
    int eventWindowMs=30;
    std::array<bool,NJ> ofMask{};
};
std::array<bool,NJ> parseMask(std::string str) {
    std::array<bool,NJ> mask{};
    if(str=="all") { mask.fill(true); return mask; }
    if(str=="none" || str.empty()) return mask;
    std::stringstream ss(str);
    std::string tok;
    while(std::getline(ss,tok,',')) {
        const int n=std::stoi(tok);
        if(n<0 || n>=NJ) throw std::runtime_error("Invalid OF joint index");
        mask[n]=true;
    }
    return mask;
}
Config parseConfig(int argc,char**argv) {
    ResourceFinder rf; rf.setVerbose(false); rf.configure(argc,argv);
    Config c;
    c.input=rf.check("data_file",Value("")).asString();
    c.dhp19=rf.check("dhp19");
    c.width=c.dhp19 ? 346:rf.check("w",Value(640)).asInt32();
    c.height=c.dhp19 ? 260:rf.check("h",Value(480)).asInt32();
    c.netPeriod=rf.check("net_period",Value(.2)).asFloat64();
    c.flowPeriod=rf.check("flow_period",Value(.005)).asFloat64();
    c.frameFps=rf.check("fps",Value(30)).asFloat64();
    c.gainA=rf.check("gain_a",Value(1.4)).asFloat64();
    c.gainB=rf.check("gain_b",Value(1.0)).asFloat64();
    c.roi=rf.check("roi",Value(20)).asInt32();
    c.pu=rf.check("pu",Value(c.dhp19?.36:.77)).asFloat64();
    c.muD=rf.check("muD",Value(c.dhp19?.01:.06)).asFloat64();
    c.muV=rf.check("muV",Value(c.dhp19?.01:.97)).asFloat64();
    c.useLc=rf.check("use_lc",Value(true)).asBool();
    c.compare=rf.check("compare",Value(true)).asBool();
    c.resetOnDetection=rf.check("reset_on_detection",Value(true)).asBool();
    c.startAt=rf.check("start_at",Value(0.0)).asFloat64();
    c.durationHint=rf.check("duration_hint",Value(-1.0)).asFloat64();
    c.eventWindowMs=rf.check("event_window_ms",Value(30)).asInt32();
    c.ofMask=parseMask(rf.check("of_joints",Value("all")).asString());
    c.python=rf.check("python",Value("python3")).asString();
    c.device=rf.check("device",Value("cuda:0")).asString();
    c.script=rf.check("movenet_script",Value(c.script)).asString();
    c.checkpoint=rf.check("checkpoint_path",Value(c.dhp19?
        "/usr/local/src/hpe-core/example/movenet/models/dhp19_allcams_e33_valacc0.87996.pth":
        "/usr/local/src/hpe-core/example/movenet/models/e97_valacc0.81209.pth")).asString();
    if(c.input.empty() || !std::filesystem::is_regular_file(c.input))
        throw std::runtime_error("data.log not found: "+c.input);
    if(!std::filesystem::is_regular_file(c.script))
        throw std::runtime_error("MoveEnet Python sidecar not found: "+c.script);
    if(!std::filesystem::is_regular_file(c.checkpoint))
        throw std::runtime_error("MoveEnet checkpoint not found: "+c.checkpoint);
    if(c.roi<3 || c.roi>120 || c.width<5 || c.height<5 || c.frameFps<1 || c.frameFps>60 ||
       c.netPeriod<=0 || c.flowPeriod<=0 || c.pu<=0 || c.muD<=0 || c.muV<=0 ||
       c.gainA<0 || c.gainB<0 || c.startAt<0 || c.eventWindowMs<=0)
        throw std::runtime_error("Invalid GUI engine parameter");
    return c;
}

void controlReader() {
    std::string command;
    while(!stopRequested && std::getline(std::cin,command)) {
        std::istringstream in(command); std::string op; in>>op;
        if(op=="PAUSE") {int v=0; in>>v; paused.store(v!=0); waitCv.notify_all();}
        else if(op=="SPEED") {double v=1; in>>v;
            if(std::isfinite(v)&&v>=0.1&&v<=8) speedFactor.store(v);
            waitCv.notify_all();}
        else if(op=="QUIT") {stopRequested=true; paused=false; waitCv.notify_all(); break;}
    }
}

std::string base64(const std::vector<unsigned char>&in) {
    static constexpr char T[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; out.reserve((in.size()+2)/3*4);
    std::size_t i=0;
    while(i+2<in.size()) {
        unsigned int n=(unsigned(in[i])<<16)|(unsigned(in[i+1])<<8)|unsigned(in[i+2]);
        out.push_back(T[(n>>18)&63]);out.push_back(T[(n>>12)&63]);
        out.push_back(T[(n>>6)&63]);out.push_back(T[n&63]); i+=3;
    }
    if(i<in.size()) {
        unsigned int n=unsigned(in[i])<<16;
        if(i+1<in.size()) n|=unsigned(in[i+1])<<8;
        out.push_back(T[(n>>18)&63]);out.push_back(T[(n>>12)&63]);
        out.push_back(i+1<in.size()?T[(n>>6)&63]:'=');out.push_back('=');
    }
    return out;
}
std::string jpeg64(const cv::Mat &m) {
    std::vector<unsigned char> binary;
    if(!cv::imencode(".jpg",m,binary,{cv::IMWRITE_JPEG_QUALITY,65}))
        throw std::runtime_error("cv::imencode failed");
    return base64(binary);
}
void printPose(std::ostringstream &out,const hpecore::skeleton13 &pose) {
    out<<'[';
    for(int i=0;i<NJ;++i){if(i)out<<',';
        const auto&j=pose[i];
        if(std::isfinite(j.u)&&std::isfinite(j.v))out<<'['<<j.u<<','<<j.v<<']';
        else out<<"[0,0]";
    }
    out<<']';
}
class MoveEnetClient {
    BufferedPort<ImageOf<PixelMono>> output;
    BufferedPort<Bottle> input;
    pid_t child=-1;
    int width=0,height=0,transportWidth=0;
    void cleanup() noexcept {
        output.interrupt();input.interrupt();output.close();input.close();
        if(child>0){
            ::kill(child,SIGTERM);int st=0;
            for(int i=0;i<30;i++){
                if(::waitpid(child,&st,WNOHANG)==child){child=-1;break;}
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if(child>0){::kill(child,SIGKILL);::waitpid(child,&st,0);child=-1;}
        }
    }
public:
    MoveEnetClient(const Config&c):width(c.width),height(c.height),transportWidth(c.dhp19?352:c.width){
        try {
            if(!Network::checkNetwork(2.0)) throw std::runtime_error("YARP server unavailable; run yarpserver &");
            if(NetworkBase::exists("/movenet/img:i") || NetworkBase::exists("/movenet/sklt:o"))
                throw std::runtime_error("Existing MoveEnet detected on /movenet/*: close it before opening GUI (no other process was modified)");
            const std::string suf=std::to_string(::getpid());
            if(!output.open("/of_gui/eros:o_"+suf) || !input.open("/of_gui/sklt:i_"+suf))
                throw std::runtime_error("Cannot open private YARP ports");
            child=::fork(); if(child<0)throw std::runtime_error("fork failed");
            if(child==0){
                const std::string log="/tmp/of_gui_movenet_"+suf+".log";
                int fd=::open(log.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0644);
                if(fd>=0){::dup2(fd,STDOUT_FILENO);::dup2(fd,STDERR_FILENO);::close(fd);}
                std::vector<std::string> args={c.python,c.script,"--checkpoint_path",c.checkpoint,
                    "--w",std::to_string(transportWidth),"--h",std::to_string(height)};
                if(c.device!="cpu"){
                    std::string ix=c.device;
                    if(ix.rfind("cuda:",0)==0)ix=ix.substr(5);
                    if(ix.empty()||ix.find_first_not_of("0123456789")!=std::string::npos)_exit(127);
                    args.push_back("--gpu");args.push_back("--GPU_ID");args.push_back(ix);
                }
                std::vector<char*>argv;
                for(auto&s:args)argv.push_back(const_cast<char*>(s.c_str()));
                argv.push_back(nullptr);::execvp(argv[0],argv.data());_exit(127);
            }
            bool ready=false;
            for(int i=0;i<600 && !stopRequested;i++){
                if(NetworkBase::exists("/movenet/img:i") && NetworkBase::exists("/movenet/sklt:o")){
                    ready=true;break;
                }
                int status=0;
                if(::waitpid(child,&status,WNOHANG)==child){child=-1;break;}
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if(!ready)throw std::runtime_error("MoveEnet failed to start (inspect /tmp/of_gui_movenet_<pid>.log)");
            if(!Network::connect(output.getName(),"/movenet/img:i","fast_tcp") ||
               !Network::connect("/movenet/sklt:o",input.getName(),"fast_tcp"))
                throw std::runtime_error("YARP connection failed");
        } catch(...) {cleanup();throw;}
    }
    ~MoveEnetClient(){cleanup();}
    bool infer(const cv::Mat&eros,double t,hpecore::stampedPose &pose) {
        cv::Mat img;eros.convertTo(img,CV_8U);
        cv::GaussianBlur(img,img,{5,5},0);
        if(img.cols!=transportWidth || img.rows!=height){
            cv::Mat padded(height,transportWidth,CV_8UC1,cv::Scalar(0));
            if(img.cols<=transportWidth && img.rows<=height)
                img.copyTo(padded(cv::Rect(0,0,img.cols,img.rows)));
            else cv::resize(img,padded,padded.size(),0,0,cv::INTER_AREA);
            img=padded;
        }
        output.prepare().copy(yarp::cv::fromCvMat<PixelMono>(img));
        const auto start=std::chrono::steady_clock::now();
        output.write();
        Bottle*reply=input.read(true);
        if(!reply)return false;
        pose.pose=hpecore::extractSkeletonFromYARP<Bottle>(*reply);
        pose.conf=hpecore::extractConfidenceFromYARP<Bottle>(*reply);
        pose.timestamp=t;
        pose.delay=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        for(auto&j:pose.pose){j.u=std::clamp(j.u,0.0f,float(width-1));j.v=std::clamp(j.v,0.0f,float(height-1));}
        return hpecore::poseNonZero(pose.pose);
    }
};
struct Lane {
    hpecore::multiJointLatComp kf;
    hpecore::pwtripletvelocity of;
    hpecore::skeleton13 fused{}, integrated{}, roiCenters{}, velocities{};
    bool integratedReady=false;
    double lastFlowTs=0, gain=1.4;
    bool ready=false;
    Lane(const Config&c,double g):gain(g) {
        if(!kf.initialise({c.pu,c.muD,c.muV,c.useLc?1.0:0.0}))
            throw std::runtime_error("KF initialization failed");
    }
    void setFirstTs(double ts){of.prev_update_ts=ts-1e-6;}
    void detect(const hpecore::stampedPose&pose){
        if(kf.poseIsInitialised())kf.updateFromPosition(pose.pose,pose.timestamp);
        else kf.set(pose.pose,pose.timestamp);
        ready=true;
    }
    void flow(const cv::Mat &sae,double ts,const Config&c,bool sawDetection){
        if(!ready)return;
        roiCenters=kf.query();
        velocities.fill({0,0});
        std::vector<hpecore::joint> selectedCenters;
        selectedCenters.reserve(NJ);
        for(int i=0;i<NJ;++i)if(c.ofMask[i])selectedCenters.push_back(roiCenters[i]);
        if(!selectedCenters.empty()){
            // vector<joint> overload DOES NOT have hard-coded *1.4;
            // this makes gain explicit and avoids modifying hpe-core.
            auto raw=of.multi_area_velocity(sae,ts,selectedCenters,c.roi);
            std::size_t k=0;
            for(int i=0;i<NJ;++i)if(c.ofMask[i]){
                velocities[i]=raw[k++]*gain;
            }
        } else of.prev_update_ts=ts;
        kf.setVelocity(velocities);
        kf.updateFromVelocity(velocities,ts);
        fused=kf.query();
        if(!integratedReady || (sawDetection && c.resetOnDetection)){
            integrated=roiCenters;integratedReady=true;
        } else {
            double dt=std::max(0.0,ts-lastFlowTs);
            for(int i=0;i<NJ;++i){
                integrated[i].u+=velocities[i].u*dt;
                integrated[i].v+=velocities[i].v*dt;
            }
        }
        lastFlowTs=ts;
    }
};

void waitForPlayback(double frameTime,double firstTs,double startAt) {
    static double playbackTime=startAt;
    static auto wallAnchor=std::chrono::steady_clock::now();
    static double speedAnchor=1.0;
    if(frameTime<startAt-EPS)return;
    // Catch up if playback becomes slower than the requested rate;
    // do not drop algorithm updates. Speed changes reset the wall clock anchor.
    if(std::abs(speedFactor.load()-speedAnchor)>EPS) {
        wallAnchor=std::chrono::steady_clock::now(); playbackTime=frameTime; speedAnchor=speedFactor.load();
    }
    while(!stopRequested) {
        std::unique_lock<std::mutex> lk(waitMutex);
        if(paused.load()) {
            waitCv.wait_for(lk,std::chrono::milliseconds(100),[]{return stopRequested.load()||!paused.load();});
            wallAnchor=std::chrono::steady_clock::now();playbackTime=frameTime;
            continue;
        }
        if(std::abs(speedAnchor-speedFactor.load())>EPS) {
            speedAnchor=speedFactor.load();wallAnchor=std::chrono::steady_clock::now();playbackTime=frameTime;
        }
        double delay=(frameTime-playbackTime)/speedAnchor;
        auto target=wallAnchor+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(delay));
        if(std::chrono::steady_clock::now()>=target)break;
        waitCv.wait_until(lk,target);
    }
}
void printFrame(double ts,double firstTs,double duration,const Config &c,
                const cv::Mat&erosSurface,const cv::Mat&lastEvents,const cv::Mat&sae,
                const hpecore::stampedPose&det,bool hasDetection,double detTs,
                const Lane &a,const Lane &b) {
    const cv::Size size(c.width,c.height);
    cv::Mat eros8; erosSurface.convertTo(eros8,CV_8U);
    cv::normalize(eros8,eros8,0,255,cv::NORM_MINMAX);
    cv::Mat recent(size,CV_8UC1,cv::Scalar(0));
    const double cutoff=ts-c.eventWindowMs*.001;
    for(int y=0;y<size.height;y++){
        auto*trow=lastEvents.ptr<double>(y);
        auto*rrow=recent.ptr<unsigned char>(y);
        for(int x=0;x<size.width;x++)if(trow[x]>=cutoff && trow[x]<=ts+EPS)rrow[x]=235;
    }
    cv::Mat saeVis(size,CV_8UC1,cv::Scalar(0));
    if(!sae.empty()) {
        const double period=std::max(.005,c.eventWindowMs*.001);
        for(int y=0;y<size.height;y++){
            const auto *src=sae.ptr<double>(y);
            auto *dst=saeVis.ptr<unsigned char>(y);
            for(int x=0;x<size.width;x++){
                if(src[x]<=ts && src[x]>ts-period)
                    dst[x]=static_cast<unsigned char>(255.0*(src[x]-(ts-period))/period);
            }
        }
    }
    std::ostringstream out;out<<std::fixed<<std::setprecision(5);
    out<<"{\"type\":\"frame\",\"time\":"<<std::max(0.0,ts-firstTs)
       <<",\"duration\":"<<duration<<",\"width\":"<<c.width<<",\"height\":"<<c.height
       <<",\"eros\":\""<<jpeg64(eros8)<<"\",\"events\":\""<<jpeg64(recent)
       <<"\",\"sae\":\""<<jpeg64(saeVis)<<"\",\"detected\":"<<(hasDetection?"true":"false")
       <<",\"detection_age\":"<<(hasDetection?std::max(0.0,ts-detTs):0)
       <<",\"roi\":"<<c.roi<<",\"gain_a\":"<<c.gainA<<",\"gain_b\":"<<c.gainB
       <<",\"compare\":"<<(c.compare?"true":"false")
       <<",\"ready_a\":"<<(a.ready?"true":"false")<<",\"ready_b\":"<<(b.ready?"true":"false");
    out<<",\"raw\":";printPose(out,det.pose);
    out<<",\"a\":{\"fused\":";printPose(out,a.fused);
    out<<",\"integrated\":";printPose(out,a.integrated);
    out<<",\"centres\":";printPose(out,a.roiCenters);
    out<<",\"velocity\":";printPose(out,a.velocities);
    out<<"},\"b\":{\"fused\":";printPose(out,b.fused);
    out<<",\"integrated\":";printPose(out,b.integrated);
    out<<",\"centres\":";printPose(out,b.roiCenters);
    out<<",\"velocity\":";printPose(out,b.velocities);
    out<<"}}";
    std::cout<<"FRAME "<<out.str()<<'\n'<<std::flush;
}
void printProgress(double elapsed) {
    std::cout<<"PROGRESS {\"time\":"<<std::fixed<<std::setprecision(3)<<elapsed<<"}"<<'\n'<<std::flush;
}
}
int main(int argc,char**argv) {
    std::signal(SIGINT,sigHandler);
    std::signal(SIGTERM,sigHandler);
    try {
        const Config c=parseConfig(argc,argv);
        speedFactor.store(1.0);
        ev::offlineLoader<ev::AE> replay;
        if(!replay.load(c.input))throw std::runtime_error("Failed to load data.log");
        double firstTs=0,lastTs=0;
        bool found=false;
        if(c.durationHint<0) {
            // Fast metadata scan: no neural inference or surface reconstruction yet.
            ev::offlineLoader<ev::AE> probe;
            if(!probe.load(c.input))throw std::runtime_error("Failed to scan data.log");
            double nxt=0;
            while(!stopRequested && probe.incrementReadTill(nxt)){
                if(probe.begin()==probe.end()){nxt+=1e-6;continue;}
                double ts=probe.begin().timestamp();
                if(!found){firstTs=ts;found=true;}
                lastTs=ts;nxt=ts+1e-6;
            }
            if(!found)throw std::runtime_error("Empty event recording");
        }
        double duration=c.durationHint>=0?c.durationHint:lastTs-firstTs;
        std::cout<<"META {\"duration\":"<<std::fixed<<std::setprecision(6)<<duration
                 <<",\"width\":"<<c.width<<",\"height\":"<<c.height<<"}"<<'\n'<<std::flush;
        if(stopRequested)return 0;
        ev::EROS eros;eros.init(c.width,c.height,7,0.3);
        hpecore::SAE sae;sae.init(c.width,c.height);
        Lane a(c,c.gainA),b(c,c.gainB);
        cv::Mat latestEvents(c.height,c.width,CV_64FC1,cv::Scalar(-1e30));
        hpecore::stampedPose detection{};
        bool pending=false,hasDetection=false;
        double detTs=-1e30,netNext=0,flowNext=0,packetNext=0,firstPacketTs=0;
        bool hasFirstPacket=false;
        std::size_t frameIndex=0,packets=0;
        double lastProgress=-1;
        // The owned MoveEnet is shared by both independent OF/KF branches.
        MoveEnetClient net(c);
        // Fork/exec the MoveEnet sidecar BEFORE starting the command reader.
        // The reader is detached because the process terminates on engine exit.
        std::thread(controlReader).detach();
        bool firstOutput=true;
        auto emit=[&](double ts){
            double rel=ts-firstPacketTs;
            if(rel+EPS<c.startAt)return;
            // Render the first frame even when the viewer is paused after seek.
            // Python sends PAUSE 1 immediately after receiving this first frame.
            if(!firstOutput)waitForPlayback(rel,0,c.startAt);
            if(stopRequested)return;
            printFrame(ts,firstPacketTs,duration,c,eros.getSurface(),latestEvents,
                       sae.getSurface(),detection,hasDetection,detTs,a,b);
            firstOutput=false;
        };
        // Use each event packet exactly once, and render scheduled frames using
        // the most recent (non-future) state. All inference remains dataset-time.
        while(!stopRequested && replay.incrementReadTill(packetNext)) {
            if(replay.begin()==replay.end()){packetNext+=1e-6;continue;}
            double ts=replay.begin().timestamp();
            if(!hasFirstPacket){
                firstPacketTs=ts;netNext=ts;flowNext=ts;
                a.setFirstTs(ts);b.setFirstTs(ts);hasFirstPacket=true;
            } else {
                while(!stopRequested && firstPacketTs+double(frameIndex)/c.frameFps<ts-EPS){
                    double frameTs=firstPacketTs+double(frameIndex)/c.frameFps;
                    emit(frameTs);frameIndex++;
                }
            }
            if(stopRequested)break;
            packetNext=ts+1e-6;packets++;
            bool hasEvents=false;
            for(auto e=replay.begin();e!=replay.end();++e){
                if(e->x>=c.width||e->y>=c.height||e->x<0||e->y<0)continue;
                eros.update(e->x,e->y);
                sae.update(e->x,e->y,ts);
                latestEvents.at<double>(e->y,e->x)=ts;
                hasEvents=true;
            }
            if(!hasEvents)continue;
            if(ts>=netNext-EPS){
                netNext=ts+c.netPeriod;
                if(net.infer(eros.getSurface(),ts,detection)){
                    hasDetection=true;pending=true;detTs=ts;
                }
            }
            if(ts>=flowNext-EPS){
                flowNext=ts+c.flowPeriod;
                const bool correction=pending;
                if(correction){a.detect(detection);if(c.compare)b.detect(detection);pending=false;}
                if(a.ready)a.flow(sae.getSurface(),ts,c,correction);
                if(c.compare && b.ready)b.flow(sae.getSurface(),ts,c,correction);
            }
            const double rel=ts-firstPacketTs;
            if(rel+EPS<c.startAt && (rel-lastProgress)>.5){
                printProgress(rel);lastProgress=rel;
            }
        }
        if(!stopRequested && hasFirstPacket){
            double d=duration>=0?duration:0;
            std::size_t maxFrames=std::max<std::size_t>(1,std::size_t(std::ceil(d*c.frameFps-1e-9)));
            while(!stopRequested&&frameIndex<maxFrames){
                emit(firstPacketTs+double(frameIndex)/c.frameFps);
                frameIndex++;
            }
        }
        std::cout<<"DONE {\"packets\":"<<packets<<"}"<<'\n'<<std::flush;
        return 0;
    } catch(const std::exception&e){
        std::ostringstream out;out<<"ERROR {\"message\":\"";
        for(char c:std::string(e.what())) {if(c=='"'||c=='\\')out<<'\\';if(c=='\n'||c=='\r')c=' ';out<<c;}
        out<<"\"}";
        std::cout<<out.str()<<'\n'<<std::flush;
        std::cerr<<"[OF GUI engine] "<<e.what()<<std::endl;
        return 1;
    }
}

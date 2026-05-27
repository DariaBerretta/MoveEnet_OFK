#include "visualization_utils.h"

#include <yarp/os/LogStream.h>

#include <algorithm>
#include <ctime>
#include <filesystem>

namespace
{
constexpr const char *kWindowName = "Visualisation";

cv::Size liveDisplaySize(const cv::Size &res)
{
    if (res.width <= 0 || res.height <= 0) {
        return res;
    }
    return res;
}

std::string buildDefaultVideoPath(const std::string &datapath_file)
{
    std::time_t now = std::time(nullptr);
    std::tm *tm = std::localtime(&now);
    char date_str[11];
    std::strftime(date_str, sizeof(date_str), "%Y-%m-%d", tm);

    std::filesystem::path csv_path(datapath_file);
    std::string folder_name = csv_path.parent_path().parent_path().filename().string();
    return std::string("/home/moveEnetFlow/mp4_files/") + date_str + "_" + folder_name + ".mp4";
}
}

bool initialiseVisualization(VisualizationContext &ctx,
                             const cv::Size &res,
                             bool is_visualize,
                             bool no_video,
                             std::string &output_video,
                             const std::string &datapath_file,
                             double output_period)
{
    ctx.visualize = is_visualize;
    ctx.display_size = liveDisplaySize(res);

    if (output_video.empty() && !no_video) {
        output_video = buildDefaultVideoPath(datapath_file);
    }

    if (is_visualize || (!output_video.empty() && !no_video)) {
        ctx.canvas = cv::Mat(res, CV_8UC3);
    }

    if (is_visualize) {
        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(kWindowName, ctx.display_size);
        cv::Mat initial_frame = ctx.canvas.empty() ? cv::Mat(res, CV_8UC3, cv::Scalar(0, 0, 0)) : ctx.canvas;
        cv::Mat display_frame;
        if (ctx.display_size.area() > 0 && ctx.display_size != initial_frame.size()) {
            cv::resize(initial_frame, display_frame, ctx.display_size, 0, 0, cv::INTER_LINEAR);
        } else {
            display_frame = initial_frame;
        }
        cv::imshow(kWindowName, display_frame);
        cv::waitKey(1);
    }

    if (!output_video.empty() && !no_video) {
        std::filesystem::path video_path(output_video);
        std::filesystem::create_directories(video_path.parent_path());

        int fps = static_cast<int>(std::max(1.0, 1.0 / output_period));
        ctx.video_writer.open(output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, res);
        if (!ctx.video_writer.isOpened()) {
            yError() << "Could not open video writer for:" << output_video;
            if (is_visualize) {
                yWarning() << "Continuing with live visualization only";
                output_video.clear();
                return true;
            }
            return false;
        }
        yInfo() << "Video output enabled ->" << output_video << " at " << fps << " FPS";
    }

    return true;
}

void renderVisualizationFrame(VisualizationContext &ctx,
                              const cv::Mat &eros_surface,
                              bool pose_is_initialised,
                              const hpecore::skeleton13 &filtered_pose,
                              const hpecore::stampedPose &detected_pose,
                              double tnow)
{
    cv::Mat eros_vis;
    eros_surface.convertTo(eros_vis, CV_8U);
    cv::GaussianBlur(eros_vis, eros_vis, {9, 9}, 0);
    cv::normalize(eros_vis, eros_vis, 0, 255, cv::NORM_MINMAX);
    if (eros_vis.channels() == 1) {
        cv::cvtColor(eros_vis, ctx.canvas, cv::COLOR_GRAY2BGR);
    } else {
        ctx.canvas = eros_vis.clone();
    }

    if (pose_is_initialised) {
        try {
            hpecore::stampedPose pose_filtered;
            pose_filtered.pose = filtered_pose;
            pose_filtered.timestamp = tnow;
            pose_filtered.conf = detected_pose.conf;
            hpecore::drawSkeleton(ctx.canvas, pose_filtered, {255, 0, 0}, 3, 0.0);
        } catch (const cv::Exception &) {
        }
    }

    if (hpecore::poseNonZero(detected_pose.pose)) {
        try {
            hpecore::stampedPose pose_raw = detected_pose;
            hpecore::drawSkeleton(ctx.canvas, pose_raw, {0, 0, 255}, 2, 0.0);
        } catch (const cv::Exception &) {
        }
    }
}

void renderVisualizationFrameOP(VisualizationContext &ctx,
                              const cv::Mat &frame,
                              bool pose_is_initialised,
                              const hpecore::skeleton13 &filtered_pose,
                              const hpecore::stampedPose &detected_pose,
                              double tnow)
{
    cv::Mat frame_vis;
    frame.convertTo(frame_vis, CV_8U);
    cv::normalize(frame_vis, frame_vis, 0, 255, cv::NORM_MINMAX);
    if (frame_vis.channels() == 1) {
        cv::cvtColor(frame_vis, ctx.canvas, cv::COLOR_GRAY2BGR);
    } else {
        ctx.canvas = frame_vis.clone();
    }

    if (pose_is_initialised) {
        try {
            hpecore::stampedPose pose_filtered;
            pose_filtered.pose = filtered_pose;
            pose_filtered.timestamp = tnow;
            pose_filtered.conf = detected_pose.conf;
            hpecore::drawSkeleton(ctx.canvas, pose_filtered, {255, 0, 0}, 3, 0.0);
        } catch (const cv::Exception &) {
        }
    }
    if (hpecore::poseNonZero(detected_pose.pose)) {
        try {
            hpecore::stampedPose pose_raw = detected_pose;
            hpecore::drawSkeleton(ctx.canvas, pose_raw, {0, 0, 255}, 2, 0.0);
        } catch (const cv::Exception &) {
        }
    }
}

void renderVisualizationFrameEPP(VisualizationContext &ctx,
                                const cv::Mat &surface,
                                bool pose_is_initialised,
                                const hpecore::skeleton13 &filtered_pose,
                                const hpecore::stampedPose &detected_pose,
                                double tnow)
{
    cv::Mat vis;
    surface.convertTo(vis, CV_8U);
    if (vis.channels() == 1) {
        cv::cvtColor(vis, ctx.canvas, cv::COLOR_GRAY2BGR);
    } else {
        ctx.canvas = vis.clone();
    }

    if (pose_is_initialised) {
        try {
            hpecore::stampedPose pose_filtered;
            pose_filtered.pose = filtered_pose;
            pose_filtered.timestamp = tnow;
            pose_filtered.conf = detected_pose.conf;
            hpecore::drawSkeleton(ctx.canvas, pose_filtered, {255, 0, 0}, 3, 0.0);
        } catch (const cv::Exception &) {
        }
    }

    if (hpecore::poseNonZero(detected_pose.pose)) {
        try {
            hpecore::stampedPose pose_raw = detected_pose;
            hpecore::drawSkeleton(ctx.canvas, pose_raw, {0, 0, 255}, 2, 0.0);
        } catch (const cv::Exception &) {
        }
    }
}

void writeVisualizationFrame(VisualizationContext &ctx, bool snapshot_ready)
{
    if (ctx.video_writer.isOpened() && snapshot_ready) {
        ctx.video_writer.write(ctx.canvas);
    }
}

bool showVisualizationFrame(VisualizationContext &ctx)
{
    if (!ctx.visualize) {
        return false;
    }

    if (ctx.display_size.area() > 0 && ctx.display_size != ctx.canvas.size()) {
        cv::Mat display_frame;
        cv::resize(ctx.canvas, display_frame, ctx.display_size, 0, 0, cv::INTER_LINEAR);
        cv::imshow(kWindowName, display_frame);
    } else {
        cv::imshow(kWindowName, ctx.canvas);
    }
    char key_pressed = cv::waitKey(1);
    return key_pressed == '\e' || key_pressed == 'q';
}

void closeVisualization(VisualizationContext &ctx, const std::string &output_video)
{
    if (ctx.visualize) {
        cv::destroyAllWindows();
    }

    if (ctx.video_writer.isOpened()) {
        ctx.video_writer.release();
        yInfo() << "Video saved to:" << output_video;
    }
}

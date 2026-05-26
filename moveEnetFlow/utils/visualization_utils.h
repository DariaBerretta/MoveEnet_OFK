#ifndef MOVEENETFLOW_VISUALIZATION_UTILS_H
#define MOVEENETFLOW_VISUALIZATION_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <hpe-core/utility.h>

struct VisualizationContext
{
    cv::Mat canvas;
    cv::Size display_size;
    cv::VideoWriter video_writer;
    bool visualize{false};
};

bool initialiseVisualization(VisualizationContext &ctx,
                             const cv::Size &res,
                             bool is_visualize,
                             bool no_video,
                             std::string &output_video,
                             const std::string &datapath_file,
                             double output_period);

void renderVisualizationFrame(VisualizationContext &ctx,
                              const cv::Mat &eros_surface,
                              bool pose_is_initialised,
                              const hpecore::skeleton13 &filtered_pose,
                              const hpecore::stampedPose &detected_pose,
                              double tnow);
void renderVisualizationFrameOP(VisualizationContext &ctx,
                              const cv::Mat &frame,
                              bool pose_is_initialised,
                              const hpecore::skeleton13 &filtered_pose,
                              const hpecore::stampedPose &detected_pose,
                              double tnow);

// Upstream EventPointPose-style renderer: assumes input already normalized
// (no blur/normalize). Converts to BGR and overlays skeletons.
void renderVisualizationFrameEPP(VisualizationContext &ctx,
                                const cv::Mat &surface,
                                bool pose_is_initialised,
                                const hpecore::skeleton13 &filtered_pose,
                                const hpecore::stampedPose &detected_pose,
                                double tnow);

void writeVisualizationFrame(VisualizationContext &ctx, bool snapshot_ready);

bool showVisualizationFrame(VisualizationContext &ctx);

void closeVisualization(VisualizationContext &ctx, const std::string &output_video);

#endif

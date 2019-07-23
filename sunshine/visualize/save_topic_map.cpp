#include "../topic_model/topic_model.hpp"
#include "utils.hpp"
#include "word_coloring.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <ros/ros.h>
#include <sunshine_msgs/TopicMap.h>

using namespace sunshine_msgs;
using namespace cv;

struct Pose {
    double x, y, z;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "save_topic_map");
    ros::NodeHandle nh("~");
    double default_pixel_scale = 1;
    try {
        nh.getParam("/" + std::string(argv[1]) + "/cell_space", default_pixel_scale);
    } catch (ros::InvalidNameException) {
    }
    auto const pixel_scale = nh.param<double>("pixel_scale", default_pixel_scale);
    auto const input_topic = nh.param<std::string>("input_topic", "/" + std::string(argv[1]) + "/topic_map");
    auto const output_prefix = nh.param<std::string>("output_prefix", std::string(argv[2]));
    auto const minWidth = nh.param<double>("min_width", 0.);
    auto const minHeight = nh.param<double>("min_height", 0.);
    auto const useColor = nh.param<bool>("use_color", false);
    auto const continuous = nh.param<bool>("continuous", false);
    sunshine::WordColorMap<decltype(TopicMap::cell_topics)::value_type> wordColorMap;
    ROS_INFO("Pixel scale: %f", pixel_scale);
    bool done = false;

    auto obsSub = nh.subscribe<TopicMap>(input_topic, 1, [&done, &wordColorMap, output_prefix, pixel_scale, minWidth, minHeight, useColor](sunshine_msgs::TopicMapConstPtr const& msg) {
        size_t const N = msg->cell_topics.size();
        static_assert(sizeof(Pose) == sizeof(double) * 3, "Pose struct has incorrect size.");
        Pose const* poseIter = reinterpret_cast<Pose const*>(msg->cell_poses.data());
        double minX = poseIter->x, minY = poseIter->y, maxX = poseIter->x, maxY = poseIter->y;
        for (size_t i = 0; i < N; i++, poseIter++) {
            minX = std::min(poseIter->x, minX);
            maxX = std::max(poseIter->x, maxX);
            minY = std::min(poseIter->y, minY);
            maxY = std::max(poseIter->y, maxY);
        }

        maxX = std::max(maxX, minX + minWidth);
        maxY = std::max(maxY, minY + minHeight);

        int const numRows = (maxY - minY) / pixel_scale + 1;
        int const numCols = (maxX - minX) / pixel_scale + 1;
        ROS_INFO("N: %lu, R: %d, C: %d", N, numRows, numCols);
        ROS_INFO("Gaps: %lu", numRows * numCols - N);
        //        assert(N == numRows * numCols);

        Mat topicMapImg(numRows, numCols, (useColor) ? sunshine::cvType<Vec4b>::value : sunshine::cvType<double>::value, Scalar(0));
        Mat ppxMapImg  (numRows, numCols, (useColor) ? sunshine::cvType<Vec4b>::value : sunshine::cvType<double>::value, Scalar(0));
        std::set<std::pair<int, int>> points;
        poseIter = reinterpret_cast<Pose const*>(msg->cell_poses.data());
        for (size_t i = 0; i < N; i++, poseIter++) {
            Point const point(std::round((poseIter->x - minX) / pixel_scale),
                std::round((maxY - poseIter->y) / pixel_scale));
            assert(points.insert({ point.x, point.y }).second);
            if (useColor) {
                auto const color = wordColorMap.colorForWord(msg->cell_topics[i]);
                topicMapImg.at<Vec4b>(point) = { color.r, color.g, color.b, color.a };
            } else {
                topicMapImg.at<double>(point) = msg->cell_topics[i] + 1;
            }
            ppxMapImg.at<double>(point) = msg->cell_ppx[i];
        }
        ROS_INFO("Colors: %lu", wordColorMap.getNumColors());
        assert(points.size() == N);

        imwrite(output_prefix + "-" + std::to_string(msg->seq) + "-topics.png", topicMapImg);
        imwrite(output_prefix + "-" + std::to_string(msg->seq) + "-ppx.png", ppxMapImg);

        done = true;
        return;
    });

    if (continuous) {
        ros::spin();
    } else {
        while (!done) {
            ros::spinOnce();
        }
    }
}

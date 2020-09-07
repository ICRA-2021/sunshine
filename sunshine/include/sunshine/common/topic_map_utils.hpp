//
// Created by stewart on 9/7/20.
//

#ifndef SUNSHINE_PROJECT_TOPIC_MAP_UTILS_HPP
#define SUNSHINE_PROJECT_TOPIC_MAP_UTILS_HPP

#include "utils.hpp"
#include "word_coloring.hpp"
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ros/console.h>
#include <sunshine_msgs/TopicMap.h>

namespace sunshine {

cv::Mat createTopicImg(const sunshine_msgs::TopicMapConstPtr& msg,
                       sunshine::WordColorMap<decltype(sunshine_msgs::TopicMap::cell_topics)::value_type>& wordColorMap,
                       double pixel_scale,
                       bool useColor,
                       double minWidth             = 0,
                       double minHeight            = 0,
                       const std::string& fixedBox = "",
                       bool debug                  = false) {
    using namespace sunshine_msgs;
    using namespace cv;
    size_t const N = msg->cell_topics.size();

    struct Pose {
        double x, y, z;
    };
    static_assert(sizeof(Pose) == sizeof(double) * 3, "Pose struct has incorrect size.");

    Pose const* poseIter = reinterpret_cast<Pose const*>(msg->cell_poses.data());
    double minX = poseIter->x, minY = poseIter->y, maxX = poseIter->x, maxY = poseIter->y;
    if (fixedBox.empty()) {
        for (size_t i = 0; i < N; i++, poseIter++) {
            minX = std::min(poseIter->x, minX);
            maxX = std::max(poseIter->x, maxX);
            minY = std::min(poseIter->y, minY);
            maxY = std::max(poseIter->y, maxY);
        }

        maxX = std::max(maxX, minX + minWidth);
        maxY = std::max(maxY, minY + minHeight);
    } else {
        auto const size_spec = sunshine::readNumbers<4, 'x'>(fixedBox);
        minX                 = size_spec[0];
        minY                 = size_spec[1];
        maxX                 = size_spec[0] + size_spec[2];
        maxY                 = size_spec[1] + size_spec[3];
    }

    ROS_INFO("Saving map over region (%f, %f) to (%f, %f) (size spec %s)", minX, minY, maxX, maxY, fixedBox.c_str());

    int const numRows = static_cast<int>((maxY - minY) / pixel_scale + 1);
    int const numCols = static_cast<int>((maxX - minX) / pixel_scale + 1);

    Mat topicMapImg(numRows, numCols, (useColor) ? sunshine::cvType<Vec4b>::value : sunshine::cvType<double>::value, Scalar(0));
    std::set<std::pair<int, int>> points;
    poseIter        = reinterpret_cast<Pose const*>(msg->cell_poses.data());
    size_t outliers = 0, overlaps = 0;
    for (size_t i = 0; i < N; i++, poseIter++) {
        Point const point(static_cast<int>(std::round((poseIter->x - minX) / pixel_scale)),
                          static_cast<int>(std::round((maxY - poseIter->y) / pixel_scale)));
        if (point.x < 0 || point.y < 0 || point.x >= numRows || point.y >= numCols) {
            outliers++;
            continue;
        }
        if (!points.insert({point.x, point.y}).second) {
            ROS_WARN_THROTTLE(1, "Duplicate cells found at (%d, %d)", point.x, point.y);
            overlaps++;
        }
        if (useColor) {
            auto const color             = wordColorMap.colorForWord(msg->cell_topics[i]);
            topicMapImg.at<Vec4b>(point) = {color.r, color.g, color.b, color.a};
        } else {
            topicMapImg.at<double>(point) = msg->cell_topics[i] + 1;
        }
    }
    ROS_INFO_COND(outliers > 0, "Discarded %lu points outside of %s", outliers, fixedBox.c_str());
    ROS_WARN_COND(overlaps > 0, "Dicarded %lu overlapped points.", overlaps);
    ROS_INFO("Points: %lu, Rows: %d, Cols: %d, Colors: %lu", N, numRows, numCols, wordColorMap.getNumColors());
    if (static_cast<unsigned long>(numRows) * static_cast<unsigned long>(numCols) < N) {
        ROS_WARN("More cells than space in grid - assuming there are overlapping cells.");
    } else {
        ROS_INFO("Gaps: %lu", static_cast<unsigned long>(numRows) * static_cast<unsigned long>(numCols) - N);
    }
    return topicMapImg;
}

template<typename WordColorMapType>
void saveTopicImg(cv::Mat const& topicMapImg,
                  const std::string& filename,
                  const std::string& colors_filename = "",
                  WordColorMapType* colorMap  = nullptr) {
    cv::imwrite(filename, topicMapImg);
    if (!colors_filename.empty()) {
        if (colorMap == nullptr) throw std::invalid_argument("Must provide color map to save colors");
        std::ofstream colorWriter(colors_filename);
        for (auto const& entry : colorMap->getAllColors()) {
            colorWriter << entry.first;
            for (auto const& v : entry.second) { colorWriter << "," << std::to_string(v); }
            colorWriter << "\n";
        }
        colorWriter.close();
    }
}

} // namespace sunshine

#endif // SUNSHINE_PROJECT_TOPIC_MAP_UTILS_HPP

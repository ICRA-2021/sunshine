#include "../topic_model/topic_model_node.hpp"
#include "utils.hpp"
#include "word_coloring.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <ros/ros.h>
#include <sunshine_msgs/SaveObservationModel.h>
#include <sunshine_msgs/GetTopicModel.h>
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
    auto const rost_namespace = nh.param<std::string>("topic_model_namespace", "rost");
    double default_pixel_scale = 1;
    try {
        nh.getParam("/" + rost_namespace + "/cell_space", default_pixel_scale);
    } catch (ros::Exception) {
        // do nothing
    }
    auto const pixel_scale = nh.param<double>("pixel_scale", default_pixel_scale);
    auto const input_topic = nh.param<std::string>("input_topic", "/" + rost_namespace + "/topic_map");
    auto const output_prefix = nh.param<std::string>("output_prefix", "topic-map");
    auto const minWidth = nh.param<double>("min_width", 0.);
    auto const minHeight = nh.param<double>("min_height", 0.);
    auto const fixedBox = nh.param<std::string>("box", "");
    auto const useColor = nh.param<bool>("use_color", false);
    auto const continuous = nh.param<bool>("continuous", false);
    auto const saveTopicTimeseries = nh.param<bool>("save_topic_timeseries", false);
    auto const saveTopicModel = nh.param<bool>("save_topic_model", true);
    auto const saveTopicCells = nh.param<bool>("save_topic_cells", true);
    sunshine::WordColorMap<decltype(TopicMap::cell_topics)::value_type> wordColorMap;
    ROS_INFO("Pixel scale: %f", pixel_scale);
    bool done = false;

    ros::ServiceClient client = nh.serviceClient<SaveObservationModel>("/" + rost_namespace + "/save_topics_by_time_csv");
    ros::ServiceClient cellClient = nh.serviceClient<SaveObservationModel>("/" + rost_namespace + "/save_topics_by_cell_csv");
    ros::ServiceClient modelClient = nh.serviceClient<GetTopicModel>("/" + rost_namespace + "/get_topic_model");

    auto obsSub = nh.subscribe<TopicMap>(input_topic, 1, [&done, &wordColorMap, &client, &modelClient, &cellClient, output_prefix, pixel_scale, minWidth,
                                                          saveTopicTimeseries, saveTopicModel, saveTopicCells, minHeight, useColor, fixedBox](sunshine_msgs::TopicMapConstPtr const& msg) {
        if (saveTopicModel) {
            GetTopicModel getTopicModel;
            if (modelClient.call(getTopicModel)) {
                std::string const filename = output_prefix + "-" + std::to_string(msg->seq) + "-modelweights.bin";
                std::fstream writer(filename, std::ios::out | std::ios::binary);
                if (writer.good()) {
                    writer.write(reinterpret_cast<char *>(getTopicModel.response.topic_model.phi.data()),
                                 sizeof(decltype(getTopicModel.response.topic_model.phi)::value_type) / sizeof(char)
                                 * getTopicModel.response.topic_model.phi.size());
                    writer.close();
                } else {
                    ROS_ERROR("Failed to save topic model to file %s", filename.c_str());
                }
            } else {
                ROS_ERROR("Failed to save topic model!");
            }
        }

        if (saveTopicTimeseries) {
            SaveObservationModel saveObservationModel;
            saveObservationModel.request.filename = output_prefix + "-" + std::to_string(msg->seq) + "-timeseries.csv";
            if (client.call(saveObservationModel)) {
                ROS_INFO("Saved timeseries to %s", saveObservationModel.request.filename.c_str());
            } else {
                ROS_ERROR("Failed to save topic timeseries to %s!", saveObservationModel.request.filename.c_str());
            }
        }

        if (saveTopicCells) {
            SaveObservationModel saveObservationModel;
            saveObservationModel.request.filename = output_prefix + "-" + std::to_string(msg->seq) + "-cells.csv";
            if (cellClient.call(saveObservationModel)) {
                ROS_INFO("Saved timeseries to %s", saveObservationModel.request.filename.c_str());
            } else {
                ROS_ERROR("Failed to save topic timeseries to %s!", saveObservationModel.request.filename.c_str());
            }
        }

        size_t const N = msg->cell_topics.size();
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
            minX = size_spec[0];
            minY = size_spec[1];
            maxX = size_spec[0] + size_spec[2];
            maxY = size_spec[1] + size_spec[3];
        }

        ROS_INFO("Saving map over region (%f, %f) to (%f, %f) (size spec %s)", minX, minY, maxX, maxY, fixedBox.c_str());

        int const numRows = static_cast<int>((maxY - minY) / pixel_scale + 1);
        int const numCols = static_cast<int>((maxX - minX) / pixel_scale + 1);

        Mat topicMapImg(numRows, numCols, (useColor) ? sunshine::cvType<Vec4b>::value : sunshine::cvType<double>::value, Scalar(0));
        Mat ppxMapImg(numRows, numCols, sunshine::cvType<double>::value, Scalar(0));
        std::set<std::pair<int, int>> points;
        poseIter = reinterpret_cast<Pose const*>(msg->cell_poses.data());
        size_t outliers = 0, overlaps = 0;
        for (size_t i = 0; i < N; i++, poseIter++) {
            Point const point(static_cast<int>(std::round((poseIter->x - minX) / pixel_scale)),
                static_cast<int>(std::round((maxY - poseIter->y) / pixel_scale)));
            if (point.x < 0 || point.y < 0 || point.x >= numRows || point.y >= numCols) {
                outliers++;
                continue;
            }
            if (!points.insert({ point.x, point.y }).second) {
                ROS_WARN_THROTTLE(1, "Duplicate cells found at (%d, %d)", point.x, point.y);
                overlaps++;
            }
            if (useColor) {
                auto const color = wordColorMap.colorForWord(msg->cell_topics[i]);
                topicMapImg.at<Vec4b>(point) = { color.r, color.g, color.b, color.a };
            } else {
                topicMapImg.at<double>(point) = msg->cell_topics[i] + 1;
            }
            if (msg->cell_ppx.size() > i) ppxMapImg.at<double>(point) = msg->cell_ppx[i];
        }
        ROS_INFO_COND(outliers > 0, "Discarded %lu points outside of %s", outliers, fixedBox.c_str());
        ROS_WARN_COND(overlaps > 0, "Dicarded %lu overlapped points.", overlaps);
        ROS_INFO("Points: %lu, Rows: %d, Cols: %d, Colors: %lu", N, numRows, numCols, wordColorMap.getNumColors());
        if (static_cast<unsigned long>(numRows) * static_cast<unsigned long>(numCols) < N){
            ROS_WARN("More cells than space in grid - assuming there are overlapping cells.");
        } else {
            ROS_INFO("Gaps: %lu", static_cast<unsigned long>(numRows) * static_cast<unsigned long>(numCols) - N);
        }

        imwrite(output_prefix + "-" + std::to_string(msg->seq) + "-topics.png", topicMapImg);
        imwrite(output_prefix + "-" + std::to_string(msg->seq) + "-ppx.png", ppxMapImg);
        std::ofstream colorWriter(output_prefix + "-" + std::to_string(msg->seq) + "-colors.csv");
        for (auto const& entry : wordColorMap.getAllColors()) {
            colorWriter << entry.first;
            for (auto const& v : entry.second) {
                colorWriter << "," << std::to_string(v);
            }
            colorWriter << "\n";
        }
        colorWriter.close();

        done = true;
        return;
    });

    if (continuous) {
        ros::spin();
    } else {
        ros::Rate idler(10);
        while (!done) {
            ros::spinOnce();
            idler.sleep();
        }
    }
}

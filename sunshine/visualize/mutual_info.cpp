#include "../topic_model/topic_model.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <ros/ros.h>
#include <sunshine_msgs/TopicMap.h>

using namespace sunshine_msgs;
using namespace cv;
using namespace sunshine;

void normalize(std::vector<double>& v)
{
    double sum = 0;
    for (auto const& x : v) {
        sum += x;
    }
    for (auto& x : v) {
        x /= sum;
    }
}

void normalize(std::vector<std::vector<double>>& v)
{
    double sum = 0;
    for (auto const& x : v) {
        for (auto const& y : x) {
            sum += y;
        }
    }
    for (auto& x : v) {
        for (auto& y : x) {
            y /= sum;
        }
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mutual_info");
    ros::NodeHandle nh("~");
    auto const input_topic = nh.param<std::string>("input_topic", "/topic_map");
    auto const pixel_scale = nh.param<double>("pixel_scale", 0.01);

    size_t K_image = 0;
    std::map<std::array<uchar, 3>, int> topicsByColor;
    std::map<word_pose_t, int> imageTopics;
    std::vector<double> imageTopicFreq;
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    for (auto y = 0; y < image.rows; y++) {
        for (auto x = 0; x < image.cols; x++) {
            int topic = K_image++;
            auto const color = image.at<Vec3b>(Point(x, y));
            auto topicIter = topicsByColor.find({color[0], color[1], color[2]});
            if (topicIter != topicsByColor.end()) {
                topic = topicIter->second;
            } else {
                topicsByColor.insert({ {color[0], color[1], color[2]}, topic });
            }
            imageTopicFreq.resize(K_image);
            imageTopicFreq[topic] += 1;
            imageTopics.insert({ word_pose_t{ 0, x * pixel_scale, y * pixel_scale, 0 }, topic });
        }
    }
    normalize(imageTopicFreq);

    ros::Subscriber obsSub = nh.subscribe<sunshine_msgs::TopicMap>(input_topic, 1, [K_image, &imageTopics, &imageTopicFreq](sunshine_msgs::TopicMapConstPtr const& msg) {
        std::map<word_pose_t, int> cellTopics;
        size_t K_msg = 0;
        size_t const N = msg->cell_topics.size();
        std::vector<double> cellTopicFreq;
        for (size_t i = 0; i < N; i++) {
            word_pose_t pose{ 0, msg->cell_poses[i * 3], msg->cell_poses[i * 3 + 1], 0 };
            auto const topic = msg->cell_topics[i];
            cellTopics.insert({ pose, topic });
            cellTopicFreq.resize(std::max(K_msg, static_cast<size_t>(topic)));
            cellTopicFreq[topic] += 1;
        }
        normalize(cellTopicFreq);

        double normConstant = 0;
        std::vector<std::vector<double>> jointPdf(K_image, std::vector<double>(K_msg, 0));
        for (auto const& cellTopicIter : cellTopics) {
            auto const& imageTopicIter = imageTopics.find(cellTopicIter.first);
            if (imageTopicIter == imageTopics.end()) {
                continue;
            }
            jointPdf[imageTopicIter->second][cellTopicIter.second] += 1;
            normConstant += 1;
        }

        double mutualInfo = 0;
        for (auto k_image = 0; k_image < K_image; k_image++) {
            for (auto k_msg = 0; k_msg < K_msg; k_msg++) {
                auto const& joint = jointPdf[k_image][k_msg];
                if (joint == 0) {
                    continue;
                }
                mutualInfo += joint * std::log(joint / (cellTopicFreq[k_msg] * imageTopicFreq[k_image]));
            }
        }
        mutualInfo /= normConstant;
        std::cout << "Mutual info: " << mutualInfo << std::endl;
    });
    ros::spin();
}

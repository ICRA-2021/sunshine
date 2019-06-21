#include "../topic_model/topic_model.hpp"
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
    assert(argc == 3);
    ros::init(argc, argv, "save_topic_map");
    ros::NodeHandle nh("~");
    std::string const input_topic = "/" + std::string(argv[1]) + "/topic_map";
    double pixel_scale;
    auto success = nh.getParam("/" + std::string(argv[1]) + "/cell_space", pixel_scale);
    ROS_INFO("Pixel scale: %f", pixel_scale);
    assert(success);
    bool done = false;

    auto obsSub = nh.subscribe<TopicMap>(input_topic, 1, [&done, argv, pixel_scale](sunshine_msgs::TopicMapConstPtr const& msg) {
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

        int const numRows = (maxY - minY) / pixel_scale + 1;
        int const numCols = (maxX - minX) / pixel_scale + 1;
        ROS_INFO("N: %lu, R: %d, C: %d", N, numRows, numCols);
        ROS_INFO("Gaps: %lu", numRows * numCols - N);
        //        assert(N == numRows * numCols);

        Mat topicMapImg(numRows, numCols, CV_64F, Scalar(0));
        Mat ppxMapImg(numRows, numCols, CV_64F, Scalar(0));
        std::set<std::pair<int, int>> points;
        poseIter = reinterpret_cast<Pose const*>(msg->cell_poses.data());
        for (size_t i = 0; i < N; i++, poseIter++) {
            Point const point(std::round((poseIter->x - minX) / pixel_scale),
                std::round((maxY - poseIter->y) / pixel_scale));
            assert(points.insert({ point.x, point.y }).second);
            topicMapImg.at<double>(point) = msg->cell_topics[i] + 1;
            ppxMapImg.at<double>(point) = msg->cell_ppx[i];
        }
        assert(points.size() == N);

        imwrite(std::string(argv[2]) + "-topics.png", topicMapImg);
        imwrite(std::string(argv[2]) + "-ppx.png", ppxMapImg);

        done = true;
        return;
    });

    while (!done) {
        ros::spinOnce();
    }
}

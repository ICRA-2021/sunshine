#include "image_utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using namespace sunshine;
typedef double_t DepthType;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_walker");

    ros::NodeHandle nh("~");
    auto const fps = nh.param<double>("fps", 30);
    auto const speed = nh.param<double>("speed", 30);
    auto const altitude = nh.param<double>("height", -1);
    auto const fov = nh.param<double>("fov", -1);
    auto const overlap = nh.param<int>("overlap", 0);
    auto const col_major = nh.param<bool>("col_major", false);
    auto const size = nh.param<std::string>("size", "0x0");
    auto const scale = nh.param<double>("scale", 1);
    auto const image_name = nh.param<std::string>("image", argv[1]);
    auto const image_topic = nh.param<std::string>("image_topic", "/camera/image");
    auto const depth_cloud_topic = nh.param<std::string>("depth_cloud_topic", "/camera/test_point_cloud");
    auto const depth_image_topic = nh.param<std::string>("depth_image_topic", "/camera/depth");
    auto const transform_topic = nh.param<std::string>("transform_topic", "/robot_tf");
    auto const frame_id = nh.param<std::string>("frame_id", "base_link");
    auto const pixel_scale = nh.param<double>("pixel_scale", 0.01);

    cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
    cv::waitKey(30);

    if (fps <= 0) {
        throw std::invalid_argument("fps cannot be non-positive!");
    }

    if (size.find("x") == size.npos) {
        throw std::invalid_argument("Size must be in the format of LxW");
    }

    auto width = std::stoi(size.substr(0, size.find("x")));
    auto height = std::stoi(size.substr(size.find("x") + 1));
    if (width <= 0 || height <= 0) {
        if (fov <= 0 || altitude <= 0) {
            throw std::invalid_argument("Either size (both L and W) must be set or both FOV and height must be set.");
        }

        throw std::runtime_error("FOV is not implemented."); // TODO: FOV calculation
    } else if (fov > 0 && altitude > 0) {
        throw std::invalid_argument("Cannot set all of size, FOV, and height! Choose at most two.");
    }

    if (overlap < 0) {
        throw std::invalid_argument("Overlap cannot be negative!");
    } else if ((col_major && overlap >= width) || (!col_major && overlap >= height)) {
        throw std::invalid_argument("Overlap is too large!");
    }

    double cx = width / 2.;
    double cy = height / 2.;
    double track = (col_major) ? cx : cy;

    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = frame_id;

    auto tfPublisher = nh.advertise<geometry_msgs::TransformStamped>(transform_topic, 1);
    auto const poseCallback = [&]() {
        tf2::Quaternion q;
        q.setRPY(0, M_PI, M_PI);
        tfPublisher.publish(broadcastTranform(frame_id,
            tf2::Vector3(cx, -cy, (altitude >= 0) ? altitude : 0) * pixel_scale,
            q,
            "map",
            header.stamp));
    };
    poseCallback();
    ros::spinOnce();

    // Publish image topic
    image_transport::ImageTransport it(nh);
    image_transport::Publisher imagePub = it.advertise(image_topic, 1);

    // Publish depth image topic
    bool const publishDepthImage = (fov > 0 || altitude > 0);
    sensor_msgs::PointCloud2Ptr depthCloud;
    ros::Publisher depthCloudPub;
    image_transport::Publisher depthImagePub;
    std::unique_ptr<sunshine::ImageScanner> image_scanner;
    if (publishDepthImage) {
        ROS_INFO("Publishing depth image.");
        image_scanner = std::make_unique<sunshine::ImageScanner3D<>>(image, width, height, getFlatHeightMap(image.cols, image.rows, 0.), scale, cx, cy, altitude, pixel_scale);
    } else {
        image_scanner = std::make_unique<sunshine::ImageScanner>(image, width, height, scale, cx, cy);
    }

    enum class Direction {
        DOWN = -2,
        LEFT = -1,
        RIGHT = 1,
        UP = 2,
        NONE = 0
    };
    Direction dir = Direction::RIGHT;

    auto& primary_axis = (col_major) ? cy : cx;
    auto const primary_axis_max = ((col_major) ? image_scanner->getMaxY() : image_scanner->getMaxX());
    auto const primary_window_extent = ((col_major) ? height : width) / 2.;
    auto& secondary_axis = (col_major) ? cx : cy;
    auto const secondary_axis_max = ((col_major) ? image_scanner->getMaxX() : image_scanner->getMaxY());
    auto const secondary_window_extent = ((col_major) ? width : height) / 2.;
    auto const step_size = speed / fps;

    auto const advanceTrack = [&]() {
        if (track >= secondary_axis_max - secondary_window_extent) {
            dir = Direction::NONE;
            return;
        }
        dir = Direction::DOWN;
        track += std::min(2 * secondary_window_extent - overlap, secondary_axis_max - secondary_window_extent - track);
    };

    auto const lawnmowerMove = [&]() {
        if (dir == Direction::RIGHT) {
            if (primary_axis >= primary_axis_max - primary_window_extent) {
                advanceTrack();
            } else {
                primary_axis += std::min(step_size, primary_axis_max - primary_window_extent - primary_axis);
            }
        } else if (dir == Direction::LEFT) {
            if (primary_axis <= primary_window_extent) {
                advanceTrack();
            } else {
                primary_axis -= std::min(step_size, primary_axis - primary_window_extent);
            }
        }

        if (dir == Direction::DOWN) {
            // Advance towards the new track
            secondary_axis += std::min(step_size, track - secondary_axis);
            if (secondary_axis >= track) {
                dir = (primary_axis <= primary_window_extent) ? Direction::RIGHT : Direction::LEFT;
            }
        }
    };

    ros::Rate rate(fps);
    while (nh.ok()) {
        header.stamp = ros::Time::now();
        poseCallback();
        image_scanner->moveTo(cx, cy);
        auto const& visibleRegion = image_scanner->getCurrentView();
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", visibleRegion).toImageMsg();
        if (publishDepthImage) {
            sunshine::ImageScanner3D<>* image_scanner_3d = dynamic_cast<sunshine::ImageScanner3D<>*>(image_scanner.get());
            depthCloud = image_scanner_3d->getCurrentPointCloud();
            depthCloud->header = header;
            depthCloudPub.publish(*depthCloud);
            depthImagePub.publish(cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, image_scanner_3d->getCurrentDepthView()).toImageMsg());
        }
        imagePub.publish(msg);

        ros::spinOnce();
        rate.sleep();

        lawnmowerMove();
    }
}

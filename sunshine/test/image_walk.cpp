#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>

cv::Mat getVisibleRegion(cv::Mat const& image, float cx, float cy, int width, int height, float scale)
{
    cv::Rect roi;
    roi.x = std::round(cx - width / 2.);
    roi.y = std::round(cy - height / 2.);
    roi.width = width;
    roi.height = height;

    cv::Mat const& region = image(roi);
    cv::Mat scaled(std::ceil(height * scale), std::ceil(width * scale), region.type());

    cv::resize(region, scaled, scaled.size(), 0, 0, cv::INTER_CUBIC);
    return scaled;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_publisher");

    ros::NodeHandle nh("~");
    auto const fps = nh.param<float>("fps", 30);
    auto const speed = nh.param<float>("speed", 30);
    auto const altitude = nh.param<float>("height", -1);
    auto const fov = nh.param<float>("fov", -1);
    auto const overlap = nh.param<int>("overlap", 0);
    auto const col_major = nh.param<bool>("col_major", false);
    auto const size = nh.param<std::string>("size", "0x0");
    auto const scale = nh.param<float>("scale", 1);
    auto const image_name = nh.param<std::string>("image", argv[1]);
    auto const image_topic = nh.param<std::string>("image_topic", "/camera/image");
    auto const depth_topic = nh.param<std::string>("depth_topic", "/camera/depth_image");

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

    float x = width / 2.f;
    float y = height / 2.f;
    float track = (col_major) ? x : y;

    bool const publishDepthImage = (fov > 0 || altitude > 0);
    cv::Mat depthImage(height, width, CV_32F);
    if (publishDepthImage) {
        for (auto row = 0; row < height; row++) {
            for (auto col = 0; col < width; col++) {
                depthImage.at<float>(row, col) = static_cast<float>(std::pow(std::pow(row - y, 2) + std::pow(width - x, 2) + std::pow(altitude, 2), 1. / 2.));
            }
        }
    }

    image_transport::ImageTransport it(nh);
    image_transport::Publisher imagePub = it.advertise(image_topic, 1);
    image_transport::ImageTransport it2(nh);
    image_transport::Publisher depthPub = it.advertise(depth_topic, 1);
    cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
    cv::waitKey(30);

    enum class Direction {
        DOWN = -2,
        LEFT = -1,
        RIGHT = 1,
        UP = 2,
        NONE = 0
    };
    Direction dir = Direction::RIGHT;

    auto& primary_axis = (col_major) ? y : x;
    auto const primary_axis_max = ((col_major) ? image.rows : image.cols);
    auto const primary_window_extent = ((col_major) ? height : width) / 2.f;
    auto& secondary_axis = (col_major) ? x : y;
    auto const secondary_axis_max = ((col_major) ? image.cols : image.rows);
    auto const secondary_window_extent = ((col_major) ? width : height) / 2.f;
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
        ROS_INFO("x,y: %f,%f", x, y);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8",
            getVisibleRegion(image, x, y, width, height, scale))
                                        .toImageMsg();
        imagePub.publish(msg);
        if (publishDepthImage) {
            depthPub.publish(cv_bridge::CvImage(std_msgs::Header(), "MONO16", depthImage).toImageMsg());
        }

        ros::spinOnce();
        lawnmowerMove();
        rate.sleep();
    }
}

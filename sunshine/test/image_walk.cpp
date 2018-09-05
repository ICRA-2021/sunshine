#include "utils.hpp"
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

cv::Mat getVisibleRegion(cv::Mat const& image, double cx, double cy, int width, int height)
{
    cv::Rect roi;
    roi.x = static_cast<int>(std::round(cx - width / 2.));
    roi.y = static_cast<int>(std::round(cy - height / 2.));
    roi.width = width;
    roi.height = height;
    return image(roi);
}

cv::Mat scaleImage(cv::Mat const& image, double scale)
{
    cv::Mat scaled(std::ceil(image.rows * scale), std::ceil(image.cols * scale), image.type());
    cv::resize(image, scaled, scaled.size(), 0, 0, cv::INTER_CUBIC);
    return scaled;
}

void copyImageColorToPointCloud(cv::Mat const& image, sensor_msgs::PointCloud2& pointCloud)
{
    auto const& colorField = std::find_if(pointCloud.fields.begin(), pointCloud.fields.end(), [](decltype(pointCloud.fields)::value_type const& field) {
        return field.name == "rgb";
    });
    if (colorField == pointCloud.fields.end()) {
        throw std::invalid_argument("Cannot find color field in point cloud.");
    }
    assert(static_cast<uint64_t>(image.rows * image.cols) == pointCloud.width * pointCloud.height);
    uint8_t* dataPtr = pointCloud.data.data() + colorField->offset;
    size_t dataStep = pointCloud.point_step;
    for (auto row = 0; row < image.rows; row++) {
        for (auto col = 0; col < image.cols; col++) {
            auto const color = image.at<cv::Vec3b>(row, col);
            for (auto i = 0u; i < 3; i++)
                *(dataPtr + i) = color.val[i];
            dataPtr += dataStep;
        }
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_publisher");

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
    int const iwidth = std::ceil(width * scale), iheight = std::ceil(height * scale);

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
    cv::Mat depthImage(iheight, iwidth, cvType<DepthType>::value);
    sensor_msgs::PointCloud2 depthCloud;
    ros::Publisher depthCloudPub;
    image_transport::Publisher depthImagePub;
    if (publishDepthImage) {
        ROS_INFO("Publishing depth image.");
        sensor_msgs::PointField basePointField;
        basePointField.datatype = basePointField.FLOAT32;
        basePointField.count = 1;

        auto offset = 0u;
        for (auto const& field : { "x", "y", "z" }) {
            sensor_msgs::PointField pointField = basePointField;
            pointField.name = field;
            pointField.offset = offset;
            depthCloud.fields.push_back(pointField);
            offset += sizeof(float);
        }

        sensor_msgs::PointField colorField;
        colorField.datatype = colorField.UINT32;
        colorField.count = 1;
        colorField.offset = offset;
        colorField.name = "rgb";
        depthCloud.fields.push_back(colorField);

        depthCloud.header.frame_id = frame_id;
        depthCloud.width = uint32_t(iwidth);
        depthCloud.height = uint32_t(iheight);
        depthCloud.is_dense = true;
        depthCloud.point_step = sizeof(float) * 3 + sizeof(uint32_t);
        depthCloud.row_step = depthCloud.point_step * depthCloud.width;
        depthCloud.data = std::vector<uint8_t>(depthCloud.row_step * depthCloud.height);
        float* depthCloudIterator = reinterpret_cast<float*>(&(depthCloud.data.data()[0]));

        depthCloudPub = nh.advertise<sensor_msgs::PointCloud2>(depth_cloud_topic, 1);
        depthImagePub = it.advertise(depth_image_topic, 1);
        auto const extentX = (width - 1) / 2., extentY = (height - 1) / 2.;
        for (auto row = 0; row < iheight; row++) {
            for (auto col = 0; col < iwidth; col++) {
                double const x = col / scale, y = row / scale;
                double const depth = std::pow(std::pow(y - extentY, 2) + std::pow(x - extentX, 2) + std::pow(altitude, 2), 1. / 2.) * pixel_scale;
                depthImage.at<DepthType>(row, col) = static_cast<DepthType>(depth);
                *(depthCloudIterator++) = static_cast<float>((x - extentX) * pixel_scale);
                *(depthCloudIterator++) = static_cast<float>((y - extentY) * pixel_scale);
                *(depthCloudIterator++) = static_cast<float>(altitude * pixel_scale);
                static_assert(sizeof(float) == sizeof(uint32_t), "Following code assumes that sizeof(float) == sizeof(uint32_t)!");
                depthCloudIterator++; // skip color
            }
        }
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
    auto const primary_axis_max = ((col_major) ? image.rows : image.cols);
    auto const primary_window_extent = ((col_major) ? height : width) / 2.;
    auto& secondary_axis = (col_major) ? cx : cy;
    auto const secondary_axis_max = ((col_major) ? image.cols : image.rows);
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
        auto const& visibleRegion = scaleImage(getVisibleRegion(image, cx, cy, width, height), scale);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", visibleRegion).toImageMsg();
        if (publishDepthImage) {
            copyImageColorToPointCloud(visibleRegion, depthCloud);
            depthCloud.header = header;
            depthCloudPub.publish(depthCloud);
            depthImagePub.publish(cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, depthImage).toImageMsg());
        }
        imagePub.publish(msg);

        ros::spinOnce();
        rate.sleep();

        lawnmowerMove();
    }
}

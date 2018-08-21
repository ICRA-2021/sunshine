#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

namespace sunshine {
using HSV = std::array<double, 3>;
using RGB = std::array<uint8_t, 3>;

template <typename ValType>
struct cvType {
};
template <>
struct cvType<float> {
    static int const value = CV_32F;
};
template <>
struct cvType<double> {
    static int const value = CV_64F;
};
template <>
struct cvType<uint8_t> {
    static int const value = CV_8U;
};
template <>
struct cvType<uint16_t> {
    static int const value = CV_16U;
};

/**
 * @brief toMat Converts two parallel vectors, of 2D poses (xy-coordinates) and 1D values, respectively, to a cv::Mat object.
 * @param idxes 2D poses of the form [x1,y1,...,xN,yN] (size of 2*N)
 * @param values Scalar values corresponding to each pose of the form [z1,...,zN] (size of N)
 * @return cv::Mat where mat.at<MatValueType>(yI, xI) = zI
 */
template <typename IdxType, typename ValueType, typename MatValueType = ValueType>
cv::Mat toMat(std::vector<IdxType> const& idxes, std::vector<ValueType> const& values)
{
    assert(idxes.size() == values.size() * 2);

    // Compute the required size of the matrix based on the largest (x,y) coordinate
    IdxType max_x = 0, max_y = 0;
    for (auto i = 0ul; i < values.size(); i++) {
        assert(idxes[i * 2] >= 0 && idxes[i * 2 + 1] >= 0);
        max_x = std::max(max_x, idxes[i * 2]);
        max_y = std::max(max_y, idxes[i * 2 + 1]);
    }
    cv::Mat mat(max_y + 1, max_x + 1, cvType<MatValueType>::value); // Don't forget to add 1 to each dimension (because 0-indexed)

    for (auto i = 0ul; i < values.size(); i++) {
        mat.at<MatValueType>(idxes[i * 2 + 1], idxes[i * 2]) = values[i]; // mat.at(y,x) = z
    }
    return mat;
}

geometry_msgs::TransformStamped broadcastTranform(std::string frame_id, tf2::Vector3 origin, tf2::Quaternion q, std::string parentFrame = "map", ros::Time stamp = ros::Time::now())
{
    static tf2_ros::TransformBroadcaster br;
    tf2::Transform transform;
    transform.setOrigin(origin);
    transform.setRotation(q);
    auto stampedTransform = tf2::toMsg(tf2::Stamped<tf2::Transform>(transform, stamp, parentFrame));
    stampedTransform.child_frame_id = frame_id;
    br.sendTransform(stampedTransform);
    return stampedTransform;
}

template <typename T>
inline void transformPose(geometry_msgs::Point& out, std::vector<T> const& poses, size_t const& idx, geometry_msgs::TransformStamped const& transform)
{
    geometry_msgs::Point point;
    point.x = poses[idx];
    point.y = poses[idx + 1];
    point.z = poses[idx + 2];
    tf2::doTransform(point, out, transform);
}

RGB hsvToRgb(HSV const& hsv)
{
    double const chroma = hsv[1] * hsv[2];
    double const hueNorm = fmod(hsv[0], 360) / 60.;
    double const secondaryChroma = chroma * (1 - fabs(fmod(hueNorm, 2) - 1));

    uint32_t const hueSectant = static_cast<uint32_t>(std::floor(hueNorm));
    std::array<double, 3> rgb = { 0, 0, 0 };
    switch (hueSectant) {
    case 0:
        rgb = { chroma, secondaryChroma, 0 };
        break;
    case 1:
        rgb = { secondaryChroma, chroma, 0 };
        break;
    case 2:
        rgb = { 0, chroma, secondaryChroma };
        break;
    case 3:
        rgb = { 0, secondaryChroma, chroma };
        break;
    case 4:
        rgb = { secondaryChroma, 0, chroma };
        break;
    case 5:
        rgb = { chroma, 0, secondaryChroma };
        break;
    }

    double const offset = hsv[2] - chroma;
    return { uint8_t((rgb[0] + offset) * 255.), uint8_t((rgb[1] + offset) * 255.), uint8_t((rgb[2] + offset) * 255.) };
}

template <typename PointContainer>
sensor_msgs::PointCloud2Ptr toPointCloud(PointContainer const& points, std::string frame_id = "/map")
{
    auto const count = points.size();
    uint32_t const height = 1, width = uint32_t(count);
    sensor_msgs::PointCloud2Ptr pc(new sensor_msgs::PointCloud2());
    sensor_msgs::PointField basePointField;
    basePointField.datatype = basePointField.FLOAT32;
    basePointField.count = 1;

    auto offset = 0u;
    for (auto const& field : { "x", "y", "z" }) {
        sensor_msgs::PointField pointField = basePointField;
        pointField.name = field;
        pointField.offset = offset;
        pc->fields.push_back(pointField);
        offset += sizeof(float);
    }

    sensor_msgs::PointField colorField;
    colorField.datatype = colorField.UINT32;
    colorField.count = 1;
    colorField.offset = offset;
    offset += sizeof(uint32_t);
    colorField.name = "rgb";
    pc->fields.push_back(colorField);

    union PointCloudElement {
        uint8_t bytes[sizeof(float) * 3 + sizeof(uint32_t)]; // to enforce size
        struct {
            float x;
            float y;
            float z;
            uint8_t rgb[3];
        } data;
    };

    pc->header.frame_id = frame_id;
    pc->width = uint32_t(width);
    pc->height = uint32_t(height);
    pc->point_step = sizeof(float) * 3 + sizeof(uint32_t);
    pc->row_step = pc->point_step * uint32_t(width);
    pc->data = std::vector<uint8_t>(pc->row_step * size_t(height));
    PointCloudElement* pcIterator = reinterpret_cast<PointCloudElement*>(pc->data.data());

    for (decltype(points.size()) i = 0; i < count; i++) {
        auto const& point = points[i];
        pcIterator->data.x = static_cast<float>(point.x);
        pcIterator->data.y = static_cast<float>(point.y);
        pcIterator->data.z = static_cast<float>(point.z);
        RGB const& color = point.color;
        std::copy(color.begin(), color.end(), std::begin(pcIterator->data.rgb));
        pcIterator++;
    }
    return pc;
}
}

#endif // UTILS_HPP

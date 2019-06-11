#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

namespace sunshine {
using HSV = std::array<double, 3>;
using ARGB = std::array<uint8_t, 4>;

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
 * @brief toMat Converts two parallel vectors, of 3D integer xyz-coordinates and 1D values, respectively, to a dense matrix representation.
 * @param idxes 3D integer poses of the form [x1,y1,z1,...,xN,yN,zN] (size of 2*N)
 * @param values Scalar values corresponding to each pose of the form [f1,...,fN] (size of N)
 * @return cv::Mat where mat.at<MatValueType>(yI, xI) = fI
 */
template <typename IdxType, typename ValueType, typename MatValueType = ValueType>
cv::Mat toMat(std::vector<IdxType> const& idxes, std::vector<ValueType> const& values)
{
    assert(idxes.size() == (values.size() * 3) || idxes.size() == (values.size() * 2));
    auto const poseDim = static_cast<int>(values.size() / idxes.size());

    // Compute the required size of the matrix based on the largest (x,y) coordinate
    IdxType max_x = 0, max_y = 0;
    for (auto i = 0ul; i < values.size(); i++) {
        assert(idxes[i * poseDim] >= 0 && idxes[i * poseDim + 1] >= 0);
        max_x = std::max(max_x, idxes[i * poseDim]);
        max_y = std::max(max_y, idxes[i * poseDim + 1]);
    }
    cv::Mat mat(max_y + 1, max_x + 1, cvType<MatValueType>::value); // Don't forget to add 1 to each dimension (because 0-indexed)

    for (auto i = 0ul; i < values.size(); i++) {
        mat.at<MatValueType>(idxes[i * poseDim + 1], idxes[i * poseDim]) = values[i]; // mat.at(y,x) = z
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

ARGB inline HSV_TO_ARGB(HSV const& hsv, double alpha = 1)
{
    double const hueNorm = fmod(hsv[0], 360) / 60.;
    uint32_t const hueSectant = static_cast<uint32_t>(std::floor(hueNorm));
    double const hueOffset = hueNorm - hueSectant;

    double const &V = hsv[2], &S = hsv[1];
    uint8_t const v = uint8_t(255. * V);
    uint8_t const p = uint8_t(255. * V * (1 - S));
    uint8_t const q = uint8_t(255. * V * (1 - S * hueOffset));
    uint8_t const t = uint8_t(255. * V * (1 - S * (1 - hueOffset)));
    uint8_t const a = uint8_t(255. * alpha);

    switch (hueSectant) {
    case 0:
        return { a, v, t, p };
    case 1:
        return { a, q, v, p };
    case 2:
        return { a, p, v, t };
    case 3:
        return { a, p, q, v };
    case 4:
        return { a, t, p, v };
    case 5:
        return { a, v, p, q };
    default:
        throw std::invalid_argument("Should be unreachable.");
    }
}

/**
 * @brief createPointCloud
 * @param width
 * @param height
 * @param colorFieldName
 * @return
 * @warning Note that "rgba" is actually stored as ARGB (the PointCloud2 docs refers to the field as "unfortunately named")
 */
sensor_msgs::PointCloud2Ptr createPointCloud(uint32_t width, uint32_t height, std::string colorFieldName = "rgba")
{
    sensor_msgs::PointCloud2Ptr pc(new sensor_msgs::PointCloud2());
    sensor_msgs::PointField basePointField;
    basePointField.datatype = basePointField.FLOAT32;
    basePointField.count = 1;

    auto offset = 0u;
    for (auto const& field : { "x", "y", "z" }) {
        sensor_msgs::PointField pointField(basePointField);
        pointField.name = field;
        pointField.offset = offset;
        pc->fields.push_back(pointField);
        pc->point_step += sizeof(float);
        offset += sizeof(float);
    }

    if (!colorFieldName.empty()) {
        sensor_msgs::PointField colorField;
        colorField.datatype = colorField.UINT32;
        colorField.count = 1;
        colorField.offset = offset;
        colorField.name = colorFieldName;
        pc->fields.push_back(colorField);
        pc->point_step += sizeof(uint32_t);
    }

    pc->width = width;
    pc->height = height;
    pc->is_dense = true;
    pc->row_step = pc->point_step * pc->width;
    pc->data = std::vector<uint8_t>(pc->row_step * pc->height);
    return pc;
}

struct Point {
    double x, y, z;
    ARGB color;
};

union ARGBPointCloudElement {
    uint8_t bytes[sizeof(float) * 3 + sizeof(uint32_t)]; // to enforce size
    struct {
        float x;
        float y;
        float z;
        uint8_t argb[4]; // see note about rgba vs argb
    } data;
};

template <typename PointContainer>
sensor_msgs::PointCloud2Ptr toRGBAPointCloud(PointContainer const& points, std::string frame_id = "map")
{
    auto const count = points.size();
    uint32_t const height = 1, width = uint32_t(count);
    sensor_msgs::PointCloud2Ptr pc = createPointCloud(width, height, "rgba"); // Note that "rgba" is actually stored as ARGB (the documentation refers to the field as "unfortunately named")

    assert(pc->point_step == sizeof(ARGBPointCloudElement));

    pc->header.frame_id = frame_id;
    ARGBPointCloudElement* pcIterator = reinterpret_cast<ARGBPointCloudElement*>(pc->data.data());

    for (decltype(points.size()) i = 0; i < count; i++) {
        Point const& point = points[i];
        pcIterator->data.x = static_cast<float>(point.x);
        pcIterator->data.y = static_cast<float>(point.y);
        pcIterator->data.z = static_cast<float>(point.z);
        ARGB const& color = point.color;
        std::copy(color.crbegin(), color.crend(), std::begin(pcIterator->data.argb));
        pcIterator++;
    }
    return pc;
}
}

#endif // UTILS_HPP

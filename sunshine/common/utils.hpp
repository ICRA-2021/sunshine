#ifndef UTILS_HPP
#define UTILS_HPP

#include "colors.hpp"
#include <exception>
#include <opencv2/core.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>

namespace sunshine {

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
template <>
struct cvType<cv::Vec4b> {
    static int const value = CV_8UC4;
};
template <>
struct cvType<cv::Vec3b> {
    static int const value = CV_8UC3;
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

/**
 * @brief createPointCloud
 * @param width
 * @param height
 * @param colorFieldName
 * @return
 */
sensor_msgs::PointCloud2Ptr createPointCloud(uint32_t width, uint32_t height, std::string colorFieldName = "rgba", std_msgs::Header const& header = {})
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
    pc->header = header;
    return pc;
}

struct RGBAPoint {
    double x, y, z;
    RGBA color;
};

union RGBAPointCloudElement {
    uint8_t bytes[sizeof(float) * 3 + sizeof(uint32_t)]; // to enforce size
    struct {
        float x;
        float y;
        float z;
        std::array<uint8_t, 4> rgba;
    } data;
};

template <typename PointContainer>
sensor_msgs::PointCloud2Ptr toRGBAPointCloud(PointContainer const& points, std::string frame_id = "map")
{
    auto const count = points.size();
    uint32_t const height = 1, width = uint32_t(count);
    sensor_msgs::PointCloud2Ptr pc = createPointCloud(width, height, "rgba");

    if (pc->point_step != sizeof(RGBAPointCloudElement)) {
        throw std::invalid_argument("Point cloud point_step suggests fields are not xyz[rgba]");
    }

    pc->header.frame_id = frame_id;
    RGBAPointCloudElement* pcIterator = reinterpret_cast<RGBAPointCloudElement*>(pc->data.data());

    for (decltype(points.size()) i = 0; i < count; i++) {
        RGBAPoint const& point = points[i];
        pcIterator->data.x = static_cast<float>(point.x);
        pcIterator->data.y = static_cast<float>(point.y);
        pcIterator->data.z = static_cast<float>(point.z);
        pcIterator->data.rgba = static_cast<std::array<uint8_t, 4>>(point.color);
        pcIterator++;
    }
    return pc;
}

template <int COUNT, char DELIM='x'>
static inline std::array<double, COUNT> readNumbers(std::string const &str)
{
    std::array<double, COUNT> nums = {0};
    size_t idx = 0;
    for (size_t i = 1; i <= COUNT; i++) {
        auto const next = (i < COUNT) ? str.find(DELIM, idx) : str.size();
        if (next == std::string::npos) {
            throw std::invalid_argument("String '" + str + "' contains too few numbers!");
        }
        nums[i - 1] = std::stod(str.substr(idx, next));
        idx = next + 1;
    }
    return nums;
}

template <int POSE_DIM>
static inline std::array<double, POSE_DIM> computeCellSize(double cell_size_time, double cell_size_space)
{
    std::array<double, POSE_DIM> cell_size = {};
    cell_size[0] = cell_size_time;
    for (size_t i = 1; i < POSE_DIM; i++) {
        cell_size[i] = cell_size_space;
    }
    return cell_size;
}
}

#endif // UTILS_HPP

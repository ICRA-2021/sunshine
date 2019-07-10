#include <byteswap.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

using namespace cv;

static std::map<uint8_t, size_t> const pointfield_sizes = {
    { sensor_msgs::PointField::INT8, 1 },
    { sensor_msgs::PointField::UINT8, 1 },
    { sensor_msgs::PointField::INT16, 2 },
    { sensor_msgs::PointField::UINT16, 2 },
    { sensor_msgs::PointField::INT32, 4 },
    { sensor_msgs::PointField::UINT32, 4 },
    { sensor_msgs::PointField::FLOAT32, 4 },
    { sensor_msgs::PointField::FLOAT64, 8 }
};

struct ColorPoint {
    double x = 0;
    double y = 0;
    double z = 0;
    Vec4b bgra = { 0, 0, 0, 0 }; // bgra

    void readField(uint8_t const* const data, uint32_t point_step, std::string name, uint32_t offset, uint8_t datatype, bool is_bigendian)
    {
        assert(offset < point_step);
        if (name == "x") {
            this->x = this->convert<double>(data + offset, datatype, is_bigendian);
        } else if (name == "y") {
            this->y = this->convert<double>(data + offset, datatype, is_bigendian);
        } else if (name == "z") {
            this->z = this->convert<double>(data + offset, datatype, is_bigendian);
        } else if (name == "rgb" || name == "rgba") {
            uint32_t bgra = this->convert<uint32_t>(data + offset, datatype, is_bigendian); // not a typo... rgb means bgr... ¯\_(ツ)_/¯¯
            uint8_t* bgraData = reinterpret_cast<uint8_t*>(&bgra);
            this->bgra[0] = bgraData[0];
            this->bgra[1] = bgraData[1];
            this->bgra[2] = bgraData[2];
            if (name == "rgba") {
                this->bgra[3] = bgraData[3];
            }
        } else {
            throw std::runtime_error("Unrecognized point cloud field: " + name);
        }
    }

private:
    template <typename From, typename To>
    static inline To _convert(uint8_t const* const data, bool const is_bigendian)
    {
        From const* castedData = reinterpret_cast<From const*>(data);
        if (is_bigendian ^ (BYTE_ORDER == BIG_ENDIAN)) {
            throw std::logic_error("Endianness conversions are unsupported.");
//            switch (sizeof(From)) {
//            case 8:
//                return static_cast<To>(bswap_64(*castedData));
//            case 4:
//                return static_cast<To>(bswap_32(*castedData));
//            case 2:
//                return static_cast<To>(bswap_16(*castedData));
//            case 1: // intentional fallthrough
//            default:
//                return static_cast<To>(*castedData);
//            }
        }
        return static_cast<To>(*castedData);
    }

    template <typename T>
    static inline T convert(uint8_t const* const data, uint8_t type, bool const is_bigendian)
    {
        switch (type) {
        case sensor_msgs::PointField::INT8:
            return _convert<int8_t, T>(data, is_bigendian);
        case sensor_msgs::PointField::INT16:
            return _convert<int16_t, T>(data, is_bigendian);
        case sensor_msgs::PointField::INT32:
            return _convert<int32_t, T>(data, is_bigendian);
        case sensor_msgs::PointField::UINT8:
            return _convert<uint8_t, T>(data, is_bigendian);
        case sensor_msgs::PointField::UINT16:
            return _convert<uint16_t, T>(data, is_bigendian);
        case sensor_msgs::PointField::UINT32:
            return _convert<uint32_t, T>(data, is_bigendian);
        case sensor_msgs::PointField::FLOAT32:
            return _convert<float_t, T>(data, is_bigendian);
        case sensor_msgs::PointField::FLOAT64:
            return _convert<double_t, T>(data, is_bigendian);
        default:
            throw std::logic_error("Missing case for datatype " + std::to_string(type));
        }
    }
};

int main(int argc, char** argv)
{
    assert(argc >= 3);
    ros::init(argc, argv, "save_topic_map");
    ros::NodeHandle nh("~");
    auto const minWidth = nh.param<double>("minWidth", 0.);
    auto const minHeight = nh.param<double>("minHeight", 0.);
    std::string const input_topic(argv[1]);
    std::string const output_file(argv[2]);
    double cell_size = (argc >= 4) ? std::stod(argv[3]) : 1;
    ROS_INFO("Cell size: %f", cell_size);
    bool done = false;

    auto obsSub = nh.subscribe<sensor_msgs::PointCloud2>(input_topic, 1, [&done, output_file, cell_size, minWidth, minHeight](sensor_msgs::PointCloud2ConstPtr msg) {
        uint32_t const pointStep = msg->point_step;
        uint8_t const* const end = &(*msg->data.end());

        if (end != msg->data.data() + (msg->row_step * msg->height)) {
            throw std::logic_error("Point cloud data has unexpected size!");
        }

        std::vector<std::function<void(ColorPoint&, uint8_t const**)>> fieldParsers;
        {
            size_t pointSize = 0;
            bool const is_bigendian = msg->is_bigendian;
            for (auto const& field : msg->fields) {
                fieldParsers.push_back([field, pointStep, is_bigendian](ColorPoint& point, uint8_t const** data) {
                    point.readField(*data, pointStep, field.name, field.offset, field.datatype, is_bigendian);
                    static_assert(sizeof(**data) == 1, "Data ptr is not byte-sized!");
                });
                pointSize += pointfield_sizes.at(field.datatype);
            }
            if (pointSize != pointStep) {
                throw std::runtime_error("Failed to construct valid parser! Size mismatch");
            }
        }

        auto const extractPoint = [pointStep, fieldParsers](uint8_t const** data) {
            ColorPoint point;
            for (auto const& parser : fieldParsers) {
                parser(point, data);
            }
            *data += pointStep;
            return point;
        };

        double minX = std::numeric_limits<double>::infinity();
        double minY = std::numeric_limits<double>::infinity();
        double maxX = -std::numeric_limits<double>::infinity();
        double maxY = -std::numeric_limits<double>::infinity();
        {
            uint8_t const* dataPtr = msg->data.data();
            do {
                auto const point = extractPoint(&dataPtr);
                minX = min(minX, point.x);
                minY = min(minY, point.y);
                maxX = max(maxX, point.x);
                maxY = max(maxY, point.y);
            } while (dataPtr < end);
            if (dataPtr != end) {
                throw std::logic_error("Heartbleed bug when reading point cloud!");
            }
        }
        ROS_INFO("X in [%f, %f], Y in [%f, %f]", minX, maxX, minY, maxY);

        maxX = max(maxX, minX + minWidth);
        maxY = max(maxY, minY + minHeight);

        uint32_t const N = static_cast<uint32_t>(msg->data.size() / msg->point_step);
        int32_t const numRows = static_cast<int32_t>(std::round((maxY - minY) / cell_size)) + 1;
        int32_t const numCols = static_cast<int32_t>(std::round((maxX - minX) / cell_size)) + 1;
        int32_t const gaps = static_cast<int32_t>(static_cast<uint32_t>(numRows) * static_cast<uint32_t>(numCols) - N);
        ROS_INFO("Points: %u, Rows: %d, Cols: %d, Gaps: %d", N, numRows, numCols, gaps);
        if (gaps < 0) {
            ROS_WARN("%d fewer rows and columns than points - is your cell_size too large?", -gaps);
        }

        assert(numRows <= std::numeric_limits<int>::max());
        assert(numCols <= std::numeric_limits<int>::max());
        Mat img(numRows, numCols, CV_8UC4, Scalar(0));
        {
            uint8_t const* dataPtr = msg->data.data();
            do {
                auto const point = extractPoint(&dataPtr);
                Point const location(static_cast<int>(std::round((point.x - minX) / cell_size)),
                    static_cast<int>(std::round((maxY - point.y) / cell_size)));
                img.at<Vec4b>(location) = point.bgra;
            } while (dataPtr < end);
            if (dataPtr != end) {
                throw std::logic_error("Heartbleed bug when reading point cloud!");
            }
        }

        imwrite(output_file, img);
        done = true;
        return;
    });

    ros::topic::waitForMessage<sensor_msgs::PointCloud2>(input_topic, nh);
    while (!done) {
        ros::spinOnce();
    }
}

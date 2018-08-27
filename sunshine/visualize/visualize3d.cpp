#include "utils.hpp"
#include <cstdlib>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sunshine_msgs/WordObservation.h>

using namespace sensor_msgs;
using namespace sunshine_msgs;

using HSV = std::array<double, 3>;
using RGB = std::array<uint8_t, 3>;
using WordType = WordObservation::_words_type::value_type;

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

sensor_msgs::PointCloud2Ptr toPointCloud(WordObservationConstPtr const& observation,
    std::function<RGB(WordType)> const& wordColorFunc,
    std::string frame_id = "/map")
{
    uint32_t const height = 1, width = uint32_t(observation->words.size());

    sensor_msgs::PointCloud2Ptr pc = PointCloud2Ptr(new PointCloud2());
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

    for (size_t i = 0; i < observation->words.size(); i++) {
        geometry_msgs::Point point;
        transformPose(point, observation->word_pose, i * 3, observation->observation_transform);
        auto const& color = wordColorFunc(observation->words[i]);

        pcIterator->data.x = static_cast<float>(point.x);
        pcIterator->data.y = static_cast<float>(point.y);
        pcIterator->data.z = static_cast<float>(point.z);
        std::copy(color.begin(), color.end(), std::begin(pcIterator->data.rgb));
        pcIterator++;
    }
    return pc;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "visualize3d");
    ros::NodeHandle nh("~");
    auto const saturation = nh.param<double>("saturation", 1.);
    auto const value = nh.param<double>("value", 1.);
    auto const input_topic = nh.param<std::string>("observation_topic", "/words");
    auto const output_topic = nh.param<std::string>("output_topic", "/word_cloud");

    std::map<WordType, RGB> colorMap;
    std::map<double, WordType> hueMap;
    auto const& colorFunc = [&](WordType word) {
        auto const colorIter = colorMap.find(word);
        if (colorIter != colorMap.end()) {
            return colorIter->second;
        }

        double hue = double(rand()) * 360. / double(RAND_MAX);
        if (hueMap.size() == 1) {
            hue = fmod(hueMap.begin()->first + 180., 360.);
        } else if (hueMap.size() > 1) {
            auto const& upper = (hueMap.upper_bound(hue) == hueMap.end())
                ? hueMap.lower_bound(0)->first + 360.
                : hueMap.upper_bound(hue)->first;
            auto const& lower = (hueMap.lower_bound(hue) == hueMap.end())
                ? hueMap.crbegin()->first
                : (--hueMap.lower_bound(hue))->first;
            hue = fmod((lower + upper) / 2., 360.);
        }
        hueMap.insert({ hue, word });

        auto const rgb = hsvToRgb({ hue, saturation, value });
        colorMap.insert({ word, rgb });
        return rgb;
    };

    ros::Publisher pcPub = nh.advertise<PointCloud2>(output_topic, 1);
    ros::Subscriber wordSub = nh.subscribe<WordObservation>(input_topic, 1, [&colorFunc, &pcPub](WordObservationConstPtr const& msg) {
        auto pc = toPointCloud(msg, colorFunc);
        pcPub.publish(pc);
    });

    ros::spin();
}

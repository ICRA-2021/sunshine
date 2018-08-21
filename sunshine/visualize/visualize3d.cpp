#include "utils.hpp"
#include <cstdlib>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sunshine_msgs/TopicMap.h>
#include <sunshine_msgs/WordObservation.h>

using namespace sunshine;
using namespace sensor_msgs;
using namespace sunshine_msgs;

using WordType = WordObservation::_words_type::value_type;

static std::map<WordType, RGB> colorMap;
static std::map<double, WordType> hueMap;

static inline RGB colorForWord(int32_t word, double saturation = 1, double value = 1)
{

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
}

struct Point {
    double x, y, z;
    RGB color;
};

class WordObservationPoints {
    WordObservation const& wordObservations;

public:
    WordObservationPoints(WordObservationConstPtr wordObservations)
        : wordObservations(*wordObservations)
    {
    }

    size_t size() const
    {
        return wordObservations.words.size();
    }

    Point operator[](size_t idx) const
    {
        Point p;
        p.x = wordObservations.word_pose[idx * 3];
        p.y = wordObservations.word_pose[idx * 3 + 1];
        p.z = wordObservations.word_pose[idx * 3 + 2];
        p.color = colorForWord(wordObservations.words[idx]);
        return p;
    }
};

class TopicMapPoints {
    TopicMap const& topicMap;
    double const max_ppx;

public:
    TopicMapPoints(TopicMapConstPtr topicMap)
        : topicMap(*topicMap)
        , max_ppx((size() > 0) ? *std::max_element(topicMap->cell_ppx.begin(), topicMap->cell_ppx.end()) : 0)
    {
    }

    size_t size() const
    {
        return topicMap.cell_topics.size();
    }

    Point operator[](size_t idx) const
    {
        Point p;
        p.x = topicMap.cell_poses[idx * 3];
        p.y = topicMap.cell_poses[idx * 3 + 1];
        p.z = topicMap.cell_poses[idx * 3 + 2];
        p.color = colorForWord(topicMap.cell_topics[idx], 1, topicMap.cell_ppx[idx] / max_ppx);
        return p;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "visualize3d");
    ros::NodeHandle nh("~");
    auto const input_topic = nh.param<std::string>("input_topic", "/words");
    auto const output_topic = nh.param<std::string>("output_topic", "/word_cloud");
    auto const input_type = nh.param<std::string>("input_type", "TopicMap");

    ros::Publisher pcPub = nh.advertise<PointCloud2>(output_topic, 1);
    ros::Subscriber obsSub;
    if (input_type == "WordObservation") {
        obsSub = nh.subscribe<WordObservation>(input_topic, 1, [&pcPub](WordObservationConstPtr const& msg) {
            auto pc = toPointCloud<WordObservationPoints>(WordObservationPoints(msg));
            pcPub.publish(pc);
        });
    } else if (input_type == "TopicMap") {
        obsSub = nh.subscribe<TopicMap>(input_topic, 1, [&pcPub](sunshine_msgs::TopicMapConstPtr const& msg) {
            auto pc = toPointCloud(TopicMapPoints(msg));
            pcPub.publish(pc);
        });
    }

    ros::spin();
}

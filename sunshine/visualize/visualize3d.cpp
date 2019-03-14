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

class Visualize3d {
    std::map<WordType, double> hueMapBackward;
    std::map<double, WordType> hueMapForward;
    ros::Publisher pcPub, ppxPub;
    ros::Subscriber obsSub;
    double ppx_display_factor, z_offset;

public:
    Visualize3d(ros::NodeHandle& nh);

    inline ARGB colorForWord(int32_t word, double saturation = 1, double value = 1, double alpha = 1)
    {
        auto const hueIter = hueMapBackward.find(word);
        if (hueIter != hueMapBackward.end()) {
            return HSV_TO_ARGB({ hueIter->second, saturation, value });
        }

        double hue = double(rand()) * 360. / double(RAND_MAX);
        if (hueMapForward.size() == 1) {
            hue = fmod(hueMapForward.begin()->first + 180., 360.);
        } else if (hueMapForward.size() > 1) {
            auto const& upper = (hueMapForward.upper_bound(hue) == hueMapForward.end())
                ? hueMapForward.lower_bound(0)->first + 360.
                : hueMapForward.upper_bound(hue)->first;
            auto const& lower = (hueMapForward.lower_bound(hue) == hueMapForward.end())
                ? hueMapForward.crbegin()->first
                : (--hueMapForward.lower_bound(hue))->first;
            hue = fmod((lower + upper) / 2., 360.);
        }
        hueMapForward.insert({ hue, word });
        hueMapBackward.insert({ word, hue });

        return HSV_TO_ARGB({ hue, saturation, value }, alpha);
    }

    double perplexity_display_factor() const
    {
        return ppx_display_factor;
    }

    double get_z_offset() const
    {
        return z_offset;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "visualize3d");
    ros::NodeHandle nh("~");
    Visualize3d visualizer(nh);
    ros::spin();
}

class WordObservationPoints {
    Visualize3d* cls;
    WordObservation const& wordObservations;

public:
    WordObservationPoints(Visualize3d* cls, WordObservationConstPtr wordObservations)
        : cls(cls)
        , wordObservations(*wordObservations)
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
        p.color = cls->colorForWord(wordObservations.words[idx]);
        return p;
    }
};

template <bool show_ppx>
class TopicMapPoints {
    Visualize3d* cls;
    TopicMap const& topicMap;
    double const max_ppx;

public:
    TopicMapPoints(Visualize3d* cls, TopicMapConstPtr topicMap)
        : cls(cls)
        , topicMap(*topicMap)
        , max_ppx((show_ppx && size() > 0) ? *std::max_element(topicMap->cell_ppx.begin(), topicMap->cell_ppx.end()) : 1)
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
        p.z = topicMap.cell_poses[idx * 3 + 2] + cls->get_z_offset();
        p.color = cls->colorForWord(topicMap.cell_topics[idx], 1, 1 + show_ppx * cls->perplexity_display_factor() * (topicMap.cell_ppx[idx] / max_ppx - 1));
        return p;
    }
};

class PerplexityPoints {
    Visualize3d* cls;
    TopicMap const& topicMap;
    double const max_ppx;

public:
    PerplexityPoints(Visualize3d* cls, TopicMapConstPtr topicMap)
        : cls(cls)
        , topicMap(*topicMap)
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
        p.z = topicMap.cell_poses[idx * 3 + 2] + cls->get_z_offset();
        uint8_t const relativePpx = uint8_t(255. * topicMap.cell_ppx[idx] / max_ppx);
        p.color = { relativePpx, relativePpx, 0, 0 };
        return p;
    }
};

Visualize3d::Visualize3d(ros::NodeHandle& nh)
{
    auto const input_topic = nh.param<std::string>("input_topic", "/words");
    auto const output_topic = nh.param<std::string>("output_topic", "/word_cloud");
    auto const input_type = nh.param<std::string>("input_type", "TopicMap");
    auto const ppx_topic = nh.param<std::string>("ppx_topic", "/ppx_cloud");
    auto const world_frame = nh.param<std::string>("world_frame", "map");
    z_offset = nh.param<double>("z_offset", 0.1);
    ppx_display_factor = nh.param<double>("ppx_display_factor", 0.5);

    pcPub = nh.advertise<PointCloud2>(output_topic, 1);
    if (ppx_topic != output_topic) {
        ppxPub = nh.advertise<PointCloud2>(ppx_topic, 1);
    }

    if (input_type == "WordObservation") {
        obsSub = nh.subscribe<WordObservation>(input_topic, 1, [this, world_frame](WordObservationConstPtr const& msg) {
            auto pc = toRGBAPointCloud<WordObservationPoints>(WordObservationPoints(this, msg), world_frame);
            pcPub.publish(pc);
        });
    } else if (input_type == "TopicMap") {
        obsSub = nh.subscribe<TopicMap>(input_topic, 1, [this, ppx_topic, output_topic, world_frame](sunshine_msgs::TopicMapConstPtr const& msg) {
            if (ppx_topic == output_topic) {
                auto pc = toRGBAPointCloud(TopicMapPoints<true>(this, msg), world_frame);
                pcPub.publish(pc);
            } else {
                auto topicPc = toRGBAPointCloud(TopicMapPoints<false>(this, msg), world_frame);
                pcPub.publish(topicPc);
                auto ppxPc = toRGBAPointCloud(PerplexityPoints(this, msg), world_frame);
                ppxPub.publish(ppxPc);
            }
        });
    }
}

#include "utils.hpp"
#include "word_coloring.hpp"
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
    WordColorMap<WordType> colorMap;
    ros::Publisher pcPub, ppxPub;
    ros::Subscriber obsSub;
    double ppx_display_factor, z_offset;

public:
    Visualize3d(ros::NodeHandle& nh);

    inline ARGB colorForWord(int32_t word, double saturation = 1, double value = 1, double alpha = 1)
    {
        return colorMap.colorForWord(word, saturation, value, alpha);
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
    auto const output_frame = nh.param<std::string>("output_frame", "map");
    z_offset = nh.param<double>("z_offset", 0.1);
    ppx_display_factor = nh.param<double>("ppx_display_factor", 0.5);

    pcPub = nh.advertise<PointCloud2>(output_topic, 1);
    if (ppx_topic != output_topic) {
        ppxPub = nh.advertise<PointCloud2>(ppx_topic, 1);
    }

    if (input_type == "WordObservation") {
        obsSub = nh.subscribe<WordObservation>(input_topic, 1, [this, output_frame](WordObservationConstPtr const& msg) {
            auto pc = toRGBAPointCloud<WordObservationPoints>(WordObservationPoints(this, msg), output_frame);
            pcPub.publish(pc);
        });
    } else if (input_type == "TopicMap") {
        obsSub = nh.subscribe<TopicMap>(input_topic, 1, [this, ppx_topic, output_topic, output_frame](sunshine_msgs::TopicMapConstPtr const& msg) {
            if (ppx_topic == output_topic) {
                auto pc = toRGBAPointCloud(TopicMapPoints<true>(this, msg), output_frame);
                pcPub.publish(pc);
            } else {
                auto topicPc = toRGBAPointCloud(TopicMapPoints<false>(this, msg), output_frame);
                pcPub.publish(topicPc);
                auto ppxPc = toRGBAPointCloud(PerplexityPoints(this, msg), output_frame);
                ppxPub.publish(ppxPc);
            }
        });
    }
}

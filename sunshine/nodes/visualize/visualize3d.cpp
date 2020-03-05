#include "ros_utils.hpp"
#include "word_coloring.hpp"
#include <cstdlib>
#include <map>
#include <utility>
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
    Visualize3d(ros::NodeHandle &nh);

    inline RGBA colorForWord(int32_t word, double saturation = 1, double value = 1, double alpha = 1) {
        return colorMap.colorForWord(word, saturation, value, alpha);
    }

    double perplexity_display_factor() const {
        return ppx_display_factor;
    }

    double get_z_offset() const {
        return z_offset;
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "visualize3d");
    ros::NodeHandle nh("~");
    Visualize3d visualizer(nh);
    ros::spin();
}

class WordObservationPoints {
    Visualize3d *cls;
    WordObservation const &wordObservations;

public:
    WordObservationPoints(Visualize3d *cls, WordObservationConstPtr wordObservations)
            : cls(cls), wordObservations(*wordObservations) {
    }

    size_t size() const {
        return wordObservations.words.size();
    }

    RGBAPoint operator[](size_t idx) const {
        auto const &poses = wordObservations.word_pose;
        return {poses[idx * 3], poses[idx * 3 + 1], poses[idx * 3 + 2], cls->colorForWord(wordObservations.words[idx])};
    }
};

template<typename T>
class MapPointsContainer {
protected:
    Visualize3d *cls;
    TopicMapConstPtr topicMap;
    double const max_ppx;
    std::array<double, 4> const cell_size;

    MapPointsContainer(Visualize3d *cls, TopicMapConstPtr topicMap, std::array<double, 4> cell_size)
            : cls(cls), topicMap(topicMap),
              max_ppx((!topicMap->cell_ppx.empty()) ? *std::max_element(topicMap->cell_ppx.begin(),
                                                                        topicMap->cell_ppx.end()) : 1),
              cell_size(cell_size) {
        assert(topicMap);
    }

public:
    size_t size() const {
        return static_cast<T const *>(this)->_size();
    }

    RGBAPoint operator[](size_t idx) const {
        return static_cast<T const *>(this)->get_point(idx);
    }
};

template<bool show_ppx>
class TopicMapPoints : public MapPointsContainer<TopicMapPoints<show_ppx>> {
public:
    TopicMapPoints(Visualize3d *cls, TopicMapConstPtr topicMap, std::array<double, 4> cell_size)
            : MapPointsContainer<TopicMapPoints<show_ppx>>(cls, std::move(topicMap), cell_size) {
    }

    size_t _size() const {
        return this->topicMap->cell_topics.size();
    }

    RGBAPoint get_point(size_t idx) const {
        if (idx >= this->topicMap->cell_topics.size()) throw std::out_of_range("Index is out of range");
        auto const &poses = this->topicMap->cell_poses;
        return {poses[idx * 3] + this->cell_size[1] / 2, poses[idx * 3 + 1] + this->cell_size[2] / 2,
                poses[idx * 3 + 2] + this->cell_size[3] / 2 + this->cls->get_z_offset(),
                this->cls->colorForWord(this->topicMap->cell_topics[idx], 1,
                                        1 + ((show_ppx) ? this->cls->perplexity_display_factor() *
                                                          (this->topicMap->cell_ppx[idx] / this->max_ppx - 1) : 0))};
    }
};

class PerplexityPoints : public MapPointsContainer<PerplexityPoints> {
public:
    PerplexityPoints(Visualize3d *cls, TopicMapConstPtr topicMap, std::array<double, 4> cell_size)
            : MapPointsContainer<PerplexityPoints>(cls, std::move(topicMap), cell_size) {
    }

    size_t _size() const {
        return this->topicMap->cell_ppx.size();
    }

    RGBAPoint get_point(size_t idx) const {
        if (idx >= topicMap->cell_ppx.size()) throw std::out_of_range("Index is out of range");
        auto const &poses = topicMap->cell_poses;
        auto const relativePpx = uint8_t(255. * topicMap->cell_ppx[idx] / max_ppx);
        return {poses[idx * 3] + cell_size[1] / 2, poses[idx * 3 + 1] + cell_size[2] / 2,
                poses[idx * 3 + 2] + cell_size[3] / 2 + cls->get_z_offset(), {relativePpx, relativePpx, 0, 0}};
    }
};

Visualize3d::Visualize3d(ros::NodeHandle &nh) {
    auto const input_topic = nh.param<std::string>("input_topic", "/words");
    auto const output_topic = nh.param<std::string>("output_topic", "/word_cloud");
    auto const input_type = nh.param<std::string>("input_type", "TopicMap");
    auto const ppx_topic = nh.param<std::string>("ppx_topic", "/ppx_cloud");
    auto const output_frame = nh.param<std::string>("output_frame", "map");
    auto const cell_size_string = nh.param<std::string>("cell_size", "");
    auto const cell_size_space = nh.param<double>("cell_space", 1);
    auto const cell_size_time = nh.param<double>("cell_time", 1);
    z_offset = nh.param<double>("z_offset", 0.1);
    ppx_display_factor = nh.param<double>("ppx_display_factor", 0.5);

    std::array<double, 4> cell_size = {0};
    if (!cell_size_string.empty()) {
        cell_size = readNumbers<4, 'x'>(cell_size_string);
    } else {
        cell_size = computeCellSize<4>(cell_size_time, cell_size_space);
    }

    pcPub = nh.advertise<PointCloud2>(output_topic, 1);
    if (ppx_topic != output_topic) {
        ppxPub = nh.advertise<PointCloud2>(ppx_topic, 1);
    }

    if (input_type == "WordObservation") {
        obsSub = nh.subscribe<WordObservation>(input_topic, 1,
                                               [this, output_frame](WordObservationConstPtr const &msg) {
                                                   auto pc = toRGBAPointCloud<WordObservationPoints>(
                                                           WordObservationPoints(this, msg), output_frame);
                                                   pcPub.publish(pc);
                                               });
    } else if (input_type == "TopicMap") {
        obsSub = nh.subscribe<TopicMap>(input_topic, 1, [=](sunshine_msgs::TopicMapConstPtr const &msg) {
            if (ppx_topic == output_topic) {
                auto pc = toRGBAPointCloud(TopicMapPoints<true>(this, msg, cell_size), output_frame);
                pcPub.publish(pc);
            } else {
                auto topicPc = toRGBAPointCloud(TopicMapPoints<false>(this, msg, cell_size), output_frame);
                pcPub.publish(topicPc);
                auto ppxPc = toRGBAPointCloud(PerplexityPoints(this, msg, cell_size), output_frame);
                ppxPub.publish(ppxPc);
            }
        });
    }
}

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "sunshine/common/adrost_utils.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/rosbag_utils.hpp"
#include "sunshine/depth_adapter.hpp"
#include "sunshine/observation_transform_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"
#include "sunshine/semantic_label_adapter.hpp"
#include "sunshine/visual_word_adapter.hpp"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sunshine/common/ros_conversions.hpp>
#include <tf2_msgs/TFMessage.h>
#include <utility>

using namespace sunshine;

class RobotSim {
    std::string const name;
    BagIterator bagIter;
    VisualWordAdapter visualWordAdapter;
    ROSTAdapter<4, double, double> rostAdapter;
    SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>> segmentationAdapter;
    ObservationTransformAdapter<WordDepthAdapter::Output> wordTransformAdapter;
    ObservationTransformAdapter<ImageDepthAdapter::Output> imageTransformAdapter;
    WordDepthAdapter wordDepthAdapter;
    ImageDepthAdapter imageDepthAdapter;
    std::unique_ptr<ImageObservation> lastRgb, lastSegmentation;
    std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>> segmentation;
    double depth_timestamp = -1;

    bool imageCallback(sensor_msgs::Image::ConstPtr image) {
        lastRgb = std::make_unique<ImageObservation>(fromRosMsg(image));
        if (lastRgb && lastRgb->timestamp == depth_timestamp) {
            lastRgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;
            return true;
        }
        return false;
    }

    bool segmentationCallback(sensor_msgs::Image::ConstPtr image) {
        lastSegmentation = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (lastSegmentation && lastSegmentation->timestamp == depth_timestamp) {
            segmentation = lastSegmentation >> imageDepthAdapter >> imageTransformAdapter >> segmentationAdapter;
            return true;
        }
        return false;
    }

    bool depthCallback(sensor_msgs::PointCloud2::ConstPtr const &msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pc);

        wordDepthAdapter.updatePointCloud(pc);
        imageDepthAdapter.updatePointCloud(pc);
        depth_timestamp = msg->header.stamp.toSec();
        bool handled    = false;
        if (lastRgb && lastRgb->timestamp == depth_timestamp) {
            lastRgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;
            handled = true;
        }
        if (lastSegmentation && lastSegmentation->timestamp == depth_timestamp) {
            segmentation = lastSegmentation >> imageDepthAdapter >> imageTransformAdapter >> segmentationAdapter;
            handled      = true;
        }
        return handled;
    };

    bool transformCallback(tf2_msgs::TFMessage::ConstPtr const &tfMsg) {
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
        }
        return false;
    }

  public:
    template<typename ParamServer>
    RobotSim(std::string name,
             std::string const &bagfile,
             ParamServer const &parameters,
             std::string const &image_topic,
             std::string const &depth_topic,
             std::string const &segmentation_topic)
        : name(std::move(name))
        , bagIter(bagfile)
        , visualWordAdapter(&parameters)
        , rostAdapter(&parameters)
        , segmentationAdapter(&parameters, true)
        , wordTransformAdapter(&parameters)
        , imageTransformAdapter(&parameters) {
        bagIter.add_callback<sensor_msgs::Image>(image_topic, [this](auto const &msg) { return this->imageCallback(msg); });
        bagIter.add_callback<sensor_msgs::Image>(segmentation_topic, [this](auto const &msg) { return this->segmentationCallback(msg); });
        bagIter.add_callback<sensor_msgs::PointCloud2>(depth_topic, [this](auto const &msg) { return this->depthCallback(msg); });
        bagIter.add_callback<tf2_msgs::TFMessage>("/tf", [this](auto const &msg) { return this->transformCallback(msg); });
        bagIter.set_logging(true);
    }

    Phi getTopicModel(bool wait_for_refine = false) const {
        if (wait_for_refine) rostAdapter.wait_for_processing(false);
        auto token = rostAdapter.get_rost().get_read_token();
        return rostAdapter.get_topic_model(*token);
    }

    auto getMap(bool wait_for_refine = true) const {
        if (wait_for_refine) rostAdapter.wait_for_processing(false);
        auto token = rostAdapter.get_rost().get_read_token();
        return rostAdapter.get_map(*token);
    }

    auto getGTMap() const {
        double const timestamp = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        segmentation->id       = ros::Time(timestamp).sec;
        return segmentation;
    }

    /**
     *
     * @return false if finished, true if there are more messages to simulate
     */
    bool next() {
        return !bagIter.play(false);
    }

    std::string getName() const {
        return name;
    }
};

template<uint32_t PoseDim = 4, typename Container>
std::unique_ptr<Segmentation<int, PoseDim, int, double>> merge(Container const &segmentations,
                                                               std::vector<std::vector<int>> const &lifting = {}) {
    auto merged = std::make_unique<Segmentation<int, PoseDim, int, double>>(segmentations[0]->frame, segmentations[0]->timestamp,
                                                                            segmentations[0]->id, segmentations[0]->cell_size,
                                                                            std::vector<int>(), segmentations[0]->observation_poses);
    for (auto i = 1; i < segmentations.size(); ++i) {
        merged->observation_poses.insert(merged->observation_poses.end(), segmentations[i]->observation_poses.begin(),
                                         segmentations[i]->observation_poses.end());
    }
    if (lifting.empty()) {
        for (auto const &map : segmentations) {
            if constexpr (std::is_integral_v<typename decltype(map->observations)::value_type>) {
                merged->observations.insert(merged->observations.end(), map->observations.begin(), map->observations.end());
            } else {
                std::transform(map->observations.begin(), map->observations.end(), std::back_inserter(merged->observations),
                               argmax<std::vector<int>>);
            }
        }
    } else {
        for (auto i = 0; i < segmentations.size(); ++i) {
            if constexpr (std::is_integral_v<typename decltype(segmentations[i]->observations)::value_type>) {
                std::transform(segmentations[i]->observations.begin(), segmentations[i]->observations.end(),
                               std::back_inserter(merged->observations), [i, &lifting](int const &obs) { return lifting[i][obs]; });
            } else {
                std::transform(segmentations[i]->observations.begin(), segmentations[i]->observations.end(),
                               std::back_inserter(merged->observations),
                               [i, &lifting](std::vector<int> const &obs) { return lifting[i][argmax(obs)]; });
            }
        }
    }
    return merged;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "multi_agent_sim");
    if (argc < 5) {
        std::cerr << "Usage: ./sunshine_eval image_topic depth_topic segmentation_topic bagfiles..." << std::endl;
        return 1;
    }
    std::string const image_topic_name(argv[1]);
    std::string const depth_topic_name(argv[2]);
    std::string const segmentation_topic_name(argv[3]);

    ros::NodeHandle nh("~");
    std::vector<std::unique_ptr<RobotSim>> robots;
    std::vector<ros::Publisher> map_pubs;
    for (auto i = 4; i < argc; ++i) {
        robots.emplace_back(std::make_unique<RobotSim>(std::to_string(i - 4), std::string(argv[i]), nh, image_topic_name,
                                                       depth_topic_name, segmentation_topic_name));
        map_pubs.push_back(nh.advertise<sunshine_msgs::TopicMap>("/" + robots.back()->getName() + "/map", 0));
    }
    ros::Publisher naive_map_pub  = nh.advertise<sunshine_msgs::TopicMap>("/naive_map", 0);
    ros::Publisher merged_map_pub = nh.advertise<sunshine_msgs::TopicMap>("/merged_map", 0);
    ros::Publisher gt_map_pub     = nh.advertise<sunshine_msgs::TopicMap>("/gt_map", 0);

    auto const fetch_new_topic_models = [&robots]() {
        std::vector<Phi> topic_models;
        for (auto const &robot : robots) { topic_models.push_back(robot->getTopicModel(true)); }
        return topic_models;
    };

    bool active = true;
    auto start  = std::chrono::steady_clock::now();
    while (active && ros::ok()) {
        active = false;
        for (auto &robot : robots) {
            auto const robot_active = robot->next();
            active                  = active || robot_active;
        }

        auto topic_models          = fetch_new_topic_models();
        auto const correspondences = match_topics("clear-l1", {topic_models.begin(), topic_models.end()});
        match_scores const scores(topic_models, correspondences.lifting, normed_dist_sq<double>);

        auto const correspondences_hungarian = match_topics("hungarian-l1", {topic_models.begin(), topic_models.end()});
        match_scores const scores_hungarian(topic_models, correspondences.lifting, normed_dist_sq<double>);

        uint32_t matched = 0;
        for (auto const size : scores.cluster_sizes) { matched += (size > 1); }
        std::cout << "Matched: " << (matched - 1) << "/" << (correspondences.num_unique - 1) << std::endl;
        std::cout << "Refine time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << std::endl;
        start = std::chrono::steady_clock::now();

        std::vector<std::unique_ptr<Segmentation<int, 4, int, double>>> segmentations;
        std::vector<std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>>> gt_segmentations;
        for (auto i = 0; i < robots.size(); ++i) {
            segmentations.emplace_back(robots[i]->getMap());
            gt_segmentations.push_back(robots[i]->getGTMap());
            map_pubs[i].publish(toRosMsg(*segmentations.back()));
        }
        naive_map_pub.publish(toRosMsg(*merge(segmentations)));
        merged_map_pub.publish(toRosMsg(*merge(segmentations, correspondences.lifting)));
        gt_map_pub.publish(toRosMsg(*merge<3>(gt_segmentations)));
    }

    return 0;
}

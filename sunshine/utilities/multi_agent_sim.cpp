#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>

#include "sunshine/common/rosbag_utils.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/visual_word_adapter.hpp"
#include "sunshine/semantic_label_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"
#include "sunshine/observation_transform_adapter.hpp"
#include "sunshine/depth_adapter.hpp"
#include "sunshine/common/adrost_utils.hpp"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_msgs/TFMessage.h>
#include <sunshine/common/ros_conversions.hpp>

using namespace sunshine;

class RobotSim
{
    BagIterator bagIter;
    VisualWordAdapter visualWordAdapter;
    ROSTAdapter<4, double, double> rostAdapter;
    SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>> segmentationAdapter;
    ObservationTransformAdapter<WordDepthAdapter::Output> wordTransformAdapter;
    ObservationTransformAdapter<ImageDepthAdapter::Output> imageTransformAdapter;
    WordDepthAdapter wordDepthAdapter;
    ImageDepthAdapter imageDepthAdapter;
    std::unique_ptr<ImageObservation> rgb, segmentation;
    double depth_timestamp = -1;

    bool imageCallback(sensor_msgs::Image::ConstPtr image) {
        rgb = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            rgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;
            return true;
        }
        return false;
    }

    bool segmentationCallback(sensor_msgs::Image::ConstPtr image) {
        segmentation = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            rgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;
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
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            rgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;
            return true;
        }
        return false;
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
    RobotSim(std::string const &bagfile,
             sunshine::Parameters const &parameters,
             std::string const &image_topic,
             std::string const &depth_topic,
             std::string const &segmentation_topic)
            : bagIter(bagfile),
              visualWordAdapter(&parameters),
              rostAdapter(&parameters),
              segmentationAdapter(&parameters),
              wordTransformAdapter(&parameters),
              imageTransformAdapter(&parameters) {
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

    /**
     *
     * @return false if finished, true if there are more messages to simulate
     */
    bool next() {
        return !bagIter.play(false);
    }
};

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: ./sunshine_eval image_topic depth_topic segmentation_topic bagfiles..." << std::endl;
        return 1;
    }
    std::string const image_topic_name(argv[1]);
    std::string const depth_topic_name(argv[2]);
    std::string const segmentation_topic_name(argv[3]);
    bool const use_orb = true;
    Parameters const parameters({{"alpha",               0.6722},
                                 {"beta",                0.078899},
                                 {"gamma",               0.00000139},
                                 {"cell_space",          1.211},
                                 {"cell_time",           3600.0},
                                 {"use_orb",             use_orb},
                                 {"V",                   436 + use_orb * 15000},
                                 {"min_obs_refine_time", 300}});

    std::vector<std::unique_ptr<RobotSim>> robots;
    for (auto i = 4; i < argc; ++i) {
        robots.emplace_back(std::make_unique<RobotSim>(std::string(argv[i]),
                                                       parameters,
                                                       image_topic_name,
                                                       depth_topic_name,
                                                       segmentation_topic_name));
    }

    auto const fetch_new_topic_models = [&robots]() {
        std::vector<Phi> topic_models;
        for (auto const &robot : robots) {
            topic_models.push_back(robot->getTopicModel(true));
        }
        return topic_models;
    };

    bool active = true;
    auto start = std::chrono::steady_clock::now();
    while (active) {
        active = false;
        for (auto &robot : robots) {
            auto const robot_active = robot->next();
            active = active || robot_active;
        }

        auto topic_models = fetch_new_topic_models();
        auto const correspondences = match_topics("clear-l1", {topic_models.begin(), topic_models.end()});
        match_scores const scores(topic_models, correspondences.lifting, normed_dist_sq<double>);
        uint32_t unmatched = 0;
        for (auto const size : scores.cluster_sizes) {
            unmatched += (size == 1);
        }
        std::cout << "Unmatched: " << unmatched << std::endl;
        std::cout << "Refine time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << std::endl;
        start = std::chrono::steady_clock::now();
    }

    return 0;
}



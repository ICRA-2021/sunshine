#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "sunshine/common/adrost_utils.hpp"
#include "sunshine/common/csv.hpp"
#include "sunshine/common/metric.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/rosbag_utils.hpp"
#include "sunshine/depth_adapter.hpp"
#include "sunshine/observation_transform_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"
#include "sunshine/benchmark.hpp"
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
    std::shared_ptr<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>> segmentationAdapter;
    ObservationTransformAdapter<WordDepthAdapter::Output> wordTransformAdapter;
    ObservationTransformAdapter<ImageDepthAdapter::Output> imageTransformAdapter;
    WordDepthAdapter wordDepthAdapter;
    ImageDepthAdapter imageDepthAdapter;
    std::unique_ptr<ImageObservation> lastRgb, lastSegmentation;
    std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>> segmentation;
    double depth_timestamp = -1;
    bool transform_found = false;

    bool imageCallback(sensor_msgs::Image::ConstPtr image) {
        lastRgb = std::make_unique<ImageObservation>(fromRosMsg(image));
        if (transform_found && lastRgb && lastRgb->timestamp == depth_timestamp) {
            lastRgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;
            return true;
        }
        return false;
    }

    bool segmentationCallback(sensor_msgs::Image::ConstPtr image) {
        lastSegmentation = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (transform_found && lastSegmentation && lastSegmentation->timestamp == depth_timestamp) {
            segmentation = lastSegmentation >> imageDepthAdapter >> imageTransformAdapter >> *segmentationAdapter;
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
        bool handled = false;
        if (transform_found && lastRgb && lastRgb->timestamp == depth_timestamp) {
            lastRgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;
            handled = true;
        }
        if (transform_found && lastSegmentation && lastSegmentation->timestamp == depth_timestamp) {
            segmentation = lastSegmentation >> imageDepthAdapter >> imageTransformAdapter >> *segmentationAdapter;
            handled = true;
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
        transform_found = true;
        return false;
    }

  public:
    template<typename ParamServer>
    RobotSim(std::string name,
             std::string const &bagfile,
             ParamServer const &parameters,
             std::string const &image_topic,
             std::string const &depth_topic,
             std::string const &segmentation_topic,
             std::shared_ptr<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>> shared_adapter = nullptr)
            : name(std::move(name)),
              bagIter(bagfile),
              visualWordAdapter(&parameters),
              rostAdapter(&parameters),
              segmentationAdapter((shared_adapter)
                                  ? shared_adapter
                                  : std::make_shared<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>>(&parameters,
                                                                                                                            true)),
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

    auto getMap(bool wait_for_refine = true) const {
        if (wait_for_refine) rostAdapter.wait_for_processing(false);
        auto token = rostAdapter.get_rost().get_read_token();
        return rostAdapter.get_map(*token);
    }

    auto getDistMap(bool wait_for_refine = true) const {
        if (wait_for_refine) rostAdapter.wait_for_processing(false);
        auto token = rostAdapter.get_rost().get_read_token();
        return rostAdapter.get_dist_map(*token);
    }

    auto getGTMap() const {
        double const timestamp = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        segmentation->id = ros::Time(timestamp).sec;
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
    auto merged = std::make_unique<Segmentation<int, PoseDim, int, double>>(segmentations[0]->frame,
                                                                            segmentations[0]->timestamp,
                                                                            segmentations[0]->id,
                                                                            segmentations[0]->cell_size,
                                                                            std::vector<int>(),
                                                                            segmentations[0]->observation_poses);
    for (auto i = 1; i < segmentations.size(); ++i) {
        merged->observation_poses
              .insert(merged->observation_poses.end(),
                      segmentations[i]->observation_poses.begin(),
                      segmentations[i]->observation_poses.end());
    }
    if (lifting.empty()) {
        for (auto const &map : segmentations) {
            if constexpr (std::is_integral_v<typename decltype(map->observations)::value_type>) {
                merged->observations.insert(merged->observations.end(), map->observations.begin(), map->observations.end());
            } else {
                std::transform(map->observations.begin(),
                               map->observations.end(),
                               std::back_inserter(merged->observations),
                               argmax<std::vector<int>>);
            }
        }
    } else {
        for (auto i = 0; i < segmentations.size(); ++i) {
            if constexpr (std::is_integral_v<typename decltype(segmentations[i]->observations)::value_type>) {
                std::transform(segmentations[i]->observations.begin(),
                               segmentations[i]->observations.end(),
                               std::back_inserter(merged->observations),
                               [i, &lifting](int const &obs) { return lifting[i][obs]; });
            } else {
                std::transform(segmentations[i]->observations.begin(),
                               segmentations[i]->observations.end(),
                               std::back_inserter(merged->observations),
                               [i, &lifting](std::vector<int> const &obs) { return lifting[i][argmax(obs)]; });
            }
        }
    }
    return merged;
}

template<uint32_t LeftPoseDim = 3, uint32_t RightPoseDim = 3>
void align(Segmentation<int, LeftPoseDim, int, double> &segmentation, Segmentation<int, RightPoseDim, int, double> const &reference) {
    auto const cooccurrence_data = compute_cooccurences(reference, segmentation);
    auto const &counts = cooccurrence_data.first;
    std::vector<std::vector<double>> costs(counts[0].size(), std::vector<double>(counts.size(), 0.0));
    for (auto i = 0ul; i < counts.size(); ++i) {
        for (auto j = 0ul; j < counts[0].size(); ++j) {
            costs[j][i] = std::accumulate(counts[i].begin(), counts[i].end(), 0.0) - 2 * counts[i][j];
        }
    }
    int num_topics = counts.size();
    auto const lifting = get_permutation(hungarian_assignments(costs), &num_topics);
    for (auto &obs : segmentation.observations) {
        obs = lifting[obs];
    }
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
    auto segmentationAdapter = std::make_shared<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>>(&nh, true);

    std::vector<std::unique_ptr<RobotSim>> robots;
    //    std::vector<ros::Publisher> map_pubs;
    for (auto i = 4; i < argc; ++i) {
        robots.emplace_back(std::make_unique<RobotSim>(std::to_string(i - 4),
                                                       std::string(argv[i]),
                                                       nh,
                                                       image_topic_name,
                                                       depth_topic_name,
                                                       segmentation_topic_name,
                                                       segmentationAdapter));
        //        map_pubs.push_back(nh.advertise<sunshine_msgs::TopicMap>("/" + robots.back()->getName() + "/map", 0));
    }
    ros::Publisher naive_map_pub = nh.advertise<sunshine_msgs::TopicMap>("/naive_map", 0);
    ros::Publisher merged_map_pub = nh.advertise<sunshine_msgs::TopicMap>("/merged_map", 0);
    ros::Publisher hungarian_map_pub = nh.advertise<sunshine_msgs::TopicMap>("/hungarian_map", 0);
    ros::Publisher gt_map_pub = nh.advertise<sunshine_msgs::TopicMap>("/gt_map", 0);

    auto const csv_filename = nh.param<std::string>("output_filename", "");
    auto writer = (csv_filename.empty()) ? nullptr : std::make_unique<csv_writer<',', '"'>>(csv_filename);
    if (writer) {
        csv_row<> header{};
        header.append("Method");
        header.append("Number of Robots");
        header.append("Number of Observations");
        header.append("Unique Topics");
        header.append("SSD");
        header.append("Cluster Size");
        header.append("Matched Mean-Square Cluster Distance");
        header.append("Matched Silhouette Index");
        header.append("Matched Davies-Bouldin Index");
        header.append("Mutual Information");
        header.append("Normalized Mutual Information");
        header.append("Adjusted Mutual Information");
        writer->write_header(header);
    }

    auto const populate_row = [&robots](std::string const &name,
                                 size_t const& n_observations,
                                 match_results const &correspondences,
                                 match_scores const &scores,
                                 std::optional<std::tuple<double, double, double>> metrics = {}) {
        csv_row<> row;
        row.append(name);
        row.append(robots.size());
        row.append(n_observations);
        row.append(correspondences.num_unique);
        row.append(correspondences.ssd);
        row.append(scores.cluster_sizes);
        row.append(scores.mscd);
        row.append(scores.silhouette);
        row.append(scores.davies_bouldin);
        if (metrics) {
            row.append(std::get<0>(metrics.value()));
            row.append(std::get<1>(metrics.value()));
            row.append(std::get<2>(metrics.value()));
        } else{
            for (auto i = 0; i < 3; ++i) row.append("");
        }
        return row;
    };

    auto const compute_metrics = [](sunshine::Segmentation<int, 3, int, double> const &gt_seg,
                                    sunshine::Segmentation<int, 4, int, double> const &topic_seg) {
        auto const contingency_table = compute_matches<4>(gt_seg, topic_seg);
        auto const &matches = std::get<0>(contingency_table);
        auto const &gt_weights = std::get<1>(contingency_table);
        auto const &topic_weights = std::get<2>(contingency_table);
        double const &total_weight = static_cast<double>(std::get<3>(contingency_table));

        double const mi = compute_mutual_info(matches, gt_weights, topic_weights, total_weight);
        double const ex = entropy<>(gt_weights, total_weight), ey = entropy<>(topic_weights, total_weight);
        double const nmi = compute_nmi(contingency_table, mi, ex, ey);
        double const emi = expected_mutual_info(gt_weights, topic_weights, total_weight);
        double const ami = compute_ami(contingency_table, mi, emi, ex, ey);
        return std::make_tuple(mi, nmi, ami);
    };

    auto const fetch_new_topic_models = [&robots](bool remove_unused = true) {
        std::vector<Phi> topic_models;
        for (auto const &robot : robots) {
            topic_models.push_back(robot->getTopicModel(true));
            if (remove_unused) topic_models.back().remove_unused();
        }
        return topic_models;
    };

    bool active = true;
    size_t n_obs = 0;
    auto start = std::chrono::steady_clock::now();
    while (active && ros::ok()) {
        active = false;
        for (auto &robot : robots) {
            auto const robot_active = robot->next();
            active = active || robot_active;
        }
        n_obs += active;

        auto topic_models = fetch_new_topic_models(true);
        auto const correspondences_clear = match_topics("clear-l1", {topic_models.begin(), topic_models.end()});
        match_scores const scores_clear(topic_models, correspondences_clear.lifting, normed_dist_sq<double>);

        auto const correspondences_hungarian = match_topics("hungarian-l1", {topic_models.begin(), topic_models.end()});
        match_scores const scores_hungarian(topic_models, correspondences_hungarian.lifting, normed_dist_sq<double>);

        auto const correspondences_naive = match_topics("id", {topic_models.begin(), topic_models.end()});
        match_scores const scores_naive(topic_models, correspondences_naive.lifting, normed_dist_sq<double>);

        uint32_t matched = 0;
        for (auto const size : scores_clear.cluster_sizes) { matched += (size > 1); }
        std::cout << "Matched: " << matched << "/" << correspondences_clear.num_unique << std::endl;
        std::cout << "Refine time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << std::endl;
        start = std::chrono::steady_clock::now();

        std::vector<std::unique_ptr<Segmentation<int, 4, int, double>>> segmentations;
        std::vector<std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>>> gt_segmentations;
        for (auto i = 0ul; i < robots.size(); ++i) {
            segmentations.emplace_back(robots[i]->getMap());
            if (!segmentation_topic_name.empty()) gt_segmentations.push_back(robots[i]->getGTMap());
            //            auto const cooccurrence_data = compute_cooccurences(*(robots[i]->getGTMap()), *(robots[i]->getDistMap()));
            //            map_pubs[i].publish(toRosMsg(*segmentations.back()));
        }

        auto naive_merged = merge(segmentations);
        auto clear_merged = merge(segmentations, correspondences_clear.lifting);
        auto hungarian_merged = merge(segmentations, correspondences_hungarian.lifting);

        if (!segmentation_topic_name.empty()) {
            auto const gt_merged = merge<3>(gt_segmentations);
            align(*naive_merged, *gt_merged);
            align(*clear_merged, *gt_merged);
            align(*hungarian_merged, *gt_merged);
            gt_map_pub.publish(toRosMsg(*gt_merged));

            if (writer) {
                auto const naive_metrics = compute_metrics(*gt_merged, *naive_merged);
                auto const clear_metrics = compute_metrics(*gt_merged, *clear_merged);
                auto const hungarian_metrics = compute_metrics(*gt_merged, *hungarian_merged);
                writer->write_row(populate_row("Naive", n_obs, correspondences_naive, scores_naive, naive_metrics));
                writer->write_row(populate_row("Hungarian", n_obs, correspondences_hungarian, scores_hungarian, hungarian_metrics));
                writer->write_row(populate_row("CLEAR", n_obs, correspondences_clear, scores_clear, clear_metrics));
            }
        } else if (writer) {
            writer->write_row(populate_row("Naive", n_obs, correspondences_naive, scores_naive));
            writer->write_row(populate_row("Hungarian", n_obs, correspondences_hungarian, scores_hungarian));
            writer->write_row(populate_row("CLEAR", n_obs, correspondences_clear, scores_clear));
        }
        if (writer) writer->flush();

        naive_map_pub.publish(toRosMsg(*naive_merged));
        merged_map_pub.publish(toRosMsg(*clear_merged));
        hungarian_map_pub.publish(toRosMsg(*hungarian_merged));
    }

    return 0;
}

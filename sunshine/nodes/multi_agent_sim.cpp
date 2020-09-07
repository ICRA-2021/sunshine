#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "sunshine/2d_adapter.hpp"
#include "sunshine/benchmark.hpp"
#include "sunshine/common/adrost_utils.hpp"
#include "sunshine/common/csv.hpp"
#include "sunshine/common/metric.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/rosbag_utils.hpp"
#include "sunshine/common/topic_map_utils.hpp"
#include "sunshine/common/utils.hpp"
#include "sunshine/depth_adapter.hpp"
#include "sunshine/observation_transform_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"
#include "sunshine/visual_word_adapter.hpp"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sunshine/common/ros_conversions.hpp>
#include <tf2_msgs/TFMessage.h>
#include <utility>

using namespace sunshine;

class RobotSim {
    std::string const name;
    std::unique_ptr<BagIterator> bagIter;
    VisualWordAdapter visualWordAdapter;
    std::shared_ptr<ROSTAdapter<4, double, double>> rostAdapter;
    std::shared_ptr<ROSTAdapter<4, double, double>> externalRostAdapter;
    std::shared_ptr<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>> segmentationAdapter;
    ObservationTransformAdapter<WordDepthAdapter::Output> wordTransformAdapter;
    ObservationTransformAdapter<ImageDepthAdapter::Output> imageTransformAdapter;
    std::unique_ptr<WordDepthAdapter> wordDepthAdapter;
    std::unique_ptr<ImageDepthAdapter> imageDepthAdapter;
    std::unique_ptr<Word2DAdapter<3>> word2dAdapter;
    std::unique_ptr<Image2DAdapter<3>> image2dAdapter;
    std::unique_ptr<ImageObservation> lastRgb, lastSegmentation;
    std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>> segmentation;
    double depth_timestamp = -1;
    bool transform_found = false;
    bool processed_rgb = false;
    bool const use_3d;
    bool use_segmentation;

    bool tryProcess() {
        if (!lastRgb || processed_rgb) return false;
        if (use_3d && (!transform_found || lastRgb->timestamp != depth_timestamp)) return false;
        if (use_segmentation && (!lastSegmentation || lastRgb->timestamp != lastSegmentation->timestamp)) return false;
        // TODO: remove duplication between if branches below
        if (use_3d) {
            auto observation = lastRgb >> visualWordAdapter >> *wordDepthAdapter >> wordTransformAdapter;
            if (externalRostAdapter) {
                observation >> *rostAdapter;
                observation >> *externalRostAdapter;
            } else {
                (*rostAdapter)(std::move(observation));
            }
            if (use_segmentation) segmentation = lastSegmentation >> *imageDepthAdapter >> imageTransformAdapter >> *segmentationAdapter;
        } else {
            auto observation = lastRgb >> visualWordAdapter >> *word2dAdapter;
            if (externalRostAdapter) {
                observation >> *rostAdapter;
                observation >> *externalRostAdapter;
            } else {
                (*rostAdapter)(std::move(observation));
            }
            if (use_segmentation) segmentation = lastSegmentation >> *image2dAdapter >> *segmentationAdapter;
        }
        processed_rgb = true;
        return true;
    }

    bool imageCallback(sensor_msgs::Image::ConstPtr image) {
        lastRgb = std::make_unique<ImageObservation>(fromRosMsg(image));
        processed_rgb = false;
        return tryProcess();
    }

    bool segmentationCallback(sensor_msgs::Image::ConstPtr image) {
        lastSegmentation = std::make_unique<ImageObservation>(fromRosMsg(image));
        return tryProcess();
    }

    bool depthCallback(sensor_msgs::PointCloud2::ConstPtr const &msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pc);

        wordDepthAdapter->updatePointCloud(pc);
        imageDepthAdapter->updatePointCloud(pc);
        depth_timestamp = msg->header.stamp.toSec();
        return tryProcess();
    };

    bool transformCallback(tf2_msgs::TFMessage::ConstPtr const &tfMsg) {
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            if (lastRgb && (tfTransform.frame_id_ == lastRgb->frame || tfTransform.child_frame_id_ == lastRgb->frame)) transform_found = true;
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
        }
        return tryProcess();
    }

  public:
    template<typename ParamServer>
    RobotSim(std::string name,
             ParamServer const &parameters,
             bool const use_3d,
             std::shared_ptr<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>> shared_seg_adapter = nullptr,
             std::shared_ptr<ROSTAdapter<4, double, double>> external_rost_adapter = nullptr)
            : name(std::move(name)),
              bagIter(nullptr),
              visualWordAdapter(&parameters),
              rostAdapter(std::make_shared<ROSTAdapter<4, double, double>>(&parameters)),
              externalRostAdapter(std::move(external_rost_adapter)),
              segmentationAdapter((shared_seg_adapter)
                                  ? std::move(shared_seg_adapter)
                                  : std::make_shared<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>>(&parameters,
                                                                                                                            true)),
              wordTransformAdapter(&parameters),
              imageTransformAdapter(&parameters),
              use_3d(use_3d) {}

    void open(std::string const& bagFilename,
              std::string const &image_topic,
              std::string const &depth_topic = "",
              std::string const &segmentation_topic= "",
              double x_offset = 0) {
        if (use_3d && depth_topic.empty()) throw std::invalid_argument("Must provide depth topic if operating in 3D mode");
        wordDepthAdapter = (depth_topic.empty()) ? nullptr : std::make_unique<WordDepthAdapter>();
        imageDepthAdapter = (depth_topic.empty()) ? nullptr : std::make_unique<ImageDepthAdapter>();
        word2dAdapter = (depth_topic.empty()) ? std::make_unique<Word2DAdapter<3>>(x_offset, 0, 0, true) : nullptr;
        image2dAdapter = (depth_topic.empty()) ? std::make_unique<Image2DAdapter<3>>(x_offset, 0, 0, true) : nullptr;
        use_segmentation = !segmentation_topic.empty();

        bagIter = std::make_unique<BagIterator>(bagFilename);
        bagIter->add_callback<sensor_msgs::Image>(image_topic, [this](auto const &msg) { return this->imageCallback(msg); });
        bagIter->add_callback<sensor_msgs::Image>(segmentation_topic, [this](auto const &msg) { return this->segmentationCallback(msg); });
        bagIter->add_callback<sensor_msgs::PointCloud2>(depth_topic, [this](auto const &msg) { return this->depthCallback(msg); });
        bagIter->add_callback<tf2_msgs::TFMessage>("/tf", [this](auto const &msg) { return this->transformCallback(msg); });
        bagIter->set_logging(true);
    }

    Phi getTopicModel(bool wait_for_refine = false) const {
        if (wait_for_refine) rostAdapter->wait_for_processing(false);
        auto token = rostAdapter->get_rost().get_read_token();
        return rostAdapter->get_topic_model(*token);
    }

    auto getRost() {
        return rostAdapter;
    }

    auto getMap(bool wait_for_refine = true) const {
        if (wait_for_refine) rostAdapter->wait_for_processing(false);
        auto token = rostAdapter->get_rost().get_read_token();
        return rostAdapter->get_map(*token);
    }

    auto getDistMap(bool wait_for_refine = true) const {
        if (wait_for_refine) rostAdapter->wait_for_processing(false);
        auto token = rostAdapter->get_rost().get_read_token();
        return rostAdapter->get_dist_map(*token);
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
        if (!bagIter) throw std::logic_error("No bag opened to play");
        return !bagIter->play(false);
    }

    void waitForProcessing() const {
        rostAdapter->wait_for_processing(false);
    }

    void pause() {
        rostAdapter->stopWorkers();
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
        merged->observation_poses.insert(merged->observation_poses.end(),
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
                               [i, &lifting](int const &obs) {
                    assert(obs < lifting[i].size());
                    return lifting[i][obs]; });
            } else {
                std::transform(segmentations[i]->observations.begin(),
                               segmentations[i]->observations.end(),
                               std::back_inserter(merged->observations),
                               [i, &lifting](std::vector<int> const &obs) {
                    assert(argmax(obs) < lifting[i].size());
                    return lifting[i][argmax(obs)]; });
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

template<uint32_t gt_pose_dim = 3>
auto compute_metrics(sunshine::Segmentation<int, gt_pose_dim, int, double> const &gt_seg,
                     sunshine::Segmentation<int, 4, int, double> const &topic_seg) {
    auto const contingency_table = compute_matches<4, gt_pose_dim>(gt_seg, topic_seg);
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

    auto output_prefix = nh.param<std::string>("output_prefix", "");
    output_prefix += (output_prefix.empty() || output_prefix.back() == '/') ? "" : "/";
    auto const csv_filename = nh.param<std::string>("output_filename", output_prefix + "stats.csv");

    auto segmentationAdapter = std::make_shared<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>>(&nh, true);
    std::vector<std::string> match_methods = {"id", "hungarian-l1", "clear-l1"};
    auto const methods_str = nh.param<std::string>("match_methods", "");
    if (!methods_str.empty()) {
        match_methods = sunshine::split(methods_str, ',');
    }

    std::vector<std::unique_ptr<RobotSim>> robots;
    std::cout << "Running single-agent simulation...." << std::endl;
    RobotSim aggregateRobot("Aggregate", nh, !depth_topic_name.empty(), segmentationAdapter);
    for (auto i = 4; i < argc; ++i) {
        aggregateRobot.open(std::string(argv[i]), image_topic_name, depth_topic_name, segmentation_topic_name);
        while (ros::ok() && aggregateRobot.next()) {
            aggregateRobot.waitForProcessing();
        }
        aggregateRobot.waitForProcessing();
    }
    aggregateRobot.pause();
    auto singleRobotSegmentation = aggregateRobot.getMap();
    WordColorMap<int> topicColorMap;
    auto const box = nh.param<std::string>("box", "");
    if (!output_prefix.empty()) {
        if (!segmentation_topic_name.empty()) {
            std::vector<std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>>> gt_segmentations;
            gt_segmentations.push_back(aggregateRobot.getGTMap());
            auto const gt_merged = merge<3>(gt_segmentations);
            auto const gtTopicImg = createTopicImg(toRosMsg(*gt_merged), topicColorMap, aggregateRobot.getRost()->get_cell_size()[1],
                                                   true, 0, 0, box, true);
            std::string const file_prefix = output_prefix + "0-ground_truth";
            saveTopicImg(gtTopicImg, file_prefix + "-map.png", file_prefix + "-colors.csv", &topicColorMap);

            align(*singleRobotSegmentation, *gt_merged);
        }
        auto const topicImg = createTopicImg(toRosMsg(*singleRobotSegmentation), topicColorMap, aggregateRobot.getRost()->get_cell_size()[1],
                                                       true, 0, 0, box, true);
        std::string const file_prefix = output_prefix + "0-single_robot";
        saveTopicImg(topicImg, file_prefix + "-map.png", file_prefix + "-colors.csv", &topicColorMap);
    }
    std::cout << "Finished single-agent simulation." << std::endl;

    //    std::vector<ros::Publisher> map_pubs;
    for (auto i = 4; i < argc; ++i) {
        robots.emplace_back(std::make_unique<RobotSim>(std::to_string(i - 4),
                                                       nh,
                                                       !depth_topic_name.empty(),
                                                       segmentationAdapter));
        robots.back()->open(std::string(argv[i]), image_topic_name, depth_topic_name, segmentation_topic_name, (i - 4) * 2000);
        //        map_pubs.push_back(nh.advertise<sunshine_msgs::TopicMap>("/" + robots.back()->getName() + "/map", 0));
    }
    std::unique_ptr<ros::Publisher> gt_map_pub = (segmentation_topic_name.empty()) ? nullptr : std::make_unique<ros::Publisher>(nh.advertise<sunshine_msgs::TopicMap>("/gt_map", 0));
    std::vector<std::unique_ptr<ros::Publisher>> publishers;
    for (auto const& method : match_methods) {
        publishers.push_back(std::make_unique<ros::Publisher>(nh.advertise<sunshine_msgs::TopicMap>("/" + sunshine::replace_all(method, "-", "_") + "_map", 0)));
    }

    auto writer = (csv_filename.empty()) ? nullptr : std::make_unique<csv_writer<',', '"'>>(csv_filename);
    if (writer) {
        csv_row<> header{};
        header.append("Method");
        header.append("Number of Robots");
        header.append("Number of Observations");
        header.append("Unique Topics");
        header.append("SSD");
        header.append("Single Robot MI");
        header.append("Single Robot NMI");
        header.append("Single Robot AMI");
        header.append("Cluster Size");
        header.append("Matched Mean-Square Cluster Distance");
        header.append("Matched Silhouette Index");
        header.append("Matched Davies-Bouldin Index");
        header.append("GT MI");
        header.append("GT NMI");
        header.append("GT AMI");
        header.append("Individual GT MIs");
        header.append("Individual GT NMIs");
        header.append("Individual GT AMIs");
        header.append("Single Robot GT MI");
        header.append("Single Robot GT NMI");
        header.append("Single Robot GT AMI");
        writer->write_header(header);
    }

    auto const populate_row = [&robots](std::string const &name,
                                        size_t const &n_observations,
                                        match_results const &correspondences,
                                        std::tuple<double, double, double> metrics,
                                        match_scores const &scores,
                                        std::optional<std::tuple<double, double, double>> gt_metrics = {},
                                        std::optional<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>> individ_gt_metrics = {},
                                        std::optional<std::tuple<double, double, double>> single_gt_metrics = {}) {
        csv_row<> row;
        row.append(name);
        row.append(robots.size());
        row.append(n_observations);
        row.append(correspondences.num_unique);
        row.append(correspondences.ssd);
        row.append(std::get<0>(metrics));
        row.append(std::get<1>(metrics));
        row.append(std::get<2>(metrics));
        row.append(scores.cluster_sizes);
        row.append(scores.mscd);
        row.append(scores.silhouette);
        row.append(scores.davies_bouldin);
        if (gt_metrics) {
            row.append(std::get<0>(gt_metrics.value()));
            row.append(std::get<1>(gt_metrics.value()));
            row.append(std::get<2>(gt_metrics.value()));
            row.append(std::get<0>(individ_gt_metrics.value()));
            row.append(std::get<1>(individ_gt_metrics.value()));
            row.append(std::get<2>(individ_gt_metrics.value()));
            row.append(std::get<0>(single_gt_metrics.value()));
            row.append(std::get<1>(single_gt_metrics.value()));
            row.append(std::get<2>(single_gt_metrics.value()));
        } else {
            for (auto i = 0; i < 9; ++i) row.append("");
        }
        return row;
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

        auto topic_models = fetch_new_topic_models(false);
        auto const refine_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        std::cout << "Refine time: " << refine_time << std::endl;

        std::vector<std::unique_ptr<Segmentation<int, 4, int, double>>> segmentations;
        std::vector<std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>>> gt_segmentations;
        for (auto i = 0ul; i < robots.size(); ++i) {
            segmentations.emplace_back(robots[i]->getMap());
            if (!segmentation_topic_name.empty()) gt_segmentations.push_back(robots[i]->getGTMap());
            //            auto const cooccurrence_data = compute_cooccurences(*(robots[i]->getGTMap()), *(robots[i]->getDistMap()));
            //            map_pubs[i].publish(toRosMsg(*segmentations.back()));
        }
        auto const gt_merged = (segmentation_topic_name.empty()) ? nullptr : merge<3>(gt_segmentations);

        for (auto i = 0ul; i < match_methods.size(); ++i) {
            auto const& method = match_methods[i];
            auto const correspondences = match_topics(method, {topic_models.begin(), topic_models.end()});
            match_scores const scores(topic_models, correspondences.lifting, normed_dist_sq<double>);

            uint32_t matched = 0;
            for (auto const size : scores.cluster_sizes) { matched += (size > 1); }
            std::cout << method << " matched clusters: " << matched << "/" << correspondences.num_unique << std::endl;

            auto merged_segmentations = merge(segmentations, correspondences.lifting);
            auto const sr_ref_metrics = compute_metrics(*singleRobotSegmentation, *merged_segmentations);
            std::cout << method << " SR AMI:" << std::get<2>(sr_ref_metrics) << std::endl;
            if (gt_merged) {
                align(*merged_segmentations, *gt_merged);

                if (writer) {
                    std::vector<double> individ_mi, individ_nmi, individ_ami;
                    for (auto const& segmentation : segmentations) {
                        auto const individ_metric = compute_metrics(*gt_merged, *segmentation);
                        individ_mi.push_back(std::get<0>(individ_metric));
                        individ_nmi.push_back(std::get<1>(individ_metric));
                        individ_ami.push_back(std::get<2>(individ_metric));
                    }
                    auto const individ_gt_metrics = std::make_tuple(individ_mi, individ_nmi, individ_ami);

                    auto const gt_ref_metrics       = compute_metrics(*gt_merged, *merged_segmentations);
                    auto const sr_gt_metrics        = compute_metrics(*gt_merged, *singleRobotSegmentation);
                    writer->write_row(populate_row(method, n_obs, correspondences, sr_ref_metrics, scores, gt_ref_metrics, individ_gt_metrics, sr_gt_metrics));

                    std::cout << method << " GT AMI:" << std::get<2>(gt_ref_metrics) << std::endl;
                }
                gt_map_pub->publish(toRosMsg(*gt_merged));
            } else if (writer) {
                writer->write_row(populate_row("Naive", n_obs, correspondences, sr_ref_metrics, scores));
            }
            if (writer) writer->flush();

            auto const merged_map = toRosMsg(*merged_segmentations);
            if (!output_prefix.empty()) {
                auto const topicImg = sunshine::createTopicImg(merged_map, topicColorMap, aggregateRobot.getRost()->get_cell_size()[1],
                                                               true, 0, 0, box, true);
                std::string const file_prefix = output_prefix + std::to_string(n_obs) + "-" + method;
                saveTopicImg(topicImg, file_prefix + "-map.png", file_prefix + "-colors.csv", &topicColorMap);
            }
            publishers[i]->publish(merged_map);
        }

        start = std::chrono::steady_clock::now();
    }

    return 0;
}

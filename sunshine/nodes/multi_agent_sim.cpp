#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "sunshine/2d_adapter.hpp"
#include "sunshine/common/simulation_utils.hpp"
#include "sunshine/common/matching_utils.hpp"
#include "sunshine/common/segmentation_utils.hpp"
#include "sunshine/common/csv.hpp"
#include "sunshine/common/metric.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/rosbag_utils.hpp"
#include "sunshine/common/data_proc_utils.hpp"
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
#include <iostream>

using namespace sunshine;

class MultiAgentSimulation {
    std::vector<std::string> const bagfiles;
    std::string const image_topic;
    std::string const depth_topic;
    std::string const segmentation_topic;
    std::map<std::string, std::string> params;

  public:
    MultiAgentSimulation() = default;
    explicit MultiAgentSimulation(std::vector<std::string> bagfiles,
                                  std::string image_topic,
                                  std::string depth_topic,
                                  std::string segmentation_topic)
                                  : bagfiles(std::move(bagfiles))
                                  , image_topic(std::move(image_topic))
                                  , depth_topic(std::move(depth_topic))
                                  , segmentation_topic(std::move(segmentation_topic)) {}

    template<typename ParamServer>
    void record(ParamServer const& paramServer, std::vector<std::string> const& match_methods = {}) {

    }

    void play(std::string const& logfile, std::vector<std::string> const& match_methods) {

    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        throw std::logic_error("Not implemented");
    }
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
            auto const gtTopicImg = createTopicImg(toRosMsg(*gt_merged),
                                                   topicColorMap,
                                                   aggregateRobot.getRost()->get_cell_size()[1],
                                                   true,
                                                   0,
                                                   0,
                                                   box,
                                                   true);
            std::string const file_prefix = output_prefix + "0-ground_truth";
            saveTopicImg(gtTopicImg, file_prefix + "-map.png", file_prefix + "-colors.csv", &topicColorMap);

            align(*singleRobotSegmentation, *gt_merged);
        }
        auto const topicImg = createTopicImg(toRosMsg(*singleRobotSegmentation),
                                             topicColorMap,
                                             aggregateRobot.getRost()->get_cell_size()[1],
                                             true,
                                             0,
                                             0,
                                             box,
                                             true);
        std::string const file_prefix = output_prefix + "0-single_robot";
        saveTopicImg(topicImg, file_prefix + "-map.png", file_prefix + "-colors.csv", &topicColorMap);
    }
    {
        CompressedFileWriter gt_map_writer("gt_map.bin");
        gt_map_writer << *aggregateRobot.getGTMap();
        CompressedFileWriter sr_map_writer("aggregate_map.bin");
        sr_map_writer << *aggregateRobot.getDistMap();
        std::cout << "Finished single-agent simulation." << std::endl;
    }

    //    std::vector<ros::Publisher> map_pubs;
    for (auto i = 4; i < argc; ++i) {
        robots.emplace_back(std::make_unique<RobotSim>(std::to_string(i - 4), nh, !depth_topic_name.empty(), segmentationAdapter));
        robots.back()->open(std::string(argv[i]), image_topic_name, depth_topic_name, segmentation_topic_name, (i - 4) * 2000);
        //        map_pubs.push_back(nh.advertise<sunshine_msgs::TopicMap>("/" + robots.back()->getName() + "/map", 0));
    }
    std::unique_ptr<ros::Publisher> gt_map_pub = (segmentation_topic_name.empty())
                                                 ? nullptr
                                                 : std::make_unique<ros::Publisher>(nh.advertise<sunshine_msgs::TopicMap>("/gt_map", 0));
    std::vector<std::unique_ptr<ros::Publisher>> publishers;
    for (auto const &method : match_methods) {
        publishers.push_back(std::make_unique<ros::Publisher>(nh.advertise<sunshine_msgs::TopicMap>(
                "/" + sunshine::replace_all(method, "-", "_") + "_map", 0)));
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

    auto const fetch_new_topic_models = [&robots](bool remove_unused = false) {
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
        std::string const topic_model_filename = output_prefix + std::to_string(n_obs) + "-models.bin";
        {
            CompressedFileWriter topic_model_writer(topic_model_filename);
            for (auto const &phi : topic_models) {
                topic_model_writer << phi;
            }
        }
        {
            CompressedFileReader topic_model_reader(topic_model_filename);
            for (auto const &phi : topic_models) {
                Phi test;
                topic_model_reader >> test;
            }
        }

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
            auto const &method = match_methods[i];
            auto const correspondences = match_topics(method, topic_models);
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
                    for (auto const &segmentation : segmentations) {
                        auto const individ_metric = compute_metrics(*gt_merged, *segmentation);
                        individ_mi.push_back(std::get<0>(individ_metric));
                        individ_nmi.push_back(std::get<1>(individ_metric));
                        individ_ami.push_back(std::get<2>(individ_metric));
                    }
                    auto const individ_gt_metrics = std::make_tuple(individ_mi, individ_nmi, individ_ami);

                    auto const gt_ref_metrics = compute_metrics(*gt_merged, *merged_segmentations);
                    auto const sr_gt_metrics = compute_metrics(*gt_merged, *singleRobotSegmentation);
                    writer->write_row(populate_row(method,
                                                   n_obs,
                                                   correspondences,
                                                   sr_ref_metrics,
                                                   scores,
                                                   gt_ref_metrics,
                                                   individ_gt_metrics,
                                                   sr_gt_metrics));

                    std::cout << method << " GT AMI:" << std::get<2>(gt_ref_metrics) << std::endl;
                }
                gt_map_pub->publish(toRosMsg(*gt_merged));
            } else if (writer) {
                writer->write_row(populate_row("Naive", n_obs, correspondences, sr_ref_metrics, scores));
            }
            if (writer) writer->flush();

            auto const merged_map = toRosMsg(*merged_segmentations);
            if (!output_prefix.empty()) {
                auto const topicImg = sunshine::createTopicImg(merged_map,
                                                               topicColorMap,
                                                               aggregateRobot.getRost()->get_cell_size()[1],
                                                               true,
                                                               0,
                                                               0,
                                                               box,
                                                               true);
                std::string const file_prefix = output_prefix + std::to_string(n_obs) + "-" + method;
                saveTopicImg(topicImg, file_prefix + "-map.png", file_prefix + "-colors.csv", &topicColorMap);
            }
            publishers[i]->publish(merged_map);
        }

        start = std::chrono::steady_clock::now();
    }

    return 0;
}

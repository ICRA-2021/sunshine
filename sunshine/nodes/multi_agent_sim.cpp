#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "sunshine/2d_adapter.hpp"
#include "sunshine/common/simulation_utils.hpp"
#include "sunshine/common/matching_utils.hpp"
#include "sunshine/common/segmentation_utils.hpp"
#include "sunshine/common/csv.hpp"
#include "sunshine/common/metric.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/data_proc_utils.hpp"
#include "sunshine/common/utils.hpp"
#include "sunshine/observation_transform_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"

#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/map.hpp>
#include <sunshine/common/ros_conversions.hpp>
#include <utility>
#include <iostream>

using namespace sunshine;

class MultiAgentSimulation {
    std::vector<std::string> bagfiles;
    std::string image_topic;
    std::string depth_topic;
    std::string segmentation_topic;
    std::map<std::string, std::string> params;
    decltype(std::declval<RobotSim>().getGTMap()) gtMap;
    decltype(std::declval<RobotSim>().getDistMap()) aggregateMap;

  public:

    template <class ParamServer>
    class ParamPassthrough {
        MultiAgentSimulation* parent;
        ParamServer const* nh;

      public:
        ParamPassthrough(MultiAgentSimulation* parent, ParamServer const* nh) : parent(parent), nh(nh) {}
        template <typename T>
        T param(std::string param_name, T default_value) const {
            T const val = nh->template param<T>(param_name, default_value);
            auto iter = parent->params.find(param_name);
            if constexpr(std::is_same_v<std::string, T>) {
                assert(iter == parent->params.end() || iter->second == val);
                parent->params.insert(iter, std::make_pair(param_name, val));
            } else {
                assert(iter == parent->params.end() || iter->second == std::to_string(val));
                parent->params.insert(iter, std::make_pair(param_name, std::to_string(val)));
            }
            return val;
        }
    };

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
    void record(ParamServer const& nh, std::vector<std::string> const& match_methods = {}) {
        ParamPassthrough<ParamServer> paramPassthrough(this, &nh);
        std::cout << "Running single-agent simulation...." << std::endl;
        auto segmentationAdapter = std::make_shared<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>>(&nh, true);
        RobotSim aggregateRobot("Aggregate", paramPassthrough, !depth_topic.empty(), segmentationAdapter);
        for (auto const& bag : bagfiles) {
            aggregateRobot.open(bag, image_topic, depth_topic, segmentation_topic);
            while (ros::ok() && aggregateRobot.next()) {
                aggregateRobot.waitForProcessing();
            }
            aggregateRobot.waitForProcessing();
        }
        aggregateRobot.pause();
        aggregateMap = aggregateRobot.getDistMap();
        WordColorMap<int> topicColorMap;
        if (!segmentation_topic.empty()) {
            gtMap = aggregateRobot.getGTMap();
        }
        auto const gt_merged = (segmentation_topic.empty()) ? nullptr : merge<3>(make_vector(gtMap.get()));
        auto const sr_merged = merge<4>(make_vector(aggregateMap.get()));

        //    std::vector<ros::Publisher> map_pubs;
        std::vector<std::unique_ptr<RobotSim>> robots;
        for (auto i = 0; i < bagfiles.size(); ++i) {
            robots.emplace_back(std::make_unique<RobotSim>(std::to_string(i), paramPassthrough, !depth_topic.empty(), segmentationAdapter));
            robots.back()->open(bagfiles[i], image_topic, depth_topic, segmentation_topic, i * 2000);
            //        map_pubs.push_back(nh.advertise<sunshine_msgs::TopicMap>("/" + robots.back()->getName() + "/map", 0));
        }

        std::unique_ptr<csv_writer<',', '"'>> writer = nullptr; //(csv_filename.empty()) ? nullptr : std::make_unique<csv_writer<',', '"'>>(csv_filename);
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

            std::vector<std::unique_ptr<Segmentation<int, 4, int, double>>> segmentations;
            std::vector<std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>>> gt_segmentations;
            for (auto & robot : robots) {
                segmentations.emplace_back(robot->getMap());
                if (!segmentation_topic.empty()) gt_segmentations.push_back(robot->getGTMap());
            }

            for (const auto & method : match_methods) {
                auto const correspondences = match_topics(method, topic_models);
                match_scores const scores(topic_models, correspondences.lifting, normed_dist_sq<double>);

                uint32_t matched = 0;
                for (auto const size : scores.cluster_sizes) { matched += (size > 1); }
                std::cout << method << " matched clusters: " << matched << "/" << correspondences.num_unique << std::endl;

                auto merged_segmentations = merge(segmentations, correspondences.lifting);
                auto const sr_ref_metrics = compute_metrics(*sr_merged, *merged_segmentations);
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
                        auto const sr_gt_metrics = compute_metrics(*gt_merged, *sr_merged);
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
                } else if (writer) {
                    writer->write_row(populate_row("Naive", n_obs, correspondences, sr_ref_metrics, scores));
                }
                if (writer) writer->flush();
            }

            start = std::chrono::steady_clock::now();
        }
    }

    void play(std::string const& logfile, std::vector<std::string> const& match_methods) {

    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & bagfiles;
        ar & image_topic;
        ar & depth_topic;
        ar & segmentation_topic;
        ar & params;
        ar & gtMap;
        ar & aggregateMap;
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

    std::vector<std::string> match_methods = {"id", "hungarian-l1", "clear-l1"};
    auto const methods_str = nh.param<std::string>("match_methods", "");
    if (!methods_str.empty()) {
        match_methods = sunshine::split(methods_str, ',');
    }

    std::vector<std::string> bagfiles;
    bagfiles.reserve(argc - 4);
    for (auto i = 4; i < argc; ++i) bagfiles.emplace_back(argv[i]);


    MultiAgentSimulation ref(bagfiles, image_topic_name, depth_topic_name, segmentation_topic_name);
    {
        CompressedFileReader reader("test.sim");
        reader >> ref;
    }

    MultiAgentSimulation sim(bagfiles, image_topic_name, depth_topic_name, segmentation_topic_name);
    sim.record(nh, match_methods);

    CompressedFileWriter writer("test.sim");
    writer << sim;

    return 0;
}

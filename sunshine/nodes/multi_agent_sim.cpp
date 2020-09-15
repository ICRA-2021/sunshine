#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "sunshine/2d_adapter.hpp"
#include "sunshine/external/json.hpp"
#include "sunshine/common/simulation_utils.hpp"
#include "sunshine/common/matching_utils.hpp"
#include "sunshine/common/segmentation_utils.hpp"
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
using json = nlohmann::ordered_json;

class MultiAgentSimulation {
    std::vector<std::string> bagfiles;
    std::string image_topic;
    std::string depth_topic;
    std::string segmentation_topic;
    std::map<std::string, std::string> params;
    decltype(std::declval<RobotSim>().getGTMap()) gtMap;
    decltype(std::declval<RobotSim>().getDistMap()) aggregateMap;
    std::vector<std::vector<decltype(std::declval<RobotSim>().getDistMap())>> robotMaps;
    std::vector<std::vector<decltype(std::declval<RobotSim>().getTopicModel())>> robotModels;

    // Derived data -- not serialized (can be recovered exactly from above data)
    std::unique_ptr<Segmentation<int, 3, int, double>> gtMLMap;
    std::unique_ptr<Segmentation<int, 4, int, double>> aggregateMLMap;
    std::set<decltype(decltype(gtMap)::element_type::observation_poses)::value_type> gtPoses;
    json results = json::object();

  protected:
    [[nodiscard]] json match(decltype(robotModels) const& topic_models, decltype(robotMaps) const& maps, std::string const& match_method) const {
        json match_results = json::array();
        for (auto i = 0ul; i < maps.size(); ++i) {
            json match_result;
            match_result["Number of Observations"] = (i + 1);

            auto const correspondences = match_topics(match_method, topic_models[i]);
            match_result["Unique Topics"] = correspondences.num_unique;
            match_result["SSD"] = correspondences.ssd;

            match_scores const scores(topic_models[i], correspondences.lifting, normed_dist_sq<double>);
            match_result["Cluster Size"] = scores.cluster_sizes;
            match_result["Mean-Square Cluster Distances"] = scores.mscd;
            match_result["Silhouette Indices"] = scores.silhouette;
            match_result["Davies-Bouldin Indices"] = scores.davies_bouldin;

            uint32_t matched = 0;
            for (auto const size : scores.cluster_sizes) { matched += (size > 1); }
            std::cout << match_method << " matched clusters: " << matched << "/" << correspondences.num_unique << std::endl;

            auto merged_segmentations = merge(maps[i], correspondences.lifting);
            {
                auto const sr_ref_metrics = compute_metrics(*aggregateMLMap, *merged_segmentations);
                match_result["SR-MI"]     = std::get<0>(sr_ref_metrics);
                match_result["SR-NMI"]    = std::get<1>(sr_ref_metrics);
                match_result["SR-AMI"]    = std::get<2>(sr_ref_metrics);
                std::cout << match_method << " SR AMI:" << std::get<2>(sr_ref_metrics) << std::endl;
            }

            if (gtMap) {
                std::set<std::array<int, 3>> merged_poses;
                for (auto const& pose : merged_segmentations->observation_poses) {
                    uint32_t const offset = pose.size() == 4;
                    merged_poses.insert({pose[offset], pose[1 + offset], pose[2 + offset]});
                }
                if (!includes(gtPoses, merged_poses)) throw std::logic_error("Ground truth is missing poses!");

                align(*merged_segmentations, *gtMLMap);
                {
                    auto const gt_ref_metrics = compute_metrics(*gtMLMap, *merged_segmentations);
                    match_result["GT-MI"]     = std::get<0>(gt_ref_metrics);
                    match_result["GT-NMI"]    = std::get<1>(gt_ref_metrics);
                    match_result["GT-AMI"]    = std::get<2>(gt_ref_metrics);
                    std::cout << match_method << " GT AMI:" << std::get<2>(gt_ref_metrics) << std::endl;
                }

                std::vector<double> individ_mi, individ_nmi, individ_ami;
                for (auto const &segmentation : maps[i]) {
                    auto const individ_metric = compute_metrics(*gtMLMap, *segmentation);
                    individ_mi.push_back(std::get<0>(individ_metric));
                    individ_nmi.push_back(std::get<1>(individ_metric));
                    individ_ami.push_back(std::get<2>(individ_metric));
                }
                match_result["Individual GT-MI"] = individ_mi;
                match_result["Individual GT-NMI"] = individ_nmi;
                match_result["Individual GT-AMI"] = individ_ami;
            }
            match_results.push_back(match_result);
        }
        return match_results;
    }

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
    bool record(ParamServer const& nh) {
        ParamPassthrough<ParamServer> paramPassthrough(this, &nh);
        std::cout << "Running single-agent simulation...." << std::endl;
        auto segmentationAdapter = std::make_shared<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>>(&nh, true);
        RobotSim aggregateRobot("Aggregate", paramPassthrough, !depth_topic.empty(), segmentationAdapter);
        for (auto const& bag : bagfiles) {
            aggregateRobot.open(bag, image_topic, depth_topic, segmentation_topic);
            while (aggregateRobot.next()) {
                if (!ros::ok()) return false;
                aggregateRobot.waitForProcessing();
            }
            aggregateRobot.waitForProcessing();
        }
        aggregateRobot.pause();
        aggregateMap = aggregateRobot.getDistMap();
        if (!segmentation_topic.empty()) {
            gtMap = aggregateRobot.getGTMap();
            std::set<std::array<int, 3>> aggregate_poses;
            gtPoses = std::set(gtMap->observation_poses.cbegin(), gtMap->observation_poses.cend());
            for (auto const& pose : aggregateMap->observation_poses) {
                uint32_t const offset = pose.size() == 4;
                aggregate_poses.insert({pose[offset], pose[1 + offset], pose[2 + offset]});
            }
            if (!includes(gtPoses, aggregate_poses)) throw std::logic_error("Ground truth is missing poses!");
        }

        //    std::vector<ros::Publisher> map_pubs;
        std::vector<std::unique_ptr<RobotSim>> robots;
        for (auto i = 0; i < bagfiles.size(); ++i) {
            robots.emplace_back(std::make_unique<RobotSim>(std::to_string(i), paramPassthrough, !depth_topic.empty(), segmentationAdapter));
            robots.back()->open(bagfiles[i], image_topic, depth_topic, segmentation_topic, i * 2000);
            //        map_pubs.push_back(nh.advertise<sunshine_msgs::TopicMap>("/" + robots.back()->getName() + "/map", 0));
        }

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
        while (active) {
            if (!ros::ok()) return false;
            active = false;
            for (auto &robot : robots) {
                auto const robot_active = robot->next();
                active = active || robot_active;
            }
            n_obs += active;

            robotModels.push_back(fetch_new_topic_models(false));
            auto const refine_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            std::cout << "Refine time: " << refine_time << std::endl;

            robotMaps.emplace_back();
            robotMaps.back().reserve(robots.size());
            for (auto const& robot : robots) {
                robotMaps.back().emplace_back(robot->getDistMap());
            }

            start = std::chrono::steady_clock::now();
        }
        return true;
    }

    void process(std::vector<std::string> const& match_methods, bool const overwrite = false) {
        if (!aggregateMap) throw std::logic_error("No simulation data to process");
        if (gtMap && !gtMLMap) gtMLMap = merge<3>(make_vector(gtMap.get()));
        if (!aggregateMLMap) aggregateMLMap = merge<4>(make_vector(aggregateMap.get()));

        results["Bagfiles"] = json(bagfiles);
        results["Image Topic"] = image_topic;
        results["Depth Topic"] = depth_topic;
        results["Segmentation Topic"] = segmentation_topic;
        results["Number of Robots"] = bagfiles.size();
        results["Parameters"] = json(params);
        if (gtMLMap) {
            auto const sr_gt_metrics = compute_metrics(*gtMLMap, *aggregateMLMap);
            results["Single Robot GT-MI"] = std::get<0>(sr_gt_metrics);
            results["Single Robot GT-NMI"] = std::get<1>(sr_gt_metrics);
            results["Single Robot GT-AMI"] = std::get<2>(sr_gt_metrics);
        }

        assert(robotMaps.size() == robotModels.size());
        for (auto const& method : match_methods) {
            if (overwrite || !results.contains(method)) results["Match Results"][method] = match(robotModels, robotMaps, method);
        }
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
        ar & robotMaps;
        ar & robotModels;
        std::string resultsStr;
        if (results.empty()) {
            ar & resultsStr;
            if (!resultsStr.empty()) results = json::parse(resultsStr);
        } else {
            resultsStr = results.dump();
            ar & resultsStr;
        }
    }

    [[nodiscard]] json const& getResults() const {
        return results;
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "multi_agent_sim");
    ros::NodeHandle nh("~");

    auto output_prefix = nh.param<std::string>("output_prefix", "");
    output_prefix += (output_prefix.empty() || output_prefix.back() == '/') ? "" : "/";
    auto const data_filename = output_prefix + "data.bin.zz";
    auto const results_filename = output_prefix + "results.json";

    std::vector<std::string> match_methods = {"id", "hungarian-l1", "clear-l1"};
    auto const methods_str = nh.param<std::string>("match_methods", "");
    if (!methods_str.empty()) {
        match_methods = sunshine::split(methods_str, ',');
    }

    std::vector<MultiAgentSimulation> simulations;
    if (argc >= 2 && std::string(argv[1]) == "record") {
        if (argc < 3) {
            std::cerr << "Usage: ./multi_agent_sim record [n_trials=1] bagfiles..." << std::endl;
            return 1;
        }
        auto const image_topic_name = nh.param<std::string>("image_topic", "/camera/rgb/image_color");
        auto const depth_topic_name = nh.param<std::string>("depth_cloud_topic", "/camera/points");
        auto const segmentation_topic_name = nh.param<std::string>("segmentation_topic", "/camera/seg/image_color");

        bool const numeric = std::string(argv[2]).find_first_not_of("0123456789") == std::string::npos;
        int const DEFAULT_TRIALS = 1;
        auto const n_trials = (numeric) ? std::stoi(argv[2]) : DEFAULT_TRIALS;
        if (numeric && argc < 4) {
            std::cerr << "Usage: ./multi_agent_sim record [n_trials=1] bagfiles..." << std::endl;
            return 1;
        }

        std::vector<std::string> bagfiles;
        int const offset = (numeric) ? 3 : 2;
        bagfiles.reserve(argc - offset);
        for (auto i = offset; i < argc; ++i) bagfiles.emplace_back(argv[i]);

        for (auto i = 0; i < n_trials; ++i) {
            ROS_INFO("Starting simulation %d", i);
            simulations.emplace_back(bagfiles, image_topic_name, depth_topic_name, segmentation_topic_name);
            try {
                if (!simulations.back().record(nh)) throw std::runtime_error("Simulation aborted");
            } catch (std::exception const& ex) {
                ROS_ERROR("Simulation %d failed -- aborting and discarding partial simulation data", i);
                simulations.pop_back();
                break;
            }
        }
    } else if (argc >= 2 && std::string(argv[1]) == "replay") {
        if (argc != 3) {
            std::cerr << "Usage: ./multi_agent_sim replay data_file.bin.zz" << std::endl;
            return 1;
        }
        CompressedFileReader reader(argv[2]);
        reader >> simulations;
    } else {
        std::cerr << "Usage: ./multi_agent_sim <record|replay> <args...>" << std::endl;
        return 1;
    }

    if (ros::ok()) {
        json results = json::array();
        for (auto& sim : simulations) {
            sim.process(match_methods);
            results.push_back(sim.getResults());
        }

        std::ofstream resultsWriter(results_filename);
        resultsWriter << results.dump(2);
        resultsWriter.close();
    } else {
        ROS_WARN("ROS is shutting down -- writing simulation data but skipping result processing");
    }
    CompressedFileWriter writer(data_filename);
    writer << simulations;

    return 0;
}

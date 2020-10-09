#include "sunshine/2d_adapter.hpp"
#include "sunshine/external/json.hpp"
#include "sunshine/common/simulation_utils.hpp"
#include "sunshine/common/matching_utils.hpp"
#include "sunshine/common/segmentation_utils.hpp"
#include "sunshine/common/metric.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/data_proc_utils.hpp"
#include "sunshine/common/thread_pool.hpp"
#include "sunshine/common/utils.hpp"
#include "sunshine/observation_transform_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"

#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/map.hpp>
#include <sunshine/common/ros_conversions.hpp>
#include <utility>
#include <iostream>
#include <random>

using namespace sunshine;
using json = nlohmann::ordered_json;


class MultiAgentSimulation {
    std::vector<std::string> bagfiles;
    std::string image_topic;
    std::string depth_topic;
    std::string segmentation_topic;
    std::map<std::string, std::string> params;
    decltype(std::declval<RobotSim>().getGTMap()) gtMap;
    std::vector<std::result_of_t<decltype(&RobotSim::getDistMap)(RobotSim, activity_manager::ReadToken const&)>> aggregateSubmaps;
    std::result_of_t<decltype(&RobotSim::getDistMap)(RobotSim, activity_manager::ReadToken const&)> aggregateMap;
    std::vector<std::vector<std::result_of_t<decltype(&RobotSim::getDistMap)(RobotSim, activity_manager::ReadToken const&)>>> robotMaps;
    std::vector<std::vector<Phi>> robotModels;

    // Derived data -- not serialized (can be recovered exactly from above data)
    std::unique_ptr<Segmentation<int, 3, int, double>> gtMLMap;
//    std::unique_ptr<Segmentation<int, 4, int, double>> aggregateMLMap;
    std::unique_ptr<std::set<decltype(decltype(gtMap)::element_type::observation_poses)::value_type>> gtPoses;
    std::shared_ptr<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>> sharedSegmentationAdapter;
    [[deprecated]] json results = json::object();

  protected:
    [[nodiscard]] static std::tuple<json, std::unique_ptr<Segmentation<int, 4, int, double>>, match_results> match(
                                    Segmentation<int, 4, int, double> const* aggregateMLMap,
                                    decltype(robotModels) const& topic_models,
                                    decltype(robotMaps) const& topic_maps,
                                    std::string const& match_method,
                                    size_t const& n_robots,
                                    Segmentation<int, 3, int, double> const* gtMLMap = nullptr,
                                    size_t subsample = 30,
                                    bool include_individ = false,
                                    bool include_ssd = false) {
        auto const aggregateLabels = aggregateMLMap->toLookupMap();
        auto const numAggregateTopics = get_num_topics(*aggregateMLMap);
        decltype(gtMLMap->toLookupMap()) gtLabels;
        if (gtMLMap) gtLabels = gtMLMap->toLookupMap();
        auto const numGTTopics = (gtMLMap) ? get_num_topics(*gtMLMap) : 0;

        json results_array = json::array();
        std::unique_ptr<Segmentation<int, 4, int, double>> finalMap;
        match_results correspondences;
        for (auto i = 0ul; i < topic_maps.size(); i += std::max(1ul, std::min(subsample, topic_maps.size() - i - 1))) {
            json match_result;
            match_result["Number of Observations"] = (i + 1);

            if (n_robots <= 0 || n_robots > topic_models[i].size()) throw std::invalid_argument("Invalid number of robots");
            decltype(robotModels)::value_type const models = (n_robots == topic_models[i].size()) ? topic_models[i] : decltype(robotModels)::value_type{topic_models[i].begin(), topic_models[i].begin() + n_robots};
            std::vector<size_t> wordsRefined, cellsRefined;
            for (auto const& model : models) {
                wordsRefined.push_back(model.word_refines);
                cellsRefined.push_back(model.cell_refines);
            }
            match_result["Word Refines"] = wordsRefined;
            match_result["Cell Refines"] = cellsRefined;
            correspondences = match_topics(match_method, models);
            match_result["Unique Topics"] = correspondences.num_unique;
            if (include_ssd) match_result["SSD"] = correspondences.ssd;

            match_scores const scores(models, correspondences.lifting, l1_distance<double>);
            match_result["Cluster Size"] = scores.cluster_sizes;
            match_result["Mean-Square Cluster Distances"] = scores.mscd;
            match_result["Silhouette Indices"] = scores.silhouette;
            match_result["Davies-Bouldin Indices"] = scores.davies_bouldin;

            uint32_t matched = 0;
            for (auto const size : scores.cluster_sizes) { matched += (size > 1); }
            ROS_DEBUG("%s matched clusters: %d/%d", match_method.c_str(), matched, correspondences.num_unique);

            std::vector<Segmentation<std::vector<int>, 4, int, double> const*> maps;
            for (auto j = 0ul; j < n_robots; ++j) maps.emplace_back(topic_maps[i][j].get());
            auto merged_segmentations = merge_segmentations(maps, correspondences.lifting);
            match_result["Number of Cells"] = merged_segmentations->observation_poses.size();
            {
                auto const sr_ref_metrics = compute_metrics(aggregateLabels, numAggregateTopics, *merged_segmentations, false);
                match_result["SR-MI"]     = std::get<0>(sr_ref_metrics);
                match_result["SR-NMI"]    = std::get<1>(sr_ref_metrics);
                match_result["SR-AMI"]    = std::get<2>(sr_ref_metrics);
                ROS_DEBUG("%s SR AMI: %f", match_method.c_str(), std::get<2>(sr_ref_metrics));
            }

            if (gtMLMap) {
                #ifndef NDEBUG
                std::set<std::array<int, 3>> merged_poses;
                for (auto const& pose : merged_segmentations->observation_poses) {
                    uint32_t const offset = pose.size() == 4;
                    merged_poses.insert({pose[offset], pose[1 + offset], pose[2 + offset]});
                }
                auto const gtPoses = std::make_unique<std::set<std::array<int, 3>>>(gtMLMap->observation_poses.cbegin(), gtMLMap->observation_poses.cend());
                if (!includes(*gtPoses, merged_poses)) {
                    ROS_ERROR_ONCE("Ground truth is missing merged poses!");
//                    throw std::logic_error("Ground truth is missing poses!");
                }
                #endif

                align(*merged_segmentations, gtLabels, numGTTopics);
                {
                    auto const gt_ref_metrics = compute_metrics(gtLabels, numGTTopics, *merged_segmentations);
                    match_result["GT-MI"]     = std::get<0>(gt_ref_metrics);
                    match_result["GT-NMI"]    = std::get<1>(gt_ref_metrics);
                    match_result["GT-AMI"]    = std::get<2>(gt_ref_metrics);
                    ROS_DEBUG("%s GT AMI: %f", match_method.c_str(), std::get<2>(gt_ref_metrics));
                }

                if (include_individ) {
                    std::vector<double> individ_mi, individ_nmi, individ_ami;
                    for (auto const &segmentation : maps) {
                        auto const individ_metric = compute_metrics(gtLabels, numGTTopics, *segmentation);
                        individ_mi.push_back(std::get<0>(individ_metric));
                        individ_nmi.push_back(std::get<1>(individ_metric));
                        individ_ami.push_back(std::get<2>(individ_metric));
                    }
                    match_result["Individual GT-MI"] = individ_mi;
                    match_result["Individual GT-NMI"] = individ_nmi;
                    match_result["Individual GT-AMI"] = individ_ami;
                }
            }
            results_array.push_back(match_result);
            finalMap = std::move(merged_segmentations);
        }
        return std::make_tuple(results_array, std::move(finalMap), correspondences);
    }

  public:
    static size_t constexpr VERSION = 1;

    template <class ParamServer>
    class ParamPassthrough {
        MultiAgentSimulation* parent;
        ParamServer const* nh;
        mutable std::mutex lock;

      public:
        ParamPassthrough(MultiAgentSimulation* parent, ParamServer const* nh) : parent(parent), nh(nh) {}
        template <typename T>
        T param(std::string param_name, T default_value) const {
            std::lock_guard<std::mutex> guard(lock);
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

    template <typename ParamServer>
    bool populate_aggregate(ParamServer const& nh) {
        if (aggregateMap) throw std::logic_error("Already populated aggregate");
        ParamPassthrough<ParamServer> paramPassthrough(this, &nh);
        std::cout << "Running single-agent simulation...." << std::endl;
        sharedSegmentationAdapter = std::make_shared<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>>(&nh, true);
        RobotSim aggregateRobot("Aggregate", paramPassthrough, !depth_topic.empty(), sharedSegmentationAdapter);
        for (auto const& bag : bagfiles) {
            aggregateRobot.open(bag, image_topic, depth_topic, segmentation_topic);
            while (aggregateRobot.next()) {
                if (!ros::ok()) return false;
                aggregateRobot.waitForProcessing();
            }
            aggregateRobot.waitForProcessing();
            aggregateSubmaps.push_back(aggregateRobot.getDistMap(*aggregateRobot.getReadToken()));
        }
        aggregateRobot.pause();
        aggregateMap = aggregateRobot.getDistMap(*aggregateRobot.getReadToken());

        if (!segmentation_topic.empty()) {
            gtMap = aggregateRobot.getGTMap();
            gtPoses = std::make_unique<std::set<std::array<int, 3>>>(gtMap->observation_poses.cbegin(), gtMap->observation_poses.cend());
            gtMLMap = merge_segmentations<3>(make_vector(gtMap.get()));
        }
        return true;
    }

    template<typename ParamServer>
    bool record(ParamServer const& nh, size_t const subsample = 1) {
        ParamPassthrough<ParamServer> paramPassthrough(this, &nh);
        if (!aggregateMap) populate_aggregate(nh); // needed to setup the sharedSegmentationAdapter
        assert(aggregateMap && sharedSegmentationAdapter);

        std::set<std::array<int, 3>> aggregate_poses;
        for (auto const& pose : aggregateMap->observation_poses) {
            uint32_t const offset = pose.size() == 4;
            aggregate_poses.insert({pose[offset], pose[1 + offset], pose[2 + offset]});
        }
        if (gtPoses && !includes(*gtPoses, aggregate_poses)) {
            ROS_ERROR_ONCE("Ground truth is missing aggregate poses!");
        }

        //    std::vector<ros::Publisher> map_pubs;
        std::vector<std::unique_ptr<RobotSim>> robots;
        for (auto i = 0; i < bagfiles.size(); ++i) {
            robots.emplace_back(std::make_unique<RobotSim>(std::to_string(i), paramPassthrough, !depth_topic.empty(), sharedSegmentationAdapter, true));
            robots.back()->open(bagfiles[i], image_topic, depth_topic, segmentation_topic, i * 2000);
            //        map_pubs.push_back(nh.advertise<sunshine_msgs::TopicMap>("/" + robots.back()->getName() + "/map", 0));
        }

        bool active = true;
        size_t n_obs = 0;
        size_t iter = 0;
        auto start = std::chrono::steady_clock::now();
        while (active) {
            active = false;
            for (auto &robot : robots) {
                if (!ros::ok()) return false;
                auto const robot_active = robot->next();
                active = active || robot_active;
            }
            n_obs += active;

            auto const read_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            std::cout << "Read time: " << read_time << std::endl;

            robotMaps.emplace_back();
            robotMaps.back().reserve(robots.size());
            robotModels.emplace_back();
            robotModels.back().reserve(robots.size());
            for (auto const& robot : robots) {
                if (!ros::ok()) return false;
                robot->waitForProcessing();
                if (iter % subsample == 0) {
                    auto token = robot->getReadToken();
                    robotModels.back().emplace_back(robot->getTopicModel(*token));
                    robotMaps.back().emplace_back(robot->getDistMap(*token));
                }
            }

            auto const refine_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            std::cout << "Refine time: " << refine_time << std::endl;

            #ifndef NDEBUG
            for (auto const& map : robotMaps.back()) {
                std::set<std::array<int, 3>> map_poses;
                for (auto const& pose : map->observation_poses) {
                    uint32_t const offset = pose.size() == 4;
                    map_poses.insert({pose[offset], pose[1 + offset], pose[2 + offset]});
                }
                if (!includes(aggregate_poses, map_poses)) {
                    std::vector<std::array<int, 3>> diff_poses;
                    std::set_difference(map_poses.begin(), map_poses.end(), aggregate_poses.begin(), aggregate_poses.end(), std::back_inserter(diff_poses));
                    ROS_ERROR("Ground truth is missing merged poses at obs %lu!", n_obs);
//                throw std::logic_error("Ground truth is missing poses!");
                }
            }
            auto const merged_map = merge_segmentations(robotMaps.back());
            std::set<std::array<int, 3>> merged_poses;
            for (auto const& pose : merged_map->observation_poses) {
                uint32_t const offset = pose.size() == 4;
                merged_poses.insert({pose[offset], pose[1 + offset], pose[2 + offset]});
            }
            if (!includes(*gtPoses, merged_poses)) {
                ROS_ERROR("Ground truth is missing merged poses at obs %lu!", n_obs);
//                throw std::logic_error("Ground truth is missing poses!");
            }
            if (!includes(aggregate_poses, merged_poses)) {
                ROS_ERROR("Aggregate poses are missing merged poses at obs %lu!", n_obs);
//                throw std::logic_error("Ground truth is missing poses!");
            }
            #endif

            ROS_INFO("Approximate simulation size: %f MB", approxBytesSize() / (1024.0 * 1024.0));
            start = std::chrono::steady_clock::now();
            iter += 1;
        }
        return true;
    }

    json process(std::vector<std::string> const& match_methods, size_t n_robots = 0, std::string const& map_prefix = "", std::string const& map_box = "", bool parallel=false) const {
        if (n_robots == 0) n_robots = bagfiles.size();
        else if (n_robots < 0 || n_robots > bagfiles.size()) throw std::invalid_argument("Invalid number of robots!");
        if (robotMaps.empty()) throw std::logic_error("No simulation data to process");

        WordColorMap<int> colorMap;
        auto const refMap = (aggregateSubmaps.empty()) ? merge_segmentations<4>(make_vector(aggregateMap.get()))
                : merge_segmentations<4>(make_vector(aggregateSubmaps[n_robots - 1].get()));
        double const pixel_scale = *std::min_element(refMap->cell_size.begin(), refMap->cell_size.end());
        if (!map_prefix.empty()) {
            auto mapImg = createTopicImg(toRosMsg(*refMap), colorMap, pixel_scale, true, 0, 0, map_box);
            saveTopicImg(mapImg, map_prefix + "-aggregate.png", map_prefix + "-aggregate-colors.csv", &colorMap);
        }

        json exp_results = json::object();
        exp_results["Bagfiles"] = json(std::vector<std::string>(bagfiles.begin(), bagfiles.begin() + n_robots));
        exp_results["Image Topic"] = image_topic;
        exp_results["Depth Topic"] = depth_topic;
        exp_results["Segmentation Topic"] = segmentation_topic;
        exp_results["Number of Robots"] = n_robots;
        exp_results["SR Number of Cells"] = refMap->observation_poses.size();
        exp_results["Parameters"] = json(params);
        if (gtMLMap) {
            exp_results["GT Number of Cells"] = gtMLMap->observation_poses.size();

            auto const gtLabels = gtMLMap->toLookupMap();
            auto const numGTTopics = get_num_topics(*gtMLMap);

            auto const sr_gt_metrics = compute_metrics(gtLabels, numGTTopics, *refMap);
            exp_results["Single Robot GT-MI"] = std::get<0>(sr_gt_metrics);
            exp_results["Single Robot GT-NMI"] = std::get<1>(sr_gt_metrics);
            exp_results["Single Robot GT-AMI"] = std::get<2>(sr_gt_metrics);
            if (!map_prefix.empty()) {
                auto mapImg = createTopicImg(toRosMsg(*gtMLMap), colorMap, pixel_scale, true, 0, 0, map_box);
                saveTopicImg(mapImg, map_prefix + "-gt.png", map_prefix + "-gt-colors.csv", &colorMap);
            }
        }

        assert(robotMaps.size() == robotModels.size());
        std::vector<std::thread> workers;
        std::mutex result_lock;
        for (auto const& method : match_methods) {
            workers.emplace_back([this, &refMap, method, n_robots, &exp_results, &result_lock, &colorMap, pixel_scale, map_box, map_prefix](){
                ROS_INFO("Matching %ld robots with %s", n_robots, method.c_str());
                auto [stats, map, final_results] = match(refMap.get(), robotModels, robotMaps, method, n_robots, gtMLMap.get());
                std::lock_guard<std::mutex> guard(result_lock);
                exp_results["Match Results"][method] = stats;
                exp_results[method]["Final Distances"] = final_results.distances;
                if (!map_prefix.empty()) {
                    auto mapImg = createTopicImg(toRosMsg(*map), colorMap, pixel_scale, true, 0, 0, map_box);
                    saveTopicImg(mapImg, map_prefix + "-" + method + ".png", map_prefix + "-" + method + "-colors.csv", &colorMap);
                }
            });
            if (!parallel) workers.back().join();
        }
        if (parallel) for (auto& worker : workers) worker.join();
        return exp_results;
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
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
        if (version == 0) {
            std::string resultsStr;
            if (results.empty()) {
                ar & resultsStr;
                if (!resultsStr.empty()) results = json::parse(resultsStr);
            } else {
                resultsStr = results.dump();
                ar & resultsStr;
            }
        } else if (version >= 1) {
            ar & aggregateSubmaps;
        }
        if (gtMap && !gtMLMap) gtMLMap = merge_segmentations<3>(make_vector(gtMap.get()));
    }
#pragma clang diagnostic pop

    [[nodiscard]] size_t approxBytesSize() const {
        auto gtMapSize = gtMap->bytesSize();
        auto aggregateMapSize = aggregateMap->bytesSize();
        size_t robotMapsSize = 0;
        size_t robotModelsSize = 0;
        for (size_t i = 0ul; i < robotMaps.size(); ++i) {
            for (size_t j = 0ul; j < robotMaps[i].size(); ++j) {
                robotMapsSize += robotMaps[i][j]->bytesSize();
                robotModelsSize += robotModels[i][j].bytesSize();
            }
        }
        return gtMapSize + aggregateMapSize + robotMapsSize + robotModelsSize;
    }

    size_t getNumRobots() const {
        return bagfiles.size();
    }
};
BOOST_CLASS_VERSION(MultiAgentSimulation, MultiAgentSimulation::VERSION)

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./multi_agent_sim <record|replay> <args...>" << std::endl;
        return 1;
    }
    std::string const mode(argv[1]);
    ros::init(argc, argv, "multi_agent_sim");
    ros::NodeHandle nh("~");

    auto output_prefix = nh.param<std::string>("output_prefix", "/home/stewart/workspace/");
    auto file_prefix = nh.param<std::string>("file_prefix", "");
    output_prefix += (output_prefix.empty() || output_prefix.back() == '/') ? "" : "/";
    file_prefix += (file_prefix.empty() || file_prefix.back() == '-') ? "" : "-";
    auto const results_filename = output_prefix + file_prefix + "results.json";

    std::vector<std::string> match_methods = {"id", "hungarian-l1", "clear-l1", "clear-l1-0.25"};
    auto const methods_str = nh.param<std::string>("match_methods", "");
    if (!methods_str.empty()) {
        match_methods = sunshine::split(methods_str, ',');
    }

    std::vector<std::string> data_files;
    if (argc >= 2 && mode == "record") {
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

        std::vector<std::unique_ptr<MultiAgentSimulation>> sims;
        sims.reserve(n_trials);
        thread_pool pool(std::max(1u, std::thread::hardware_concurrency() / (1 + nh.param<int>("num_threads", 2))));
        auto const batch_size = pool.size();
        assert(batch_size > 0);
        size_t n_ready = 0;
        size_t n_done = 0;
        auto rng = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
        while (n_done < n_trials && ros::ok())
        {
            for (size_t i = 0; i < batch_size && n_ready < n_trials; ++i, ++n_ready)
            {
                std::shuffle(bagfiles.begin(), bagfiles.end(), rng);
                sims.push_back (std::make_unique<MultiAgentSimulation> (bagfiles,
                                                                        image_topic_name,
                                                                        depth_topic_name,
                                                                        segmentation_topic_name));
                pool.enqueue ([ptr = sims.back ().get (), &nh] { ptr->populate_aggregate (nh); });
            }
            pool.join ();

            for (; n_done < n_ready && ros::ok (); ++n_done)
            {
                auto const data_filename =
                    output_prefix + file_prefix + "raw-" + std::to_string (n_done + 1) + "-of-" + std::to_string (n_trials) + ".bin.zz";
                ROS_INFO("Starting simulation %ld", n_done + 1);
                try
                {
                    auto const cell_space = nh.param<double>("cell_space", 0.8);
                    if (!sims[n_done]->record(nh, (cell_space < 0.8) ? 2 : 1)) throw std::runtime_error ("Simulation aborted");
                }
                catch (std::exception const &ex)
                {
                    ROS_ERROR("Simulation %ld failed.", n_done + 1);
                    continue;
                }
                CompressedFileWriter writer (data_filename);
                writer << *(sims[n_done]);
                data_files.push_back (data_filename);
                sims[n_done].reset (); // free up some memory
                ROS_INFO("Finished simulation %ld", n_done + 1);
            }
        }
    } else if (mode == "replay") {
        for (auto i = 2; i < argc; ++i) {
            data_files.emplace_back(argv[i]);
        }
    }

    if (!ros::ok()) {
        ROS_WARN("ROS is shutting down -- writing simulation data but skipping result processing");
        return 2;
    }
    json results = json::array();
    std::mutex resultsMutex;
    std::vector<std::thread> workers;

    auto map_prefix = nh.param<std::string>("map_prefix", "/home/stewart/workspace/");
    map_prefix += (map_prefix.empty() || map_prefix.back() == '/') ? "" : "/";
    map_prefix += (map_prefix.empty()) ? "" : file_prefix;
    auto const map_box = nh.param<std::string>("box", "-150x-150x300x300");
    auto const parallel_match = nh.param<bool>("parallel_match", false);
    size_t const n_parallel_match = (parallel_match) ? match_methods.size() : 1;
    size_t const cap_parallel_sim = nh.param<int>("max_parallel_sim", 1);
    size_t const max_parallel_sim = std::min(cap_parallel_sim, std::max(1ul, std::thread::hardware_concurrency() / n_parallel_match));
    auto const parallel_nrobots = nh.param<bool>("parallel_nrobots", true);
    std::mutex simMutex;
    size_t n_parallel_sim = 0;
    std::condition_variable sim_cv;
    size_t run = 1;
    for (auto const& file : data_files) {
        {
            std::lock_guard<std::mutex> guard(simMutex);
            n_parallel_sim++;
        }
        workers.emplace_back([&resultsMutex, &results, map_prefix, map_box, run, file, match_methods, parallel_match, parallel_nrobots, &simMutex, &sim_cv, &n_parallel_sim](){
            ROS_INFO("Reading file %s", file.c_str());
            MultiAgentSimulation sim;
            try
            {
                CompressedFileReader reader(file);
                reader >> sim;
            } catch (boost::archive::archive_exception const& e) {
                CompressedFileReader reader(file);
                std::unique_ptr<MultiAgentSimulation> simPtr;
                reader >> simPtr;
                sim = std::move(*simPtr);
            }
            std::vector<std::thread> subworkers;
            for (auto i = 1; i <= sim.getNumRobots(); ++i) {
                subworkers.emplace_back([file, i, map_prefix, run, match_methods, map_box, &sim, &resultsMutex, &results, parallel_match](){
                    ROS_INFO("Matching file %s with %d robots", file.c_str(), i);
                    std::string prefix = map_prefix;
                    if (!prefix.empty()) {
                        prefix += "exp" + std::to_string(run) + "-" + std::to_string(i) + "robots";
                    }
                    auto const results_i = sim.process(match_methods, i, prefix, map_box, parallel_match);
                    if (!ros::ok()) {
                        ROS_WARN("Stopping file %s...", file.c_str());
                        return;
                    }
                    std::lock_guard<std::mutex> guard(resultsMutex);
                    results.push_back(results_i);
                    ROS_INFO("Finished file %s with %d robots", file.c_str(), i);
                });
                if (!parallel_nrobots) subworkers.back().join();
            }
            if (parallel_nrobots) for (auto& worker : subworkers) worker.join();
            {
                std::unique_lock<std::mutex> lk(simMutex);
                n_parallel_sim--;
            }
            sim_cv.notify_all();
        });
        std::unique_lock<std::mutex> lk(simMutex);
        if (n_parallel_sim >= max_parallel_sim) sim_cv.wait(lk, [&n_parallel_sim, max_parallel_sim]{return n_parallel_sim < max_parallel_sim;});
        run += 1;
    }
    for (auto& worker : workers) worker.join();
    if (!ros::ok()) ROS_WARN("ROS is shutting down -- saving partial results");

    std::ofstream resultsWriter(results_filename);
    resultsWriter << results.dump(2);
    resultsWriter.close();

    return 0;
}

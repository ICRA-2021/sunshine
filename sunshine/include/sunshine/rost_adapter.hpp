#ifndef TOPIC_MODEL_HPP
#define TOPIC_MODEL_HPP

#include <array>
#include <memory>
#include <rost/hlda.hpp>
#include <rost/rost.hpp>
#include <utility>
#include <future>
#include <list>
#include "sunshine/common/utils.hpp"
#include "sunshine/common/observation_types.hpp"
#include "sunshine/common/observation_adapters.hpp"
#include "sunshine/common/sunshine_types.hpp"
#include <ros/console.h>

namespace sunshine {

using warp::ROST;
using warp::hROST;

template<size_t _POSEDIM = 4, typename WordInputDimType = double, typename WordOutputDimType = double>
class ROSTAdapter : public Adapter<ROSTAdapter<_POSEDIM>, CategoricalObservation<int,
      _POSEDIM - 1, WordInputDimType>, std::future<Segmentation<std::vector<int>, _POSEDIM, int32_t, WordOutputDimType>>> {
  public:
    static size_t constexpr POSEDIM = _POSEDIM;
    typedef WordOutputDimType WordDimType;
    typedef int32_t CellDimType;
    typedef std::array<CellDimType, POSEDIM> cell_pose_t;
    typedef std::array<WordDimType, POSEDIM> word_pose_t;
    typedef neighbors<cell_pose_t> neighbors_t;
    typedef warp::SpatioTemporalTopicModel<cell_pose_t, neighbors_t, hash_container<cell_pose_t>> ROST_t;
    using WordObservation = CategoricalObservation<int, POSEDIM - 1, WordInputDimType>;
  private:
    mutable std::mutex wordsReceivedLock;
    std::chrono::steady_clock::time_point lastWordsAdded;
    mutable int consecutive_rate_violations = 0;

    int K, V;
    double last_time = -1; //number of topic types, number of word types
    std::array<WordDimType, POSEDIM> cell_size;
    double k_alpha, k_beta, k_gamma, k_tau, p_refine_rate_local, p_refine_rate_global;
    CellDimType G_time, G_space;
    int num_threads, min_obs_refine_time, obs_queue_size;
    bool polled_refine, update_topic_model;
    size_t last_refine_count = 0;
    std::unique_ptr<ROST_t> rost;
    std::string world_frame;

    std::vector<double> observation_times; //list of all time seq ids observed thus far.
    std::vector<cell_pose_t> current_cell_poses, last_poses;

    std::atomic<bool> stopWork;
    std::vector<std::shared_ptr<std::thread>> workers;
    std::function<void(ROSTAdapter *)> newObservationCallback;
    std::list<std::unique_ptr<std::thread>> observationThreads;

    static std::map<cell_pose_t, std::vector<int>> words_for_cell_poses(WordObservation const &wordObs,
                                                                        std::array<WordDimType, POSEDIM> cell_size) {
        using namespace std;
        map<cell_pose_t, vector<int>> words_by_cell_pose;

        for (size_t i = 0; i < wordObs.observations.size(); ++i) {
            word_pose_t wordPose;
            if constexpr(POSEDIM == 4) {
                wordPose = {wordObs.timestamp, wordObs.observation_poses[i][0], wordObs.observation_poses[i][1],
                      wordObs.observation_poses[i][2]};
            } else if constexpr(POSEDIM == 3) {
                wordPose = {wordObs.timestamp, wordObs.observation_poses[i][0], wordObs.observation_poses[i][1]};
            } else {
                static_assert(always_false<POSEDIM>);
            }
            cell_pose_t const cell_stamped_point = toCellId(wordPose, cell_size);
            words_by_cell_pose[cell_stamped_point].emplace_back(wordObs.observations[i]);
        }
        return words_by_cell_pose;
    }

    std::vector<std::vector<int>> getTopicDistsForPoses(const std::vector<cell_pose_t> &cell_poses) const {
        std::vector<std::vector<int>> topics;
        topics.reserve(cell_poses.size());
        for (auto const &pose : cell_poses) {
            auto const cell = rost->get_cell(pose);
            if (cell->nZ.size() == this->K) {
                topics.push_back(cell->nZ);
            } else {
                ROS_WARN_THROTTLE(1, "ROSTAdapter::getObservationForPoses() : Skipping cells with wrong number of topics");
                continue;
            }
        }

        return topics;
    }

  public:
#ifndef NDEBUG
    std::vector<std::vector<int>> externalTopicCounts = {}; // TODO Delete?
#endif

    template<typename ParamServer>
    explicit ROSTAdapter(ParamServer *nh, decltype(newObservationCallback) callback = nullptr)
          : newObservationCallback(std::move(callback)) {
        K = nh->template param<int>("K", 100); // number of topics
        V = nh->template param<int>("V", 1500); // vocabulary size
        bool const is_hierarchical = nh->template param<bool>("hierarchical", false);
        int const num_levels = nh->template param<int>("num_levels", 3);
        k_alpha = nh->template param<double>("alpha", 0.1);
        k_beta = nh->template param<double>("beta", 1.0);
        k_gamma = nh->template param<double>("gamma", 0.0001);
        k_tau = nh->template param<double>("tau", 0.5); // beta(1,tau) is used to pick cells for global refinement
        p_refine_rate_local = nh->template param<double>("p_refine_rate_local", 0.5); // probability of refining last observation
        p_refine_rate_global = nh->template param<double>("p_refine_rate_global", 0.5);
        num_threads = nh->template param<int>("num_threads", 4); // beta(1,tau) is used to pick cells for refinement
        double const cell_size_space = nh->template param<double>("cell_space", 1);
        double const cell_size_time = nh->template param<double>("cell_time", 1);
        std::string const cell_size_string = nh->template param<std::string>("cell_size", "");
        G_time = nh->template param<CellDimType>("G_time", 1);
        G_space = nh->template param<CellDimType>("G_space", 1);
        polled_refine = nh->template param<bool>("polled_refine", false);
        update_topic_model = nh->template param<bool>("update_topic_model", true);
        min_obs_refine_time = nh->template param<int>("min_obs_refine_time", 200);
        obs_queue_size = nh->template param<int>("word_obs_queue_size", 1);
        world_frame = nh->template param<std::string>("world_frame", "");

        if (!cell_size_string.empty()) {
            cell_size = readNumbers<POSEDIM, 'x', WordDimType>(cell_size_string);
        } else {
            cell_size = computeCellSize<POSEDIM, WordDimType>(cell_size_time, cell_size_space);
        }

//        ROS_INFO("Starting online topic modelling with parameters: K=%u, alpha=%f, beta=%f, tau=%f", K, k_alpha, k_beta, k_tau);

        cell_pose_t const G = computeCellSize<POSEDIM, CellDimType>(G_time, G_space);
        if (is_hierarchical) {
//            ROS_INFO("Enabling hierarchical ROST with %d levels, gamma=%f", num_levels, k_gamma);
            rost = std::make_unique<hROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t >>>(static_cast<size_t>(V),
                                                                                                   static_cast<size_t>(K),
                                                                                                   static_cast<size_t>(num_levels),
                                                                                                   k_alpha,
                                                                                                   k_beta,
                                                                                                   k_gamma,
                                                                                                   neighbors_t(G));
        } else {
            rost = std::make_unique<ROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t >>>(static_cast<size_t>(V),
                                                                                                  static_cast<size_t>(K),
                                                                                                  k_alpha,
                                                                                                  k_beta,
                                                                                                  neighbors_t(G));
            if (k_gamma > 0) {
//                ROS_INFO("Enabling HDP with gamma=%f", k_gamma);
                auto rost_concrete = dynamic_cast<ROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t>> *>(rost.get());
                rost_concrete->gamma = k_gamma;
                rost_concrete->enable_auto_topics_size(true);
            }
        }

        if (polled_refine) { //refine when requested
            throw std::runtime_error("Not implemented. Requires services.");
//            ROS_INFO("Topics will be refined on request.");
        } else { //refine automatically
//            ROS_INFO("Topics will be refined online.");
            stopWork = false;
            workers = parallel_refine_online_exp_beta(rost.get(), k_tau, p_refine_rate_local, p_refine_rate_global, num_threads, &stopWork);
        }
    }

    ~ROSTAdapter() override {
        stopWork = true; //signal workers to stop
        for (auto const &t : workers) { //wait for them to stop
            if (t) t->join();
        }
    }

    std::future<Segmentation<std::vector<int>, POSEDIM, CellDimType, WordDimType>> operator()(WordObservation const &wordObs) {
        auto time_checkpoint = std::chrono::steady_clock::now();
        auto const time_start = time_checkpoint;

        if (wordObs.frame.empty()) ROS_WARN("Received WordObservation with empty frame!");

        if (world_frame.empty()) { world_frame = wordObs.frame; }
        else if (wordObs.frame != world_frame) {
            ROS_ERROR("Word observation in wrong frame! Skipping...\nFound: %s\nExpected: %s", wordObs.frame.c_str(), world_frame.c_str());
            throw std::invalid_argument("Word observation in invalid frame.");
        }

        using namespace std;
        lock_guard<mutex> guard(wordsReceivedLock);
        auto duration_words_lock = record_lap(time_checkpoint);
        long duration_write_lock;

        // TODO Can observation transform ever be invalid?

        double observation_time = wordObs.timestamp;
        //update the  list of observed time step ids
        observation_times.push_back(observation_time);
        if (!observation_times.empty() && last_time > observation_time) {
            ROS_WARN("Observation received that is older than previous observation!");
        }

        //if we are receiving observations from the next time step, then spit out
        //topics for the current time step.
        if (last_time >= 0) {
            ROS_DEBUG("Received more word observations - broadcasting observations for time %f", last_time);
            newObservationCallback(this);
            size_t const refine_count = rost->get_refine_count();
            ROS_DEBUG("#cells_refined: %u", static_cast<unsigned>(refine_count - last_refine_count));
            last_refine_count = refine_count;
            current_cell_poses.clear();
        }
        last_time = std::max(last_time, observation_time);
        auto const duration_broadcast = record_lap(time_checkpoint);

        ROS_DEBUG("Adding %lu word observations from time %f", wordObs.observations.size(), observation_time);
        {
            auto rostWriteGuard = rost->get_write_token();
            duration_write_lock = record_lap(time_checkpoint);

            auto const &words_by_cell_pose = words_for_cell_poses(wordObs, cell_size);
            current_cell_poses.reserve(current_cell_poses.size() + words_by_cell_pose.size());
            for (auto const &entry : words_by_cell_pose) {
                auto const &cell_pose = entry.first;
                auto const &cell_words = entry.second;
                rost->add_observation(cell_pose, cell_words.begin(), cell_words.end(), update_topic_model);
                current_cell_poses.push_back(cell_pose);
            }
        }
        auto const duration_add_observations = record_lap(time_checkpoint);
        ROS_DEBUG("Refining %lu cells", current_cell_poses.size());

        auto const total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - time_start).count();
        if (total_duration > this->min_obs_refine_time) {
            ROS_WARN(
                  "Words observation overhead: %lu ms (%lu words lock, %lu write lock, %lu broadcast, %lu updating cells) exceeds min refine time %d",
                  total_duration,
                  duration_words_lock,
                  duration_write_lock,
                  duration_broadcast,
                  duration_add_observations,
                  min_obs_refine_time);
        } else {
            ROS_INFO("Words observation overhead: %lu ms (%lu lock, %lu write lock, %lu broadcast, %lu updating cells)",
                     total_duration,
                     duration_words_lock,
                     duration_write_lock,
                     duration_broadcast,
                     duration_add_observations);
        }

        lastWordsAdded = chrono::steady_clock::now();

        typedef Segmentation<std::vector<int>, POSEDIM, CellDimType, WordDimType> Segmentation;
        std::promise<Segmentation> promisedTopics;
        auto futureTopics = promisedTopics.get_future();
        observationThreads.push_back(std::make_unique<std::thread>([this, id = wordObs.id, startTime = lastWordsAdded, poses = current_cell_poses, refineTime = min_obs_refine_time, cell_poses = current_cell_poses, promisedTopics{
              std::move(promisedTopics)}]() mutable {
            using namespace std::chrono;
            wait_for_processing(false);
            auto topics = getTopicDistsForPoses(cell_poses);
            double const timestamp = duration<double>(steady_clock::now().time_since_epoch()).count();
            promisedTopics.set_value(Segmentation("map", timestamp, id, cell_size, std::move(topics), std::move(poses)));
        }));
        return futureTopics;
    }

    std::map<CellDimType, std::vector<int>> get_topics_by_time() const {
        auto rostReadToken = rost->get_read_token();
        auto const poses_by_time = rost->get_poses_by_time();
        std::map<typename ROST_t::pose_dim_t, std::vector<int>> topics_by_time;
        for (auto const &entry : poses_by_time) {
            std::vector<int> topics(static_cast<size_t>(K), 0);
            for (auto const &pose : entry.second) {
                for (auto const &topic : rost->get_topics_for_pose(pose)) {
                    topics[static_cast<size_t>(topic)] += 1;
                }
            }
            topics_by_time.insert({entry.first * cell_size[0], topics}); // IMPORTANT: Real time not cell time
        }
        return topics_by_time;
    }

    std::map<cell_pose_t, std::vector<int>> get_topics_by_cell() const {
        auto rostReadToken = rost->get_read_token();
        auto const &poses = rost->cell_pose;
        std::map<cell_pose_t, std::vector<int>> topics_by_cell;
        for (auto const &pose : poses) {
            auto const cell = rost->get_cell(pose);
            if (cell->nZ.size() == this->K) {
                topics_by_cell.insert({pose, cell->nZ});
            } else {
                ROS_WARN_THROTTLE(1, "topic_model::get_topics_by_cell() : Skipping cells with wrong number of topics");
                continue;
            }
        }
        return topics_by_cell;
    }

    Phi get_topic_model(activity_manager::ReadToken const &read_token) const {
        Phi phi("", rost->get_num_topics(), rost->get_num_words(), rost->get_topic_model(), rost->get_topic_weights());
        phi.validate(false);
        return phi;
    }

    void set_topic_model(activity_manager::WriteToken const &write_token, Phi const &phi) {
        rost->set_topic_model(write_token, phi.counts, phi.topic_weights);
    }

    void wait_for_processing(bool new_data = true) const {
        using namespace std::chrono;
        auto const elapsedSinceAdd = steady_clock::now() - lastWordsAdded;
        ROS_DEBUG("Time elapsed since last observation added (minimum set to %d ms): %lu ms",
                  min_obs_refine_time,
                  duration_cast<milliseconds>(elapsedSinceAdd).count());
        if (duration_cast<milliseconds>(elapsedSinceAdd).count() < min_obs_refine_time) {
            consecutive_rate_violations++;
            if (new_data) {
                ROS_WARN("New word observation received too soon! Delaying...");
                ROS_ERROR_COND(consecutive_rate_violations > obs_queue_size,
                               "A word observation will likely be dropped. Increase queue size, or reduce observation rate or processing time.");
            }
            std::this_thread::sleep_for(milliseconds(min_obs_refine_time) - elapsedSinceAdd);
        } else {
            consecutive_rate_violations = 0;
        }
    }

    decltype(K) get_num_topics() const {
        return K;
    }

    decltype(last_time) get_last_observation_time() const {
        return last_time;
    }

    decltype(current_cell_poses) const &get_current_cell_poses() const {
        return current_cell_poses;
    }

    decltype(cell_size) const &get_cell_size() const {
        return cell_size;
    }

    ROST_t const &get_rost() const {
        return *rost;
    }

    auto get_cell_topics_and_ppx(activity_manager::ReadToken const &read_token, cell_pose_t const &pose) {
        return rost->get_ml_topics_and_ppx_for_pose(pose);
    }

    static inline cell_pose_t toCellId(word_pose_t const &word, std::array<WordDimType, POSEDIM> cell_size) {
        static_assert(std::numeric_limits<CellDimType>::max() <= std::numeric_limits<WordDimType>::max(),
                      "Word dim type must be larger than cell dim type!");
        if constexpr (POSEDIM == 4) {
            return {static_cast<CellDimType>(std::fmod(word[0] / cell_size[0], std::numeric_limits<CellDimType>::max())),
                  static_cast<CellDimType>(std::fmod(word[1] / cell_size[1], std::numeric_limits<CellDimType>::max())),
                  static_cast<CellDimType>(std::fmod(word[2] / cell_size[2], std::numeric_limits<CellDimType>::max())),
                  static_cast<CellDimType>(std::fmod(word[3] / cell_size[3], std::numeric_limits<CellDimType>::max()))};
        } else if constexpr (POSEDIM == 3) {
            return {static_cast<CellDimType>(std::fmod(word[0] / cell_size[0], std::numeric_limits<CellDimType>::max())),
                  static_cast<CellDimType>(std::fmod(word[1] / cell_size[1], std::numeric_limits<CellDimType>::max())),
                  static_cast<CellDimType>(std::fmod(word[2] / cell_size[2], std::numeric_limits<CellDimType>::max()))};
        } else if constexpr (POSEDIM == 2) {
            return {static_cast<CellDimType>(std::fmod(word[0] / cell_size[0], std::numeric_limits<CellDimType>::max())),
                  static_cast<CellDimType>(std::fmod(word[1] / cell_size[1], std::numeric_limits<CellDimType>::max()))};
        } else {
            static_assert(always_false<POSEDIM>);
        }
    }

    static inline word_pose_t toWordPose(cell_pose_t const &cell, std::array<double, POSEDIM> cell_size) {
        if constexpr(POSEDIM == 4) {
            return {static_cast<WordDimType>(cell[0] * cell_size[0]), static_cast<WordDimType>(cell[1] * cell_size[1]),
                  static_cast<WordDimType>(cell[2] * cell_size[2]), static_cast<WordDimType>(cell[3] * cell_size[3])};
        } else if constexpr(POSEDIM == 3) {
            return {static_cast<WordDimType>(cell[0] * cell_size[0]), static_cast<WordDimType>(cell[1] * cell_size[1]),
                  static_cast<WordDimType>(cell[2] * cell_size[2])};
        } else if constexpr(POSEDIM == 2) {
            return {static_cast<WordDimType>(cell[0] * cell_size[0]), static_cast<WordDimType>(cell[1] * cell_size[1])};
        } else {
            static_assert(always_false<POSEDIM>);
        }
    }
};
}

#endif // TOPIC_MODEL_HPP

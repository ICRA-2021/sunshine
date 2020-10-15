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
        _POSEDIM - 1, WordInputDimType>, std::future<Segmentation<std::vector<int>, _POSEDIM, int32_t, WordOutputDimType>>>
{
  public:
    static size_t constexpr POSEDIM = _POSEDIM;
    typedef WordOutputDimType WordDimType;
    typedef int32_t CellDimType;
    typedef std::array<CellDimType, POSEDIM> cell_pose_t;
    typedef std::array<WordDimType, POSEDIM> word_pose_t;
    typedef neighbors<cell_pose_t> neighbors_t;
//    typedef warp::SpatioTemporalTopicModel<cell_pose_t, neighbors_t, hash_container<cell_pose_t>> ROST_t;
    typedef ROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t >> ROST_t;
    using WordObservation = CategoricalObservation<int, POSEDIM - 1, WordInputDimType>;

    constexpr static double DEFAULT_CELL_SPACE = 1;
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
    bool update_topic_model;
    size_t last_refine_count = 0, min_refines_per_obs = 0;
    std::unique_ptr<ROST_t> rost;
    std::string world_frame;

    std::vector<double> observation_times; //list of all time seq ids observed thus far.
    std::vector<cell_pose_t> current_cell_poses, last_poses;

    std::atomic<bool> stopWork;
    std::vector<std::shared_ptr<std::thread>> workers;
    std::function<void(ROSTAdapter *)> newObservationCallback;
    bool const broadcastMode;
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
            cell_pose_t const cell_stamped_point = sunshine::toCellId<POSEDIM, CellDimType>(wordPose, cell_size);
            words_by_cell_pose[cell_stamped_point].emplace_back(wordObs.observations[i]);
        }
        return words_by_cell_pose;
    }

    template<typename T>
    std::vector<T> getTopicDistsForPoses(const std::vector<cell_pose_t> &cell_poses, std::function<T(std::vector<int>)> const& f) const {
        auto rostReadToken = rost->get_read_token();
        std::vector<T> topics;
        topics.reserve(cell_poses.size());
        for (auto const &pose : cell_poses) {
            auto const cell = rost->get_cell(pose);
            if (cell->nZ.size() == this->K) {
                topics.push_back(f(cell->nZ));
            } else {
                ROS_WARN_THROTTLE(1, "ROSTAdapter::getObservationForPoses() : Skipping cells with wrong number of topics");
                continue;
            }
        }

        return topics;
    }

    std::vector<std::vector<int>> getTopicDistsForPoses(std::vector<cell_pose_t> const& cell_poses) const {
        return getTopicDistsForPoses<std::vector<int>>(cell_poses, [](std::vector<int>const & nZ){return nZ;});
    }

    std::vector<int> getMLTopicsForPoses(const std::vector<cell_pose_t> &cell_poses) const {
        return getTopicDistsForPoses<int>(cell_poses, [](std::vector<int> const& nZ){return argmax(nZ);});
    }

  public:
#ifndef NDEBUG
    std::vector<std::vector<int>> externalTopicCounts = {}; // TODO Delete?
#endif

    template<typename ParamServer>
    explicit ROSTAdapter(ParamServer *nh,
                         decltype(newObservationCallback) callback = nullptr,
                         const std::vector<std::vector<int>> &init_model = {},
                         bool const broadcastMode = true)
            : newObservationCallback(std::move(callback)), stopWork(false), broadcastMode(broadcastMode) {
        K = nh->template param<int>("K", 32); // number of topics
        V = nh->template param<int>("V", 15436); // vocabulary size
        bool const is_hierarchical = nh->template param<bool>("hierarchical", false);
        int const num_levels = nh->template param<int>("num_levels", 3);
        k_alpha = nh->template param<double>("alpha", 0.00803958);
        k_beta = nh->template param<double>("beta", 0.420008);
        k_gamma = nh->template param<double>("gamma", 6.27228e-07);
        k_tau = nh->template param<double>("tau", 0.5); // beta(1,tau) is used to pick cells for global refinement
        p_refine_rate_local = nh->template param<double>("p_refine_rate_local", 0.5); // probability of refining last observation
        p_refine_rate_global = nh->template param<double>("p_refine_rate_global", 0.5);
        num_threads = nh->template param<int>("num_threads", 2); // beta(1,tau) is used to pick cells for refinement
        double const cell_size_space = nh->template param<double>("cell_space", sunshine::ROSTAdapter<>::DEFAULT_CELL_SPACE);
        double const cell_size_time = nh->template param<double>("cell_time", 1);
        std::string const cell_size_string = nh->template param<std::string>("cell_size", "");
        G_time = nh->template param<CellDimType>("G_time", 1);
        G_space = nh->template param<CellDimType>("G_space", 1);
        update_topic_model = nh->template param<bool>("update_topic_model", true);
        min_obs_refine_time = nh->template param<int>("min_obs_refine_time", 30);
        min_refines_per_obs = nh->template param<int>("min_refines_per_obs", 200000);
        obs_queue_size = nh->template param<int>("word_obs_queue_size", 1);
        world_frame = nh->template param<std::string>("world_frame", "map");

        if (!cell_size_string.empty()) {
            cell_size = readNumbers<POSEDIM, 'x', WordDimType>(cell_size_string);
        } else {
            cell_size = computeCellSize<POSEDIM, WordDimType>(cell_size_time, cell_size_space);
        }

//        ROS_INFO("Starting online topic modelling with parameters: K=%u, alpha=%f, beta=%f, tau=%f", K, k_alpha, k_beta, k_tau);

        cell_pose_t const G = computeCellSize<POSEDIM, CellDimType>(G_time, G_space);
//        if (is_hierarchical) {
//            ROS_INFO("Enabling hierarchical ROST with %d levels, gamma=%f", num_levels, k_gamma);
//            rost = std::make_unique<hROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t >>>(static_cast<size_t>(V),
//                                                                                                   static_cast<size_t>(K),
//                                                                                                   static_cast<size_t>(num_levels),
//                                                                                                   k_alpha,
//                                                                                                   k_beta,
//                                                                                                   k_gamma,
//                                                                                                   neighbors_t(G));
//        } else {
        rost = std::make_unique<ROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t >>>(static_cast<size_t>(V),
                                                                                              static_cast<size_t>(K),
                                                                                              k_alpha,
                                                                                              k_beta,
                                                                                              neighbors_t(G),
                                                                                              hash_container<cell_pose_t>(),
                                                                                              (k_gamma > 0) ? k_gamma : 1.0);
        if (k_gamma > 0) {
//                ROS_INFO("Enabling HDP with gamma=%f", k_gamma);
            auto rost_concrete = dynamic_cast<ROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t>> *>(rost.get());
            rost_concrete->enable_auto_topics_size(true);
        }
//        }

        if (!init_model.empty()) {
            if (init_model.size() != K) throw std::invalid_argument("Initial topic model dimension mismatch with number of topics.");
            if (init_model[0].size() != V) throw std::invalid_argument("Initial topic model dimension mismatch with vocabulary size.");

            auto rostWriteGuard = rost->get_write_token();
            rost->set_topic_model(*rostWriteGuard, init_model);
        }

        stopWorkers();
        if (num_threads <= 0) { //refine when requested
            ROS_INFO("Topics will only be refined on service request.");
        } else { //refine automatically
            ROS_DEBUG("Topics will be refined online.");
            startWorkers();
        }
    }

    void stopWorkers() {
        if (!stopWork) {
            stopWork = true;                // signal workers to stop
            for (auto const &t : workers) { // wait for them to stop
                if (t) t->join();
            }
        }
    }

    void startWorkers() {
        if (stopWork) {
            stopWork = false;
            workers = parallel_refine_online_exp_beta(rost.get(), k_tau, p_refine_rate_local, p_refine_rate_global, num_threads, &stopWork);
        }
    }

    ~ROSTAdapter() override {
        stopWorkers();
        for (auto const &t : observationThreads) {
            if (t) t->join();
        }
    }

    std::future<std::unique_ptr<Segmentation<std::vector<int>, POSEDIM, CellDimType, WordDimType>>> operator()(WordObservation const *wordObs) {
        auto time_checkpoint = std::chrono::steady_clock::now();
        auto const time_start = time_checkpoint;

        if (wordObs->frame.empty()) ROS_WARN("Received WordObservation with empty frame!");

        if (world_frame.empty()) { world_frame = wordObs->frame; }
        else if (wordObs->frame != world_frame) {
            ROS_ERROR("Word observation in wrong frame! Skipping...\nFound: %s\nExpected: %s", wordObs->frame.c_str(), world_frame.c_str());
            throw std::invalid_argument("Word observation in invalid frame.");
        }

        ROS_ERROR_COND(wordObs->vocabulary_size > V,
                       "Word observation vocabulary size (%lu) is larger than ROST vocabulary (%d)! May cause crash...",
                       wordObs->vocabulary_size,
                       V);
        ROS_WARN_COND(wordObs->vocabulary_size < V,
                      "Word observation vocabulary size (%lu) is smaller than ROST vocabulary (%d).",
                      wordObs->vocabulary_size,
                      V);

        using namespace std;
        lock_guard<mutex> guard(wordsReceivedLock);
        auto duration_words_lock = record_lap(time_checkpoint);
        long duration_write_lock;

        // TODO Can observation transform ever be invalid?

        double observation_time = wordObs->timestamp;
        //update the  list of observed time step ids
        observation_times.push_back(observation_time);
        if (!observation_times.empty() && last_time > observation_time) {
            ROS_WARN_THROTTLE(30, "Observation received that is older than previous observation!");
        }

        //if we are receiving observations from the next time step, then spit out
        //topics for the current time step.
        static decltype(std::chrono::steady_clock::now()) timeFirstAdd;
        if (last_time >= 0) {
            ROS_DEBUG("Received more word observations - broadcasting observations for time %f", last_time);
            if (newObservationCallback) newObservationCallback(this);
            this->wait_for_processing(true);
            size_t const refine_count = rost->get_refine_count();
            ROS_DEBUG("time since last add: %ld ms", chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - lastWordsAdded));
            ROS_DEBUG("#words_refined since last add: %ld", refine_count - last_refine_count);
            auto timeSinceFirstAdd = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - timeFirstAdd).count();
//            ROS_INFO("Running Refine Rate: %f", static_cast<double>(refine_count - last_refine_count) / chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - lastWordsAdded).count());
            ROS_DEBUG("(%s) Avg. Refine Rate: %f", (rost->get_num_words() >= 15000) ? "ORB" : "NO ORB", static_cast<double>(refine_count) / timeSinceFirstAdd);
            current_cell_poses.clear();
        } else {
            timeFirstAdd = std::chrono::steady_clock::now();
        }
        last_time = std::max(last_time, observation_time);
        auto const duration_broadcast = record_lap(time_checkpoint);

        ROS_DEBUG("Adding %lu word observations from time %f", wordObs->observations.size(), observation_time);
        {
            auto rostWriteGuard = rost->get_write_token();
            duration_write_lock = record_lap(time_checkpoint);

            auto const &words_by_cell_pose = words_for_cell_poses(*wordObs, cell_size);
            auto const old_num_cells = rost->cells.size();
            current_cell_poses.reserve(current_cell_poses.size() + words_by_cell_pose.size());
            for (auto const &entry : words_by_cell_pose) {
                auto const &cell_pose = entry.first;
                auto const &cell_words = entry.second;
                rost->add_observation(cell_pose, cell_words.begin(), cell_words.end(), update_topic_model);
                current_cell_poses.push_back(cell_pose);
            }
//            ROS_INFO_STREAM("Thread " << std::this_thread::get_id() << " added " << (rost->cells.size() - old_num_cells) << " cells (" << wordObs->observations.size() << " words)");
        }
        auto const duration_add_observations = record_lap(time_checkpoint);
        last_refine_count = rost->get_refine_count();
        lastWordsAdded = chrono::steady_clock::now();
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
            ROS_DEBUG("Words observation overhead: %lu ms (%lu lock, %lu write lock, %lu broadcast, %lu updating cells)",
                      total_duration,
                      duration_words_lock,
                      duration_write_lock,
                      duration_broadcast,
                      duration_add_observations);
        }

        typedef Segmentation<std::vector<int>, POSEDIM, CellDimType, WordDimType> Segmentation;
        std::promise<std::unique_ptr<Segmentation>> promisedTopics;
        auto futureTopics = promisedTopics.get_future();
        if (broadcastMode) {
            observationThreads.push_back(std::make_unique<std::thread>([this, id = wordObs->id, cell_poses = current_cell_poses, promisedTopics{
                    std::move(promisedTopics)}]() mutable {
                using namespace std::chrono;
                wait_for_processing(false);
                auto topics = getTopicDistsForPoses(cell_poses);
                double const timestamp = duration<double>(steady_clock::now().time_since_epoch()).count();
                promisedTopics.set_value(std::make_unique<Segmentation>("map",
                                                                        timestamp,
                                                                        id,
                                                                        cell_size,
                                                                        std::move(topics),
                                                                        std::move(cell_poses)));
            }));
        } else {
            promisedTopics.set_value(nullptr);
        }
        return futureTopics;
    }

    using Adapter<ROSTAdapter<_POSEDIM>, CategoricalObservation<int,
            _POSEDIM - 1, WordInputDimType>, std::future<Segmentation<std::vector<int>, _POSEDIM, int32_t, WordOutputDimType>>>::operator();

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

    auto get_map(activity_manager::ReadToken const &read_token) const {
        using namespace std::chrono;
        double const timestamp = duration<double>(steady_clock::now().time_since_epoch()).count();
        auto map = std::make_unique<Segmentation<int, POSEDIM, CellDimType, WordDimType>>("map", timestamp, ros::Time(timestamp).sec, cell_size, std::vector<int>(), rost->cell_pose);
        map->observations = getMLTopicsForPoses(map->observation_poses);
        return map;
    }

    auto get_dist_map(activity_manager::ReadToken const &read_token) const {
        using namespace std::chrono;
        double const timestamp = duration<double>(steady_clock::now().time_since_epoch()).count();
        auto map = std::make_unique<Segmentation<std::vector<int>, POSEDIM, CellDimType, WordDimType>>("map", timestamp, ros::Time(timestamp).sec, cell_size, std::vector<std::vector<int>>(), rost->cell_pose);
        map->observations = getTopicDistsForPoses(map->observation_poses);
        return map;
    }

    Phi get_topic_model(activity_manager::ReadToken const &read_token) const {
        Phi phi("", rost->get_num_topics(), rost->get_num_words(), rost->get_topic_model(), rost->get_topic_weights(), rost->get_refine_count(), rost->get_word_refine_count());
        phi.validate(false);
        return phi;
    }

    void set_topic_model(activity_manager::WriteToken const &write_token, Phi const &phi) {
        rost->set_topic_model(write_token, (std::vector<std::vector<int>>) phi, phi.topic_weights);
    }

    void inline wait_for_processing(bool const new_data = false) const {
        using namespace std::chrono;
        int64_t elapsedSinceAdd = duration_cast<milliseconds>(steady_clock::now() - lastWordsAdded).count();
        uint64_t refinedSinceAdd = rost->get_refine_count() - last_refine_count;
        bool const delay = elapsedSinceAdd < min_obs_refine_time || refinedSinceAdd < min_refines_per_obs;
        if (new_data) {
            ROS_DEBUG("Time elapsed since last observation added (minimum set to %d ms): %l ms", min_obs_refine_time, elapsedSinceAdd);
            ROS_DEBUG("Refines since last observation added (minimum set to %d): %lu ms", min_refines_per_obs, refinedSinceAdd);
            if (delay) {
                consecutive_rate_violations++;
                ROS_WARN("New word observation received too soon! Delaying...");
                ROS_ERROR_COND(consecutive_rate_violations > obs_queue_size,
                               "A word observation will likely be dropped. Increase queue size, or reduce observation rate or processing time.");
            } else {
                consecutive_rate_violations = 0;
            }
        }
        if (delay) {
            if (elapsedSinceAdd < min_obs_refine_time) {
                std::this_thread::sleep_for(milliseconds(min_obs_refine_time - elapsedSinceAdd));
            }
            do {
                refinedSinceAdd = rost->get_refine_count() - last_refine_count;
                if (refinedSinceAdd >= min_refines_per_obs) break;
                elapsedSinceAdd = duration_cast<milliseconds>(steady_clock::now() - lastWordsAdded).count();
                auto const ms_per_refine = static_cast<double>(elapsedSinceAdd) / refinedSinceAdd;
                auto const required_refine_time = ms_per_refine * (min_refines_per_obs - refinedSinceAdd);
                std::this_thread::sleep_for(milliseconds(strictNumericCast<int64_t>(std::max(1.0, std::min(50.0, ceil(required_refine_time))))));
            } while (true);
        }
    }

    decltype(K) get_num_topics() const {
        return K;
    }

    decltype(K) get_num_active_topics() const {
        return rost->get_active_topics();
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

    decltype(world_frame) const &get_world_frame() const {
        return world_frame;
    }

    ROST_t const &get_rost() const {
        return *rost;
    }

    auto get_cell_topics_and_ppx(activity_manager::ReadToken const &read_token, cell_pose_t const &pose) {
        return rost->get_ml_topics_and_ppx_for_pose(pose);
    }

    inline cell_pose_t toCellId(word_pose_t const &word_pose) const {
        return sunshine::toCellId<POSEDIM, CellDimType, WordDimType>(word_pose, cell_size);
    }

    inline word_pose_t toWordPose(cell_pose_t const &cell_pose) const {
        return sunshine::toWordPose<POSEDIM, CellDimType, WordDimType>(cell_pose, cell_size);
    }
};
}

#endif // TOPIC_MODEL_HPP

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

#define POSEDIM 4

namespace sunshine {

typedef double_t WordDimType;
typedef int32_t CellDimType;
typedef std::array<CellDimType, POSEDIM> cell_pose_t;
typedef std::array<WordDimType, POSEDIM> word_pose_t;
typedef neighbors<cell_pose_t> neighbors_t;
typedef warp::SpatioTemporalTopicModel<cell_pose_t, neighbors_t, hash_container<cell_pose_t>> ROST_t;
using warp::ROST;
using warp::hROST;

class ROSTAdapter : public Adapter<ROSTAdapter, CategoricalObservation<int, 3, WordDimType>, std::future<Segmentation<std::vector<int>, POSEDIM, CellDimType, WordDimType>>> {
    mutable std::mutex wordsReceivedLock;
    std::chrono::steady_clock::time_point lastWordsAdded;
    mutable int consecutive_rate_violations = 0;

    int K, V;
    double last_time = -1; //number of topic types, number of word types
    std::array<double, POSEDIM> cell_size;
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

    std::vector<std::vector<int>> getTopicDistsForPoses(const std::vector<cell_pose_t>& cell_poses);

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
            cell_size = readNumbers<POSEDIM, 'x'>(cell_size_string);
        } else {
            cell_size = computeCellSize<POSEDIM>(cell_size_time, cell_size_space);
        }

//        ROS_INFO("Starting online topic modelling with parameters: K=%u, alpha=%f, beta=%f, tau=%f", K, k_alpha, k_beta, k_tau);

        cell_pose_t G{{G_time, G_space, G_space, G_space}};
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

    ~ROSTAdapter() override;

    std::future<Segmentation<std::vector<int>, POSEDIM, CellDimType, WordDimType>> operator()(CategoricalObservation<int, 3, WordDimType> const &words);

    std::map<CellDimType, std::vector<int>> get_topics_by_time() const;

    std::map<cell_pose_t, std::vector<int>> get_topics_by_cell() const;

    Phi get_topic_model(activity_manager::ReadToken const &read_token) const {
        Phi phi("", rost->get_num_topics(), rost->get_num_words(), rost->get_topic_model(), rost->get_topic_weights());
        phi.validate(false);
        return phi;
    }

    void set_topic_model(activity_manager::WriteToken const &write_token, Phi const &phi) {
        rost->set_topic_model(write_token, phi.counts, phi.topic_weights);
    }

    void wait_for_processing(bool new_data = true) const;

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
};

static inline cell_pose_t toCellId(word_pose_t const &word, std::array<double, POSEDIM> cell_size) {
    static_assert(std::numeric_limits<CellDimType>::max() <= std::numeric_limits<WordDimType>::max(),
                  "Word dim type must be larger than cell dim type!");
    return {static_cast<CellDimType>(std::fmod(word[0] / cell_size[0], std::numeric_limits<CellDimType>::max())),
          static_cast<CellDimType>(std::fmod(word[1] / cell_size[1], std::numeric_limits<CellDimType>::max())),
          static_cast<CellDimType>(std::fmod(word[2] / cell_size[2], std::numeric_limits<CellDimType>::max())),
          static_cast<CellDimType>(std::fmod(word[3] / cell_size[3], std::numeric_limits<CellDimType>::max()))};
}

static inline word_pose_t toWordPose(cell_pose_t const &cell, std::array<double, POSEDIM> cell_size) {
    return {static_cast<WordDimType>(cell[0] * cell_size[0]), static_cast<WordDimType>(cell[1] * cell_size[1]),
          static_cast<WordDimType>(cell[2] * cell_size[2]), static_cast<WordDimType>(cell[3] * cell_size[3])};
}
}

#endif // TOPIC_MODEL_HPP

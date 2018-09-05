#ifndef TOPIC_MODEL_HPP
#define TOPIC_MODEL_HPP

#include <array>
#include <memory>
#include <ros/ros.h>
#include <rost/rost.hpp>
#include <sunshine_msgs/LocalSurprise.h>
#include <sunshine_msgs/Perplexity.h>
#include <sunshine_msgs/TopicWeights.h>
#include <sunshine_msgs/WordObservation.h>

#define POSEDIM 4

namespace sunshine {

typedef double_t WordDimType;
typedef int32_t CellDimType;
typedef std::array<CellDimType, POSEDIM> cell_pose_t;
typedef std::array<WordDimType, POSEDIM> word_pose_t;
typedef neighbors<cell_pose_t> neighbors_t;
typedef ROST<cell_pose_t, neighbors_t, hash_container<cell_pose_t>> ROST_t;

class topic_model {
    ros::NodeHandle* nh;
    ros::Publisher scene_pub, global_perplexity_pub, global_surprise_pub, local_surprise_pub, topic_weights_pub;
    ros::Subscriber word_sub;

    std::mutex wordsReceivedLock;
    std::chrono::steady_clock::time_point lastWordsAdded;
    int consecutive_rate_violations = 0;

    std::string words_topic_name;
    int K, V, last_time; //number of topic types, number of word types
    double cell_size_time, cell_size_space, k_alpha, k_beta, k_gamma, k_tau, p_refine_rate_local, p_refine_rate_global;
    CellDimType G_time, G_space;
    int num_threads, min_obs_refine_time, obs_queue_size;
    bool polled_refine, update_topic_model, publish_topics, publish_local_surprise, publish_global_surprise, publish_ppx;
    size_t last_refine_count;
    std::unique_ptr<ROST_t> rost;

    std::vector<CellDimType> observation_times; //list of all time seq ids observed thus far.
    std::vector<cell_pose_t> current_cell_poses, last_poses;
    sunshine_msgs::WordObservation topicObs;

    std::atomic<bool> stopWork;
    std::vector<std::shared_ptr<std::thread>> workers;

    void wait_for_processing();
    void words_callback(const sunshine_msgs::WordObservation::ConstPtr& words);
    void broadcast_topics();

public:
    topic_model(ros::NodeHandle* nh);
    ~topic_model();
};
}

#endif // TOPIC_MODEL_HPP

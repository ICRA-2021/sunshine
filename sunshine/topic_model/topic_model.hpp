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

typedef float_t DimType;
typedef std::array<DimType, POSEDIM> pose_t;
typedef neighbors<pose_t> neighbors_t;
typedef ROST<pose_t, neighbors_t, hash_container<pose_t>> ROST_t;

class topic_model {
    ros::NodeHandle* nh;
    ros::Publisher scene_pub, global_perplexity_pub, global_surprise_pub, local_surprise_pub, topic_weights_pub;
    ros::Subscriber word_sub;

    std::mutex wordsReceivedLock;
    std::chrono::steady_clock::time_point lastWordsAdded;
    int consecutive_rate_violations = 0;

    std::string words_topic_name;
    int K, V, cell_space; //number of topic types, number of word types
    double k_alpha, k_beta, k_gamma, k_tau, p_refine_rate_local, p_refine_rate_global;
    DimType G_time, G_space, last_time;
    int num_threads, min_obs_refine_time, obs_queue_size;
    bool polled_refine, update_topic_model;
    size_t last_refine_count;
    std::unique_ptr<ROST_t> rost;

    std::vector<DimType> observation_times; //list of all time seq ids observed thus far.
    std::map<pose_t, std::vector<pose_t>> word_poses_by_cell; //stores [pose]-> {x_i,y_i,scale_i,.....} for the current time
    std::vector<pose_t> current_cell_poses, last_poses;
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

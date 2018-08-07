#include "topic_model.hpp"

#include <opencv2/core.hpp>
#include <rost/refinery.hpp>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/transform_storage.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using namespace sunshine;
using namespace sunshine_msgs;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "topic_model");
    ros::NodeHandle nh("~");

    topic_model model(&nh);

    ros::MultiThreadedSpinner spinner;
    ROS_INFO("Spinning...");
    spinner.spin();
}

topic_model::topic_model(ros::NodeHandle* nh)
    : nh(nh)
{
    nh->param<int>("K", K, 100); // number of topics
    nh->param<int>("V", V, 1500); // vocabulary size
    nh->param<double>("alpha", k_alpha, 0.1);
    nh->param<double>("beta", k_beta, 1.0);
    nh->param<double>("gamma", k_gamma, 0.001);
    nh->param<double>("tau", k_tau, 0.5); // beta(1,tau) is used to pick cells for global refinement
    nh->param<double>("p_refine_rate_local", p_refine_rate_local, 0.5); // probability of refining last observation
    nh->param<double>("p_refine_rate_global", p_refine_rate_global, 0.5);
    nh->param<int>("num_threads", num_threads, 4); // beta(1,tau) is used to pick cells for refinement
    nh->param<int>("cell_space", cell_space, 32);
    nh->param<float>("G_time", G_time, 1);
    nh->param<float>("G_space", G_space, 1);
    nh->param<bool>("polled_refine", polled_refine, false);
    nh->param<bool>("update_topic_model", update_topic_model, true);
    nh->param<int>("min_obs_refine_time", min_obs_refine_time, 200);
    nh->param<int>("word_obs_queue_size", obs_queue_size, 1);

    nh->param<std::string>("words_topic",words_topic_name,"/word_extractor/words");
    
    ROS_INFO("Starting online topic modelling with parameters: K=%u, alpha=%f, beta=%f, gamma=%f tau=%f", K, k_alpha, k_beta, k_gamma, k_tau);

    scene_pub = nh->advertise<WordObservation>("topics", 10);
    global_perplexity_pub = nh->advertise<Perplexity>("perplexity_score", 10);
    global_surprise_pub = nh->advertise<LocalSurprise>("scene_perplexity", 10);
    local_surprise_pub = nh->advertise<LocalSurprise>("cell_perplexity", 10);
    topic_weights_pub = nh->advertise<TopicWeights>("topic_weight", 10);

    word_sub = nh->subscribe(words_topic_name, static_cast<uint32_t>(obs_queue_size), &topic_model::words_callback, this);

    pose_t G{ { G_time, G_space, G_space } };
    rost = std::unique_ptr<ROST_t>(new ROST_t(static_cast<size_t>(V), static_cast<size_t>(K), k_alpha, k_beta, neighbors_t(G)));
    last_time = -1;

    if (polled_refine) { //refine when requested
        throw std::runtime_error("Not implemented. Requires services.");
        ROS_INFO("Topics will be refined on request.");
    } else { //refine automatically
        ROS_INFO("Topics will be refined online.");
        stopWork = false;
        workers = parallel_refine_online_exp_beta(rost.get(), k_tau, p_refine_rate_local, p_refine_rate_global, num_threads, &stopWork);
    }
}

topic_model::~topic_model()
{
    stopWork = true; //signal workers to stop
    for (auto t : workers) { //wait for them to stop
        t->join();
    }
}

static std::pair<std::map<pose_t, std::vector<int>>,
    std::map<pose_t, std::vector<pose_t>>>
words_for_cell_poses(WordObservation const& wordObs, int cell_size)
{
    using namespace std;
    map<pose_t, vector<int>> words_by_cell_pose;
    map<pose_t, vector<pose_t>> word_poses_by_cell_pose;

    for (size_t i = 0; i < wordObs.words.size(); ++i) {
        geometry_msgs::Point cell_point, word_point;
        cell_point.x = wordObs.word_pose[i * 3 + 0] / cell_size;
        word_point.x = wordObs.word_pose[i * 3 + 0];
        cell_point.y = wordObs.word_pose[i * 3 + 1] / cell_size;
        word_point.y = wordObs.word_pose[i * 3 + 1];
        cell_point.z = wordObs.word_pose[i * 3 + 2] / cell_size;
        word_point.z = wordObs.word_pose[i * 3 + 2];
        tf2::doTransform(cell_point, cell_point, wordObs.observation_transform);
        tf2::doTransform(word_point, word_point, wordObs.observation_transform);

        pose_t cell_stamped_point, word_stamped_point;
        cell_stamped_point[0] = static_cast<DimType>(wordObs.observation_transform.header.stamp.toSec()); // TODO (Stewart Jamieson): Use the time
        word_stamped_point[0] = static_cast<DimType>(wordObs.observation_transform.header.stamp.toSec()); // TODO (Stewart Jamieson): Use the time
        cell_stamped_point[1] = static_cast<DimType>(cell_point.x);
        word_stamped_point[1] = static_cast<DimType>(word_point.x);
        cell_stamped_point[2] = static_cast<DimType>(cell_point.y);
        word_stamped_point[2] = static_cast<DimType>(word_point.y);
        cell_stamped_point[3] = static_cast<DimType>(cell_point.z);
        word_stamped_point[3] = static_cast<DimType>(word_point.z);

        words_by_cell_pose[cell_stamped_point].push_back(wordObs.words[i]);
        word_poses_by_cell_pose[cell_stamped_point].push_back(word_stamped_point);
    }
    return make_pair(words_by_cell_pose, word_poses_by_cell_pose);
}

void topic_model::wait_for_processing()
{
    using namespace std::chrono;
    auto const elapsedSinceAdd = steady_clock::now() - lastWordsAdded;
    ROS_DEBUG("Time elapsed since last observation added (minimum set to %d ms): %lu ms",
              min_obs_refine_time, duration_cast<milliseconds>(elapsedSinceAdd).count());
    if (duration_cast<milliseconds>(elapsedSinceAdd).count() < min_obs_refine_time) {
        ROS_WARN("New word observation received too soon! Delaying...");
        if (++consecutive_rate_violations > obs_queue_size) {
            ROS_ERROR("Next word observation will likely be dropped. Increase queue size, or reduce observation rate or processing time.");
        }
        std::this_thread::sleep_for(milliseconds(min_obs_refine_time) - elapsedSinceAdd);
    } else {
        consecutive_rate_violations = 0;
    }
}

void topic_model::words_callback(const WordObservation::ConstPtr& wordObs)
{
    using namespace std;
    lock_guard<mutex> guard(wordsReceivedLock);

    if (false) { // TODO Can observation transform ever be invalid?
        ROS_ERROR("Word observations are missing observation poses! Skipping...");
        return;
    }

    DimType observation_time = static_cast<DimType>(wordObs->observation_transform.header.stamp.toSec());
    //update the  list of observed time step ids
    if (observation_times.empty() || observation_times.back() < observation_time) {
        observation_times.push_back(observation_time);
    } else if (observation_times.back() > observation_time) {
        ROS_WARN("Observation received that is older than previous observation! Skipping...");
        return;
    }

    //if we are receiving observations from the next time step, then spit out
    //topics for the current time step.
    if (last_time >= 0 && (last_time != observation_time)) {
        broadcast_topics();
        size_t refine_count = rost->get_refine_count();
        ROS_DEBUG("#cells_refined: %u", static_cast<unsigned>(refine_count - last_refine_count));
        last_refine_count = refine_count;
        word_poses_by_cell.clear();
        current_cell_poses.clear();
    }
    last_time = observation_time;

    auto celldata = words_for_cell_poses(*wordObs, cell_space);
    auto const& words_by_cell_pose = celldata.first;
    auto const& word_poses_by_cell_pose = celldata.second;
    for (auto const& entry : words_by_cell_pose) {
        auto const& cell_pose = entry.first;
        auto const& cell_words = entry.second;
        rost->add_observation(cell_pose, cell_words.begin(), cell_words.end(), update_topic_model);
        current_cell_poses.push_back(cell_pose);
    }

    lastWordsAdded = chrono::steady_clock::now();

    word_poses_by_cell.insert(word_poses_by_cell_pose.begin(), word_poses_by_cell_pose.end());
}

void topic_model::broadcast_topics()
{
    using namespace std;
    wait_for_processing();

    auto const time = last_time;

    WordObservation::Ptr topicObs(new WordObservation);
    topicObs->source = "topics";
    topicObs->vocabulary_begin = 0;
    topicObs->vocabulary_size = static_cast<int32_t>(K);
    topicObs->seq = static_cast<uint32_t>(time);
    topicObs->observation_transform.transform.rotation.w = 1; // Identity rotation (global frame)
    topicObs->observation_transform.header.stamp = ros::Time::now();

    Perplexity::Ptr global_perplexity(new Perplexity);
    global_perplexity->seq = static_cast<uint32_t>(time);
    global_perplexity->perplexity = -1;

    LocalSurprise::Ptr global_surprise(new LocalSurprise);
    global_surprise->seq = static_cast<uint32_t>(time);
    global_surprise->cell_width = cell_space;
    global_surprise->surprise.resize(word_poses_by_cell.size(), 0);
    global_surprise->surprise_poses.resize(word_poses_by_cell.size() * (POSEDIM - 1), 0);

    LocalSurprise::Ptr local_surprise(new LocalSurprise);
    local_surprise->seq = static_cast<uint32_t>(time);
    local_surprise->cell_width = cell_space;
    local_surprise->surprise.resize(word_poses_by_cell.size(), 0);
    local_surprise->surprise_poses.resize(word_poses_by_cell.size() * (POSEDIM - 1), 0);

    TopicWeights::Ptr msg_topic_weights(new TopicWeights);
    msg_topic_weights->seq = static_cast<uint32_t>(time);
    msg_topic_weights->weight = rost->get_topic_weights();

    auto n_words = 0ul;
    double sum_log_p_word = 0;
    auto entryIdx = 0u;

    for (auto const& entry : word_poses_by_cell) {
        const pose_t& cell_pose = entry.first;
        vector<int> topics; //topic labels for each word in the cell
        double log_likelihood; //cell's sum_w log(p(w | model) = log p(cell | model)
        tie(topics, log_likelihood) = rost->get_ml_topics_and_ppx_for_pose(cell_pose);

        auto const& cell = rost->get_cell(cell_pose);
        for (auto i = 0u; i < POSEDIM - 1; i++) {
            auto const poseIdx = entryIdx * (POSEDIM - 1) + i;
            local_surprise->surprise_poses[poseIdx] = cell_pose[i + 1];
            global_surprise->surprise_poses[poseIdx] = cell_pose[i + 1];
        }

        auto const cell_perplexity = rost->cell_perplexity_word(cell->W, rost->neighborhood(*cell));
        auto const scene_perplexity = rost->cell_perplexity_word(cell->W, rost->get_topic_weights());
        local_surprise->surprise[entryIdx] = cell_perplexity;
        global_surprise->surprise[entryIdx] = scene_perplexity;

        //populate the topic label message
        topicObs->words.insert(topicObs->words.end(), topics.begin(), topics.end());
        vector<pose_t> const& word_poses = entry.second;
        for (auto const& word_pose : word_poses) {
            for (auto i = 1u; i < POSEDIM; i++) {
                topicObs->word_pose.push_back(word_pose[i]);
            }
        }

        n_words += topics.size();
        sum_log_p_word += log_likelihood;
        assert(n_words * (POSEDIM - 1) == topicObs->word_pose.size());
        entryIdx++;
    }

    global_perplexity->perplexity = exp(-sum_log_p_word / n_words);
    ROS_INFO("Perplexity: %f", global_perplexity->perplexity);

    scene_pub.publish(topicObs);
    topic_weights_pub.publish(msg_topic_weights);
    global_perplexity_pub.publish(global_perplexity);
    global_surprise_pub.publish(global_surprise);
    local_surprise_pub.publish(local_surprise);
}

#include "sunshine/rost_adapter.hpp"

#include <fstream>
#include <exception>
#include <functional>
#include <rost/refinery.hpp>
#include <ros/console.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"

using namespace sunshine;
using WordObservation = CategoricalObservation<int, 3, WordDimType>;

template<typename T>
long record_lap(T &time_checkpoint) {
    auto const duration = std::chrono::steady_clock::now() - time_checkpoint;
    time_checkpoint = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

ROSTAdapter::~ROSTAdapter() {
    stopWork = true; //signal workers to stop
    for (auto const &t : workers) { //wait for them to stop
        if (t) t->join();
    }
}

std::map<ROST_t::pose_dim_t, std::vector<int>> ROSTAdapter::get_topics_by_time() const {
    auto rostReadToken = rost->get_read_token();
    auto const poses_by_time = rost->get_poses_by_time();
    std::map<ROST_t::pose_dim_t, std::vector<int>> topics_by_time;
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

std::map<cell_pose_t, std::vector<int>> ROSTAdapter::get_topics_by_cell() const {
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

static std::map<cell_pose_t, std::vector<int>> words_for_cell_poses(WordObservation const &wordObs, std::array<double, POSEDIM> cell_size) {
    using namespace std;
    map<cell_pose_t, vector<int>> words_by_cell_pose;

    for (size_t i = 0; i < wordObs.observations.size(); ++i) {
        // Don't need this since we check that the word obs has the right frame
//        geometry_msgs::Point word_point;
//        transformPose(word_point, wordObs.observation_poses, i * 3, wordObs.observation_transform);
//        word_pose_t word_stamped_point;
//        word_stamped_point[0] = static_cast<WordDimType>(wordObs.timestamp);
//        word_stamped_point[1] = static_cast<WordDimType>(word_point.x);
//        word_stamped_point[2] = static_cast<WordDimType>(word_point.y);
//        word_stamped_point[3] = static_cast<WordDimType>(word_point.z);

        cell_pose_t const cell_stamped_point = toCellId({wordObs.timestamp, wordObs.observation_poses[i][0],
                                                              wordObs.observation_poses[i][1], wordObs.observation_poses[i][2]}, cell_size);
        words_by_cell_pose[cell_stamped_point].emplace_back(wordObs.observations[i]);
    }
    return words_by_cell_pose;
}

void ROSTAdapter::wait_for_processing(bool new_data) const {
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

std::future<Segmentation<std::vector<int>, POSEDIM, CellDimType, WordDimType>> ROSTAdapter::operator()(WordObservation const &wordObs) {
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
        promisedTopics.set_value(Segmentation("map", timestamp, id, cell_size, topics, poses));
    }));
    return futureTopics;
}

std::vector<std::vector<int>> ROSTAdapter::getTopicDistsForPoses(const std::vector<cell_pose_t>& cell_poses) {
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

#include "topic_model_node.hpp"

#include "sunshine/common/ros_conversions.hpp"
#include "sunshine/common/ros_utils.hpp"
#include <exception>
#include <fstream>
#include <functional>
#include <rost/refinery.hpp>
#include <sunshine_msgs/LocalSurprise.h>
#include <sunshine_msgs/Perplexity.h>
#include <sunshine_msgs/TopicMap.h>
#include <sunshine_msgs/TopicWeights.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/transform_storage.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"

using namespace sunshine;
using namespace sunshine_msgs;

using warp::hROST;
using warp::ROST;

static std::string const NO_PPX                                      = "none";
static std::string const CELL_PPX                                    = "cell";
static std::string const NEIGHBORHOOD_PPX                            = "neighborhood";
static std::string const GLOBAL_PPX                                  = "global";
std::vector<std::string> const topic_model_node::VALID_MAP_PPX_TYPES = {NO_PPX, CELL_PPX, NEIGHBORHOOD_PPX, /*"scene" ,*/ GLOBAL_PPX};

template<typename T>
long record_lap(T &time_checkpoint) {
    auto const duration = std::chrono::steady_clock::now() - time_checkpoint;
    time_checkpoint     = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

bool _save_topics_by_time_csv(topic_model_node const *topic_model, SaveObservationModelRequest &req, SaveObservationModelResponse &) {
    auto const &rost = topic_model->get_adapter();
    std::ofstream writer(req.filename);
    writer << "time";
    for (auto k = 0; k < rost.get_num_topics(); k++) { writer << ",topic_" << std::to_string(k) << "_count"; }
    auto const topics = rost.get_topics_by_time();
    for (auto const &entry : topics) {
        writer << "\n";
        writer << std::to_string(entry.first);
        for (auto const &count : entry.second) { writer << "," + std::to_string(count); }
    }
    writer.close();
    return true;
}

bool _save_topics_by_cell_csv(topic_model_node const *topic_model, SaveObservationModelRequest &req, SaveObservationModelResponse &) {
    auto const &rost = topic_model->get_adapter();
    std::ofstream writer(req.filename);
    writer << "pose_dim_0";
    for (auto i = 1; i < POSEDIM; i++) { writer << ",pose_dim_" + std::to_string(i); }
    for (auto k = 0; k < rost.get_num_topics(); k++) { writer << ",k_" << std::to_string(k); }
    auto const topics = rost.get_topics_by_cell();
    for (auto const &entry : topics) {
        writer << "\n";
        writer << std::to_string(entry.first[0]);
        for (auto dim = 1u; dim < POSEDIM; dim++) { writer << "," + std::to_string(entry.first[dim]); }
        for (auto const &count : entry.second) { writer << "," + std::to_string(count); }
    }
    writer.close();
    return true;
}

bool _generate_topic_summary(topic_model_node const *topic_model, GetTopicSummaryRequest &request, GetTopicSummaryResponse &response) {
    auto const &rost      = topic_model->get_adapter();
    response.num_topics   = rost.get_num_topics();
    response.last_seq     = rost.get_last_observation_time();
    response.header.stamp = ros::Time::now();
    if (request.grouping == "cell") {
        auto const topics         = rost.get_topics_by_cell();
        response.num_observations = topics.size();
        response.pose_fields      = "t,x,y,z";
        response.topic_counts.reserve(rost.get_num_topics() * topics.size());
        response.topic_pose.reserve(POSEDIM * topics.size());
        for (auto const &entry : topics) {
            response.topic_pose.insert(response.topic_pose.end(), entry.first.begin(), entry.first.end());
            response.topic_counts.insert(response.topic_counts.end(), entry.second.begin(), entry.second.end());
        }
    } else if (request.grouping == "time") {
        auto const topics         = rost.get_topics_by_time();
        response.num_observations = topics.size();
        response.pose_fields      = "t";
        response.topic_counts.reserve(rost.get_num_topics() * topics.size());
        response.topic_pose.reserve(topics.size());
        for (auto const &entry : topics) {
            response.topic_pose.push_back(entry.first);
            response.topic_counts.insert(response.topic_counts.end(), entry.second.begin(), entry.second.end());
        }
    } else if (request.grouping == "global") {
        auto const topics         = rost.get_rost().get_topic_weights();
        response.num_observations = 1;
        response.pose_fields      = "";
        assert(topics.size() == rost.get_num_topics());
        response.topic_counts.reserve(topics.size());
        for (auto const &entry : topics) { response.topic_counts = topics; }
    } else if (request.grouping == "observation") {
        ROS_ERROR("'observation' grouping is not yet implemented.");
        return false;
    } else {
        ROS_ERROR("Unrecognized topic grouping: %s", request.grouping.c_str());
        return false;
    }
    return true;
}

bool _get_topic_map(topic_model_node const *topic_model, GetTopicMapRequest &, GetTopicMapResponse &response) {
    response.topic_map = *topic_model->generate_topic_map(topic_model->get_adapter().get_last_observation_time());
    return true;
}

bool _get_topic_model(topic_model_node *topic_model,
                      std::unique_ptr<activity_manager::WriteToken> const &rostLock,
                      std::string name,
                      GetTopicModelRequest &,
                      GetTopicModelResponse &response) {
    ROS_DEBUG("Sending topic model! Global lock held: %s", (rostLock) ? "true" : "false");
    response.topic_model            = sunshine_msgs::TopicModel();
    response.topic_model.identifier = name;

    auto &rost = topic_model->get_adapter().get_rost();
    std::unique_ptr<activity_manager::ReadToken> readToken;
    if (!rostLock) readToken = rost.get_read_token();
    assert(readToken || rostLock);
    auto const weights     = rost.get_topic_weights();
    response.topic_model.V = rost.get_num_words();
    auto const phi         = rost.get_topic_model();

    int const K = rost.get_num_topics();
    //    for (K = rost.get_num_topics(); K > 0 && weights[K - 1] == 0; --K); // WARNING: it is not so simple to skip empty topics!! the
    //    model translator has issues merging if topics that it added disappear
    response.topic_model.K = K;
    response.topic_model.topic_weights.reserve(K);
    response.topic_model.phi.reserve(K * response.topic_model.V);
    int empty = 0;
    for (auto k = 0ul; k < K; ++k) {
        if (weights[k] < 0) {
            ROS_ERROR("How is this weight negative? %d", weights[k]);
        } else if (weights[k] == 0)
            empty++;
        response.topic_model.topic_weights.push_back(weights[k]);
        response.topic_model.phi.insert(response.topic_model.phi.end(), phi[k].begin(), phi[k].end());
    }
    if (empty > 0) ROS_INFO("Sent %d empty topics. Try using HDP to minimize this expense.", empty); // TODO optimize?
    return response.topic_model.K >= 1;
}

bool _set_topic_model(topic_model_node *topic_model,
                      std::unique_ptr<activity_manager::WriteToken> const &rostLock,
                      SetTopicModelRequest &request,
                      SetTopicModelResponse &) {
    ROS_INFO("New topic model received. Global lock held: %s", (rostLock) ? "true" : "false");
    std::stringstream weights;
    for (auto const &weight : request.topic_model.topic_weights) weights << weight << " ";
    ROS_INFO("New topic weights: %s", weights.str().c_str());
    std::vector<std::vector<int>> nZW;
    assert(request.topic_model.K > 0);
    nZW.reserve(request.topic_model.K);
    auto const V = request.topic_model.V;
    for (auto k = 0ul; k < request.topic_model.K; ++k) {
        nZW.emplace_back(request.topic_model.phi.cbegin() + k * V, request.topic_model.phi.cbegin() + (k + 1) * V);
    }
    auto const &rost = topic_model->get_adapter().get_rost();
    auto writeLock   = (rostLock) ? std::unique_ptr<activity_manager::WriteToken>() : rost.get_write_token();
    assert(V == rost.get_num_words());
    if (request.topic_model.K < rost.get_num_topics()) {
        nZW.resize(rost.get_num_topics(), std::vector<int>(V, 0));
        request.topic_model.topic_weights.resize(rost.get_num_topics(), 0);
    }

#ifndef NDEBUG
    ROS_WARN("Running exhaustive topic_model set verifications -- use release build to skip");
    auto const ref = rost.get_topic_weights();
    ROS_INFO("Topic counts old: %d. Topic counts new: %d.", std::accumulate(ref.begin(), ref.end(), 0),
             std::accumulate(request.topic_model.topic_weights.begin(), request.topic_model.topic_weights.end(), 0));
    topic_model->externalTopicCounts.resize(rost.get_num_topics(), std::vector<int>(V, 0));
    auto const &ref_model = rost.get_topic_model();
    for (auto i = 0ul; i < ref_model.size(); ++i) {
        for (auto j = 0ul; j < ref_model[i].size(); ++j) {
            topic_model->externalTopicCounts[i][j] += nZW[i][j] - ref_model[i][j];
            if (topic_model->externalTopicCounts[i][j] < 0) ROS_WARN_THROTTLE(1, "New topic model invalidates previous topic labels!");
        }
    }
#endif

    ROS_DEBUG("Setting topic model with dimen: %lu,%lu vs %u,%u", nZW.size(), nZW[0].size(), rost.get_num_topics(), rost.get_num_words());
    topic_model->get_adapter().set_topic_model((rostLock) ? *rostLock : *writeLock, Phi("", rost.get_num_topics(), rost.get_num_words(),
                                                                                        std::move(nZW), request.topic_model.topic_weights));
    return true;
}

bool _pause_topic_model(topic_model_node *topic_model,
                        std::unique_ptr<activity_manager::WriteToken> &rostLock,
                        PauseRequest &request,
                        PauseResponse &) {
    ROS_DEBUG("Changing topic model global pause state");
    auto const &rost = topic_model->get_adapter().get_rost();
    if (request.pause == (bool) rostLock) return false;
    if (request.pause) rostLock = rost.get_write_token();
    if (!request.pause) rostLock.reset();
    ROS_INFO("Global topic model pause lock %s", (rostLock) ? "locked" : "released");
    return true;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "topic_model");
    ros::NodeHandle nh("~");

    topic_model_node model(&nh);

    ros::MultiThreadedSpinner spinner;
    ROS_INFO("Spinning...");
    spinner.spin();
}

topic_model_node::topic_model_node(ros::NodeHandle *nh)
    : nh(nh)
    , rostAdapter(
          nh,
          [this](ROSTAdapter<POSEDIM> *adapter) {
              ROS_DEBUG("Received newer word observations - broadcasting observations for time %f", adapter->get_last_observation_time());
              if (broadcast_thread && broadcast_thread->joinable()) { broadcast_thread->join(); }
              broadcast_thread = std::make_shared<std::thread>(&topic_model_node::broadcast_topics, this,
                                                               adapter->get_last_observation_time(), adapter->get_current_cell_poses());
          },
          nh->param<bool>("init_identity", false) ? identity_mat<int>(nh->param<int>("K", 0)) : identity_mat<int>(0))
    , save_topics_by_time_csv([this](auto &&PH1, auto &&PH2) { return _save_topics_by_time_csv(this, PH1, PH2); })
    , save_topics_by_cell_csv([this](auto &&PH1, auto &&PH2) { return _save_topics_by_cell_csv(this, PH1, PH2); })
    , generate_topic_summary([this](auto &&PH1, auto &&PH2) { return _generate_topic_summary(this, PH1, PH2); })
    , get_topic_map([this](auto &&PH1, auto &&PH2) { return _get_topic_map(this, PH1, PH2); })
    , get_topic_model([this, capture0 = boost::cref(externalRostLock)](auto &&PH1, auto &&PH2) {
        return _get_topic_model(this, capture0, ros::this_node::getName(), PH1, PH2);
    })
    , set_topic_model(
          [this, capture0 = boost::cref(externalRostLock)](auto &&PH1, auto &&PH2) { return _set_topic_model(this, capture0, PH1, PH2); })
    , pause_topic_model([this, capture0 = boost::ref(externalRostLock)](auto &&PH1, auto &&PH2) {
        return _pause_topic_model(this, capture0, PH1, PH2);
    }) {
    nh->param<int>("word_obs_queue_size", obs_queue_size, 1);
    nh->param<bool>("publish_topics", publish_topics, true);
    nh->param<bool>("publish_local_surprise", publish_local_surprise, false);
    nh->param<bool>("publish_global_surprise", publish_global_surprise, false);
    nh->param<bool>("publish_ppx", publish_ppx, false);
    nh->param<std::string>("words_topic", words_topic_name, "/word_extractor/words");
    nh->param<std::string>("map_ppx", map_ppx_type, NO_PPX);
    nh->param<float>("map_publish_period", map_publish_period, -1);
    nh->param<float>("save_topics_period", save_topics_period, -1);
    nh->param<std::string>("save_topics_path", save_topics_path, "");
    if (std::find(VALID_MAP_PPX_TYPES.cbegin(), VALID_MAP_PPX_TYPES.cend(), map_ppx_type) == VALID_MAP_PPX_TYPES.cend()) {
        throw std::invalid_argument("Invalid map perplexity type: " + map_ppx_type);
    }

    topic_weights_pub = nh->advertise<TopicWeights>("topic_weight", 10);
    if (publish_topics) scene_pub = nh->advertise<sunshine_msgs::WordObservation>("topics", 10);
    if (publish_ppx) global_perplexity_pub = nh->advertise<Perplexity>("perplexity_score", 10);
    if (publish_global_surprise) global_surprise_pub = nh->advertise<LocalSurprise>("scene_perplexity", 10);
    if (publish_local_surprise) local_surprise_pub = nh->advertise<LocalSurprise>("cell_perplexity", 10);
    if (map_publish_period >= 0) map_pub = nh->advertise<TopicMap>("topic_map", 1);

    this->word_sub = nh->subscribe(words_topic_name, static_cast<uint32_t>(obs_queue_size), &topic_model_node::words_callback, this);
    this->time_topic_server      = nh->advertiseService<>("save_topics_by_time_csv", this->save_topics_by_time_csv);
    this->cell_topic_server      = nh->advertiseService<>("save_topics_by_cell_csv", this->save_topics_by_cell_csv);
    this->topic_summary_server   = nh->advertiseService<>("get_topic_summary", this->generate_topic_summary);
    this->topic_map_server       = nh->advertiseService<>("get_topic_map", this->get_topic_map);
    this->get_topic_model_server = nh->advertiseService<>("get_topic_model", this->get_topic_model);
    this->set_topic_model_server = nh->advertiseService<>("set_topic_model", this->set_topic_model);
    this->pause_server           = nh->advertiseService<>("pause_topic_model", this->pause_topic_model);

    if (map_publish_period > 0) {
        map_publish_timer = nh->createTimer(ros::Duration(map_publish_period), [this](ros::TimerEvent const &) {
            map_pub.publish(*generate_topic_map(rostAdapter.get_last_observation_time()));
        });
    }

    if (save_topics_period > 0) {
        auto const last_slash = ros::this_node::getName().find_last_of('/');
        auto const nodeName   = ros::this_node::getName().substr((last_slash == std::string::npos) ? 0 : last_slash + 1);
        save_topics_timer     = nh->createTimer(ros::Duration(save_topics_period), [this, nodeName](ros::TimerEvent const &) {
            GetTopicModel serviceObj{};
            std::unique_ptr<activity_manager::WriteToken> nptr{};
            auto const success = _get_topic_model(this, nptr, ros::this_node::getName(), serviceObj.request, serviceObj.response);
            if (success) {
                auto const millis = std::to_string(static_cast<int>(ros::Time::now().nsec / 1E6));
                assert(millis.size() <= 3);
                std::string const filename = save_topics_path + "/" + std::to_string(ros::Time::now().sec) + "_"
                                             + std::string(3 - millis.size(), '0') + millis + "_" + nodeName + ".bin";
                std::fstream writer(filename, std::ios::out | std::ios::binary);
                if (writer.good()) {
                    writer.write(reinterpret_cast<char *>(serviceObj.response.topic_model.phi.data()),
                                 sizeof(decltype(serviceObj.response.topic_model.phi)::value_type) / sizeof(char)
                                     * serviceObj.response.topic_model.phi.size());
                    writer.close();
                } else {
                    ROS_WARN("Failed to save topic model to file %s", filename.c_str());
                }
            }
        });
    }
}

topic_model_node::~topic_model_node() {
    map_publish_timer.stop();
    if (broadcast_thread && broadcast_thread->joinable()) broadcast_thread->join();
}

void topic_model_node::words_callback(const sunshine_msgs::WordObservation::ConstPtr &wordMsg) {
    if (current_source.empty()) {
        current_source = wordMsg->source;
    } else if (current_source != wordMsg->source)
        ROS_WARN("Received words from new source! Expected \"%s\", received \"%s\"", current_source.c_str(), wordMsg->source.c_str());
    auto const wordObs = fromRosMsg<int, POSEDIM - 1, ROSTAdapter<POSEDIM>::WordDimType>(*wordMsg);
    last_obs_time = std::chrono::steady_clock::now();
    start_refine_time = (start_refine_time.time_since_epoch().count() == 0) ? last_obs_time : start_refine_time;
    rostAdapter(&wordObs);
}

TopicMapPtr topic_model_node::generate_topic_map(int const obs_time) const {
    auto const &rost            = rostAdapter.get_rost();
    auto rostReadToken          = rost.get_read_token();
    TopicMap::Ptr topic_map     = boost::make_shared<TopicMap>();
    topic_map->seq              = static_cast<uint32_t>(obs_time);
    topic_map->header.frame_id  = rostAdapter.get_world_frame();
    topic_map->vocabulary_begin = 0;
    topic_map->vocabulary_size  = static_cast<int32_t>(rost.get_num_topics());
    topic_map->ppx_type         = map_ppx_type;
    topic_map->cell_topics.reserve(rost.cell_pose.size());
    if (map_ppx_type != NO_PPX) topic_map->cell_ppx.reserve(rost.cell_pose.size());
    topic_map->cell_poses.reserve(rost.cell_pose.size() * (POSEDIM - 1));
    topic_map->cell_width                                 = {rostAdapter.get_cell_size().begin() + 1, rostAdapter.get_cell_size().end()};
    topic_map->observation_transform.transform.rotation.w = 1; // Identity rotation (global frame)
    topic_map->observation_transform.header.stamp         = ros::Time::now();

    auto const poses      = rost.cell_pose;
    auto const BATCH_SIZE = 500ul;
    auto i                = 0ul;
    while (i < poses.size()) {
        auto const BATCH_END = std::min(i + BATCH_SIZE, poses.size());
        for (; i < BATCH_END; ++i) {
            auto const &cell_pose    = poses[i];
            auto const &cell         = rost.get_cell(cell_pose);
            auto const word_pose     = rostAdapter.toWordPose(cell_pose);
            auto const ml_cell_topic = std::max_element(cell->nZ.cbegin(), cell->nZ.cend());
            if (ml_cell_topic == cell->nZ.cend()) {
                ROS_ERROR("Cell has no topics! Map will contain invalid topic labels.");
                continue;
            } else {
                topic_map->cell_topics.push_back(int32_t(ml_cell_topic - cell->nZ.cbegin()));
            }
            for (size_t j = 1; j < POSEDIM; j++) { topic_map->cell_poses.push_back(word_pose[j]); }
            if (map_ppx_type != NO_PPX) {
                // TODO: Support cell_ppx
                auto const map_ppx = (map_ppx_type == NEIGHBORHOOD_PPX) ? rost.cell_perplexity_word(cell->W, rost.neighborhood(*cell))
                                                                        : rost.cell_perplexity_word(cell->W, rost.get_topic_weights());
                topic_map->cell_ppx.push_back(map_ppx);
            }
        }
    }
    return topic_map;
}

void topic_model_node::broadcast_topics(int const obs_time, const std::vector<ROSTAdapter<POSEDIM>::cell_pose_t> &broadcast_poses) {
    if (!publish_global_surprise && !publish_local_surprise && !publish_ppx && !publish_topics) return;
    auto &rost = rostAdapter.get_rost();

    static size_t refine_count = rost.get_refine_count();
    ROS_INFO("Running refine rate %.2f cells/ms (avg %.2f), %d active topics, %ld cells",
             static_cast<double>(rost.get_refine_count() - refine_count)
             / std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last_obs_time).count(),
             static_cast<double>(rost.get_refine_count())
             / std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_refine_time).count(),
             rost.get_active_topics(),
             rost.cells.size());
    refine_count = rost.get_refine_count();

    auto time_checkpoint  = std::chrono::steady_clock::now();
    auto const time_start = time_checkpoint;

    using namespace std;
    rostAdapter.wait_for_processing(false);
    auto const duration_wait = record_lap(time_checkpoint);

    sunshine_msgs::WordObservation::Ptr topicObs(new sunshine_msgs::WordObservation);
    topicObs->header.frame_id                            = rostAdapter.get_world_frame();
    topicObs->seq                                        = static_cast<uint32_t>(obs_time);
    topicObs->source                                     = current_source;
    topicObs->vocabulary_begin                           = 0;
    topicObs->vocabulary_size                            = static_cast<int32_t>(rost.get_num_topics());
    topicObs->observation_transform.transform.rotation.w = 1; // Identity rotation (global frame)
    topicObs->observation_transform.header.stamp         = ros::Time::now();

    Perplexity::Ptr global_perplexity(new Perplexity);
    global_perplexity->seq        = static_cast<uint32_t>(obs_time);
    global_perplexity->perplexity = -1;

    auto const &cell_size          = rostAdapter.get_cell_size();
    auto const &current_cell_poses = rostAdapter.get_current_cell_poses();

    LocalSurprise::Ptr global_surprise(new LocalSurprise);
    global_surprise->seq        = static_cast<uint32_t>(obs_time);
    global_surprise->cell_width = {cell_size.begin() + 1, cell_size.end()};
    global_surprise->surprise.reserve(current_cell_poses.size());
    global_surprise->surprise_poses.reserve(current_cell_poses.size() * (POSEDIM - 1));

    LocalSurprise::Ptr local_surprise(new LocalSurprise);
    local_surprise->seq        = static_cast<uint32_t>(obs_time);
    local_surprise->cell_width = {cell_size.begin() + 1, cell_size.end()};
    local_surprise->surprise.reserve(current_cell_poses.size());
    local_surprise->surprise_poses.reserve(current_cell_poses.size() * (POSEDIM - 1));

    TopicWeights::Ptr msg_topic_weights(new TopicWeights);
    msg_topic_weights->seq    = static_cast<uint32_t>(obs_time);
    msg_topic_weights->weight = rost.get_topic_weights();

    auto n_words          = 0ul;
    double sum_log_p_word = 0;
    auto entryIdx         = 0u;

    bool const topics_required   = publish_topics;
    bool const cell_ppx_required = publish_ppx;

    auto const duration_init = record_lap(time_checkpoint);
    long duration_lock;
    {
        auto rostReadToken = rost.get_read_token(); // TODO add a timeout here?
        duration_lock      = record_lap(time_checkpoint);
        for (auto const &cell_pose : broadcast_poses) {
            auto const &cell     = rost.get_cell(cell_pose);
            auto const word_pose = rostAdapter.toWordPose(cell_pose);

            vector<int> topics;             // topic labels for each word in the cell
            double cell_log_likelihood = 0; // cell's sum_w log(p(w | model) = log p(cell | model)
            double cell_ppx, neighborhood_ppx, global_ppx;

            if (topics_required || cell_ppx_required) {
                tie(topics, cell_log_likelihood) = rostAdapter.get_cell_topics_and_ppx(*rostReadToken, cell_pose);

                if (publish_topics) {
                    // populate the topic label message
                    topicObs->words.reserve(topicObs->words.size() + topics.size());
                    topicObs->word_pose.reserve(topicObs->word_pose.size() + topics.size() * POSEDIM);
                    topicObs->words.insert(topicObs->words.end(), topics.begin(), topics.end());
                    for (size_t i = 0; i < topics.size(); i++) {
                        for (size_t j = 1; j < POSEDIM; j++) { topicObs->word_pose.push_back(word_pose[j]); }
                    }
                }

                cell_ppx = exp(-cell_log_likelihood / topics.size()); // TODO: Publish somewhere
                n_words += topics.size();                             // used to compute total ppx
                assert(n_words * (POSEDIM - 1) == topicObs->word_pose.size());
            }

            if (publish_local_surprise) {
                for (size_t i = 1; i < POSEDIM; i++) local_surprise->surprise_poses.push_back(cell_pose[i]);
                local_surprise->surprise.push_back(rost.cell_perplexity_word(cell->W, rost.neighborhood(*cell)));
            }

            if (publish_global_surprise) {
                for (size_t i = 1; i < POSEDIM; i++) global_surprise->surprise_poses.push_back(cell_pose[i]);
                global_surprise->surprise.push_back(rost.cell_perplexity_word(cell->W, rost.get_topic_weights()));
            }

            sum_log_p_word += cell_log_likelihood;
            entryIdx++;
        }
    }
    auto const duration_populate = record_lap(time_checkpoint);

    if (publish_topics) {
        scene_pub.publish(topicObs);
        topic_weights_pub.publish(msg_topic_weights);
    }

    if (publish_ppx) {
        global_perplexity->perplexity = exp(-sum_log_p_word / n_words);
        ROS_INFO("Perplexity: %f", global_perplexity->perplexity);
        global_perplexity_pub.publish(global_perplexity);
    }

    if (publish_global_surprise) { global_surprise_pub.publish(global_surprise); }

    if (publish_local_surprise) { local_surprise_pub.publish(local_surprise); }
    auto const duration_publish = record_lap(time_checkpoint);

    if (map_publish_period == 0) {
        auto topic_map = generate_topic_map(obs_time);
        map_pub.publish(topic_map);
    }
    auto const duration_map = record_lap(time_checkpoint);

    auto const total_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_start).count();
    ROS_INFO("Broadcast overhead: %lu ms (%lu waiting, %lu initializing, %lu locked, %lu populating messages, %lu publishing, %lu mapping)",
             total_duration, duration_wait, duration_init, duration_lock, duration_populate, duration_publish, duration_map);
}

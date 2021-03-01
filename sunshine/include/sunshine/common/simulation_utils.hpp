//
// Created by stewart on 4/28/20.
//

#ifndef SUNSHINE_PROJECT_SIMULATION_UTILS_HPP
#define SUNSHINE_PROJECT_SIMULATION_UTILS_HPP

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <tf2_msgs/TFMessage.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <utility>

#include <set>
#include <cmath>
#include "sunshine/rost_adapter.hpp"

#include "sunshine/visual_word_adapter.hpp"
#include "sunshine/semantic_label_adapter.hpp"
#include "sunshine/segmentation_adapter.hpp"
#include "sunshine/observation_transform_adapter.hpp"
#include "sunshine/2d_adapter.hpp"
#include "sunshine/depth_adapter.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/ros_conversions.hpp"
#include "sunshine/common/rosbag_utils.hpp"

namespace sunshine {

class RobotSim {
    std::string const name;
    std::unique_ptr<BagIterator> bagIter;
    VisualWordAdapter visualWordAdapter;
    std::shared_ptr<ROSTAdapter<4, double, double>> rostAdapter;
    std::shared_ptr<ROSTAdapter<4, double, double>> externalRostAdapter = nullptr;
    std::shared_ptr<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>> segmentationAdapter;
    ObservationTransformAdapter<WordDepthAdapter::Output> wordTransformAdapter;
    ObservationTransformAdapter<ImageDepthAdapter::Output> imageTransformAdapter;
    tf::TransformBroadcaster tfBroadcaster;
    std::unique_ptr<WordDepthAdapter> wordDepthAdapter;
    std::unique_ptr<ImageDepthAdapter> imageDepthAdapter;
    std::unique_ptr<Word2DAdapter<3>> word2dAdapter;
    std::unique_ptr<Image2DAdapter<3>> image2dAdapter;
    sensor_msgs::Image::ConstPtr lastRgb, lastSegmentation;
    sensor_msgs::PointCloud2::ConstPtr lastPc;
    std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>> segmentation;
    ros::Time depth_timestamp = ros::Time(0.0);
//    bool transform_found = false;
    bool processed_rgb = false;
    bool const use_3d;
    bool const read_only_segmentation;
    bool use_segmentation = false;
    ros::Time latestTransformTime = ros::Time(0);
    size_t bag_num = 0;
    decltype(std::chrono::steady_clock::now()) clock = std::chrono::steady_clock::now();
    bool broadcast_tf = false;

    bool tryProcess() {
        if (!lastRgb || processed_rgb) return false;
        if (use_3d && (latestTransformTime < lastRgb->header.stamp || lastRgb->header.stamp != depth_timestamp)) return false;
        if (use_segmentation && (!lastSegmentation || lastRgb->header.stamp != lastSegmentation->header.stamp)) return false;
        assert(!use_segmentation || (lastSegmentation->header.frame_id == lastRgb->header.frame_id));
        #ifndef NDEBUG
        static char ID_LETTER = 'A';
        thread_local char THREAD_LETTER = ID_LETTER++;
        ROS_DEBUG("THREAD %c PROCESSING NEW OBSERVATION", THREAD_LETTER);
        ROS_DEBUG("RGB: #%d, %f s, %s", lastRgb->header.seq, lastRgb->header.stamp.toSec(), lastRgb->header.frame_id.c_str());
        if (use_segmentation) ROS_DEBUG("Seg: #%d, %f s, %s", lastSegmentation->header.seq, lastSegmentation->header.stamp.toSec(), lastSegmentation->header.frame_id.c_str());
        #endif
        ROS_DEBUG("%ld ms since last observation", record_lap(clock));
        auto newRgb = std::make_unique<ImageObservation>(fromRosMsg(lastRgb));
        newRgb->frame = fixFrame(newRgb->frame);
        auto newSegmentation = (use_segmentation) ? std::make_unique<ImageObservation>(fromRosMsg(lastSegmentation)) : nullptr;
        if (newSegmentation) newSegmentation->frame = fixFrame(newSegmentation->frame);
        ROS_DEBUG("%ld ms parsing images", record_lap(clock));

        auto const word_poses_toCellSet = [this](auto const& poses, double timestamp){
            std::set<std::array<int, 3>> reduced_dim_poses;
            for (auto const& pose : poses) {
                auto const cell_pose = rostAdapter->toCellId({timestamp, pose[0], pose[1], pose[2]});
                reduced_dim_poses.insert({cell_pose[1], cell_pose[2], cell_pose[3]});
            }
            return reduced_dim_poses;
        };

        auto const cells_toCellSet = [](auto const& poses){
            std::set<std::array<int, 3>> reduced_dim_poses;
            for (auto const& pose : poses) {
                reduced_dim_poses.insert({pose[1], pose[2], pose[3]});
            }
            return reduced_dim_poses;
        };

        // TODO: remove duplication between if branches below
        if (use_3d) {
            tf::StampedTransform transform;
            try {
                transform = wordTransformAdapter.getLatestTransform(fixFrame(lastRgb->header.frame_id), lastRgb->header.stamp);
                #ifndef NDEBUG
                ROS_DEBUG("Depth: #%d, %f s, %s", lastPc->header.seq, lastPc->header.stamp.toSec(), lastPc->header.frame_id.c_str());
                ROS_DEBUG("Transform: %f s, %s, child: %s", transform.stamp_.toSec(), transform.frame_id_.c_str(), transform.child_frame_id_.c_str());
                #endif
            } catch (tf::ExtrapolationException const& ex) {
                ROS_WARN("Extrapolation exceptions shouldn't happen since we have latestTransformTime...");
                return false;
            } catch (...) {
                ROS_ERROR("Unexpected error when checking for transform!");
                return false;
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::PCLPointCloud2 pcl_pc2;
            pcl_conversions::toPCL(*lastPc, pcl_pc2);
            pcl::fromPCLPointCloud2(pcl_pc2, *pc);
            pc->header.frame_id = fixFrame(pc->header.frame_id);
            wordDepthAdapter->updatePointCloud(pc);
            imageDepthAdapter->updatePointCloud(pc);
            ROS_DEBUG("%ld ms parsing depth cloud", record_lap(clock)); // ~3ms optimized

            auto observation = wordTransformAdapter(newRgb >> visualWordAdapter >> *wordDepthAdapter, transform);
#ifndef NDEBUG
            auto old_observation_poses = cells_toCellSet(rostAdapter->get_rost().cell_pose);
            auto new_observation_poses = word_poses_toCellSet(observation->observation_poses, observation->timestamp);
            std::vector<std::array<int, 3>> new_poses;
            std::set_difference(new_observation_poses.begin(), new_observation_poses.end(), old_observation_poses.begin(), old_observation_poses.end(), std::back_inserter(new_poses));
            ROS_DEBUG("Adding %lu observations across %lu new cell poses (%lu unique)", observation->observations.size(), new_observation_poses.size(), new_poses.size());
#endif
            if (externalRostAdapter) {
                observation >> *rostAdapter;
                observation >> *externalRostAdapter;
            } else {
                (*rostAdapter)(std::move(observation));
            }
            ROS_DEBUG("%ld ms adding to topic model", record_lap(clock)); // ~40-80ms optimized
            if (use_segmentation && !read_only_segmentation) {
                auto single_segmentation = imageTransformAdapter(newSegmentation >> *imageDepthAdapter, transform);
                [[maybe_unused]] auto lock = segmentationAdapter->getLock();
                segmentation = std::move(single_segmentation) >> *segmentationAdapter;
            }
            ROS_DEBUG("%ld ms parsing segmentation", record_lap(clock)); // ~30-40ms optimized
            auto final_observation_poses = cells_toCellSet(rostAdapter->get_rost().cell_pose);
            {
                [[maybe_unused]] auto lock = segmentationAdapter->getLock();
                if (use_segmentation && !includes(segmentationAdapter->getRawCounts(), final_observation_poses)) {
                    throw std::runtime_error("Latest observation includes unrecongized poses!");
                }
            }
            ROS_DEBUG("%ld ms validating poses", record_lap(clock));
        } else {
            auto observation = newRgb >> visualWordAdapter >> *word2dAdapter;
            if (externalRostAdapter) {
                observation >> *rostAdapter;
                observation >> *externalRostAdapter;
            } else {
                (*rostAdapter)(std::move(observation));
            }
//            ROS_INFO("%ld ms adding to 2d topic model", record_lap(clock));
            if (use_segmentation && !read_only_segmentation) {
                [[maybe_unused]] auto lock = segmentationAdapter->getLock();
                segmentation = newSegmentation >> *image2dAdapter >> *segmentationAdapter;
            }
//            ROS_INFO("%ld ms parsing 2d segmentation", record_lap(clock));
        }
        processed_rgb = true;
        return true;
    }

    bool imageCallback(sensor_msgs::Image::ConstPtr image) {
//        ROS_INFO("%ld ms before entering imageCallback", record_lap(clock));
        lastRgb = std::move(image);
        processed_rgb = false;
        return tryProcess();
    }

    bool segmentationCallback(sensor_msgs::Image::ConstPtr image) {
//        ROS_INFO("%ld ms before entering segmentationCallback", record_lap(clock));
        lastSegmentation = std::move(image);
        return tryProcess();
    }

    bool depthCallback(sensor_msgs::PointCloud2::ConstPtr msg) {
//        ROS_INFO("%ld ms before entering depthCallback", record_lap(clock));
        depth_timestamp = msg->header.stamp;
        lastPc = std::move(msg);
        return tryProcess();
    };

    bool transformCallback(tf2_msgs::TFMessage::ConstPtr tfMsg) {
//        ROS_INFO("%ld ms before entering transformCallback", record_lap(clock));
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            if (lastRgb && tfTransform.child_frame_id_ == lastRgb->header.frame_id) {
                latestTransformTime = tfTransform.stamp_;
//                transform_found = true;
            }
            tfTransform.child_frame_id_ = fixFrame(tfTransform.child_frame_id_);
//            ROS_INFO("Adding transform at %f from %s to %s", tfTransform.stamp_.toSec(), tfTransform.frame_id_.c_str(), tfTransform.child_frame_id_.c_str());
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
            if (broadcast_tf) {
                tfTransform.stamp_ = ros::Time::now();
                tfBroadcaster.sendTransform(tfTransform);
            }
        }
//        ROS_INFO("%ld ms processing transforms", record_lap(clock));
        return tryProcess();
    }

  public:
    template<typename ParamServer>
    RobotSim(std::string name,
             ParamServer const &parameters,
             bool const use_3d,
             std::shared_ptr<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>> shared_seg_adapter = nullptr,
             bool const read_only_segmentation = false,
             bool const broadcast_tf = false)
            : name(std::move(name)),
              bagIter(nullptr),
              visualWordAdapter(&parameters),
              rostAdapter(std::make_shared<ROSTAdapter<4, double, double>>(&parameters, nullptr, std::vector<std::vector<int>>(), false)),
              segmentationAdapter((shared_seg_adapter)
                                  ? std::move(shared_seg_adapter)
                                  : std::make_shared<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>>(&parameters,
                                                                                                                            true)),
              wordTransformAdapter(&parameters),
              imageTransformAdapter(&parameters),
              use_3d(use_3d),
              read_only_segmentation(read_only_segmentation),
              broadcast_tf(broadcast_tf) {}

    void open(std::string const& bagFilename,
              std::string const &image_topic,
              std::string const &depth_topic = "",
              std::string const &segmentation_topic= "",
              double const _2d_x_offset = 0) {
        if (use_3d && depth_topic.empty()) throw std::invalid_argument("Must provide depth topic if operating in 3D mode");
        wordDepthAdapter = (depth_topic.empty()) ? nullptr : std::make_unique<WordDepthAdapter>();
        imageDepthAdapter = (depth_topic.empty()) ? nullptr : std::make_unique<ImageDepthAdapter>();
        word2dAdapter = (depth_topic.empty()) ? std::make_unique<Word2DAdapter<3>>(_2d_x_offset, 0, 0, true) : nullptr;
        image2dAdapter = (depth_topic.empty()) ? std::make_unique<Image2DAdapter<3>>(_2d_x_offset, 0, 0, true) : nullptr;
        use_segmentation = !segmentation_topic.empty();

        bag_num++;
        latestTransformTime = ros::Time(0);

        bagIter = std::make_unique<BagIterator>(bagFilename);
        bagIter->add_callback(image_topic, [this](rosbag::MessageInstance const &msg) { return this->imageCallback(msg.instantiate<sensor_msgs::Image>()); });
        bagIter->add_callback(segmentation_topic, [this](rosbag::MessageInstance const &msg) { return this->segmentationCallback(msg.instantiate<sensor_msgs::Image>()); });
        bagIter->add_callback(depth_topic, [this](rosbag::MessageInstance const &msg) { return this->depthCallback(msg.instantiate<sensor_msgs::PointCloud2>()); });
        bagIter->add_callback("/tf", [this](rosbag::MessageInstance const &msg) { return this->transformCallback(msg.instantiate<tf2_msgs::TFMessage>()); });
        bagIter->set_logging(true);
    }

    Phi getTopicModel(activity_manager::ReadToken const& token) const {
        return rostAdapter->get_topic_model(token);
    }

    auto getRost() {
        return rostAdapter;
    }

    auto getMap(activity_manager::ReadToken const& token) const {
        return rostAdapter->get_map(token);
    }

    auto getDistMap(activity_manager::ReadToken const& token) const {
        return rostAdapter->get_dist_map(token);
    }

    auto getGTMap() const {
        double const timestamp = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (segmentation) segmentation->id = ros::Time(timestamp).sec;
        return segmentation;
    }

    auto getLastPc() const {
        return lastPc;
    }

    auto getLatestTransform(std::string const& frame) const {
        return wordTransformAdapter.getLatestTransform(frame, ros::Time(0));
    }

    [[nodiscard]] inline std::string fixFrame(std::string const& frame) const {
        return frame + "-" + name + "-" + std::to_string(bag_num);
    }

    /**
     *
     * @return false if finished, true if there are more messages to simulate
     */
    bool next() {
//        ROS_INFO("%ld ms entering next()", record_lap(clock));
        if (!bagIter) throw std::logic_error("No bag opened to play");
        auto const start = std::chrono::steady_clock::now();
        auto const ret = !bagIter->play(false);
        auto const duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        ROS_WARN_COND(duration.count() > 400, "%s spent %ld ms playing bag", name.c_str(), duration.count());
        return ret;
    }

    void waitForProcessing() const {
        rostAdapter->wait_for_processing(false);
    }

    [[nodiscard]] auto getReadToken() const {
        return rostAdapter->get_rost().get_read_token();
    }

    void pause() {
        rostAdapter->stopWorkers();
    }

    std::string getName() const {
        return name;
    }
};

template<typename Metric>
std::tuple<double, size_t, size_t> benchmark(std::string const &bagfile,
                 std::string const &image_topic,
                 std::string const &segmentation_topic,
                 std::string const &depth_topic,
                 sunshine::Parameters const &parameters,
                 Metric const &metric,
                 uint32_t const warmup = 0,
                 uint32_t const max_iter = std::numeric_limits<uint32_t>::max(),
                 bool const average = false) {
    // TODO Use RobotSim
    using namespace sunshine;

    auto visualWordAdapter = VisualWordAdapter(&parameters);
//    auto labelSegmentationAdapter = SemanticSegmentationAdapter<int, std::vector<int>>(&parameters);
    auto rostAdapter = ROSTAdapter<4, double, double>(&parameters, nullptr, {}, average);
    auto segmentationAdapter = SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>(&parameters, !average);
    auto wordDepthAdapter = WordDepthAdapter();
    auto imageDepthAdapter = ImageDepthAdapter();
    auto wordTransformAdapter = ObservationTransformAdapter<WordDepthAdapter::Output>(&parameters);
    auto imageTransformAdapter = ObservationTransformAdapter<ImageDepthAdapter::Output>(&parameters);
    uint32_t count = 0;
    double av_metric = 0;

    sensor_msgs::Image::ConstPtr lastRgb, lastSeg;
    sensor_msgs::PointCloud2::ConstPtr lastDepth;
    ros::Time lastMsgTime(0);
    auto const processPair = [&]() {
        if (!lastRgb || lastMsgTime == lastRgb->header.stamp || !lastSeg || !lastDepth || lastRgb->header.stamp != lastDepth->header.stamp || lastRgb->header.stamp != lastSeg->header.stamp) return false;
        tf::StampedTransform transform;
        try {
            transform = wordTransformAdapter.getLatestTransform(lastRgb->header.frame_id, lastRgb->header.stamp);
        } catch (tf::ExtrapolationException const& ex) {
            ROS_INFO("Waiting for appropriate transformation.");
            return false;
        } catch (tf::LookupException const& ex) {
            ROS_INFO("Waiting for appropriate transformation.");
            return false;
        }
        ROS_DEBUG("Processing t=%f", lastRgb->header.stamp.toSec());
        lastMsgTime = lastRgb->header.stamp;

        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*lastDepth, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pc);
        wordDepthAdapter.updatePointCloud(pc);
        imageDepthAdapter.updatePointCloud(pc);

        assert(lastRgb->header.frame_id == lastSeg->header.frame_id);
        auto rgb = std::make_unique<ImageObservation>(fromRosMsg(lastRgb));
        auto segmentation = std::make_unique<ImageObservation>(fromRosMsg(lastSeg));

        if (count > 0) rostAdapter.wait_for_processing(false);
        auto topicsFuture = wordTransformAdapter(rgb >> visualWordAdapter >> wordDepthAdapter, transform) >> rostAdapter;
        auto gtSeg = imageTransformAdapter(segmentation >> imageDepthAdapter, transform) >> segmentationAdapter;

        if (average && count >= warmup) {
            rostAdapter.wait_for_processing(false);
            auto topicsSeg = topicsFuture.get();
            double const result = metric(*gtSeg, *topicsSeg);
            av_metric = (av_metric * (count - warmup) + result) / (count - warmup + 1);
        }
        count += 1;

        if (count == warmup + 1) {
            std::cout << threadString("Finished warmup.") << std::endl;
        } else if (count % 100 == 0) {
            std::cout << threadString("Processed observations: ") << count << std::endl;
        }
        return count >= max_iter;
    };

    auto const imageCallback = [&](sensor_msgs::Image::ConstPtr image) {
        lastRgb = std::move(image);
        return processPair();
    };

    auto const segmentationCallback = [&](sensor_msgs::Image::ConstPtr image) {
        lastSeg = std::move(image);
        return processPair();
    };

    auto const depthCallback = [&](sensor_msgs::PointCloud2::ConstPtr msg) {
        lastDepth = std::move(msg);
        return processPair();
    };

    auto const transformCallback = [&](tf2_msgs::TFMessage::ConstPtr const &tfMsg) {
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
        }
        return processPair();
    };

    ROS_WARN("Extracting images from %s", bagfile.c_str());
    sunshine::BagIterator bagIter(bagfile);
    bagIter.add_callback<sensor_msgs::Image>(image_topic, imageCallback);
    bagIter.add_callback<sensor_msgs::Image>(segmentation_topic, segmentationCallback);
    bagIter.add_callback<sensor_msgs::PointCloud2>(depth_topic, depthCallback);
    bagIter.add_callback<tf2_msgs::TFMessage>("/tf", transformCallback);
    bagIter.set_logging(true);
    auto const finished = bagIter.play();
    ROS_ERROR_COND(!finished, "Failed to finish playing bagfile!");
    ROS_INFO("Processed %u images from rosbag.", count);

    if (!average) {
        auto readToken = rostAdapter.get_rost().get_read_token();
        auto topicsSeg = rostAdapter.get_dist_map(*readToken);
        auto gtSeg = segmentationAdapter(nullptr);
        av_metric = metric(*gtSeg, *topicsSeg);
    }
    return {av_metric, rostAdapter.get_rost().get_refine_count(), rostAdapter.get_rost().get_word_refine_count()};
}
}

#endif //SUNSHINE_PROJECT_SIMULATION_UTILS_HPP

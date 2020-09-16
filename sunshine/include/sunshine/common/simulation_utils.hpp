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
    std::shared_ptr<ROSTAdapter<4, double, double>> externalRostAdapter;
    std::shared_ptr<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>> segmentationAdapter;
    ObservationTransformAdapter<WordDepthAdapter::Output> wordTransformAdapter;
    ObservationTransformAdapter<ImageDepthAdapter::Output> imageTransformAdapter;
    std::unique_ptr<WordDepthAdapter> wordDepthAdapter;
    std::unique_ptr<ImageDepthAdapter> imageDepthAdapter;
    std::unique_ptr<Word2DAdapter<3>> word2dAdapter;
    std::unique_ptr<Image2DAdapter<3>> image2dAdapter;
    sensor_msgs::Image::ConstPtr lastRgb, lastSegmentation;
    sensor_msgs::PointCloud2::ConstPtr lastPc;
    std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>> segmentation;
    double depth_timestamp = -1;
    bool transform_found = false;
    bool processed_rgb = false;
    bool const use_3d;
    bool use_segmentation = false;
    decltype(std::chrono::steady_clock::now()) clock = std::chrono::steady_clock::now();

    bool tryProcess() {
        if (!lastRgb || processed_rgb) return false;
        if (use_3d && (!transform_found || lastRgb->header.stamp.toSec() != depth_timestamp)) return false;
        if (use_segmentation && (!lastSegmentation || lastRgb->header.stamp.toSec() != lastSegmentation->header.stamp.toSec())) return false;
        assert(!use_segmentation || (lastSegmentation->header.frame_id == lastRgb->header.frame_id));

        ROS_DEBUG("%ld ms since last observation", record_lap(clock));
        auto newRgb = std::make_unique<ImageObservation>(fromRosMsg(lastRgb));
        auto newSegmentation = (use_segmentation) ? std::make_unique<ImageObservation>(fromRosMsg(lastSegmentation)) : nullptr;
        newRgb->timestamp = ros::Time::now().toSec();
        ROS_DEBUG("%ld ms parsing images", record_lap(clock));

        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*lastPc, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pc);
        wordDepthAdapter->updatePointCloud(pc);
        imageDepthAdapter->updatePointCloud(pc);
        ROS_DEBUG("%ld ms parsing depth cloud", record_lap(clock)); // ~3ms optimized

        // TODO: remove duplication between if branches below
        if (use_3d) {
            auto observation = newRgb >> visualWordAdapter >> *wordDepthAdapter >> wordTransformAdapter;
            if (externalRostAdapter) {
                observation >> *rostAdapter;
                observation >> *externalRostAdapter;
            } else {
                (*rostAdapter)(std::move(observation));
            }
            ROS_DEBUG("%ld ms adding to topic model", record_lap(clock)); // ~40-80ms optimized
            if (use_segmentation) segmentation = newSegmentation >> *imageDepthAdapter >> imageTransformAdapter >> *segmentationAdapter;
            ROS_DEBUG("%ld ms parsing segmentation", record_lap(clock)); // ~30-40ms optimized
            std::vector<std::array<int, 3>> reduced_dim_poses;
            reduced_dim_poses.reserve(rostAdapter->get_rost().cell_pose.size());
            for (auto const& pose : rostAdapter->get_rost().cell_pose) {
                reduced_dim_poses.push_back({pose[1], pose[2], pose[3]});
            }
            if (use_segmentation && !includes(segmentationAdapter->getRawCounts(), reduced_dim_poses)) throw std::runtime_error("Latest observation includes unrecongized poses!");
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
            if (use_segmentation) segmentation = newSegmentation >> *image2dAdapter >> *segmentationAdapter;
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
        depth_timestamp = msg->header.stamp.toSec();
        lastPc = std::move(msg);
        return tryProcess();
    };

    bool transformCallback(tf2_msgs::TFMessage::ConstPtr tfMsg) {
//        ROS_INFO("%ld ms before entering transformCallback", record_lap(clock));
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            tfTransform.stamp_ = ros::Time::now();
            if (lastRgb && (tfTransform.frame_id_ == lastRgb->header.frame_id || tfTransform.child_frame_id_ == lastRgb->header.frame_id)) transform_found = true;
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
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
             std::shared_ptr<ROSTAdapter<4, double, double>> external_rost_adapter = nullptr)
            : name(std::move(name)),
              bagIter(nullptr),
              visualWordAdapter(&parameters),
              rostAdapter(std::make_shared<ROSTAdapter<4, double, double>>(&parameters)),
              externalRostAdapter(std::move(external_rost_adapter)),
              segmentationAdapter((shared_seg_adapter)
                                  ? std::move(shared_seg_adapter)
                                  : std::make_shared<SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>>(&parameters,
                                                                                                                            true)),
              wordTransformAdapter(&parameters),
              imageTransformAdapter(&parameters),
              use_3d(use_3d) {}

    void open(std::string const& bagFilename,
              std::string const &image_topic,
              std::string const &depth_topic = "",
              std::string const &segmentation_topic= "",
              double x_offset = 0) {
        if (use_3d && depth_topic.empty()) throw std::invalid_argument("Must provide depth topic if operating in 3D mode");
        wordDepthAdapter = (depth_topic.empty()) ? nullptr : std::make_unique<WordDepthAdapter>();
        imageDepthAdapter = (depth_topic.empty()) ? nullptr : std::make_unique<ImageDepthAdapter>();
        word2dAdapter = (depth_topic.empty()) ? std::make_unique<Word2DAdapter<3>>(x_offset, 0, 0, true) : nullptr;
        image2dAdapter = (depth_topic.empty()) ? std::make_unique<Image2DAdapter<3>>(x_offset, 0, 0, true) : nullptr;
        use_segmentation = !segmentation_topic.empty();

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
        ROS_WARN_COND(duration.count() > 300, "%s spent %ld ms playing bag", name.c_str(), duration.count());
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
double benchmark(std::string const &bagfile,
                 std::string const &image_topic,
                 std::string const &segmentation_topic,
                 std::string const &depth_topic,
                 sunshine::Parameters const &parameters,
                 Metric const &metric,
                 uint32_t const warmup = 0,
                 uint32_t const max_iter = std::numeric_limits<uint32_t>::max()) {
    // TODO Use RobotSim
    using namespace sunshine;

    auto visualWordAdapter = VisualWordAdapter(&parameters);
//    auto labelSegmentationAdapter = SemanticSegmentationAdapter<int, std::vector<int>>(&parameters);
    auto rostAdapter = ROSTAdapter<4, double, double>(&parameters);
    auto segmentationAdapter = SemanticSegmentationAdapter<std::array<uint8_t, 3>, std::vector<int>>(&parameters);
    auto wordDepthAdapter = WordDepthAdapter();
    auto imageDepthAdapter = ImageDepthAdapter();
    auto wordTransformAdapter = ObservationTransformAdapter<WordDepthAdapter::Output>(&parameters);
    auto imageTransformAdapter = ObservationTransformAdapter<ImageDepthAdapter::Output>(&parameters);
    uint32_t count = 0;
    double av_metric = 0;

    std::unique_ptr<ImageObservation> rgb, segmentation;
    double depth_timestamp = -1;
    auto const processPair = [&]() {
        auto topicsFuture = rgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;
        auto gtSeg = segmentation >> imageDepthAdapter >> imageTransformAdapter >> segmentationAdapter;
        auto topicsSeg = topicsFuture.get();

        if (count >= warmup) {
            double const result = metric(*gtSeg, *topicsSeg);
            av_metric = (av_metric * (count - warmup) + result) / (count - warmup + 1);
        }
        count += 1;
        return count >= max_iter;
    };

    auto const imageCallback = [&](sensor_msgs::Image::ConstPtr image) {
        rgb = std::make_unique<ImageObservation>(fromRosMsg(image));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            return processPair();
        }
        return false;
    };

    auto const segmentationCallback = [&](sensor_msgs::Image::ConstPtr image) {
        segmentation = std::make_unique<ImageObservation>(fromRosMsg(image));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            return processPair();
        }
        return false;
    };

    auto const depthCallback = [&](sensor_msgs::PointCloud2::ConstPtr const &msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pc);

        wordDepthAdapter.updatePointCloud(pc);
        imageDepthAdapter.updatePointCloud(pc);
        depth_timestamp = msg->header.stamp.toSec();
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            return processPair();
        }
        return false;
    };

    auto const transformCallback = [&](tf2_msgs::TFMessage::ConstPtr const &tfMsg) {
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
        }
        return false;
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
    return av_metric;
}
}

#endif //SUNSHINE_PROJECT_SIMULATION_UTILS_HPP

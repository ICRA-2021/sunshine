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
    std::unique_ptr<ImageObservation> lastRgb, lastSegmentation;
    std::shared_ptr<Segmentation<std::vector<int>, 3, int, double>> segmentation;
    double depth_timestamp = -1;
    bool transform_found = false;
    bool processed_rgb = false;
    bool const use_3d;
    bool use_segmentation;

    bool tryProcess() {
        if (!lastRgb || processed_rgb) return false;
        if (use_3d && (!transform_found || lastRgb->timestamp != depth_timestamp)) return false;
        if (use_segmentation && (!lastSegmentation || lastRgb->timestamp != lastSegmentation->timestamp)) return false;
        // TODO: remove duplication between if branches below
        lastRgb->timestamp = ros::Time::now().toSec();
        if (use_3d) {
            auto observation = lastRgb >> visualWordAdapter >> *wordDepthAdapter >> wordTransformAdapter;
            if (externalRostAdapter) {
                observation >> *rostAdapter;
                observation >> *externalRostAdapter;
            } else {
                (*rostAdapter)(std::move(observation));
            }
            if (use_segmentation) segmentation = lastSegmentation >> *imageDepthAdapter >> imageTransformAdapter >> *segmentationAdapter;
        } else {
            auto observation = lastRgb >> visualWordAdapter >> *word2dAdapter;
            if (externalRostAdapter) {
                observation >> *rostAdapter;
                observation >> *externalRostAdapter;
            } else {
                (*rostAdapter)(std::move(observation));
            }
            if (use_segmentation) segmentation = lastSegmentation >> *image2dAdapter >> *segmentationAdapter;
        }
        processed_rgb = true;
        return true;
    }

    bool imageCallback(sensor_msgs::Image::ConstPtr image) {
        lastRgb = std::make_unique<ImageObservation>(fromRosMsg(image));
        processed_rgb = false;
        return tryProcess();
    }

    bool segmentationCallback(sensor_msgs::Image::ConstPtr image) {
        lastSegmentation = std::make_unique<ImageObservation>(fromRosMsg(image));
        return tryProcess();
    }

    bool depthCallback(sensor_msgs::PointCloud2::ConstPtr const &msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *pc);

        wordDepthAdapter->updatePointCloud(pc);
        imageDepthAdapter->updatePointCloud(pc);
        depth_timestamp = msg->header.stamp.toSec();
        return tryProcess();
    };

    bool transformCallback(tf2_msgs::TFMessage::ConstPtr const &tfMsg) {
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            tfTransform.stamp_ = ros::Time::now();
            if (lastRgb && (tfTransform.frame_id_ == lastRgb->frame || tfTransform.child_frame_id_ == lastRgb->frame)) transform_found = true;
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
        }
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
        bagIter->add_callback<sensor_msgs::Image>(image_topic, [this](auto const &msg) { return this->imageCallback(msg); });
        bagIter->add_callback<sensor_msgs::Image>(segmentation_topic, [this](auto const &msg) { return this->segmentationCallback(msg); });
        bagIter->add_callback<sensor_msgs::PointCloud2>(depth_topic, [this](auto const &msg) { return this->depthCallback(msg); });
        bagIter->add_callback<tf2_msgs::TFMessage>("/tf", [this](auto const &msg) { return this->transformCallback(msg); });
        bagIter->set_logging(true);
    }

    Phi getTopicModel(bool wait_for_refine = false) const {
        if (wait_for_refine) rostAdapter->wait_for_processing(false);
        auto token = rostAdapter->get_rost().get_read_token();
        return rostAdapter->get_topic_model(*token);
    }

    auto getRost() {
        return rostAdapter;
    }

    auto getMap(bool wait_for_refine = true) const {
        if (wait_for_refine) rostAdapter->wait_for_processing(false);
        auto token = rostAdapter->get_rost().get_read_token();
        return rostAdapter->get_map(*token);
    }

    auto getDistMap(bool wait_for_refine = true) const {
        if (wait_for_refine) rostAdapter->wait_for_processing(false);
        auto token = rostAdapter->get_rost().get_read_token();
        return rostAdapter->get_dist_map(*token);
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
        if (!bagIter) throw std::logic_error("No bag opened to play");
        return !bagIter->play(false);
    }

    void waitForProcessing() const {
        rostAdapter->wait_for_processing(false);
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
        rgb = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            return processPair();
        }
        return false;
    };

    auto const segmentationCallback = [&](sensor_msgs::Image::ConstPtr image) {
        segmentation = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
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

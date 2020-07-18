//
// Created by stewart on 4/28/20.
//

#ifndef SUNSHINE_PROJECT_BENCHMARK_HPP
#define SUNSHINE_PROJECT_BENCHMARK_HPP

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
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
#include "sunshine/depth_adapter.hpp"
#include "sunshine/common/parameters.hpp"
#include "sunshine/common/ros_conversions.hpp"

namespace sunshine {
template<typename Metric>
double benchmark(std::string const &bagfile,
                 std::string const &image_topic,
                 std::string const &segmentation_topic,
                 std::string const &depth_topic,
                 sunshine::Parameters const &parameters,
                 Metric const &metric,
                 uint32_t const warmup = 0,
                 uint32_t const max_iter = std::numeric_limits<uint32_t>::max()) {
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
//        std::cout << (std::equal(segmentation->image.begin<uint8_t>(), segmentation->image.end<uint8_t>(), rgb->image.begin<uint8_t>())
//                      ? "match"
//                      : "not match") << std::endl;
//        auto wordObs = wordTransformAdapter(wordDepthAdapter(visualWordAdapter(rgb.get())));
        auto topicsFuture = rgb >> visualWordAdapter >> wordDepthAdapter >> wordTransformAdapter >> rostAdapter;

//        auto const segObs = imageTransformAdapter(imageDepthAdapter(segmentation));
        auto gtSeg = segmentation >> imageDepthAdapter >> imageTransformAdapter >> segmentationAdapter;
        auto topicsSeg = topicsFuture.get();
//        auto topicsSeg = rgb >> imageDepthAdapter >> imageTransformAdapter >> segmentationAdapter;
        if (count >= warmup) {
            double const result = metric(*gtSeg, *topicsSeg);
//            std::cerr << result << std::endl;
            av_metric = (av_metric * (count - warmup) + result) / (count - warmup + 1);
        }
        count += 1;
    };

    auto const imageCallback = [&](sensor_msgs::Image::ConstPtr image) {
        rgb = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            processPair();
        }
    };

    auto const segmentationCallback = [&](sensor_msgs::Image::ConstPtr image) {
        segmentation = std::make_unique<ImageObservation>(fromRosMsg(std::move(image)));
        if (rgb && segmentation && rgb->timestamp == segmentation->timestamp && rgb->timestamp == depth_timestamp) {
            processPair();
        }
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
            processPair();
        }
    };

    auto const transformCallback = [&](tf2_msgs::TFMessage::ConstPtr const &tfMsg) {
        for (auto const &transform : tfMsg->transforms) {
            tf::StampedTransform tfTransform;
            tf::transformStampedMsgToTF(transform, tfTransform);
            wordTransformAdapter.addTransform(tfTransform);
            imageTransformAdapter.addTransform(tfTransform);
        }
    };

    ROS_WARN("Extracting images from %s", bagfile.c_str());
    rosbag::Bag bag;
    bag.open(bagfile);

    std::set<std::string> skipped_topics;
    for (rosbag::MessageInstance const m: rosbag::View(bag)) {
        bool handled = false;
        if (m.getTopic() == image_topic || m.getTopic() == segmentation_topic) {
            sensor_msgs::Image::ConstPtr imgMsg = m.instantiate<sensor_msgs::Image>();
            if (imgMsg != nullptr) {
                if (m.getTopic() == image_topic) imageCallback(imgMsg);
                if (m.getTopic() == segmentation_topic) segmentationCallback(imgMsg);
            } else {
                ROS_ERROR("Non-image message found in topic %s", m.getTopic().c_str());
            }
            handled = true;
        }
        if (m.getTopic() == depth_topic) {
            depthCallback(m.instantiate<sensor_msgs::PointCloud2>());
            handled = true;
        }
        if (m.getTopic() == "/tf") {
            transformCallback(m.instantiate<tf2_msgs::TFMessage>());
            handled = true;
        }
        if (!handled && skipped_topics.find(m.getTopic()) == skipped_topics.end()) {
            ROS_INFO("Skipped message from topic %s", m.getTopic().c_str());
            skipped_topics.insert(m.getTopic());
        }
        if (count >= max_iter) break;
    }
    bag.close();
    ROS_INFO("Processed %u images from rosbag.", count);
    return av_metric;
}

template<typename Container>
double entropy(Container const &container, double weight = 1.0) {
    double sum = 0;
    for (auto const &val : container) {
        if (val > 0) {
            double const pv = (weight == 1.0) ? val : (val / weight);
            sum += pv * log(pv);
        } else {
            assert(val == 0);
        }
    }
    return -sum;
}

template<size_t pose_dimen = 4>
std::tuple<std::vector<std::vector<uint32_t>>, std::vector<uint32_t>, std::vector<uint32_t>, uint32_t> compute_matches(sunshine::Segmentation<std::vector<int>, 3, int, double> const &gt_seg,
                                                                                                                       sunshine::Segmentation<std::vector<int>, pose_dimen, int, double> const &topic_seg) {
    std::map<std::array<int, 3>, uint32_t> gt_labels, topic_labels;
    std::vector<uint32_t> gt_weights(gt_seg.observations[0].size(), 0), topic_weights(topic_seg.observations[0].size(), 0);
    std::vector<std::vector<uint32_t>> matches(topic_seg.observations[0].size(), std::vector<uint32_t>(gt_seg.observations[0].size(), 0));
    double total_weight = 0;
    for (auto i = 0; i < gt_seg.observations.size(); ++i) {
        auto const label = argmax<>(gt_seg.observations[i]);
        gt_labels.insert({gt_seg.observation_poses[i], label});
    }
    for (auto i = 0; i < topic_seg.observations.size(); ++i) {
        static_assert(pose_dimen == 3 || pose_dimen == 4);
        constexpr size_t offset = (pose_dimen == 4) ? 1 : 0;
        std::array<int, 3> const pose{topic_seg.observation_poses[i][offset], topic_seg.observation_poses[i][1 + offset],
                                      topic_seg.observation_poses[i][2 + offset]};
        auto const topic_label = argmax<>(topic_seg.observations[i]);
        topic_labels.insert({pose, topic_label});

        auto iter = gt_labels.find(pose);
        if (iter != gt_labels.end()) {
            auto const &gt_label = iter->second;
            matches[topic_label][gt_label] += 1;
            gt_weights[gt_label] += 1;
            topic_weights[topic_label] += 1;
            total_weight += 1;
        } else {
            std::cerr << "Failed to find gt gt_seg for pose!" << std::endl;
        }
    }
    return {matches, gt_weights, topic_weights, total_weight};
}

double compute_mutual_info(std::vector<std::vector<uint32_t>> const& matches, std::vector<uint32_t> const& gt_weights, std::vector<uint32_t> const& topic_weights, double const total_weight) {
    std::vector<double> px, py;
    px.reserve(topic_weights.size());
    py.reserve(gt_weights.size());
    for (auto const &topic_weight : topic_weights) px.push_back(topic_weight / total_weight);
    for (auto const &gt_weight : gt_weights) py.push_back(gt_weight / total_weight);

    double mi = 0;
    double sum_pxy = 0;
    for (auto i = 0; i < matches.size(); ++i) {
        for (auto j = 0; j < matches[i].size(); ++j) {
            auto const pxy = matches[i][j] / total_weight;
            if (pxy > 0) {
                mi += pxy * log(pxy / (px[i] * py[j]));
            }
            sum_pxy += pxy;
        }
    }
    assert(sum_pxy >= 0.99 && sum_pxy <= 1.01);

    return mi;
}

double compute_average_entropy_y(std::vector<std::vector<uint32_t>> const& matches, std::vector<uint32_t> const& topic_weights, double const total_weight) {
    std::vector<double> px;
    px.reserve(topic_weights.size());
    for (auto const &topic_weight : topic_weights) px.push_back(topic_weight / total_weight);

    double exp_ey = 0;
    for (auto i = 0; i < matches.size(); ++i) {
        exp_ey += entropy<>(matches[i], topic_weights[i]) * px[i];
    }
    return exp_ey;
}

template<size_t pose_dimen = 4>
double nmi(sunshine::Segmentation<std::vector<int>, 3, int, double> const &gt_seg,
           sunshine::Segmentation<std::vector<int>, pose_dimen, int, double> const &topic_seg) {
    auto const contingency_table = compute_matches<pose_dimen>(gt_seg, topic_seg);
    auto const& matches = std::get<0>(contingency_table);
    auto const& gt_weights = std::get<1>(contingency_table);
    auto const& topic_weights = std::get<2>(contingency_table);
    double const& total_weight = static_cast<double>(std::get<3>(contingency_table));

    double const mi = compute_mutual_info(matches, gt_weights, topic_weights, total_weight);
    double const ex = entropy<>(gt_weights, total_weight), ey = entropy<>(topic_weights, total_weight);
    double const nmi = (ex <= 0 || ey <= 0) ? 1. : ((mi == 0) ? 0. : (mi / sqrt(ex * ey)));

    return nmi;
}

double expected_mutual_info(std::vector<uint32_t> const& gt_weights, std::vector<uint32_t> const& topic_weights, uint32_t const total_weight) {
//    auto const& n = matches;
    auto const& a = topic_weights;
    auto const& b = gt_weights;
    auto const& N = total_weight;

    auto const log_factorial = [](uint32_t val){return std::lgamma(val + 1);};

    std::vector<double> lg_b, lg_a, lg_Nb, lg_Na;
    double const lg_N = log_factorial(N);
    lg_a.reserve(topic_weights.size());
    lg_Na.reserve(topic_weights.size());
    lg_b.reserve(gt_weights.size());
    lg_Nb.reserve(gt_weights.size());
    for (auto const &topic_weight : topic_weights) {
        lg_a.push_back(log_factorial(topic_weight));
        lg_Na.push_back(log_factorial(N - topic_weight));
    }
    for (auto const &gt_weight : gt_weights) {
        lg_b.push_back(log_factorial(gt_weight));
        lg_Nb.push_back(log_factorial(N - gt_weight));
    }

    double emi = 0;
    for (auto i = 0; i < topic_weights.size(); ++i) {
        for (auto j = 0; j < gt_weights.size(); ++j) {
            uint32_t const min_n = std::max(1u, a[i] + b[j] - N);
            uint32_t const max_n = std::min(a[i], b[j]);
            for (auto n_ij = min_n; n_ij <= max_n; ++n_ij) {
                double const lg_n = log_factorial(n_ij);
                double const lg_an = log_factorial(a[i] - n_ij);
                double const lg_bn = log_factorial(b[j] - n_ij);
                double const lg_Nabn = log_factorial(N - a[i] - b[j] + n_ij);

                double const first_term = static_cast<double>(n_ij) * std::log(N * static_cast<double>(n_ij) / (a[i] * b[j])) / N;
                double const second_term = std::exp((lg_a[i] + lg_b[j] + lg_Na[i] + lg_Nb[j]) - (lg_N + lg_n + lg_an + lg_bn + lg_Nabn));
                emi += first_term * second_term;
            }
        }
    }
    return emi;
}

template<size_t pose_dimen = 4>
double ami(sunshine::Segmentation<std::vector<int>, 3, int, double> const &gt_seg,
           sunshine::Segmentation<std::vector<int>, pose_dimen, int, double> const &topic_seg) {
    auto const contingency_table = compute_matches<pose_dimen>(gt_seg, topic_seg);
    auto const& matches = std::get<0>(contingency_table);
    auto const& gt_weights = std::get<1>(contingency_table);
    auto const& topic_weights = std::get<2>(contingency_table);
    double const& total_weight = static_cast<double>(std::get<3>(contingency_table));

    double const mi = compute_mutual_info(matches, gt_weights, topic_weights, total_weight);
    double const emi = expected_mutual_info(gt_weights, topic_weights, total_weight);
    double const ex = entropy<>(gt_weights, total_weight), ey = entropy<>(topic_weights, total_weight);
    double const ami = (ex <= 0. || ey <= 0.) ? 1 : ((mi - emi) / (std::max(ex, ey) - emi));

//    double const compare = nmi<pose_dimen>(gt_seg, topic_seg);
//    double const compare_entropy = compute_average_entropy_y(matches, topic_weights, total_weight);
    return std::isnan(ami) ? 0. : ami;
}
}

#endif //SUNSHINE_PROJECT_BENCHMARK_HPP

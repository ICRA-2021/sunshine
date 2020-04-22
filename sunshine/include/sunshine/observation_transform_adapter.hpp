//
// Created by stewart on 4/22/20.
//

#ifndef SUNSHINE_PROJECT_OBSERVATION_TRANSFORM_ADAPTER_HPP
#define SUNSHINE_PROJECT_OBSERVATION_TRANSFORM_ADAPTER_HPP

#include "sunshine/common/observation_adapters.hpp"
#include <string>
#include <tf/tf.h>

namespace sunshine {

template <typename Type>
class ObservationTransformAdapter : public Adapter<ObservationTransformAdapter<Type>, Type, Type> {
    std::string target_frame;
    tf::Transformer transformer;

  public:
    template <typename ParamServer>
    explicit ObservationTransformAdapter(ParamServer* nh) {
        static_assert(Type::PoseDim == 3);
        target_frame = nh->template param<std::string>("world_frame", "map");
    }

    Type operator()(Type && in) const {
//        tf::StampedTransform transform;
//        transformer.lookupTransform(target_frame, in.frame, ros::Time(in.timestamp), transform);
        tf::Stamped<tf::Point> outputPoint;
        for (auto i = 0; i < in.observation_poses.size(); ++i) {
            // Using ros::Time(0) gets the latest transform
            tf::Stamped<tf::Point> inputPoint(tf::Point(in.observation_poses[i][0], in.observation_poses[i][1], in.observation_poses[i][2]), ros::Time(0), in.frame);
            transformer.transformPoint(target_frame, inputPoint, outputPoint);
            in.observation_poses[i][0] = outputPoint.x();
            in.observation_poses[i][1] = outputPoint.y();
            in.observation_poses[i][2] = outputPoint.z();
        }
        in.frame = target_frame;
        return in;
    }

    Type operator()(Type const& in) const {
        tf::StampedTransform transform;
        transformer.lookupTransform(target_frame, in.frame_id, ros::Time(in.timestamp), transform);
        tf::Stamped<tf::Point> outputPoint;
        Type out{in};
        for (auto i = 0; i < in.observation_poses.size(); ++i) {
            tf::Stamped<tf::Point> inputPoint(tf::Point(in.observation_poses[i][0], in.observation_poses[i][1], in.observation_poses[i][2]), ros::Time(in.timestamp), in.frame);
            transformer.transformPoint(target_frame, inputPoint, outputPoint);
            out.observation_poses[i][0] = outputPoint.x();
            out.observation_poses[i][1] = outputPoint.y();
            out.observation_poses[i][2] = outputPoint.z();
        }
        out.frame = target_frame;
        return out;
    }

    void addTransform(tf::StampedTransform transform) {
        transformer.setTransform(transform);
    }
};

}

#endif //SUNSHINE_PROJECT_OBSERVATION_TRANSFORM_ADAPTER_HPP

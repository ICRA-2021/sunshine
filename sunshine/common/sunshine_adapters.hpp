//
// Created by stewart on 3/4/20.
//

#ifndef SUNSHINE_PROJECT_SUNSHINE_ADAPTERS_HPP
#define SUNSHINE_PROJECT_SUNSHINE_ADAPTERS_HPP

#include <functional>
#include "sunshine_types.hpp"

namespace sunshine {

template<typename ImplClass, typename InputType, typename OutputType>
class Adapter {
  public:
    typedef InputType Input;
    typedef OutputType Output;

    virtual ~Adapter() = default;
//    virtual Output operator()(Input const &input) const = 0; // Not needed with CRTP

    friend Output operator>>(Input input, ImplClass adapter) {
        return adapter(input);
    }
};

template<typename ImplClass, typename FeatureType, uint32_t PoseDim, typename PoseType = double>
class FeatureExtractorAdapter : public Adapter<ImplClass, ImageObservation, CategoricalObservation<FeatureType, PoseDim, PoseType>> {
};

template<typename ImplClass, typename WordType, typename TopicType, uint32_t PoseDim, typename WordPoseType = double, typename TopicPoseType=WordPoseType>
class TopicModelAdapter : public Adapter<ImplClass, CategoricalObservation<WordType, PoseDim, WordPoseType>, CategoricalObservation<TopicType, PoseDim, TopicPoseType>> {
};

template <typename Input>
class LogAdapter : public Adapter<LogAdapter<Input>, Input, Input> {
    std::function<void(Input const&)> log_function;
  public:
    explicit LogAdapter(std::function<void(Input const&)> log_function) : log_function(log_function) {}
    Input operator()(Input const& input) const {
        log_function(input);
        return input;
    }
};

template <typename Input>
class UnaryOpAdapter : public Adapter<UnaryOpAdapter<Input>, Input, Input> {
  std::function<Input(Input)> unary_op;
  public:
    explicit UnaryOpAdapter(std::function<Input(Input)> unary_op) : unary_op(unary_op) {}
    Input operator()(Input input) {
        return unary_op(input);
    }
};

template<typename Init, class... Adapter>
auto process(Init i, Adapter... adapters) {
    return (i >> ... >> adapters);
}

}

#endif //SUNSHINE_PROJECT_SUNSHINE_ADAPTERS_HPP

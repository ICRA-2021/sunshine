//
// Created by stewart on 3/4/20.
//

#ifndef SUNSHINE_PROJECT_OBSERVATION_ADAPTERS_HPP
#define SUNSHINE_PROJECT_OBSERVATION_ADAPTERS_HPP

#include <functional>
#include "sunshine/common/observation_types.hpp"

namespace sunshine {

template<typename ImplClass, typename InputType, typename OutputType>
class Adapter {
  public:
    typedef InputType Input;
    typedef OutputType Output;

    virtual ~Adapter() = default;
//    virtual Output operator()(Input const &input) const = 0; // Not needed with CRTP

    friend auto operator>>(std::unique_ptr<Input>&& input, ImplClass& adapter) {
        return adapter(std::move(input));
    }

    friend auto operator>>(std::unique_ptr<Input> const& input, ImplClass& adapter) {
        return adapter(input.get());
    }

  public:
    auto operator()(std::unique_ptr<Input>&& input) {
        Input const* const inputPtr = input.get();
        return dynamic_cast<ImplClass*>(this)->operator()(inputPtr);
    }
};

template<typename Input>
class LogAdapter : public Adapter<LogAdapter<Input>, Input, Input> {
    std::function<void(Input const &)> log_function;
  public:
    explicit LogAdapter(std::function<void(Input const &)> log_function)
          : log_function(log_function) {}

    std::unique_ptr<Input> operator()(std::unique_ptr<Input>&& input) const {
        log_function(*input);
        return std::move(input);
    }
};

template<typename Init, class... Adapter>
auto process(Init i, Adapter... adapters) {
    return (i >> ... >> adapters);
}

}

#endif //SUNSHINE_PROJECT_OBSERVATION_ADAPTERS_HPP

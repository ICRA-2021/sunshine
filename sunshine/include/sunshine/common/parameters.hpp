//
// Created by stewart on 3/24/20.
//

#ifndef SUNSHINE_PROJECT_PARAMETERS_HPP
#define SUNSHINE_PROJECT_PARAMETERS_HPP

#include <map>
#include <variant>

namespace sunshine {

class Parameters {
    std::map<std::string, std::variant<std::string, int, double, bool>> parameters;
  public:
    explicit Parameters(decltype(parameters) params)
          : parameters(params) {}

    std::variant<std::string, int, double, bool>& operator[](std::string const& key) {
        return parameters[key];
    }

    template<typename T>
    T param(std::string const &key, T default_value) const {
        if (auto const &iter = parameters.find(key); iter != parameters.end()) {
            return std::get<T>(iter->second);
        } else { return default_value; }
    }
};

}

#endif //SUNSHINE_PROJECT_PARAMETERS_HPP

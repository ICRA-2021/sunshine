//
// Created by stewart on 3/25/20.
//

#ifndef SUNSHINE_PROJECT_OBJECT_STORES_HPP
#define SUNSHINE_PROJECT_OBJECT_STORES_HPP

namespace sunshine {
template<typename value_type, typename label_type = size_t>
class UniqueStore {
    std::map<value_type, label_type> items;

  public:
    UniqueStore() {
        static_assert(std::is_integral_v<label_type>, "Label type must be integral!");
    }

    label_type get_id(value_type const &item) {
        if (auto iter = items.find(item); iter != items.end()) {
            return iter->second;
        } else {
            items.emplace(item, items.size());
            if (items.size() - 1 > std::numeric_limits<label_type>::max()) throw std::runtime_error("UniqueStore ran out of labels");
            return items.size() - 1;
        }
    }

    label_type get_id(value_type &&item) {
        if (auto iter = items.find(item); iter != items.end()) {
            return iter->second;
        } else {
            items.emplace(std::move(item), items.size());
            if (items.size() - 1 > std::numeric_limits<label_type>::max()) throw std::runtime_error("UniqueStore ran out of labels");
            return items.size() - 1;
        }
    }

    label_type lookup_id(value_type const &item) const {
        if (auto iter = items.find(item); iter != items.end()) {
            return iter->second;
        } else {
            throw std::invalid_argument("Not found");
        }
    }

    size_t size() const {
        return items.size();
    }
};
}

#endif //SUNSHINE_PROJECT_OBJECT_STORES_HPP

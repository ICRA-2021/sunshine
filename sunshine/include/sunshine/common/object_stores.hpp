//
// Created by stewart on 3/25/20.
//

#ifndef SUNSHINE_PROJECT_OBJECT_STORES_HPP
#define SUNSHINE_PROJECT_OBJECT_STORES_HPP

namespace sunshine {
template<typename value_type>
class UniqueStore {
    std::map<value_type, size_t> items;

  public:
    size_t get_id(value_type const &item) {
        if (auto iter = items.find(item); iter != items.end()) {
            return iter->second;
        } else {
            items.emplace(item, items.size());
            return items.size() - 1;
        }
    }

    size_t get_id(value_type &&item) {
        if (auto iter = items.find(item); iter != items.end()) {
            return iter->second;
        } else {
            items.emplace(std::move(item), items.size());
            return items.size() - 1;
        }
    }

    size_t lookup_id(value_type const &item) const {
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

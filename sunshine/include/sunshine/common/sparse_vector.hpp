//
// Created by stewart on 9/15/20.
//

#ifndef SUNSHINE_PROJECT_SPARSE_VECTOR_H
#define SUNSHINE_PROJECT_SPARSE_VECTOR_H

#include <map>
#include <vector>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

namespace sunshine {
template<typename T, typename IndexT = size_t>
class sparse_vector {
    IndexT vec_size = 0;
    T default_value = T();
    std::map<IndexT, T> sparse;

  public:
    typedef T value_type;
    typedef IndexT size_type;

    explicit sparse_vector(IndexT n = 0, T default_value = T()) : vec_size(n), default_value(default_value) { }

    explicit sparse_vector(std::vector<T> const& v, T const default_value = T()) : vec_size(v.size()), default_value(default_value) {
        assert(v.size() < std::numeric_limits<IndexT>::max());
        for (IndexT i = 0; i < v.size(); ++i) {
            if (v[i] != default_value) sparse.insert(std::make_pair(i, v[i]));
        }
    }

    template<typename Iter>
    explicit sparse_vector(Iter first, Iter const last, T const default_value = T()) : default_value(default_value) {
        for (; first != last; first++) {
            auto const& val = *first;
            if (val != default_value) sparse.insert(std::make_pair(vec_size, val));
            vec_size++;
        }
    }

    explicit operator std::vector<T>() const {
        std::vector<T> out(vec_size, default_value);
        for (auto const& e : sparse) { out[e.first] = e.second; }
        return out;
    }

    auto map_begin() const {
        return sparse.cbegin();
    }

    auto map_end() const {
        return sparse.cend();
    }

    auto const& as_map() const {
        return sparse;
    }

    IndexT size() const {
        return vec_size;
    }

    template<typename V>
    V accumulate(V init = V()) {
        init += (vec_size - sparse.size()) * default_value;
        return std::accumulate(sparse.begin(), sparse.end(), init,
                               [](V const& running_total, auto const& next) { return running_total + next.second; });
    }

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & vec_size;
        ar & default_value;
        ar & sparse;
    }

    size_t bytesSize() const {
        auto const headerSize = sizeof(this->vec_size) + sizeof(this->default_value) + sizeof(sparse);
        auto const sparseSize = sparse.size() * (sizeof(T) + sizeof(IndexT)); // not exact but good enough
        return headerSize + sparseSize;
    }

    T& operator[](IndexT const& idx) {
        if (idx < 0 || idx >= vec_size) throw std::invalid_argument("Index out of range");
        auto iter = sparse.find(idx);
        if (iter == sparse.end()) {
            auto insertIter = sparse.insert(std::make_pair(idx, default_value));
            assert(insertIter.second);
        }
        return sparse.at(idx);
    }

    T operator[](IndexT const& idx) const {
        if (idx < 0 || idx >= vec_size) throw std::invalid_argument("Index out of range");
        auto const iter = sparse.find(idx);
        return (iter == sparse.end()) ? default_value : iter->second;
    }
};
} // namespace sunshine

#endif // SUNSHINE_PROJECT_SPARSE_VECTOR_H

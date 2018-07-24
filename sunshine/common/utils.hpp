#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>

template <typename IdxType, typename ValueType>
static cv::Mat toMat(std::vector<IdxType> const& idxes, std::vector<ValueType> const& values) {
    assert(idxes.size() == values.size() * 2);
    IdxType max_x = 0, max_y = 0;
    for (auto i = 0ul; i < values.size(); i++) {
        max_x = std::max(max_x, idxes[i*2]);
        max_y = std::max(max_y, idxes[i*2+1]);
    }
    cv::Mat mat(max_y + 1, max_x + 1, CV_32F);
    for (auto i = 0ul; i < values.size(); i++) {
        mat.at<float>(idxes[i*2+1], idxes[i*2]) = values[i];
    }
    return mat;
}

#endif // UTILS_HPP

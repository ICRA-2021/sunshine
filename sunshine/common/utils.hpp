#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/core.hpp>

template <typename ValType> struct cvType {};
template <> struct cvType<float> { static int const value = CV_32F; };
template <> struct cvType<double> { static int const value = CV_64F; };

/**
 * @brief toMat Converts two parallel vectors, of 2D poses (xy-coordinates) and 1D values, respectively, to a cv::Mat object.
 * @param idxes 2D poses of the form [x1,y1,...,xN,yN] (size of 2*N)
 * @param values Scalar values corresponding to each pose of the form [z1,...,zN] (size of N)
 * @return cv::Mat where mat.at<MatValueType>(yI, xI) = zI
 */
template <typename IdxType, typename ValueType, typename MatValueType = ValueType>
cv::Mat toMat(std::vector<IdxType> const& idxes, std::vector<ValueType> const& values) {
    assert(idxes.size() == values.size() * 2);

    // Compute the required size of the matrix based on the largest (x,y) coordinate
    IdxType max_x = 0, max_y = 0;
    for (auto i = 0ul; i < values.size(); i++) {
        assert(idxes[i*2] >= 0 && idxes[i*2+1] >= 0);
        max_x = std::max(max_x, idxes[i*2]);
        max_y = std::max(max_y, idxes[i*2+1]);
    }
    cv::Mat mat(max_y + 1, max_x + 1, cvType<MatValueType>::value); // Don't forget to add 1 to each dimension (because 0-indexed)

    for (auto i = 0ul; i < values.size(); i++) {
        mat.at<MatValueType>(idxes[i*2+1], idxes[i*2]) = values[i]; // mat.at(y,x) = z
    }
    return mat;
}

#endif // UTILS_HPP

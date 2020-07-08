#ifndef UTILS_HPP
#define UTILS_HPP

#include "colors.hpp"
#include <exception>
#include <array>
#include <chrono>
#include <opencv2/core.hpp>

namespace sunshine {

template <class...> constexpr std::false_type always_false{};

template<typename ValType>
struct cvType {
};
template<>
struct cvType<float> {
  static int const value = CV_32F;
};
template<>
struct cvType<double> {
  static int const value = CV_64F;
};
template<>
struct cvType<uint8_t> {
  static int const value = CV_8U;
};
template<>
struct cvType<uint16_t> {
  static int const value = CV_16U;
};
template<>
struct cvType<cv::Vec4b> {
  static int const value = CV_8UC4;
};
template<>
struct cvType<cv::Vec3b> {
  static int const value = CV_8UC3;
};

/**
 * @brief toMat Converts two parallel vectors, of 3D integer xyz-coordinates and 1D values, respectively, to a dense matrix representation.
 * @param idxes 3D integer poses of the form [x1,y1,z1,...,xN,yN,zN] (size of 2*N)
 * @param values Scalar values corresponding to each pose of the form [f1,...,fN] (size of N)
 * @return cv::Mat where mat.at<MatValueType>(yI, xI) = fI
 */
template<typename IdxType, typename ValueType, typename MatValueType = ValueType>
cv::Mat toMat(std::vector<IdxType> const &idxes, std::vector<ValueType> const &values) {
    assert(idxes.size() == (values.size() * 3) || idxes.size() == (values.size() * 2));
    auto const poseDim = static_cast<int>(values.size() / idxes.size());

    // Compute the required size of the matrix based on the largest (x,y) coordinate
    IdxType max_x = 0, max_y = 0;
    for (auto i = 0ul; i < values.size(); i++) {
        assert(idxes[i * poseDim] >= 0 && idxes[i * poseDim + 1] >= 0);
        max_x = std::max(max_x, idxes[i * poseDim]);
        max_y = std::max(max_y, idxes[i * poseDim + 1]);
    }
    cv::Mat mat(max_y + 1, max_x + 1, cvType<MatValueType>::value); // Don't forget to add 1 to each dimension (because 0-indexed)

    for (auto i = 0ul; i < values.size(); i++) {
        mat.at<MatValueType>(idxes[i * poseDim + 1], idxes[i * poseDim]) = values[i]; // mat.at(y,x) = z
    }
    return mat;
}

template<int COUNT, char DELIM = 'x', typename WordDimType = double>
static inline std::array<WordDimType, COUNT> readNumbers(std::string const &str) {
    std::array<WordDimType, COUNT> nums = {0};
    size_t idx = 0;
    for (size_t i = 1; i <= COUNT; i++) {
        auto const next = (i < COUNT)
                          ? str.find(DELIM, idx)
                          : str.size();
        if (next == std::string::npos) {
            throw std::invalid_argument("String '" + str + "' contains too few numbers!");
        }
        nums[i - 1] = std::stod(str.substr(idx, next));
        idx = next + 1;
    }
    return nums;
}

template<int POSE_DIM, typename WordDimType = double>
static inline std::array<WordDimType, POSE_DIM> computeCellSize(double cell_size_time, double cell_size_space) {
    std::array<WordDimType, POSE_DIM> cell_size = {};
    cell_size[0] = cell_size_time;
    for (size_t i = 1; i < POSE_DIM; i++) {
        cell_size[i] = cell_size_space;
    }
    return cell_size;
}

namespace _make_array {
template<std::size_t... Indices>
struct indices {
  using next = indices<Indices..., sizeof...(Indices)>;
};
template<std::size_t N>
struct build_indices {
  using type = typename build_indices<N - 1>::type::next;
};

template<>
struct build_indices<0> {
  using type = indices<>;
};
template<std::size_t N> using BuildIndices = typename build_indices<N>::type;

template<std::size_t... I, typename Iter, typename ValueType=typename Iter::value_type, typename Array = std::array<ValueType, sizeof...(I)>>
Array make_array(Iter first, indices<I...>) {
    return Array{{first[I]...}};
}
}

template<std::size_t N, typename Iter, typename ValueType=typename Iter::value_type>
std::array<ValueType, N> make_array(Iter start) {
    using namespace _make_array;
    return make_array(start, BuildIndices<N>{});
}

template<typename T>
long record_lap(T &time_checkpoint) {
    auto const duration = std::chrono::steady_clock::now() - time_checkpoint;
    time_checkpoint = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

template <typename T>
std::vector<std::vector<T>> identity_mat(size_t const dimen) {
    std::vector<std::vector<T>> eye(dimen, std::vector<T>(dimen, 0));
    for (auto i = 0; i < dimen; ++i) eye[i][i] = 1;
    return eye;
}

}

#endif // UTILS_HPP

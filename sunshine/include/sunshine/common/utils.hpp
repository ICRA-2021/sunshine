#ifndef UTILS_HPP
#define UTILS_HPP

#include "colors.hpp"
#include <exception>
#include <numeric>
#include <array>
#include <chrono>
#include <opencv2/core.hpp>
#include <boost/functional/hash.hpp>
#include <ostream>

namespace sunshine {

template <class...> constexpr std::false_type always_false{};

template<typename T>
struct is_vector : public std::false_type { };

template<typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type { };

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
inline std::array<WordDimType, COUNT> readNumbers(std::string const &str) {
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
inline std::array<WordDimType, POSE_DIM> computeCellSize(double cell_size_time, double cell_size_space) {
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
inline Array make_array(Iter first, indices<I...>) {
    return Array{{first[I]...}};
}
}

template<std::size_t N, typename Iter, typename ValueType=typename Iter::value_type>
inline std::array<ValueType, N> make_array(Iter start) {
    using namespace _make_array;
    return make_array(start, BuildIndices<N>{});
}


template<typename T, int N>
struct hasharray {
    std::size_t operator()(std::array<T, N> const& arr) const {
        return boost::hash_range(arr.cbegin(), arr.cend());
    }
};

template<typename T>
inline long record_lap(T &time_checkpoint) {
    auto const new_time = std::chrono::steady_clock::now();
    auto const duration = new_time - time_checkpoint;
    time_checkpoint = new_time;
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

template <typename T>
std::vector<std::vector<T>> identity_mat(size_t const dimen) {
    std::vector<std::vector<T>> eye(dimen, std::vector<T>(dimen, 0));
    for (auto i = 0; i < dimen; ++i) eye[i][i] = 1;
    return eye;
}

//template<typename Container>
//uint32_t argmax(Container const &container) {
//    uint32_t idx_max = 0;
//    auto max = container[0];
//    for (auto i = 1; i < container.size(); ++i) {
//        if (container[i] > max) {
//            idx_max = i;
//            max = container[i];
//        }
//    }
//    return idx_max;
//}

template<typename T>
inline typename T::value_type argmax(T const& array) {
    return std::max_element(array.begin(), array.end()) - array.begin();
}

template <typename T>
inline std::vector<T> make_vector(T val) {
    std::vector<T> vec;
    vec.emplace_back(std::move(val));
    return vec;
}

template<typename T, typename C = T>
inline C normalize(T const& array) {
    static_assert(std::is_floating_point_v<typename C::value_type>);
    C out;
    typename T::value_type const sum = std::accumulate(array.begin(), array.end(), typename T::value_type(0));
    std::transform(array.begin(), array.end(), std::back_inserter(out),
                   [sum](typename T::value_type const& v){return static_cast<typename C::value_type>(v) / sum;});
    return out;
}

template<typename T, typename Ret = typename T::value_type>
inline Ret dotp(T const& left, T const& right) {
    return std::inner_product(left.begin(), left.end(), right.begin(), Ret(0));
}

std::vector<std::string> split(const std::string &txt, char ch = ' ')
{
    size_t pos = txt.find( ch );
    size_t initialPos = 0;
    std::vector<std::string> strs = {};

    // Decompose statement
    while( pos != std::string::npos ) {
        strs.push_back( txt.substr( initialPos, pos - initialPos ) );
        initialPos = pos + 1;
        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    strs.push_back( txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ) );
    return strs;
}

[[nodiscard]] std::string replace_all(std::string str, const std::string& old_str, const std::string& new_str) {
    size_t pos = 0;
    while ((pos = str.find(old_str, pos)) != std::string::npos) {
        str.replace(pos, old_str.length(), new_str);
        pos += new_str.length();
    }
    return str;
}

template<typename To>
To safeNumericCast(uintmax_t val) {
    if (std::is_same_v<To, double> && val > (1ull << 53u)) throw std::logic_error(std::to_string(val) + " cannot be safely cast");
    if (std::is_same_v<To, float> && val > (1ull << 24u)) throw std::logic_error(std::to_string(val) + " cannot be safely cast");
    if (val > std::numeric_limits<To>::max()) throw std::logic_error(std::to_string(val) + " overflows target type");
    if (val < std::numeric_limits<To>::min()) throw std::logic_error(std::to_string(val) + " underflows target type");
    return static_cast<To>(val);
}

template<typename To>
To safeNumericCast(intmax_t val) {
    if (std::is_same_v<To, double> && val > (1ull << 53u)) throw std::logic_error(std::to_string(val) + " cannot be safely cast");
    if (std::is_same_v<To, float> && val > (1ull << 24u)) throw std::logic_error(std::to_string(val) + " cannot be safely cast");
    if (val > std::numeric_limits<To>::max()) throw std::logic_error(std::to_string(val) + " overflows target type");
    if (val < std::numeric_limits<To>::min()) throw std::logic_error(std::to_string(val) + " underflows target type");
    return static_cast<To>(val);
}

template<typename To>
To safeNumericCast(double val) {
    if (val >= (1ull << 53u)) throw std::logic_error(std::to_string(val) + " cannot be safely cast");
    if (std::is_same_v<To, float> && val > (1ull << 24u)) throw std::logic_error(std::to_string(val) + " cannot be safely cast");
    if (val > std::numeric_limits<To>::max()) throw std::logic_error(std::to_string(val) + " overflows target type");
    if (val < std::numeric_limits<To>::min()) throw std::logic_error(std::to_string(val) + " underflows target type");
    if constexpr (std::is_integral_v<To>) return static_cast<To>(std::round(val));
    else return static_cast<To>(val);
}

template<typename T, size_t N>
std::ostream& operator<<(std::ostream& stream, std::array<T, N> const& arr) {
    stream << "[";
    bool first = true;
    for (auto const& a : arr) {
        if (!first) stream << ",";
        if constexpr (std::is_same_v<T, std::string>) stream << "\"";
        stream << a;
        if constexpr (std::is_same_v<T, std::string>) stream << "\"";
        first = false;
    }
    return (stream << "]");
}

template<typename T>
bool includes(std::vector<T> parent, std::vector<T> child) {
    std::sort(parent.begin(), parent.end());
    std::sort(child.begin(), child.end());
    return std::includes(parent.begin(), parent.end(), child.begin(), child.end());
}

template<typename T>
bool includes(std::set<T> const& parent, std::set<T> const& child) {
    return std::includes(parent.begin(), parent.end(), child.begin(), child.end());
}

template<typename T>
bool includes(std::set<T> const& parent, std::vector<T> child) {
    std::sort(child.begin(), child.end());
    return std::includes(parent.begin(), parent.end(), child.begin(), child.end());
}

template<typename T, typename V, typename H, typename Iterable>
bool includes(std::unordered_map<T, V, H> const& parent, Iterable const& child) {
    for (auto const& e : child) {
        if (parent.find(e) == parent.end()) {
            return false;
        }
    }
    return true;
}

template<typename T, typename V>
bool includes(std::map<T, V> const& parent, std::vector<T> const& child) {
    return includes(std::set<T>(parent.begin(), parent.end()), child);
}

}

#endif // UTILS_HPP

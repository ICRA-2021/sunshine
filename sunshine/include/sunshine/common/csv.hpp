//
// Created by stewart on 2/7/20.
//

#ifndef SUNSHINE_PROJECT_CSV_HPP
#define SUNSHINE_PROJECT_CSV_HPP

#include <string>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <exception>
#include <mutex>

template<char delimiter = ',', char str_delimiter = '"'>
class csv_row {
    std::string data;
    std::vector<size_t> delimiters = {};

    template<char delim>
    static constexpr bool check_delim(char i) noexcept {
        return delim == i;
    }

    template<char delim = delimiter>
    static size_t get_next_delimiter(std::string const &str, size_t idx) {
        bool in_str = false;
        for (; idx < str.size(); ++idx) {
            in_str = in_str ^ check_delim<str_delimiter>(str[idx]);
            if (!in_str && check_delim<delim>(str[idx])) return idx;
        }
        return str.size();
    }

    static std::vector<size_t> get_all_delimiters(std::string const &str) {
        std::vector<size_t> out(1, 0);
        if (str.empty()) return out;
        size_t idx = 0;
        do {
            idx = get_next_delimiter(str, idx);
            out.push_back(idx);
        } while (idx < str.size());
        assert(idx == str.size());
        return out;
    }

    static std::string escape(std::string const &str) {
        return str_delimiter + str + str_delimiter;
    }

  public:
    explicit csv_row(const std::string &data = "")
          : data(data)
          , delimiters(get_all_delimiters(data)) {
        static_assert(!check_delim<str_delimiter>(delimiter), "Entry delimiter must be different than string delimiter!");
    }

    size_t size() const {
        return delimiters.size() - 1;
    }

    std::string get(size_t n) const {
        if (n >= size()) throw std::out_of_range("Index out of range of row");
        return data.substr(delimiters[n], delimiters[n + 1] - delimiters[n]);
    }

    std::string const &str() const {
        return data;
    }

    void append(std::string const &value) {
        if (get_next_delimiter<str_delimiter>(value, 0) != value.size()) throw std::invalid_argument("String contains unexpected escapes");
        data += ((data.empty())
                 ? escape(value)
                 : (delimiter + escape(value)));
        delimiters.push_back(data.size());
    }

    void append(const char *const &value) {
        append(std::string(value));
    }

    void append(char const &value) {
        if (check_delim<str_delimiter>(value)) throw std::invalid_argument("String contains unexpected escapes");
        if (check_delim<delimiter>(value)) {
            data += ((data.empty())
                     ? escape(std::string(1, value))
                     : (delimiter + escape(std::string(1, value))));
        } else {
            data += (data.empty())
                    ? std::string(1, value)
                    : (delimiter + std::string(1, value));
        }
        delimiters.push_back(data.size());
    }

    template<typename T>
    void append(std::vector<T> const &values) {
        std::stringstream ss;
        ss << "[ ";
        size_t i = 0;
        for (auto const &value : values) {
            if (i++ > 0) ss << ", ";
            ss << value;
        }
        ss << " ]";
        append(ss.str());
    }

    template<typename T, std::enable_if_t<std::is_scalar<T>::value, int> = 0>
    void append(T const &value) {
        data += (data.empty())
                ? std::to_string(value)
                : (delimiter + std::to_string(value));
        delimiters.push_back(data.size());
    }
};

template<char delimiter = ',', char str_delimiter = '"'>
class csv_writer {
    std::ostream *out;
    bool _constructed = false;
    bool _has_header = false;
    std::mutex write_lock;

  public:
    typedef csv_row<delimiter, str_delimiter> Row;

    explicit csv_writer(const std::string &path)
          : out(new std::ofstream(path))
          , _constructed(true) {
        if (!out->good()) throw std::invalid_argument("Failed to open " + path + " for writing.");
    }

    explicit csv_writer(std::ostream *out)
          : out(out)
          , _constructed(false) {
        if (!out->good()) throw std::invalid_argument("Cannot write to bad stream.");
    }

    void write_row(csv_row<delimiter, str_delimiter> const &row) {
        std::lock_guard<std::mutex> guard(write_lock);
        (*out) << row.str() << '\n';
    }

    csv_row<delimiter, str_delimiter> new_row() const {
        return csv_row<delimiter, str_delimiter>();
    }

    void write_row(std::string const &row) {
        std::lock_guard<std::mutex> guard(write_lock);
        (*out) << row << '\n';
    }

    template<typename T>
    void write_header(T const &header) {
        _has_header = true;
        write_row(header);
    }

    bool has_header() const {
        return _has_header;
    }

    void flush() {
        std::lock_guard<std::mutex> guard(write_lock);
        out->flush();
    }

    void close() {
        std::lock_guard<std::mutex> guard(write_lock);
        if (_constructed) {
            auto *ofstream_ptr = dynamic_cast<std::ofstream *>(out);
            ofstream_ptr->close();
        }
    }

    ~csv_writer() {
        if (_constructed) close();
        else flush();
    }
};

template<char delimiter = ',', char str_delimiter = '"'>
class csv_reader {
    std::string const path;
    std::ifstream in;
    std::mutex read_lock;

    std::string next_line = "";
    bool eof = false;

  public:
    explicit csv_reader(const std::string &path)
          : path(path)
          , in(path)
          , eof(!std::getline(in, next_line)) {
        if (!in.good()) throw std::invalid_argument("Failed to open " + path + " for reading.");
    }

    bool has_next() const {
        std::lock_guard<std::mutex> guard(read_lock);
        return !eof;
    }

    csv_row<delimiter, str_delimiter> next_row() {
        std::lock_guard<std::mutex> guard(read_lock);
        auto row = csv_row<delimiter, str_delimiter>(next_line);
        if (has_next()) eof = !std::getline(in, next_line);
        return row;
    }

    void close() {
        std::lock_guard<std::mutex> guard(read_lock);
        in.close();
    }

    ~csv_reader() {
        close();
    }
};

#endif //SUNSHINE_PROJECT_CSV_HPP

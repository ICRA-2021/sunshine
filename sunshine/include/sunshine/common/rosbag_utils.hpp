//
// Created by stewart on 2020-07-23.
//

#ifndef SUNSHINE_PROJECT_ROSBAG_UTILS_HPP
#define SUNSHINE_PROJECT_ROSBAG_UTILS_HPP

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <string>
#include <map>
#include <memory>

namespace sunshine {

class BagIterator
{
    rosbag::Bag bag;
    std::unique_ptr<rosbag::View> view;
    std::unique_ptr<rosbag::View::const_iterator> iterator;
    std::multimap<std::string, std::function<bool(rosbag::MessageInstance const &)>> callbacks;
    bool logging = false;
    decltype(std::chrono::steady_clock::now()) clock = std::chrono::steady_clock::now();

    void open(std::string const& filename) {
        if (bag.isOpen()) bag.close();
        if (!filename.empty()) {
            bag.open(filename, rosbag::bagmode::Read);
            if (!bag.isOpen()) throw std::invalid_argument("Failed to read bagfile " + filename);
        }
    }

  public:
    explicit BagIterator(const std::string &filename) {
        open(filename);
    }

    void set_logging(bool enable = true) {
        logging = enable;
    }

    /**
     *
     * @param topic Name of the topic to assign the callback to
     * @param callback Function that returns true if the playback should break
     */
    void add_callback(std::string const &topic, std::function<bool(rosbag::MessageInstance const&)> callback) {
        callbacks.emplace(std::make_pair(topic, std::move(callback)));
    }

    /**
     *
     * @tparam MessageClass Type that message should be instantiated as
     * @param topic Name of the topic to assign the callback to
     * @param callback Function that returns true if the playback should break
     */
    template<class MessageClass>
    void add_callback(std::string const &topic, std::function<bool(boost::shared_ptr<MessageClass>)> callback) {
        callbacks.emplace(std::make_pair(topic, [callback](rosbag::MessageInstance const &msg) {
            return callback(msg.instantiate<MessageClass>());
        }));
    }

    /**
     *
     * @param ignore_breakpoints
     * @return true if finished playing bag up to a maximum of max_msgs, false otherwise
     */
    bool play(bool ignore_breakpoints = false) {
//        ROS_INFO("%ld ms since last played", record_lap(clock));
        if (!view) view = std::make_unique<rosbag::View>(bag, [this](rosbag::ConnectionInfo const* ci){
                       bool const matches = callbacks.find(ci->topic) != callbacks.end();
                       ROS_INFO_COND(!matches && logging, "Skipped message from topic %s", ci->topic.c_str());
                       return matches;
            });
        if (!iterator) iterator = std::make_unique<rosbag::View::const_iterator>(view->begin());
        bool flagged = false;
        for (; *iterator != view->end() && (ignore_breakpoints || !flagged); (*iterator)++) {
            auto const dt = record_lap(clock);
            auto const &m = **iterator;
//            ROS_INFO("parsing %s; %ld ms since last msg parsed", m.getTopic().c_str(), dt);
            if (auto callback_range = callbacks.equal_range(m.getTopic()); callback_range.first != callbacks.end()) {
//                ROS_INFO("%ld ms spent finding callback", record_lap(clock));
                for (auto callback = callback_range.first; callback != callback_range.second; ++callback) {
                    flagged = (callback->second(m)) || flagged;
                }
                auto const dt2 = record_lap(clock);
//                ROS_INFO("%ld ms spent in callbacks", dt2);
            }
        }
//        ROS_INFO("%ld ms since last callback finished", record_lap(clock));
        return !flagged;
    }

    void restart() {
        iterator.reset();
    }

    bool finished() const {
        return view && iterator && *iterator == view->end();
    }

    ~BagIterator() {
        if (bag.isOpen()) bag.close();
    }
};

}

#endif //SUNSHINE_PROJECT_ROSBAG_UTILS_HPP

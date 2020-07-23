//
// Created by stewart on 2020-07-23.
//

#ifndef SUNSHINE_PROJECT_ROSBAG_UTILS_HPP
#define SUNSHINE_PROJECT_ROSBAG_UTILS_HPP

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <string>
#include <unordered_map>
#include <memory>

namespace sunshine {

class BagIterator
{
    rosbag::Bag bag;
    std::unique_ptr<rosbag::View> view;
    std::unique_ptr<rosbag::View::const_iterator> iterator;
    std::unordered_multimap<std::string, std::function<bool(rosbag::MessageInstance const&)>> callbacks;
    bool logging = false;

  public:
    explicit BagIterator(const std::string& filename) {
        bag.open(filename, rosbag::bagmode::Read);
        if (!bag.isOpen()) throw std::invalid_argument("Failed to read bagfile " + filename);
    }

    void set_logging(bool enable = true) {
        logging = enable;
    }

    /**
     *
     * @tparam Callback Callback class (implied by function argument>
     * @tparam MessageClass Type that message should be instantiated as
     * @param topic Name of the topic to assign the callback to
     * @param callback Function that returns true if the playback should break
     */
    template<class MessageClass = rosbag::MessageInstance, typename Callback>
    void add_callback(std::string const& topic, Callback const& callback) {
        callbacks.emplace(std::make_pair(topic, [&callback](rosbag::MessageInstance const& msg){
            if constexpr (std::is_same_v<MessageClass, rosbag::MessageInstance>) return callback(msg);
            else return callback(msg.instantiate<MessageClass>());
        }));
    }

    /**
     *
     * @param ignore_breakpoints
     * @return true if finished playing bag up to a maximum of max_msgs, false otherwise
     */
    bool play(bool ignore_breakpoints = false) {
        if (!view) view = std::make_unique<rosbag::View>(bag);
        if (!iterator) iterator = std::make_unique<rosbag::View::const_iterator>(view->begin());
        for (;*iterator != view->end(); (*iterator)++) {
            auto const& m = **iterator;
            if (auto callback_range = callbacks.equal_range(m.getTopic()); callback_range.first != callbacks.end()) {
                bool flag = false;
                for (auto callback = callback_range.first; callback != callback_range.second; ++callback) flag = flag || callback->second(m);
                if (flag && !ignore_breakpoints) return false;
            } else {
                ROS_INFO_COND(logging, "Skipped message from topic %s", m.getTopic().c_str());
                add_callback(m.getTopic(), [](rosbag::MessageInstance const& msg){return false;});
            }
        }
        return true;
    }

    void restart() {
        iterator.reset();
    }

    ~BagIterator() {
        bag.close();
    }
};

}

#endif //SUNSHINE_PROJECT_ROSBAG_UTILS_HPP

#ifndef WORD_COLORING_HPP
#define WORD_COLORING_HPP

#include "utils.hpp"
#include <cstdlib>

namespace sunshine {

template <typename WordType>
class WordColorMap {
    std::map<WordType, double> hueMapBackward;
    std::map<double, WordType> hueMapForward;

public:
    inline RGBA colorForWord(WordType word, double saturation = 1, double value = 1, double alpha = 1)
    {
        auto const hueIter = hueMapBackward.find(word);
        if (hueIter != hueMapBackward.end()) {
            return HSV_TO_RGBA({ hueIter->second, saturation, value }, alpha);
        }

        double hue = double(rand()) * 360. / double(RAND_MAX);
        if (hueMapForward.size() == 1) {
            hue = fmod(hueMapForward.begin()->first + 180., 360.);
        } else if (hueMapForward.size() > 1) {
            auto const& upper = (hueMapForward.upper_bound(hue) == hueMapForward.end())
                ? hueMapForward.lower_bound(0)->first + 360.
                : hueMapForward.upper_bound(hue)->first;
            auto const& lower = (hueMapForward.lower_bound(hue) == hueMapForward.end())
                ? hueMapForward.crbegin()->first
                : (--hueMapForward.lower_bound(hue))->first;
            hue = fmod((lower + upper) / 2., 360.);
        }
        hueMapForward.insert({ hue, word });
        hueMapBackward.insert({ word, hue });

        return sunshine::HSV_TO_RGBA({ hue, saturation, value }, alpha);
    }

    std::map<WordType, std::array<uint8_t, 3>> getAllColors(double saturation = 1, double value = 1)
    {
        std::map<WordType, std::array<uint8_t, 3>> colorMap;
        for (auto const& entry : hueMapBackward) {
            RGBA const colorVal = sunshine::HSV_TO_RGBA({entry.second, saturation, value});
            colorMap.insert({entry.first, {colorVal.r, colorVal.g, colorVal.b}});
        }
        return colorMap;
    }

    inline size_t getNumColors() const {
        return hueMapForward.size();
    }
};
}

#endif // WORD_COLORING_HPP

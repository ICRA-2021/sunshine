#ifndef WORD_COLORING_HPP
#define WORD_COLORING_HPP

#include "sunshine/common/utils.hpp"
#include <cstdlib>
#include <map>

namespace sunshine {

template <typename WordType>
class WordColorMap {
    std::map<WordType, double> hueMapBackward;
    std::map<double, WordType> hueMapForward;

    inline double hueForWord(WordType word) {
        static_assert(std::is_integral_v<WordType>);
        assert(word >= WordType(0));
        double num = 0;
        double den = 1;
        for (WordType i = 0; i != word && i < std::numeric_limits<WordType>::max(); ++i) {
            num += 2;
            if (num >= den) {
                num = 1;
                den *= 2;
            }
        }
        return (num * 360.) / den;
    }

public:
    inline RGBA colorForWord(WordType word, double saturation = 1, double value = 1, double alpha = 1)
    {
        auto const hueIter = hueMapBackward.find(word);
        if (hueIter != hueMapBackward.end()) {
            return HSV_TO_RGBA({ hueIter->second, saturation, value }, alpha);
        }

        double const hue = hueForWord(word);
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

    [[nodiscard]] inline size_t getNumColors() const {
        return hueMapForward.size();
    }
};
}

#endif // WORD_COLORING_HPP

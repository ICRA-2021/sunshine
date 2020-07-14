#ifndef COLORS_HPP
#define COLORS_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sunshine {

struct HSV {
  double h;
  double s;
  double v;
};

struct RGBA;
struct ARGB;

struct RGBA {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;

  explicit operator std::array<uint8_t, 4>() const {
      return {r, g, b, a};
  }

  inline explicit operator ARGB() const;
};

struct ARGB {
  uint8_t a;
  uint8_t r;
  uint8_t g;
  uint8_t b;

  explicit operator std::array<uint8_t, 4>() const {
      return {a, r, g, b};
  }

  inline explicit operator RGBA() const;
};

inline RGBA::operator ARGB() const {
    return {a, r, g, b};
}

inline ARGB::operator RGBA() const {
    return {r, g, b, a};
}

RGBA inline HSV_TO_RGBA(HSV const &hsv, double alpha = 1) {
    double const hueNorm = std::fmod(hsv.h, 360) / 60.;
    auto const hueSectant = static_cast<uint32_t>(std::floor(hueNorm));
    double const hueOffset = hueNorm - hueSectant;

    double const &V = hsv.v, &S = hsv.s;
    auto const v = uint8_t(255. * V);
    auto const p = uint8_t(255. * V * (1 - S));
    auto const q = uint8_t(255. * V * (1 - S * hueOffset));
    auto const t = uint8_t(255. * V * (1 - S * (1 - hueOffset)));
    auto const a = uint8_t(255. * alpha);

    switch (hueSectant) {
        case 0:
            return {v, t, p, a};
        case 1:
            return {q, v, p, a};
        case 2:
            return {p, v, t, a};
        case 3:
            return {p, q, v, a};
        case 4:
            return {t, p, v, a};
        case 5:
            return {v, p, q, a};
        default:
            throw std::invalid_argument("Should be unreachable.");
    }
}
}

#endif // COLORS_HPP

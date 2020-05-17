#pragma once

namespace motis {

struct projection {
  projection(float const low, float const high,
             float const scale_factor = 100.0,  // percent
             float const input_low = 0.0, float const input_high = 1.0)
      : low_{low},
        high_{high},
        scale_factor_{scale_factor},
        input_low_{input_low},
        input_high_{input_high} {}

  int operator()(float const in) const {
    auto const normalized = (in - input_low_) / (input_high_ - input_low_);
    return static_cast<int>(scale_factor_ *
                            (low_ + normalized * (high_ - low_)));
  }

  float low_, high_;
  float scale_factor_;
  float input_low_, input_high_;
};

}  // namespace motis
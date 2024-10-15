#pragma once

#include "cista/reflection/comparable.h"

namespace motis::paxforecast {

template <typename Type>
struct fixed_distribution {
  CISTA_COMPARABLE()
  using result_type = Type;

  struct param_type {
    CISTA_COMPARABLE()
    using distribution_type = fixed_distribution;
    result_type value_{};
  };

  void reset() {}
  param_type param() const { return {value_}; }
  void param(param_type const& p) { value_ = p.value_; }
  result_type min() const { return value_; }
  result_type max() const { return value_; }

  template <typename Gen>
  result_type operator()(Gen& gen) {
    return value_;
  }

  template <typename Gen>
  result_type operator()(Gen& gen, param_type const& p) {
    return p.value_;
  }

  result_type value_{};
};

}  // namespace motis::paxforecast

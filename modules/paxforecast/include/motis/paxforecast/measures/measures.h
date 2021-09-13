#pragma once

#include <algorithm>
#include <vector>

#include "motis/paxforecast/measures/announcements.h"

#include "motis/paxforecast/measures/trip_load_information.h"
#include "motis/paxforecast/measures/trip_recommendation.h"

namespace motis::paxforecast::measures {

struct measures {
  void clear() {
    recommendations_.clear();
    load_infos_.clear();
  }

  void add(measures const& o) {
    auto const copy = [](auto const& from, auto& to) {
      to.reserve(to.size() + from.size());
      std::copy(begin(from), end(from), std::back_inserter(to));
    };

    copy(o.recommendations_, recommendations_);
    copy(o.load_infos_, load_infos_);
  }

  std::vector<trip_recommendation> recommendations_;
  std::vector<trip_load_information> load_infos_;
};

}  // namespace motis::paxforecast::measures

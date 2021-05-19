#pragma once

#include <bitset>

#include "motis/hash_map.h"
#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/common/flat_matrix.h"
#include "motis/core/schedule/time.h"

namespace motis {

struct waiting_time_rules {
  int waiting_time_category(mcd::string const& train_category) const {
    auto it = category_map_.find(train_category);
    if (it == end(category_map_)) {
      return default_group_;
    } else {
      return it->second;
    }
  }

  inline int waiting_time_category(int family) const {
    if (static_cast<std::size_t>(family) < family_to_wtr_category_.size()) {
      return family_to_wtr_category_[family];
    } else {
      return default_group_;
    }
  }

  inline int waiting_time(int connecting_category, int feeder_category) const {
    return waiting_time_matrix_[connecting_category][feeder_category];
  }

  inline int waiting_time_family(int connecting_family,
                                 int feeder_family) const {
    return waiting_time_matrix_[waiting_time_category(connecting_family)]
                               [waiting_time_category(feeder_family)];
  }

  inline bool waits_for_other_trains(int connecting_category) const {
    return waits_for_other_trains_[connecting_category];
  }

  inline bool other_trains_wait_for(int feeder_category) const {
    return other_trains_wait_for_[feeder_category];
  }

  int default_group_{0};
  mcd::hash_map<mcd::string, int> category_map_;
  mcd::vector<int> family_to_wtr_category_;
  flat_matrix<uint32_t> waiting_time_matrix_;
  mcd::vector<bool> waits_for_other_trains_;
  mcd::vector<bool> other_trains_wait_for_;
};

}  // namespace motis

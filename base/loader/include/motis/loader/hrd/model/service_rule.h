#pragma once

#include <cinttypes>
#include <map>
#include <utility>
#include <vector>

#include "motis/schedule/bitfield.h"
#include "motis/loader/hrd/model/hrd_service.h"

namespace motis::loader::hrd {

struct resolved_rule_info {
  bitfield traffic_days_;
  int eva_num_1_{0}, eva_num_2_{0};
  uint8_t type_{0};
  bool day_switch_{false};
  unsigned s1_traffic_days_offset_{0};
  unsigned s2_traffic_days_offset_{0};
};

using service_combination =
    std::tuple<hrd_service*, hrd_service*, resolved_rule_info>;

struct service_rule {
  service_rule(service_rule const&) = default;
  service_rule(service_rule&&) = default;
  service_rule& operator=(service_rule const&) = delete;
  service_rule& operator=(service_rule&&) = delete;

  explicit service_rule(bitfield const& mask) : mask_(mask) {}
  virtual ~service_rule() = default;
  virtual int applies(hrd_service const&) const = 0;
  virtual void add(hrd_service*, int info) = 0;
  virtual std::vector<service_combination> service_combinations() const = 0;
  virtual resolved_rule_info rule_info() const = 0;

  bitfield const& mask_;
};

using service_id = std::pair<int, uint64_t>;  // (train_num, admin)
using service_rules =
    std::map<service_id, std::vector<std::shared_ptr<service_rule>>>;

}  // namespace motis::loader::hrd

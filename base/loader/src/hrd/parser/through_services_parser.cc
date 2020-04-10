#include "motis/loader/hrd/parser/through_services_parser.h"

#include <optional>

#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"

#include "motis/core/common/logging.h"
#include "motis/loader/util.h"
#include "motis/schedule-format/RuleService_generated.h"

using namespace utl;
using namespace motis::logging;

namespace motis::loader::hrd {

struct ts_rule : public service_rule {
  ts_rule(service_id id_1, service_id id_2, int eva_num, bitfield const& mask)
      : service_rule(mask),
        id_1_(std::move(id_1)),
        id_2_(std::move(id_2)),
        eva_num_(eva_num) {}

  ~ts_rule() override = default;

  ts_rule(ts_rule const&) = delete;
  ts_rule(ts_rule&&) = delete;
  ts_rule& operator=(ts_rule const&) = delete;
  ts_rule& operator=(ts_rule&&) = delete;

  int applies(hrd_service const& s) const override {
    // Check for non-empty intersection.

    if ((s.traffic_days_at_stop(s.stops_.size() - 1, event_type::ARR) & mask_)
            .any()) {
      // Assuming s is service (1): Check last stop.
      auto last_stop = s.stops_.back();
      auto last_section = s.sections_.back();
      if (last_stop.eva_num_ == eva_num_ &&
          last_section.train_num_ == id_1_.first &&
          raw_to_int<uint64_t>(last_section.admin_) == id_1_.second) {
        return 1;
      }
    }

    // Assuming s is service (2).
    for (auto section_idx = 0UL; section_idx < s.sections_.size();
         ++section_idx) {
      auto from_stop = s.stops_[section_idx];
      auto section = s.sections_[section_idx];

      if (from_stop.eva_num_ == eva_num_ && section.train_num_ == id_2_.first &&
          raw_to_int<uint64_t>(section.admin_) == id_2_.second &&
          (s.traffic_days_at_stop(section_idx, event_type::DEP) & mask_)
              .any()) {
        return 2;
      }
    }

    // No match.
    return 0;
  }

  void add(hrd_service* s, int info) override {
    if (info == 1) {
      participants_1_.push_back(s);
    } else {
      participants_2_.push_back(s);
    }
  }

  std::vector<service_combination> service_combinations() const override {
    std::vector<service_combination> comb;
    for (auto const& s1 : participants_1_) {
      auto const s1_stop_index = static_cast<int>(s1->stops_.size()) - 1;
      auto const s1_traffic_days =
          s1->traffic_days_at_stop(s1_stop_index, event_type::ARR);

      std::optional<service_combination> combination;
      auto min_time_diff = 1440;

      for (auto const& s2 : participants_2_) {
        auto const s2_stop_index = s2->first_stop_index_at(eva_num_);
        auto const s2_traffic_days =
            s2->traffic_days_at_stop(s2_stop_index, event_type::DEP);

        auto const arr_time =
            s1->event_time(s1_stop_index, event_type::ARR) % 1440;
        auto const dep_time =
            s2->event_time(s2_stop_index, event_type::DEP) % 1440;
        auto const day_switch = dep_time < arr_time;
        auto const time_diff = (dep_time + (day_switch ? 1440 : 0)) - arr_time;

        if (time_diff > min_time_diff) {
          continue;
        }

        auto const intersection =
            day_switch ? (s1_traffic_days << 1) & s2_traffic_days & mask_
                       : s1_traffic_days & s2_traffic_days & mask_;

        if (intersection.any()) {
          combination = {s1, s2,
                         resolved_rule_info{intersection, eva_num_, eva_num_,
                                            RuleType_THROUGH, day_switch}};
          min_time_diff = time_diff;
        }
      }

      if (combination) {
        comb.push_back(combination.value());
      }
    }

    return comb;
  }

  resolved_rule_info rule_info() const override {
    return resolved_rule_info{mask_, eva_num_, eva_num_, RuleType_THROUGH};
  }

  service_id id_1_, id_2_;
  int eva_num_;
  std::vector<hrd_service*> participants_1_;
  std::vector<hrd_service*> participants_2_;
};

void parse_through_service_rules(loaded_file const& file,
                                 std::map<int, bitfield> const& hrd_bitfields,
                                 service_rules& rules, config const& c) {
  scoped_timer timer("parsing through trains");
  for_each_line_numbered(file.content(), [&](cstr line, int line_number) {
    if (line.len < 40) {
      return;
    }

    auto it = hrd_bitfields.find(parse<int>(line.substr(c.th_s_.bitfield_)));

    utl::verify(it != std::end(hrd_bitfields), "missing bitfield: {}:{}",
                file.name(), line_number);

    auto key_1 =
        std::make_pair(parse<int>(line.substr(c.th_s_.key1_nr_)),
                       raw_to_int<uint64_t>(line.substr(c.th_s_.key1_admin_)));
    auto key_2 =
        std::make_pair(parse<int>(line.substr(c.th_s_.key2_nr_)),
                       raw_to_int<uint64_t>(line.substr(c.th_s_.key2_admin_)));
    std::shared_ptr<service_rule> rule(new ts_rule(
        key_1, key_2, parse<int>(line.substr(c.th_s_.eva_)), it->second));

    rules[key_1].push_back(rule);
    rules[key_2].push_back(rule);
  });
}

}  // namespace motis::loader::hrd

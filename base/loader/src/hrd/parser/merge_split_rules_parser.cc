#include "motis/loader/hrd/parser/merge_split_rules_parser.h"

#include <set>
#include <vector>

#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"

#include "motis/core/common/logging.h"
#include "motis/loader/util.h"
#include "motis/schedule-format/RuleService_generated.h"

using namespace utl;
using namespace motis::logging;

namespace motis::loader::hrd {

struct mss_rule : public service_rule {
  mss_rule(service_id id_1, service_id id_2, int eva_num_begin, int eva_num_end,
           bitfield const& mask)
      : service_rule(mask),
        id_1_(std::move(id_1)),
        id_2_(std::move(id_2)),
        eva_num_begin_(eva_num_begin),
        eva_num_end_(eva_num_end) {}

  mss_rule(mss_rule const&) = delete;
  mss_rule(mss_rule&&) = delete;
  mss_rule& operator=(mss_rule const&) = delete;
  mss_rule& operator=(mss_rule&&) = delete;

  ~mss_rule() override = default;

  int applies(hrd_service const& s) const override {
    // Check for non-empty intersection.
    try {
      auto const stop_idx = s.find_first_stop_at(eva_num_begin_);
      if (stop_idx == hrd_service::NOT_SET) {
        return 0;
      }
      if ((s.traffic_days_at_stop(stop_idx, event_type::DEP) & mask_).none()) {
        return 0;
      }
    } catch (std::runtime_error&) {
      return 0;
    }

    // Check if first and last stop of the common part are contained with the
    // correct service id.
    bool begin_found = false, end_found = false;
    for (unsigned section_idx = 0;
         section_idx < s.sections_.size() && !(begin_found && end_found);
         ++section_idx) {
      auto const& section = s.sections_[section_idx];
      auto const& from_stop = s.stops_[section_idx];
      auto const& to_stop = s.stops_[section_idx + 1];
      auto service_id = std::make_pair(section.train_num_,
                                       raw_to_int<uint64_t>(section.admin_));

      if (service_id != id_1_ && service_id != id_2_) {
        continue;
      }
      if (!end_found && from_stop.eva_num_ == eva_num_begin_) {
        begin_found = true;
      }
      if (begin_found && to_stop.eva_num_ == eva_num_end_) {
        end_found = true;
      }
    }
    return static_cast<int>(begin_found && end_found);
  }

  void add(hrd_service* s, int /* info */) override {
    participants_.push_back(s);
  }

  static std::pair<int /* event time */, size_t /* stop index */>
  get_event_time(hrd_service const* s, int eva_num,
                 hrd_service::event hrd_service::stop::*ev) {
    auto stop_it = std::find_if(
        begin(s->stops_), end(s->stops_),
        [&](hrd_service::stop const& st) { return st.eva_num_ == eva_num; });
    utl::verify(stop_it != end(s->stops_), "merge/split stop not found");
    return std::make_pair(((*stop_it).*ev).time_,
                          std::distance(begin(s->stops_), stop_it));
  }

  static bool all_ms_events_exist(hrd_service const* s1, hrd_service const* s2,
                                  int mss_begin, int mss_end) {
    auto const [merge_time_s1, merge_idx_s1] =
        get_event_time(s1, mss_begin, &hrd_service::stop::dep_);
    auto const [merge_time_s2, merge_idx_s2] =
        get_event_time(s2, mss_begin, &hrd_service::stop::dep_);

    if (merge_time_s1 % 1440 != merge_time_s2 % 1440) {
      return false;
    }

    auto const day_offset_s1 = merge_time_s1 / 1440;
    auto const day_offset_s2 = merge_time_s2 / 1440;
    auto const time_offset_s1 = day_offset_s1 * 1440;
    auto const time_offset_s2 = day_offset_s2 * 1440;

    auto const [split_time_s1, split_idx_s1] =
        get_event_time(s1, mss_end, &hrd_service::stop::arr_);
    auto const [split_time_s2, split_idx_s2] =
        get_event_time(s2, mss_end, &hrd_service::stop::arr_);

    if (split_time_s1 - time_offset_s1 != split_time_s2 - time_offset_s2 ||
        split_idx_s1 - merge_idx_s1 != split_idx_s2 - merge_idx_s2) {
      return false;
    }

    // ensure that all stops between the merge and split match
    int const stop_count = split_idx_s1 - merge_idx_s1 + 1;
    for (int i = 0; i < stop_count; ++i) {
      auto const& stop_s1 = s1->stops_[merge_idx_s1 + i];
      auto const& stop_s2 = s2->stops_[merge_idx_s2 + i];
      if (stop_s1.eva_num_ != stop_s2.eva_num_ ||
          (i != 0 && (stop_s1.arr_.time_ - time_offset_s1) !=
                         (stop_s2.arr_.time_ - time_offset_s2)) ||
          (i != stop_count - 1 && (stop_s1.dep_.time_ - time_offset_s1) !=
                                      (stop_s2.dep_.time_ - time_offset_s2))) {
        return false;
      }
    }
    return true;
  }

  std::vector<service_combination> service_combinations() const override {
    std::vector<service_combination> unordered_pairs;
    std::set<std::pair<hrd_service*, hrd_service*>> combinations;
    for (auto s1 : participants_) {
      auto const s1_traffic_days = s1->traffic_days_at_stop(
          s1->get_first_stop_index_at(eva_num_begin_), event_type::DEP);
      for (auto s2 : participants_) {
        if (s1 == s2 ||
            combinations.find(std::make_pair(s2, s1)) != end(combinations)) {
          continue;
        }
        combinations.emplace(s1, s2);

        auto const s2_traffic_days = s2->traffic_days_at_stop(
            s2->get_first_stop_index_at(eva_num_begin_), event_type::DEP);
        auto const intersection = s1_traffic_days & s2_traffic_days & mask_;
        if (intersection.any() &&
            all_ms_events_exist(s1, s2, eva_num_begin_, eva_num_end_)) {
          unordered_pairs.emplace_back(
              s1, s2,
              resolved_rule_info{intersection, eva_num_begin_, eva_num_end_,
                                 RuleType_MERGE_SPLIT});
        }
      }
    }
    return unordered_pairs;
  }

  resolved_rule_info rule_info() const override {
    return resolved_rule_info{mask_, eva_num_begin_, eva_num_end_,
                              RuleType_MERGE_SPLIT};
  }

  service_id id_1_, id_2_;
  int eva_num_begin_, eva_num_end_;
  std::vector<hrd_service*> participants_;
};

void parse_merge_split_service_rules(
    loaded_file const& file, std::map<int, bitfield> const& hrd_bitfields,
    service_rules& rules, config const& c) {
  scoped_timer const timer("parsing merge split rules");

  for_each_line_numbered(file.content(), [&](cstr line, int line_number) {
    if (line.len < c.merge_spl_.line_length_) {
      return;
    }

    auto const bitfield_idx =
        c.merge_spl_.bitfield_.from == std::numeric_limits<size_t>::max()
            ? 0
            : parse<int>(line.substr(c.merge_spl_.bitfield_));
    auto it = hrd_bitfields.find(bitfield_idx);
    utl::verify(it != std::end(hrd_bitfields), "missing bitfield: {}:{}",
                file.name(), line_number);

    auto key_1 = std::make_pair(
        parse<int>(line.substr(c.merge_spl_.key1_nr_)),
        raw_to_int<uint64_t>(line.substr(c.merge_spl_.key1_admin_).trim()));
    auto key_2 = std::make_pair(
        parse<int>(line.substr(c.merge_spl_.key2_nr_)),
        raw_to_int<uint64_t>(line.substr(c.merge_spl_.key2_admin_).trim()));

    auto eva_num_begin = parse<int>(line.substr(c.merge_spl_.eva_begin_));
    auto eva_num_end = parse<int>(line.substr(c.merge_spl_.eva_end_));
    std::shared_ptr<service_rule> const rule(
        new mss_rule(key_1, key_2, eva_num_begin, eva_num_end, it->second));

    rules[key_1].push_back(rule);
    rules[key_2].push_back(rule);
  });
}

}  // namespace motis::loader::hrd

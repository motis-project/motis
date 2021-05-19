#pragma once

#include <vector>

#include "utl/parser/cstr.h"

#include "motis/schedule/bitfield.h"
#include "motis/loader/hrd/model/specification.h"
#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

struct parser_info {
  char const* filename_;
  int line_number_from_;
  int line_number_to_;
};

enum class event_type : uint8_t { DEP, ARR };

struct hrd_service {
  static const constexpr auto NOT_SET = -1;  // NOLINT

  struct event {
    int time_;
    bool in_out_allowed_;
  };

  struct stop {
    int eva_num_;
    event arr_, dep_;
  };

  struct attribute {
    attribute(int bitfield_num, utl::cstr code)
        : bitfield_num_(bitfield_num), code_(code) {}

    int bitfield_num_;
    utl::cstr code_;

    friend bool operator==(attribute const& lhs, attribute const& rhs) {
      return lhs.bitfield_num_ == rhs.bitfield_num_ && lhs.code_ == rhs.code_;
    }
  };

  enum direction_type { EVA_NUMBER, DIRECTION_CODE };
  struct section {
    section() = default;
    section(int train_num, utl::cstr admin)
        : train_num_(train_num), admin_(admin) {}

    int train_num_{0};
    utl::cstr admin_;
    std::vector<attribute> attributes_;
    std::vector<utl::cstr> category_;
    std::vector<utl::cstr> line_information_;
    std::vector<std::pair<uint64_t, int>> directions_;
    std::vector<int> traffic_days_;
  };

  hrd_service(parser_info origin, int num_repetitions, int interval,
              std::vector<stop> stops, std::vector<section> sections,
              bitfield traffic_days, int initial_train_num)
      : origin_(origin),
        num_repetitions_(num_repetitions),
        interval_(interval),
        stops_(std::move(stops)),
        sections_(std::move(sections)),
        traffic_days_(traffic_days),
        initial_train_num_(initial_train_num) {}

  hrd_service(specification const& spec, config const&);

  void verify_service();

  std::vector<std::pair<int, uint64_t>> get_ids() const {
    std::vector<std::pair<int, uint64_t>> ids;

    // Add first service id.
    auto const& first_section = sections_.front();
    ids.emplace_back(std::make_pair(
        first_section.train_num_, raw_to_int<uint64_t>(first_section.admin_)));

    // Add new service id if it changed.
    for (size_t i = 1; i < sections_.size(); ++i) {
      auto id = std::make_pair(sections_[i].train_num_,
                               raw_to_int<uint64_t>(sections_[i].admin_));
      if (id != ids.back()) {
        ids.emplace_back(id);
      }
    }

    return ids;
  }

  inline int find_first_stop_at(int eva_num) const {
    for (auto i = 0UL; i < stops_.size(); ++i) {
      if (stops_[i].eva_num_ == eva_num) {
        return i;
      }
    }
    return NOT_SET;
  }

  inline int get_first_stop_index_at(int eva_num) const {
    auto const idx = find_first_stop_at(eva_num);
    utl::verify(idx != NOT_SET,
                "hrd_service::get_first_stop_index_at: stop not found");
    return idx;
  }

  inline int event_time(int stop_index, event_type evt) const {
    return evt == event_type::DEP ? stops_[stop_index].dep_.time_
                                  : stops_[stop_index].arr_.time_;
  }

  inline unsigned traffic_days_offset_at_stop(int stop_index,
                                              event_type evt) const {
    return static_cast<unsigned>(event_time(stop_index, evt) / 1440);
  }

  inline bitfield traffic_days_at_stop(int stop_index, event_type evt) const {
    return traffic_days_ << traffic_days_offset_at_stop(stop_index, evt);
  }

  parser_info origin_;
  int num_repetitions_;
  int interval_;
  std::vector<stop> stops_;
  std::vector<section> sections_;
  bitfield traffic_days_;
  int initial_train_num_;
};

}  // namespace motis::loader::hrd

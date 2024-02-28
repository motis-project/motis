#include "motis/paxmon/api/get_check_data.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "utl/enumerate.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/string.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::paxmon::eval::forecast;

using namespace flatbuffers;

namespace motis::paxmon::api {

namespace {

struct section_data {
  std::vector<pax_check_entry const*> check_entries_;
  std::uint16_t checks_{};
  std::uint16_t checkins_{};
  std::uint16_t possible_additional_pax_{};  // from CHECKED_DEVIATION_NO_MATCH
};

Offset<Station> optional_station(message_creator& mc, schedule const& sched,
                                 std::uint32_t const station_id) {
  return station_id != 0 ? to_fbs(mc, *sched.stations_.at(station_id)) : 0;
};

unixtime optional_time(schedule const& sched, time const t) {
  return t == INVALID_TIME ? 0 : motis_to_unixtime(sched, t);
};

}  // namespace

Offset<PaxMonCheckEntry> check_entry_to_fbs(message_creator& mc,
                                            schedule const& sched,
                                            pax_check_entry const& entry) {
  return CreatePaxMonCheckEntry(
      mc, entry.ref_, mc.CreateString(entry.order_id_.view()),
      mc.CreateString(entry.trip_id_.view()), entry.passengers_,
      static_cast<PaxMonCheckType>(entry.check_type_), entry.check_count_,
      static_cast<PaxMonCheckLegStatus>(entry.leg_status_),
      static_cast<PaxMonCheckDirection>(entry.direction_), entry.planned_train_,
      entry.checked_in_train_, entry.canceled_,
      optional_station(mc, sched, entry.leg_start_station_),
      optional_station(mc, sched, entry.leg_destination_station_),
      optional_time(sched, entry.leg_start_time_),
      optional_time(sched, entry.leg_destination_time_),
      optional_station(mc, sched, entry.checkin_start_station_),
      optional_station(mc, sched, entry.checkin_destination_station_),
      optional_time(sched, entry.check_min_time_),
      optional_time(sched, entry.check_max_time_),
      optional_time(sched, entry.schedule_train_start_time_),
      mc.CreateString(entry.category_.view()), entry.train_nr_,
      entry.planned_trip_ref_);
}

msg_ptr get_check_data(paxmon_data& data, schedule const& sched,
                       msg_ptr const& msg) {
  auto const req = motis_content(PaxMonCheckDataRequest, msg);

  auto const trp = from_fbs(sched, req->trip_id());
  auto const trp_sections = access::sections{trp};
  utl::verify(trp_sections.size() != 0, "trip has no active sections");

  auto const trp_id_start_time = trp->id_.primary_.get_time();
  auto const trp_id_destination_time = trp->id_.secondary_.target_time_;

  auto const first_trp_section = *trp_sections.begin();
  auto const current_first_departure = first_trp_section.lcon().d_time_;
  auto const schedule_first_departure =
      get_schedule_time(sched, first_trp_section.ev_key_from());
  auto const min_first_departure = std::min(
      {current_first_departure, schedule_first_departure, trp_id_start_time});

  auto const last_trp_section = *std::prev(trp_sections.end());
  auto const current_last_arrival = last_trp_section.lcon().a_time_;
  auto const schedule_last_arrival =
      get_schedule_time(sched, last_trp_section.ev_key_to());
  auto const max_last_arrival = std::max(
      {current_last_arrival, schedule_last_arrival, trp_id_destination_time});

  auto sec_data = std::vector<section_data>(trp_sections.size());

  message_creator mc;
  auto fbs_entries = std::vector<Offset<PaxMonCheckEntry>>{};
  auto matched_entry_count = 0U;
  auto unmatched_entry_count = 0U;

  auto const is_matching_entry = [&](pax_check_entry const& entry) {
    if (entry.schedule_train_start_time_ == schedule_first_departure ||
        entry.schedule_train_start_time_ == trp_id_start_time) {
      return true;
    }
    if (entry.all_checks_between(min_first_departure, max_last_arrival) ||
        entry.leg_between(min_first_departure, max_last_arrival)) {
      return true;
    }
    return false;
  };

  auto const check_and_add_matching_section =
      [&](pax_check_entry const& entry, access::trip_section const& sec,
          std::size_t sec_idx) {
        auto const schedule_dep = get_schedule_time(sched, sec.ev_key_from());
        auto const schedule_arr = get_schedule_time(sched, sec.ev_key_to());
        auto current_dep = sec.lcon().d_time_;
        auto const current_arr = sec.lcon().a_time_;

        auto const in_leg = entry.in_leg(schedule_dep, schedule_arr);
        auto const checked =
            entry.maybe_checked_between(current_dep, current_arr);
        auto const checked_in_section =
            entry.definitely_checked_between(current_dep, current_arr);

        auto& sd = sec_data.at(sec_idx);
        if (in_leg || checked) {
          sd.check_entries_.emplace_back(&entry);
          if (checked_in_section) {
            if ((entry.check_type_ == check_type::TICKED_CHECKED ||
                 entry.check_type_ == check_type::BOTH)) {
              ++sd.checks_;
            }
            if ((entry.check_type_ == check_type::CHECKIN ||
                 entry.check_type_ == check_type::BOTH)) {
              ++sd.checkins_;
            }
          }
        } else if (!entry.has_leg_info() &&
                   entry.leg_status_ ==
                       leg_status::CHECKED_DEVIATION_NO_MATCH) {
          ++sd.possible_additional_pax_;
        }
      };

  auto con_info = first_trp_section.lcon().full_con_->con_info_;
  auto train_nr = con_info->train_nr_;
  auto category = sched.categories_.at(con_info->family_)->name_;

  while (con_info != nullptr && matched_entry_count == 0) {
    if (auto const map_entry = data.pax_check_data_.trains_.find(
            train_pax_data_key{category, train_nr});
        map_entry != end(data.pax_check_data_.trains_)) {
      for (auto const& entry : map_entry->second.entries_) {
        if (is_matching_entry(entry)) {
          fbs_entries.emplace_back(check_entry_to_fbs(mc, sched, entry));
          ++matched_entry_count;

          for (auto const& [sec_idx, sec] : utl::enumerate(trp_sections)) {
            check_and_add_matching_section(entry, sec, sec_idx);
          }
        } else {
          ++unmatched_entry_count;
        }
      }
    }
    if (matched_entry_count == 0) {
      con_info = con_info->merged_with_;
      if (con_info != nullptr) {
        train_nr = con_info->train_nr_;
        category = sched.categories_.at(con_info->family_)->name_;
      }
    }
  }

  auto fbs_sections = std::vector<Offset<PaxMonCheckSectionData>>{};
  fbs_sections.reserve(trp_sections.size());
  for (auto const& [sec_idx, sec] : utl::enumerate(trp_sections)) {
    auto const& sd = sec_data.at(sec_idx);
    auto total_group_count = 0U;
    auto total_pax_count = 0U;
    auto checked_group_count = 0U;
    auto checked_pax_count = 0U;
    auto unchecked_but_covered_group_count = 0U;
    auto unchecked_but_covered_pax_count = 0U;
    auto unchecked_uncovered_group_count = 0U;
    auto unchecked_uncovered_pax_count = 0U;

    for (auto const* ce : sd.check_entries_) {
      ++total_group_count;
      total_pax_count += ce->passengers_;
      if (ce->check_type_ != check_type::NOT_CHECKED) {
        ++checked_group_count;
        checked_pax_count += ce->passengers_;
      } else {
        if (ce->leg_status_ == leg_status::NOT_CHECKED_COVERED) {
          ++unchecked_but_covered_group_count;
          unchecked_but_covered_pax_count += ce->passengers_;
        } else if (ce->leg_status_ == leg_status::NOT_CHECKED_NOT_COVERED) {
          ++unchecked_uncovered_group_count;
          unchecked_uncovered_pax_count += ce->passengers_;
        }
      }
    }

    auto const min_pax_count = checked_pax_count;
    auto const max_pax_count = checked_pax_count +
                               unchecked_uncovered_pax_count +
                               sd.possible_additional_pax_;
    auto const avg_pax_count = (min_pax_count + max_pax_count) / 2;

    fbs_sections.emplace_back(CreatePaxMonCheckSectionData(
        mc, to_fbs(mc, sec.from_station(sched)),
        to_fbs(mc, sec.to_station(sched)),
        motis_to_unixtime(sched, get_schedule_time(sched, sec.ev_key_from())),
        motis_to_unixtime(sched, sec.lcon().d_time_),
        motis_to_unixtime(sched, get_schedule_time(sched, sec.ev_key_to())),
        motis_to_unixtime(sched, sec.lcon().a_time_),
        mc.CreateVector(utl::to_vec(sd.check_entries_,
                                    [](auto const& ce) { return ce->ref_; })),
        total_group_count, total_pax_count, checked_group_count,
        checked_pax_count, unchecked_but_covered_group_count,
        unchecked_but_covered_pax_count, unchecked_uncovered_group_count,
        unchecked_uncovered_pax_count, sd.possible_additional_pax_,
        min_pax_count, avg_pax_count, max_pax_count, sd.checks_, sd.checkins_));
  }

  mc.create_and_finish(
      MsgContent_PaxMonCheckDataResponse,
      CreatePaxMonCheckDataResponse(
          mc, mc.CreateString(category.view()), train_nr, matched_entry_count,
          unmatched_entry_count, mc.CreateVector(fbs_entries),
          mc.CreateVector(fbs_sections))
          .Union());
  return make_msg(mc);
}

msg_ptr get_check_data_by_order(paxmon_data& data, schedule const& sched,
                                msg_ptr const& msg) {
  auto const req = motis_content(PaxMonCheckDataByOrderRequest, msg);
  auto const order_id = mcd::string{req->order_id()->view()};

  message_creator mc;
  auto fbs_entries =
      utl::to_vec(data.pax_check_data_.entries_by_order_id_.at(order_id),
                  [&](pax_check_entry const* entry) {
                    return check_entry_to_fbs(mc, sched, *entry);
                  });
  mc.create_and_finish(
      MsgContent_PaxMonCheckDataByOrderResponse,
      CreatePaxMonCheckDataByOrderResponse(mc, mc.CreateVector(fbs_entries))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api

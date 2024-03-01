#include "motis/paxmon/api/revise_compact_journey.h"

#include <cstdint>
#include <algorithm>
#include <iterator>
#include <optional>

#include "utl/to_vec.h"

#include "motis/core/common/interval_map.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/journeys_to_message.h"

#include "motis/routing/output/to_journey.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace flatbuffers;

namespace mro = motis::routing::output;
namespace mroi = motis::routing::output::intermediate;

namespace motis::paxmon::api {

struct event_info {
  time cur_time_{};
  time sched_time_{};
  std::uint16_t cur_track_{};
  std::uint16_t sched_track_{};
  timestamp_reason reason_{timestamp_reason::SCHEDULE};

  [[nodiscard]] journey::stop::event_info to_journey(
      schedule const& sched, bool const valid = true) const {
    return {.valid_ = valid,
            .timestamp_ = motis_to_unixtime(sched, cur_time_),
            .schedule_timestamp_ = motis_to_unixtime(sched, sched_time_),
            .timestamp_reason_ = reason_,
            .track_ = sched.tracks_.at(cur_track_).str(),
            .schedule_track_ = sched.tracks_.at(sched_track_).str()};
  }
};

struct trip_section_info {
  light_connection const* lcon_{};
  station const& from_;
  station const& to_;
  event_info dep_{};
  event_info arr_{};
};

struct used_trip_sections {
  std::vector<trip_section_info> sections_;
  std::optional<std::size_t> enter_section_;
  std::optional<std::size_t> exit_section_;
  bool exact_enter_section_{};
  bool exact_exit_section_{};
};

used_trip_sections get_trip_sections(schedule const& sched,
                                     journey_leg const& leg, trip const* trp) {
  auto result = used_trip_sections{};
  result.sections_ = utl::to_vec(access::sections{trp}, [&](auto const& sec) {
    auto const& lc = sec.lcon();
    auto const ek_dep = sec.ev_key_from();
    auto const ek_arr = sec.ev_key_to();
    auto const di_dep = get_delay_info(sched, ek_dep);
    auto const di_arr = get_delay_info(sched, ek_arr);
    return trip_section_info{
        .lcon_ = &lc,
        .from_ = sec.from_station(sched),
        .to_ = sec.to_station(sched),
        .dep_ = event_info{.cur_time_ = lc.d_time_,
                           .sched_time_ = di_dep.get_schedule_time(),
                           .cur_track_ = lc.full_con_->d_track_,
                           .sched_track_ = get_schedule_track(sched, ek_dep),
                           .reason_ = di_dep.get_reason()},
        .arr_ = event_info{.cur_time_ = lc.a_time_,
                           .sched_time_ = di_arr.get_schedule_time(),
                           .cur_track_ = lc.full_con_->a_track_,
                           .sched_track_ = get_schedule_track(sched, ek_arr),
                           .reason_ = di_arr.get_reason()}};
  });

  auto enter_section_it =
      std::find_if(begin(result.sections_), end(result.sections_),
                   [&](trip_section_info const& sec) {
                     return sec.from_.index_ == leg.enter_station_id_ &&
                            sec.dep_.sched_time_ == leg.enter_time_;
                   });
  result.exact_enter_section_ = enter_section_it != end(result.sections_);

  auto exit_section_it =
      std::find_if(begin(result.sections_), end(result.sections_),
                   [&](trip_section_info const& sec) {
                     return sec.to_.index_ == leg.exit_station_id_ &&
                            sec.arr_.sched_time_ == leg.exit_time_;
                   });
  result.exact_exit_section_ = exit_section_it != end(result.sections_);

  if (!result.exact_enter_section_) {
    // no exact match, see if a stop with a different scheduled time at the
    // enter station exists (possible reroute)
    // if multiple stops at the same station exist, choose the first one
    // after the scheduled departure time, or the last if none are after
    // the scheduled departure time
    for (auto it = begin(result.sections_); it != end(result.sections_); ++it) {
      if (it->from_.index_ == leg.enter_station_id_) {
        enter_section_it = it;
        if (it->dep_.sched_time_ >= leg.enter_time_) {
          break;
        }
      }
    }
  }

  if (!result.exact_exit_section_) {
    // same logic
    for (auto it = begin(result.sections_); it != end(result.sections_); ++it) {
      if (it->to_.index_ == leg.exit_station_id_) {
        exit_section_it = it;
        if (enter_section_it != end(result.sections_)) {
          if (it->arr_.sched_time_ >= enter_section_it->dep_.sched_time_) {
            break;
          }
        } else if (it->arr_.sched_time_ >= leg.exit_time_) {
          break;
        }
      }
    }
  }

  if (enter_section_it != end(result.sections_)) {
    result.enter_section_ =
        std::distance(begin(result.sections_), enter_section_it);
  }

  if (exit_section_it != end(result.sections_)) {
    result.exit_section_ =
        std::distance(begin(result.sections_), exit_section_it);
  }

  return result;
}

journey revise_compact_journey(schedule const& sched,
                               compact_journey const& cj) {
  auto j = journey{};
  auto last_time = unixtime{};

  auto const push_station = [&](station const& st) -> journey::stop& {
    if (j.stops_.empty() || j.stops_.back().eva_no_ != st.eva_nr_) {
      return j.stops_.emplace_back(journey::stop{.exit_ = false,
                                                 .enter_ = false,
                                                 .name_ = st.name_.str(),
                                                 .eva_no_ = st.eva_nr_.str(),
                                                 .lat_ = st.lat(),
                                                 .lng_ = st.lng()});
    } else {
      return j.stops_.back();
    }
  };

  auto itransports = std::vector<mroi::transport>{};

  for (auto const& leg : cj.legs()) {
    auto const& enter_station = *sched.stations_.at(leg.enter_station_id_);
    auto const& exit_station = *sched.stations_.at(leg.exit_station_id_);

    if (!j.stops_.empty() && j.stops_.back().eva_no_ != enter_station.eva_nr_) {
      if (leg.enter_transfer_ &&
          leg.enter_transfer_->type_ == transfer_info::type::FOOTPATH) {
        auto const prev_idx = static_cast<unsigned>(j.stops_.size() - 1);
        itransports.emplace_back(prev_idx, prev_idx + 1,
                                 leg.enter_transfer_->duration_, 0, 0, 0);
      }
    }

    auto const* trp = get_trip(sched, leg.trip_idx_);
    auto const sections = get_trip_sections(sched, leg, trp);

    if (!sections.sections_.empty() && sections.enter_section_ &&
        sections.exit_section_ &&
        *sections.enter_section_ <= *sections.exit_section_) {
      // enter + exit found -> add all trip stops
      auto const& enter_sec = sections.sections_.at(*sections.enter_section_);
      auto& enter_stop = push_station(enter_sec.from_);
      enter_stop.enter_ = true;

      for (auto i = *sections.enter_section_; i <= *sections.exit_section_;
           ++i) {
        auto const& sec = sections.sections_[i];
        auto& dep_stop = j.stops_.back();
        dep_stop.departure_ = sec.dep_.to_journey(sched);
        auto& arr_stop = push_station(sec.to_);
        arr_stop.arrival_ = sec.arr_.to_journey(sched);
        itransports.emplace_back(static_cast<unsigned>(j.stops_.size() - 2),
                                 static_cast<unsigned>(j.stops_.size() - 1),
                                 sec.lcon_);
      }

      auto& exit_stop = j.stops_.back();
      exit_stop.exit_ = true;
      last_time = exit_stop.arrival_.schedule_timestamp_;
    } else {
      // enter and/or exit not found -> add planned enter/exit stops
      auto& enter_stop = push_station(enter_station);
      auto const enter_idx = static_cast<unsigned>(j.stops_.size() - 1);
      enter_stop.enter_ = true;
      auto& dep = enter_stop.departure_;
      dep.valid_ = sections.exact_enter_section_;
      dep.schedule_timestamp_ = motis_to_unixtime(sched, leg.enter_time_);
      dep.timestamp_ = dep.schedule_timestamp_;
      if (sections.enter_section_) {
        enter_stop.departure_ =
            sections.sections_[*sections.enter_section_].dep_.to_journey(sched);
      }

      auto& exit_stop = push_station(exit_station);
      auto const exit_idx = static_cast<unsigned>(j.stops_.size() - 1);
      exit_stop.exit_ = true;
      auto& arr = exit_stop.arrival_;
      arr.valid_ = sections.exact_exit_section_;
      arr.schedule_timestamp_ = motis_to_unixtime(sched, leg.exit_time_);
      arr.timestamp_ = arr.schedule_timestamp_;
      if (sections.exit_section_) {
        exit_stop.arrival_ =
            sections.sections_[*sections.exit_section_].arr_.to_journey(sched);
      }
      last_time = arr.schedule_timestamp_;

      if (sections.enter_section_) {
        itransports.emplace_back(
            enter_idx, exit_idx,
            sections.sections_.at(*sections.enter_section_).lcon_);
      } else if (sections.exit_section_) {
        itransports.emplace_back(
            enter_idx, exit_idx,
            sections.sections_.at(*sections.exit_section_).lcon_);
      } else {
        itransports.emplace_back(enter_idx, exit_idx, 0, 0, 0, 0);
      }
    }
  }

  if (cj.final_footpath().is_footpath()) {
    auto const& fp = cj.final_footpath();
    auto const& from_station = *sched.stations_.at(fp.from_station_id_);
    auto const& to_station = *sched.stations_.at(fp.to_station_id_);

    auto& from_stop = push_station(from_station);
    auto const from_idx = static_cast<unsigned>(j.stops_.size() - 1);
    auto& dep = from_stop.departure_;
    dep.valid_ = true;
    dep.schedule_timestamp_ = last_time;
    dep.timestamp_ = dep.schedule_timestamp_;

    auto& to_stop = push_station(to_station);
    auto const to_idx = static_cast<unsigned>(j.stops_.size() - 1);
    auto& arr = to_stop.arrival_;
    arr.valid_ = true;
    arr.schedule_timestamp_ = last_time + fp.duration_ * 60;
    arr.timestamp_ = arr.schedule_timestamp_;

    itransports.emplace_back(from_idx, to_idx, fp.duration_, 0, 0, 0);
  }

  j.transports_ = mro::generate_journey_transports(itransports, sched);
  j.trips_ = mro::generate_journey_trips(itransports, sched);
  j.attributes_ = mro::generate_journey_attributes(itransports);

  return j;
}

msg_ptr revise_compact_journey(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonReviseCompactJourneyRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;

  auto const cjs = utl::to_vec(*req->journeys(), [&](auto const& fbs_cj) {
    return from_fbs(sched, fbs_cj);
  });

  auto const revised = utl::to_vec(
      cjs, [&](auto const& cj) { return revise_compact_journey(sched, cj); });

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonReviseCompactJourneyResponse,
      CreatePaxMonReviseCompactJourneyResponse(
          mc, mc.CreateVector(utl::to_vec(
                  revised,
                  [&](auto const& j) { return to_connection(mc, j, true); })))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api

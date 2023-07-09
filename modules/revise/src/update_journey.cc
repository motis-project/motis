#include "motis/revise/update_journey.h"

#include <fstream>
#include <numeric>
#include <utility>

#include "utl/concat.h"
#include "utl/verify.h"

#include "motis/core/common/interval_map.h"
#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/generate_journey_transport.h"
#include "motis/core/journey/journeys_to_message.h"

#include "motis/revise/get_all_stops.h"
#include "motis/revise/get_ev_key.h"
#include "motis/revise/update_journey_status.h"

#include "motis/module/module.h"

namespace motis::revise {

struct trp_cmp {
  bool operator()(trip const* a, trip const* b) const {
    return a->id_ < b->id_;
  }
};

struct con_info_cmp {
  bool operator()(connection_info const* a, connection_info const* b) const {
    auto train_nr_a = output_train_nr(a->train_nr_, a->original_train_nr_);
    auto train_nr_b = output_train_nr(b->train_nr_, b->original_train_nr_);
    return std::tie(a->line_identifier_, a->family_, train_nr_a, a->dir_) <
           std::tie(b->line_identifier_, b->family_, train_nr_b, b->dir_);
  }
};

struct free_text_cmp {
  bool operator()(free_text const* a, free_text const* b) const {
    return std::tie(a->code_, a->text_, a->type_) <
           std::tie(b->code_, b->text_, b->type_);
  }
};

void add_to_intervals(
    schedule const& sched, ev_key const& ev, int const interval,
    interval_map<trip const*, trp_cmp>& trip_intervals,
    interval_map<connection_info const*, con_info_cmp>& transport_intervals,
    interval_map<attribute const*>& attribute_intervals) {
  if (!ev) {
    return;
  }

  // trips
  for (auto const& trp : *sched.merged_trips_.at(ev.lcon()->trips_)) {
    trip_intervals.add_entry(trp, interval);
  }

  // attributes
  for (auto const& attribute : ev.lcon()->full_con_->con_info_->attributes_) {
    attribute_intervals.add_entry(attribute, interval);
  }

  // transports
  auto con_info = ev.lcon()->full_con_->con_info_;
  while (con_info != nullptr) {
    transport_intervals.add_entry(con_info, interval);
    con_info = con_info->merged_with_;
  }
}

void edges_to_journey(
    schedule const& sched, journey const& j, journey& new_journey,
    interval_map<free_text const*, free_text_cmp>& free_texts,
    interval_map<trip const*, trp_cmp>& trip_intervals,
    interval_map<connection_info const*, con_info_cmp>& transport_intervals,
    interval_map<attribute const*>& attribute_intervals) {
  auto const stops = get_all_stops(sched, j);
  for (auto const& stop : stops) {
    auto& new_stop = new_journey.stops_.emplace_back(stop->get_stop(sched));
    auto const stop_size = new_journey.stops_.size() - 1;

    if (stop->get_type() == stop_type_t::WALK_STOP) {
      continue;  // Walk stops don't have attributes, trips, etc.
    }

    // Add free texts that apply to
    // either departure or arrival of the journey stop.
    auto const it_free_text_dep =
        sched.graph_to_free_texts_.find(stop->get_dep());
    auto const it_free_text_arr =
        sched.graph_to_free_texts_.find(stop->get_arr());
    if (it_free_text_dep != end(sched.graph_to_free_texts_) ||
        it_free_text_arr != end(sched.graph_to_free_texts_)) {
      for (auto const& free_text :
           it_free_text_dep != end(sched.graph_to_free_texts_)
               ? it_free_text_dep->second
               : it_free_text_arr->second) {
        free_texts.add_entry(&free_text, stop_size);
      }
    }

    // Add trips from this stop in the old journey
    // matched by (station_id, arrival time).
    if (!new_stop.arrival_.valid_) {
      auto const stop_it = std::find_if(
          begin(j.stops_), end(j.stops_), [&](journey::stop const& s) {
            return s.arrival_.schedule_timestamp_ ==
                       new_stop.arrival_.schedule_timestamp_ &&
                   s.eva_no_ == new_stop.eva_no_;
          });
      auto const stop_idx = std::distance(begin(j.stops_), stop_it);
      for (auto const& t : j.trips_) {
        if (stop_idx > t.from_ && stop_idx <= t.to_) {
          trip_intervals.add_entry(from_extern_trip(sched, &t.extern_trip_),
                                   stop_size);
        }
      }
    }

    // Add trips from this stop in the old journey
    // matched by (station_id, departure time).
    if (!new_stop.departure_.valid_) {
      auto const stop_it = std::find_if(
          begin(j.stops_), end(j.stops_), [&](journey::stop const& s) {
            return s.departure_.schedule_timestamp_ ==
                       new_stop.departure_.schedule_timestamp_ &&
                   s.eva_no_ == new_stop.eva_no_;
          });
      auto const stop_idx = std::distance(begin(j.stops_), stop_it);
      for (auto const& t : j.trips_) {
        if (stop_idx >= t.from_ && stop_idx < t.to_) {
          trip_intervals.add_entry(from_extern_trip(sched, &t.extern_trip_),
                                   stop_size);
        }
      }
    }

    add_to_intervals(sched, stop->get_arr(), stop_size, trip_intervals,
                     transport_intervals, attribute_intervals);

    add_to_intervals(sched, stop->get_dep(), stop_size, trip_intervals,
                     transport_intervals, attribute_intervals);
  }
}

int get_stop_idx(journey const& con, int const schedule_time,
                 std::string const& eva_nr, event_type const type) {
  auto const event_info = std::find_if(
      begin(con.stops_), end(con.stops_), [&](journey::stop const& s) {
        return type == event_type::ARR
                   ? s.arrival_.schedule_timestamp_ == schedule_time &&
                         s.eva_no_ == eva_nr
                   : s.departure_.schedule_timestamp_ == schedule_time &&
                         s.eva_no_ == eva_nr;
      });
  utl::verify(event_info != end(con.stops_), "user : no event found");
  return std::distance(begin(con.stops_), event_info);
}

journey update_journey(schedule const& sched, journey const& j) {
  if (j.stops_.empty() ||
      std::none_of(begin(j.stops_), end(j.stops_),
                   [](journey::stop const& s) { return s.enter_; })) {
    return j;
  }

  auto updated_journey = journey{};
  interval_map<free_text const*, free_text_cmp> free_text_intervals;
  interval_map<trip const*, trp_cmp> trip_intervals;
  interval_map<connection_info const*, con_info_cmp> transport_intervals;
  interval_map<attribute const*> attribute_intervals;

  // compute stations and intervals
  edges_to_journey(sched, j, updated_journey, free_text_intervals,
                   trip_intervals, transport_intervals, attribute_intervals);

  // compute free_texts
  for (auto const& [free_text, range] :
       free_text_intervals.get_attribute_ranges()) {
    for (auto const& r : range) {
      updated_journey.free_texts_.push_back(
          {static_cast<unsigned>(r.from_),
           static_cast<unsigned>(r.to_),
           {free_text->code_, free_text->text_, free_text->type_}});
    }
  }
  std::sort(begin(updated_journey.free_texts_),
            end(updated_journey.free_texts_),
            [](journey::ranged_free_text const& lhs,
               journey::ranged_free_text const& rhs) {
              return lhs.from_ < rhs.from_;
            });

  // compute trips
  for (auto const& t : trip_intervals.get_attribute_ranges()) {
    auto const& p = t.first->id_.primary_;
    auto const& s = t.first->id_.secondary_;
    for (auto const& range : t.second) {
      updated_journey.trips_.push_back(journey::trip{
          static_cast<unsigned>(range.from_), static_cast<unsigned>(range.to_),
          extern_trip{sched.stations_.at(p.station_id_)->eva_nr_,
                      t.first->gtfs_trip_id_, p.get_train_nr(),
                      motis_to_unixtime(sched, p.get_time()),
                      sched.stations_.at(s.target_station_id_)->eva_nr_,
                      motis_to_unixtime(sched, s.target_time_), s.line_id_},
          t.first->dbg_.str()});
    }
  }
  std::sort(begin(updated_journey.trips_), end(updated_journey.trips_),
            [](journey::trip const& lhs, journey::trip const& rhs) {
              return lhs.from_ < rhs.from_;
            });

  // compute transports
  for (auto const& t : transport_intervals.get_attribute_ranges()) {
    for (auto const& range : t.second) {
      updated_journey.transports_.push_back(
          generate_journey_transport(range.from_, range.to_, t.first, sched));
    }
  }
  std::vector<journey::transport> old_walk_transports;
  for (auto const& t : j.transports_) {
    if (t.is_walk_) {
      old_walk_transports.push_back(t);
    }
  }
  if (!old_walk_transports.empty()) {
    auto walk_section = !updated_journey.stops_.empty() &&
                        !updated_journey.stops_.front().enter_;
    auto walk_transport_idx = 0U;
    for (auto stop_it = begin(updated_journey.stops_);
         stop_it != end(updated_journey.stops_); ++stop_it) {
      auto const idx = std::distance(begin(updated_journey.stops_), stop_it);
      if (walk_section && old_walk_transports.size() > walk_transport_idx &&
          idx > 0) {
        updated_journey.transports_.push_back(
            old_walk_transports[walk_transport_idx]);
        updated_journey.transports_.back().from_ = idx - 1;
        updated_journey.transports_.back().to_ = idx;
        walk_transport_idx++;
      }
      if (stop_it->enter_ && !stop_it->exit_) {
        walk_section = false;
      }
      if (stop_it->exit_ && !stop_it->enter_) {
        walk_section = true;
      }
    }
  }
  for (auto const& s : get_sections(updated_journey)) {
    if (s.type_ == section_type::WALK ||
        updated_journey.stops_.at(s.from_).departure_.valid_ ||
        updated_journey.stops_.at(s.to_).arrival_.valid_) {
      continue;
    }
    auto const first_idx = get_stop_idx(
        j, updated_journey.stops_.at(s.from_).departure_.schedule_timestamp_,
        updated_journey.stops_.at(s.from_).eva_no_, event_type::DEP);
    auto const last_idx = get_stop_idx(
        j, updated_journey.stops_.at(s.to_).arrival_.schedule_timestamp_,
        updated_journey.stops_.at(s.to_).eva_no_, event_type::ARR);
    for (auto const& t : j.transports_) {
      if (static_cast<unsigned>(first_idx) >= t.from_ &&
          static_cast<unsigned>(last_idx) <= t.to_) {
        updated_journey.transports_.emplace_back(t);
        updated_journey.transports_.back().from_ = first_idx;
        updated_journey.transports_.back().to_ = last_idx;
      }
    }
  }
  std::sort(begin(updated_journey.transports_),
            end(updated_journey.transports_),
            [](journey::transport const& lhs, journey::transport const& rhs) {
              return lhs.from_ < rhs.from_;
            });

  // compute attributes
  for (auto const& [attr, ranges] :
       attribute_intervals.get_attribute_ranges()) {
    for (auto const& range : ranges) {
      updated_journey.attributes_.push_back({static_cast<unsigned>(range.from_),
                                             static_cast<unsigned>(range.to_),
                                             {attr->code_, attr->text_}});
    }
  }
  std::sort(begin(updated_journey.attributes_),
            end(updated_journey.attributes_),
            [](journey::ranged_attribute const& lhs,
               journey::ranged_attribute const& rhs) {
              return lhs.from_ < rhs.from_;
            });

  updated_journey.duration_ =
      updated_journey.stops_.back().arrival_.timestamp_ -
      updated_journey.stops_.front().departure_.timestamp_;
  updated_journey.db_costs_ = j.db_costs_;
  updated_journey.night_penalty_ = j.night_penalty_;
  updated_journey.price_ = j.price_;
  updated_journey.transfers_ = j.transfers_;
  update_journey_status(sched, updated_journey);
  return updated_journey;
}

}  // namespace motis::revise

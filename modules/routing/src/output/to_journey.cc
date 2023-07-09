#include "motis/routing/output/to_journey.h"

#include <string>

#include "motis/core/common/interval_map.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/generate_journey_transport.h"

namespace motis::routing::output {

std::vector<journey::transport> generate_journey_transports(
    std::vector<intermediate::transport> const& transports,
    schedule const& sched) {
  struct con_info_cmp {
    bool operator()(connection_info const* a, connection_info const* b) const {
      auto train_nr_a = output_train_nr(a->train_nr_, a->original_train_nr_);
      auto train_nr_b = output_train_nr(b->train_nr_, b->original_train_nr_);
      return std::tie(a->line_identifier_, a->family_, train_nr_a, a->dir_) <
             std::tie(b->line_identifier_, b->family_, train_nr_b, b->dir_);
    }
  };

  std::vector<journey::transport> journey_transports;
  interval_map<connection_info const*, con_info_cmp> intervals;
  for (auto const& t : transports) {
    if (t.con_ != nullptr) {
      auto con_info = t.con_->full_con_->con_info_;
      while (con_info != nullptr) {
        intervals.add_entry(con_info, t.from_, t.to_);
        con_info = con_info->merged_with_;
      }
    } else {
      journey_transports.push_back(generate_journey_transport(
          t.from_, t.to_, nullptr, sched, t.duration_, t.mumo_id_,
          t.mumo_price_, t.mumo_accessibility_));
    }
  }

  for (auto const& t : intervals.get_attribute_ranges()) {
    for (auto const& range : t.second) {
      journey_transports.push_back(
          generate_journey_transport(range.from_, range.to_, t.first, sched));
    }
  }

  std::sort(begin(journey_transports), end(journey_transports),
            [](journey::transport const& lhs, journey::transport const& rhs) {
              return lhs.from_ < rhs.from_;
            });

  return journey_transports;
}

std::vector<journey::trip> generate_journey_trips(
    std::vector<intermediate::transport> const& transports,
    schedule const& sched) {
  struct trp_cmp {
    bool operator()(trip const* a, trip const* b) const {
      return a->id_ < b->id_;
    }
  };

  interval_map<trip const*, trp_cmp> intervals;
  for (auto const& t : transports) {
    if (t.con_ == nullptr) {
      continue;
    }

    for (auto const& trp : *sched.merged_trips_.at(t.con_->trips_)) {
      intervals.add_entry(trp, t.from_, t.to_);
    }
  }

  std::vector<journey::trip> journey_trips;
  for (auto const& t : intervals.get_attribute_ranges()) {
    auto const& p = t.first->id_.primary_;
    auto const& s = t.first->id_.secondary_;
    for (auto const& range : t.second) {
      journey_trips.push_back(journey::trip{
          static_cast<unsigned>(range.from_), static_cast<unsigned>(range.to_),
          extern_trip{t.first->gtfs_trip_id_,
                      sched.stations_.at(p.station_id_)->eva_nr_,
                      p.get_train_nr(), motis_to_unixtime(sched, p.get_time()),
                      sched.stations_.at(s.target_station_id_)->eva_nr_,
                      motis_to_unixtime(sched, s.target_time_), s.line_id_},
          t.first->dbg_.str()});
    }
  }

  std::sort(begin(journey_trips), end(journey_trips),
            [](journey::trip const& lhs, journey::trip const& rhs) {
              return lhs.from_ < rhs.from_;
            });

  return journey_trips;
}

std::vector<journey::stop> generate_journey_stops(
    std::vector<intermediate::stop> const& stops, schedule const& sched) {
  std::vector<journey::stop> journey_stops;
  for (auto const& stop : stops) {
    auto const& station = *sched.stations_[stop.station_id_];
    journey_stops.push_back(
        {stop.exit_, stop.enter_, station.name_.str(), station.eva_nr_.str(),
         station.width_, station.length_,
         stop.a_time_ != INVALID_TIME
             ? journey::stop::event_info{true,
                                         motis_to_unixtime(
                                             sched.schedule_begin_,
                                             stop.a_time_),
                                         motis_to_unixtime(
                                             sched.schedule_begin_,
                                             stop.a_sched_time_),
                                         stop.a_reason_,
                                         sched.tracks_[stop.a_track_].str(),
                                         sched.tracks_[stop.a_sched_track_]
                                             .str()}
             : journey::stop::event_info{false, 0, 0,
                                         timestamp_reason::SCHEDULE, "", ""},
         stop.d_time_ != INVALID_TIME
             ? journey::stop::event_info{true,
                                         motis_to_unixtime(
                                             sched.schedule_begin_,
                                             stop.d_time_),
                                         motis_to_unixtime(
                                             sched.schedule_begin_,
                                             stop.d_sched_time_),
                                         stop.d_reason_,
                                         sched.tracks_[stop.d_track_].str(),
                                         sched.tracks_[stop.d_sched_track_]
                                             .str()}
             : journey::stop::event_info{false, 0, 0,
                                         timestamp_reason::SCHEDULE, "", ""}});
  }
  return journey_stops;
}

std::vector<journey::ranged_attribute> generate_journey_attributes(
    std::vector<intermediate::transport> const& transports) {
  interval_map<attribute const*> attributes;
  for (auto const& t : transports) {
    if (t.con_ == nullptr) {
      continue;
    } else {
      for (auto const& attribute : t.con_->full_con_->con_info_->attributes_) {
        attributes.add_entry(attribute, t.from_, t.to_);
      }
    }
  }

  std::vector<journey::ranged_attribute> journey_attributes;
  for (auto const& attribute_range : attributes.get_attribute_ranges()) {
    auto const& attribute = attribute_range.first;
    auto const& attribute_ranges = attribute_range.second;
    auto const& code = attribute->code_;
    auto const& text = attribute->text_;

    for (auto const& range : attribute_ranges) {
      journey_attributes.push_back({static_cast<unsigned>(range.from_),
                                    static_cast<unsigned>(range.to_),
                                    {code, text}});
    }
  }

  std::sort(begin(journey_attributes), end(journey_attributes));

  return journey_attributes;
}

}  // namespace motis::routing::output

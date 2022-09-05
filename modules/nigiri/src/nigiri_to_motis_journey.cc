#include "motis/nigiri/nigiri_to_motis_journey.h"

#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/parser/csv.h"

#include "nigiri/routing/journey.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/core/common/interval_map.h"
#include "motis/core/common/unixtime.h"

#include "motis/nigiri/unixtime_conv.h"

namespace n = ::nigiri;

namespace motis::nigiri {

struct transport_display_info {
  CISTA_COMPARABLE()
  n::clasz clasz_;
  n::string display_name_;
};

mcd::string get_station_id(std::vector<std::string> const& tags,
                           n::timetable const& tt, n::location_idx_t const l) {
  return tags.at(to_idx(tt.locations_.src_.at(l))) +
         tt.locations_.ids_.at(l).str();
}

extern_trip nigiri_trip_to_extern_trip(std::vector<std::string> const& tags,
                                       n::timetable const& tt,
                                       n::trip_idx_t const trip,
                                       n::day_idx_t const day) {
  auto const [transport, stop_range] = tt.trip_ref_transport_[trip];
  auto const first_location =
      n::timetable::stop{
          tt.route_location_seq_[tt.transport_route_[transport]].front()}
          .location_idx();
  auto const last_location =
      n::timetable::stop{
          tt.route_location_seq_[tt.transport_route_[transport]].back()}
          .location_idx();
  auto const& id = tt.trip_id_strings_.at(tt.trip_ids_.at(trip).back());
  auto const [admin, train_nr, first_stop_eva, fist_start_time, last_stop_eva,
              last_stop_time, line] =
      utl::split<'/', utl::cstr, unsigned, utl::cstr, unsigned, utl::cstr,
                 unsigned, utl::cstr>(id.view());
  return extern_trip{
      .station_id_ = get_station_id(tags, tt, first_location),
      .train_nr_ = train_nr,
      .time_ = to_motis_unixtime(tt.event_time(
          {transport, day}, stop_range.from_, n::event_type::kDep)),
      .target_station_id_ = get_station_id(tags, tt, last_location),
      .target_time_ = to_motis_unixtime(tt.event_time(
          {transport, day}, stop_range.to_ - 1, n::event_type::kArr)),
      .line_id_ = line.to_str()};
}

motis::journey nigiri_to_motis_journey(n::timetable const& tt,
                                       std::vector<std::string> const& tags,
                                       n::routing::journey const& nj) {
  journey mj;

  auto const fill_stop_info = [&](motis::journey::stop& s,
                                  n::location_idx_t const l) {
    auto const& l_name = tt.locations_.names_.at(l);
    auto const& pos = tt.locations_.coordinates_.at(l);
    s.name_ = l_name.str();
    s.eva_no_ = get_station_id(tags, tt, l);
    s.lat_ = pos.lat_;
    s.lng_ = pos.lng_;
  };

  auto const add_walk = [&](n::routing::journey::leg const& leg, int mumo_id) {
    auto& from_stop =
        mj.stops_.empty() ? mj.stops_.emplace_back() : mj.stops_.back();
    auto const from_idx = static_cast<unsigned>(mj.stops_.size() - 1);
    fill_stop_info(from_stop, leg.from_);

    auto& to_stop = mj.stops_.emplace_back();
    auto const to_idx = static_cast<unsigned>(mj.stops_.size() - 1);
    fill_stop_info(to_stop, leg.to_);

    auto t = journey::transport{};
    t.from_ = from_idx;
    t.to_ = to_idx;
    t.is_walk_ = true;
    t.duration_ =
        static_cast<unsigned>((leg.arr_time_ - leg.dep_time_).count());
    t.mumo_id_ = mumo_id;
    mj.transports_.emplace_back(std::move(t));
  };

  interval_map<transport_display_info> transports;
  interval_map<std::pair<extern_trip, std::string>> extern_trips;

  auto const add_transports = [&](n::transport const t, unsigned section_idx) {
    auto x = journey::transport{};
    x.from_ = x.to_ = section_idx;

    auto const trips_on_section = tt.transport_to_trip_section_.at(t.t_idx_);
    auto const merged_trips_idx =
        trips_on_section.at(trips_on_section.size() == 1U ? 0U : section_idx);
    for (auto const trip : tt.merged_trips_.at(merged_trips_idx)) {
      auto const clasz_sections =
          tt.route_section_clasz_.at(tt.transport_route_.at(t.t_idx_));
      auto const clasz =
          clasz_sections.at(clasz_sections.size() == 1U ? 0U : section_idx);
      transports.add_entry(
          transport_display_info{
              .clasz_ = clasz,
              .display_name_ = tt.trip_display_names_.at(trip).view()},
          mj.stops_.size() - 1);

      // TODO(felix) maybe the day index needs to be changed according to the
      // offset between the occurance in a rule service expanded trip vs. the
      // reference trip. For now, no rule services are implemented.
      extern_trips.add_entry(
          std::pair{
              nigiri_trip_to_extern_trip(tags, tt, trip, t.day_),
              std::string{tt.source_file_names_
                              .at(tt.trip_debug_.at(trip)[0].source_file_idx_)
                              .view()} +
                  std::to_string(tt.trip_debug_.at(trip)[0].line_number_)},
          mj.stops_.size() - 1);
    }
  };

  for (auto const& leg : nj.legs_) {
    leg.uses_.apply(utl::overloaded{
        [&](n::routing::journey::transport_enter_exit const& t) {
          auto const& route_idx = tt.transport_route_.at(t.t_.t_idx_);
          auto const& stop_seq = tt.route_location_seq_.at(route_idx);

          for (auto const& stop_idx : t.stop_range_) {
            auto const exit = (stop_idx == t.stop_range_.to_ - 1U);
            auto const enter = (stop_idx == t.stop_range_.from_);

            // for entering: create a new stop if it's the first stop in journey
            // otherwise: create a new stop
            auto const reuse_arrival = enter && !mj.stops_.empty();
            auto& stop =
                reuse_arrival ? mj.stops_.back() : mj.stops_.emplace_back();
            fill_stop_info(
                stop, n::timetable::stop{stop_seq.at(stop_idx)}.location_idx());

            if (exit) {
              stop.exit_ = true;
            }
            if (enter) {
              stop.enter_ = true;
            }

            if (!enter) {
              auto const time = to_motis_unixtime(
                  tt.event_time(t.t_, stop_idx, n::event_type::kArr));
              stop.arrival_ = journey::stop::event_info{
                  .valid_ = true,
                  .timestamp_ = time,
                  .schedule_timestamp_ = time,
                  .timestamp_reason_ = timestamp_reason::SCHEDULE,
                  .track_ = "",
                  .schedule_track_ = ""};
            }

            if (!exit) {
              auto const time = to_motis_unixtime(
                  tt.event_time(t.t_, stop_idx, n::event_type::kDep));
              stop.departure_ = journey::stop::event_info{
                  .valid_ = true,
                  .timestamp_ = time,
                  .schedule_timestamp_ = time,
                  .timestamp_reason_ = timestamp_reason::SCHEDULE,
                  .track_ = "",
                  .schedule_track_ = ""};
            }

            if (!exit) {
              add_transports(t.t_, stop_idx);
            }
          }
        },
        [&](n::footpath_idx_t const) { add_walk(leg, -1); },
        [&](std::uint8_t const x) { add_walk(leg, x); }});
  }

  for (auto const& [x, ranges] : transports.get_attribute_ranges()) {
    for (auto const& r : ranges) {
      auto t = journey::transport{};
      t.from_ = r.from_;
      t.to_ = r.to_;
      t.clasz_ = static_cast<std::underlying_type_t<n::clasz>>(x.clasz_);
      t.name_ = x.display_name_;
      mj.transports_.emplace_back(std::move(t));
    }
  }

  for (auto const& [et, ranges] : extern_trips.get_attribute_ranges()) {
    for (auto const& r : ranges) {
      mj.trips_.emplace_back(journey::trip{.from_ = r.from_,
                                           .to_ = r.to_,
                                           .extern_trip_ = et.first,
                                           .debug_ = et.second});
    }
  }

  return mj;
}

}  // namespace motis::nigiri

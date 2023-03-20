#include "motis/nigiri/nigiri_to_motis_journey.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/parser/split.h"

#include "nigiri/routing/journey.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/core/common/interval_map.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/journey/print_journey.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = ::nigiri;

namespace motis::nigiri {

struct transport_display_info {
  CISTA_COMPARABLE()
  unsigned duration_;
  n::clasz clasz_;
  std::string display_name_;
  std::string direction_;
  std::string provider_;
  std::string line_;
};

n::location_idx_t resolve_parent(n::timetable const& tt,
                                 n::location_idx_t const x) {
  return tt.locations_.types_.at(x) == n::location_type::kTrack
             ? tt.locations_.parents_.at(x)
             : x;
}

mcd::string get_station_id(std::vector<std::string> const& tags,
                           n::timetable const& tt, n::location_idx_t const l) {
  return tags.at(to_idx(tt.locations_.src_.at(l))) +
         std::string{tt.locations_.ids_.at(l).view()};
}

extern_trip nigiri_trip_to_extern_trip(std::vector<std::string> const& tags,
                                       n::timetable const& tt,
                                       n::trip_idx_t const trip,
                                       n::day_idx_t const day) {
  auto const [transport, stop_range] = tt.trip_ref_transport_[trip];
  auto const first_location = resolve_parent(
      tt,
      n::timetable::stop{
          tt.route_location_seq_[tt.transport_route_[transport]].front()}
          .location_idx());
  auto const last_location = resolve_parent(
      tt,
      n::timetable::stop{
          tt.route_location_seq_[tt.transport_route_[transport]].back()}
          .location_idx());
  auto const& id = tt.trip_id_strings_.at(tt.trip_ids_.at(trip).back()).view();
  auto const [train_nr, first_stop_eva, fist_start_time, last_stop_eva,
              last_stop_time, line] =
      utl::split<'/', unsigned, utl::cstr, unsigned, utl::cstr, unsigned,
                 utl::cstr>(id);
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
                                  n::location_idx_t const x) {
    auto const l = resolve_parent(tt, x);
    auto const& l_name = tt.locations_.names_.at(l);
    auto const& pos = tt.locations_.coordinates_.at(l);
    s.name_ = l_name.view();
    s.eva_no_ = get_station_id(tags, tt, l);
    s.lat_ = pos.lat_;
    s.lng_ = pos.lng_;
  };

  auto const add_walk = [&](n::routing::journey::leg const& leg,
                            n::duration_t const duration, int mumo_id,
                            bool const is_last) {
    auto const is_transfer =
        leg.from_ == leg.to_ ||
        (leg.from_ == tt.locations_.parents_.at(leg.to_)) ||
        (leg.to_ == tt.locations_.parents_.at(leg.from_)) ||
        (tt.locations_.parents_.at(leg.from_) ==
             tt.locations_.parents_.at(leg.to_) &&
         (tt.locations_.types_.at(leg.from_) == n::location_type::kTrack ||
          tt.locations_.types_.at(leg.to_) == n::location_type::kTrack));

    if (is_transfer && is_last) {
      return;
    }

    auto& from_stop =
        mj.stops_.empty() ? mj.stops_.emplace_back() : mj.stops_.back();
    auto const from_idx = static_cast<unsigned>(mj.stops_.size() - 1);
    fill_stop_info(from_stop, leg.from_);
    from_stop.departure_.valid_ = true;
    from_stop.departure_.timestamp_ = to_motis_unixtime(leg.dep_time_);
    from_stop.departure_.schedule_timestamp_ = to_motis_unixtime(leg.dep_time_);

    if (!is_transfer) {
      auto& to_stop = is_transfer ? mj.stops_.back() : mj.stops_.emplace_back();
      auto const to_idx = static_cast<unsigned>(mj.stops_.size() - 1);
      fill_stop_info(to_stop, leg.to_);
      to_stop.arrival_.valid_ = true;
      to_stop.arrival_.timestamp_ = to_motis_unixtime(leg.arr_time_);
      to_stop.arrival_.schedule_timestamp_ = to_motis_unixtime(leg.arr_time_);

      auto t = journey::transport{};
      t.from_ = from_idx;
      t.to_ = to_idx;
      t.is_walk_ = true;
      t.duration_ = duration.count();
      t.mumo_id_ = mumo_id;
      mj.transports_.emplace_back(std::move(t));
    }
  };

  interval_map<transport_display_info> transports;
  interval_map<std::pair<extern_trip, std::string /* debug */>> extern_trips;
  interval_map<attribute> attributes;

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

      auto const provider_sections =
          tt.transport_section_providers_.at(t.t_idx_);
      auto const provider_idx = provider_sections.at(
          provider_sections.size() == 1U ? 0U : section_idx);
      auto const provider =
          std::string{tt.providers_.at(provider_idx).long_name_.view()};

      auto const direction_sections =
          tt.transport_section_directions_.at(t.t_idx_);
      std::string direction;
      if (!direction_sections.empty()) {
        auto const direction_idx = direction_sections.size() == 1U
                                       ? direction_sections.at(0)
                                       : direction_sections.at(section_idx);
        if (direction_idx != n::trip_direction_idx_t::invalid()) {
          direction = tt.trip_directions_.at(direction_idx)
                          .apply(utl::overloaded{
                              [&](n::trip_direction_string_idx_t const i) {
                                return tt.trip_direction_strings_.at(i).view();
                              },
                              [&](n::location_idx_t const i) {
                                return tt.locations_.names_.at(i).view();
                              }});
        }
      }

      auto const line_sections = tt.transport_section_lines_.at(t.t_idx_);
      std::string line;
      if (!line_sections.empty()) {
        auto const line_idx = line_sections.size() == 1U
                                  ? line_sections.at(0U)
                                  : line_sections.at(section_idx);
        if (line_idx != n::trip_line_idx_t::invalid()) {
          line = tt.trip_lines_.at(line_idx).view();
        }
      }

      transports.add_entry(
          transport_display_info{
              .duration_ = 0U,
              .clasz_ = clasz,
              .display_name_ =
                  std::string{tt.trip_display_names_.at(trip).view()},
              .direction_ = direction,
              .provider_ = provider,
              .line_ = line},
          mj.stops_.size() - 1, mj.stops_.size());

      // TODO(felix) maybe the day index needs to be changed according to the
      // offset between the occurrence in a rule service expanded trip vs. the
      // reference trip. For now, no rule services are implemented.
      extern_trips.add_entry(
          std::pair{
              nigiri_trip_to_extern_trip(tags, tt, trip, t.day_),
              fmt::format(
                  "{}:{}:{}",
                  tt.source_file_names_
                      .at(tt.trip_debug_.at(trip)[0].source_file_idx_)
                      .view(),
                  std::to_string(tt.trip_debug_.at(trip)[0].line_number_from_),
                  std::to_string(tt.trip_debug_.at(trip)[0].line_number_to_))},
          mj.stops_.size() - 1, mj.stops_.size());

      auto const section_attributes =
          tt.transport_section_attributes_.at(t.t_idx_);
      if (!section_attributes.empty()) {
        auto const attribute_combi = section_attributes.size() == 1U
                                         ? section_attributes.at(0)
                                         : section_attributes.at(section_idx);

        for (auto const& attr :
             tt.attribute_combinations_.at(attribute_combi)) {
          attributes.add_entry(
              attribute{.code_ = tt.attributes_.at(attr).code_.view(),
                        .text_ = tt.attributes_.at(attr).text_.view()},
              mj.stops_.size() - 1, mj.stops_.size());
        }
      }
    }
  };

  for (auto const [i, leg] : utl::enumerate(nj.legs_)) {
    std::visit(
        utl::overloaded{
            [&](n::routing::journey::transport_enter_exit const& t) {
              auto const& route_idx = tt.transport_route_.at(t.t_.t_idx_);
              auto const& stop_seq = tt.route_location_seq_.at(route_idx);

              for (auto const& stop_idx : t.stop_range_) {
                auto const exit = (stop_idx == t.stop_range_.to_ - 1U);
                auto const enter = (stop_idx == t.stop_range_.from_);

                // for entering: create a new stop if it's the first stop in
                // journey otherwise: create a new stop
                auto const reuse_arrival = enter && !mj.stops_.empty();
                auto& stop =
                    reuse_arrival ? mj.stops_.back() : mj.stops_.emplace_back();
                auto const l =
                    n::timetable::stop{stop_seq.at(stop_idx)}.location_idx();
                fill_stop_info(stop, l);

                if (exit) {
                  stop.exit_ = true;
                }
                if (enter) {
                  stop.enter_ = true;
                }

                if (!enter) {
                  auto const time = to_motis_unixtime(
                      tt.event_time(t.t_, stop_idx, n::event_type::kArr));
                  auto const track =
                      tt.locations_.types_.at(l) == n::location_type::kTrack
                          ? tt.locations_.names_.at(l).view()
                          : "";
                  stop.arrival_.valid_ = true;
                  stop.arrival_.timestamp_ = time;
                  stop.arrival_.schedule_timestamp_ = time;
                  stop.arrival_.timestamp_reason_ = timestamp_reason::SCHEDULE;
                  stop.arrival_.track_ = std::string{track};
                  stop.arrival_.schedule_track_ = std::string{track};
                }

                if (!exit) {
                  auto const time = to_motis_unixtime(
                      tt.event_time(t.t_, stop_idx, n::event_type::kDep));
                  auto const track =
                      tt.locations_.types_.at(l) == n::location_type::kTrack
                          ? tt.locations_.names_.at(l).view()
                          : "";
                  stop.departure_.valid_ = true;
                  stop.departure_.timestamp_ = time;
                  stop.departure_.schedule_timestamp_ = time;
                  stop.departure_.timestamp_reason_ =
                      timestamp_reason::SCHEDULE;
                  stop.departure_.track_ = std::string{track};
                  stop.departure_.schedule_track_ = std::string{track};
                }

                if (!exit) {
                  add_transports(t.t_, stop_idx);
                }
              }
            },
            [&, i = i, leg = leg](n::footpath const fp) {
              add_walk(leg, fp.duration_, -1, i == nj.legs_.size() - 1U);
            },
            [&, leg = leg](n::routing::offset const x) {
              add_walk(leg, x.duration_, x.type_, false);
            }},
        leg.uses_);
  }

  for (auto const& [x, ranges] : transports.get_attribute_ranges()) {
    for (auto const& r : ranges) {
      auto t = journey::transport{};
      t.from_ = r.from_;
      t.to_ = r.to_;
      t.clasz_ = static_cast<std::underlying_type_t<n::clasz>>(x.clasz_);
      t.name_ = x.display_name_;
      t.provider_ = x.provider_;
      t.direction_ = x.direction_;
      t.line_identifier_ = x.line_;
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

  for (auto const& [attr, ranges] : attributes.get_attribute_ranges()) {
    for (auto const& r : ranges) {
      mj.attributes_.emplace_back(journey::ranged_attribute{
          .from_ = r.from_, .to_ = r.to_, .attr_ = attr});
    }
  }

  std::sort(begin(mj.transports_), end(mj.transports_),
            [](auto&& a, auto&& b) { return a.from_ < b.from_; });
  std::sort(begin(mj.trips_), end(mj.trips_));
  std::sort(begin(mj.attributes_), end(mj.attributes_));

  return mj;
}

}  // namespace motis::nigiri

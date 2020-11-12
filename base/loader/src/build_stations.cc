#include "motis/loader/build_stations.h"

#include "geo/point_rtree.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/verify.h"

#include "motis/memory.h"

#include "motis/core/schedule/schedule.h"

#include "motis/loader/filter/local_stations.h"
#include "motis/loader/interval_util.h"
#include "motis/loader/timezone_util.h"

#include "motis/schedule-format/Schedule_generated.h"

namespace f = flatbuffers64;

namespace motis::loader {

constexpr auto const kLinkNearbyMaxDistance = 300;  // [m];

struct stations_builder {
  explicit stations_builder(schedule& sched, bool no_local_stations)
      : sched_{sched}, no_local_stations_{no_local_stations} {}

  void add_dummy_node(std::string const& name) {
    auto const station_idx = sched_.station_nodes_.size();

    // Create dummy station node.
    sched_.station_nodes_.emplace_back(mcd::make_unique<station_node>(
        make_station_node(static_cast<unsigned>(station_idx))));

    // Create dummy station object.
    auto s = mcd::make_unique<station>();
    s->index_ = station_idx;
    s->eva_nr_ = name;
    s->name_ = name;

    sched_.eva_to_station_.emplace(name, s.get());
    sched_.stations_.emplace_back(std::move(s));
  }

  void add_station(uint32_t const source_schedule, Station const* fbs_station) {
    if (skip_station(fbs_station)) {
      return;
    }

    auto const station_idx = sched_.station_nodes_.size();

    // Create station node.
    auto node_ptr = mcd::make_unique<station_node>(
        make_station_node(static_cast<unsigned>(station_idx)));
    station_nodes_[fbs_station] = node_ptr.get();
    sched_.station_nodes_.emplace_back(std::move(node_ptr));

    // Create station object.
    auto s = mcd::make_unique<station>();
    s->index_ = station_idx;
    s->name_ = fbs_station->name()->str();
    s->width_ = fbs_station->lat();
    s->length_ = fbs_station->lng();
    s->eva_nr_ = std::string{sched_.prefixes_[source_schedule]} +
                 fbs_station->id()->str();
    s->transfer_time_ = std::max(2, fbs_station->interchange_time());
    s->timez_ = fbs_station->timezone() != nullptr
                    ? get_or_create_timezone(fbs_station->timezone())
                    : nullptr;
    s->equivalent_.push_back(s.get());
    s->source_schedule_ = source_schedule;

    // Store DS100.
    if (fbs_station->external_ids() != nullptr) {
      s->external_ids_.reserve(fbs_station->external_ids()->size());
      for (auto const& ds100 : *fbs_station->external_ids()) {
        sched_.ds100_to_station_.emplace(ds100->str(), s.get());
        s->external_ids_.emplace_back(ds100->str());
      }
    }

    utl::verify(
        sched_.eva_to_station_.find(s->eva_nr_) == end(sched_.eva_to_station_),
        "add_station: have non-unique station_id: {}", s->eva_nr_);

    sched_.eva_to_station_.emplace(s->eva_nr_, s.get());
    sched_.stations_.emplace_back(std::move(s));
  }

  timezone const* get_or_create_timezone(Timezone const* input_timez) {
    return utl::get_or_create(timezones_, input_timez, [&] {
      auto const tz =
          input_timez->season() != nullptr
              ? create_timezone(
                    input_timez->general_offset(),
                    input_timez->season()->offset(),  //
                    first_day_, last_day_,
                    input_timez->season()->day_idx_first_day(),
                    input_timez->season()->day_idx_last_day(),
                    input_timez->season()->minutes_after_midnight_first_day(),
                    input_timez->season()->minutes_after_midnight_last_day())
              : timezone{input_timez->general_offset()};
      sched_.timezones_.emplace_back(mcd::make_unique<timezone>(tz));
      return sched_.timezones_.back().get();
    });
  }

  void link_meta_stations(
      f::Vector<f::Offset<MetaStation>> const* meta_stations) {
    for (auto const& meta : *meta_stations) {
      if (skip_station(meta->station())) {
        continue;
      }
      auto& station =
          *sched_.stations_[station_nodes_.at(meta->station())->id_];
      for (auto const& fbs_equivalent : *meta->equivalent()) {
        if (skip_station(fbs_equivalent)) {
          continue;
        }
        auto& equivalent =
            *sched_.stations_[station_nodes_.at(fbs_equivalent)->id_];
        if (station.index_ != equivalent.index_) {
          station.equivalent_.push_back(&equivalent);
        }
      }
    }
  }

  void link_nearby_stations() {
    auto const station_rtree =
        geo::make_point_rtree(sched_.stations_, [](auto const& s) {
          return geo::latlng{s->lat(), s->lng()};
        });

    for (auto const& [from_idx, from_station] :
         utl::enumerate(sched_.stations_)) {
      if (from_station->source_schedule_ == NO_SOURCE_SCHEDULE) {
        continue;  // no dummy stations
      }

      for (auto const& to_idx : station_rtree.in_radius(
               geo::latlng{from_station->lat(), from_station->lng()},
               kLinkNearbyMaxDistance)) {
        if (from_idx == to_idx) {
          continue;
        }

        auto& to_station = sched_.stations_.at(to_idx);
        if (to_station->source_schedule_ == NO_SOURCE_SCHEDULE) {
          continue;  // no dummy stations
        }

        if (from_station->source_schedule_ == to_station->source_schedule_) {
          continue;  // don't shortcut yourself
        }

        from_station->equivalent_.push_back(to_station.get());
      }
    }
  }

  inline bool skip_station(Station const* station) {
    return no_local_stations_ && is_local_station(station);
  }

  schedule& sched_;
  int first_day_{0}, last_day_{0};
  mcd::hash_map<Station const*, station_node*> station_nodes_;
  mcd::hash_map<Timezone const*, timezone const*> timezones_;
  bool no_local_stations_{false};
};

mcd::hash_map<Station const*, station_node*> build_stations(
    schedule& sched, std::vector<Schedule const*> const& fbs_schedules,
    bool no_local_stations) {
  stations_builder b{sched, no_local_stations};

  // Add dummy stations.
  b.add_dummy_node(STATION_START);
  b.add_dummy_node(STATION_END);
  b.add_dummy_node(STATION_VIA0);
  b.add_dummy_node(STATION_VIA1);
  b.add_dummy_node(STATION_VIA2);
  b.add_dummy_node(STATION_VIA3);
  b.add_dummy_node(STATION_VIA4);
  b.add_dummy_node(STATION_VIA5);
  b.add_dummy_node(STATION_VIA6);
  b.add_dummy_node(STATION_VIA7);
  b.add_dummy_node(STATION_VIA8);
  b.add_dummy_node(STATION_VIA9);

  // Add actual stations.
  for (auto const& [src_index, fbs_schedule] : utl::enumerate(fbs_schedules)) {
    std::tie(b.first_day_, b.last_day_) =
        first_last_days(sched, fbs_schedule->interval());

    for (auto const* fbs_station : *fbs_schedule->stations()) {
      b.add_station(src_index, fbs_station);
    }

    if (fbs_schedule->meta_stations() != nullptr) {
      b.link_meta_stations(fbs_schedule->meta_stations());
    }
  }

  if (fbs_schedules.size() > 1) {
    b.link_nearby_stations();
  }

  sched.node_count_ = sched.station_nodes_.size();

  return std::move(b.station_nodes_);
}

}  // namespace motis::loader

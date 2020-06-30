#include "motis/loader/graph_builder.h"

#include <cassert>
#include <algorithm>
#include <functional>
#include <numeric>

#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "date/date.h"

#include "utl/parser/cstr.h"

#include "motis/core/common/constants.h"
#include "motis/core/common/logging.h"
#include "motis/core/schedule/build_route_node.h"
#include "motis/core/schedule/category.h"
#include "motis/core/schedule/price.h"
#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/loader/build_graph.h"
#include "motis/loader/classes.h"
#include "motis/loader/footpaths.h"
#include "motis/loader/rule_route_builder.h"
#include "motis/loader/rule_service_graph_builder.h"
#include "motis/loader/timezone_util.h"
#include "motis/loader/util.h"
#include "motis/loader/wzr_loader.h"

using namespace motis::logging;
using namespace flatbuffers64;

namespace motis::loader {

char const* c_str(flatbuffers64::String const* str) {
  return str == nullptr ? nullptr : str->c_str();
}

std::string format_date(time_t const t, char const* format = "%Y-%m-%d") {
  return date::format(
      format, std::chrono::system_clock::time_point{std::chrono::seconds{t}});
}

graph_builder::graph_builder(schedule& sched, Interval const* schedule_interval,
                             loader_options const& opt,
                             unsigned progress_offset)
    : progress_offset_{progress_offset},
      sched_{sched},
      first_day_{
          static_cast<int>((sched.schedule_begin_ - schedule_interval->from()) /
                           (MINUTES_A_DAY * 60))},
      last_day_{
          static_cast<int>((sched.schedule_end_ - schedule_interval->from()) /
                               (MINUTES_A_DAY * 60) -
                           1)},
      apply_rules_{opt.apply_rules_},
      adjust_footpaths_{opt.adjust_footpaths_},
      expand_trips_{opt.expand_trips_} {
  utl::verify(sched.schedule_end_ > sched.schedule_begin_ &&
                  schedule_interval->from() <=
                      static_cast<uint64_t>(sched.schedule_end_) &&
                  schedule_interval->to() >=
                      static_cast<uint64_t>(sched.schedule_begin_),
              "schedule (from={}, to={}) out of interval (from={}, to={})",
              format_date(schedule_interval->from()),
              format_date(schedule_interval->to()),
              format_date(sched.schedule_begin_),
              format_date(sched.schedule_end_));
}

void graph_builder::add_dummy_node(std::string const& name) {
  auto const station_idx = sched_.station_nodes_.size();
  auto s = mcd::make_unique<station>(
      make_station(station_idx, 0.0, 0.0, 0, name, name, nullptr));
  sched_.eva_to_station_.emplace(name, s.get());
  sched_.stations_.emplace_back(std::move(s));
  sched_.station_nodes_.emplace_back(
      mcd::make_unique<station_node>(make_station_node(station_idx)));
}

void graph_builder::add_stations(Vector<Offset<Station>> const* stations) {
  // Add dummy stations.
  add_dummy_node(STATION_START);
  add_dummy_node(STATION_END);
  add_dummy_node(STATION_VIA0);
  add_dummy_node(STATION_VIA1);
  add_dummy_node(STATION_VIA2);
  add_dummy_node(STATION_VIA3);
  add_dummy_node(STATION_VIA4);
  add_dummy_node(STATION_VIA5);
  add_dummy_node(STATION_VIA6);
  add_dummy_node(STATION_VIA7);
  add_dummy_node(STATION_VIA8);
  add_dummy_node(STATION_VIA9);

  // Add schedule stations.
  auto const dummy_station_count = sched_.station_nodes_.size();
  for (auto i = flatbuffers64::uoffset_t{}; i < stations->size(); ++i) {
    auto const& input_station = stations->Get(i);
    auto const station_index = i + dummy_station_count;

    // Create station node.
    auto node_ptr = mcd::make_unique<station_node>(
        make_station_node(static_cast<unsigned int>(station_index)));
    stations_[input_station] = node_ptr.get();
    sched_.station_nodes_.emplace_back(std::move(node_ptr));

    // Create station object.
    auto s = mcd::make_unique<station>();
    s->index_ = station_index;
    s->name_ = input_station->name()->str();
    s->width_ = input_station->lat();
    s->length_ = input_station->lng();
    s->eva_nr_ = input_station->id()->str();
    s->transfer_time_ = std::max(2, input_station->interchange_time());
    s->timez_ = input_station->timezone() != nullptr
                    ? get_or_create_timezone(input_station->timezone())
                    : nullptr;
    s->equivalent_.push_back(s.get());
    sched_.eva_to_station_.emplace(input_station->id()->str(), s.get());

    // Store DS100.
    if (input_station->external_ids() != nullptr) {
      for (auto const& ds100 : *input_station->external_ids()) {
        sched_.ds100_to_station_.emplace(ds100->str(), s.get());
      }
    }

    sched_.stations_.emplace_back(std::move(s));
  }

  // First regular node id:
  // first id after station node ids
  next_node_id_ = sched_.stations_.size();
}

void graph_builder::link_meta_stations(
    Vector<Offset<MetaStation>> const* meta_stations) {
  for (auto const& meta : *meta_stations) {
    auto& station = sched_.eva_to_station_.at(meta->station()->id()->str());
    for (auto const& fbs_equivalent : *meta->equivalent()) {
      auto& equivalent = *sched_.stations_[stations_[fbs_equivalent]->id_];
      if (station->index_ != equivalent.index_) {
        station->equivalent_.push_back(&equivalent);
      }
    }
  }
}

timezone const* graph_builder::get_or_create_timezone(
    Timezone const* input_timez) {
  return utl::get_or_create(timezones_, input_timez, [&]() {
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

station_node* graph_builder::get_station_node(Station const* station) const {
  auto it = stations_.find(station);
  utl::verify(it != end(stations_), "station not found");
  return it->second;
}

full_trip_id graph_builder::get_full_trip_id(Service const* s, int day,
                                             int section_idx) {
  auto const& stops = s->route()->stations();
  auto const dep_station = stops->Get(section_idx);
  auto const arr_station = stops->Get(stops->size() - 1);
  auto const dep_station_idx = get_station_node(dep_station)->id_;
  auto const arr_station_idx = get_station_node(arr_station)->id_;

  auto const dep_tz = sched_.stations_[dep_station_idx]->timez_;
  auto const provider_first_section = s->sections()->Get(0)->provider();
  auto const dep_time = get_adjusted_event_time(
      tz_cache_, sched_.schedule_begin_, day - first_day_,
      s->times()->Get(section_idx * 2 + 1), dep_tz,
      c_str(dep_station->timezone_name()),
      provider_first_section == nullptr
          ? nullptr
          : c_str(provider_first_section->timezone_name()));

  auto const arr_tz = sched_.stations_[arr_station_idx]->timez_;
  auto const provider_last_section =
      s->sections()->Get(s->sections()->size() - 1)->provider();
  auto const arr_time = get_adjusted_event_time(
      tz_cache_, sched_.schedule_begin_, day - first_day_,
      s->times()->Get(s->times()->size() - 2), arr_tz,
      c_str(arr_station->timezone_name()),
      provider_last_section == nullptr
          ? nullptr
          : c_str(provider_last_section->timezone_name()));

  auto const train_nr = s->sections()->Get(section_idx)->train_nr();
  auto const line_id_ptr = s->sections()->Get(0)->line_id();
  auto const line_id = line_id_ptr != nullptr ? line_id_ptr->str() : "";

  full_trip_id id;
  id.primary_ = primary_trip_id{dep_station_idx,
                                static_cast<uint32_t>(train_nr), dep_time};
  id.secondary_ = secondary_trip_id{arr_station_idx, arr_time, line_id};
  return id;
}

merged_trips_idx graph_builder::create_merged_trips(Service const* s,
                                                    int day_idx) {
  return static_cast<motis::merged_trips_idx>(
      push_mem(sched_.merged_trips_,
               mcd::vector<ptr<trip>>({register_service(s, day_idx)})));
}

trip* graph_builder::register_service(Service const* s, int day_idx) {
  auto const stored =
      sched_.trip_mem_
          .emplace_back(mcd::make_unique<trip>(
              get_full_trip_id(s, day_idx), nullptr, 0U,
              s->debug() == nullptr
                  ? trip_debug{}
                  : trip_debug{utl::get_or_create(
                                   filenames_, s->debug()->file(),
                                   [&]() {
                                     return sched_.filenames_
                                         .emplace_back(
                                             mcd::make_unique<mcd::string>(
                                                 s->debug()->file()->str()))
                                         .get();
                                   }),
                               s->debug()->line_from(), s->debug()->line_to()}))
          .get();
  sched_.trips_.emplace_back(stored->id_.primary_, stored);

  if (s->trip_id() != nullptr) {
    auto const motis_time = to_motis_time(day_idx - first_day_ - 5, 0);
    auto const date = motis_to_unixtime(sched_, motis_time);
    sched_.gtfs_trip_ids_[{s->trip_id()->str(), date}] = stored;
  }

  for (auto i = 1UL; i < s->sections()->size(); ++i) {
    auto curr_section = s->sections()->Get(i);
    auto prev_section = s->sections()->Get(i - 1);

    if (curr_section->train_nr() != prev_section->train_nr()) {
      sched_.trips_.emplace_back(get_full_trip_id(s, day_idx, i).primary_,
                                 stored);
    }
  }

  if (s->initial_train_nr() != stored->id_.primary_.train_nr_) {
    auto primary = stored->id_.primary_;
    primary.train_nr_ = s->initial_train_nr();
    sched_.trips_.emplace_back(primary, stored);
  }

  return stored;
}

void graph_builder::add_services(Vector<Offset<Service>> const* services) {
  mcd::vector<Service const*> sorted(services->size());
  std::copy(std::begin(*services), std::end(*services), begin(sorted));
  std::stable_sort(begin(sorted), end(sorted),
                   [](Service const* lhs, Service const* rhs) {
                     return lhs->route() < rhs->route();
                   });

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Build Services")
      .out_bounds(progress_offset_, 90)
      .in_high(sorted.size());

  auto it = begin(sorted);
  mcd::vector<Service const*> route_services;
  while (it != end(sorted)) {
    auto route = (*it)->route();
    do {
      if (!apply_rules_ || !(*it)->rule_participant()) {
        route_services.push_back(*it);
      }
      ++it;
    } while (it != end(sorted) && route == (*it)->route());

    if (!route_services.empty()) {
      add_route_services(route_services);
    }

    route_services.clear();
    progress_tracker->update(std::distance(begin(sorted), it));
  }
}

void graph_builder::index_first_route_node(route const& r) {
  assert(!r.empty());
  auto route_index = r[0].from_route_node_->route_;
  if (static_cast<int>(sched_.route_index_to_first_route_node_.size()) <=
      route_index) {
    sched_.route_index_to_first_route_node_.resize(route_index + 1);
  }
  sched_.route_index_to_first_route_node_[route_index] = r[0].from_route_node_;
}

void graph_builder::add_route_services(
    mcd::vector<Service const*> const& services) {
  if (services.empty()) {
    return;
  }
  add_route_services(
      mcd::to_vec(begin(services), end(services), [&](Service const* service) {
        return std::make_pair(service,
                              get_or_create_bitfield(service->traffic_days()));
      }));
}

void graph_builder::add_route_services(
    mcd::vector<std::pair<Service const*, bitfield>> const& services) {
  if (services.empty()) {
    return;
  }

  mcd::vector<route_lcs> alt_routes;
  for (auto const& [s, traffic_days] : services) {
    auto const first_day_offset =
        s->times()->Get(s->times()->size() - 2) / 1440;
    auto const first_day = std::max(0, first_day_ - first_day_offset);

    for (int day = first_day; day <= last_day_; ++day) {
      if (!traffic_days.test(day)) {
        continue;
      }

      time prev_arr = 0;
      bool adjusted = false;
      mcd::vector<light_connection> lcons;
      auto t = create_merged_trips(s, day);
      for (unsigned section_idx = 0; section_idx < s->sections()->size();
           ++section_idx) {
        lcons.push_back(section_to_connection(
            t, {{participant{s, section_idx}}}, day, prev_arr, adjusted));
        prev_arr = lcons.back().a_time_;
      }

      add_to_routes(alt_routes, lcons);
    }
  }

  for (auto const& route : alt_routes) {
    if (route.empty() || route[0].empty()) {
      continue;
    }

    auto const route_id = next_route_index_++;
    auto r = create_route(services[0].first->route(), route, route_id);
    index_first_route_node(*r);
    write_trip_info(*r);
    if (expand_trips_) {
      add_expanded_trips(*r);
    }
  }
}

void graph_builder::add_expanded_trips(route const& r) {
  assert(!r.empty());
  auto trips_added = false;
  auto const& re = r.front().get_route_edge();
  if (re != nullptr) {
    for (auto const& lc : re->m_.route_edge_.conns_) {
      auto const& merged_trips = sched_.merged_trips_[lc.trips_];
      assert(merged_trips != nullptr);
      assert(merged_trips->size() == 1);
      auto const trp = merged_trips->front();
      if (check_trip(trp)) {
        sched_.expanded_trips_.push_back(trp);
        trips_added = true;
      }
    }
  }
  if (trips_added) {
    sched_.expanded_trips_.finish_key();
  }
}

bool graph_builder::check_trip(trip const* trp) {
  auto last_time = 0U;
  for (auto const& section : motis::access::sections(trp)) {
    auto const& lc = section.lcon();
    if (lc.d_time_ > lc.a_time_ || last_time > lc.d_time_) {
      ++broken_trips_;
      return false;
    }
    last_time = lc.a_time_;
  }
  return true;
}

int graph_builder::get_index(
    mcd::vector<mcd::vector<light_connection>> const& alt_route,
    mcd::vector<light_connection> const& sections) {
  assert(!sections.empty());
  assert(!alt_route.empty());

  if (alt_route[0].empty()) {
    return 0;
  }

  int index = -1;
  for (auto section_idx = 0UL; section_idx < sections.size(); ++section_idx) {
    auto const& route_section = alt_route[section_idx];
    auto const& lc = sections[section_idx];
    if (index == -1) {
      index = std::distance(
          begin(route_section),
          std::lower_bound(begin(route_section), end(route_section),
                           sections[section_idx]));
      --section_idx;
    } else {
      // Check if departures stay sorted.
      bool earlier_eq_dep =
          index > 0 && lc.d_time_ <= route_section[index - 1].d_time_;
      bool later_eq_dep = static_cast<unsigned>(index) < route_section.size() &&
                          lc.d_time_ >= route_section[index].d_time_;

      // Check if arrivals stay sorted.
      bool earlier_eq_arr =
          index > 0 && lc.a_time_ <= route_section[index - 1].a_time_;
      bool later_eq_arr = static_cast<unsigned>(index) < route_section.size() &&
                          lc.a_time_ >= route_section[index].a_time_;

      if (earlier_eq_dep || later_eq_dep || earlier_eq_arr || later_eq_arr) {
        return -1;
      }
    }
  }
  return index;
}

void graph_builder::add_to_route(
    mcd::vector<mcd::vector<light_connection>>& route,
    mcd::vector<light_connection> const& sections, int index) {
  for (auto section_idx = 0UL; section_idx < sections.size(); ++section_idx) {
    route[section_idx].insert(std::next(begin(route[section_idx]), index),
                              sections[section_idx]);
  }
}

void graph_builder::add_to_routes(
    mcd::vector<mcd::vector<mcd::vector<light_connection>>>& alt_routes,
    mcd::vector<light_connection> const& sections) {
  for (auto& alt_route : alt_routes) {
    int index = get_index(alt_route, sections);
    if (index == -1) {
      continue;
    }

    add_to_route(alt_route, sections, index);
    return;
  }

  alt_routes.emplace_back(sections.size());
  add_to_route(alt_routes.back(), sections, 0);
}

connection_info* graph_builder::get_or_create_connection_info(
    Section const* section, int dep_day_index, connection_info* merged_with) {
  con_info_.line_identifier_ =
      section->line_id() != nullptr ? section->line_id()->str() : "";
  con_info_.train_nr_ = section->train_nr();
  con_info_.family_ = get_or_create_category_index(section->category());
  con_info_.dir_ = get_or_create_direction(section->direction());
  con_info_.provider_ = get_or_create_provider(section->provider());
  con_info_.merged_with_ = merged_with;
  read_attributes(dep_day_index, section->attributes(), con_info_.attributes_);

  return mcd::set_get_or_create(con_infos_, &con_info_, [&]() {
    sched_.connection_infos_.emplace_back(
        mcd::make_unique<connection_info>(con_info_));
    return sched_.connection_infos_.back().get();
  });
}

connection_info* graph_builder::get_or_create_connection_info(
    std::array<participant, 16> const& services, int dep_day_index) {
  connection_info* prev_con_info = nullptr;

  for (auto service : services) {
    if (service.service_ == nullptr) {
      return prev_con_info;
    }

    auto const& s = service;
    prev_con_info = get_or_create_connection_info(
        s.service_->sections()->Get(s.section_idx_), dep_day_index,
        prev_con_info);
  }

  return prev_con_info;
}

light_connection graph_builder::section_to_connection(
    merged_trips_idx trips, std::array<participant, 16> const& services,
    int day, time prev_arr, bool& adjusted) {
  auto const& ref = services[0].service_;
  auto const& section_idx = services[0].section_idx_;

  assert(ref != nullptr);

  assert(std::all_of(begin(services), end(services), [&](participant const& s) {
    if (s.service_ == nullptr) {
      return true;
    }

    auto const& ref_stops = ref->route()->stations();
    auto const& s_stops = s.service_->route()->stations();
    auto stations_match =
        s_stops->Get(s.section_idx_) == ref_stops->Get(section_idx) &&
        s_stops->Get(s.section_idx_ + 1) == ref_stops->Get(section_idx + 1);

    auto times_match =
        s.service_->times()->Get(s.section_idx_ * 2 + 1) % 1440 ==
            ref->times()->Get(section_idx * 2 + 1) % 1440 &&
        s.service_->times()->Get(s.section_idx_ * 2 + 2) % 1440 ==
            ref->times()->Get(section_idx * 2 + 2) % 1440;

    return stations_match && times_match;
  }));
  assert(std::is_sorted(begin(services), end(services)));

  auto const from_station = ref->route()->stations()->Get(section_idx);
  auto const to_station = ref->route()->stations()->Get(section_idx + 1);
  auto& from = *sched_.stations_.at(stations_[from_station]->id_);
  auto& to = *sched_.stations_.at(stations_[to_station]->id_);

  auto plfs = ref->tracks();
  auto dep_platf =
      plfs != nullptr ? plfs->Get(section_idx)->dep_tracks() : nullptr;
  auto arr_platf =
      plfs != nullptr ? plfs->Get(section_idx + 1)->arr_tracks() : nullptr;

  auto section = ref->sections()->Get(section_idx);
  auto dep_time = ref->times()->Get(section_idx * 2 + 1);
  auto arr_time = ref->times()->Get(section_idx * 2 + 2);

  // Day indices for shifted bitfields (tracks, attributes)
  int dep_day_index = day + (dep_time / MINUTES_A_DAY);
  int arr_day_index = day + (arr_time / MINUTES_A_DAY);

  // Build full connection.
  auto clasz_it = sched_.classes_.find(section->category()->name()->str());
  con_.clasz_ = (clasz_it == end(sched_.classes_)) ? service_class::OTHER
                                                   : clasz_it->second;
  con_.price_ = get_distance(from, to) * get_price_per_km(con_.clasz_);
  con_.d_track_ = get_or_create_track(dep_day_index, dep_platf);
  con_.a_track_ = get_or_create_track(arr_day_index, arr_platf);
  con_.con_info_ = get_or_create_connection_info(services, dep_day_index);

  // Build light connection.
  time dep_motis_time = 0, arr_motis_time = 0;
  auto const section_timezone = section->provider() == nullptr
                                    ? nullptr
                                    : section->provider()->timezone_name();
  std::tie(dep_motis_time, arr_motis_time) = get_event_times(
      tz_cache_, sched_.schedule_begin_, day - first_day_, prev_arr, dep_time,
      arr_time, from.timez_, c_str(from_station->timezone_name()),
      c_str(section_timezone), to.timez_, c_str(to_station->timezone_name()),
      c_str(section_timezone), adjusted);

  // Count events.
  ++from.dep_class_events_.at(static_cast<service_class_t>(con_.clasz_));
  ++to.arr_class_events_.at(static_cast<service_class_t>(con_.clasz_));

  // Track first event.
  sched_.first_event_schedule_time_ = std::min(
      sched_.first_event_schedule_time_,
      motis_to_unixtime(sched_, dep_motis_time) - SCHEDULE_OFFSET_MINUTES * 60);
  sched_.last_event_schedule_time_ = std::max(
      sched_.last_event_schedule_time_,
      motis_to_unixtime(sched_, arr_motis_time) - SCHEDULE_OFFSET_MINUTES * 60);

  return light_connection(
      dep_motis_time, arr_motis_time,
      mcd::set_get_or_create(connections_, &con_,
                             [&]() {
                               sched_.full_connections_.emplace_back(
                                   mcd::make_unique<connection>(con_));
                               return sched_.full_connections_.back().get();
                             }),
      trips);
}

void graph_builder::add_footpaths(Vector<Offset<Footpath>> const* footpaths) {
  auto skipped = 0U;
  for (auto const& footpath : *footpaths) {
    auto const from_node_it = stations_.find(footpath->from());
    auto const to_node_it = stations_.find(footpath->to());

    utl::verify(
        from_node_it != end(stations_), "footpath from node not found {} [{}]",
        footpath->from()->name()->c_str(), footpath->from()->id()->c_str());
    utl::verify(to_node_it != end(stations_),
                "footpath to node not found {} [{}]",
                footpath->to()->name()->c_str(), footpath->to()->id()->c_str());

    auto duration = footpath->duration();
    auto const from_node = from_node_it->second;
    auto const to_node = to_node_it->second;
    auto const& from_station = *sched_.stations_.at(from_node->id_);
    auto const& to_station = *sched_.stations_.at(to_node->id_);

    if (from_node == to_node) {
      LOG(warn) << "Footpath loop at station " << from_station.eva_nr_
                << " ignored";
      continue;
    }

    if (adjust_footpaths_) {
      uint32_t max_transfer_time =
          std::max(from_station.transfer_time_, to_station.transfer_time_);

      duration = std::max(max_transfer_time, footpath->duration());

      auto const distance = get_distance(from_station, to_station) * 1000;
      auto const max_distance_adjust = duration * 60 * WALK_SPEED;
      auto const max_distance = 2 * duration * 60 * WALK_SPEED;

      if (distance > max_distance) {
        continue;
      } else if (distance > max_distance_adjust) {
        duration = std::round(distance / (60 * WALK_SPEED));
      }

      if (duration > 15) {
        continue;
      }
    }

    next_node_id_ = from_node->add_foot_edge(
        next_node_id_, make_fwd_edge(from_node, to_node, duration));
  }
  LOG(info) << "Skipped " << skipped
            << " footpaths connecting stations with no events";
}

void graph_builder::connect_reverse() {
  for (auto& station_node : sched_.station_nodes_) {
    for (auto& station_edge : station_node->edges_) {
      station_edge.to_->incoming_edges_.push_back(&station_edge);
      if (station_edge.to_->get_station() != station_node.get()) {
        continue;
      }
      for (auto& edge : station_edge.to_->edges_) {
        edge.to_->incoming_edges_.push_back(&edge);
      }
    }
  }
}

void graph_builder::sort_trips() {
  std::sort(
      begin(sched_.trips_), end(sched_.trips_),
      [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
}

bitfield const& graph_builder::get_or_create_bitfield(
    String const* serialized_bitfield) {
  return utl::get_or_create(bitfields_, serialized_bitfield, [&]() {
    return deserialize_bitset<BIT_COUNT>(
        {serialized_bitfield->c_str(),
         static_cast<size_t>(serialized_bitfield->Length())});
  });
}

void graph_builder::read_attributes(
    int day, Vector<Offset<Attribute>> const* attributes,
    mcd::vector<ptr<attribute const>>& active_attributes) {
  active_attributes.clear();
  active_attributes.reserve(attributes->size());
  for (auto const& attr : *attributes) {
    if (!get_or_create_bitfield(attr->traffic_days()).test(day)) {
      continue;
    }
    auto const attribute_info = attr->info();
    active_attributes.push_back(
        utl::get_or_create(attributes_, attribute_info, [&]() {
          auto new_attr = mcd::make_unique<attribute>();
          new_attr->code_ = attribute_info->code()->str();
          new_attr->text_ = attribute_info->text()->str();
          sched_.attributes_.emplace_back(std::move(new_attr));
          return sched_.attributes_.back().get();
        }));
  }
}

mcd::string const* graph_builder::get_or_create_direction(
    Direction const* dir) {
  if (dir == nullptr) {
    return nullptr;
  } else if (dir->station() != nullptr) {
    return &sched_.stations_[stations_[dir->station()]->id_]->name_;
  } else /* direction text */ {
    return utl::get_or_create(directions_, dir->text(), [&]() {
      sched_.directions_.emplace_back(
          mcd::make_unique<mcd::string>(dir->text()->str()));
      return sched_.directions_.back().get();
    });
  }
}

provider const* graph_builder::get_or_create_provider(Provider const* p) {
  if (p == nullptr) {
    return nullptr;
  } else {
    return utl::get_or_create(providers_, p, [&]() {
      sched_.providers_.emplace_back(mcd::make_unique<provider>(
          p->short_name()->str(), p->long_name()->str(),
          p->full_name()->str()));
      return sched_.providers_.back().get();
    });
  }
}

int graph_builder::get_or_create_category_index(Category const* c) {
  return utl::get_or_create(categories_, c, [&]() {
    int index = sched_.categories_.size();
    sched_.categories_.emplace_back(mcd::make_unique<category>(
        c->name()->str(), static_cast<uint8_t>(c->output_rule())));
    return index;
  });
}

int graph_builder::get_or_create_track(int day,
                                       Vector<Offset<Track>> const* tracks) {
  static constexpr int no_track = 0;
  if (sched_.tracks_.empty()) {
    sched_.tracks_.emplace_back("");
  }

  if (tracks == nullptr) {
    return no_track;
  }

  auto track_it = std::find_if(
      std::begin(*tracks), std::end(*tracks), [&](Track const* track) {
        return get_or_create_bitfield(track->bitfield()).test(day);
      });
  if (track_it == std::end(*tracks)) {
    return no_track;
  } else {
    auto name = track_it->name()->str();
    return utl::get_or_create(tracks_, name, [&]() {
      int index = sched_.tracks_.size();
      sched_.tracks_.emplace_back(name);
      return index;
    });
  }
}

void graph_builder::write_trip_info(route& r) {
  auto const edges = mcd::to_vec(begin(r), end(r), [](route_section& s) {
    return trip::route_edge(s.get_route_edge());
  });

  sched_.trip_edges_.emplace_back(
      mcd::make_unique<mcd::vector<trip::route_edge>>(edges));
  auto edges_ptr = sched_.trip_edges_.back().get();

  auto& lcons = edges_ptr->front().get_edge()->m_.route_edge_.conns_;
  for (auto lcon_idx = lcon_idx_t{}; lcon_idx < lcons.size(); ++lcon_idx) {
    auto trp = sched_.merged_trips_[lcons[lcon_idx].trips_]->front();
    trp->edges_ = edges_ptr;
    trp->lcon_idx_ = lcon_idx;
  }
}

mcd::unique_ptr<route> graph_builder::create_route(Route const* r,
                                                   route_lcs const& lcons,
                                                   unsigned route_index) {
  auto const& stops = r->stations();
  auto const& in_allowed = r->in_allowed();
  auto const& out_allowed = r->out_allowed();
  auto route_sections = mcd::make_unique<route>();

  route_section last_route_section;
  for (auto i = 0UL; i < r->stations()->size() - 1; ++i) {
    auto from = i;
    auto to = i + 1;
    route_sections->push_back(
        add_route_section(route_index, lcons[i],  //
                          stops->Get(from), in_allowed->Get(from) != 0U,
                          out_allowed->Get(from) != 0U, stops->Get(to),
                          in_allowed->Get(to) != 0U, out_allowed->Get(to) != 0U,
                          last_route_section.to_route_node_, nullptr));
    last_route_section = route_sections->back();
  }

  return route_sections;
}

route_section graph_builder::add_route_section(
    int route_index, mcd::vector<light_connection> const& connections,
    Station const* from_stop, bool from_in_allowed, bool from_out_allowed,
    Station const* to_stop, bool to_in_allowed, bool to_out_allowed,
    node* from_route_node, node* to_route_node) {
  route_section section;

  auto const from_station_node = stations_[from_stop];
  auto const to_station_node = stations_[to_stop];

  if (from_route_node != nullptr) {
    section.from_route_node_ = from_route_node;
  } else {
    section.from_route_node_ = build_route_node(
        route_index, next_node_id_++, from_station_node,
        sched_.stations_[from_station_node->id_]->transfer_time_,
        from_in_allowed, from_out_allowed);
  }

  if (to_route_node != nullptr) {
    section.to_route_node_ = to_route_node;
  } else {
    section.to_route_node_ =
        build_route_node(route_index, next_node_id_++, to_station_node,
                         sched_.stations_[to_station_node->id_]->transfer_time_,
                         to_in_allowed, to_out_allowed);
  }

  section.outgoing_route_edge_index_ = section.from_route_node_->edges_.size();
  section.from_route_node_->edges_.push_back(make_route_edge(
      section.from_route_node_, section.to_route_node_, connections));
  lcon_count_ += connections.size();

  return section;
}

schedule_ptr build_graph(Schedule const* serialized, loader_options const& opt,
                         unsigned progress_offset) {
  scoped_timer timer("building graph");

  LOG(info) << "schedule: " << serialized->name()->str();

  schedule_ptr sched(new schedule());
  sched->classes_ = class_mapping();
  std::tie(sched->schedule_begin_, sched->schedule_end_) = opt.interval();

  if (serialized->name() != nullptr) {
    sched->name_ = serialized->name()->str();
  }

  auto progress_tracker = utl::get_active_progress_tracker();

  progress_tracker->status("Build Graph").out_bounds(progress_offset, 90);
  graph_builder builder(*sched, serialized->interval(), opt, progress_offset);
  builder.add_stations(serialized->stations());
  if (serialized->meta_stations() != nullptr) {
    builder.link_meta_stations(serialized->meta_stations());
  }

  builder.add_services(serialized->services());
  if (opt.apply_rules_) {
    scoped_timer timer("rule services");
    progress_tracker->status("Rule Services").out_bounds(90, 92);
    build_rule_routes(builder, serialized->rule_services());
  }

  if (opt.expand_trips_) {
    sched->expanded_trips_.finish_map();
  }

  progress_tracker->status("Footpaths").out_bounds(92, 93);
  builder.add_footpaths(serialized->footpaths());

  progress_tracker->status("Connect Reverse").out_bounds(93, 94);
  builder.connect_reverse();

  progress_tracker->status("Sort Trips").out_bounds(94, 95);
  builder.sort_trips();

  if (opt.expand_footpaths_) {
    progress_tracker->status("Expand Footpaths").out_bounds(95, 96);
    calc_footpaths(*sched);
  }

  sched->hash_ = serialized->hash();
  sched->route_count_ = builder.next_route_index_;
  sched->node_count_ = builder.next_node_id_;

  progress_tracker->status("Lower Bounds").out_bounds(96, 100).in_high(4);
  sched->transfers_lower_bounds_fwd_ = build_interchange_graph(
      sched->station_nodes_, sched->route_count_, search_dir::FWD);
  progress_tracker->increment();
  sched->transfers_lower_bounds_bwd_ = build_interchange_graph(
      sched->station_nodes_, sched->route_count_, search_dir::BWD);
  progress_tracker->increment();
  sched->travel_time_lower_bounds_fwd_ =
      build_station_graph(sched->station_nodes_, search_dir::FWD);
  progress_tracker->increment();
  sched->travel_time_lower_bounds_bwd_ =
      build_station_graph(sched->station_nodes_, search_dir::BWD);
  progress_tracker->increment();

  sched->waiting_time_rules_ = load_waiting_time_rules(
      opt.wzr_classes_path_, opt.wzr_matrix_path_, sched->categories_);
  sched->schedule_begin_ -= SCHEDULE_OFFSET_MINUTES * 60;
  calc_waits_for(*sched, opt.planned_transfer_delta_);

  LOG(info) << sched->connection_infos_.size() << " connection infos";
  LOG(info) << builder.lcon_count_ << " light connections";
  LOG(info) << builder.next_route_index_ << " routes";
  LOG(info) << sched->trip_mem_.size() << " trips";
  if (opt.expand_trips_) {
    LOG(info) << sched->expanded_trips_.index_size() - 1 << " expanded routes";
    LOG(info) << sched->expanded_trips_.data_size() << " expanded trips";
    LOG(info) << builder.broken_trips_ << " broken trips ignored";
  }

  utl::verify(
      std::all_of(begin(sched->trips_), end(sched->trips_),
                  [](auto const& t) { return t.second->edges_ != nullptr; }),
      "missing trip edges");
  return sched;
}

}  // namespace motis::loader

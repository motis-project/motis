#include "motis/loader/graph_builder.h"

#include <cassert>
#include <algorithm>
#include <functional>
#include <numeric>

#include "boost/uuid/nil_generator.hpp"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "date/date.h"

#include "utl/parser/cstr.h"

#include "motis/core/common/constants.h"
#include "motis/core/common/logging.h"
#include "motis/core/schedule/build_platform_node.h"
#include "motis/core/schedule/build_route_node.h"
#include "motis/core/schedule/category.h"
#include "motis/core/schedule/price.h"
#include "motis/core/schedule/validate_graph.h"
#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/debug/route_graph.h"
#include "motis/core/debug/trip.h"

#include "motis/loader/build_footpaths.h"
#include "motis/loader/build_graph.h"
#include "motis/loader/build_stations.h"
#include "motis/loader/classes.h"
#include "motis/loader/filter/local_stations.h"
#include "motis/loader/interval_util.h"
#include "motis/loader/rule_route_builder.h"
#include "motis/loader/rule_service_graph_builder.h"
#include "motis/loader/util.h"
#include "motis/loader/wzr_loader.h"

using namespace motis::logging;
using namespace flatbuffers64;

namespace motis::loader {

char const* c_str(flatbuffers64::String const* str) {
  return str == nullptr ? nullptr : str->c_str();
}

graph_builder::graph_builder(schedule& sched, loader_options const& opt)
    : sched_{sched},
      apply_rules_{opt.apply_rules_},
      no_local_transport_{opt.no_local_transport_},
      debug_broken_trips_{opt.debug_broken_trips_} {}

full_trip_id graph_builder::get_full_trip_id(Service const* s, int day,
                                             int section_idx) {
  auto const& stops = s->route()->stations();
  auto const dep_station = stops->Get(section_idx);
  auto const arr_station = stops->Get(stops->size() - 1);
  auto const dep_station_idx = stations_.at(dep_station)->id_;
  auto const arr_station_idx = stations_.at(arr_station)->id_;

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
              get_full_trip_id(s, day_idx),
              s->trip_id() == nullptr ? "" : s->trip_id()->str(), nullptr, 0U,
              static_cast<trip_idx_t>(sched_.trip_mem_.size()),
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
                               s->debug()->line_from(), s->debug()->line_to()},
              s->seq_numbers() == nullptr ? mcd::vector<uint32_t>{}
                                          : mcd::to_vec(*s->seq_numbers()),
              boost::uuids::nil_uuid(),
              s->schedule_relationship() == ScheduleRelationship_UNSCHEDULED))
          .get();
  sched_.trips_.emplace_back(stored->id_.primary_, stored);

  if (s->trip_id() != nullptr) {
    auto const motis_time = to_motis_time(day_idx - first_day_ - 5, 0);
    auto const date = motis_to_unixtime(sched_, motis_time);
    sched_.gtfs_trip_ids_[dataset_prefix_ + s->trip_id()->str()].emplace_back(
        date, stored);
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
  progress_tracker->in_high(sorted.size());

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

    if (!route_services.empty() && !skip_route(route)) {
      add_route_services(mcd::to_vec(route_services, [&](Service const* s) {
        return std::make_pair(s, get_or_create_bitfield(s->traffic_days()));
      }));
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
    mcd::vector<std::pair<Service const*, bitfield>> const& services) {
  mcd::vector<route_lcs> alt_routes;
  for (auto const& [s, traffic_days] : services) {
    auto const first_day_offset =
        s->times()->Get(s->times()->size() - 2) / 1440;
    auto const first_day = std::max(0, first_day_ - first_day_offset);

    auto const stations =
        mcd::to_vec(*s->route()->stations(), [&](auto const& s) {
          return sched_.stations_[stations_.find(s)->second->id_].get();
        });

    for (int day = first_day; day <= last_day_; ++day) {
      if (day >= traffic_days.size() || !traffic_days.test(day)) {
        continue;
      }

      time prev_arr = 0;
      bool adjusted = false;
      mcd::vector<light_connection> lcons;
      auto const merged_trips_idx = sched_.merged_trips_.size();
      for (unsigned section_idx = 0; section_idx < s->sections()->size();
           ++section_idx) {
        lcons.push_back(section_to_connection(merged_trips_idx,
                                              {{participant{s, section_idx}}},
                                              day, prev_arr, adjusted));
        prev_arr = lcons.back().a_time_;
      }

      if (has_duplicate(s, lcons)) {
        continue;
      }

      utl::verify(merged_trips_idx == create_merged_trips(s, day),
                  "unexpected merged_trips_idx");
      add_to_routes(alt_routes, lcons, stations);
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
    add_expanded_trips(*r);
  }
}

bool graph_builder::has_duplicate(Service const* service,
                                  mcd::vector<light_connection> const& lcons) {
  auto const& first_station = sched_.stations_.at(
      stations_.at(service->route()->stations()->Get(0))->id_);
  for (auto const& eq : first_station->equivalent_) {
    if (eq->source_schedule_ == first_station->source_schedule_) {
      continue;  // Ignore duplicates from same schedule.
    }

    for (auto const& route_node :
         sched_.station_nodes_[eq->index_]->child_nodes_) {
      if (!route_node->is_route_node()) {
        continue;
      }
      for (auto const& route_edge : route_node->edges_) {
        if (route_edge.type() != edge::ROUTE_EDGE) {
          continue;
        }

        for (auto* lc = route_edge.get_connection(lcons.front().d_time_);
             lc != nullptr && lc->d_time_ == lcons.front().d_time_;
             lc = route_edge.get_next_valid_lcon(lc, 1U)) {
          for (auto const& trp : *sched_.merged_trips_[lc->trips_]) {
            if (are_duplicates(service, lcons, trp)) {
              return true;
            }
          }
        }
      }
    }
  }

  return false;
}

bool graph_builder::are_duplicates(Service const* service_a,
                                   mcd::vector<light_connection> const& lcons_a,
                                   trip const* trp_b) {
  auto const* stations_a = service_a->route()->stations();
  auto const stops_b = access::stops{trp_b};
  auto const stop_count_b = std::distance(begin(stops_b), end(stops_b));

  if (stations_a->size() != stop_count_b) {
    return false;
  }

  auto const are_equivalent = [&](Station const* st_a, station const& s_b) {
    auto const& s_a = sched_.stations_.at(stations_.at(st_a)->id_);
    return s_a->source_schedule_ != s_b.source_schedule_ &&
           std::any_of(
               begin(s_a->equivalent_), end(s_a->equivalent_),
               [&](auto const& eq_a) { return eq_a->index_ == s_b.index_; });
  };

  auto const& last_stop_b = *std::next(begin(stops_b), stop_count_b - 1);
  if (lcons_a.back().a_time_ != last_stop_b.arr_lcon().a_time_ ||
      !are_equivalent(stations_a->Get(stations_a->size() - 1),
                      last_stop_b.get_station(sched_))) {
    return false;
  }

  for (auto [i_a, it_b] = std::tuple{1ULL, std::next(begin(stops_b))};
       std::next(it_b) != end(stops_b); ++i_a, ++it_b) {
    if (lcons_a[i_a - 1].a_time_ != (*it_b).arr_lcon().a_time_ ||
        lcons_a[i_a].d_time_ != (*it_b).dep_lcon().d_time_ ||
        !are_equivalent(stations_a->Get(i_a), (*it_b).get_station(sched_))) {
      return false;
    }
  }

  return true;
}

void graph_builder::add_expanded_trips(route const& r) {
  assert(!r.empty());
  auto const& re = r.front().get_route_edge();
  if (re != nullptr) {
    auto const route_id = re->from_->route_;
    auto route_trips = sched_.expanded_trips_.emplace_back();
    for (auto const& lc : re->m_.route_edge_.conns_) {
      auto const& merged_trips = sched_.merged_trips_[lc.trips_];
      assert(merged_trips != nullptr);
      assert(merged_trips->size() == 1);
      auto const trp = merged_trips->front();
      if (check_trip(trp)) {
        route_trips.push_back(trp);
      }
    }
    if (!route_trips.empty()) {
      sched_.route_to_expanded_routes_[route_id].emplace_back(
          route_trips.index());
    }
  }
}

bool graph_builder::check_trip(trip const* trp) {
  if (trp->edges_->empty()) {
    LOG(info) << "broken trip: " << debug::trip{sched_, trp} << ": no edges";
    return false;
  }
  time last_time = 0;
  for (auto const sec : access::sections{trp}) {
    auto const& lc = sec.lcon();
    auto const section_times_ok = lc.d_time_ <= lc.a_time_;
    auto const stop_times_ok = last_time <= lc.d_time_;
    if (!section_times_ok || !stop_times_ok) {
      if (debug_broken_trips_) {
        LOG(info) << "broken trip: " << debug::trip{sched_, trp}
                  << ": lc={dep=" << format_time(lc.d_time_)
                  << ", arr=" << format_time(lc.a_time_)
                  << "}, last_time=" << format_time(last_time) << ":\n"
                  << debug::rule_service_trip_with_sections{sched_, trp}
                  << "\nroute graph:\n\n"
                  << debug::route_graph{sched_, trp} << "\n";
      }
      ++broken_trips_;
      return false;
    }
    last_time = lc.a_time_;
  }
  return true;
}

int graph_builder::get_index(
    mcd::vector<mcd::vector<light_connection>> const& alt_route,
    mcd::vector<light_connection> const& sections,
    mcd::vector<station*> const& stations) {
  assert(!sections.empty());
  assert(!alt_route.empty());
  assert(stations.size() == sections.size() + 1);

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

      // Check if both tracks have the same platform or both tracks have no
      // platform information.
      auto const& dep_station = stations[section_idx];
      auto const& arr_station = stations[section_idx + 1];
      auto const different_dep_platform =
          dep_station->get_platform(lc.full_con_->d_track_) !=
          dep_station->get_platform(route_section[0].full_con_->d_track_);
      auto const different_arr_platform =
          arr_station->get_platform(lc.full_con_->a_track_) !=
          arr_station->get_platform(route_section[0].full_con_->a_track_);

      if (earlier_eq_dep || later_eq_dep || earlier_eq_arr || later_eq_arr ||
          different_dep_platform || different_arr_platform) {
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
    mcd::vector<light_connection> const& sections,
    mcd::vector<station*> const& stations) {
  for (auto& alt_route : alt_routes) {
    int index = get_index(alt_route, sections, stations);
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
  std::optional<std::string> error;
  std::tie(dep_motis_time, arr_motis_time, error) = get_event_times(
      tz_cache_, sched_.schedule_begin_, day - first_day_, prev_arr, dep_time,
      arr_time, from.timez_, c_str(from_station->timezone_name()),
      c_str(section_timezone), to.timez_, c_str(to_station->timezone_name()),
      c_str(section_timezone), adjusted);
  if (error.has_value()) {
    LOG(logging::error) << "invalid times: " << *error
                        << ", section_idx=" << section_idx << ", trip="
                        << (ref->trip_id() != nullptr ? ref->trip_id()->str()
                                                      : "unkown");
  }

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

  return {dep_motis_time, arr_motis_time,
          mcd::set_get_or_create(connections_, &con_,
                                 [&]() {
                                   sched_.full_connections_.emplace_back(
                                       mcd::make_unique<connection>(con_));
                                   return sched_.full_connections_.back().get();
                                 }),
          trips};
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
    for (auto& platform_node : station_node->platform_nodes_) {
      if (platform_node != nullptr) {
        for (auto& edge : platform_node->edges_) {
          edge.to_->incoming_edges_.push_back(&edge);
        }
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

void graph_builder::write_trip_info(route const& r) {
  auto edges_ptr =
      sched_.trip_edges_
          .emplace_back(mcd::make_unique<mcd::vector<trip::route_edge>>(
              mcd::to_vec(r,
                          [](route_section const& s) {
                            return trip::route_edge{s.get_route_edge()};
                          })))
          .get();

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
  auto const from_station = sched_.stations_[from_station_node->id_].get();
  auto const to_station = sched_.stations_[to_station_node->id_].get();

  if (from_route_node != nullptr) {
    section.from_route_node_ = from_route_node;
  } else {
    section.from_route_node_ = build_route_node(
        route_index, sched_.next_node_id_++, from_station_node,
        from_station->transfer_time_, from_in_allowed, from_out_allowed);
  }
  auto const from_platform =
      from_station->get_platform(connections[0].full_con_->d_track_);
  if (from_in_allowed && from_platform) {
    add_platform_enter_edge(sched_, section.from_route_node_, from_station_node,
                            from_station->platform_transfer_time_,
                            from_platform.value());
  }

  if (to_route_node != nullptr) {
    section.to_route_node_ = to_route_node;
  } else {
    section.to_route_node_ = build_route_node(
        route_index, sched_.next_node_id_++, to_station_node,
        to_station->transfer_time_, to_in_allowed, to_out_allowed);
  }
  auto const to_platform =
      to_station->get_platform(connections[0].full_con_->a_track_);
  if (to_out_allowed && to_platform) {
    add_platform_exit_edge(sched_, section.to_route_node_, to_station_node,
                           to_station->platform_transfer_time_,
                           to_platform.value());
  }

  section.outgoing_route_edge_index_ = section.from_route_node_->edges_.size();
  section.from_route_node_->edges_.push_back(make_route_edge(
      section.from_route_node_, section.to_route_node_, connections));
  lcon_count_ += connections.size();

  return section;
}

bool graph_builder::skip_station(Station const* station) const {
  return no_local_transport_ && is_local_station(station);
}

bool graph_builder::skip_route(Route const* route) const {
  return no_local_transport_ &&
         std::any_of(
             route->stations()->begin(), route->stations()->end(),
             [&](Station const* station) { return skip_station(station); });
}

schedule_ptr build_graph(std::vector<Schedule const*> const& fbs_schedules,
                         loader_options const& opt) {
  utl::verify(!fbs_schedules.empty(), "build_graph: no schedules");

  scoped_timer timer("building graph");
  for (auto const* fbs_schedule : fbs_schedules) {
    LOG(info) << "schedule: " << fbs_schedule->name()->str();
  }

  auto sched = mcd::make_unique<schedule>();
  sched->classes_ = class_mapping();
  std::tie(sched->schedule_begin_, sched->schedule_end_) = opt.interval();

  for (auto const& [index, fbs_schedule] : utl::enumerate(fbs_schedules)) {
    sched->names_.push_back(
        fbs_schedule->name() != nullptr
            ? fbs_schedule->name()->str()
            : std::string{"unknown-"}.append(std::to_string(index)));
  }

  if (fbs_schedules.size() == 1 && opt.dataset_prefix_.empty()) {
    sched->prefixes_.emplace_back();  // dont force prefix for single
  } else {
    utl::verify(std::set<std::string>{begin(opt.dataset_prefix_),
                                      end(opt.dataset_prefix_)}
                        .size() == fbs_schedules.size(),
                "graph_builder: some prefixes are missing or non-unique");
    sched->prefixes_ = mcd::to_vec(
        opt.dataset_prefix_,
        [](auto const& s) -> mcd::string { return s.empty() ? s : s + "_"; });
  }

  auto progress_tracker = utl::get_active_progress_tracker();
  graph_builder builder{*sched, opt};

  progress_tracker->status("Add Stations").out_bounds(0, 5);
  builder.stations_ =
      build_stations(*sched, fbs_schedules, builder.tracks_, opt.use_platforms_,
                     opt.no_local_transport_);

  for (auto const& [i, fbs_schedule] : utl::enumerate(fbs_schedules)) {
    auto const dataset_prefix =
        opt.dataset_prefix_.empty() ? "" : opt.dataset_prefix_[i];
    auto const out_low = 5.F + (80.F / fbs_schedules.size()) * i;
    auto const out_high = 5.F + (80.F / fbs_schedules.size()) * (i + 1);
    auto const out_mid = out_low + (out_high - out_low) * .9F;
    progress_tracker->status(fmt::format("Add Services {}", dataset_prefix))
        .out_bounds(out_low, out_mid);

    builder.dataset_prefix_ =
        dataset_prefix.empty() ? dataset_prefix : dataset_prefix + "_";

    std::tie(builder.first_day_, builder.last_day_) =
        first_last_days(*sched, i, fbs_schedule->interval());
    builder.add_services(fbs_schedule->services());
    if (opt.apply_rules_) {
      scoped_timer timer("rule services");
      progress_tracker->status(fmt::format("Rule Services {}", dataset_prefix))
          .out_bounds(out_mid, out_high);
      build_rule_routes(builder, fbs_schedule->rule_services());
    }
  }

  progress_tracker->status("Footpaths").out_bounds(85, 90);
  build_footpaths(*sched, opt, builder.stations_, fbs_schedules);

  progress_tracker->status("Connect Reverse").out_bounds(90, 93);
  builder.connect_reverse();

  progress_tracker->status("Sort Trips").out_bounds(93, 95);
  builder.sort_trips();

  auto hash = cista::BASE_HASH;
  for (auto const* fbs_schedule : fbs_schedules) {
    hash = cista::hash_combine(hash, fbs_schedule->hash());
  }
  for (auto const& prefix : sched->prefixes_) {
    hash = cista::hash(prefix, hash);
  }
  sched->hash_ = hash;

  sched->route_count_ = builder.next_route_index_;

  progress_tracker->status("Lower Bounds").out_bounds(96, 100).in_high(4);
  sched->transfers_lower_bounds_fwd_ = build_interchange_graph(
      sched->station_nodes_, sched->non_station_node_offset_,
      sched->route_count_, search_dir::FWD);
  progress_tracker->increment();
  sched->transfers_lower_bounds_bwd_ = build_interchange_graph(
      sched->station_nodes_, sched->non_station_node_offset_,
      sched->route_count_, search_dir::BWD);
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
  LOG(info) << sched->expanded_trips_.index_size() << " expanded routes";
  LOG(info) << sched->expanded_trips_.element_count() << " expanded trips";
  LOG(info) << builder.broken_trips_ << " broken trips ignored";

  validate_graph(*sched);
  utl::verify(
      std::all_of(begin(sched->trips_), end(sched->trips_),
                  [](auto const& t) { return t.second->edges_ != nullptr; }),
      "missing trip edges");
  return sched;
}

}  // namespace motis::loader

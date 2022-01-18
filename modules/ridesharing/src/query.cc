#include "motis/ridesharing/query.h"

#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"
#include "motis/module/module.h"

#include "motis/protocol/Message_generated.h"
#include "motis/ridesharing/leg_idx.h"

#include <numeric>
#include <tuple>

#include "utl/concat.h"
#include "utl/to_vec.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::osrm;
using namespace motis::lookup;
using namespace motis::logging;
using namespace motis::parking;

namespace motis::ridesharing {

constexpr int LOCATIONS_PER_QUERY = 12;  // 50

inline Position to_pos(geo::latlng const& loc) { return {loc.lat_, loc.lng_}; }

query::query(RidesharingRequest const* const& req, int close_station_radius)
    : start_(*req->start()),
      target_(*req->target()),
      t_(req->time()),
      passengers_(req->passengers()),
      mode_((query_mode)req->query_mode()),
      ppr_search_options_(
          req->ppr_search_options()),  // mode_(req->query_mode()),
      close_station_radius_(close_station_radius) {}

std::vector<unsigned> convert(
    Vector<Offset<Station>> const& v,
    std::unordered_map<std::string, int> const& lookup_station_evas) {
  std::vector<unsigned> res;
  for (auto const& st : v) {
    if (st->id()->str().rfind("80", 0) == 0) {
      auto const it = lookup_station_evas.find(st->id()->str());
      if (it != lookup_station_evas.end()) {
        res.push_back(it->second);
      }
    }
  }
  return res;
}

template <typename T, typename V>
std::unordered_multimap<T, V> select(std::vector<T> const& v,
                                     std::unordered_multimap<T, V> const& m) {
  std::unordered_multimap<T, V> result;
  for (auto& e : v) {
    auto range = m.equal_range(e);
    for (auto it = range.first; it != range.second; ++it) {
      result.emplace(e, it->second);
    }
  }
  return result;
}

// TODO *1.2
ridesharing_edge query::make_direct_ridesharing_edge(std::string const& from,
                                                     std::string const& to,
                                                     lift const& li,
                                                     connection_eval const& ce,
                                                     double const approach_time,
                                                     connection const& con) {
  auto const price = calculate_price(
      ce.from_leg_detour_.duration_ + ce.to_leg_detour_.duration_,
      ce.common_.duration_);
  auto const prior_time = std::accumulate(
      li.rrs_.begin(), li.rrs_.begin() + con.from_leg_, 0.0,
      [](double const acc, auto const& rr) { return acc + rr.duration_; });
  auto const t = static_cast<long>(li.t_ + prior_time + approach_time + 0.5);
  return {from,
          geo::latlng{start_.lat(), start_.lng()},
          con.from_leg_,
          0,
          {},
          to,
          geo::latlng{target_.lat(), target_.lng()},
          con.to_leg_,
          price,
          t,
          static_cast<uint16_t>(ce.common_.duration_),
          0,
          0,
          lift_key{li.t_, li.driver_id_}.to_string()};

  // return CreateRidesharingEdge(
  //   mc, mc.CreateString(from), &start_,
  //   0, &start_,
  //   mc.CreateString(to), &target_,
  //   price, t, ce.common_.duration_, 0,
  //   0, mc.CreateString(lift_key{li.t_, li.driver_id_}.to_string())
  // );
}

ridesharing_edge query::make_ridesharing_edge(
    message_creator& mc, std::string const& from, std::string const& to,
    lift const& li, connection_eval const& ce, double const approach_time,
    connection const& con, geo::latlng const& parking, int const parking_id,
    geo::latlng const& station_location) {
  auto const price = calculate_price(
      ce.from_leg_detour_.duration_ + ce.to_leg_detour_.duration_,
      ce.common_.duration_);
  auto const prior_time = std::accumulate(
      li.rrs_.begin(), li.rrs_.begin() + con.from_leg_, 0.0,
      [](double const acc, auto const& rr) { return acc + rr.duration_; });
  auto const t = static_cast<long>(li.t_ + prior_time + approach_time + 0.5);
  if (parking == station_location) {
    return {from,
            from == "START" ? geo::latlng{start_.lat(), start_.lng()}
                            : station_location,
            con.from_leg_,
            parking_id,
            parking,
            to,
            from == "START" ? station_location
                            : geo::latlng{target_.lat(), target_.lng()},
            con.to_leg_,
            price,
            t,
            from == "START" ? static_cast<uint16_t>(ce.common_.duration_ * 1.2)
                            : static_cast<uint16_t>(ce.common_.duration_),
            0,
            0,
            lift_key{li.t_, li.driver_id_}.to_string()};
  }

  Position myps{parking.lat_, parking.lng_};
  Position myss{station_location.lat_, station_location.lng_};
  mc.create_and_finish(
      MsgContent_ParkingPprRequest,
      parking::CreateParkingPprRequest(
          mc, parking_id, &myps,
          mc.CreateString(ppr_search_options_->profile()->str()),
          mc.CreateString(from == "START" ? to : from), &myss, from == "START",
          motis::ppr::CreateSearchOptions(
              mc, mc.CreateString(ppr_search_options_->profile()->str()),
              ppr_search_options_->duration_limit()))
          .Union(),
      "/parking/ppr_lookup");
  auto const parking_msg = motis_call(make_msg(mc))->val();
  auto const parking_response = motis_content(ParkingPprResponse, parking_msg);
  auto const ppr_duration = parking_response->duration();
  auto const ppr_accessibility = parking_response->accessibility();
  if (parking_response->db()) {
    parking_time_db_ += parking_response->request_duration();
    ++parking_db_;
  } else {
    parking_time_not_db_ += parking_response->request_duration();
    ++parking_not_db_;
  }
  return {from,
          from == "START" ? geo::latlng{start_.lat(), start_.lng()}
                          : station_location,
          con.from_leg_,
          parking_id,
          parking,
          to,
          from == "START" ? station_location
                          : geo::latlng{target_.lat(), target_.lng()},
          con.to_leg_,
          price,
          t,
          from == "START" ? static_cast<uint16_t>(ce.common_.duration_ * 1.2)
                          : static_cast<uint16_t>(ce.common_.duration_),
          ppr_duration,
          ppr_accessibility,
          lift_key{li.t_, li.driver_id_}.to_string()};
}

auto query::close_stations(
    std::unordered_map<std::string, int> const& lookup_station_evas) {
  MOTIS_START_TIMING(total);

  MOTIS_START_TIMING(cs);
  message_creator mc;
  std::vector<Offset<lookup::LookupGeoStationRequest>> c;
  c.push_back(lookup::CreateLookupGeoStationRequest(mc, &start_, 0.0,
                                                    close_station_radius_));
  c.push_back(lookup::CreateLookupGeoStationRequest(mc, &target_, 0.0,
                                                    close_station_radius_));

  mc.create_and_finish(
      MsgContent_LookupBatchGeoStationRequest,
      lookup::CreateLookupBatchGeoStationRequest(mc, mc.CreateVector(c))
          .Union(),
      "/lookup/geo_station_batch");
  auto const lookup_msg = motis_call(make_msg(mc))->val();
  auto const lookup_response =
      motis_content(LookupBatchGeoStationResponse, lookup_msg);
  MOTIS_STOP_TIMING(cs);
  close_station_time_query_ = MOTIS_TIMING_US(cs);
  auto const cs = MOTIS_TIMING_US(cs);
  auto station_lookup = lookup_response->responses()->Get(0)->stations();
  auto start_station_ids = convert(*station_lookup, lookup_station_evas);
  station_lookup = lookup_response->responses()->Get(1)->stations();
  auto target_station_ids = convert(*station_lookup, lookup_station_evas);
  MOTIS_STOP_TIMING(total);
  auto const total = MOTIS_TIMING_US(total);
  LOG(logging::info) << "In Close-Station timing: "
                     << (static_cast<double>(cs) / total);
  return make_pair(start_station_ids, target_station_ids);
}

template <typename T>
inline std::vector<T> add_all(std::vector<T> const& v1,
                              std::vector<T> const& v2) {
  std::vector<unsigned int> temp;
  std::set_union(v1.begin(), v1.end(), v2.begin(), v2.end(),
                 std::back_inserter(temp));
  return temp;
}

template <typename T>
inline std::map<T, unsigned> label_index(std::vector<T> const& v) {
  std::map<T, unsigned> result;
  for (auto e : v) {
    result.emplace(e, result.size());
  }
  return result;
}

bool find_direct(connection_map const& start, connection_map const& target) {
  return std::any_of(std::begin(start), std::end(start), [&](auto const& st) {
    return !select(st.second.stations_, target).empty();
  });
}

std::map<leg_idx, connection> potential_connections(leg_idx sz,
                                                    connection_map const& cm) {
  std::map<leg_idx, connection> connections;
  for (leg_idx i = 0; i < sz; i++) {
    for (leg_idx j = i; j < sz; j++) {
      std::pair<leg_idx, connection> con = {i * sz + j, connection{{}, i, j}};
      connections.insert(con);
    }
  }
  for (auto const s : cm) {
    connections[s.second.from_leg_ * sz + s.second.to_leg_].stations_ = add_all(
        connections[s.second.from_leg_ * sz + s.second.to_leg_].stations_,
        s.second.stations_);
  }
  return connections;
}

void query::setup_potential_edges(
    con_lookup_it& low, con_lookup_it const& high,
    std::unique_ptr<database> const& db,
    std::vector<unsigned> const& start_station_ids,
    std::vector<unsigned> const& target_station_ids) {
  std::vector<unsigned int> pick_up_station_indices;
  std::vector<unsigned int> drop_off_station_indices;
  unsigned lift_waypoint_counter = 0;
  for (; low != high; ++low) {
    auto const clookup = low->second;
    auto const start_stations = select(start_station_ids, clookup.from_to_);
    auto const target_stations = select(target_station_ids, clookup.to_from_);

    if ((mode_ == query_mode::START && start_stations.empty()) ||
        (mode_ == query_mode::DESTINATION && target_stations.empty()) ||
        (mode_ == query_mode::BOTH && start_stations.empty() &&
         target_stations.empty())) {
      continue;
    }
    auto const opt_lift = db->get_lift(low->first);
    if (!opt_lift.has_value()) {
      continue;
    }
    auto const li = opt_lift.value();
    if (li.max_passengers_ < passengers_) {
      continue;
    }
    if (li.t_ + li.max_total_duration_ < t_) {
      continue;
    }
    auto const sz = li.rrs_.size();
    auto const drop_off_connections_by_li =
        (mode_ == query_mode::BOTH || mode_ == query_mode::START)
            ? potential_connections(sz, start_stations)
            : std::map<leg_idx, connection>{};
    bool direct_found = mode_ == query_mode::BOTH &&
                        find_direct(start_stations, target_stations);
    auto const pick_up_connections_by_li =
        (mode_ == query_mode::BOTH || mode_ == query_mode::DESTINATION)
            ? potential_connections(sz, target_stations)
            : std::map<leg_idx, connection>{};
    for (auto& entry : drop_off_connections_by_li) {
      if (!entry.second.stations_.empty()) {
        // If a direct connection exist, there is going to be a routing bc of
        // lift_dest
        lift_depart_.push_back(std::make_tuple(
            li, entry.second, direct_found ? lift_waypoint_counter : -1));
      }
    }
    bool lift_noted = false;
    for (auto& entry : pick_up_connections_by_li) {
      if (!entry.second.stations_.empty()) {
        lift_dest_.push_back(std::make_tuple(li, entry.second));
        if (!lift_noted) {
          lift_waypoint_counter += sz;
          lift_noted = true;
        }
      }
    }
    auto const drop_off_idxs = std::accumulate(
        drop_off_connections_by_li.begin(), drop_off_connections_by_li.end(),
        std::vector<unsigned>{}, [](std::vector<unsigned>& result, auto& con) {
          return add_all(result, con.second.stations_);
        });
    drop_off_station_indices = add_all(drop_off_station_indices, drop_off_idxs);
    auto const pick_up_idxs = std::accumulate(
        pick_up_connections_by_li.begin(), pick_up_connections_by_li.end(),
        std::vector<unsigned>{}, [](std::vector<unsigned>& result, auto& con) {
          return add_all(result, con.second.stations_);
        });
    pick_up_station_indices = add_all(pick_up_station_indices, pick_up_idxs);
  }
  drop_off_stations_ = label_index(drop_off_station_indices);
  pick_up_stations_ = label_index(pick_up_station_indices);
}

void one_to_many(std::vector<ctx::future_ptr<ctx_data, void>>& futures,
                 std::mutex& mutex, std::vector<routing_result>& result,
                 Position const& one, std::vector<Position> const& many,
                 osrm::Direction const direction) {
  for (auto offset = 0u; offset < many.size(); offset += LOCATIONS_PER_QUERY) {
    auto const end_offset = offset + LOCATIONS_PER_QUERY;
    auto const end_it =
        end_offset < many.size() ? many.begin() + end_offset : many.end();
    auto const m =
        utl::to_vec(many.begin() + offset, end_it, [](auto const& p) {
          return geo::latlng{p.lat(), p.lng()};
        });
    futures.emplace_back(spawn_job_void([&, offset, m] {
      auto idx = offset;
      message_creator mc;
      mc.create_and_finish(MsgContent_OSRMOneToManyRequest,
                           osrm::CreateOSRMOneToManyRequest(
                               mc, mc.CreateString("car"), direction, &one,
                               mc.CreateVectorOfStructs(utl::to_vec(
                                   m,
                                   [](auto const& location) {
                                     Position pos{location.lat_, location.lng_};
                                     return pos;
                                   })))
                               .Union(),
                           "/osrm/one_to_many");
      try {
        auto const osrm_msg = motis_call(make_msg(mc))->val();
        auto const osrm_resp = motis_content(OSRMOneToManyResponse, osrm_msg);
        std::lock_guard<std::mutex> lock(mutex);
        for (auto const& cost : *osrm_resp->costs()) {
          result[idx++] = routing_result{cost->duration(), cost->distance()};
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex);
        for (; idx < offset + m.size(); ++idx) {
          result[idx] = routing_result{1000000, 1000000};
        }
      }
    }));
  }
}

template <typename T>
std::vector<Position> lift_waypoints(T const& lift_data, bool front) {
  std::vector<Position> lift_wps;
  for (uint16_t i = 0; i < lift_data.size(); ++i) {
    auto const li = std::get<0>(lift_data[i]);
    auto const begi = li.waypoints_.begin() + !front;
    auto const en = li.waypoints_.end() - front;
    utl::concat(lift_wps, utl::to_vec(begi, en, [](auto& wp) {
                  Position pos{wp.lat_, wp.lng_};
                  return pos;
                }));
    while (i < lift_data.size() - 1 &&
           li == lift_key{std::get<0>(lift_data[i + 1]).t_,
                          std::get<0>(lift_data[i + 1]).driver_id_}) {
      ++i;
    }
  }
  return lift_wps;
}

void query::routing(std::vector<std::pair<geo::latlng, int>> const& parkings) {
  auto const lift_starts = lift_waypoints(lift_depart_, true);
  lifts_to_query_start_.resize(lift_starts.size());

  auto drop_offs = utl::to_vec(drop_off_stations_, [&](auto e) {
    auto const parking = parkings[e.first].first;
    auto const pos = Position{parking.lat_, parking.lng_};
    return pos;
  });
  utl::concat(drop_offs, lift_waypoints(lift_depart_, false));
  drop_offs.push_back(target_);
  to_drop_off_.resize(drop_offs.size());
  auto pick_ups = utl::to_vec(pick_up_stations_, [&](auto e) {
    auto const parking = parkings[e.first].first;
    auto const pos = Position{parking.lat_, parking.lng_};
    return pos;
  });
  utl::concat(pick_ups, lift_waypoints(lift_dest_, true));
  from_pick_up_.resize(pick_ups.size());

  auto const lift_destinations = lift_waypoints(lift_dest_, false);
  query_target_to_lifts_.resize(lift_destinations.size());

  std::vector<ctx::future_ptr<ctx_data, void>> futures;
  std::mutex mutex;
  one_to_many(futures, mutex, lifts_to_query_start_, start_, lift_starts,
              osrm::Direction_Backward);
  one_to_many(futures, mutex, to_drop_off_, start_, drop_offs,
              osrm::Direction_Forward);
  one_to_many(futures, mutex, from_pick_up_, target_, pick_ups,
              osrm::Direction_Backward);
  one_to_many(futures, mutex, query_target_to_lifts_, target_,
              lift_destinations, osrm::Direction_Forward);
  ctx::await_all(futures);
}

connection_eval query::evaluate_same_leg(query_mode mode,
                                         routing_result const& common,
                                         lift const& li, unsigned lift_idx,
                                         connection con, unsigned st,
                                         int direct_connection) {
  auto from_leg_detour = routing_result{0.0, 0.0};
  switch (mode) {
    case query_mode::START:
      from_leg_detour = lifts_to_query_start_[lift_idx] + common +
                        li.from_routings_[con.to_leg_].at(st) -
                        li.rrs_[con.from_leg_];
      break;
    case query_mode::DESTINATION:
      from_leg_detour = li.to_routings_[con.from_leg_].at(st) + common +
                        query_target_to_lifts_[lift_idx] -
                        li.rrs_[con.from_leg_];
      break;
    case query_mode::BOTH:
      from_leg_detour =
          lifts_to_query_start_[lift_idx] + common +
          query_target_to_lifts_[direct_connection + con.to_leg_] -
          li.rrs_[con.from_leg_];
      break;
    default: break;
  }
  return {common, from_leg_detour, routing_result{0.0, 0.0}};
}

connection_eval query::evaluate_different_leg(query_mode mode,
                                              routing_result const& common,
                                              lift const& li, unsigned lift_idx,
                                              connection con, unsigned st,
                                              int direct_connection,
                                              unsigned j) {
  auto const station_offset = mode == query_mode::DESTINATION
                                  ? pick_up_stations_.size()
                                  : drop_off_stations_.size();
  auto from_leg_detour = routing_result{0.0, 0.0};
  auto to_leg_detour = routing_result{0.0, 0.0};
  switch (mode) {
    case query_mode::START:
      from_leg_detour = lifts_to_query_start_[lift_idx] +
                        to_drop_off_[station_offset + lift_idx] -
                        li.rrs_[con.from_leg_];
      to_leg_detour = li.from_routings_[con.to_leg_].at(st) +
                      li.to_routings_[con.to_leg_].at(st) -
                      li.rrs_[con.to_leg_];
      break;
    case query_mode::DESTINATION:
      /*if (query_target_to_lifts_.size() <= lift_idx) {
        LOG(logging::info) << query_target_to_lifts_.size() << "qttl " <<
      lift_idx;
      }
      if (from_pick_up_.size() <= station_offset + lift_idx) {

        LOG(logging::info) << from_pick_up_.size() << "fpu " << station_offset +
      lift_idx;
      }
      if (li.rrs_.size() <= con.from_leg_ || li.rrs_.size() <= con.to_leg_ ||
         li.to_routings_.size() <= con.from_leg_ || li.from_routings_.size() <=
      con.from_leg_) { LOG(logging::info) << li.rrs_.size() << "qmd rrs " <<
      con.from_leg_ << "->" << con.to_leg_
           << " " << li.from_routings_.size() << " " << li.to_routings_.size();
      }*/
      // common = from_pick_up_[station_offset + lift_idx] +
      // li.from_routings_[con.from_leg_].at(st);
      from_leg_detour = li.to_routings_[con.from_leg_].at(st) +
                        li.from_routings_[con.from_leg_].at(st) -
                        li.rrs_[con.from_leg_];
      to_leg_detour = from_pick_up_[station_offset + lift_idx] +
                      query_target_to_lifts_[lift_idx] - li.rrs_[con.to_leg_];
      break;
    case query_mode::BOTH:
      // common = to_drop_off_[station_offset + lift_idx] +
      //                     from_pick_up_[pick_up_stations_.size() +
      //                     con.to_leg_ + j];
      /*if (lifts_to_query_start_.size() <= lift_idx) {
        LOG(logging::info) << lifts_to_query_start_.size() << "dir ltqs" <<
      lift_idx;
      }
      if (query_target_to_lifts_.size() <= direct_connection + con.to_leg_) {
        LOG(logging::info) << query_target_to_lifts_.size() << "dir qttl" <<
      lift_idx;
      }
      if (to_drop_off_.size() <= station_offset + lift_idx) {

        LOG(logging::info) << to_drop_off_.size() << "dir tdo" << station_offset
      + lift_idx;
      }
      if (from_pick_up_.size() <= station_offset + lift_idx) {

        LOG(logging::info) << from_pick_up_.size() << "dir fpu " <<
      pick_up_stations_.size() + con.to_leg_ + j;
      }
      if (li.rrs_.size() <= con.from_leg_ || li.rrs_.size() <= con.to_leg_ ||
         li.to_routings_.size() <= con.from_leg_ || li.from_routings_.size() <=
      con.from_leg_) { LOG(logging::info) << li.rrs_.size() << "qmboth rrs " <<
      con.from_leg_ << "->" << con.to_leg_
           << " " << li.from_routings_.size() << " " << li.to_routings_.size();
      }*/
      from_leg_detour = lifts_to_query_start_[lift_idx] +
                        to_drop_off_[station_offset + lift_idx] -
                        li.rrs_[con.from_leg_];
      to_leg_detour =
          query_target_to_lifts_[direct_connection + con.to_leg_] +
          from_pick_up_[pick_up_stations_.size() + con.to_leg_ + j] -
          li.rrs_[con.to_leg_];
      break;
    default: break;
  }
  return {common,
          //+ std::accumulate(li.rrs_.begin() + con.from_leg_ + 1,
          // li.rrs_.begin() + con.to_leg_, routing_result{0, 0}),
          from_leg_detour, to_leg_detour};
}

query_response query::make_edges(
    message_creator& mc, std::vector<std::string> const& station_evas,
    std::vector<std::pair<geo::latlng, int>> const& parkings,
    std::vector<geo::latlng> const& station_locations) {
  MOTIS_START_TIMING(make_edges_time_query);

  query_response qr{routing_time_query_};
  qr.close_station_time_ = close_station_time_query_;
  unsigned j = 0;
  // pick-up -> station
  for (unsigned i = 0; i < lift_depart_.size(); i++) {
    auto const& [li, con, direct_connection] = lift_depart_[i];
    auto const lift_idx = j + con.from_leg_;

    for (auto st : con.stations_) {
      auto const ce =
          con.from_leg_ == con.to_leg_
              ? evaluate_same_leg(query_mode::START,
                                  to_drop_off_[drop_off_stations_.at(st)], li,
                                  lift_idx, con, st)
              : evaluate_different_leg(query_mode::START,
                                       to_drop_off_[drop_off_stations_.at(st)],
                                       li, lift_idx, con, st);

      if (valid_edge(li, con.from_leg_, con.to_leg_, ce, passengers_)) {
        auto const prior_time =
            std::accumulate(li.rrs_.begin(), li.rrs_.begin() + con.from_leg_,
                            0.0, [](double const acc, auto const& rr) {
                              return acc + rr.duration_;
                            });
        auto const t =
            static_cast<long>(li.t_ + prior_time +
                              lifts_to_query_start_[lift_idx].duration_ + 0.5);
        if (t < t_) {
          continue;
        }
        qr.deps_.push_back(make_ridesharing_edge(
            mc, "START", station_evas[st], li, ce,
            lifts_to_query_start_[lift_idx].duration_, con, parkings[st].first,
            parkings[st].second, station_locations[st]));
        // LOG(logging::info) << "DEPS: " << station_evas[st] <<
        // qr.deps_.size();
      }
    }
    // direct_connections;
    if (direct_connection >= 0) {
      auto const ce =
          con.from_leg_ == con.to_leg_
              ? evaluate_same_leg(query_mode::BOTH, to_drop_off_.back(), li,
                                  lift_idx, con, 0, direct_connection)
              : evaluate_different_leg(query_mode::BOTH, to_drop_off_.back(),
                                       li, lift_idx, con, 0, direct_connection,
                                       j);
      if (valid_edge(li, con.from_leg_, con.to_leg_, ce, passengers_)) {
        auto const prior_time =
            std::accumulate(li.rrs_.begin(), li.rrs_.begin() + con.from_leg_,
                            0.0, [](double const acc, auto const& rr) {
                              return acc + rr.duration_;
                            });
        auto const t =
            static_cast<long>(li.t_ + prior_time +
                              lifts_to_query_start_[lift_idx].duration_ + 0.5);
        if (t < t_) {
          continue;
        }
        qr.direct_connections_.push_back(make_direct_ridesharing_edge(
            "START", "END", li, ce, lifts_to_query_start_[lift_idx].duration_,
            con));
      }
    }
    if (i != lift_depart_.size() - 1 &&
        !(li == lift_key{std::get<0>(lift_depart_[i + 1]).t_,
                         std::get<0>(lift_depart_[i + 1]).driver_id_})) {
      j += li.rrs_.size();
    }
  }
  // station -> drop-off

  j = 0;
  for (unsigned i = 0; i < lift_dest_.size(); i++) {
    auto const& [li, con] = lift_dest_[i];
    auto const lift_idx = j + con.to_leg_;
    for (auto st : con.stations_) {
      auto const ce =
          con.from_leg_ == con.to_leg_
              ? evaluate_same_leg(query_mode::DESTINATION,
                                  from_pick_up_[pick_up_stations_.at(st)], li,
                                  lift_idx, con, st)
              : evaluate_different_leg(query_mode::DESTINATION,
                                       from_pick_up_[pick_up_stations_.at(st)],
                                       li, lift_idx, con, st);
      if (valid_edge(li, con.from_leg_, con.to_leg_, ce, passengers_)) {
        auto const prior_time =
            std::accumulate(li.rrs_.begin(), li.rrs_.begin() + con.from_leg_,
                            0.0, [](double const acc, auto const& rr) {
                              return acc + rr.duration_;
                            });
        auto const t = static_cast<long>(
            li.t_ + prior_time +
            li.to_routings_[con.from_leg_].at(st).duration_ + 0.5);
        if (t < t_) {
          continue;
        }
        qr.arrs_.push_back(make_ridesharing_edge(
            mc, station_evas[st], "END", li, ce,
            li.to_routings_[con.from_leg_].at(st).duration_, con,
            parkings[st].first, parkings[st].second, station_locations[st]));
      }
    }
    if (i != lift_dest_.size() - 1 &&
        !(li == lift_key{std::get<0>(lift_dest_[i + 1]).t_,
                         std::get<0>(lift_dest_[i + 1]).driver_id_})) {
      j += li.rrs_.size();
    }
  }
  MOTIS_STOP_TIMING(make_edges_time_query);
  qr.edges_time_ = MOTIS_TIMING_US(make_edges_time_query);
  return qr;
}

query_response query::apply(
    message_creator& mc,
    std::map<lift_key, connection_lookup> const& lift_connections,
    std::unique_ptr<database> const& db,
    std::vector<geo::latlng> const& station_locations,
    std::vector<std::pair<geo::latlng, int>> const& parkings,
    std::unordered_map<std::string, int> const& lookup_station_evas,
    std::vector<std::string> const& station_evas) {
  MOTIS_START_TIMING(total);

  MOTIS_START_TIMING(cs);
  auto station_lookup = close_stations(lookup_station_evas);
  MOTIS_STOP_TIMING(cs);
  auto const cs = MOTIS_TIMING_US(cs);
  close_station_time_query_ = MOTIS_TIMING_US(cs);

  MOTIS_START_TIMING(ts);
  auto k = lift_key{t_ - 3600 * 8, 0};
  auto low = lift_connections.lower_bound(k);
  k.t_ += 3600 * 16;
  auto const high = lift_connections.upper_bound(k);
  MOTIS_STOP_TIMING(ts);
  auto const ts = MOTIS_TIMING_US(ts);

  MOTIS_START_TIMING(spe);
  LOG(logging::info) << "pre setup";
  setup_potential_edges(low, high, db, station_lookup.first,
                        station_lookup.second);
  MOTIS_STOP_TIMING(spe);
  auto const spe = MOTIS_TIMING_US(spe);

  LOG(logging::info) << "pre routing";
  MOTIS_START_TIMING(rt);
  routing(parkings);
  MOTIS_STOP_TIMING(rt);
  auto const rt = MOTIS_TIMING_US(rt);
  MOTIS_STOP_TIMING(total);
  auto const total = MOTIS_TIMING_US(total);
  routing_time_query_ = MOTIS_TIMING_US(rt);
  LOG(logging::info) << "pre edges";
  LOG(logging::info) << "Time Profile: "
                     << "Cs " << ((double)cs) / total << "; Ts "
                     << ((double)ts) / total << "; Spe "
                     << ((double)spe) / total << "; Routing "
                     << ((double)rt) / total;
  return make_edges(mc, station_evas, parkings, station_locations);
}

}  // namespace motis::ridesharing

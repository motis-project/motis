#include "motis/ridesharing/ridesharing.h"

#include "boost/algorithm/string/predicate.hpp"

#include "motis/core/common/timing.h"

#include "motis/core/schedule/schedule.h"

#include "motis/module/context/motis_publish.h"

#include "utl/to_vec.h"

namespace po = boost::program_options;
using namespace flatbuffers;
using namespace motis::module;
using namespace motis::osrm;
using namespace motis::lookup;
using namespace motis::logging;
using namespace motis::parking;

namespace motis::ridesharing {

ridesharing::ridesharing() : module("Ridesharing Options", "ridesharing") {
  param(database_path_, "database_path",  //"ridesharing.mdb"
        "Location of the Ridesharing Database (folder or ':memory:')");
  param(close_station_radius_, "max_station_lookup_range",
        "maximal distance to consider a location as similar to another for "
        "routing (see LookUp)");
  param(max_stations_, "stations",
        "maximal stations to consider as a meeting points");
  param(db_max_size_, "db_max_size", "virtual memory map size");
  param(use_parking_, "use_parking",
        "if true uses parking spots with their ppr edges (stored in db) "
        "proximate to stations"
        " as the actual meeting point. Otherwise station location is assumed");
}

ridesharing::~ridesharing() = default;

void ridesharing::init(motis::module::registry& reg) {
  reg.subscribe("/osrm/initialized", [&](auto&& m) { return init_module(m); });
  reg.register_op("/ridesharing/edges", [&](auto&& m) { return edges(m); });
  reg.register_op("/ridesharing/create", [&](auto&& m) { return create(m); });
  reg.register_op("/ridesharing/remove", [&](auto&& m) { return remove(m); });
  reg.register_op("/ridesharing/book", [&](auto&& m) { return book(m); });
  reg.register_op("/ridesharing/unbook", [&](auto&& m) { return unbook(m); });
  reg.register_op("/ridesharing/timeout",
                  [&](auto&& m) { return time_out(m); });
  reg.register_op("/ridesharing/stats",
                  [&](auto&& m) { return statistics(m); });
}

msg_ptr ridesharing::init_module(msg_ptr const&) {
  MOTIS_START_TIMING(init_time);
  auto const& sched = get_sched();
  auto station_data = std::vector<std::tuple<int, std::string, geo::latlng>>{};
  for (auto const& s : sched.stations_) {
    if (boost::algorithm::starts_with(s->eva_nr_.str(), "80")) {
      station_data.emplace_back(
          -std::accumulate(s->arr_class_events_.begin(),
                           s->arr_class_events_.end(), 0) -
              std::accumulate(s->dep_class_events_.begin(),
                              s->dep_class_events_.end(), 0),
          s->eva_nr_, geo::latlng{s->lat(), s->lng()});
    }
  }
  if (max_stations_ >= station_data.size()) {
    station_locations_ =
        utl::to_vec(station_data, [](auto const& e) { return std::get<2>(e); });
    station_evas_ =
        utl::to_vec(station_data, [](auto const& e) { return std::get<1>(e); });
    for (int i = 0; i < station_evas_.size(); i++) {
      lookup_station_evas_.emplace(station_evas_[i], i);
    }
  } else {
    std::sort(station_data.begin(), station_data.end());
    station_locations_ =
        utl::to_vec(station_data.begin(), station_data.begin() + max_stations_,
                    [](auto const& e) { return std::get<2>(e); });
    station_evas_ =
        utl::to_vec(station_data.begin(), station_data.begin() + max_stations_,
                    [](auto const& e) { return std::get<1>(e); });
    for (int i = 0; i < station_evas_.size(); i++) {
      lookup_station_evas_.emplace(station_evas_[i], i);
    }
  }
  if (!database_path_.empty()) {
    database_ = std::make_unique<database>(database_path_, db_max_size_);
    LOG(info) << "Database created at: " + database_path_;
  }
  MOTIS_START_TIMING(parking_time);
  parkings_ = utl::to_vec(station_locations_,
                          [this](auto const& loc) { return add_parking(loc); });
  MOTIS_STOP_TIMING(parking_time);
  stats_.parking_time_ = MOTIS_TIMING_US(parking_time);
  initialize_routing_matrix();
  load_lifts_from_db();
  MOTIS_STOP_TIMING(init_time);
  stats_.init_time_ = MOTIS_TIMING_US(init_time);
  LOG(info) << "Initialiation complete with " << parkings_.size()
            << " considered Stations";
  motis_publish(make_no_msg("/ridesharing/initialized"));
  return nullptr;
}

std::pair<geo::latlng, int> ridesharing::add_parking(geo::latlng const& loc) {
  if (!use_parking_) {
    return {loc, -1};
  }
  auto const station_pos = Position{loc.lat_, loc.lng_};
  message_creator mc;
  mc.create_and_finish(MsgContent_ParkingGeoRequest,
                       CreateParkingGeoRequest(mc, &station_pos, 1000).Union(),
                       "/parking/geo");
  auto const parking_msg = motis_call(make_msg(mc))->val();
  auto const parking_response = motis_content(ParkingGeoResponse, parking_msg);
  auto park_loc = geo::latlng{loc.lat_, loc.lng_};
  auto park_id = -1;
  if (parking_response->parkings()->size() != 0) {
    auto const pfirst = parking_response->parkings()->Get(0);
    park_loc = {pfirst->pos()->lat(), pfirst->pos()->lng()};
    park_id = pfirst->id();
  }
  return {park_loc, park_id};
}

msg_ptr make_ridesharing_response(ResponseType res_type,
                                  std::string const& lift_key) {
  message_creator mc;
  mc.create_and_finish(
      MsgContent_RidesharingLiftResponse,
      CreateRidesharingLiftResponse(mc, res_type, mc.CreateString(lift_key))
          .Union());
  return make_msg(mc);
}

msg_ptr make_ridesharing_response(ResponseType res_type, lift_key const& key) {
  message_creator mc;
  mc.create_and_finish(MsgContent_RidesharingLiftResponse,
                       CreateRidesharingLiftResponse(
                           mc, res_type, mc.CreateString(key.to_string()))
                           .Union());
  return make_msg(mc);
}

msg_ptr ridesharing::remove(msg_ptr const& msg) {
  MOTIS_START_TIMING(deletion_time);
  auto const req = motis_content(RidesharingRemove, msg);
  lift_key lk = {req->time(), req->driver()};
  database_->remove_lift(lk);
  auto const it = lift_connections_.find(lk);
  if (it != std::end(lift_connections_)) {
    lift_connections_.erase(it);
    MOTIS_STOP_TIMING(deletion_time);
    stats_.total_deletion_time_ += MOTIS_TIMING_US(deletion_time);
    ++stats_.deletions_;
    return make_ridesharing_response(ResponseType_Success, lk);
  }
  stats_.deletions_ += 1000000;
  return make_ridesharing_response(ResponseType_Not_Found, lk);
}

msg_ptr ridesharing::create(msg_ptr const& msg) {
  MOTIS_START_TIMING(creation_time);
  auto const req = motis_content(RidesharingCreate, msg);
  auto const start = geo::latlng{req->start()->lat(), req->start()->lng()};
  auto const target = geo::latlng{req->target()->lat(), req->target()->lng()};
  auto li =
      lift{start, target, req->time(), req->driver(), req->max_passengers()};
  if (li.rrs_[0].duration_ <= 0) {
    ++stats_.invalid_lifts_;
    return make_ridesharing_response(ResponseType_Not_Available, "");
  }
  li.initial_station_routing(parkings_);
  stats_.total_routing_time_create_ += li.creation_routing_time_;
  auto const lk = lift_key{li.t_, li.driver_id_};
  if (auto lift_db = database_->get_lift(lk)) {
    LOG(logging::error) << "Created Lift already exists.";
    return make_ridesharing_response(ResponseType_Already_Exists, lk);
  }
  lift_connections_.insert(
      std::make_pair(lk, setup_acceptable_stations(li, routing_matrix_)));
  database_->put_lift(make_db_lift(li), lk);
  MOTIS_STOP_TIMING(creation_time);
  stats_.total_creation_time_ += MOTIS_TIMING_US(creation_time);
  ++stats_.creations_;
  return make_ridesharing_response(ResponseType_Success, lk);
}

msg_ptr ridesharing::book(msg_ptr const& msg) {
  MOTIS_START_TIMING(booking_time);
  auto const req = motis_content(RidesharingBook, msg);
  auto const lk = lift_key{req->time_lift_start(), req->driver()};
  if (auto li = database_->get_lift(lk)) {
    auto const pick_up =
        geo::latlng{req->pick_up()->lat(), req->pick_up()->lng()};
    auto const drop_off =
        geo::latlng{req->drop_off()->lat(), req->drop_off()->lng()};
    auto const pas = passenger{req->passenger(),
                               pick_up,
                               drop_off,
                               req->price(),
                               req->required_arrival(),
                               req->passenger_count()};

    if (!li->process_booking(pas, pick_up, req->pick_up_on_leg_index(),
                             drop_off, req->drop_off_on_leg_index(),
                             parkings_)) {
      stats_.bookings_ += 100000;
      return make_ridesharing_response(ResponseType_Not_Available, lk);
    }
    stats_.total_routing_time_book_ += li->booking_routing_time_;
    switch (li->passengers_.size()) {
      case 2: ++stats_.two_passenger_; break;
      case 3:
        ++stats_.three_passenger_;
        --stats_.two_passenger_;
        break;
      case 4:
        ++stats_.four_passenger_;
        --stats_.three_passenger_;
        break;
      case 5: --stats_.four_passenger_; break;
      default:
        stats_.max_passengers_ =
            std::max(stats_.max_passengers_, li->passengers_.size());
        break;
    }

    auto const lc = lift_connections_.find(lk);
    if (lc != lift_connections_.end()) {
      lc->second = setup_acceptable_stations(li.value(), routing_matrix_);
    }
    database_->put_lift(make_db_lift(li.value()),
                        lift_key{li->t_, li->driver_id_});
    MOTIS_STOP_TIMING(booking_time);
    stats_.total_booking_time_ += MOTIS_TIMING_US(booking_time);
    ++stats_.bookings_;
    return make_ridesharing_response(ResponseType_Success, lk);
  } else {
    stats_.bookings_ += 100000;
    LOG(logging::error) << "Lift not found!";
    return make_ridesharing_response(ResponseType_Not_Found, lk);
  }
}

msg_ptr ridesharing::unbook(msg_ptr const& msg) {
  auto const req = motis_content(RidesharingUnbook, msg);
  auto const lk = lift_key{req->time_lift_start(), req->driver()};
  if (auto li = database_->get_lift(lk)) {
    if (!li->remove_booking(req->passenger())) {
      LOG(logging::error) << "Passenger not found";
      return make_ridesharing_response(ResponseType_Not_Yet_Booked, lk);
    }
    auto const lc = lift_connections_.find(lk);
    if (lc != lift_connections_.end()) {
      lc->second = setup_acceptable_stations(li.value(), routing_matrix_);
    }
    database_->put_lift(make_db_lift(li.value()),
                        lift_key{li->t_, li->driver_id_});
    return make_ridesharing_response(ResponseType_Success, lk);
  } else {
    LOG(logging::error) << "Lift not found!";
    return make_ridesharing_response(ResponseType_Not_Found, lk);
  }
}

msg_ptr ridesharing::edges(msg_ptr const& msg) {
  MOTIS_START_TIMING(query_time);
  message_creator mc;
  query q{motis_content(RidesharingRequest, msg), close_station_radius_};
  auto response = q.apply(mc, lift_connections_, database_, station_locations_,
                          parkings_, lookup_station_evas_, station_evas_);
  auto const to_fbs = [&](auto const& e) {
    Position start{e.from_pos_.lat_, e.from_pos_.lng_};
    Position parking_pos{e.parking_pos_.lat_, e.parking_pos_.lng_};
    Position target{e.to_pos_.lat_, e.to_pos_.lng_};
    return CreateRidesharingEdge(
        mc, mc.CreateString(e.from_station_id_), &start, e.from_leg_,
        e.parking_id_, &parking_pos, mc.CreateString(e.to_station_id_), &target,
        e.to_leg_, e.rs_price_, e.rs_t_, e.rs_duration_, e.ppr_duration_,
        e.ppr_accessibility_, mc.CreateString(e.lift_key_));
  };
  auto const edges = response.arrs_.size() +
                     response.direct_connections_.size() +
                     response.deps_.size();
  if (edges == 0) {
    ++stats_.requests_without_edges_;
  } else {
    stats_.total_edges_constructed_ += edges;
    stats_.direct_edges_constructed_ += response.direct_connections_.size();
    stats_.door_in_edges_constructed_ += response.deps_.size();
    stats_.door_out_edges_constructed_ += response.arrs_.size();
  }
  stats_.parking_time_db_ += q.parking_time_db_;
  stats_.parking_time_not_db_ += q.parking_time_not_db_;
  stats_.parking_db_ += q.parking_db_;
  stats_.parking_not_db_ += q.parking_not_db_;

  mc.create_and_finish(
      MsgContent_RidesharingResponse,
      CreateRidesharingResponse(
          mc, mc.CreateVector(utl::to_vec(response.arrs_, to_fbs)),
          mc.CreateVector(utl::to_vec(response.deps_, to_fbs)),
          mc.CreateVector(utl::to_vec(response.direct_connections_, to_fbs)))
          .Union());
  MOTIS_STOP_TIMING(query_time);
  stats_.total_routing_time_query_ += response.routing_time_;
  stats_.total_edges_time_query_ += response.edges_time_;
  stats_.total_close_station_time_query_ += response.close_station_time_;
  stats_.total_query_time_ += MOTIS_TIMING_US(query_time);
  ++stats_.queries_;
  return make_msg(mc);
}

void ridesharing::load_lifts_from_db() {
  auto db_lifts = database_->get_lifts();
  for (auto& li : db_lifts) {
    if (!same_stations_) {
      li.recompute_routings(parkings_);
      database_->put_lift(make_db_lift(li), lift_key{li.t_, li.driver_id_});
    }
    lift_connections_.insert({lift_key{li.t_, li.driver_id_},
                              setup_acceptable_stations(li, routing_matrix_)});
  }
}

long station_hash(std::vector<std::pair<geo::latlng, int>> const& parkings) {
  return std::accumulate(
      parkings.begin(), parkings.end(), 12345, [](int acc, auto const& st) {
        return (acc ^ (std::hash<double>{}(st.first.lat_) << 1) ^
                (std::hash<double>{}(st.first.lng_) >> 1))
               << 1;
      });
}

void ridesharing::initialize_routing_matrix() {
  auto hashcode = station_hash(parkings_);
  if (database_->is_initialized()) {
    same_stations_ = hashcode == database_->get_station_hashcode();
    if (same_stations_) {
      LOG(info) << "Loading Routing Matrix from Database!";
      routing_matrix_ = database_->get_routing_table();
      return;
    }
  }
  LOG(info) << "    Recomputing Routing Matrix!";
  auto const positions = utl::to_vec(parkings_, [](auto const& p) {
    return Position{p.first.lat_, p.first.lng_};
  });
  message_creator mc;
  mc.create_and_finish(
      MsgContent_OSRMManyToManyRequest,
      CreateOSRMManyToManyRequest(mc, mc.CreateString("car"),
                                  mc.CreateVectorOfStructs(positions))
          .Union(),
      "/osrm/many_to_many");
  auto const osrm_msg = motis_call(make_msg(mc))->val();
  auto const routing_response = motis_content(OSRMManyToManyResponse, osrm_msg);
  for (auto const& fb_rr_row : *routing_response->routing_matrix()) {
    routing_matrix_.push_back(
        utl::to_vec(*fb_rr_row->costs(), [](auto const& rr) {
          return routing_result{rr->duration(), rr->distance()};
        }));
  }
  database_->put_routing_table(make_routing_table(routing_matrix_), hashcode);
}

msg_ptr ridesharing::time_out([[maybe_unused]] msg_ptr const& msg) {
  auto const t = std::time(0);
  auto const lk = lift_key{t - 3600 * 12, 0};
  auto low = lift_connections_.lower_bound(lk);
  for (auto it = begin(lift_connections_); it != low; it++) {
    database_->remove_lift(it->first);
  }
  lift_connections_.erase(begin(lift_connections_), low);

  return make_ridesharing_response(ResponseType_Success, lk);
}

msg_ptr ridesharing::statistics([[maybe_unused]] msg_ptr const& msg) {
  message_creator mc;
  mc.create_and_finish(
      MsgContent_RidesharingStatistics,
      CreateRidesharingStatistics(mc, to_fbs(mc, "rs", stats_)).Union());
  return make_msg(mc);
}

}  // namespace motis::ridesharing

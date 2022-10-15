#include "motis/intermodal/intermodal.h"

#include <algorithm>
#include <functional>
#include <mutex>
#include <optional>
#include <utility>

#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/constants.h"
#include "motis/core/common/timing.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"

#include "motis/intermodal/direct_connections.h"
#include "motis/intermodal/error.h"
#include "motis/intermodal/eval/commands.h"
#include "motis/intermodal/mumo_edge.h"
#include "motis/intermodal/query_bounds.h"
#include "motis/intermodal/statistics.h"
#include "motis/intermodal/ondemand_availability.h"

#include "motis/protocol/Message_generated.h"

using namespace flatbuffers;
using namespace motis::module;
using namespace motis::lookup;
using namespace motis::osrm;
using namespace motis::routing;
using namespace motis::revise;

namespace motis::intermodal {

intermodal::intermodal() : module("Intermodal Options", "intermodal") {
  param(router_, "router", "routing module");
  param(revise_, "revise", "revise connections");
  param(ondemand_infos_, "ondemand", "ondemand server infos");
}

intermodal::~intermodal() = default;

void intermodal::reg_subc(motis::module::subc_reg& r) {
  r.register_cmd("intermodal_generate", "generate routing queries",
                 eval::generate);
  r.register_cmd("intermodal_compare", "print difference between results",
                 eval::compare);
}

void intermodal::init(motis::module::registry& r) {
  r.register_op("/intermodal", [this](msg_ptr const& m) { return route(m); },
                {kScheduleReadAccess});
  if (router_.empty()) {
    router_ = "/routing";
  } else if (router_[0] != '/') {
    router_ = "/" + router_;
  }
  r.subscribe("/init", [this]() { ppr_profiles_.update(); }, {});
}

int doctorwho = 0;
int masterwho = 0;

std::vector<Offset<Connection>> revise_connections(
    std::vector<journey> const& journeys, statistics& stats,
    message_creator& mc) {
  MOTIS_START_TIMING(revise_timing);
  message_creator rmc;
  rmc.create_and_finish(
      MsgContent_ReviseRequest,
      CreateReviseRequest(
          rmc, rmc.CreateVector(utl::to_vec(
                   journeys,
                   [&](journey const& j) { return to_connection(rmc, j); })))
          .Union(),
      "/revise");
  auto const res = motis_call(make_msg(rmc))->val();
  auto result = utl::to_vec(*motis_content(ReviseResponse, res)->connections(),
                            [&mc](Connection const* con) {
                              return motis_copy_table(Connection, mc, con);
                            });
  MOTIS_STOP_TIMING(revise_timing);
  stats.revise_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_MS(revise_timing));
  return result;
}

struct parking_patch {
  parking_patch(mumo_edge const* e, unsigned transport_from,
                unsigned transport_to)
      : e_(e), from_(transport_from), to_(transport_to) {}

  mumo_edge const* e_{};
  unsigned from_{};
  unsigned to_{};
};

struct ondemand_patch {
  ondemand_patch(mumo_edge const* e, unsigned transport_from,
           unsigned transport_to, journey j_todelete, int j_id)
      : e_(e), from_(transport_from), to_(transport_to), j_(std::move(j_todelete)),
        j_id_(j_id) {}

  mumo_edge const* e_{};
  unsigned from_{};
  unsigned to_{};
  journey j_{};
  int j_id_{};
};

struct split_transport_result {
  journey::stop& parking_stop_;
  journey::transport& first_transport_;
  journey::transport& second_transport_;
};

split_transport_result split_transport(journey& j,
                                       std::vector<parking_patch>& patches,
                                       journey::transport& transport) {
  utl::verify(transport.to_ == transport.from_ + 1, "invalid transport");
  auto const transport_idx = std::distance(
      begin(j.transports_),
      std::find_if(
          begin(j.transports_), end(j.transports_), [&](auto const& t) {
            return t.from_ == transport.from_ && t.to_ == transport.to_;
          }));
  auto const new_stop_idx = transport.to_;
  auto& new_stop = *j.stops_.emplace(j.stops_.begin() + new_stop_idx);

  auto const update_idx = [&](unsigned& i) {
    if (i >= new_stop_idx) {
      ++i;
    }
  };

  auto const update_indices = [&](auto& v) {
    for (auto& e : v) {
      update_idx(e.from_);
      update_idx(e.to_);
    }
  };

  update_indices(j.transports_);
  update_indices(j.trips_);
  update_indices(j.attributes_);
  update_indices(j.free_texts_);
  update_indices(j.problems_);
  update_indices(patches);

  j.transports_.insert(begin(j.transports_) + transport_idx,
                       j.transports_[transport_idx]);

  auto& t1 = j.transports_[transport_idx];
  auto& t2 = j.transports_[transport_idx + 1];

  t1.to_--;
  t2.from_++;

  return {new_stop, t1, t2};
}

std::string get_parking_station(int index) {
  switch (index) {
    case 0: return STATION_VIA0;
    case 1: return STATION_VIA1;
    case 2: return STATION_VIA2;
    case 3: return STATION_VIA3;
    default: throw std::system_error(error::parking_edge_error);
  }
}

journey::transport& get_transport(journey& j, unsigned const from,
                                  unsigned const to) {
  for (auto& t : j.transports_) {
    if (t.from_ == from && t.to_ == to) {
      return t;
    }
  }
  throw std::system_error(error::parking_edge_error);
}

bool is_virtual_station(journey::stop const& s) {
  return s.name_ == STATION_START || s.name_ == STATION_END;
}

void apply_parking_patches(journey& j, std::vector<parking_patch>& patches) {
  auto parking_idx = 0;
  for (auto& p : patches) {
    auto t = get_transport(j, p.from_, p.to_);
    auto const car_first = is_virtual_station(j.stops_[p.from_]);

    auto const first_edge_duration =
        (car_first ? p.e_->car_parking_->car_duration_
                   : p.e_->car_parking_->foot_duration_) *
        60;

    auto str = split_transport(j, patches, t);
    auto const parking_station = get_parking_station(parking_idx++);
    str.parking_stop_.eva_no_ = parking_station;
    str.parking_stop_.name_ = parking_station;
    str.parking_stop_.lat_ = p.e_->car_parking_->parking_pos_.lat_;
    str.parking_stop_.lng_ = p.e_->car_parking_->parking_pos_.lng_;
    str.parking_stop_.arrival_.valid_ = true;
    str.parking_stop_.arrival_.timestamp_ =
        j.stops_[p.from_].departure_.timestamp_ + first_edge_duration;
    str.parking_stop_.arrival_.schedule_timestamp_ =
        j.stops_[p.from_].departure_.schedule_timestamp_ + first_edge_duration;
    str.parking_stop_.arrival_.timestamp_reason_ =
        j.stops_[p.from_].departure_.timestamp_reason_;
    str.parking_stop_.departure_ = str.parking_stop_.arrival_;

    auto& car_transport =
        car_first ? str.first_transport_ : str.second_transport_;
    auto& foot_transport =
        car_first ? str.second_transport_ : str.first_transport_;

    car_transport.mumo_type_ = to_string(mumo_type::CAR);
    foot_transport.mumo_type_ = to_string(mumo_type::FOOT);
  }
}

void apply_gbfs_patches(journey& j, std::vector<parking_patch>& patches) {
  for (auto const& p : patches) {
    // station bike:
    // replace: X --walk[type:gbfs]--> P
    // to: X --walk--> (SX) --bike--> (SP) --walk--> P
    // replace: P -->walk[type:gbfs]--> X
    // to: P --walk--> (SP) --bike--> (SX) --walk--> X
    if (std::holds_alternative<gbfs_edge::station_bike>(p.e_->gbfs_->bike_)) {
      auto const& s = std::get<gbfs_edge::station_bike>(p.e_->gbfs_->bike_);

      auto& t = get_transport(j, p.from_, p.to_);
      auto str1 = split_transport(j, patches, t);
      split_transport(j, patches, str1.second_transport_);

      auto& s1 = j.stops_.at(p.from_ + 1);
      s1.eva_no_ = s.from_station_id_;
      s1.name_ = s.from_station_name_;
      s1.lat_ = s.from_station_pos_.lat_;
      s1.lng_ = s.from_station_pos_.lng_;
      s1.arrival_.valid_ = true;
      s1.arrival_.timestamp_ =
          j.stops_[p.from_].departure_.timestamp_ + s.first_walk_duration_ * 60;
      s1.arrival_.schedule_timestamp_ = s1.arrival_.timestamp_;
      s1.arrival_.timestamp_reason_ =
          j.stops_[p.from_].departure_.timestamp_reason_;
      s1.departure_ = s1.arrival_;

      auto& s2 = j.stops_.at(p.from_ + 2);
      s2.eva_no_ = s.to_station_id_;
      s2.name_ = s.to_station_name_;
      s2.lat_ = s.to_station_pos_.lat_;
      s2.lng_ = s.to_station_pos_.lng_;
      s2.arrival_.valid_ = true;
      s2.arrival_.timestamp_ = s1.departure_.timestamp_ + s.bike_duration_ * 60;
      s2.arrival_.schedule_timestamp_ = s2.arrival_.timestamp_;
      s2.arrival_.timestamp_reason_ =
          j.stops_[p.from_ + 1].departure_.timestamp_reason_;
      s2.departure_ = s2.arrival_;

      get_transport(j, p.from_, p.from_ + 1).mumo_type_ =
          to_string(mumo_type::FOOT);
      get_transport(j, p.from_ + 1, p.from_ + 2).mumo_type_ =
          p.e_->gbfs_->vehicle_type_;
      get_transport(j, p.from_ + 2, p.from_ + 3).mumo_type_ =
          to_string(mumo_type::FOOT);
    }

    // free bike:
    // replace: X --walk[type:gbfs]--> P
    // to: X --walk--> (B) --bike--> P
    // replace: P -->walk[type:gbfs]--> X
    // to: P --walk--> (B) --bike--> X
    else if (std::holds_alternative<gbfs_edge::free_bike>(p.e_->gbfs_->bike_)) {
      auto const& b = std::get<gbfs_edge::free_bike>(p.e_->gbfs_->bike_);

      auto& t = get_transport(j, p.from_, p.to_);
      auto str = split_transport(j, patches, t);

      str.parking_stop_.eva_no_ = b.id_;
      str.parking_stop_.name_ = b.id_;
      str.parking_stop_.lat_ = b.pos_.lat_;
      str.parking_stop_.lng_ = b.pos_.lng_;
      str.parking_stop_.arrival_.valid_ = true;
      str.parking_stop_.arrival_.timestamp_ =
          j.stops_[p.from_].departure_.timestamp_ + b.walk_duration_ * 60;
      str.parking_stop_.arrival_.schedule_timestamp_ =
          str.parking_stop_.arrival_.timestamp_;
      str.parking_stop_.arrival_.timestamp_reason_ =
          j.stops_[p.from_].departure_.timestamp_reason_;
      str.parking_stop_.departure_ = str.parking_stop_.arrival_;

      str.first_transport_.mumo_type_ = to_string(mumo_type::FOOT);
      str.second_transport_.mumo_type_ = to_string(mumo_type::BIKE);
    }
  }
}

std::size_t remove_not_available_od_journeys(std::vector<journey>& journeys,
                                  std::vector<ondemand_patch>& od_patches,
                                  std::vector<availability_response>& ares) {
  if (ares.empty() || od_patches.empty()) {
    return 0;
  }
  auto const all = journeys.size();
  int journey_id = 0;
  utl::erase_if(journeys, [&](motis::journey j) {
    bool found_journey = false, to_short = false, not_available = false;
    int idx_count = 0;
    if (!std::any_of(begin(j.transports_), end(j.transports_),
                     [&](journey::transport const& t) {
                       return t.mumo_type_ == to_string(mumo_type::ON_DEMAND);
                     })) {
      //printf("how often?\n");
      return false;
    } else {
      for (auto const& p : od_patches) {
        if (journey_id == p.j_id_) {
          found_journey = true;
          if (p.e_->duration_ <= 5.0) {
            to_short = true;
          }
          if (!ares.at(idx_count).available_) {
            not_available = true;
            break;
          }
        }
        idx_count++;
      }
      journey_id++;
    }
    return found_journey && (to_short || not_available);
  });
  return all - journeys.size();
}

void apply_ondemand_patches(journey& j, std::vector<parking_patch>& patches,
                            std::vector<availability_response> v_ares) {
  int i = 0;
  for(auto const& patch : patches) {
    availability_response ares = v_ares.at(i);
    if(ares.walk_dur_.empty() || !ares.available_
        || (ares.walk_dur_.at(0) == 0 && ares.walk_dur_.at(1) == 0)) {
      i++;
      continue;
    }
    /*
     *  replace: S --od--> T
     *  with:    S --walk--> PU --od--> T
     *  replace: T --od--> S
     *  with:    T --walk--> PU --od--> S
    */
    if(ares.walk_dur_.at(0) != 0 && ares.walk_dur_.at(1) == 0) {
      auto& t1 = get_transport(j, patch.from_, patch.to_);
      auto splitted_one = split_transport(j, patches, t1);

      splitted_one.parking_stop_.eva_no_ = ares.codenumber_id_;
      splitted_one.parking_stop_.name_ = ares.codenumber_id_;
      splitted_one.parking_stop_.lat_ = ares.startpoint_.lat_;
      splitted_one.parking_stop_.lng_ = ares.startpoint_.lng_;
      splitted_one.parking_stop_.arrival_.valid_ = true;
      splitted_one.parking_stop_.arrival_.timestamp_ =
          j.stops_[patch.from_].departure_.timestamp_ +
          ares.walk_dur_.at(0);
      splitted_one.parking_stop_.arrival_.schedule_timestamp_ =
          splitted_one.parking_stop_.arrival_.timestamp_;
      splitted_one.parking_stop_.arrival_.timestamp_reason_ =
          j.stops_[patch.from_].departure_.timestamp_reason_;
      splitted_one.parking_stop_.departure_ = splitted_one.parking_stop_.arrival_;

      splitted_one.first_transport_.mumo_type_ = to_string(mumo_type::FOOT);
      splitted_one.second_transport_.mumo_type_ = to_string(mumo_type::ON_DEMAND);
    }
    /*
     *  replace: S --od--> T
     *  with:    S --od--> DO --walk--> T
     *  replace: T --od--> S
     *  with:    T --od--> DO --walk--> S
    */
    else if(ares.walk_dur_.at(0) == 0 && ares.walk_dur_.at(1) != 0) {
      auto& t2 = get_transport(j, patch.from_, patch.to_);
      auto splitted_two = split_transport(j, patches, t2);

      splitted_two.parking_stop_.eva_no_ = ares.codenumber_id_;
      splitted_two.parking_stop_.name_ = ares.codenumber_id_;
      splitted_two.parking_stop_.lat_ = ares.startpoint_.lat_;
      splitted_two.parking_stop_.lng_ = ares.startpoint_.lng_;
      splitted_two.parking_stop_.arrival_.valid_ = true;
      splitted_two.parking_stop_.arrival_.timestamp_ =
          j.stops_[patch.from_].departure_.timestamp_ +
          ares.walk_dur_.at(1);
      splitted_two.parking_stop_.arrival_.schedule_timestamp_ =
          splitted_two.parking_stop_.arrival_.timestamp_;
      splitted_two.parking_stop_.arrival_.timestamp_reason_ =
          j.stops_[patch.from_].departure_.timestamp_reason_;
      splitted_two.parking_stop_.departure_ = splitted_two.parking_stop_.arrival_;

      splitted_two.first_transport_.mumo_type_ = to_string(mumo_type::ON_DEMAND);
      splitted_two.second_transport_.mumo_type_ = to_string(mumo_type::FOOT);
    }
    /*
     *  replace: S --od--> T
     *  with:    S --walk--> PU --od--> DO --walk--> T
     *  replace: T --od--> S
     *  with:    T --walk--> PU --od--> DO --walk--> S
    */
    else if(ares.walk_dur_.at(0) != 0 && ares.walk_dur_.at(1) != 0) {
      auto& t3 = get_transport(j, patch.from_, patch.to_);
      auto splitted_three = split_transport(j, patches, t3);
      split_transport(j, patches, splitted_three.second_transport_);

      auto& split3 = j.stops_.at(patch.from_ + 1);
      split3.eva_no_ = ares.codenumber_id_;
      split3.name_ = ares.codenumber_id_;
      split3.lat_ = ares.startpoint_.lat_;
      split3.lng_ = ares.startpoint_.lng_;
      split3.arrival_.valid_ = true;
      split3.arrival_.timestamp_ = j.stops_[patch.from_].departure_.timestamp_
          + ares.walk_dur_.at(0);
      split3.arrival_.schedule_timestamp_ = split3.arrival_.timestamp_;
      split3.arrival_.timestamp_reason_ =
          j.stops_[patch.from_].departure_.timestamp_reason_;
      split3.departure_ = split3.arrival_;

      auto& split4 = j.stops_.at(patch.from_ + 2);
      split4.eva_no_ = ares.codenumber_id_;
      split4.name_ = ares.codenumber_id_;
      split4.lat_ = ares.endpoint_.lat_;
      split4.lng_ = ares.endpoint_.lng_;
      split4.arrival_.valid_ = true;
      split4.arrival_.timestamp_ =
          split3.departure_.timestamp_ + (ares.dropoff_time_ - ares.pickup_time_);
      split4.arrival_.schedule_timestamp_ = split4.arrival_.timestamp_;
      split4.arrival_.timestamp_reason_ =
          j.stops_[patch.from_ + 1].departure_.timestamp_reason_;
      split4.departure_ = split4.arrival_;

      get_transport(j, patch.from_, patch.from_ + 1).mumo_type_ =
          to_string(mumo_type::FOOT);
      get_transport(j, patch.from_ + 1, patch.from_ + 2).mumo_type_ =
          to_string(mumo_type::ON_DEMAND);
      get_transport(j, patch.from_ + 2, patch.from_ + 3).mumo_type_ =
          to_string(mumo_type::FOOT);
    }
    i++;
  }
}

availability_response ondemand_availability(journey j, bool start, mumo_edge const& e,
                                             statistics& stats, std::vector<std::string> const& server_info) {
  availability_request areq;
  if(j.transports_.size() == 1) {
    areq.direct_con_ = true;
    areq.duration_ = static_cast<int>(round(j.transports_.at(0).duration_ * 60));
  } else {
    areq.direct_con_ = false;
    areq.duration_ = static_cast<int>(round(e.duration_ * 60));
  }
  areq.startpoint_.lat_ = e.from_pos_.lat_;
  areq.startpoint_.lng_ = e.from_pos_.lng_;
  areq.endpoint_.lat_ = e.to_pos_.lat_;
  areq.endpoint_.lng_ = e.to_pos_.lng_;
  if(start) {
    areq.start_ = true;
    areq.departure_time_ = j.stops_.front().departure_.timestamp_;
    areq.arrival_time_onnext_ = j.stops_.at(1).arrival_.timestamp_;
  }
  else {
    areq.start_ = false;
    int lastindex = static_cast<int>(j.stops_.size()) - 1;
    areq.departure_time_ = j.stops_.at(lastindex - 1).departure_.timestamp_;
    areq.arrival_time_ = j.stops_.at(lastindex - 1).arrival_.timestamp_;
    areq.arrival_time_onnext_ = j.stops_.back().arrival_.timestamp_;
  }
  MOTIS_START_TIMING(ondemand_check);
  availability_response ares = check_od_availability(areq, server_info, stats);
  MOTIS_STOP_TIMING(ondemand_check);
  stats.ondemand_check_availability_ +=
      static_cast<uint64_t>(MOTIS_TIMING_MS(ondemand_check));
  return ares;
}


msg_ptr postprocess_response(msg_ptr const& response_msg,
                             query_start const& q_start,
                             query_dest const& q_dest,
                             IntermodalRoutingRequest const* req,
                             std::vector<mumo_edge const*> const& edge_mapping,
                             statistics& stats, bool const revise,
                             std::vector<stats_category> const& mumo_stats,
                             ppr_profiles const& profiles,
                             std::vector<std::string> const& server_infos) {
  doctorwho++;
  printf("----------------------------------------------------------------------COUNT: %d\n", doctorwho);
  MOTIS_START_TIMING(post_timing);
  auto const dir = req->search_dir();
  auto routing_response =
      response_msg ? motis_content(RoutingResponse, response_msg) : nullptr;
  auto journeys = routing_response == nullptr
                      ? std::vector<journey>{}
                      : message_to_journeys(routing_response);
  printf("    JOURNEYS: %llu\n", journeys.size());
  stats.journey_count_begin_ += journeys.size();

  MOTIS_START_TIMING(direct_connection_timing);
  auto const direct =
      get_direct_connections(q_start, q_dest, req, profiles, edge_mapping);
  stats.dominated_by_direct_connection_ =
      remove_dominated_journeys(journeys, direct);
  add_direct_connections(journeys, direct, q_start, q_dest, req);
  MOTIS_STOP_TIMING(direct_connection_timing);
  stats.direct_connection_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_MS(direct_connection_timing));

  auto ondemand_patches = std::vector<ondemand_patch>{};
  std::vector<availability_response> vares;
  int journey_id = 0;
  auto checked_to = std::vector<geo::latlng>{};
  auto checked_from = std::vector<geo::latlng>{};
  bool area = false;

  message_creator mc;
  for (auto& journey : journeys) {
    auto& stops = journey.stops_;
    if (stops.size() < 2) {
      continue;
    }

    if (q_start.is_intermodal_) {
      auto& start = (dir == SearchDir_Forward) ? stops.front() : stops.back();
      start.lat_ = q_start.pos_.lat_;
      start.lng_ = q_start.pos_.lng_;
    }

    if (q_dest.is_intermodal_) {
      auto& dest = (dir == SearchDir_Forward) ? stops.back() : stops.front();
      dest.lat_ = q_dest.pos_.lat_;
      dest.lng_ = q_dest.pos_.lng_;
    }

    bool ondemand_journey = false;
    auto gbfs_patches = std::vector<parking_patch>{};
    auto parking_patches = std::vector<parking_patch>{};
    auto ondemand_parking_patches = std::vector<parking_patch>{};
    std::vector<availability_response> v_ares;
    availability_response ares;
    for (auto& t : journey.transports_) {
      if (!t.is_walk_ || t.mumo_id_ < 0) {
        continue;
      }
      auto const e = edge_mapping.at(static_cast<std::size_t>(t.mumo_id_));
      t.mumo_type_ = to_string(e->type_);
      t.mumo_id_ = e->id_;

      if (e->type_ == mumo_type::CAR_PARKING && e->car_parking_) {
        if (!e->car_parking_->uses_car_) {
          t.mumo_type_ = to_string(mumo_type::FOOT);
          continue;
        }
        parking_patches.emplace_back(e, t.from_, t.to_);
      } else if (e->type_ == mumo_type::GBFS) {
        gbfs_patches.emplace_back(e, t.from_, t.to_);
      } else if(e->type_ == mumo_type::ON_DEMAND) {
        ondemand_journey = true;
        if(!std::any_of(std::begin(checked_to), std::end(checked_to),
                        [&](geo::latlng pos){
                          return pos == e->to_pos_;})
            && !std::any_of(std::begin(checked_from), std::end(checked_from),
                        [&](geo::latlng pos){
                          return pos == e->from_pos_;})) {
          area = check_od_area(e->from_pos_, e->to_pos_, server_infos, stats);
          checked_from.emplace_back(e->from_pos_);
          checked_to.emplace_back(e->to_pos_);
        }
        if(q_start.is_intermodal_ && q_start.pos_.lat_ == e->from_pos_.lat_
            && q_start.pos_.lng_ == e->from_pos_.lng_ && area) {
          ares = ondemand_availability(journey, true, *e, stats, server_infos);
        }
        if(q_dest.is_intermodal_ && q_dest.pos_.lat_ == e->to_pos_.lat_
            && q_dest.pos_.lng_ == e->to_pos_.lng_ && area) {
          ares = ondemand_availability(journey, false, *e, stats, server_infos);
        }
        vares.emplace_back(ares);
        v_ares.emplace_back(ares);
        ondemand_patches.emplace_back(e, t.from_, t.to_, journey, journey_id);
        ondemand_parking_patches.emplace_back(e, t.from_, t.to_);
      }
    }
    apply_parking_patches(journey, parking_patches);
    apply_gbfs_patches(journey, gbfs_patches);
    apply_ondemand_patches(journey, ondemand_parking_patches, v_ares);
    journey_id++;
    if(ondemand_journey) {
      stats.ondemand_journey_count_++;
    }
  }
  MOTIS_START_TIMING(ondemand_remove);
  stats.ondemand_removed_journeys_ =
   remove_not_available_od_journeys(journeys, ondemand_patches, vares);
  MOTIS_STOP_TIMING(ondemand_remove);
  stats.ondemand_remove_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_US(ondemand_remove));
  stats.journey_count_end_ += journeys.size();

  utl::erase_if(journeys, [](journey const& j) { return j.stops_.empty(); });
  std::sort(
      begin(journeys), end(journeys), [](journey const& a, journey const& b) {
        return std::make_pair(a.stops_.front().departure_.schedule_timestamp_,
                              a.stops_.back().arrival_.schedule_timestamp_) <
               std::make_pair(b.stops_.front().departure_.schedule_timestamp_,
                              b.stops_.back().arrival_.schedule_timestamp_);
      });

  auto const connections = revise
                               ? revise_connections(journeys, stats, mc)
                               : utl::to_vec(journeys, [&mc](journey const& j) {
                                   return to_connection(mc, j);
                                 });
  MOTIS_STOP_TIMING(post_timing);
  stats.postprocess_timing_ = static_cast<uint64_t>(MOTIS_TIMING_MS(post_timing));

  auto all_stats =
      routing_response == nullptr
          ? std::vector<Offset<Statistics>>{}
          : utl::to_vec(*routing_response->statistics(),
                        [&](Statistics const* stats) {
                          return motis_copy_table(Statistics, mc, stats);
                        });
  for (auto const& s : mumo_stats) {
    all_stats.emplace_back(to_fbs(mc, s));
  }
  all_stats.emplace_back(to_fbs(mc, to_stats_category("intermodal", stats)));

  auto interval_begin = uint64_t{};
  auto interval_end = uint64_t{};
  if (routing_response != nullptr) {
    interval_begin = routing_response->interval_begin();
    interval_end = routing_response->interval_end();
  } else {
    switch (req->start_type()) {
      case IntermodalStart_PretripStart:
        interval_begin =
            reinterpret_cast<IntermodalPretripStart const*>(req->start())
                ->interval()
                ->begin();
        interval_end =
            reinterpret_cast<IntermodalPretripStart const*>(req->start())
                ->interval()
                ->begin();
        break;

      case IntermodalStart_IntermodalPretripStart:
        interval_begin = reinterpret_cast<PretripStart const*>(req->start())
                             ->interval()
                             ->begin();
        interval_end = reinterpret_cast<PretripStart const*>(req->start())
                           ->interval()
                           ->begin();
        break;

      case IntermodalStart_IntermodalOntripStart:
        interval_begin = interval_end =
            reinterpret_cast<IntermodalOntripStart const*>(req->start())
                ->departure_time();
        break;

      case IntermodalStart_OntripStationStart:
        interval_begin = interval_end =
            reinterpret_cast<OntripStationStart const*>(req->start())
                ->departure_time();
        break;

      case IntermodalStart_OntripTrainStart:
        interval_begin = interval_end =
            reinterpret_cast<OntripTrainStart const*>(req->start())
                ->arrival_time();
        break;

      case IntermodalStart_NONE: break;
    }
  }

  mc.create_and_finish(
      MsgContent_RoutingResponse,
      CreateRoutingResponse(
          mc, mc.CreateVectorOfSortedTables(&all_stats),
          mc.CreateVector(connections), interval_begin, interval_end,
          mc.CreateVector(utl::to_vec(
              direct,
              [&mc](direct_connection const& c) { return to_fbs(mc, c); })))
          .Union());
  return make_msg(mc);
}

msg_ptr intermodal::route(msg_ptr const& msg) {
  auto const req = motis_content(IntermodalRoutingRequest, msg);
  message_creator mc;
  statistics stats{};

  masterwho++;
  printf("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| master %d\n", masterwho);

  auto const& sched = get_sched();
  auto const start = parse_query_start(mc, req, sched);
  auto const dest = parse_query_dest(mc, req, sched);

  stats.linear_distance_ =
      static_cast<uint64_t>(distance(start.pos_, dest.pos_));

  auto appender = [](auto& vec, auto const& from, auto const& to,
                     auto const& from_pos, auto const& to_pos, auto const dur,
                     auto const accessibility, mumo_type const type,
                     int const id) -> mumo_edge& {
    return vec.emplace_back(from, to, from_pos, to_pos, dur, accessibility,
                            type, id);
  };

  MOTIS_START_TIMING(mumo_edge_timing);
  std::vector<mumo_edge> deps;
  std::vector<mumo_edge> arrs;
  std::vector<stats_category> mumo_stats;
  std::mutex mumo_stats_mutex;

  auto const mumo_stats_appender = [&](stats_category&& s) {
    std::lock_guard guard(mumo_stats_mutex);
    mumo_stats.emplace_back(s);
  };

  std::vector<ctx::future_ptr<ctx_data, void>> futures;

  using namespace std::placeholders;
  if (req->search_dir() == SearchDir_Forward) {
    if (start.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_starts(
            req, start.pos_, dest.pos_,
            std::bind(appender, std::ref(deps),  // NOLINT
                      STATION_START, _1, start.pos_, _2, _3, _4, _5, _6),
            mumo_stats_appender, ppr_profiles_);
      }));
    }
    if (dest.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_dests(req, dest.pos_, start.pos_,
                   std::bind(appender, std::ref(arrs),  // NOLINT
                             _1, STATION_END, _2, dest.pos_, _3, _4, _5, _6),
                   mumo_stats_appender, ppr_profiles_);
      }));
    }
  } else {
    if (start.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_starts(
            req, start.pos_, dest.pos_,
            std::bind(appender, std::ref(deps),  // NOLINT
                      _1, STATION_START, _2, start.pos_, _3, _4, _5, _6),
            mumo_stats_appender, ppr_profiles_);
      }));
    }
    if (dest.is_intermodal_) {
      futures.emplace_back(spawn_job_void([&]() {
        make_dests(req, dest.pos_, start.pos_,
                   std::bind(appender, std::ref(arrs),  // NOLINT
                             STATION_END, _1, dest.pos_, _2, _3, _4, _5, _6),
                   mumo_stats_appender, ppr_profiles_);
      }));
    }
  }

  ctx::await_all(futures);
  MOTIS_STOP_TIMING(mumo_edge_timing);
  stats.start_edges_ = deps.size();
  stats.destination_edges_ = arrs.size();
  stats.mumo_edge_duration_ =
      static_cast<uint64_t>(MOTIS_TIMING_MS(mumo_edge_timing));

  std::vector<mumo_edge const*> edge_mapping;
  auto edges = write_edges(mc, deps, arrs, edge_mapping);

  auto routing_resp = msg_ptr{};
  if ((!start.is_intermodal_ || !deps.empty()) &&
      (!dest.is_intermodal_ || !arrs.empty())) {
    auto const router = ((req->search_type() == SearchType_Default ||
                          req->search_type() == SearchType_Accessibility) &&
                         start.start_type_ != Start_OntripTrainStart)
                            ? router_
                            : "/routing";

    mc.create_and_finish(
        MsgContent_RoutingRequest,
        CreateRoutingRequest(mc, start.start_type_, start.start_, dest.station_,
                             req->search_type(), req->search_dir(),
                             mc.CreateVector(std::vector<Offset<Via>>{}),
                             mc.CreateVector(edges))
            .Union(),
        router);

    MOTIS_START_TIMING(routing_timing);
    routing_resp = motis_call(make_msg(mc))->val();
    MOTIS_STOP_TIMING(routing_timing);

    stats.routing_duration_ =
        static_cast<uint64_t>(MOTIS_TIMING_MS(routing_timing));
  }
  return postprocess_response(routing_resp, start, dest, req, edge_mapping,
                              stats, revise_, mumo_stats, ppr_profiles_, ondemand_infos_);
}

}  // namespace motis::intermodal

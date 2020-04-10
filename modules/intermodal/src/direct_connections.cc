#include "motis/intermodal/direct_connections.h"

#include <algorithm>
#include <utility>

#include "utl/erase_if.h"

#include "ppr/routing/search_profile.h"

using namespace geo;
using namespace flatbuffers;
using namespace ppr::routing;
using namespace motis::routing;
using namespace motis::ppr;
using namespace motis::module;

namespace motis::intermodal {

std::pair<std::string, double> get_ppr_settings(
    Vector<Offset<ModeWrapper>> const* modes) {
  std::string profile;
  double duration = 0;

  for (auto const& m : *modes) {
    if (m->mode_type() == Mode_FootPPR) {
      auto const options =
          reinterpret_cast<FootPPR const*>(m->mode())->search_options();
      profile = options->profile()->str();
      duration = options->duration_limit();
    }
  }

  return {profile, duration};
}

std::pair<std::string, double> get_direct_ppr_profile(
    IntermodalRoutingRequest const* req) {
  auto const [start_profile, start_duration] =  // NOLINT
      get_ppr_settings(req->start_modes());
  auto const [dest_profile, dest_duration] =  // NOLINT
      get_ppr_settings(req->destination_modes());

  auto const total_duration = start_duration + dest_duration;
  auto const profile =
      req->search_dir() == SearchDir_Forward ? start_profile : dest_profile;

  return {profile, total_duration};
}

inline double get_max_distance(std::string const& profile, double duration,
                               ppr_profiles const& profiles) {
  return profiles.get_walking_speed(profile) * duration;
}

msg_ptr make_direct_ppr_request(geo::latlng const& start,
                                geo::latlng const& dest,
                                std::string const& ppr_profile,
                                double ppr_duration_limit,
                                SearchDir direction) {
  Position const start_pos{start.lat_, start.lng_};
  Position const dest_pos{dest.lat_, dest.lng_};
  message_creator mc;

  auto const dir = direction == SearchDir_Forward ? SearchDirection_Forward
                                                  : SearchDirection_Backward;

  mc.create_and_finish(
      MsgContent_FootRoutingRequest,
      CreateFootRoutingRequest(
          mc, &start_pos,
          mc.CreateVectorOfStructs(std::vector<Position>{dest_pos}),
          CreateSearchOptions(mc, mc.CreateString(ppr_profile),
                              ppr_duration_limit),
          dir, false, false, false)
          .Union(),
      "/ppr/route");
  return make_msg(mc);
}

std::vector<direct_connection> get_direct_connections(
    query_start const& q_start, query_dest const& q_dest,
    IntermodalRoutingRequest const* req, ppr_profiles const& profiles) {
  std::vector<direct_connection> direct;

  auto const [profile, duration] = get_direct_ppr_profile(req);  // NOLINT
  if (duration == 0 || distance(q_start.pos_, q_dest.pos_) >
                           get_max_distance(profile, duration, profiles)) {
    return direct;
  }

  auto const ppr_msg =
      motis_call(make_direct_ppr_request(q_start.pos_, q_dest.pos_, profile,
                                         duration, req->search_dir()))
          ->val();
  auto const ppr_resp = motis_content(FootRoutingResponse, ppr_msg);
  auto const routes = ppr_resp->routes();
  if (routes->size() == 1) {
    for (auto const& route : *routes->Get(0)->routes()) {
      direct.emplace_back(mumo_type::FOOT, route->duration(),
                          route->accessibility());
    }
  }

  return direct;
}

std::size_t remove_dominated_journeys(
    std::vector<journey>& journeys,
    std::vector<direct_connection> const& direct) {
  auto const before = journeys.size();
  utl::erase_if(journeys, [&](journey const& j) {
    auto const jd = static_cast<unsigned>(1.2 * j.duration_);
    return std::any_of(
        begin(direct), end(direct), [&](direct_connection const& d) {
          return (d.duration_ < jd && d.accessibility_ <= j.accessibility_) ||
                 (d.duration_ <= jd && d.accessibility_ < j.accessibility_);
        });
  });
  return before - journeys.size();
}

void add_direct_connections(std::vector<journey>& journeys,
                            std::vector<direct_connection> const& direct,
                            query_start const& q_start,
                            query_dest const& q_dest,
                            IntermodalRoutingRequest const* req) {
  auto const fwd = req->search_dir() == SearchDir_Forward;
  for (auto const& d : direct) {
    auto const dep_time =
        fwd ? q_start.time_
            : static_cast<std::time_t>(q_start.time_ - d.duration_ * 60);
    auto const arr_time =
        fwd ? static_cast<std::time_t>(q_start.time_ + d.duration_ * 60)
            : q_start.time_;

    auto& j = journeys.emplace_back();
    auto& start = j.stops_.emplace_back();
    start.name_ = STATION_START;
    start.eva_no_ = STATION_START;
    start.lat_ = q_start.pos_.lat_;
    start.lng_ = q_start.pos_.lng_;
    start.departure_.valid_ = true;
    start.departure_.timestamp_ = dep_time;
    start.departure_.schedule_timestamp_ = dep_time;
    auto& dest = j.stops_.emplace_back();
    dest.name_ = STATION_END;
    dest.eva_no_ = STATION_END;
    dest.lat_ = q_dest.pos_.lat_;
    dest.lng_ = q_dest.pos_.lng_;
    dest.arrival_.valid_ = true;
    dest.arrival_.timestamp_ = arr_time;
    dest.arrival_.schedule_timestamp_ = arr_time;
    auto& transport = j.transports_.emplace_back();
    transport.from_ = 0;
    transport.to_ = 1;
    transport.is_walk_ = true;
    transport.duration_ = d.duration_;
    transport.mumo_accessibility_ = d.accessibility_;
    transport.mumo_type_ = to_string(d.type_);
  }
}

Offset<DirectConnection> to_fbs(FlatBufferBuilder& fbb,
                                direct_connection const& c) {
  return CreateDirectConnection(fbb, c.duration_, c.accessibility_,
                                fbb.CreateString(to_string(c.type_)));
}

}  // namespace motis::intermodal
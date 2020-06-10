#include "motis/intermodal/direct_connections.h"

#include <algorithm>
#include <mutex>
#include <utility>

#include "utl/erase_if.h"
#include "utl/struct/comparable.h"
#include "utl/verify.h"

#include "ppr/routing/search_profile.h"

#include "motis/module/context/motis_spawn.h"

using namespace geo;
using namespace flatbuffers;
using namespace ppr::routing;
using namespace motis::routing;
using namespace motis::ppr;
using namespace motis::module;
using namespace motis::osrm;

namespace motis::intermodal {

struct ppr_settings {
  std::string profile_;
  double max_duration_{};  // seconds
  double max_distance_{};  // meters
};

inline double get_max_distance(std::string const& profile, double duration,
                               ppr_profiles const& profiles) {
  return profiles.get_walking_speed(profile) * duration;
}

ppr_settings get_ppr_settings(Vector<Offset<ModeWrapper>> const* modes,
                              ppr_profiles const& profiles) {
  auto settings = ppr_settings{};

  for (auto const& m : *modes) {
    if (m->mode_type() == Mode_FootPPR) {
      auto const options =
          reinterpret_cast<FootPPR const*>(m->mode())->search_options();
      settings.profile_ = options->profile()->str();
      settings.max_duration_ = options->duration_limit();
      settings.max_distance_ =
          get_max_distance(settings.profile_, settings.max_duration_, profiles);
    }
  }

  return settings;
}

ppr_settings get_direct_ppr_settings(IntermodalRoutingRequest const* req,
                                     ppr_profiles const& profiles) {
  auto const start_settings = get_ppr_settings(req->start_modes(), profiles);
  auto const dest_settings =
      get_ppr_settings(req->destination_modes(), profiles);

  auto const& profile = req->search_dir() == SearchDir_Forward
                            ? start_settings.profile_
                            : dest_settings.profile_;
  auto const max_duration =
      start_settings.max_duration_ + dest_settings.max_duration_;
  auto const max_distance =
      max_duration > 0 ? get_max_distance(profile, max_duration, profiles) : 0;

  return {profile, max_duration, max_distance};
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

struct osrm_settings {
  MAKE_COMPARABLE()
  int max_duration_{};  // seconds
  double max_distance_{};  // meters
};

inline osrm_settings operator+(osrm_settings const& lhs,
                               osrm_settings const& rhs) {
  return {lhs.max_duration_ + rhs.max_duration_,
          lhs.max_distance_ + rhs.max_distance_};
}

template <Mode ModeType>
osrm_settings get_osrm_settings(Vector<Offset<ModeWrapper>> const* modes) {
  auto settings = osrm_settings{};

  for (auto const& m : *modes) {
    if (m->mode_type() == ModeType) {
      switch (ModeType) {
        case Mode_Bike: {
          settings.max_duration_ =
              reinterpret_cast<Bike const*>(m->mode())->max_duration();
          settings.max_distance_ = settings.max_duration_ * BIKE_SPEED;
          break;
        }
        case Mode_Car: {
          settings.max_duration_ =
              reinterpret_cast<Car const*>(m->mode())->max_duration();
          settings.max_distance_ = settings.max_duration_ * CAR_SPEED;
          break;
        }
        case Mode_CarParking: {
          settings.max_duration_ =
              reinterpret_cast<CarParking const*>(m->mode())
                  ->max_car_duration();
          settings.max_distance_ = settings.max_duration_ * CAR_SPEED;
          break;
        }
        case Mode_Foot: {
          settings.max_duration_ =
              reinterpret_cast<Foot const*>(m->mode())->max_duration();
          settings.max_distance_ = settings.max_duration_ * WALK_SPEED;
          break;
        }
        default: {
          throw utl::fail("direct connections: unsupported osrm mode");
        }
      }
    }
  }

  return settings;
}

template <Mode ModeType>
osrm_settings get_direct_osrm_settings(IntermodalRoutingRequest const* req) {
  return get_osrm_settings<ModeType>(req->start_modes()) +
         get_osrm_settings<ModeType>(req->destination_modes());
}

osrm_settings get_direct_osrm_car_settings(
    IntermodalRoutingRequest const* req) {
  auto const start_settings =
      std::max(get_osrm_settings<Mode_Car>(req->start_modes()),
               get_osrm_settings<Mode_CarParking>(req->start_modes()));
  auto const dest_settings =
      std::max(get_osrm_settings<Mode_Car>(req->destination_modes()),
               get_osrm_settings<Mode_CarParking>(req->destination_modes()));
  return start_settings + dest_settings;
}

msg_ptr make_direct_osrm_request(geo::latlng const& start,
                                 geo::latlng const& dest,
                                 std::string const& profile,
                                 SearchDir direction) {
  Position const start_pos{start.lat_, start.lng_};
  Position const dest_pos{dest.lat_, dest.lng_};
  message_creator mc;

  auto const dir =
      direction == SearchDir_Forward ? Direction_Forward : Direction_Backward;

  mc.create_and_finish(
      MsgContent_OSRMOneToManyRequest,
      CreateOSRMOneToManyRequest(
          mc, mc.CreateString(profile), dir, &start_pos,
          mc.CreateVectorOfStructs(std::vector<Position>{dest_pos}))
          .Union(),
      "/osrm/one_to_many");
  return make_msg(mc);
}

std::vector<direct_connection> get_direct_connections(
    query_start const& q_start, query_dest const& q_dest,
    IntermodalRoutingRequest const* req, ppr_profiles const& profiles) {
  auto direct = std::vector<direct_connection>{};
  auto const beeline = distance(q_start.pos_, q_dest.pos_);

  auto direct_mutex = std::mutex{};
  auto futures = std::vector<ctx::future_ptr<ctx_data, void>>{};

  auto const ppr_settings = get_direct_ppr_settings(req, profiles);
  if (ppr_settings.max_duration_ > 0 && beeline <= ppr_settings.max_distance_) {
    futures.emplace_back(spawn_job_void([&]() {
      auto const ppr_msg =
          motis_call(make_direct_ppr_request(
                         q_start.pos_, q_dest.pos_, ppr_settings.profile_,
                         ppr_settings.max_duration_, req->search_dir()))
              ->val();
      auto const ppr_resp = motis_content(FootRoutingResponse, ppr_msg);
      auto const routes = ppr_resp->routes();
      if (routes->size() == 1) {
        std::lock_guard guard{direct_mutex};
        for (auto const& route : *routes->Get(0)->routes()) {
          direct.emplace_back(mumo_type::FOOT, route->duration(),
                              route->accessibility());
        }
      }
    }));
  }

  auto const osrm_bike_settings = get_direct_osrm_settings<Mode_Bike>(req);
  if (osrm_bike_settings.max_duration_ > 0 &&
      beeline <= osrm_bike_settings.max_distance_) {
    futures.emplace_back(spawn_job_void([&]() {
      auto const osrm_msg =
          motis_call(make_direct_osrm_request(q_start.pos_, q_dest.pos_,
                                              to_string(mumo_type::BIKE),
                                              req->search_dir()))
              ->val();
      auto const osrm_resp = motis_content(OSRMOneToManyResponse, osrm_msg);
      utl::verify(osrm_resp->costs()->size() == 1,
                  "direct connetions: invalid osrm response");
      auto const duration =
          static_cast<unsigned>(osrm_resp->costs()->Get(0)->duration());
      if (duration <= osrm_bike_settings.max_duration_) {
        std::lock_guard guard{direct_mutex};
        direct.emplace_back(mumo_type::BIKE, duration / 60, 0);
      }
    }));
  }

  auto const osrm_car_settings = get_direct_osrm_car_settings(req);
  if (osrm_car_settings.max_duration_ > 0 &&
      beeline <= osrm_car_settings.max_distance_) {
    futures.emplace_back(spawn_job_void([&]() {
      auto const osrm_msg =
          motis_call(make_direct_osrm_request(q_start.pos_, q_dest.pos_,
                                              to_string(mumo_type::CAR),
                                              req->search_dir()))
              ->val();
      auto const osrm_resp = motis_content(OSRMOneToManyResponse, osrm_msg);
      utl::verify(osrm_resp->costs()->size() == 1,
                  "direct connetions: invalid osrm response");
      auto const duration =
          static_cast<unsigned>(osrm_resp->costs()->Get(0)->duration());
      if (duration <= osrm_car_settings.max_duration_) {
        std::lock_guard guard{direct_mutex};
        direct.emplace_back(mumo_type::CAR, duration / 60, 0);
      }
    }));
  }

  ctx::await_all(futures);
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

#include "motis/osrm/router.h"

#include "osrm/engine_config.hpp"
#include "osrm/multi_target_parameters.hpp"
#include "osrm/osrm.hpp"
#include "osrm/route_parameters.hpp"
#include "osrm/smooth_via_parameters.hpp"
#include "osrm/table_parameters.hpp"
#include "util/coordinate.hpp"
#include "util/json_container.hpp"
#include "util/json_util.hpp"

#include "utl/to_vec.h"

#include "motis/osrm/error.h"

using namespace flatbuffers;
using namespace motis::module;

using namespace osrm;
using namespace osrm::util;
using namespace osrm::util::json;

namespace motis::osrm {

struct router::impl {
public:
  explicit impl(std::string const& path) {
    EngineConfig config;
    config.storage_config = {path};
    config.use_shared_memory = false;

    osrm_ = std::make_unique<OSRM>(config);
  }

  static FloatCoordinate make_coord(double lat, double lng) {
    return FloatCoordinate{FloatLongitude{lng}, FloatLatitude{lat}};
  }

  msg_ptr table(OSRMManyToManyRequest const* req) const {
    TableParameters params;
    for (auto const& loc : *req->from()) {
      params.sources.emplace_back(params.sources.size());
      params.coordinates.emplace_back(make_coord(loc->lat(), loc->lng()));
    }
    for (auto const& loc : *req->to()) {
      params.destinations.emplace_back(params.sources.size() +
                                       params.destinations.size());
      params.coordinates.emplace_back(make_coord(loc->lat(), loc->lng()));
    }

    Object result;
    osrm_->Table(params, result);

    std::vector<double> durations;
    for (auto const& duration_row :
         result.values["durations"].get<Array>().values) {
      for (auto const& d : duration_row.get<Array>().values) {
        durations.emplace_back(d.get<Number>().value);
      }
    }

    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_OSRMManyToManyResponse,
        CreateOSRMManyToManyResponse(
            fbb, fbb.CreateVector(durations.data(), durations.size()))
            .Union());
    return make_msg(fbb);
  }

  msg_ptr one_to_many(OSRMOneToManyRequest const* req) const {
    MultiTargetParameters params;
    params.forward = req->direction() == SearchDir_Forward;

    params.coordinates.reserve(req->many()->size() + 1);
    params.coordinates.emplace_back(
        make_coord(req->one()->lat(), req->one()->lng()));

    for (auto const& loc : *req->many()) {
      params.coordinates.emplace_back(make_coord(loc->lat(), loc->lng()));
    }

    Object result;
    auto const status = osrm_->MultiTarget(params, result);

    if (status != Status::Ok) {
      throw std::system_error(error::no_routing_response);
    }

    std::vector<Cost> costs;
    for (auto const& cost : result.values["costs"].get<Array>().values) {
      auto const& cost_obj = cost.get<Object>();
      costs.emplace_back(cost_obj.values.at("duration").get<Number>().value,
                         cost_obj.values.at("distance").get<Number>().value);
    }

    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_OSRMOneToManyResponse,
        CreateOSRMOneToManyResponse(fbb, fbb.CreateVectorOfStructs(costs))
            .Union());
    return make_msg(fbb);
  }

  msg_ptr via(OSRMViaRouteRequest const* req) const {
    RouteParameters params;
    params.geometries = RouteParameters::GeometriesType::CoordVec1D;
    params.overview = RouteParameters::OverviewType::Full;

    for (auto const& waypoint : *req->waypoints()) {
      params.coordinates.emplace_back(
          make_coord(waypoint->lat(), waypoint->lng()));
    }

    Object result;
    auto const status = osrm_->Route(params, result);

    if (status != Status::Ok) {
      throw std::system_error(error::no_routing_response);
    }

    auto& all_routes = result.values["routes"];
    if (all_routes.get<Array>().values.empty()) {
      throw std::system_error(error::no_routing_response);
    }

    message_creator mc;
    auto& route = get(all_routes, 0U);
    auto const& polyline =
        utl::to_vec(get(route, "geometry").get<Array>().values,
                    [](auto&& jc) { return jc.template get<Number>().value; });

    mc.create_and_finish(MsgContent_OSRMViaRouteResponse,
                         CreateOSRMViaRouteResponse(
                             mc, get(route, "duration").get<Number>().value,
                             get(route, "distance").get<Number>().value,
                             CreatePolyline(mc, mc.CreateVector(polyline)))
                             .Union());
    return make_msg(mc);
  }

  msg_ptr smooth_via(OSRMSmoothViaRouteRequest const* req) const {
    SmoothViaParameters params;

    for (auto const& waypoint : *req->waypoints()) {
      std::vector<Coordinate> coords;
      for (auto const& pos : *waypoint->positions()) {
        coords.emplace_back(make_coord(pos->lat(), pos->lng()));
      }
      params.waypoints.emplace_back(std::move(coords));
    }

    Object result;
    auto const status = osrm_->SmoothVia(params, result);

    if (status != Status::Ok) {
      throw std::system_error(error::no_routing_response);
    }

    message_creator mc;
    std::vector<Offset<Polyline>> segments;
    for (auto const& json_polyline :
         result.values["geometry"].get<Array>().values) {
      auto const& polyline = utl::to_vec(
          json_polyline.get<Array>().values,
          [](auto&& jc) { return jc.template get<Number>().value; });
      segments.emplace_back(CreatePolyline(mc, mc.CreateVector(polyline)));
    }

    mc.create_and_finish(MsgContent_OSRMSmoothViaRouteResponse,
                         CreateOSRMSmoothViaRouteResponse(
                             mc, result.values["duration"].get<Number>().value,
                             result.values["distance"].get<Number>().value,
                             mc.CreateVector(segments))
                             .Union());
    return make_msg(mc);
  }

  std::unique_ptr<OSRM> osrm_;
};

router::router(std::string const& path)
    : impl_(std::make_unique<router::impl>(path)) {}

router::~router() = default;

msg_ptr router::table(OSRMManyToManyRequest const* req) const {
  return impl_->table(req);
}

msg_ptr router::one_to_many(OSRMOneToManyRequest const* req) const {
  return impl_->one_to_many(req);
}

msg_ptr router::via(OSRMViaRouteRequest const* req) const {
  return impl_->via(req);
}

msg_ptr router::smooth_via(OSRMSmoothViaRouteRequest const* req) const {
  return impl_->smooth_via(req);
}

}  // namespace motis::osrm

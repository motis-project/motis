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

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_spawn.h"
#include "motis/osrm/error.h"

using namespace flatbuffers;
using namespace motis::module;

using namespace osrm;
using namespace osrm::util;
using namespace osrm::util::json;
using namespace motis;

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

  msg_ptr one_to_many(OSRMOneToManyRequest const* req) const {
    MultiTargetParameters params;
    params.forward = req->direction() == Direction_Forward;

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

  void emplace_one_to_many_future(
      std::vector<ctx::future_ptr<ctx_data, void>>& futures,
      OSRMManyToManyRequest const* req, int i, std::vector<Cost>& result) {
    futures.emplace_back(spawn_job_void([&, req, i] {
      message_creator mc;
      Position start = {req->coordinates()->Get(i)->lat(),
                        req->coordinates()->Get(i)->lng()};
      mc.create_and_finish(
          MsgContent_OSRMOneToManyRequest,
          CreateOSRMOneToManyRequest(
              mc, mc.CreateString(req->profile()), Direction_Forward, &start,
              mc.CreateVectorOfStructs(
                  utl::to_vec(*req->coordinates(),
                              [](auto* location) {
                                Position pos{location->lat(), location->lng()};
                                return pos;
                              })))
              .Union(),
          "/osrm/one_to_many");
      try {
        auto const osrm_msg = motis_call(make_msg(mc))->val();
        auto const osrm_resp_in =
            motis_content(OSRMOneToManyResponse, osrm_msg);
        result = utl::to_vec(*osrm_resp_in->costs(), [](auto* cost) {
          return Cost{cost->duration(), cost->distance()};
        });
      } catch (...) {
        std::cout << "Exception on One-To-Many.";
        result = utl::to_vec(*req->coordinates(), [](auto const*) {
          return Cost{100000, 100000};
        });
      }
    }));
  }

  msg_ptr many_to_many(OSRMManyToManyRequest const* req) {
    message_creator mc;
    std::vector<std::vector<Cost>> routing_matrix;
    for (int i = 0; i < req->coordinates()->size();) {
      std::vector<ctx::future_ptr<ctx_data, void>> futures;
      std::cout << "Iteration: " << i << std::endl;
      std::vector<Cost> c1{};
      std::vector<Cost> c2{};
      std::vector<Cost> c3{};
      std::vector<Cost> c4{};
      std::vector<Cost> c5{};
      std::vector<Cost> c6{};
      std::vector<Cost> c7{};
      std::vector<Cost> c8{};

      emplace_one_to_many_future(futures, req, i++, c1);
      auto const is_multi = i + 10 < req->coordinates()->size();
      if (is_multi) {
        emplace_one_to_many_future(futures, req, i++, c2);
        emplace_one_to_many_future(futures, req, i++, c3);
        emplace_one_to_many_future(futures, req, i++, c4);
        emplace_one_to_many_future(futures, req, i++, c5);
        emplace_one_to_many_future(futures, req, i++, c6);
        emplace_one_to_many_future(futures, req, i++, c7);
        emplace_one_to_many_future(futures, req, i++, c8);
      }
      ctx::await_all(futures);
      routing_matrix.push_back(c1);
      if (is_multi) {
        routing_matrix.push_back(c2);
        routing_matrix.push_back(c3);
        routing_matrix.push_back(c4);
        routing_matrix.push_back(c5);
        routing_matrix.push_back(c6);
        routing_matrix.push_back(c7);
        routing_matrix.push_back(c8);
      }
    }
    LOG(logging::info) << routing_matrix.size();
    std::vector<Offset<OSRMManyToManyResponseRow>> fbs_routing_matrix;
    for (auto const& row : routing_matrix) {
      fbs_routing_matrix.emplace_back(
          CreateOSRMManyToManyResponseRow(mc, mc.CreateVectorOfStructs(row)));
    }
    mc.create_and_finish(
        MsgContent_OSRMManyToManyResponse,
        CreateOSRMManyToManyResponse(mc, mc.CreateVector(fbs_routing_matrix))
            .Union());
    return make_msg(mc);
  }

  msg_ptr route(OSRMRouteRequest const* req) {
    RouteParameters params;
    params.steps = true;

    for (auto const& waypoint : *req->coordinates()) {
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

    auto& route = get(all_routes, 0u).get<Object>();

    std::vector<Cost> costs;
    for (auto const& cost : route.values["legs"].get<Array>().values) {
      auto const& cost_obj = cost.get<Object>();
      costs.emplace_back(cost_obj.values.at("duration").get<Number>().value,
                         cost_obj.values.at("distance").get<Number>().value);
    }

    message_creator mc;

    mc.create_and_finish(
        MsgContent_OSRMRouteResponse,
        CreateOSRMRouteResponse(mc, mc.CreateVectorOfStructs(costs)).Union());
    return make_msg(mc);
  }

  std::unique_ptr<OSRM> osrm_;
};

router::router(std::string const& path)
    : impl_(std::make_unique<router::impl>(path)) {}

router::~router() = default;

msg_ptr router::one_to_many(OSRMOneToManyRequest const* req) const {
  return impl_->one_to_many(req);
}

msg_ptr router::many_to_many(OSRMManyToManyRequest const* req) const {
  return impl_->many_to_many(req);
}

msg_ptr router::route(OSRMRouteRequest const* req) const {
  return impl_->route(req);
}

msg_ptr router::via(OSRMViaRouteRequest const* req) const {
  return impl_->via(req);
}

msg_ptr router::smooth_via(OSRMSmoothViaRouteRequest const* req) const {
  return impl_->smooth_via(req);
}

}  // namespace motis::osrm

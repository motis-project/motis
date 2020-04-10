#include "motis/ppr/ppr.h"

#include <cmath>
#include <limits>
#include <map>

#include "utl/to_vec.h"

#include "ppr/common/verify.h"
#include "ppr/profiles/parse_search_profile.h"
#include "ppr/routing/route_steps.h"
#include "ppr/routing/search.h"
#include "ppr/routing/search_profile.h"
#include "ppr/serialization/reader.h"

#include "motis/core/common/logging.h"
#include "motis/ppr/error.h"
#include "motis/ppr/profiles.h"

using namespace motis::module;
using namespace motis::logging;
using namespace ppr;
using namespace ppr::routing;
using namespace ppr::serialization;
using namespace flatbuffers;

namespace motis::ppr {

location to_location(Position const* pos) {
  return make_location(pos->lng(), pos->lat());
}

Position to_position(location const& loc) { return {loc.lat(), loc.lon()}; }

Offset<Polyline> write_polyline(FlatBufferBuilder& fbb,
                                std::vector<location> const& path) {
  std::vector<double> polyline;
  polyline.reserve(path.size() * 2);
  for (auto const& l : path) {
    polyline.push_back(l.lat());
    polyline.push_back(l.lon());
  }
  return CreatePolyline(fbb, fbb.CreateVector(polyline));
}

Offset<RouteStep> write_route_step(FlatBufferBuilder& fbb,
                                   route_step const& rs) {
  return CreateRouteStep(
      fbb, static_cast<RouteStepType>(rs.step_type_),
      fbb.CreateString(rs.street_name_),
      static_cast<StreetType>(rs.street_type_),
      static_cast<CrossingType>(rs.crossing_), rs.distance_, rs.time_,
      rs.accessibility_, write_polyline(fbb, rs.path_), rs.elevation_up_,
      rs.elevation_down_, rs.incline_up_, static_cast<TriState>(rs.handrail_));
}

Offset<Edge> write_edge(FlatBufferBuilder& fbb, route::edge const& e) {
  return CreateEdge(fbb, e.distance_, e.duration_, e.accessibility_,
                    write_polyline(fbb, e.path_), fbb.CreateString(e.name_),
                    e.osm_way_id_, static_cast<EdgeType>(e.edge_type_),
                    static_cast<StreetType>(e.street_type_),
                    static_cast<CrossingType>(e.crossing_type_),
                    e.elevation_up_, e.elevation_down_, e.incline_up_,
                    static_cast<TriState>(e.handrail_));
}

Offset<Route> write_route(FlatBufferBuilder& fbb, route const& r,
                          bool include_steps, bool include_edges,
                          bool include_path) {
  auto const start = to_position(r.edges_.front().path_.front());
  auto const destination = to_position(r.edges_.back().path_.back());
  auto const steps =
      include_steps
          ? fbb.CreateVector(utl::to_vec(get_route_steps(r),
                                         [&](route_step const& rs) {
                                           return write_route_step(fbb, rs);
                                         }))
          : fbb.CreateVector(std::vector<Offset<RouteStep>>{});
  auto const edges =
      include_edges ? fbb.CreateVector(utl::to_vec(r.edges_,
                                                   [&](route::edge const& e) {
                                                     return write_edge(fbb, e);
                                                   }))
                    : fbb.CreateVector(std::vector<Offset<Edge>>{});
  auto const path = write_polyline(
      fbb, include_path ? get_route_path(r) : std::vector<location>{});
  auto const duration_min = static_cast<duration>(
      std::min(std::round(r.duration_ / 60),
               static_cast<double>(std::numeric_limits<duration>::max())));
  auto const accessibility_disc =
      static_cast<uint16_t>(std::ceil(r.accessibility_));
  return CreateRoute(
      fbb, r.distance_, duration_min, r.orig_duration_ / 60.0, r.disc_duration_,
      accessibility_disc, r.orig_accessibility_, r.disc_accessibility_, &start,
      &destination, steps, edges, path, r.elevation_up_, r.elevation_down_);
}

Offset<Routes> write_routes(FlatBufferBuilder& fbb,
                            std::vector<route> const& routes,
                            bool include_steps, bool include_edges,
                            bool include_path) {
  return CreateRoutes(
      fbb, fbb.CreateVector(utl::to_vec(routes, [&](struct route const& r) {
        return write_route(fbb, r, include_steps, include_edges, include_path);
      })));
}

struct ppr::impl {
  explicit impl(std::string const& rg_path,
                std::vector<std::string> const& profile_files,
                std::size_t edge_rtree_max_size,
                std::size_t area_rtree_max_size, rtree_options rtree_opt,
                bool verify_routing_graph) {
    {
      scoped_timer timer("loading ppr routing graph");
      read_routing_graph(rg_, rg_path);
    }
    {
      scoped_timer timer("preparing ppr r-trees");
      rg_.prepare_for_routing(edge_rtree_max_size, area_rtree_max_size,
                              rtree_opt);
    }
    if (verify_routing_graph) {
      scoped_timer timer("verifying ppr routing graph");
      if (!verify_graph(rg_)) {
        throw std::runtime_error("invalid ppr routing graph");
      }
    }
    read_profile_files(profile_files, profiles_);
  }

  msg_ptr route(msg_ptr const& msg) {
    switch (msg->get()->content_type()) {
      case MsgContent_FootRoutingRequest: return route_normal(msg);
      case MsgContent_FootRoutingSimpleRequest: return route_simple(msg);
      default:
        LOG(motis::logging::error) << "/ppr/route: unexpected message type: "
                                   << msg->get()->content_type();
        throw std::system_error(motis::module::error::unexpected_message_type);
    }
  }

  msg_ptr get_profiles() {
    message_creator fbb;
    auto profiles = utl::to_vec(profiles_, [&](auto const& e) {
      auto const& p = e.second;
      return CreateFootRoutingProfileInfo(fbb, fbb.CreateString(e.first),
                                          p.walking_speed_);
    });
    fbb.create_and_finish(MsgContent_FootRoutingProfilesResponse,
                          CreateFootRoutingProfilesResponse(
                              fbb, fbb.CreateVectorOfSortedTables(&profiles))
                              .Union());
    return make_msg(fbb);
  }

private:
  msg_ptr route_normal(msg_ptr const& msg) {
    auto const req = motis_content(FootRoutingRequest, msg);

    auto start = to_location(req->start());
    std::vector<location> destinations;
    for (auto const& dest : *req->destinations()) {
      destinations.emplace_back(to_location(dest));
    }

    auto const profile = get_search_profile(req->search_options());
    auto const dir = req->search_direction() == SearchDirection_Forward
                         ? search_direction::FWD
                         : search_direction::BWD;

    auto const result = find_routes(rg_, start, destinations, profile, dir);

    message_creator fbb;
    auto const include_steps = req->include_steps();
    auto const include_edges = req->include_edges();
    auto const include_path = req->include_path();
    fbb.create_and_finish(
        MsgContent_FootRoutingResponse,
        CreateFootRoutingResponse(
            fbb, fbb.CreateVector(utl::to_vec(
                     result.routes_,
                     [&](std::vector<struct route> const& rs) {
                       return write_routes(fbb, rs, include_steps,
                                           include_edges, include_path);
                     })))
            .Union());
    return make_msg(fbb);
  }

  msg_ptr route_simple(msg_ptr const& msg) {
    auto const req = motis_content(FootRoutingSimpleRequest, msg);

    auto start = to_location(req->start());
    std::vector<location> destinations{to_location(req->destination())};

    search_profile profile{};
    if (req->max_duration() != 0) {
      profile.duration_limit_ = req->max_duration();
    }
    auto const result =
        find_routes(rg_, start, destinations, profile, search_direction::FWD);

    message_creator fbb;
    auto const include_steps = req->include_steps();
    auto const include_edges = false;
    auto const include_path = req->include_path();
    assert(result.routes_.size() == 1);
    fbb.create_and_finish(
        MsgContent_FootRoutingSimpleResponse,
        CreateFootRoutingSimpleResponse(
            fbb, fbb.CreateVector(utl::to_vec(result.routes_[0],
                                              [&](struct route const& r) {
                                                return write_route(
                                                    fbb, r, include_steps,
                                                    include_edges,
                                                    include_path);
                                              })))
            .Union());
    return make_msg(fbb);
  }

  search_profile get_search_profile(SearchOptions const* opt) {
    auto profile = search_profile{};
    auto const name = opt->profile()->str();
    auto const it = profiles_.find(name);
    if (it != end(profiles_)) {
      profile = it->second;
    } else if (!name.empty() && name != "default") {
      throw std::system_error(error::profile_not_available);
    }
    profile.duration_limit_ = opt->duration_limit();
    return profile;
  }

  routing_graph rg_;
  std::map<std::string, ::ppr::routing::search_profile> profiles_;
};

ppr::ppr() : module("Foot Routing", "ppr") {
  param(graph_file_, "graph", "ppr routing graph path");
  param(profile_files_, "profile", "Search profile");
  param(edge_rtree_max_size_, "edge-rtree-max-size",
        "Maximum size for edge r-tree file");
  param(area_rtree_max_size_, "area-rtree-max-size",
        "Maximum size for area r-tree file");
  param(lock_rtrees_, "lock-rtrees", "Prefetch and lock r-trees in memory");
  param(prefetch_rtrees_, "prefetch-rtrees", "Prefetch r-trees");
  param(verify_graph_, "verify-graph", "Verify routing graph");
}

ppr::~ppr() = default;

void ppr::init(motis::module::registry& reg) {
  rtree_options rtree_opt = lock_rtrees_
                                ? rtree_options::LOCK
                                : (prefetch_rtrees_ ? rtree_options::PREFETCH
                                                    : rtree_options::DEFAULT);

  try {
    impl_ = std::make_unique<impl>(graph_file_, profile_files_,
                                   edge_rtree_max_size_, area_rtree_max_size_,
                                   rtree_opt, verify_graph_);
    reg.register_op("/ppr/route", std::bind(&impl::route, impl_.get(),
                                            std::placeholders::_1));
    reg.register_op("/ppr/profiles",
                    std::bind(&impl::get_profiles, impl_.get()));
  } catch (std::exception const& e) {
    LOG(logging::error) << "ppr module not initialized (" << e.what() << ")";
  }
}

}  // namespace motis::ppr

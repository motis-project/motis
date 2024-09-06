#include "motis/ppr/ppr.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>

#include "cista/hash.h"
#include "cista/mmap.h"

#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"

#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "ppr/common/verify.h"
#include "ppr/preprocessing/preprocessing.h"
#include "ppr/profiles/parse_search_profile.h"
#include "ppr/routing/route_steps.h"
#include "ppr/routing/search.h"
#include "ppr/routing/search_profile.h"
#include "ppr/serialization/reader.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"
#include "motis/core/otel/tracer.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/ppr/data.h"
#include "motis/ppr/error.h"
#include "motis/ppr/profiles.h"

using namespace motis::module;
using namespace motis::logging;
using namespace ppr;
using namespace ppr::routing;
using namespace ppr::serialization;
using namespace flatbuffers;

namespace fs = std::filesystem;
namespace pp = ::ppr::preprocessing;

namespace motis::ppr {

struct import_state {
  CISTA_COMPARABLE()
  named<std::string, MOTIS_NAME("osm_path")> osm_path_;
  named<cista::hash_t, MOTIS_NAME("osm_hash")> osm_hash_;
  named<size_t, MOTIS_NAME("osm_size")> osm_size_;
  named<std::string, MOTIS_NAME("dem_path")> dem_path_;
  named<cista::hash_t, MOTIS_NAME("dem_hash")> dem_hash_;
};

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
      rs.elevation_down_, rs.incline_up_, static_cast<TriState>(rs.handrail_),
      static_cast<DoorType>(rs.door_type_),
      static_cast<AutomaticDoorType>(rs.automatic_door_type_),
      static_cast<TriState>(rs.traffic_signals_sound_),
      static_cast<TriState>(rs.traffic_signals_vibration_), rs.beeline_);
}

Offset<Edge> write_edge(FlatBufferBuilder& fbb, routing_graph_data const& rg,
                        route::edge const& e) {
  auto const& lvl = e.levels_;

  return CreateEdge(
      fbb, e.distance_, e.duration_, e.accessibility_,
      write_polyline(fbb, e.path_), fbb.CreateString(e.name_), e.osm_way_id_,
      e.from_node_osm_id_, e.to_node_osm_id_,
      static_cast<EdgeType>(e.edge_type_),
      static_cast<StreetType>(e.street_type_),
      static_cast<CrossingType>(e.crossing_type_), e.elevation_up_,
      e.elevation_down_, e.incline_up_, static_cast<TriState>(e.handrail_),
      static_cast<DoorType>(e.door_type_),
      static_cast<AutomaticDoorType>(e.automatic_door_type_),
      static_cast<TriState>(e.traffic_signals_sound_),
      static_cast<TriState>(e.traffic_signals_vibration_), e.area_,
      e.free_crossing_, e.is_additional_edge_,
      fbb.CreateVector(
          lvl.has_level()
              ? (lvl.has_single_level()
                     ? std::vector{to_human_level(lvl.single_level())}
                     : utl::to_vec(rg.levels_.at(lvl.multi_level_index()),
                                   [](auto const level) {
                                     return to_human_level(level);
                                   }))
              : std::vector<double>{}));
}

Offset<Route> write_route(FlatBufferBuilder& fbb, routing_graph_data const& rg,
                          route const& r, bool include_steps,
                          bool include_edges, bool include_path) {
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
      include_edges
          ? fbb.CreateVector(utl::to_vec(
                r.edges_,
                [&](route::edge const& e) { return write_edge(fbb, rg, e); }))
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
                            routing_graph_data const& rg,
                            std::vector<route> const& routes,
                            bool include_steps, bool include_edges,
                            bool include_path) {
  return CreateRoutes(
      fbb, fbb.CreateVector(utl::to_vec(routes, [&](struct route const& r) {
        return write_route(fbb, rg, r, include_steps, include_edges,
                           include_path);
      })));
}

struct ppr::impl {
  explicit impl(std::string const& rg_path,
                std::map<std::string, profile_info>& profiles,
                std::size_t edge_rtree_max_size,
                std::size_t area_rtree_max_size, rtree_options rtree_opt,
                bool verify_routing_graph, bool check_integrity) {
    data_.profiles_ = profiles;
    {
      scoped_timer const timer("loading ppr routing graph");
      read_routing_graph(data_.rg_, rg_path, check_integrity);
    }
    {
      scoped_timer const timer("preparing ppr r-trees");
      data_.rg_.prepare_for_routing(edge_rtree_max_size, area_rtree_max_size,
                                    rtree_opt);
    }
    if (verify_routing_graph) {
      scoped_timer const timer("verifying ppr routing graph");
      if (!verify_graph(data_.rg_)) {
        throw std::runtime_error("invalid ppr routing graph");
      }
    }
  }

  msg_ptr route(msg_ptr const& msg) const {
    auto span = motis_tracer->StartSpan("ppr::route");
    auto scope = opentelemetry::trace::Scope{span};

    switch (msg->get()->content_type()) {
      case MsgContent_FootRoutingRequest: return route_normal(msg);
      case MsgContent_FootRoutingSimpleRequest: return route_simple(msg);
      default:
        LOG(motis::logging::error) << "/ppr/route: unexpected message type: "
                                   << msg->get()->content_type();
        throw std::system_error(motis::module::error::unexpected_message_type);
    }
  }

  msg_ptr get_profiles() const {
    message_creator fbb;
    auto profiles = utl::to_vec(data_.profiles_, [&](auto const& e) {
      auto const& p = e.second.profile_;
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
  msg_ptr route_normal(msg_ptr const& msg) const {
    auto span = motis_tracer->StartSpan("ppr::route_normal");
    auto scope = opentelemetry::trace::Scope{span};

    auto const req = motis_content(FootRoutingRequest, msg);

    auto start = to_location(req->start());
    std::vector<location> destinations;
    for (auto const& dest : *req->destinations()) {
      destinations.emplace_back(to_location(dest));
    }

    auto const dir = req->search_direction() == SearchDir_Forward
                         ? search_direction::FWD
                         : search_direction::BWD;

    span->SetAttribute("motis.ppr.request.profile",
                       req->search_options()->profile()->view());
    span->SetAttribute("motis.ppr.request.duration_limit",
                       req->search_options()->duration_limit());
    span->SetAttribute("motis.ppr.request.direction",
                       dir == search_direction::FWD ? "forward" : "backward");

    auto const profile = get_search_profile(req->search_options());

    auto const result =
        find_routes(data_.rg_, start, destinations, profile, dir);

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
                       return write_routes(fbb, *data_.rg_.data_, rs,
                                           include_steps, include_edges,
                                           include_path);
                     })))
            .Union());
    return make_msg(fbb);
  }

  msg_ptr route_simple(msg_ptr const& msg) const {
    auto span = motis_tracer->StartSpan("ppr::route_simple");
    auto scope = opentelemetry::trace::Scope{span};

    auto const req = motis_content(FootRoutingSimpleRequest, msg);

    auto start = to_location(req->start());
    std::vector<location> const destinations{to_location(req->destination())};

    search_profile profile{};
    if (req->max_duration() != 0) {
      profile.duration_limit_ = req->max_duration();
    }
    auto const result = find_routes(data_.rg_, start, destinations, profile,
                                    search_direction::FWD);

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
                                                    fbb, *data_.rg_.data_, r,
                                                    include_steps,
                                                    include_edges,
                                                    include_path);
                                              })))
            .Union());
    return make_msg(fbb);
  }

  search_profile get_search_profile(SearchOptions const* opt) const {
    auto profile = search_profile{};
    auto const name = opt->profile()->str();
    auto const it = data_.profiles_.find(name);
    if (it != end(data_.profiles_)) {
      profile = it->second.profile_;
    } else if (!name.empty() && name != "default") {
      throw std::system_error(error::profile_not_available);
    }
    profile.duration_limit_ = opt->duration_limit();
    return profile;
  }

public:
  ppr_data data_;
};

ppr::ppr() : module("Foot Routing", "ppr") {
  param(use_dem_, "import.use_dem", "Use elevation data");
  param(profile_files_, "profile", "Search profile");
  param(edge_rtree_max_size_, "edge_rtree_max_size",
        "Maximum size for edge r-tree file");
  param(area_rtree_max_size_, "area_rtree_max_size",
        "Maximum size for area r-tree file");
  param(lock_rtrees_, "lock_rtrees", "Prefetch and lock r-trees in memory");
  param(prefetch_rtrees_, "prefetch_rtrees", "Prefetch r-trees");
  param(verify_graph_, "verify_graph", "Verify routing graph");
  param(check_integrity_, "check_integrity",
        "Check routing graph file integrity");
}

ppr::~ppr() = default;
void ppr::import(import_dispatcher& reg) {
  auto const collector = std::make_shared<event_collector>(
      get_data_directory().generic_string(), "ppr", reg,
      [this](event_collector::dependencies_map_t const& dependencies,
             event_collector::publish_fn_t const& publish) {
        using import::OSMEvent;
        using import::DEMEvent;

        auto span = motis_tracer->StartSpan("ppr::import");
        auto scope = opentelemetry::trace::Scope{span};

        auto const dir = get_data_directory() / "ppr";
        auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const osm_path = osm->path()->str();

        span->SetAttribute("motis.osm.file", osm_path);
        span->SetAttribute("motis.osm.size", osm->size());

        auto dem_path = std::string{};
        auto dem_hash = cista::hash_t{};
        if (use_dem_) {
          auto const dem = motis_content(DEMEvent, dependencies.at("DEM"));
          dem_path = data_path(dem->path()->str());
          dem_hash = dem->hash();
          span->SetAttribute("motis.dem.path", dem->path()->str());
        }

        auto progress_tracker = utl::get_active_progress_tracker();

        auto const state =
            import_state{data_path(osm_path), osm->hash(), osm->size(),
                         data_path(dem_path), dem_hash};
        auto const state_changed =
            read_ini<import_state>(dir / "import.ini") != state;

        span->SetAttribute("motis.import.state",
                           state_changed ? "changed" : "unchanged");

        auto const rtree_opt =
            lock_rtrees_ ? rtree_options::LOCK
                         : (prefetch_rtrees_ ? rtree_options::PREFETCH
                                             : rtree_options::DEFAULT);

        auto profiles = std::map<std::string, profile_info>{};
        read_profile_files(profile_files_, profiles);
        auto profiles_hash = cista::BASE_HASH;
        for (auto const& p : profiles) {
          profiles_hash =
              cista::hash_combine(profiles_hash, p.second.file_hash_);
        }

        auto graph_hash = cista::hash_t{};
        auto graph_size = std::size_t{};

        auto const load_graph = [&]() {
          span->AddEvent("load graph", {{"file", graph_file()}});
          impl_ = std::make_unique<impl>(
              graph_file(), profiles, edge_rtree_max_size_,
              area_rtree_max_size_, rtree_opt, verify_graph_, check_integrity_);

          auto& mem_holder = impl_->data_.rg_.data_.mem_;
          utl::verify(
              std::holds_alternative<cista::buf<cista::mmap>>(mem_holder),
              "expected memory mapped ppr graph");
          auto& mmap_buf = std::get<cista::buf<cista::mmap>>(mem_holder);
          graph_hash = cista::hash(std::string_view{
              reinterpret_cast<char const*>(mmap_buf.base()),
              std::min(static_cast<std::size_t>(50 * 1024 * 1024),
                       mmap_buf.size())});
          graph_size = mmap_buf.size();
        };

        if (!state_changed) {
          // try to load existing routing graph - this will fail if it is
          // incompatible with the current ppr version
          try {
            load_graph();
          } catch (std::exception const& e) {
            span->AddEvent("exception", {{"exception.message", e.what()},
                                         {"during", "load_graph"}});
            impl_.reset();
            LOG(logging::error) << "loading existing ppr routing graph failed ("
                                << e.what() << "), will be re-created";
          }
        }

        if (!impl_) {
          span->AddEvent("import");
          fs::create_directories(dir);

          pp::options opt;
          opt.osm_file_ = osm_path;
          opt.graph_file_ = graph_file();
          if (use_dem_) {
            opt.dem_files_ = {dem_path};
          }
          opt.edge_rtree_max_size_ = edge_rtree_max_size_;
          opt.area_rtree_max_size_ = area_rtree_max_size_;

          pp::logging log;
          log.out_ = &std::clog;
          log.total_progress_updates_only_ = true;

          auto step_span = std::shared_ptr<opentelemetry::trace::Span>{};
          auto step_scope = std::unique_ptr<opentelemetry::trace::Scope>{};

          log.step_started_ = [&](pp::logging const& log,
                                  pp::step_info const& step) {
            step_span = motis_tracer->StartSpan(
                step.name(), {{"step.current", (log.current_step() + 1)},
                              {"step.total", log.step_count()}});
            step_scope =
                std::make_unique<opentelemetry::trace::Scope>(step_span);
            std::clog << "Step " << (log.current_step() + 1) << "/"
                      << log.step_count() << ": " << step.name() << ": Starting"
                      << '\n';
            progress_tracker->status(step.name());
          };

          log.step_progress_ = [&](pp::logging const& log,
                                   pp::step_info const&) {
            progress_tracker->update(
                static_cast<int>(log.total_progress() * 100));
          };

          log.step_finished_ = [&](pp::logging const& log,
                                   pp::step_info const& step) {
            step_scope.reset(nullptr);
            step_span.reset();
            std::clog << "Step " << (log.current_step() + 1) << "/"
                      << log.step_count() << ": " << step.name()
                      << ": Finished in " << static_cast<int>(step.duration_)
                      << "ms" << '\n';
          };

          auto const result = pp::create_routing_data(opt, log);
          utl::verify(result.successful(), "{}", result.error_msg_);
          write_ini(dir / "import.ini", state);

          try {
            load_graph();
          } catch (std::exception const& e) {
            impl_.reset();
            LOG(logging::error)
                << "loading ppr routing graph failed: " << e.what();
          }
        }

        import_successful_ = impl_ != nullptr;
        progress_tracker->update(100);

        if (impl_ == nullptr) {
          return;
        }

        add_shared_data(to_res_id(global_res_id::PPR_DATA), &impl_->data_);

        message_creator fbb;
        fbb.create_and_finish(
            MsgContent_PPREvent,
            motis::import::CreatePPREvent(
                fbb, fbb.CreateString(graph_file()), graph_hash, graph_size,
                fbb.CreateVector(utl::to_vec(
                    profiles,
                    [&](auto const& p) {
                      return motis::import::CreatePPRProfile(
                          fbb, fbb.CreateString(p.first),
                          fbb.CreateString(p.second.file_path_.string()),
                          p.second.file_hash_, p.second.file_size_);
                    })),
                profiles_hash)
                .Union(),
            "/import", DestinationType_Topic);
        publish(make_msg(fbb));
      });
  collector->require("OSM", [](msg_ptr const& msg) {
    return msg->get()->content_type() == MsgContent_OSMEvent;
  });
  if (use_dem_) {
    collector->require("DEM", [](msg_ptr const& msg) {
      return msg->get()->content_type() == MsgContent_DEMEvent;
    });
  }
}

void ppr::init(motis::module::registry& reg) {
  if (impl_) {
    reg.register_op("/ppr/route",
                    [this](msg_ptr const& msg) { return impl_->route(msg); },
                    {});
    reg.register_op("/ppr/profiles",
                    [this](msg_ptr const&) { return impl_->get_profiles(); },
                    {});
  } else {
    LOG(logging::error) << "ppr module not initialized";
  }
}

std::string ppr::graph_file() const {
  return (get_data_directory() / "ppr" / "routing_graph.ppr").generic_string();
}

}  // namespace motis::ppr

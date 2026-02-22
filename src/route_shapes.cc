#include "motis/route_shapes.h"

#include <chrono>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <optional>
#include <ranges>
#include <set>
#include <vector>

#include "boost/stacktrace.hpp"

#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"
#include "utl/sorted_diff.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "geo/box.h"
#include "geo/latlng.h"

#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/rt/frun.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"

#include "osr/routing/map_matching.h"
#include "osr/routing/parameters.h"
#include "osr/routing/profile.h"
#include "osr/routing/route.h"
#include "osr/types.h"

#include "motis/match_platforms.h"
#include "motis/types.h"

namespace n = nigiri;

namespace motis {

std::optional<osr::search_profile> get_profile(n::clasz const clasz) {
  switch (clasz) {
    case n::clasz::kBus:
    case n::clasz::kCoach:
    case n::clasz::kRideSharing:
    case n::clasz::kODM: return osr::search_profile::kBus;
    case n::clasz::kTram:
    case n::clasz::kHighSpeed:
    case n::clasz::kLongDistance:
    case n::clasz::kNight:
    case n::clasz::kRegional:
    case n::clasz::kSuburban:
    case n::clasz::kSubway:
    case n::clasz::kFunicular: return osr::search_profile::kRailway;
    case n::clasz::kShip: return osr::search_profile::kFerry;
    default: return std::nullopt;
  }
}

struct route_shape_result {
  n::vector<geo::latlng> shape_;
  n::vector<n::shape_offset_t> offsets_;
  geo::box route_bbox_{};
  n::vector<geo::box> segment_bboxes_;

  unsigned segments_routed_{};
  unsigned segments_beelined_{};
  unsigned dijkstra_early_terminations_{};
  unsigned dijkstra_full_runs_{};
};

route_shape_result route_shape(
    osr::ways const& w,
    osr::lookup const& lookup,
    n::timetable& tt,
    std::vector<osr::location> const& match_points,
    osr::search_profile const profile,
    osr::profile_parameters const& profile_params,
    n::clasz const clasz,
    n::route_idx_t const route_idx,
    std::optional<config::timetable::shapes_debug> const& debug,
    bool const debug_enabled) {
  auto r = route_shape_result{};
  r.offsets_.reserve(
      static_cast<decltype(r.offsets_)::size_type>(match_points.size()));
  r.segment_bboxes_.reserve(static_cast<decltype(r.segment_bboxes_)::size_type>(
      match_points.size() - 1U));

  auto debug_path_fn = std::function<std::optional<std::filesystem::path>(
      osr::matched_route const&)>{nullptr};

  if (debug_enabled) {
    debug_path_fn = [&debug, route_idx, clasz,
                     &tt](osr::matched_route const& res)
        -> std::optional<std::filesystem::path> {
      auto include =
          debug->all_ || (debug->all_with_beelines_ && res.n_beelined_ > 0U);
      auto tags = std::set<std::string>{};

      if (debug->route_indices_ && !debug->route_indices_->empty()) {
        auto const& debug_route_indices = *debug->route_indices_;
        if (std::ranges::contains(debug_route_indices, to_idx(route_idx))) {
          include = true;
        }
      }

      if (debug->route_ids_ && !debug->route_ids_->empty()) {
        auto const& debug_route_ids = *debug->route_ids_;
        for (auto const transport_idx : tt.route_transport_ranges_[route_idx]) {
          auto const frun = n::rt::frun{
              tt, nullptr,
              n::rt::run{.t_ = n::transport{transport_idx, n::day_idx_t{0}},
                         .stop_range_ =
                             n::interval{
                                 n::stop_idx_t{0U},
                                 static_cast<n::stop_idx_t>(
                                     tt.route_location_seq_[route_idx].size())},
                         .rt_ = n::rt_transport_idx_t::invalid()}};

          auto const rsn = frun[0].get_route_id(n::event_type::kDep);
          if (std::ranges::contains(debug_route_ids, rsn)) {
            tags.emplace(fmt::format("route_{}", rsn));
            include = true;
            break;
          }
        }
      }

      if (debug->trips_ && !debug->trips_->empty()) {
        auto const& debug_trip_ids = *debug->trips_;
        for (auto const transport_idx : tt.route_transport_ranges_[route_idx]) {
          auto const frun = n::rt::frun{
              tt, nullptr,
              n::rt::run{.t_ = n::transport{transport_idx},
                         .stop_range_ =
                             n::interval{
                                 n::stop_idx_t{0U},
                                 static_cast<n::stop_idx_t>(
                                     tt.route_location_seq_[route_idx].size())},
                         .rt_ = n::rt_transport_idx_t::invalid()}};

          frun.for_each_trip([&](n::trip_idx_t const trip_idx,
                                 n::interval<n::stop_idx_t> const) {
            for (auto const trip_id_idx : tt.trip_ids_[trip_idx]) {
              auto const trip_id = tt.trip_id_strings_.at(trip_id_idx).view();
              if (std::ranges::contains(debug_trip_ids, trip_id)) {
                tags.emplace(fmt::format("trip_{}", trip_id));
                include = true;
                return;
              }
            }
          });
        }
      }

      if (debug->slow_ != 0U && res.total_duration_.count() > debug->slow_) {
        include = true;
        tags.emplace("slow");
      }

      if (include) {
        auto fn = fmt::format("r_{}_{}", to_idx(route_idx), to_str(clasz));
        for (auto const& tag : tags) {
          fn += fmt::format("_{}", tag);
        }
        return debug->path_ / fn;
      } else {
        return {};
      }
    };
  }

  auto const matched_route =
      osr::map_match(w, lookup, profile, profile_params, match_points, nullptr,
                     nullptr, debug_path_fn);

  r.segments_routed_ = matched_route.n_routed_;
  r.segments_beelined_ = matched_route.n_beelined_;
  r.dijkstra_early_terminations_ = matched_route.n_dijkstra_early_terminations_;
  r.dijkstra_full_runs_ = matched_route.n_dijkstra_full_runs_;

  utl::verify(matched_route.segment_offsets_.size() == match_points.size(),
              "[route_shapes] segment offsets ({}) != match points ({})",
              matched_route.segment_offsets_.size(), match_points.size());

  r.segment_bboxes_.resize(static_cast<decltype(r.segment_bboxes_)::size_type>(
      match_points.size() - 1U));
  r.shape_.clear();
  r.shape_.reserve(static_cast<decltype(r.shape_)::size_type>(
      matched_route.path_.segments_.size() * 8U));
  r.offsets_.clear();
  r.offsets_.reserve(
      static_cast<decltype(r.offsets_)::size_type>(match_points.size()));
  r.offsets_.emplace_back(0U);

  for (auto seg_idx = 0U; seg_idx < match_points.size() - 1U; ++seg_idx) {
    auto& seg_bbox = r.segment_bboxes_[seg_idx];

    if (!r.shape_.empty()) {
      seg_bbox.extend(r.shape_.back());
    }

    auto const start = matched_route.segment_offsets_[seg_idx];
    auto const end = (seg_idx + 1U < match_points.size() - 1U)
                         ? matched_route.segment_offsets_[seg_idx + 1U]
                         : matched_route.path_.segments_.size();

    for (auto ps_idx = start; ps_idx < end; ++ps_idx) {
      auto const& ps = matched_route.path_.segments_[ps_idx];
      for (auto const& pt : ps.polyline_) {
        r.route_bbox_.extend(pt);
        seg_bbox.extend(pt);
      }
      if (!ps.polyline_.empty()) {
        auto first = ps.polyline_.begin();
        if (!r.shape_.empty() && r.shape_.back() == *first) {
          ++first;
        }
        r.shape_.insert(r.shape_.end(), first, ps.polyline_.end());
      }
    }

    r.offsets_.emplace_back(static_cast<std::uint32_t>(r.shape_.size() - 1U));
  }

  utl::verify(r.offsets_.size() == match_points.size(),
              "[route_shapes] mismatch: offsets.size()={}, stops.size()={}",
              r.offsets_.size(), match_points.size());
  return r;
}

void route_shapes(osr::ways const& w,
                  osr::lookup const& lookup,
                  n::timetable& tt,
                  n::shapes_storage& shapes,
                  config::timetable::route_shapes const& conf,
                  std::array<bool, n::kNumClasses> const& clasz_enabled,
                  shape_cache_t* cache) {
  fmt::println(std::clog, "computing shapes");

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Computing shapes")
      .out_bounds(0.F, 100.F)
      .in_high(tt.n_routes());

  auto routes_matched = 0ULL;
  auto segments_routed = 0ULL;
  auto segments_beelined = 0ULL;
  auto dijkstra_early_terminations = 0ULL;
  auto dijkstra_full_runs = 0ULL;
  auto routes_with_existing_shapes = 0ULL;
  auto cache_hits = 0ULL;

  auto const& debug = conf.debug_;
  auto const debug_enabled =
      debug && !debug->path_.empty() &&
      (debug->all_ || debug->all_with_beelines_ ||
       (debug->trips_ && !debug->trips_->empty()) ||
       (debug->route_ids_ && !debug->route_ids_->empty()) ||
       (debug->route_indices_ && !debug->route_indices_->empty()) ||
       debug->slow_ != 0U);

  if (debug_enabled) {
    std::filesystem::create_directories(debug->path_);
  }

  std::clog << "\n** route_shapes [start] **\n"
            << "  routes=" << tt.n_routes() << "\n  trips=" << tt.n_trips()
            << "\n  shapes.trip_offset_indices_="
            << shapes.trip_offset_indices_.size()
            << "\n  shapes.route_bboxes_=" << shapes.route_bboxes_.size()
            << "\n  shapes.route_segment_bboxes_="
            << shapes.route_segment_bboxes_.size()
            << "\n  shapes.data=" << shapes.data_.size()
            << "\n  shapes.routed_data=" << shapes.routed_data_.size()
            << "\n  shapes.offsets=" << shapes.offsets_.size()
            << "\n  shapes.trip_offset_indices_="
            << shapes.trip_offset_indices_.size() << "\n\n";

  shapes.trip_offset_indices_.resize(tt.n_trips());
  shapes.route_bboxes_.resize(tt.n_routes());
  shapes.route_segment_bboxes_.resize(tt.n_routes());

  auto shapes_mutex = std::mutex{};

  auto const store_shape =
      [&](n::route_idx_t const r, n::scoped_shape_idx_t const shape_idx,
          n::vector<n::shape_offset_t> const& offsets,
          geo::box const& route_bbox, n::vector<geo::box> const& segment_bboxes,
          std::vector<osr::location> const& match_points,
          n::interval<n::transport_idx_t> const& transports) {
        shapes.route_bboxes_[r] = route_bbox;
        auto rsb = shapes.route_segment_bboxes_[r];
        if (!rsb.empty()) {
          if (rsb.size() != segment_bboxes.size()) {
            fmt::println(std::clog,
                         "[route_shapes] route {}: segment bbox size "
                         "mismatch: storage={}, computed={}",
                         r, rsb.size(), segment_bboxes.size());
          } else {
            for (auto i = 0U; i < segment_bboxes.size(); ++i) {
              rsb[i] = segment_bboxes[i];
            }
          }
        }

        auto range_to_offsets =
            hash_map<std::pair<n::stop_idx_t, n::stop_idx_t>,
                     n::shape_offset_idx_t>{};

        for (auto const transport_idx : transports) {
          auto const frun = n::rt::frun{
              tt, nullptr,
              n::rt::run{.t_ = n::transport{transport_idx, n::day_idx_t{0}},
                         .stop_range_ = n::interval{n::stop_idx_t{0U},
                                                    static_cast<n::stop_idx_t>(
                                                        match_points.size())},
                         .rt_ = n::rt_transport_idx_t::invalid()}};
          frun.for_each_trip([&](n::trip_idx_t const trip_idx,
                                 n::interval<n::stop_idx_t> const range) {
            auto const key = std::pair{range.from_, range.to_};
            auto it = range_to_offsets.find(key);
            if (it == end(range_to_offsets)) {
              auto trip_offsets = std::vector<n::shape_offset_t>{};
              trip_offsets.reserve(static_cast<std::size_t>(range.size()));
              for (auto const i : range) {
                trip_offsets.push_back(offsets.at(i));
              }
              auto const offsets_idx = shapes.add_offsets(trip_offsets);
              it = range_to_offsets.emplace(key, offsets_idx).first;
            }

            shapes.trip_offset_indices_[trip_idx] = {shape_idx, it->second};
          });
        }
      };

  auto const process_route = [&](std::size_t const route_idx) {
    auto const r =
        n::route_idx_t{static_cast<n::route_idx_t::value_t>(route_idx)};

    auto const clasz = tt.route_clasz_[r];
    auto profile = get_profile(clasz);
    if (!profile || !clasz_enabled[static_cast<std::size_t>(clasz)]) {
      progress_tracker->increment();
      return;
    }
    auto const profile_params = osr::get_parameters(*profile);

    auto const stops = tt.route_location_seq_[r];
    if (stops.size() < 2U ||
        (conf.max_stops_ != 0U && stops.size() > conf.max_stops_)) {
      auto l = std::scoped_lock{shapes_mutex};
      std::clog << "skipping route " << r << ", " << stops.size() << " stops\n";
      progress_tracker->increment();
      return;
    }

    auto const transports = tt.route_transport_ranges_[r];

    if (!conf.replace_shapes_) {
      auto existing_shapes = true;
      for (auto const transport_idx : transports) {
        auto const frun = n::rt::frun{
            tt, nullptr,
            n::rt::run{.t_ = n::transport{transport_idx, n::day_idx_t{0}},
                       .stop_range_ = n::interval{n::stop_idx_t{0U},
                                                  static_cast<n::stop_idx_t>(
                                                      stops.size())},
                       .rt_ = n::rt_transport_idx_t::invalid()}};
        frun.for_each_trip(
            [&](n::trip_idx_t const trip_idx, n::interval<n::stop_idx_t>) {
              auto const shape_idx = shapes.get_shape_idx(trip_idx);
              if (shape_idx == n::scoped_shape_idx_t::invalid()) {
                existing_shapes = false;
              }
            });
        if (existing_shapes) {
          ++routes_with_existing_shapes;
          progress_tracker->increment();
          return;
        }
      }
    }

    auto const match_points = utl::to_vec(stops, [&](auto const stop_idx) {
      auto const loc_idx = n::stop{stop_idx}.location_idx();
      auto const pos = tt.locations_.coordinates_[loc_idx];
      return osr::location{pos, osr::level_t{osr::kNoLevel}};
    });

    try {
      auto cache_key =
          cache != nullptr
              ? std::optional{shape_cache_key{
                    *profile,
                    cista::raw::to_vec(match_points,
                                       [](auto const& mp) { return mp.pos_; })}}
              : std::nullopt;

      if (cache != nullptr) {
        auto const l = std::scoped_lock{shapes_mutex};
        if (auto it = cache->find(*cache_key); it != end(*cache)) {
          auto const& ce = it->second;
          ++cache_hits;
          auto const local_shape_idx = n::get_local_shape_idx(ce.shape_idx_);
          utl::verify(ce.shape_idx_ != n::scoped_shape_idx_t::invalid() &&
                          n::get_shape_source(ce.shape_idx_) ==
                              n::shape_source::kRouted,
                      "[route_shapes] invalid cached shape index: {}",
                      ce.shape_idx_);
          utl::verify(
              local_shape_idx != n::shape_idx_t::invalid() &&
                  static_cast<std::size_t>(to_idx(local_shape_idx)) <
                      shapes.routed_data_.size(),
              "[route_shapes] cache routed shape idx out of bounds: {} >= {}",
              local_shape_idx, shapes.routed_data_.size());
          store_shape(r, ce.shape_idx_, ce.offsets_, ce.route_bbox_,
                      ce.segment_bboxes_, match_points, transports);
          progress_tracker->increment();
          return;
        }
      }

      auto rsr = route_shape(w, lookup, tt, match_points, *profile,
                             profile_params, clasz, r, debug, debug_enabled);
      ++routes_matched;
      segments_routed += rsr.segments_routed_;
      segments_beelined += rsr.segments_beelined_;
      dijkstra_early_terminations += rsr.dijkstra_early_terminations_;
      dijkstra_full_runs += rsr.dijkstra_full_runs_;

      auto const l = std::scoped_lock{shapes_mutex};

      auto const local_shape_idx =
          static_cast<n::shape_idx_t>(shapes.routed_data_.size());
      auto const shape_idx =
          n::to_scoped_shape_idx(local_shape_idx, n::shape_source::kRouted);
      shapes.routed_data_.emplace_back(rsr.shape_);

      store_shape(r, shape_idx, rsr.offsets_, rsr.route_bbox_,
                  rsr.segment_bboxes_, match_points, transports);
      if (cache != nullptr) {
        cache->emplace(*cache_key,
                       shape_cache_entry{
                           .shape_idx_ = shape_idx,
                           .offsets_ = std::move(rsr.offsets_),
                           .route_bbox_ = rsr.route_bbox_,
                           .segment_bboxes_ = std::move(rsr.segment_bboxes_)});
      }
    } catch (std::exception const& e) {
      fmt::println(std::clog,
                   "[route_shapes] route {}: map matching failed: {}", r,
                   e.what());

      if (auto const trace =
              boost::stacktrace::stacktrace::from_current_exception();
          trace) {
        std::clog << trace << std::endl;
      }
    }
    progress_tracker->increment();
  };

  utl::parallel_for_run(
      tt.n_routes(), process_route, utl::noop_progress_update{},
      utl::parallel_error_strategy::QUIT_EXEC, conf.n_threads_);

  std::clog << "\n** route_shapes [end] **\n"
            << "  routes=" << tt.n_routes() << "\n  trips=" << tt.n_trips()
            << "\n  shapes.trip_offset_indices_="
            << shapes.trip_offset_indices_.size()
            << "\n  shapes.route_bboxes_=" << shapes.route_bboxes_.size()
            << "\n  shapes.route_segment_bboxes_="
            << shapes.route_segment_bboxes_.size()
            << "\n  shapes.data=" << shapes.data_.size()
            << "\n  shapes.routed_data=" << shapes.routed_data_.size()
            << "\n  shapes.offsets=" << shapes.offsets_.size()
            << "\n  shapes.trip_offset_indices_="
            << shapes.trip_offset_indices_.size() << "\n\n";

  fmt::println(std::clog,
               "{} routes matched, {} segments routed, {} segments beelined, "
               "{} dijkstra early terminations, {} dijkstra full runs\n{} "
               "routes with existing shapes skipped\n{} cache hits",
               routes_matched, segments_routed, segments_beelined,
               dijkstra_early_terminations, dijkstra_full_runs,
               routes_with_existing_shapes, cache_hits);
}

}  // namespace motis

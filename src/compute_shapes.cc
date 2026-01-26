#include "motis/compute_shapes.h"

#include <algorithm>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <vector>

#include "boost/stacktrace.hpp"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/erase_if.h"
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

#include "motis/constants.h"
#include "motis/match_platforms.h"
#include "motis/osr/max_distance.h"
#include "motis/types.h"

namespace n = nigiri;

namespace motis {

std::optional<osr::search_profile> get_profile(n::clasz const clasz) {
  switch (clasz) {
    case n::clasz::kBus:
    case n::clasz::kCoach: return osr::search_profile::kBus;
    case n::clasz::kTram:
    case n::clasz::kHighSpeed:
    case n::clasz::kLongDistance:
    case n::clasz::kNight:
    case n::clasz::kRegional:
    case n::clasz::kRegionalFast:
    case n::clasz::kSuburban:
    case n::clasz::kFunicular: return osr::search_profile::kRailway;
    default: return std::nullopt;
  }
}

void compute_shapes(
    osr::ways const& w,
    osr::lookup const& lookup,
    osr::platforms const& pl,
    n::timetable& tt,
    n::shapes_storage& shapes,
    std::optional<config::timetable::shapes_debug> const& debug) {
  fmt::println(std::clog, "computing shapes");

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Computing shapes")
      .out_bounds(0.F, 100.F)
      .in_high(tt.n_routes());

  auto routes_matched = 0ULL;
  auto segments_routed = 0ULL;
  auto segments_beelined = 0ULL;

  auto const debug_enabled =
      debug && !debug->path_.empty() &&
      (debug->all_ || debug->all_with_beelines_ ||
       (debug->trips_ && !debug->trips_->empty()) ||
       (debug->route_ids_ && !debug->route_ids_->empty()) ||
       (debug->route_indices_ && !debug->route_indices_->empty()));

  if (debug_enabled) {
    std::filesystem::create_directories(debug->path_);
  }

  shapes.trip_offset_indices_.resize(tt.n_trips());
  shapes.route_bboxes_.resize(tt.n_routes());
  shapes.route_segment_bboxes_.resize(tt.n_routes());

  for (auto r = n::route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto profile = get_profile(tt.route_clasz_[r]);
    if (!profile) {
      progress_tracker->increment();
      continue;
    }
    auto const profile_params = osr::get_parameters(*profile);

    auto const stops = tt.route_location_seq_[r];
    if (stops.size() < 2U) {
      std::cerr << "[compute_shapes] skipping route " << r << ", "
                << stops.size() << " stops\n";
      progress_tracker->increment();
      continue;
    }
    auto const transports = tt.route_transport_ranges_[r];
    auto debug_path_fn = std::function<std::optional<std::filesystem::path>(
        osr::matched_route const&)>{nullptr};

    if (debug_enabled) {
      debug_path_fn = [&debug, r, &tt](osr::matched_route const& res)
          -> std::optional<std::filesystem::path> {
        auto include =
            debug->all_ || (debug->all_with_beelines_ && res.n_beelined_ > 0U);
        auto tags = std::set<std::string>{};

        if (debug->route_indices_ && !debug->route_indices_->empty()) {
          auto const& debug_route_indices = *debug->route_indices_;
          if (std::ranges::contains(debug_route_indices, to_idx(r))) {
            include = true;
          }
        }

        if (debug->route_ids_ && !debug->route_ids_->empty()) {
          auto const& debug_route_ids = *debug->route_ids_;
          for (auto const transport_idx : tt.route_transport_ranges_[r]) {
            auto const frun = n::rt::frun{
                tt, nullptr,
                n::rt::run{
                    .t_ = n::transport{transport_idx, n::day_idx_t{0}},
                    .stop_range_ =
                        n::interval{n::stop_idx_t{0U},
                                    static_cast<n::stop_idx_t>(
                                        tt.route_location_seq_[r].size())},
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
          for (auto const transport_idx : tt.route_transport_ranges_[r]) {
            auto const frun = n::rt::frun{
                tt, nullptr,
                n::rt::run{
                    .t_ = n::transport{transport_idx},
                    .stop_range_ =
                        n::interval{n::stop_idx_t{0U},
                                    static_cast<n::stop_idx_t>(
                                        tt.route_location_seq_[r].size())},
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

        if (include) {
          auto fn = fmt::format("r_{}", to_idx(r));
          for (auto const& tag : tags) {
            fn += fmt::format("_{}", tag);
          }
          return debug->path_ / fn;
        } else {
          return {};
        }
      };
    }

    auto shape = std::vector<geo::latlng>{};

    auto offsets = std::vector<n::shape_offset_t>{};
    offsets.reserve(stops.size());

    auto route_bbox = geo::box{};
    auto segment_bboxes = std::vector<geo::box>{};
    segment_bboxes.reserve(stops.size() - 1U);

    auto const match_points = utl::to_vec(stops, [&](auto const stop_idx) {
      auto const loc_idx = n::stop{stop_idx}.location_idx();
      auto const pos = tt.locations_.coordinates_[loc_idx];
      return osr::location{pos, osr::level_t{osr::kNoLevel}};
    });

    // std::cerr << "[compute_shapes] route " << r << ", " << stops.size()
    //           << " stops, " << transports.size()
    //           << " transports, profile: " << to_str(*profile)
    //           << ", clasz: " << static_cast<int>(tt.route_clasz_[r]) <<
    //           "\n";

    try {
      auto const max_segment_cost = profile == osr::search_profile::kRailway
                                        ? osr::cost_t{10000U}
                                        : osr::cost_t{5000U};
      auto const matched_route =
          osr::map_match(w, lookup, *profile, profile_params, match_points,
                         max_segment_cost, nullptr, nullptr, debug_path_fn);

      ++routes_matched;
      segments_routed += matched_route.n_routed_;
      segments_beelined += matched_route.n_beelined_;

      // std::cerr << "  routed=" << matched_route.n_routed_
      //           << ", beelined=" << matched_route.n_beelined_ << "\n";

      utl::verify(matched_route.segment_offsets_.size() == match_points.size(),
                  "[compute_shapes] segment offsets ({}) != match points ({})",
                  matched_route.segment_offsets_.size(), match_points.size());

      segment_bboxes.resize(stops.size() - 1U);
      shape.clear();
      shape.reserve(matched_route.path_.segments_.size() * 8U);
      offsets.clear();
      offsets.reserve(stops.size());
      offsets.emplace_back(0U);

      for (auto seg_idx = 0U; seg_idx < stops.size() - 1U; ++seg_idx) {
        auto& seg_bbox = segment_bboxes[seg_idx];

        if (!shape.empty()) {
          seg_bbox.extend(shape.back());
        }

        auto const start = matched_route.segment_offsets_[seg_idx];
        auto const end = (seg_idx + 1U < stops.size() - 1U)
                             ? matched_route.segment_offsets_[seg_idx + 1U]
                             : matched_route.path_.segments_.size();

        for (auto ps_idx = start; ps_idx < end; ++ps_idx) {
          auto const& ps = matched_route.path_.segments_[ps_idx];
          for (auto const& pt : ps.polyline_) {
            route_bbox.extend(pt);
            seg_bbox.extend(pt);
          }
          if (!ps.polyline_.empty()) {
            auto first = ps.polyline_.begin();
            if (!shape.empty() && shape.back() == *first) {
              ++first;
            }
            shape.insert(shape.end(), first, ps.polyline_.end());
          }
        }

        offsets.emplace_back(static_cast<std::uint32_t>(shape.size() - 1U));
      }

      utl::verify(
          offsets.size() == stops.size(),
          "[compute_shapes] mismatch: offsets.size()={}, stops.size()={}",
          offsets.size(), stops.size());

      auto const shape_idx = static_cast<n::shape_idx_t>(shapes.data_.size());
      shapes.data_.emplace_back(shape);

      shapes.route_bboxes_[r] = route_bbox;
      // TODO
      // shapes.route_segment_bboxes_[r] = segment_bboxes;

      auto range_to_offsets = std::map<std::pair<n::stop_idx_t, n::stop_idx_t>,
                                       n::shape_offset_idx_t>{};

      for (auto const transport_idx : transports) {
        auto const frun = n::rt::frun{
            tt, nullptr,
            n::rt::run{.t_ = n::transport{transport_idx, n::day_idx_t{0}},
                       .stop_range_ = n::interval{n::stop_idx_t{0U},
                                                  static_cast<n::stop_idx_t>(
                                                      stops.size())},
                       .rt_ = n::rt_transport_idx_t::invalid()}};
        frun.for_each_trip([&](n::trip_idx_t const trip_idx,
                               n::interval<n::stop_idx_t> const range) {
          auto const key = std::pair{range.from_, range.to_};
          auto it = range_to_offsets.find(key);
          if (it == end(range_to_offsets)) {
            auto trip_offsets = std::vector<n::shape_offset_t>{};
            trip_offsets.reserve(range.size());
            for (auto const i : range) {
              trip_offsets.push_back(offsets.at(i));
            }
            auto const offsets_idx = shapes.add_offsets(trip_offsets);
            it = range_to_offsets.emplace(key, offsets_idx).first;
          }

          shapes.trip_offset_indices_[trip_idx] = {shape_idx, it->second};
        });
      }
    } catch (std::exception const& e) {
      fmt::println(std::clog,
                   "[compute_shapes] route {}: map matching failed: {}", r,
                   e.what());

      if (auto const trace =
              boost::stacktrace::stacktrace::from_current_exception();
          trace) {
        std::clog << trace << std::endl;
      }
    }
    progress_tracker->increment();
  }

  fmt::println(std::clog,
               "{} routes matched, {} segments routed, {} segments beelined",
               routes_matched, segments_routed, segments_beelined);
}

}  // namespace motis

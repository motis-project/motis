#include "motis/compute_footpaths.h"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include "utl/erase_if.h"
#include "utl/parallel_for.h"

#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"
#include "osr/util/infinite.h"
#include "osr/util/reverse.h"

#include "geo/latlng.h"

#include "utl/helpers/algorithm.h"
#include "utl/zip.h"

#include "motis/constants.h"
#include "motis/get_loc.h"
#include "motis/match_platforms.h"
#include "motis/osr/max_distance.h"
#include "motis/osr/parameters.h"
#include "motis/point_rtree.h"
#include "motis/route_coverage.h"

namespace n = nigiri;

namespace motis {

constexpr auto const kMaxMissingFootpathDistance = 100.0;

void compute_footpaths(
    osr::ways const& w,
    osr::lookup const& lookup,
    osr::platforms const& pl,
    nigiri::timetable& tt,
    n::mutable_fws_multimap<n::location_idx_t, n::footpath>& walks,
    platform_matches_t const& matches,
    way_matches_storage const* way_matches,
    osr::elevation_storage const* elevations,
    std::vector<routed_transfers_settings> const& settings) {
  fmt::println(std::clog, "creating r-tree");
  auto const loc_rtree = [&]() {
    auto t = point_rtree<n::location_idx_t>{};
    for (auto i = n::location_idx_t{0U}; i != tt.n_locations(); ++i) {
      t.add(tt.locations_.coordinates_[i], i);
    }
    return t;
  }();

  auto const pt = utl::get_active_progress_tracker();
  pt->in_high(2U * tt.n_locations() * settings.size());

  struct state {
    std::vector<n::location_idx_t> neighbors_;
    std::vector<osr::location> neighbors_loc_;
    osr::match_result neighbor_candidates_;
    std::vector<std::optional<osr::path>> results_;
    route_coverage route_coverage_;
  };

  auto n_done = 0U;
  // One flat SoA store for all locations; bucket index == location index.
  auto candidates = osr::match_result{};

  auto transfers = n::vector_map<n::location_idx_t, std::vector<n::footpath>>(
      tt.n_locations());
  for (auto const& mode : settings) {
    auto const is_default_profile = mode.profile_idx_ == n::kFootProfile;
    for (auto& fps : transfers) {
      fps.clear();
    }

    auto default_estimates =
        n::vector_map<n::location_idx_t, std::vector<n::footpath>>(
            is_default_profile ? tt.n_locations() : 0U);

    auto const is_candidate = [&](n::location_idx_t const l) {
      return !mode.is_candidate_ || mode.is_candidate_(l);
    };

    auto const covers_routes =
        mode.min_reachable_routes_ != 0U &&
        mode.max_route_covering_duration_ > mode.max_duration_;
    auto route_covering =
        n::vector_map<n::location_idx_t, std::vector<n::footpath>>(
            covers_routes ? tt.n_locations() : 0U);

    // The preprocessed candidates are geometric and capped at the
    // preprocessing radius, so they are only usable when that radius covers
    // this profile's matching distance.
    auto const use_raw_matches =
        way_matches != nullptr && !way_matches->matches_.empty() &&
        way_matches->max_matching_distance_ >= mode.max_matching_distance_;

    {
      auto const timer = utl::scoped_timer{
          fmt::format("matching timetable locations for profile={}",
                      to_str(mode.profile_))};

      candidates.clear();
      utl::parallel_ordered_collect_threadlocal<osr::match_result>(
          tt.n_locations(),
          [&](osr::match_result& tmp, std::size_t const x) {
            auto const l =
                n::location_idx_t{static_cast<n::location_idx_t::value_t>(x)};
            tmp.clear();
            if (is_candidate(l)) {
              auto raw = std::span<osr::raw_way_candidate const>{};
              if (use_raw_matches) {
                auto const& m = way_matches->matches_[l];
                raw = {m.begin(), m.end()};
              }
              lookup.match(to_profile_parameters(mode.profile_, {}),
                           get_loc(tt, w, pl, matches, l), false,
                           osr::direction::kForward,
                           mode.max_matching_distance_, nullptr, mode.profile_,
                           raw, tmp);
            } else {
              tmp.start(osr::kNoLevel);
              tmp.finish();
            }
            return tmp;
          },
          [&](std::size_t, osr::match_result&& m) {
            candidates.append(m, osr::match_idx_t{0U});
          },
          pt->update_fn());

      n_done += tt.n_locations();
    }

    utl::parallel_for_run_threadlocal<state>(
        tt.n_locations(), [&](state& s, auto const i) {
          cista::for_each_field(s, [](auto& f) { f.clear(); });

          auto const l = n::location_idx_t{i};
          if (!is_candidate(l)) {
            pt->update_monotonic(n_done + i);
            return;
          }

          auto const route_neighbors = [&](std::chrono::seconds const max) {
            s.neighbors_.clear();
            s.neighbors_loc_.clear();
            loc_rtree.in_radius(
                tt.locations_.coordinates_[l],
                get_max_distance(mode.profile_, osr_parameters{}, max),
                [&](n::location_idx_t const x) {
                  if (x != l && is_candidate(x)) {
                    s.neighbors_.emplace_back(x);
                  }
                });
            s.results_ =
                osr::route_one_to_many(
                    to_profile_parameters(mode.profile_, {}), w, lookup,
                    mode.profile_, get_loc(tt, w, pl, matches, l),
                    utl::transform_to(s.neighbors_, s.neighbors_loc_,
                                      [&](n::location_idx_t const x) {
                                        return get_loc(tt, w, pl, matches, x);
                                      }),
                    candidates[osr::match_idx_t{to_idx(l)}],
                    [&]() -> osr::match_result const& {
                      s.neighbor_candidates_.clear();
                      for (auto const x : s.neighbors_) {
                        s.neighbor_candidates_.append(
                            candidates, osr::match_idx_t{to_idx(x)});
                      }
                      return s.neighbor_candidates_;
                    }(),
                    max, osr::direction::kForward, nullptr, nullptr, elevations,
                    [](osr::path const& p) { return p.uses_elevator_; })
                    ->results();
          };

          auto const to_minutes = [](osr::path const& p) {
            return n::duration_t{
                static_cast<unsigned>(std::ceil(p.duration_.count() / 60.0))};
          };

          auto& coverage = s.route_coverage_;
          auto const has_routes = !tt.location_routes_[l].empty();
          for (auto max = mode.max_duration_;;
               max = std::min(2 * max, mode.max_route_covering_duration_)) {
            route_neighbors(max);
            if (!covers_routes || !has_routes) {
              break;
            }

            coverage.reached_.clear();
            for (auto const [n, r] : utl::zip(s.neighbors_, s.results_)) {
              if (r.has_value() && !tt.location_routes_[n].empty()) {
                coverage.reached_.emplace_back(to_minutes(*r), n);
              }
            }
            if (coverage.select(tt, l, mode.min_reachable_routes_) ||
                max >= mode.max_route_covering_duration_) {
              break;
            }
          }
          auto const& results = s.results_;

          for (auto const [n, r] : utl::zip(s.neighbors_, results)) {
            if (!r.has_value()) {
              continue;
            }

            auto const duration = to_minutes(*r);
            if (duration > mode.max_duration_ && !coverage.is_selected(n)) {
              continue;
            }
            auto const fp =
                n::footpath{n, std::min(duration, n::footpath::kMax)};
            transfers[l].push_back(fp);
            if (covers_routes && coverage.is_selected(n)) {
              route_covering[l].push_back(fp);
            }
          }

          if (mode.extend_missing_ || is_default_profile) {
            for (auto const [n, r] : utl::zip(s.neighbors_, results)) {
              if (r.has_value()) {
                continue;
              }
              auto const dist = geo::distance(tt.locations_.coordinates_[l],
                                              tt.locations_.coordinates_[n]);
              auto const estimate = n::footpath{n, n::walk_duration(dist)};
              auto const is_close = dist < kMaxMissingFootpathDistance;
              if (mode.extend_missing_ && is_close) {
                transfers[l].push_back(estimate);
              } else if (is_default_profile &&
                         (is_close || tt.locations_.get_root_idx(l) ==
                                          tt.locations_.get_root_idx(n))) {
                default_estimates[l].push_back(estimate);
              }
            }
          }

          utl::erase_if(transfers[l], [&](n::footpath fp) {
            return fp.duration() > mode.max_duration_ &&
                   !coverage.is_selected(fp.target());
          });
          utl::sort(transfers[l]);

          pt->update_monotonic(n_done + i);
        });

    // Mirror route covering footpaths (assuming symmetric walking times) so
    // that stops picked by a remote stop can also walk back to it.
    for (auto l = n::location_idx_t{0U};
         l != n::location_idx_t{route_covering.size()}; ++l) {
      for (auto const fp : route_covering[l]) {
        auto& back = transfers[fp.target()];
        if (utl::none_of(
                back, [&](n::footpath const x) { return x.target() == l; })) {
          back.push_back(n::footpath{l, fp.duration()});
          utl::sort(back);
        }
      }
    }

    for (auto const& x : transfers) {
      tt.locations_.footpaths_out_[mode.profile_idx_].emplace_back(x);
    }

    if (is_default_profile) {
      walks.clear();
      for (auto const [routed, estimates] :
           utl::zip(transfers, default_estimates)) {
        auto bucket = walks.emplace_back();
        for (auto const fp : routed) {
          bucket.push_back(fp);
        }
        for (auto const fp : estimates) {
          bucket.push_back(fp);
        }
      }
    }

    n_done += tt.n_locations();
  }
}

}  // namespace motis

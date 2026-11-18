#include "motis/compute_footpaths.h"

#include <atomic>

#include "cista/mmap.h"
#include "cista/serialization.h"

#include "utl/enumerate.h"
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
#include "motis/osr/footpath_restriction.h"
#include "motis/osr/max_distance.h"
#include "motis/osr/parameters.h"
#include "motis/point_rtree.h"

namespace n = nigiri;

namespace motis {

constexpr auto const kMaxMissingFootpathDistance = 100.0;

void compute_footpaths(
    osr::ways const& w,
    osr::lookup const& lookup,
    osr::platforms const& pl,
    nigiri::timetable& tt,
    n::mutable_fws_multimap<n::location_idx_t, n::footpath>& walks,
    n::loader::station_footpaths_t& restricted,
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
    std::vector<n::loader::station_footpath> pareto_;
  };

  auto n_done = 0U;
  // One flat SoA store for all locations; bucket index == location index.
  auto candidates = osr::match_result{};

  auto transfers = n::vector_map<n::location_idx_t, std::vector<n::footpath>>(
      tt.n_locations());
  auto fastest = n::vector_map<n::location_idx_t, std::vector<n::footpath>>(
      tt.n_locations());
  for (auto const& mode : settings) {
    auto const is_default_profile = mode.profile_idx_ == n::kFootProfile;
    // Foot: one multi-criteria search per location, one footpath per Pareto
    // result (wheelchair queries use them with their restrictions).
    auto const with_alternatives = mode.profile_ == osr::search_profile::kFoot;
    auto n_pairs = std::atomic_uint64_t{0U};
    auto n_restricted = std::atomic_uint64_t{0U};
    if (with_alternatives) {
      restricted[mode.profile_idx_].clear();
      restricted[mode.profile_idx_].resize(tt.n_locations());
    }
    for (auto& fps : transfers) {
      fps.clear();
    }
    for (auto& fps : fastest) {
      fps.clear();
    }

    auto default_estimates =
        n::vector_map<n::location_idx_t, std::vector<n::footpath>>(
            is_default_profile ? tt.n_locations() : 0U);

    auto const is_candidate = [&](n::location_idx_t const l) {
      return !mode.is_candidate_ || mode.is_candidate_(l);
    };

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

    auto const routing_timer = utl::scoped_timer{
        fmt::format("routing footpaths for profile={}", to_str(mode.profile_))};
    utl::parallel_for_run_threadlocal<state>(
        tt.n_locations(), [&](state& s, auto const i) {
          cista::for_each_field(s, [](auto& f) { f.clear(); });

          auto const l = n::location_idx_t{i};
          if (!is_candidate(l)) {
            pt->update_monotonic(n_done + i);
            return;
          }

          loc_rtree.in_radius(tt.locations_.coordinates_[l],
                              get_max_distance(mode.profile_, osr_parameters{},
                                               mode.max_duration_),
                              [&](n::location_idx_t const x) {
                                if (x != l && is_candidate(x)) {
                                  s.neighbors_.emplace_back(x);
                                }
                              });

          auto const from = get_loc(tt, w, pl, matches, l);
          utl::transform_to(s.neighbors_, s.neighbors_loc_,
                            [&](n::location_idx_t const x) {
                              return get_loc(tt, w, pl, matches, x);
                            });
          s.neighbor_candidates_.clear();
          for (auto const x : s.neighbors_) {
            s.neighbor_candidates_.append(candidates,
                                          osr::match_idx_t{to_idx(x)});
          }
          auto const route_minutes = [](osr::duration_t const d) {
            return n::duration_t{
                static_cast<unsigned>(std::ceil(d.count() / 60.0))};
          };

          // Reached neighbors (for the estimates of missing walks below).
          auto is_reached = std::vector<bool>(s.neighbors_.size(), false);
          if (!with_alternatives) {
            auto const results =
                osr::route_one_to_many(
                    to_profile_parameters(mode.profile_, {}), w, lookup,
                    mode.profile_, from, s.neighbors_loc_,
                    candidates[osr::match_idx_t{to_idx(l)}],
                    s.neighbor_candidates_, mode.max_duration_,
                    osr::direction::kForward, nullptr, nullptr, elevations)
                    ->results();
            for (auto const [j, r] : utl::enumerate(results)) {
              if (r.has_value()) {
                is_reached[j] = true;
                transfers[l].emplace_back(
                    n::footpath{s.neighbors_[j], route_minutes(r->duration_)});
              }
            }
          } else {
            auto const results = osr::route_pareto(
                to_profile_parameters(mode.profile_, {}), w, lookup,
                mode.profile_, from, s.neighbors_loc_,
                candidates[osr::match_idx_t{to_idx(l)}], s.neighbor_candidates_,
                mode.max_duration_, osr::direction::kForward, nullptr,
                elevations);
            for (auto const [j, paths] : utl::enumerate(results)) {
              if (paths.empty()) {
                continue;
              }
              is_reached[j] = true;
              auto const target = s.neighbors_[j];
              s.pareto_.clear();
              for (auto const& p : paths) {
                s.pareto_.push_back(n::loader::station_footpath{
                    .target_ = target,
                    .duration_ = to_minutes(p),
                    .restriction_ = to_restriction(p)});
              }
              fastest[l].emplace_back(n::footpath{
                  target, utl::min_element(s.pareto_, [](auto&& a, auto&& b) {
                            return a.duration_ < b.duration_;
                          })->duration_});
              n::loader::remove_dominated(s.pareto_);
              ++n_pairs;
              if (s.pareto_.size() == 1U &&
                  s.pareto_.front().restriction_ == n::footpath_restriction{}) {
                transfers[l].emplace_back(
                    n::footpath{target, s.pareto_.front().duration_});
                continue;
              }
              for (auto const& fp : s.pareto_) {
                if (fp.duration_ <= mode.max_duration_) {
                  restricted[mode.profile_idx_][l].push_back(fp);
                  ++n_restricted;
                }
              }
            }
          }

          if (mode.extend_missing_ || is_default_profile) {
            for (auto const [j, n] : utl::enumerate(s.neighbors_)) {
              if (is_reached[j]) {
                continue;
              }
              auto const dist = geo::distance(tt.locations_.coordinates_[l],
                                              tt.locations_.coordinates_[n]);
              auto const estimate = n::footpath{n, n::walk_duration(dist)};
              auto const is_close = dist < kMaxMissingFootpathDistance;
              if (mode.extend_missing_ && is_close) {
                transfers[l].push_back(estimate);
                if (with_alternatives) {
                  fastest[l].push_back(estimate);
                }
              } else if (is_default_profile &&
                         (is_close || tt.locations_.get_root_idx(l) ==
                                          tt.locations_.get_root_idx(n))) {
                default_estimates[l].push_back(estimate);
              }
            }
          }

          for (auto* fps : {&transfers[l], &fastest[l]}) {
            utl::erase_if(*fps, [&](n::footpath fp) {
              return fp.duration() > mode.max_duration_;
            });
            utl::sort(*fps);
          }

          pt->update_monotonic(n_done + i);
        });

    for (auto const& x : transfers) {
      tt.locations_.footpaths_out_[mode.profile_idx_].emplace_back(x);
    }
    if (with_alternatives) {
      fmt::println(std::clog,
                   "profile={}: {} stop pairs, {} footpaths with restriction "
                   "records",
                   to_str(mode.profile_), n_pairs.load(), n_restricted.load());
    }

    if (is_default_profile) {
      walks.clear();
      for (auto const [routed, estimates] :
           utl::zip(fastest, default_estimates)) {
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

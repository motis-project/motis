#include "motis/compute_footpaths.h"

#include "nigiri/loader/build_lb_graph.h"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include "utl/concat.h"
#include "utl/erase_if.h"
#include "utl/parallel_for.h"
#include "utl/sorted_diff.h"

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

namespace n = nigiri;

namespace motis {

// below this, a missing routing result is an OSM data error, not a real gap
constexpr auto const kMaxMissingFootpathDistance = 100.0;

elevator_footpath_map_t compute_footpaths(
    osr::ways const& w,
    osr::lookup const& lookup,
    osr::platforms const& pl,
    nigiri::timetable& tt,
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

  auto elevator_in_paths_mutex = std::mutex{};
  auto elevator_in_paths = elevator_footpath_map_t{};
  auto const add_if_elevator = [&](osr::node_idx_t const n,
                                   n::location_idx_t const a,
                                   n::location_idx_t const b) {
    if (n != osr::node_idx_t::invalid() &&
        w.r_->node_properties_[n].is_elevator()) {
      auto l = std::unique_lock{elevator_in_paths_mutex};
      elevator_in_paths[n].emplace(a, b);
    }
  };

  struct state {
    std::vector<n::footpath> sorted_tt_fps_;
    std::vector<n::footpath> missing_;
    std::vector<n::location_idx_t> neighbors_;
    std::vector<osr::location> neighbors_loc_;
    osr::match_result neighbor_candidates_;
  };

  auto n_done = 0U;
  // One flat SoA store for all locations; bucket index == location index.
  auto candidates = osr::match_result{};

  auto transfers = n::vector_map<n::location_idx_t, std::vector<n::footpath>>(
      tt.n_locations());
  for (auto const& mode : settings) {
    for (auto& fps : transfers) {
      fps.clear();
    }

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

          auto const results = osr::route(
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
                  s.neighbor_candidates_.append(candidates,
                                                osr::match_idx_t{to_idx(x)});
                }
                return s.neighbor_candidates_;
              }(),
              static_cast<osr::cost_t>(mode.max_duration_.count()),
              osr::direction::kForward, nullptr, nullptr, elevations,
              [](osr::path const& p) { return p.uses_elevator_; });

          for (auto const [n, r] : utl::zip(s.neighbors_, results)) {
            if (!r.has_value()) {
              continue;
            }

            auto const duration = n::duration_t{
                static_cast<unsigned>(std::ceil(r->cost_ / 60.0))};
            transfers[l].emplace_back(n::footpath{n, duration});

            if (mode.profile_ == osr::search_profile::kWheelchair) {
              for (auto const& seg : r->segments_) {
                add_if_elevator(seg.from_, l, n);
                add_if_elevator(seg.from_, n, l);
              }
            }
          }

          // transfers.txt is authoritative: a rule fixes the transfer time,
          // which may be shorter or longer than the walk, so routing may only
          // fill the pairs it does not speak about.
          if (mode.profile_idx_ == n::kFootProfile &&
              to_idx(l) < tt.locations_.transfer_rule_fps_.size()) {
            auto const rules = tt.locations_.transfer_rule_fps_[l];
            if (rules.begin() != rules.end()) {
              utl::erase_if(transfers[l], [&](n::footpath const fp) {
                return utl::any_of(rules, [&](n::footpath const r) {
                  return r.target() == fp.target();
                });
              });
              for (auto const r : rules) {
                transfers[l].emplace_back(r);
              }
            }
          }

          // A pair the router cannot connect over a few meters is an OSM data
          // error rather than a real gap: bridge it with the beeline.
          if (mode.extend_missing_) {
            for (auto const [n, r] : utl::zip(s.neighbors_, results)) {
              if (r.has_value()) {
                continue;
              }
              auto const dist = geo::distance(
                  tt.locations_.coordinates_[l], tt.locations_.coordinates_[n]);
              if (dist >= kMaxMissingFootpathDistance ||
                  utl::any_of(transfers[l], [&, n = n](n::footpath const fp) {
                    return fp.target() == n;
                  })) {
                continue;
              }
              transfers[l].emplace_back(
                  n::footpath{n, n::duration_t{static_cast<int>(
                                     std::ceil((dist / 0.7) / 60.0))}});
            }
          }

          utl::erase_if(transfers[l], [&](n::footpath fp) {
            return fp.duration() > mode.max_duration_;
          });
          utl::sort(transfers[l]);

          pt->update_monotonic(n_done + i);
        });

    auto transfers_in =
        n::vector_map<n::location_idx_t, std::vector<n::footpath>>{};
    transfers_in.resize(tt.n_locations());
    for (auto const [i, out] : utl::enumerate(transfers)) {
      auto const l = n::location_idx_t{i};
      for (auto const fp : out) {
        assert(fp.target() < tt.n_locations());
        transfers_in[fp.target()].push_back(n::footpath{l, fp.duration()});
      }
    }
    for (auto const& x : transfers) {
      tt.locations_.footpaths_out_[mode.profile_idx_].emplace_back(x);
    }
    for (auto const& x : transfers_in) {
      tt.locations_.footpaths_in_[mode.profile_idx_].emplace_back(x);
    }

    n::loader::build_lb_graph<n::direction::kForward>(tt, mode.profile_idx_);
    n::loader::build_lb_graph<n::direction::kBackward>(tt, mode.profile_idx_);

    n_done += tt.n_locations();
  }

  return elevator_in_paths;
}

}  // namespace motis

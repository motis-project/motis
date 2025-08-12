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

#include "motis/constants.h"
#include "motis/get_loc.h"
#include "motis/match_platforms.h"
#include "motis/max_distance.h"
#include "motis/point_rtree.h"

namespace n = nigiri;

namespace motis {

elevator_footpath_map_t compute_footpaths(
    osr::ways const& w,
    osr::lookup const& lookup,
    osr::platforms const& pl,
    nigiri::timetable& tt,
    osr::elevation_storage const* elevations,
    bool const update_coordinates,
    std::vector<routed_transfers_settings> const& settings) {
  fmt::println(std::clog, "creating matches");
  auto const matches = get_matches(tt, pl, w);

  fmt::println(std::clog, "  -> creating r-tree");
  auto const loc_rtree = [&]() {
    auto t = point_rtree<n::location_idx_t>{};
    for (auto i = n::location_idx_t{0U}; i != tt.n_locations(); ++i) {
      if (update_coordinates && matches[i] != osr::platform_idx_t::invalid()) {
        auto const center = get_platform_center(pl, w, matches[i]);
        if (center.has_value() &&
            geo::distance(*center, tt.locations_.coordinates_[i]) <
                kMaxAdjust) {
          tt.locations_.coordinates_[i] = *center;
        }
      }
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
    std::vector<osr::match_t> neighbor_candidates_;
  };

  auto n_done = 0U;
  auto candidates = vector_map<n::location_idx_t, osr::match_t>{};
  auto transfers = n::vector_map<n::location_idx_t, std::vector<n::footpath>>(
      tt.n_locations());
  for (auto const& mode : settings) {
    candidates.clear();
    candidates.resize(tt.n_locations());
    for (auto& fps : transfers) {
      fps.clear();
    }

    auto const is_candidate = [&](n::location_idx_t const l) {
      return !mode.is_candidate_ || mode.is_candidate_(l);
    };

    {
      auto const timer = utl::scoped_timer{
          fmt::format("matching timetable locations for profile={}",
                      to_str(mode.profile_))};

      utl::parallel_for_run(tt.n_locations(), [&](std::size_t const x) {
        pt->update_monotonic(n_done + x);

        auto const l =
            n::location_idx_t{static_cast<n::location_idx_t::value_t>(x)};
        if (!is_candidate(l)) {
          return;
        }
        candidates[l] = lookup.match(
            get_loc(tt, w, pl, matches, l), false, osr::direction::kForward,
            mode.max_matching_distance_, nullptr, mode.profile_);
      });

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

          loc_rtree.in_radius(
              tt.locations_.coordinates_[l],
              get_max_distance(mode.profile_, mode.max_duration_),
              [&](n::location_idx_t const x) {
                if (x != l && is_candidate(x)) {
                  s.neighbors_.emplace_back(x);
                }
              });

          auto const results = osr::route(
              w, lookup, mode.profile_, get_loc(tt, w, pl, matches, l),
              utl::transform_to(s.neighbors_, s.neighbors_loc_,
                                [&](n::location_idx_t const x) {
                                  return get_loc(tt, w, pl, matches, x);
                                }),
              candidates[l],
              utl::transform_to(
                  s.neighbors_, s.neighbor_candidates_,
                  [&](n::location_idx_t const x) { return candidates[x]; }),
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

          if (mode.extend_missing_) {
            auto const& tt_fps = tt.locations_.footpaths_out_[0].at(l);
            s.sorted_tt_fps_.resize(tt_fps.size());
            std::copy(begin(tt_fps), end(tt_fps), begin(s.sorted_tt_fps_));
            utl::sort(s.sorted_tt_fps_);
            utl::sort(transfers[l]);

            utl::sorted_diff(
                s.sorted_tt_fps_, transfers[l],
                [](auto&& a, auto&& b) { return a.target() < b.target(); },
                [](auto&& a, auto&& b) { return a.target() == b.target(); },
                utl::overloaded{
                    [](n::footpath, n::footpath) { assert(false); },
                    [&](utl::op const op, n::footpath const x) {
                      if (op == utl::op::kDel) {
                        auto const dist = geo::distance(
                            tt.locations_.coordinates_[l],
                            tt.locations_.coordinates_[x.target()]);
                        if (mode.extend_missing_ < 100.0) {
                          auto const duration = n::duration_t{
                              static_cast<int>(std::ceil((dist / 0.7) / 60.0))};
                          s.missing_.emplace_back(x.target(), duration);
                        }
                      }
                    }});

            utl::concat(transfers[l], s.missing_);
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

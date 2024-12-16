#include "motis/match_platforms.h"

#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"
#include "utl/parser/arg_parser.h"

#include "osr/geojson.h"

#include "motis/location_routes.h"

namespace n = nigiri;

namespace motis {

bool is_number(char const x) { return x >= '0' && x <= '9'; }

template <typename Fn>
void for_each_number(std::string_view x, Fn&& fn) {
  for (auto i = 0U; i < x.size(); ++i) {
    if (!is_number(x[i])) {
      continue;
    }

    auto j = i + 1U;
    for (; j != x.size(); ++j) {
      if (!is_number(x[j])) {
        break;
      }
    }

    fn(utl::parse<unsigned>(x.substr(i, j - i)));
    i = j;
  }
}

bool has_number_match(std::string_view a, std::string_view b) {
  auto match = false;
  for_each_number(a, [&](unsigned const x) {
    for_each_number(b, [&](unsigned const y) { match = (x == y); });
  });
  return match;
}

template <typename Collection>
bool has_number_match(Collection&& a, std::string_view b) {
  return std::any_of(a.begin(), a.end(),
                     [&](auto&& x) { return has_number_match(x.view(), b); });
}

template <typename Collection>
bool has_exact_match(Collection&& a, std::string_view b) {
  return std::any_of(a.begin(), a.end(),
                     [&](auto&& x) { return x.view() == b; });
}

template <typename Collection>
bool has_contains_match(Collection&& a, std::string_view b) {
  return std::any_of(a.begin(), a.end(),
                     [&](auto&& x) { return x.view().contains(b); });
}

std::optional<std::string_view> get_track(std::string_view s) {
  if (s.size() == 0 || std::isdigit(s.back()) == 0) {
    return std::nullopt;
  }
  for (auto i = 0U; i != s.size(); ++i) {
    auto const j = s.size() - i - 1U;
    if (std::isdigit(s[j]) == 0U) {
      return s.substr(j + 1U);
    }
  }
  return s;
}

template <typename Collection>
double get_routes_bonus(n::timetable const& tt,
                        n::location_idx_t const l,
                        Collection&& names) {
  auto matches = 0U;
  for (auto const& r : get_location_routes(tt, l)) {
    for (auto const& x : names) {
      if (r == x.view()) {
        ++matches;
      }

      utl::for_each_token(x.view(), ' ', [&](auto&& token) {
        if (r == token.view()) {
          ++matches;
        }
      });
    }
  }

  return matches * 20U;
}

template <typename Collection>
double get_match_bonus(Collection&& names,
                       std::string_view ref,
                       std::string_view name) {
  auto bonus = 0U;
  auto const size = static_cast<double>(name.size());
  if (has_exact_match(names, ref)) {
    bonus += std::max(0.0, 200.0 - size);
  }
  if (has_number_match(names, name)) {
    bonus += std::max(0.0, 140.0 - size);
  }
  if (auto const track = get_track(ref);
      track.has_value() && has_number_match(names, *track)) {
    bonus += std::max(0.0, 60.0 - size);
  }
  if (has_exact_match(names, name)) {
    bonus += std::max(0.0, 15.0 - size);
  }
  if (has_contains_match(names, ref)) {
    bonus += std::max(0.0, 5.0 - size);
  }
  return bonus;
}

struct center {
  template <typename T>
  void add(T const& polyline) {
    for (auto const& x : polyline) {
      add(geo::latlng(x));
    }
  }

  void add(geo::latlng const& pos) {
    sum_.lat_ += pos.lat();
    sum_.lng_ += pos.lng();
    n_ += 1U;
  }

  geo::latlng get_center() const { return {sum_.lat_ / n_, sum_.lng_ / n_}; }

  geo::latlng sum_;
  std::size_t n_;
};

std::optional<geo::latlng> get_platform_center(osr::platforms const& pl,
                                               osr::ways const& w,
                                               osr::platform_idx_t const x) {
  auto c = center{};
  for (auto const p : pl.platform_ref_[x]) {
    std::visit(utl::overloaded{[&](osr::node_idx_t const node) {
                                 c.add(pl.get_node_pos(node).as_latlng());
                               },
                               [&](osr::way_idx_t const way) {
                                 c.add(w.way_polylines_[way]);
                               }},
               osr::to_ref(p));
  }
  if (c.n_ == 0U) {
    return std::nullopt;
  }

  auto const center = c.get_center();
  auto closest = geo::latlng{};
  auto update_closest = [&, dist = std::numeric_limits<double>::max()](
                            geo::latlng const& candidate) mutable {
    auto const candidate_dist = geo::distance(candidate, center);
    if (candidate_dist < dist) {
      closest = candidate;
      dist = candidate_dist;
    }
  };
  for (auto const p : pl.platform_ref_[x]) {
    std::visit(
        utl::overloaded{
            [&](osr::node_idx_t const node) {
              update_closest(pl.get_node_pos(node).as_latlng());
            },
            [&](osr::way_idx_t const way) {
              for (auto const [a, b] : utl::pairwise(w.way_polylines_[way])) {
                update_closest(geo::closest_on_segment(center, a, b));
              }
            }},
        osr::to_ref(p));
  }
  return closest;
}

vector_map<n::location_idx_t, osr::platform_idx_t> get_matches(
    nigiri::timetable const& tt, osr::platforms const& pl, osr::ways const& w) {
  auto m = n::vector_map<n::location_idx_t, osr::platform_idx_t>{};
  m.resize(tt.n_locations());
  utl::parallel_for_run(tt.n_locations(), [&](auto const i) {
    auto const l = n::location_idx_t{i};
    m[l] = get_match(tt, pl, w, l);
  });
  return m;
}

osr::platform_idx_t get_match(n::timetable const& tt,
                              osr::platforms const& pl,
                              osr::ways const& w,
                              n::location_idx_t const l) {
  auto const ref = tt.locations_.coordinates_[l];
  auto best = osr::platform_idx_t::invalid();
  auto best_score = std::numeric_limits<double>::max();

  pl.find(ref, [&](osr::platform_idx_t const x) {
    auto const center = get_platform_center(pl, w, x);
    if (!center.has_value()) {
      return;
    }

    auto const dist = geo::distance(*center, ref);
    auto const match_bonus =
        get_match_bonus(pl.platform_names_[x], tt.locations_.ids_[l].view(),
                        tt.locations_.names_[l].view());
    auto const lvl = pl.get_level(w, x);
    auto const lvl_bonus =
        lvl != osr::kNoLevel && lvl.to_float() != 0.0F ? 5 : 0;
    auto const way_bonus = osr::is_way(pl.platform_ref_[x].front()) ? 20 : 0;
    auto const routes_bonus = get_routes_bonus(tt, l, pl.platform_names_[x]);
    auto const score =
        dist - match_bonus - way_bonus - lvl_bonus - routes_bonus;
    if (score < best_score) {
      best = x;
      best_score = score;
    }
  });

  if (best != osr::platform_idx_t::invalid()) {
    get_match_bonus(pl.platform_names_[best], tt.locations_.ids_[l].view(),
                    tt.locations_.names_[l].view());
  }

  return best;
}

}  // namespace motis
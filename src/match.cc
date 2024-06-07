#include "icc/match.h"

#include "utl/helpers/algorithm.h"
#include "utl/parser/arg_parser.h"

#include "osr/geojson.h"

namespace n = nigiri;

namespace icc {

constexpr auto const kNumberMatchBonus = 200.0;

bool is_number(char const x) { return x >= '0' && x <= '9'; }

template <typename Fn>
void for_each_number(std::string_view x, Fn&& fn) {
  for (auto i = 0U; i <= x.size(); ++i) {
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

template <typename Collection>
double get_match_bonus(Collection&& names,
                       std::string_view ref,
                       std::string_view name) {
  if (has_exact_match(names, ref)) {
    return 200.0 - names.size();
  }
  if (has_number_match(names, name)) {
    return 150.0 - names.size();
  }
  if (has_exact_match(names, name)) {
    return 15.0 - names.size();
  }
  if (has_contains_match(names, ref)) {
    return 5.0 - names.size();
  }
  return 0.0;
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
  return c.get_center();
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
        lvl != osr::level_t::invalid() && osr::to_float(lvl) != 0.0F ? 5 : 0;
    auto const way_bonus = osr::is_way(pl.platform_ref_[x].front()) ? 20 : 0;
    auto const score = dist - match_bonus - way_bonus - lvl_bonus;
    if (score < best_score) {
      best = x;
      best_score = score;
    }
  });

  return best;
}

matching_t match(n::timetable const& tt,
                 osr::platforms const& pl,
                 osr::ways const& w) {
  auto m = matching_t{};
  m.resize(tt.n_locations());
  utl::fill(m, osr::platform_idx_t::invalid());
  for (auto l = n::location_idx_t{0U}; l != tt.n_locations(); ++l) {
    m[l] = get_match(tt, pl, w, l);
  }
  return m;
}

}  // namespace icc
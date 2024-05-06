#include "icc/match.h"

#include "utl/parser/arg_parser.h"

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

matching match(nigiri::timetable const& tt,
               osr::platforms const& pl,
               osr::ways const& w) {
  auto const platform_center = [&](osr::platform_idx_t const x) {
    auto c = center{};
    for (auto const p : pl.platform_ref_[x]) {
      std::visit(utl::overloaded{[&](osr::node_idx_t const node) {
                                   c.add(w.get_node_pos(node).as_latlng());
                                 },
                                 [&](osr::way_idx_t const way) {
                                   c.add(w.way_polylines_[way]);
                                 }},
                 osr::to_ref(p));
    }
    return c.get_center();
  };

  auto m = matching{};
  for (auto l = n::location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto const ref = tt.locations_.coordinates_[l];
    auto best = osr::platform_idx_t::invalid();
    auto best_score = std::numeric_limits<double>::max();

    pl.find(ref, [&](osr::platform_idx_t const x) {
      auto const center = platform_center(x);
      auto const dist = geo::distance(center, ref);
      auto const is_number_match = has_number_match(
          pl.platform_names_[x], tt.locations_.names_[l].view());
      auto const score = dist - (is_number_match ? kNumberMatchBonus : 0.0);
      if (score < best_score) {
        best = x;
        best_score = score;
      }
    });

    if (best != osr::platform_idx_t ::invalid()) {
      m.pl_[best] = l;
      m.lp_[l] = best;
    }
  }
  return m;
}

}  // namespace icc
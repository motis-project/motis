#include "osr/geojson.h"

#include "motis/adr_extend_tt.h"

#include "nigiri/special_stations.h"

#include <string>
#include <utility>
#include <vector>

#include "utl/get_or_create.h"
#include "utl/parallel_for.h"
#include "utl/timer.h"
#include "utl/to_vec.h"

#include "nigiri/timetable.h"
#include "nigiri/translations_view.h"

#include "adr/area_database.h"
#include "adr/score.h"
#include "adr/typeahead.h"

#include "motis/types.h"

namespace a = adr;
namespace n = nigiri;
namespace json = boost::json;

namespace motis {

constexpr auto const kClaszMax =
    static_cast<std::underlying_type_t<n::clasz>>(n::kNumClasses);

date::time_zone const* get_tz(n::timetable const& tt,
                              adr_ext const* ae,
                              tz_map_t const* tz,
                              n::location_idx_t const l) {
  auto const p = tt.locations_.parents_[l];
  auto const x = p == n::location_idx_t::invalid() ? l : p;

  auto const p_idx =
      !ae || !tz ? adr_extra_place_idx_t::invalid() : ae->location_place_.at(x);
  if (p_idx != adr_extra_place_idx_t::invalid()) {
    return tz->at(p_idx);
  }

  return nullptr;
}

void normalize(std::string& x) {
  auto const replace_str = [&](std::string_view search,
                               std::string_view replace) {
    auto const pos = x.find(search);
    if (pos != std::string::npos) {
      x.replace(pos, search.size(), replace);
    }
  };

  replace_str("Int.", "Hbf");

  x = adr::normalize(x);

  auto const removals = std::initializer_list<std::string_view>{
      "tief", "oben",    "gleis", "platform", "gl",
      "gare", "bahnhof", "bhf",   "strasse",  "gasse"};
  for (auto const r : removals) {
    auto const pos = x.find(r);
    if (pos != std::string::npos) {
      x.erase(pos, r.size());
    }
  }

  auto const replacements =
      std::initializer_list<std::pair<std::string_view, std::string_view>>{
          {"flixtrain", "hbf"}, {"hauptbf", "hbf"},       {"haupt", "hbf"},
          {"centrale", "hbf"},  {"station", "hbf"},       {"zob", "hbf"},
          {"int", "hbf"},       {"international", "hbf"}, {"anleger", "fähre"}};
  for (auto const& [search, replace] : replacements) {
    replace_str(search, replace);
  }
}

adr::score_t get_diff(std::string str1,
                      std::string str2,
                      std::vector<adr::sift_offset>& sift4_offset_arr) {
  str1 = adr::normalize(str1);
  str2 = adr::normalize(str2);

  normalize(str1);
  normalize(str2);

  if (str1.contains("hbf") && str2.contains("hbf")) {
    return 0;
  }

  auto a = std::vector<std::string_view>{};
  auto b = std::vector<std::string_view>{};

  adr::for_each_token(
      str1,
      [&](auto&& p_token) mutable {
        if (!p_token.empty()) {
          a.emplace_back(p_token);
        }
        return utl::continue_t::kContinue;
      },
      ' ', '-');
  adr::for_each_token(
      str2,
      [&](auto&& p_token) mutable {
        if (!p_token.empty()) {
          b.emplace_back(p_token);
        }
        return utl::continue_t::kContinue;
      },
      ' ', '-');

  auto covered = std::uint32_t{};
  auto score = adr::score_t{0U};

  for (auto i = 0U; i != a.size(); ++i) {
    auto best = std::numeric_limits<adr::score_t>::max();
    auto best_j = 0U;
    for (auto j = 0U; j != b.size(); ++j) {
      if ((covered & (1U << j)) != 0U) {
        continue;
      }

      auto const dist = std::min(
          static_cast<adr::edit_dist_t>(std::max(a[i].size(), b[i].size())),
          adr::sift4(a[i], b[j], 3,
                     static_cast<adr::edit_dist_t>(
                         std::min(a[i].size(), b[j].size()) / 2U + 2U),
                     sift4_offset_arr));
      if (dist < best) {
        best = dist;
        best_j = j;
      }
    }

    covered |= (1U << best_j);
    score += best;
  }

  for (auto j = 0U; j != b.size(); ++j) {
    if ((covered & (1U << j)) == 0U) {
      score += b[j].size();
    }
  }

  return static_cast<float>(score) /
         static_cast<float>(std::max(str1.length(), str2.length()));
}

adr_ext adr_extend_tt(nigiri::timetable const& tt,
                      a::area_database const* area_db,
                      a::typeahead& t) {
  if (tt.n_locations() == 0) {
    return {};
  }

  auto const timer = utl::scoped_timer{"guesser candidates"};

  auto ret = adr_ext{};

  auto area_set_lookup = [&]() {
    auto x = hash_map<basic_string<a::area_idx_t>, a::area_set_idx_t>{};
    for (auto const [i, area_set] : utl::enumerate(t.area_sets_)) {
      x.emplace(area_set.view(), a::area_set_idx_t{i});
    }
    return x;
  }();

  // mapping: location_idx -> place_idx
  // reverse: place_idx -> location_idx
  auto place_location = n::vecvec<adr_extra_place_idx_t, n::location_idx_t>{};
  auto const add_place = [&](n::location_idx_t const l) {
    auto const i = adr_extra_place_idx_t{place_location.size()};
    ret.location_place_[l] = i;
    place_location.emplace_back({l});
    return i;
  };

  auto const get_transitive_equivalences = [&](n::location_idx_t const l) {
    auto q = std::vector<n::location_idx_t>{};
    auto visited = hash_set<n::location_idx_t>{};

    auto const visit = [&](n::location_idx_t const x) {
      auto const [_, inserted] = visited.insert(x);
      if (!inserted) {
        return;
      }
      for (auto const eq : tt.locations_.equivalences_[x]) {
        if (!visited.contains(eq)) {
          q.push_back(eq);
        }
      }
    };

    visit(l);
    while (!q.empty()) {
      auto const next = q.back();
      q.resize(q.size() - 1);
      visit(next);
    }

    return visited;
  };

  {
    ret.location_place_.resize(tt.n_locations(),
                               adr_extra_place_idx_t::invalid());

    // Map each location + its equivalents with the same name to one
    // place_idx.
    auto sift4_dist = std::vector<adr::sift_offset>{};
    for (auto l = n::location_idx_t{nigiri::kNSpecialStations};
         l != tt.n_locations(); ++l) {
      if (ret.location_place_[l] != adr_extra_place_idx_t::invalid() ||
          tt.locations_.parents_[l] != n::location_idx_t::invalid()) {
        continue;
      }

      auto const place_idx = add_place(l);

      auto const name = tt.get_default_translation(tt.locations_.names_[l]);
      for (auto const eq : get_transitive_equivalences(l)) {
        if (ret.location_place_[eq] != adr_extra_place_idx_t::invalid() ||
            tt.locations_.parents_[eq] != n::location_idx_t::invalid()) {
          continue;
        }

        if (tt.get_default_translation(tt.locations_.names_[eq]) == name) {
          fmt::println(std::clog, "adding to {}: {}  *** name match",
                       n::loc{tt, l}, n::loc{tt, eq});
          ret.location_place_[eq] = place_idx;
        } else {
          auto const dist = geo::distance(tt.locations_.coordinates_[l],
                                          tt.locations_.coordinates_[eq]);
          auto const eq_name =
              tt.get_default_translation(tt.locations_.names_[eq]);
          auto const str_diff =
              get_diff(std::string{name}, std::string{eq_name}, sift4_dist);
          auto const cutoff = (500.F - 1750.F * str_diff);
          auto const good = dist < cutoff;

          if (good) {
            fmt::println(std::clog, "adding to {}: {}  *** fuzzy match",
                         n::loc{tt, l}, n::loc{tt, eq});

            ret.location_place_[eq] = place_idx;

            auto existing = place_location.back();
            if (utl::find_if(existing, [&](n::location_idx_t const x) {
                  return tt.get_default_translation(tt.locations_.names_[x]) ==
                         tt.get_default_translation(tt.locations_.names_[eq]);
                }) == end(existing)) {
              place_location.back().push_back(eq);
            }
          } else {
            fmt::println(std::clog, "NO MATCH {}: {}", n::loc{tt, l},
                         n::loc{tt, eq});
          }
        }
      }
    }

    // Map all children to root.
    for (auto l = n::location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.parents_[l] != n::location_idx_t::invalid()) {
        ret.location_place_[l] =
            ret.location_place_[tt.locations_.get_root_idx(l)];
      }
    }
  }

  for (auto const [i, place] : utl::enumerate(ret.location_place_)) {
    if (place == adr_extra_place_idx_t::invalid()) {
      place = adr_extra_place_idx_t{0U};
    }
  }

  for (auto const [i, p] : utl::enumerate(ret.location_place_)) {
    auto const l = n::location_idx_t{i};
    if (l >= n::kNSpecialStations && p == adr_extra_place_idx_t::invalid()) {
      auto const parent =
          tt.locations_.parents_[l] == n::location_idx_t::invalid()
              ? n::get_special_station(n::special_station::kEnd)
              : tt.locations_.parents_[l];
      auto const grand_parent =
          tt.locations_.parents_[parent] == n::location_idx_t::invalid()
              ? n::get_special_station(n::special_station::kEnd)
              : tt.locations_.parents_[parent];

      utl::log_error(
          "adr_extend", "invalid place for {} (parent={}, grand_parent={})",
          n::loc{tt, l}, n::loc{tt, parent}, n::loc{tt, grand_parent});

      ret.location_place_[l] = adr_extra_place_idx_t{0U};
    }
  }

  // For each station without parent:
  // Compute importance = transport count weighted by clasz.
  ret.place_importance_.resize(place_location.size());
  ret.place_clasz_.resize(place_location.size());
  {
    auto const event_counts = utl::scoped_timer{"guesser event_counts"};
    for (auto i = n::kNSpecialStations; i < tt.n_locations(); ++i) {
      auto const l = n::location_idx_t{i};

      auto transport_counts = std::array<unsigned, n::kNumClasses>{};
      for (auto const& r : tt.location_routes_[l]) {
        auto const clasz =
            static_cast<std::underlying_type_t<n::clasz>>(tt.route_clasz_[r]);
        for (auto const tr : tt.route_transport_ranges_[r]) {
          transport_counts[clasz] +=
              tt.bitfields_[tt.transport_traffic_days_[tr]].count();
        }
      }

      constexpr auto const prio =
          std::array<float, kClaszMax>{/* Air */ 20,
                                       /* HighSpeed */ 30,
                                       /* LongDistance */ 25,
                                       /* Coach */ 22,
                                       /* Night */ 25,
                                       /* RegionalFast */ 20,
                                       /* Regional */ 20,
                                       /* Suburban */ 15,
                                       /* Subway */ 12,
                                       /* Tram */ 3,
                                       /* Bus  */ 2,
                                       /* Ship  */ 10,
                                       /* CableCar */ 5,
                                       /* Funicular */ 5,
                                       /* AerialLift */ 5,
                                       /* Other  */ 1};
      auto const root = tt.locations_.get_root_idx(l);
      auto const place_idx = ret.location_place_[root];

      for (auto const [clasz, t_count] : utl::enumerate(transport_counts)) {
        ret.place_importance_[place_idx] +=
            prio[clasz] * static_cast<float>(t_count);

        auto const c = n::clasz{static_cast<std::uint8_t>(clasz)};
        if (t_count != 0U) {
          ret.place_clasz_[place_idx] |= n::routing::to_mask(c);
        }
      }
    }
  }

  // Update counts of meta-stations with the sum of their priorities.
  // Meta stations have equivalence relations to other stops and are at (0,0)
  for (auto i = n::kNSpecialStations; i < tt.n_locations(); ++i) {
    auto const l = n::location_idx_t{i};
    auto const is_meta =
        tt.locations_.coordinates_[l] == geo::latlng{} &&
        tt.locations_.parents_[l] == n::location_idx_t::invalid() &&
        !tt.locations_.equivalences_[l].empty();
    if (!is_meta) {
      continue;
    }

    auto const place_idx = ret.location_place_[l];
    for (auto const eq : get_transitive_equivalences(l)) {
      auto const eq_root = tt.locations_.get_root_idx(eq);
      auto const eq_place_idx = ret.location_place_[eq_root];
      ret.place_importance_[place_idx] += ret.place_importance_[eq_place_idx];
      ret.place_clasz_[place_idx] |= ret.place_clasz_[eq_place_idx];
    }
  }

  utl::verify(!ret.place_importance_.empty(), "no places");

  // Normalize to interval [0, 1] by dividing by max. importance.
  {
    auto const normalize = utl::scoped_timer{"guesser normalize"};
    auto const max_it = std::max_element(begin(ret.place_importance_),
                                         end(ret.place_importance_));
    auto const max_importance = std::max(*max_it, 1.F);
    for (auto& i : ret.place_importance_) {
      i /= max_importance;
    }
  }

  // Add to typeahead.
  auto const add_string = [&](std::string_view s,
                              a::place_idx_t const place_idx) {
    auto const str_idx = a::string_idx_t{t.strings_.size()};
    t.strings_.emplace_back(s);
    t.string_to_location_.emplace_back(
        std::initializer_list<std::uint32_t>{to_idx(place_idx)});
    t.string_to_type_.emplace_back(
        std::initializer_list<a::location_type_t>{a::location_type_t::kPlace});
    return str_idx;
  };
  auto areas = basic_string<a::area_idx_t>{};
  auto no_areas_idx = adr::area_set_idx_t{t.area_sets_.size()};
  if (area_db == nullptr) {
    t.area_sets_.emplace_back(areas);
  }

  for (auto const [prio, locations] :
       utl::zip(ret.place_importance_, place_location)) {
    auto const place_idx = a::place_idx_t{t.place_names_.size()};

    auto names = std::vector<std::pair<a::string_idx_t, a::language_idx_t>>{};
    auto const add_names = [&](n::location_idx_t const loc) {
      for (auto const [lang, text] :
           n::get_translation_view(tt, tt.locations_.names_[loc])) {
        names.emplace_back(add_string(text, place_idx),
                           t.get_or_create_lang_idx(tt.languages_.get(lang)));
      }
    };

    for (auto const l : locations) {
      add_names(l);
      for (auto const& c : tt.locations_.children_[l]) {
        if (tt.locations_.types_[c] == nigiri::location_type::kStation &&
            tt.get_default_translation(tt.locations_.names_[c]) !=
                tt.get_default_translation(tt.locations_.names_[l])) {
          add_names(c);
        }
      }

      auto const is_null_island = [](geo::latlng const& pos) {
        return pos.lat() < 3.0 && pos.lng() < 3.0;
      };
      auto pos = tt.locations_.coordinates_[l];
      if (is_null_island(pos)) {
        for (auto const c : tt.locations_.children_[l]) {
          if (!is_null_island(tt.locations_.coordinates_[c])) {
            pos = tt.locations_.coordinates_[c];
            break;
          }
        }
      }
    }

    auto const pos =
        a::coordinates::from_latlng(tt.locations_.coordinates_[locations[0]]);

    fmt::println(std::clog, "names: {}, stops={}, prio={}",
                 names | std::views::transform([&](auto&& n) {
                   return t.strings_[n.first].view();
                 }),
                 locations | std::views::transform(
                                 [&](auto&& l) { return n::loc{tt, l}; }),
                 prio);

    t.place_type_.emplace_back(a::amenity_category::kExtra);
    t.place_names_.emplace_back(
        names | std::views::transform([](auto&& n) { return n.first; }));
    t.place_name_lang_.emplace_back(
        names | std::views::transform([](auto&& n) { return n.second; }));
    t.place_coordinates_.emplace_back(pos);
    t.place_osm_ids_.emplace_back(
        locations | std::views::transform([&](auto&& l) { return to_idx(l); }));
    t.place_population_.emplace_back(static_cast<std::uint16_t>(
        (prio * 1'000'000) / a::population::kCompressionFactor));
    t.place_is_way_.resize(t.place_is_way_.size() + 1U);

    if (area_db == nullptr) {
      t.place_areas_.emplace_back(no_areas_idx);
    } else {
      area_db->lookup(t, a::coordinates::from_latlng(pos), areas);
      t.place_areas_.emplace_back(
          utl::get_or_create(area_set_lookup, areas, [&]() {
            auto const set_idx = a::area_set_idx_t{t.area_sets_.size()};
            t.area_sets_.emplace_back(areas);
            return set_idx;
          }));
    }
  }

  t.build_ngram_index();

  return ret;
}

}  // namespace motis

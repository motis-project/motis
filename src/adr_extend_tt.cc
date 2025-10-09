#include "motis/adr_extend_tt.h"

#include <string>
#include <utility>
#include <vector>

#include "utl/get_or_create.h"
#include "utl/parallel_for.h"
#include "utl/timer.h"
#include "utl/to_vec.h"

#include "nigiri/timetable.h"

#include "adr/area_database.h"
#include "adr/typeahead.h"

#include "motis/types.h"

namespace a = adr;
namespace n = nigiri;

namespace motis {

constexpr auto const kClaszMax =
    static_cast<std::underlying_type_t<n::clasz>>(n::kNumClasses);

date::time_zone const* get_tz(n::timetable const& tt,
                              adr_ext const* lp,
                              tz_map_t const* tz,
                              n::location_idx_t const l) {
  auto const p = tt.locations_.parents_[l];
  auto const x = p == n::location_idx_t::invalid() ? l : p;

  auto const p_idx =
      !lp || !tz ? adr_extra_place_idx_t::invalid() : lp->location_place_.at(x);
  if (p_idx != adr_extra_place_idx_t::invalid()) {
    return tz->at(p_idx);
  }

  return nullptr;
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
  auto place_location = vector_map<adr_extra_place_idx_t, n::location_idx_t>{};
  {
    ret.location_place_.resize(tt.n_locations(),
                               adr_extra_place_idx_t::invalid());
    place_location.resize(tt.n_locations(), n::location_idx_t::invalid());

    // Map each location + its equivalents with the same name to one place_idx.
    auto i = adr_extra_place_idx_t{0U};
    for (auto l = n::location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (ret.location_place_[l] == adr_extra_place_idx_t::invalid() &&
          tt.locations_.parents_[l] == n::location_idx_t::invalid()) {
        ret.location_place_[l] = i;
        place_location[i] = l;

        auto const name = tt.locations_.names_[l].view();
        for (auto const eq : tt.locations_.equivalences_[l]) {
          if (tt.locations_.names_[eq].view() == name) {
            ret.location_place_[eq] = i;
          }
        }

        ++i;
      }
    }
    place_location.resize(to_idx(i));

    // Map all children to root.
    for (auto l = n::location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.parents_[l] != n::location_idx_t::invalid()) {
        ret.location_place_[l] =
            ret.location_place_[tt.locations_.get_root_idx(l)];
      }
    }
  }

  // For each station without parent:
  // Compute importance = transport count weighted by clasz.
  ret.place_importance_.resize(place_location.size());
  ret.place_clasz_.resize(place_location.size());
  {
    auto const event_counts = utl::scoped_timer{"guesser event_counts"};
    for (auto i = 0U; i != tt.n_locations(); ++i) {
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
                                       /* HighSpeed */ 20,
                                       /* LongDistance */ 20,
                                       /* Coach */ 20,
                                       /* Night */ 20,
                                       /* RegionalFast */ 16,
                                       /* Regional */ 15,
                                       /* Suburban */ 10,
                                       /* Subway */ 10,
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

        if (t_count != 0U) {
          ret.place_clasz_[place_idx] |=
              static_cast<n::routing::clasz_mask_t>(clasz << 1U);
        }
      }
    }
  }

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
  auto const add_string = [&](auto const& s, a::place_idx_t const place_idx) {
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

  for (auto const [prio, l] : utl::zip(ret.place_importance_, place_location)) {
    auto const place_idx = a::place_idx_t{t.place_names_.size()};

    auto names = std::vector<std::pair<a::string_idx_t, a::language_idx_t>>{
        {add_string(tt.locations_.names_[l].view(), place_idx),
         a::kDefaultLang}};
    auto const add_alt_names = [&](n::location_idx_t const loc) {
      for (auto const& an : tt.locations_.alt_names_[loc]) {
        names.emplace_back(
            add_string(tt.locations_.alt_name_strings_[an].view(), place_idx),
            t.get_or_create_lang_idx(
                tt.languages_[tt.locations_.alt_name_langs_[an]].view()));
      }
    };
    add_alt_names(l);
    for (auto const& c : tt.locations_.children_[l]) {
      if (tt.locations_.types_[c] == nigiri::location_type::kStation &&
          tt.locations_.names_[c].view() != tt.locations_.names_[l].view()) {
        names.emplace_back(
            add_string(tt.locations_.names_[c].view(), place_idx),
            a::kDefaultLang);
        add_alt_names(c);
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

    t.place_type_.emplace_back(a::place_type::kExtra);
    t.place_names_.emplace_back(
        utl::to_vec(names, [](auto const& n) { return n.first; }));
    t.place_coordinates_.emplace_back(a::coordinates::from_latlng(pos));
    t.place_osm_ids_.emplace_back(to_idx(l));
    t.place_name_lang_.emplace_back(
        utl::to_vec(names, [](auto const& n) { return n.second; }));
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

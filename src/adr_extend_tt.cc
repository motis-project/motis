#include "motis/adr_extend_tt.h"

#include "utl/get_or_create.h"
#include "utl/parallel_for.h"
#include "utl/timer.h"

#include "nigiri/timetable.h"

#include "adr/area_database.h"
#include "adr/typeahead.h"

#include "motis/types.h"

namespace a = adr;
namespace n = nigiri;

namespace motis {

constexpr auto const kClaszMax =
    static_cast<std::underlying_type_t<n::clasz>>(n::kNumClasses);

void adr_extend_tt(nigiri::timetable const& tt,
                   a::area_database const& area_db,
                   a::typeahead& t) {
  auto const timer = utl::scoped_timer{"guesser candidates"};

  auto area_set_lookup = [&]() {
    auto x = hash_map<std::basic_string<a::area_idx_t>, a::area_set_idx_t>{};
    for (auto const& [i, area_set] : utl::enumerate(t.area_sets_)) {
      x.emplace(area_set.view(), a::area_set_idx_t{i});
    }
    return x;
  }();

  auto place_location = vector_map<a::place_idx_t, n::location_idx_t>{};
  if (tt.n_locations() == 0) {
    return;
  }

  auto const base_offset = t.place_names_.size();

  // Map each location + its equivalents with the same name to one place_idx
  // mapping: location_idx -> place_idx
  // reverse: place_idx -> location_idx
  auto location_place = n::vector_map<n::location_idx_t, a::place_idx_t>{};
  {
    location_place.resize(tt.n_locations(), a::place_idx_t::invalid());
    place_location.resize(tt.n_locations());

    auto i = a::place_idx_t{0U};
    for (auto l = n::location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (location_place[l] == a::place_idx_t::invalid() &&
          tt.locations_.parents_[l] == n::location_idx_t::invalid()) {
        location_place[l] = i;
        place_location[i] = l;

        auto const name = tt.locations_.names_[l].view();
        for (auto const eq : tt.locations_.equivalences_[l]) {
          if (tt.locations_.names_[eq].view() == name) {
            location_place[eq] = i;
          }
        }

        ++i;
      }
    }

    place_location.resize(to_idx(i));
  }

  // For each station without parent:
  // Compute importance = transport count weighted by clasz.
  auto importance = vector_map<a::place_idx_t, float>{};
  importance.resize(place_location.size());
  {
    auto const event_counts = utl::scoped_timer{"guesser event_counts"};
    for (auto i = 0U; i != tt.n_locations(); ++i) {
      auto const l = n::location_idx_t{i};

      auto transport_counts = std::array<unsigned, n::kNumClasses>{};
      for (auto const& r : tt.location_routes_[l]) {
        for (auto const& tr : tt.route_transport_ranges_[r]) {
          auto const clasz = static_cast<std::underlying_type_t<n::clasz>>(
              tt.route_section_clasz_[r][0]);
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
                                       /* Metro */ 10,
                                       /* Subway */ 10,
                                       /* Tram */ 3,
                                       /* Bus  */ 2,
                                       /* Ship  */ 10,
                                       /* Other  */ 1};
      auto const p = tt.locations_.parents_[l];
      auto const x = (p == n::location_idx_t::invalid()) ? l : p;
      for (auto const [clasz, t_count] : utl::enumerate(transport_counts)) {
        importance[location_place[x]] += prio[clasz] * t_count;
      }
    }
  }

  // Normalize to interval [0, 1] by dividing by max. importance.
  {
    auto const normalize = utl::scoped_timer{"guesser normalize"};
    auto const max_it = std::max_element(begin(importance), end(importance));
    auto const max_importance = std::max(*max_it, 1.F);
    for (auto& i : importance) {
      i /= max_importance;
    }
  }

  // Add to typeahead.
  auto areas = std::basic_string<a::area_idx_t>{};
  for (auto const& [prio, l] : utl::zip(importance, place_location)) {
    auto const str_idx = a::string_idx_t{t.strings_.size()};
    auto const place_idx = a::place_idx_t{t.place_names_.size()};
    t.place_type_.emplace_back(a::place_type::kExtra);
    t.strings_.emplace_back(tt.locations_.names_[l].view());
    t.place_names_.emplace_back(
        std::initializer_list<a::string_idx_t>{str_idx});
    t.place_coordinates_.emplace_back(
        a::coordinates::from_latlng(tt.locations_.coordinates_[l]));
    t.place_osm_ids_.emplace_back(to_idx(l));
    t.place_name_lang_.emplace_back(
        std::initializer_list<a::language_idx_t>{a::kDefaultLang});
    t.place_population_.emplace_back(static_cast<std::uint16_t>(
        (prio * 1'000'000) / a::population::kCompressionFactor));
    t.place_is_way_.resize(t.place_is_way_.size() + 1U);
    t.string_to_location_.emplace_back(
        std::initializer_list<std::uint32_t>{to_idx(place_idx)});
    t.string_to_type_.emplace_back(
        std::initializer_list<a::location_type_t>{a::location_type_t::kPlace});

    area_db.lookup(
        t, a::coordinates::from_latlng(tt.locations_.coordinates_[l]), areas);
    t.place_areas_.emplace_back(
        utl::get_or_create(area_set_lookup, areas, [&]() {
          auto const set_idx = a::area_set_idx_t{t.area_sets_.size()};
          t.area_sets_.emplace_back(areas);
          return set_idx;
        }));
  }

  t.build_ngram_index();
}

}  // namespace motis
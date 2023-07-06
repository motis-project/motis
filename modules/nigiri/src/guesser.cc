#include "motis/nigiri/guesser.h"

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "guess/guesser.h"

#include "nigiri/timetable.h"

#include "motis/core/common/logging.h"
#include "motis/nigiri/location.h"

namespace mm = motis::module;
namespace n = nigiri;

namespace {

constexpr auto const kClaszMax =
    static_cast<std::underlying_type_t<n::clasz>>(n::kNumClasses);

std::vector<std::pair<std::string, float>> get_candidates(
    n::timetable const& tt, std::vector<::nigiri::location_idx_t>& mapping) {
  auto const timer = motis::logging::scoped_timer{"guesser candidates"};

  auto candidates = std::vector<std::pair<std::string, float>>{};
  if (tt.n_locations() == 0) {
    return candidates;
  }

  // Mapping: candidate index -> location_idx_t
  // Reverse mapping: location_idx_t -> candidate index
  // candidates[reverse_mapping[location_idx_t]]
  auto reverse_mapping = n::vector_map<n::location_idx_t, int>{};
  {
    reverse_mapping.resize(tt.n_locations(), -1);
    mapping.resize(tt.n_locations());

    auto i = 0U;
    for (auto l = n::location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (reverse_mapping[l] == -1 &&
          tt.locations_.parents_[l] == n::location_idx_t::invalid()) {
        reverse_mapping[l] = i;
        mapping[i] = l;

        auto const name = tt.locations_.names_[l].view();
        for (auto const eq : tt.locations_.equivalences_[l]) {
          if (tt.locations_.names_[eq].view() == name) {
            reverse_mapping[eq] = i;
          }
        }

        ++i;
      }
    }
    mapping.resize(i);
    candidates.resize(i, std::pair<std::string, float>{"", 0.F});
  }

  // For each station without parent:
  // Compute importance = transport count weighted by clasz.
  {
    auto const event_counts =
        motis::logging::scoped_timer{"guesser event_counts"};
    for (auto i = 0U; i != tt.n_locations(); ++i) {
      auto const l = n::location_idx_t{i};

      auto transport_counts = std::array<unsigned, n::kNumClasses>{};
      for (auto const& r : tt.location_routes_[l]) {
        for (auto const& t : tt.route_transport_ranges_[r]) {
          auto const clasz = static_cast<std::underlying_type_t<n::clasz>>(
              tt.route_section_clasz_[r][0]);
          transport_counts[clasz] +=
              tt.bitfields_[tt.transport_traffic_days_[t]].count();
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
      auto importance = 0.0F;
      for (auto const [clasz, t_count] : utl::enumerate(transport_counts)) {
        importance += prio[clasz] * t_count;
      }
      candidates[reverse_mapping[x]].second += importance;
    }
  }

  {
    auto const normalize = motis::logging::scoped_timer{"guesser names"};
    for (auto i = 0U; i != candidates.size(); ++i) {
      candidates[i].first = tt.locations_.names_[mapping[i]].view();
    }
  }

  // Normalize to interval [0, 1] by dividing by max. importance.
  {
    auto const normalize = motis::logging::scoped_timer{"guesser normalize"};
    auto const max_it = std::max_element(
        begin(candidates), end(candidates),
        [](auto&& a, auto&& b) { return a.second < b.second; });
    auto const max_importance = std::max(max_it->second, 1.F);
    for (auto& [name, factor] : candidates) {
      factor = 1 + 2 * (factor / max_importance);
    }
  }

  return candidates;
}

std::string trim(std::string const& s) {
  auto first = s.find_first_not_of(' ');
  auto last = s.find_last_not_of(' ');
  if (first == last) {
    return "";
  } else {
    return s.substr(first, (last - first + 1));
  }
}

}  // namespace

namespace motis::nigiri {

guesser::guesser(tag_lookup const& tags, n::timetable const& tt)
    : tt_{tt}, tags_{tags} {
  guess_ = std::make_unique<guess::guesser>(get_candidates(tt, mapping_));
}

guesser::~guesser() = default;

mm::msg_ptr guesser::guess(mm::msg_ptr const& msg) {
  using ::motis::guesser::StationGuesserRequest;
  auto const req = motis_content(StationGuesserRequest, msg);
  mm::message_creator fbb;
  fbb.create_and_finish(
      MsgContent_StationGuesserResponse,
      ::motis::guesser::CreateStationGuesserResponse(
          fbb, fbb.CreateVector(utl::to_vec(
                   guess_->guess_match(trim(req->input()->str()),
                                       req->guess_count()),
                   [&](guess::guesser::match const& m) {
                     auto const p = mapping_[m.index];
                     auto const name = tt_.locations_.names_.at(p).view();
                     auto const id = get_station_id(tags_, tt_, p);
                     auto const coord = tt_.locations_.coordinates_[p];
                     auto const pos = Position(coord.lat_, coord.lng_);
                     return CreateStation(fbb, fbb.CreateString(id),
                                          fbb.CreateString(name), &pos);
                   })))
          .Union());
  return mm::make_msg(fbb);
}

}  // namespace motis::nigiri
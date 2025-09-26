#include "motis/odm/journeys.h"

#include <charconv>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/routing/pareto_set.h"

#include "motis/odm/odm.h"
#include "motis/transport_mode_ids.h"

namespace motis::odm {

struct csv_journey {
  utl::csv_col<utl::cstr, UTL_NAME("departure")> departure_time_;
  utl::csv_col<utl::cstr, UTL_NAME("arrival")> arrival_time_;
  utl::csv_col<std::uint8_t, UTL_NAME("transfers")> transfers_;
  utl::csv_col<utl::cstr, UTL_NAME("first_mile_mode")> first_mile_mode_;
  utl::csv_col<nigiri::duration_t::rep, UTL_NAME("first_mile_duration")>
      first_mile_duration_;
  utl::csv_col<utl::cstr, UTL_NAME("last_mile_mode")> last_mile_mode_;
  utl::csv_col<nigiri::duration_t::rep, UTL_NAME("last_mile_duration")>
      last_mile_duration_;
};

std::optional<nigiri::transport_mode_id_t> read_transport_mode(
    std::string_view m) {
  if (m == "taxi") {
    return kOdmTransportModeId;
  } else if (m == "walk") {
    return kWalkTransportModeId;
  } else {
    return std::nullopt;
  }
}

nigiri::routing::journey make_dummy(
    nigiri::unixtime_t const departure,
    nigiri::unixtime_t const arrival,
    std::uint8_t const transfers,
    nigiri::transport_mode_id_t const first_mile_mode,
    nigiri::duration_t const first_mile_duration,
    nigiri::transport_mode_id_t const last_mile_mode,
    nigiri::duration_t const last_mile_duration) {
  return nigiri::routing::journey{
      .legs_ = {{nigiri::direction::kForward, nigiri::location_idx_t::invalid(),
                 nigiri::location_idx_t::invalid(), departure,
                 departure + first_mile_duration,
                 nigiri::routing::offset{nigiri::location_idx_t::invalid(),
                                         first_mile_duration, first_mile_mode}},
                {nigiri::direction::kForward, nigiri::location_idx_t::invalid(),
                 nigiri::location_idx_t::invalid(),
                 arrival - last_mile_duration, arrival,
                 nigiri::routing::offset{nigiri::location_idx_t::invalid(),
                                         last_mile_duration, last_mile_mode}}},
      .start_time_ = departure,
      .dest_time_ = arrival,
      .transfers_ = transfers};
}

std::vector<nigiri::routing::journey> from_csv(std::string_view const csv) {
  auto journeys = std::vector<nigiri::routing::journey>{};
  utl::line_range{utl::make_buf_reader(csv)} | utl::csv<csv_journey>() |
      utl::for_each([&](csv_journey const& cj) {
        try {
          auto const departure =
              nigiri::parse_time(cj.departure_time_->trim().view(), "%F %R");

          auto const arrival =
              nigiri::parse_time(cj.arrival_time_->trim().view(), "%F %R");

          auto const first_mile_duration =
              nigiri::duration_t{*cj.first_mile_duration_};
          auto const first_mile_mode =
              read_transport_mode(cj.first_mile_mode_->trim().view());
          if (!first_mile_mode) {
            fmt::println("Invalid first-mile transport mode: {}",
                         cj.first_mile_mode_->view());
            return;
          }

          auto const last_mile_duration =
              nigiri::duration_t{*cj.last_mile_duration_};
          auto const last_mile_mode =
              read_transport_mode(cj.last_mile_mode_->trim().view());
          if (!last_mile_mode) {
            fmt::println("Invalid last-mile transport mode: {}",
                         cj.last_mile_mode_->view());
            return;
          }

          journeys.push_back(make_dummy(departure, arrival, *cj.transfers_,
                                        *first_mile_mode, first_mile_duration,
                                        *last_mile_mode, last_mile_duration));

        } catch (std::exception const& e) {
          fmt::println("could not parse csv_journey: {}", e.what());
        }
      });

  return journeys;
}

nigiri::pareto_set<nigiri::routing::journey> separate_pt(
    std::vector<nigiri::routing::journey>& journeys) {
  auto pt_journeys = nigiri::pareto_set<nigiri::routing::journey>{};
  for (auto j = begin(journeys); j != end(journeys);) {
    if (is_pure_pt(*j)) {
      pt_journeys.add(std::move(*j));
      j = journeys.erase(j);
    } else {
      ++j;
    }
  }
  return pt_journeys;
}

std::string to_csv(nigiri::routing::journey const& j) {
  auto const mode_str = [&](nigiri::transport_mode_id_t const mode) {
    return mode == kOdmTransportModeId ? "taxi" : "walk";
  };

  auto const first_mile_mode =
      !j.legs_.empty() && std::holds_alternative<nigiri::routing::offset>(
                              j.legs_.front().uses_)
          ? mode_str(std::get<nigiri::routing::offset>(j.legs_.front().uses_)
                         .transport_mode_id_)
          : "walk";

  auto const first_mile_duration =
      !j.legs_.empty() && std::holds_alternative<nigiri::routing::offset>(
                              j.legs_.front().uses_)
          ? std::get<nigiri::routing::offset>(j.legs_.front().uses_)
                .duration()
                .count()
          : nigiri::duration_t::rep{0};

  auto const last_mile_mode =
      j.legs_.size() > 1 && std::holds_alternative<nigiri::routing::offset>(
                                j.legs_.back().uses_)
          ? mode_str(std::get<nigiri::routing::offset>(j.legs_.back().uses_)
                         .transport_mode_id_)
          : "walk";

  auto const last_mile_duration =
      j.legs_.size() > 1 && std::holds_alternative<nigiri::routing::offset>(
                                j.legs_.back().uses_)
          ? std::get<nigiri::routing::offset>(j.legs_.back().uses_)
                .duration()
                .count()
          : nigiri::duration_t::rep{0};

  return fmt::format("{}, {}, {}, {}, {:0>2}, {}, {:0>2}", j.start_time_,
                     j.dest_time_, j.transfers_, first_mile_mode,
                     first_mile_duration, last_mile_mode, last_mile_duration);
}

std::string to_csv(std::vector<nigiri::routing::journey> const& jv) {
  auto ss = std::stringstream{};
  ss << "departure, arrival, transfers, first_mile_mode, "
        "first_mile_duration, last_mile_mode, last_mile_duration\n";

  for (auto const& j : jv) {
    ss << to_csv(j) << "\n";
  }

  return ss.str();
}

nigiri::routing::journey make_odm_direct(nigiri::location_idx_t const from,
                                         nigiri::location_idx_t const to,
                                         nigiri::unixtime_t const departure,
                                         nigiri::unixtime_t const arrival) {
  return nigiri::routing::journey{
      .legs_ = {{nigiri::direction::kForward, from, to, departure, arrival,
                 nigiri::routing::offset{to,
                                         std::chrono::abs(arrival - departure),
                                         kOdmTransportModeId}}},
      .start_time_ = departure,
      .dest_time_ = arrival,
      .dest_ = to,
      .transfers_ = 0U};
}

}  // namespace motis::odm
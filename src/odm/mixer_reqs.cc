#include "motis/odm/mixer_reqs.h"

#include <charconv>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"

#include "nigiri/common/parse_time.h"

#include "motis/odm/odm.h"
#include "motis/transport_mode_ids.h"

namespace motis::odm {

struct csv_journey {
  utl::csv_col<utl::cstr, UTL_NAME("departure_time")> departure_time_;
  utl::csv_col<utl::cstr, UTL_NAME("arrival_time")> arrival_time_;
  utl::csv_col<std::uint8_t, UTL_NAME("transfers")> transfers_;
  utl::csv_col<utl::cstr, UTL_NAME("first_mile_mode")> first_mile_mode_;
  utl::csv_col<nigiri::duration_t::rep, UTL_NAME("first_mile_duration")>
      first_mile_duration_;
  utl::csv_col<utl::cstr, UTL_NAME("last_mile_mode")> last_mile_mode_;
  utl::csv_col<nigiri::duration_t::rep, UTL_NAME("last_mile_duration")>
      last_mile_duration_;
};

template <typename T>
std::optional<T> read_number(const char* first, const char* last) {
  T n;
  auto [ptr, ec] = std::from_chars(first, last, n);
  if (ec == std::errc::invalid_argument ||
      ec == std::errc::result_out_of_range) {
    return std::nullopt;
  }
  return n;
}

std::optional<nigiri::unixtime_t> read_hours_minutes(std::string_view t) {
  auto const colon_idx = t.find(':');
  if (colon_idx == std::string_view::npos) {
    return std::nullopt;
  }

  auto hours = read_number<unsigned>(t.data(), t.data() + colon_idx);
  auto minutes =
      read_number<unsigned>(t.data() + colon_idx + 1, t.data() + t.size());
  if (!hours || !minutes) {
    return std::nullopt;
  }

  return nigiri::unixtime_t{std::chrono::hours{*hours} +
                            std::chrono::minutes{*minutes}};
}

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

std::vector<nigiri::routing::journey> read(std::string_view csv) {
  auto journeys = std::vector<nigiri::routing::journey>{};
  utl::line_range{utl::make_buf_reader(csv)} | utl::csv<csv_journey>() |
      utl::for_each([&](csv_journey& cj) {
        try {
          auto const departure_time =
              read_hours_minutes(cj.departure_time_->trim().view());
          if (!departure_time) {
            fmt::println("Invalid departure time: {}",
                         cj.departure_time_->view());
            return;
          }

          auto const arrival_time =
              read_hours_minutes(cj.arrival_time_->trim().view());
          if (!arrival_time) {
            fmt::println("Invalid arrival time: {}", cj.arrival_time_->view());
            return;
          }

          auto j = nigiri::routing::journey{.start_time_ = *departure_time,
                                            .dest_time_ = *arrival_time,
                                            .transfers_ = *cj.transfers_};

          auto const first_mile_duration =
              nigiri::duration_t{*cj.first_mile_duration_};
          auto const first_mile_mode =
              read_transport_mode(cj.first_mile_mode_->trim().view());
          if (!first_mile_mode) {
            fmt::println("Invalid first-mile transport mode: {}",
                         cj.first_mile_mode_->view());
            return;
          }
          j.legs_.emplace_back(
              nigiri::direction::kForward, nigiri::location_idx_t::invalid(),
              nigiri::location_idx_t::invalid(), j.start_time_,
              j.start_time_ + first_mile_duration,
              nigiri::routing::offset{nigiri::location_idx_t::invalid(),
                                      first_mile_duration, *first_mile_mode});

          auto const last_mile_duration =
              nigiri::duration_t{*cj.last_mile_duration_};
          auto const last_mile_mode =
              read_transport_mode(cj.last_mile_mode_->trim().view());
          if (!last_mile_mode) {
            fmt::println("Invalid last-mile transport mode: {}",
                         cj.last_mile_mode_->view());
            return;
          }
          j.legs_.emplace_back(
              nigiri::direction::kForward, nigiri::location_idx_t::invalid(),
              nigiri::location_idx_t::invalid(),
              j.dest_time_ - last_mile_duration, j.dest_time_,
              nigiri::routing::offset{nigiri::location_idx_t::invalid(),
                                      last_mile_duration, *last_mile_mode});

          journeys.push_back(std::move(j));

        } catch (std::exception const& e) {
          fmt::println("could not parse csv_journey: {}", e.what());
        }
      });

  return journeys;
}

std::string to_csv(std::vector<nigiri::routing::journey> const& jv) {
  auto ss = std::stringstream{};
  ss << "departure_time, arrival_time, transfers, first_mile_mode, "
        "first_mile_duration, last_mile_mode, last_mile_duration\n";

  for (auto const& j : jv) {
    auto const time_str = [&](nigiri::unixtime_t t) {
      auto const [hours, minutes] = std::div(t.time_since_epoch().count(), 60);
      return fmt::format("{}:{}", hours, minutes);
    };

    auto const mode_str = [&](nigiri::transport_mode_id_t const mode) {
      return mode == kOdmTransportModeId ? "taxi" : "walk";
    };

    ss << fmt::format(
        "{}, {}, {}, {}, {}, {}, {}\n", time_str(j.start_time_),
        time_str(j.dest_time_), j.transfers_,
        mode_str(std::get<nigiri::routing::offset>(j.legs_.front().uses_)
                     .transport_mode_id_),
        std::get<nigiri::routing::offset>(j.legs_.front().uses_)
            .duration()
            .count(),
        mode_str(std::get<nigiri::routing::offset>(j.legs_.back().uses_)
                     .transport_mode_id_),
        std::get<nigiri::routing::offset>(j.legs_.back().uses_)
            .duration()
            .count());
  }

  return ss.str();
}

}  // namespace motis::odm
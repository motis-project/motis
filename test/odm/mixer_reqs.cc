#include "./mixer_reqs.h"

#include "utl/parser/csv_range.h"

namespace motis {

struct csv_journey {
  utl::csv_col<utl::cstr, UTL_NAME("departure_time")> departure_time_;
  utl::csv_col<utl::cstr, UTL_NAME("arrival_time")> arrival_time_;
  utl::csv_col<std::uint8_t, UTL_NAME("transfers")> transfers_;
  utl::csv_col<utl::cstr, UTL_NAME("first_mile_mode")> first_mile_mode_;
  utl::csv_col<std::uint16_t, UTL_NAME("first_mile_duration")>
      first_mile_duration_;
  utl::csv_col<utl::cstr, UTL_NAME("last_mile_mode")> last_mile_mode_;
  utl::csv_col<utl::cstr, UTL_NAME("last_mile_duration")> last_mile_duration_;
};

std::vector<nigiri::routing::journey> read(std::string_view csv) {
  auto journeys = std::vector<nigiri::routing::journey>{};
  utl::line_range{utl::make_buf_reader(csv)} | utl::csv<csv_journey>{} |
      utl::for_each([&](csv_journey const& cj) {

      });

  return journeys;
}

std::string write(std::vector<nigiri::routing::journey> const& jv) {

  auto ss = std::stringstream{};

  return ss.str();
}

}  // namespace motis
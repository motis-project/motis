#include "./mixer_reqs.h"

#include "utl/parser/csv_range.h"

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

nigiri::transport_mode_id_t read_transport_mode(utl::cstr const& m) {
  if (m == "taxi") {
    return kOdmTransportModeId;
  } else {
    return kWalkTransportModeId;
  }
}

std::vector<nigiri::routing::journey> read(std::string_view csv) {
  auto journeys = std::vector<nigiri::routing::journey>{};
  utl::line_range{utl::make_buf_reader(csv)} | utl::csv<csv_journey>{} |
      utl::for_each([&](csv_journey const& cj) {
        try {
          auto j = nigiri::routing::journey{
              .start_time_ =
                  nigiri::parse_time(cj.departure_time_->view(), "%H:%M"),
              .dest_time_ =
                  nigiri::parse_time(cj.arrival_time_->view(), "%H:%M"),
              .transfers_ = *cj.transfers_};

          auto const first_mile_duration =
              nigiri::duration_t{*cj.first_mile_duration_};
          if (first_mile_duration > nigiri::duration_t{0}) {
            j.legs_.emplace_back(
                nigiri::direction::kForward, nigiri::location_idx_t::invalid(),
                nigiri::location_idx_t::invalid(), j.start_time_,
                j.start_time_ + first_mile_duration,
                nigiri::routing::offset{
                    nigiri::location_idx_t::invalid(), first_mile_duration,
                    read_transport_mode(*cj.first_mile_mode_)});
          }

          auto const last_mile_duration =
              nigiri::duration_t{*cj.last_mile_duration_};
          if (last_mile_duration > nigiri::duration_t{0}) {
            j.legs_.emplace_back(
                nigiri::direction::kForward, nigiri::location_idx_t::invalid(),
                nigiri::location_idx_t::invalid(),
                j.dest_time_ - last_mile_duration, j.dest_time_,
                nigiri::routing::offset{
                    nigiri::location_idx_t::invalid(), last_mile_duration,
                    read_transport_mode(*cj.last_mile_mode_)});
          }

          journeys.push_back(std::move(j));

        } catch (std::exception const& e) {
          std::println("could not parse csv_journey: {}", e.what());
        }
      });

  return journeys;
}

std::string write(std::vector<nigiri::routing::journey> const& jv) {

  auto ss = std::stringstream{};

  return ss.str();
}

}  // namespace motis::odm
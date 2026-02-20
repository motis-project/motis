#include "motis/odm/prima.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/json.hpp"

#include "utl/erase_duplicates.h"

#include "motis/http_req.h"
#include "motis/odm/odm.h"
#include "motis/transport_mode_ids.h"

namespace n = nigiri;
namespace nr = nigiri::routing;
namespace json = boost::json;
using namespace std::chrono_literals;

namespace motis::odm {

std::string prima::make_whitelist_taxi_request(
    std::vector<nr::start> const& first_mile,
    std::vector<nr::start> const& last_mile,
    n::timetable const& tt) const {
  return make_whitelist_request(from_, to_, first_mile, last_mile, direct_taxi_,
                                fixed_, cap_, tt);
}

void extract_taxis(std::vector<nr::journey> const& journeys,
                   std::vector<nr::start>& first_mile_taxi_rides,
                   std::vector<nr::start>& last_mile_taxi_rides) {
  for (auto const& j : journeys) {
    if (!j.legs_.empty()) {
      if (is_odm_leg(j.legs_.front(), kOdmTransportModeId)) {
        first_mile_taxi_rides.push_back({
            .time_at_start_ = j.legs_.front().dep_time_,
            .time_at_stop_ = j.legs_.front().arr_time_,
            .stop_ = j.legs_.front().to_,
        });
      }
    }

    if (j.legs_.size() > 1) {
      if (is_odm_leg(j.legs_.back(), kOdmTransportModeId)) {
        last_mile_taxi_rides.push_back({
            .time_at_start_ = j.legs_.back().arr_time_,
            .time_at_stop_ = j.legs_.back().dep_time_,
            .stop_ = j.legs_.back().from_,
        });
      }
    }
  }

  utl::erase_duplicates(first_mile_taxi_rides, by_stop, std::equal_to<>{});
  utl::erase_duplicates(last_mile_taxi_rides, by_stop, std::equal_to<>{});
}

void prima::extract_taxis_for_persisting(
    std::vector<nr::journey> const& journeys) {
  whitelist_first_mile_locations_.clear();
  whitelist_last_mile_locations_.clear();

  for (auto const& j : journeys) {
    if (j.legs_.size() <= 1) {
      continue;
    }

    if (is_odm_leg(j.legs_.front(), kOdmTransportModeId)) {
      whitelist_first_mile_locations_.push_back(j.legs_.front().to_);
    }

    if (is_odm_leg(j.legs_.back(), kOdmTransportModeId)) {
      whitelist_last_mile_locations_.push_back(j.legs_.back().from_);
    }
  }

  utl::erase_duplicates(whitelist_first_mile_locations_, std::less<>{},
                        std::equal_to<>{});
  utl::erase_duplicates(whitelist_last_mile_locations_, std::less<>{},
                        std::equal_to<>{});
}

bool prima::consume_whitelist_taxi_response(
    std::string_view json,
    std::vector<nr::journey>& journeys,
    std::vector<nr::start>& first_mile_taxi_rides,
    std::vector<nr::start>& last_mile_taxi_rides) {
  auto const update_first_mile = [&](json::array const& update) {
    auto const n_pt_udpates = n_rides_in_response(update);
    if (first_mile_taxi_rides.size() != n_pt_udpates) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] first mile taxi #rides != #updates ({} != {})",
             first_mile_taxi_rides.size(), n_pt_udpates);
      return true;
    }

    auto const prev_first_mile =
        std::exchange(first_mile_taxi_rides, std::vector<nr::start>{});

    auto prev_it = std::begin(prev_first_mile);
    for (auto const& stop : update) {
      for (auto const& event : stop.as_array()) {
        if (event.is_null()) {
          first_mile_taxi_rides.push_back({.time_at_start_ = kInfeasible,
                                           .time_at_stop_ = kInfeasible,
                                           .stop_ = prev_it->stop_});
        } else {
          first_mile_taxi_rides.push_back({
              .time_at_start_ =
                  to_unix(event.as_object().at("pickupTime").as_int64()),
              .time_at_stop_ =
                  to_unix(event.as_object().at("dropoffTime").as_int64()),
              .stop_ = prev_it->stop_,
          });
        }
        ++prev_it;
      }
    }
    fix_first_mile_duration(journeys, first_mile_taxi_rides, prev_first_mile,
                            kOdmTransportModeId);
    return false;
  };

  auto const update_last_mile = [&](json::array const& update) {
    auto const n_pt_udpates = n_rides_in_response(update);
    if (last_mile_taxi_rides.size() != n_pt_udpates) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] last mile taxi #rides != #updates ({} != {})",
             last_mile_taxi_rides.size(), n_pt_udpates);
      return true;
    }

    auto const prev_last_mile =
        std::exchange(last_mile_taxi_rides, std::vector<nr::start>{});

    auto prev_it = std::begin(prev_last_mile);
    for (auto const& stop : update) {
      for (auto const& event : stop.as_array()) {
        if (event.is_null()) {
          last_mile_taxi_rides.push_back({
              .time_at_start_ = kInfeasible,
              .time_at_stop_ = kInfeasible,
              .stop_ = prev_it->stop_,
          });
        } else {
          last_mile_taxi_rides.push_back({
              .time_at_start_ =
                  to_unix(event.as_object().at("dropoffTime").as_int64()),
              .time_at_stop_ =
                  to_unix(event.as_object().at("pickupTime").as_int64()),
              .stop_ = prev_it->stop_,
          });
        }
        ++prev_it;
      }
    }

    fix_last_mile_duration(journeys, last_mile_taxi_rides, prev_last_mile,
                           kOdmTransportModeId);
    return false;
  };

  auto const update_direct_rides = [&](json::array const& update) {
    if (direct_taxi_.size() != update.size()) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] direct taxi #rides != #updates ({} != {})",
             direct_taxi_.size(), update.size());
      direct_taxi_.clear();
      return true;
    }

    direct_taxi_.clear();
    for (auto const& ride : update) {
      if (!ride.is_null()) {
        direct_taxi_.push_back(
            {to_unix(ride.as_object().at("pickupTime").as_int64()),
             to_unix(ride.as_object().at("dropoffTime").as_int64())});
      }
    }

    return false;
  };

  auto with_errors = false;
  try {
    auto const o = json::parse(json).as_object();
    with_errors |= update_first_mile(o.at("start").as_array());
    with_errors |= update_last_mile(o.at("target").as_array());
    with_errors |= update_direct_rides(o.at("direct").as_array());
  } catch (std::exception const&) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] could not parse response: {}", json);
    return false;
  }
  if (with_errors) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] parsed response with errors: {}", json);
    return false;
  }

  // adjust journey start/dest times after adjusting legs
  for (auto& j : journeys) {
    if (!j.legs_.empty()) {
      j.start_time_ = j.legs_.front().dep_time_;
      j.dest_time_ = j.legs_.back().arr_time_;
    }
  }

  return true;
}

bool prima::whitelist_taxi(std::vector<nr::journey>& taxi_journeys,
                           n::timetable const& tt) {
  auto first_mile_taxi_rides = std::vector<nr::start>{};
  auto last_mile_taxi_rides = std::vector<nr::start>{};
  extract_taxis(taxi_journeys, first_mile_taxi_rides, last_mile_taxi_rides);
  extract_taxis_for_persisting(taxi_journeys);

  auto whitelist_response = std::optional<std::string>{};
  auto ioc = boost::asio::io_context{};
  try {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] request for {} rides",
           first_mile_taxi_rides.size() + last_mile_taxi_rides.size() +
               direct_taxi_.size());

    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const prima_msg = co_await http_POST(
              taxi_whitelist_, kReqHeaders,
              make_whitelist_request(from_, to_, first_mile_taxi_rides,
                                     last_mile_taxi_rides, direct_taxi_, fixed_,
                                     cap_, tt),
              10s);
          whitelist_response = get_http_body(prima_msg);
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] networking failed: {}", e.what());
    whitelist_response = std::nullopt;
  }
  if (!whitelist_response) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] failed, discarding taxi journeys");
    return false;
  }
  auto const was_whitelist_response_valid = consume_whitelist_taxi_response(
      *whitelist_response, taxi_journeys, first_mile_taxi_rides,
      last_mile_taxi_rides);
  if (!was_whitelist_response_valid) {
    return false;
  }
  whitelist_response_ = json::parse(whitelist_response.value()).as_object();
  return true;
}

}  // namespace motis::odm
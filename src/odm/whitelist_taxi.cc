#include "motis/odm/prima.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/json.hpp"

#include "utl/erase_duplicates.h"

#include "motis/http_req.h"
#include "motis/odm/fix_duration.h"
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

std::tuple<std::vector<nr::start>, std::vector<nr::start>> extract_taxis(
    std::vector<nr::journey> const& journeys) {
  auto first_mile_taxi_rides = std::vector<nr::start>{};
  auto last_mile_taxi_rides = std::vector<nr::start>{};

  for (auto const& j : journeys) {
    if (!j.legs_.empty()) {
      if (is_odm_leg(j.legs_.front(), kOdmTransportModeId)) {
        first_mile_taxi_rides.push_back(
            {.time_at_start_ = j.legs_.front().dep_time_,
             .time_at_stop_ = j.legs_.front().arr_time_,
             .stop_ = j.legs_.front().to_});
      }
    }
    if (j.legs_.size() > 1) {
      if (is_odm_leg(j.legs_.back(), kOdmTransportModeId)) {
        last_mile_taxi_rides.push_back(
            {.time_at_start_ = j.legs_.back().arr_time_,
             .time_at_stop_ = j.legs_.back().dep_time_,
             .stop_ = j.legs_.back().from_});
      }
    }
  }

  utl::erase_duplicates(first_mile_taxi_rides, by_stop, std::equal_to{});
  utl::erase_duplicates(last_mile_taxi_rides, by_stop, std::equal_to{});

  return std::tuple{first_mile_taxi_rides, last_mile_taxi_rides};
}

bool prima::consume_whitelist_taxi_response(
    std::string_view json,
    std::vector<nr::journey>& journeys,
    std::vector<nr::start> const& first_mile_in,
    std::vector<nr::start> const& last_mile_in) {
  auto const read_prima_data = [](auto const& o) {
    return prima_data{
        .passenger_delta_ = o.at("passengerDuration").as_int64(),
        .approach_return_delta_ =
            o.at("approachPlusReturnDurationDelta").as_int64(),
        .fully_paid_delta_ = o.at("fullyPayedDurationDelta").as_int64(),
        .waiting_time_delta_ = o.at("taxiWaitingTime").as_int64(),
        .cost_ = value_to<double>(o.at("cost"))};
  };

  auto const update_first_mile = [&](json::array const& update) {
    auto const n_pt_udpates = n_rides_in_response(update);
    if (first_mile_in.size() != n_pt_udpates) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] first mile taxi #rides != #updates ({} != {})",
             first_mile_in.size(), n_pt_udpates);
      return true;
    }

    auto i = std::begin(first_mile_in);
    for (auto const& stop : update) {
      for (auto const& event : stop.as_array()) {
        if (event.is_null()) {
          first_mile_taxi_.push_back(
              {{kInfeasible, kInfeasible, i->stop_}, {}});
        } else {
          auto const o = event.as_object();
          first_mile_taxi_.push_back(
              {{.time_at_start_ = to_unix(o.at("pickupTime").as_int64()),
                .time_at_stop_ = to_unix(o.at("dropoffTime").as_int64()),
                .stop_ = i->stop_},
               read_prima_data(o)});
        }
        ++i;
      }
    }

    fix_first_mile_duration<ride>(journeys, first_mile_taxi_, first_mile_in,
                                  kOdmTransportModeId);

    return false;
  };

  auto const update_last_mile = [&](json::array const& update) {
    auto const n_pt_udpates = n_rides_in_response(update);
    if (last_mile_in.size() != n_pt_udpates) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] last mile taxi #rides != #updates ({} != {})",
             last_mile_in.size(), n_pt_udpates);
      return true;
    }

    auto i = std::begin(last_mile_in);
    for (auto const& stop : update) {
      for (auto const& event : stop.as_array()) {
        if (event.is_null()) {
          last_mile_taxi_.push_back({{kInfeasible, kInfeasible, i->stop_}, {}});
        } else {
          auto const o = event.as_object();
          last_mile_taxi_.push_back(
              {{.time_at_start_ = to_unix(o.at("dropoffTime").as_int64()),
                .time_at_stop_ = to_unix(o.at("pickupTime").as_int64()),
                .stop_ = i->stop_},
               read_prima_data(o)});
        }
        ++i;
      }
    }

    fix_last_mile_duration<ride>(journeys, last_mile_taxi_, last_mile_in,
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
        auto const o = ride.as_object();
        direct_taxi_.push_back({to_unix(o.at("pickupTime").as_int64()),
                                to_unix(o.at("dropoffTime").as_int64()),
                                read_prima_data(o)});
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

bool prima::whitelist_taxi(std::vector<nr::journey>& journeys,
                           n::timetable const& tt) {
  auto const [first_mile, last_mile] = extract_taxis(journeys);

  auto whitelist_response = std::optional<std::string>{};
  auto ioc = boost::asio::io_context{};
  try {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist taxi] request for {} rides",
           first_mile.size() + last_mile.size() + direct_taxi_.size());
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const prima_msg = co_await http_POST(
              taxi_whitelist_, kReqHeaders,
              make_whitelist_request(from_, to_, first_mile, last_mile,
                                     direct_taxi_, fixed_, cap_, tt),
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

  return consume_whitelist_taxi_response(*whitelist_response, journeys,
                                         first_mile, last_mile);
}

}  // namespace motis::odm
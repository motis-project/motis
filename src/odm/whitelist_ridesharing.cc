#include "motis/odm/prima.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/json.hpp"

#include "motis/http_req.h"
#include "motis/odm/odm.h"

namespace n = nigiri;
namespace nr = nigiri::routing;
namespace json = boost::json;
using namespace std::chrono_literals;

namespace motis::odm {

std::string prima::make_ride_sharing_request(n::timetable const& tt) const {
  return make_whitelist_request(from_, to_, first_mile_ride_sharing_,
                                last_mile_ride_sharing_, direct_ride_sharing_,
                                fixed_, cap_, tt);
}

bool prima::consume_ride_sharing_response(std::string_view json) {
  auto const update_first_mile = [&](json::array const& update) {
    auto const n = n_rides_in_response(update);
    if (first_mile_ride_sharing_.size() != n) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] first mile ride-sharing #rides != #updates ({} "
             "!= {})",
             first_mile_ride_sharing_.size(), n);
      first_mile_ride_sharing_.clear();
      return true;
    }

    auto prev_first_mile =
        std::exchange(first_mile_ride_sharing_, std::vector<nr::start>{});
    auto prev_it = std::begin(prev_first_mile);
    for (auto const& stop : update) {
      for (auto const& time : stop.as_array()) {
        if (!time.is_null() && time.is_array()) {
          for (auto const& event : time.as_array()) {
            first_mile_ride_sharing_.push_back(
                {.time_at_start_ =
                     to_unix(event.as_object().at("pickupTime").as_int64()),
                 .time_at_stop_ =
                     to_unix(event.as_object().at("dropoffTime").as_int64()) +
                     kODMTransferBuffer,
                 .stop_ = prev_it->stop_});
            first_mile_ride_sharing_tour_ids_.emplace_back(
                event.as_object().at("tripId").as_string());
          }
        }
        ++prev_it;
      }
    }
    return false;
  };

  auto const update_last_mile = [&](json::array const& update) {
    auto const n = n_rides_in_response(update);
    if (last_mile_ride_sharing_.size() != n) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist taxi] last mile ride-sharing #rides != #updates ({} "
             "!= {})",
             last_mile_ride_sharing_.size(), n);
      last_mile_ride_sharing_.clear();
      return true;
    }

    auto prev_last_mile =
        std::exchange(last_mile_ride_sharing_, std::vector<nr::start>{});
    auto prev_it = std::begin(prev_last_mile);
    for (auto const& stop : update) {
      for (auto const& time : stop.as_array()) {
        if (!time.is_null() && time.is_array()) {
          for (auto const& event : time.as_array()) {
            last_mile_ride_sharing_.push_back(
                {.time_at_start_ =
                     to_unix(event.as_object().at("dropoffTime").as_int64()),
                 .time_at_stop_ =
                     to_unix(event.as_object().at("pickupTime").as_int64()) -
                     kODMTransferBuffer,
                 .stop_ = prev_it->stop_});
            last_mile_ride_sharing_tour_ids_.emplace_back(
                event.as_object().at("tripId").as_string());
          }
          ++prev_it;
        }
      }
    }
    return false;
  };

  auto const update_direct = [&](json::array const& update) {
    if (direct_ride_sharing_.size() != update.size()) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[whitelist ride-sharing] direct ride-sharing #rides != "
             "#updates "
             "({} != {})",
             direct_ride_sharing_.size(), update.size());
      direct_ride_sharing_.clear();
      return true;
    }

    direct_ride_sharing_.clear();
    for (auto const& time : update) {
      if (time.is_array()) {
        for (auto const& ride : time.as_array()) {
          if (!ride.is_null()) {
            direct_ride_sharing_.push_back(
                {to_unix(ride.as_object().at("pickupTime").as_int64()),
                 to_unix(ride.as_object().at("dropoffTime").as_int64())});
            direct_ride_sharing_tour_ids_.emplace_back(
                ride.as_object().at("tripId").as_string());
          }
        }
      }
    }

    return false;
  };

  auto with_errors = false;
  try {
    auto const o = json::parse(json).as_object();
    with_errors |= update_first_mile(o.at("start").as_array());
    with_errors |= update_last_mile(o.at("target").as_array());
    with_errors |= update_direct(o.at("direct").as_array());
  } catch (std::exception const&) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride-sharing] could not parse response: {}", json);
    return false;
  }
  if (with_errors) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride-sharing] parsed response with errors: {}", json);
    return false;
  }

  return true;
}

bool prima::whitelist_ride_sharing(n::timetable const& tt) {
  auto response = std::optional<std::string>{};
  auto ioc = boost::asio::io_context{};
  try {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride-sharing] request for {} events",
           n_ride_sharing_events());
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const prima_msg =
              co_await http_POST(ride_sharing_whitelist_, kReqHeaders,
                                 make_ride_sharing_request(tt), 30s);
          response = get_http_body(prima_msg);
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride-sharing] networking failed: {}", e.what());
    response = std::nullopt;
  }
  if (!response) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[whitelist ride share] failed, discarding ride share journeys");
    return false;
  }

  return consume_ride_sharing_response(*response);
}

}  // namespace motis::odm
#include "motis/odm/prima.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/json.hpp"

#include "nigiri/timetable.h"

#include "utl/erase_if.h"

#include "motis/http_req.h"

namespace n = nigiri;
namespace json = boost::json;
using namespace std::chrono_literals;

namespace motis::odm {

json::array to_json(std::vector<n::routing::offset> const& offsets,
                    n::timetable const& tt) {
  auto a = json::array{};
  for (auto const& o : offsets) {
    auto const& pos = tt.locations_.coordinates_[o.target_];
    a.emplace_back(json::value{{"lat", pos.lat_}, {"lng", pos.lng_}});
  }
  return a;
}

std::string prima::make_blacklist_taxi_request(
    n::timetable const& tt,
    n::interval<n::unixtime_t> const& taxi_intvl) const {
  return json::serialize(json::value{
      {"start", {{"lat", from_.pos_.lat_}, {"lng", from_.pos_.lng_}}},
      {"target", {{"lat", to_.pos_.lat_}, {"lng", to_.pos_.lng_}}},
      {"startBusStops", to_json(first_mile_taxi_, tt)},
      {"targetBusStops", to_json(last_mile_taxi_, tt)},
      {"earliest", to_millis(taxi_intvl.from_)},
      {"latest", to_millis(taxi_intvl.to_)},
      {"startFixed", fixed_ == n::event_type::kDep},
      {"capacities", json::value_from(cap_)}});
}

n::interval<n::unixtime_t> read_intvl(json::value const& jv) {
  return n::interval{to_unix(jv.as_object().at("startTime").as_int64()),
                     to_unix(jv.as_object().at("endTime").as_int64())};
}

bool prima::consume_blacklist_taxi_response(std::string_view json) {
  auto const read_service_times = [&](json::array const& blacklist_times,
                                      auto const& offsets, auto& taxi_times) {
    if (blacklist_times.size() != offsets.size()) {
      n::log(n::log_lvl::debug, "motis.prima",
             "[blacklist taxi] #intervals mismatch");
      taxi_times.clear();
      return;
    }

    taxi_times.resize(offsets.size());

    for (auto [blacklist_time, taxi_time] :
         utl::zip(blacklist_times, taxi_times)) {
      for (auto const& t : blacklist_time.as_array()) {
        taxi_time.emplace_back(read_intvl(t));
      }
    }
  };

  auto const update_direct_rides = [&](json::array const& direct_times) {
    utl::erase_if(direct_taxi_, [&](auto const& ride) {
      return utl::none_of(direct_times, [&](auto const& t) {
        auto const i = read_intvl(t);
        return i.contains(ride.dep_) && i.contains(ride.arr_);
      });
    });
  };

  try {
    auto const o = json::parse(json).as_object();

    read_service_times(o.at("start").as_array(), first_mile_taxi_,
                       first_mile_taxi_times_);
    read_service_times(o.at("target").as_array(), last_mile_taxi_,
                       last_mile_taxi_times_);

    if (direct_duration_ && *direct_duration_ < kODMMaxDuration) {
      update_direct_rides(o.at("direct").as_array());
    }

  } catch (std::exception const&) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[blacklist taxi] could not parse response: {}", json);
    return false;
  }

  return true;
}

bool prima::blacklist_taxi(n::timetable const& tt,
                           n::interval<n::unixtime_t> const& taxi_intvl) {
  auto blacklist_response = std::optional<std::string>{};
  auto ioc = boost::asio::io_context{};
  try {
    n::log(n::log_lvl::debug, "motis.prima", "[blacklist taxi] request for {}",
           taxi_intvl);
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          auto const prima_msg = co_await http_POST(
              taxi_blacklist_, kReqHeaders,
              make_blacklist_taxi_request(tt, taxi_intvl), 10s);
          blacklist_response = get_http_body(prima_msg);
        },
        boost::asio::detached);
    ioc.run();
  } catch (std::exception const& e) {
    n::log(n::log_lvl::debug, "motis.prima",
           "[blacklist taxi] networking failed: {}", e.what());
    blacklist_response = std::nullopt;
  }
  if (!blacklist_response) {
    return false;
  }

  return consume_blacklist_taxi_response(*blacklist_response);
}

}  // namespace motis::odm
#include "motis/odm/calibration/json.h"

#include <chrono>
#include <sstream>

#include "boost/json.hpp"

#include "nigiri/special_stations.h"

#include "motis/odm/odm.h"

namespace motis::odm {

namespace n = nigiri;

constexpr auto const kIntermediateDummyStation =
    n::get_special_station(n::special_station::kVia0);

n::unixtime_t read_time(std::string_view s) {
  auto ss = std::stringstream{};
  ss << s;
  unsigned h, m;
  ss >> h;
  ss.get();
  ss >> m;
  return n::unixtime_t{n::duration_t{h * 60 + m}};
}

n::routing::journey read_journey(boost::json::object const& o) {
  auto j = n::routing::journey{};

  j.start_time_ = read_time(value_to<std::string_view>(o.at("departure")));
  j.dest_time_ = read_time(value_to<std::string_view>(o.at("arrival")));
  j.dest_ = n::get_special_station(n::special_station::kEnd);
  j.transfers_ = value_to<std::uint8_t>(o.at("transfers"));

  auto const start_length =
      n::duration_t{value_to<n::duration_t::rep>(o.at("startLength"))};
  auto const end_length =
      n::duration_t{value_to<n::duration_t::rep>(o.at("endLength"))};

  auto const parse_mode = [](std::string_view s) {
    return s == "taxi" ? kODM : kWalk;
  };
  auto const start_mode = parse_mode(value_to<std::string>(o.at("startMode")));
  auto const end_mode = parse_mode(value_to<std::string>(o.at("endMode")));

  auto const direct_leg = [&](n::transport_mode_id_t mode) {
    return n::routing::journey::leg{
        n::direction::kForward,
        n::get_special_station(n::special_station::kStart),
        n::get_special_station(n::special_station::kEnd),
        j.start_time_,
        j.dest_time_,
        n::routing::offset{n::get_special_station(n::special_station::kEnd),
                           j.travel_time(), mode}};
  };

  if (start_length == j.travel_time()) {
    j.legs_.push_back(direct_leg(start_mode));
  } else if (end_length == j.travel_time()) {
    j.legs_.push_back(direct_leg(end_mode));
  } else {
    if (start_length != n::duration_t{0}) {
      j.legs_.emplace_back(n::direction::kForward,
                           n::get_special_station(n::special_station::kStart),
                           kIntermediateDummyStation, j.start_time_,
                           j.start_time_ + start_length,
                           n::routing::offset{kIntermediateDummyStation,
                                              start_length, start_mode});
    }
    if (end_length != n::duration_t{0}) {
      j.legs_.emplace_back(
          n::direction::kForward, kIntermediateDummyStation,
          n::get_special_station(n::special_station::kEnd),
          j.dest_time_ - end_length, j.dest_time_,
          n::routing::offset{kIntermediateDummyStation, end_length, end_mode});
    }
  }

  return j;
}

bool uses_odm(n::routing::journey const& j) {
  return std::any_of(begin(j.legs_), end(j.legs_), [](auto const& l) {
    return std::holds_alternative<n::routing::offset>(l.uses_) &&
           std::get<n::routing::offset>(l.uses_).transport_mode_id_ == kODM;
  });
}

std::vector<requirement> read_requirements(std::string_view json) {
  auto reqs = std::vector<requirement>{};

  auto const o = boost::json::parse(json).as_object();
  for (auto const& cs : o.at("conSets").as_array()) {
    reqs.emplace_back();
    for (auto const& c : cs.as_array()) {
      auto j = read_journey(c.as_object());
      if (uses_odm(j)) {
        reqs.back().odm_.push_back(std::move(j));
        auto const to_dom = try_value_to<bool>(c.as_object().at("toDom"));
        if (to_dom) {
          reqs.back().odm_to_dom_.set(reqs.back().odm_.size() - 1, *to_dom);
        }
      } else {
        reqs.back().pt_.add(std::move(j));
      }
    }
  }
  
  return reqs;
}

}  // namespace motis::odm
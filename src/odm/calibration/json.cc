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

n::unixtime_t read_time(boost::json::value const& v) {
  auto ss = std::stringstream{};
  ss << v;
  unsigned h, m;
  ss.get();
  ss >> h;
  ss.get();
  ss >> m;
  return n::unixtime_t{n::duration_t{h * 60 + m}};
}

n::routing::journey read_journey(boost::json::object const& o) {  // FIXME
  auto j = n::routing::journey{};
  try {
    j.start_time_ = read_time(o.at("departure"));
    j.dest_time_ = read_time(o.at("arrival"));

    std::cout << "start_time: " << j.start_time_
              << ", dest_time: " << j.dest_time_ << std::endl;

    j.dest_ = n::get_special_station(n::special_station::kEnd);
    j.transfers_ = static_cast<std::uint8_t>(o.at("transfers").as_uint64());

    auto const start_length = n::duration_t{o.at("startLength").as_uint64()};
    auto const end_length = n::duration_t{o.at("endLength").as_uint64()};
    auto const start_mode_str = o.at("startMode").as_string();
    auto const end_mode_str = o.at("endMode").as_string();

    auto const parse_mode = [](std::string_view s) {
      return s == "taxi" ? kODM : kWalk;
    };

    auto const direct_leg = [&](std::string_view s) {
      return n::routing::journey::leg{
          n::direction::kForward,
          n::get_special_station(n::special_station::kStart),
          n::get_special_station(n::special_station::kEnd),
          j.start_time_,
          j.dest_time_,
          n::routing::offset{n::get_special_station(n::special_station::kEnd),
                             j.travel_time(), parse_mode(s)}};
    };

    if (start_length == j.travel_time()) {
      j.legs_.push_back(direct_leg(start_mode_str));
    } else if (end_length == j.travel_time()) {
      j.legs_.push_back(direct_leg(end_mode_str));
    } else {
      if (start_length != n::duration_t{0}) {
        j.legs_.emplace_back(
            n::direction::kForward,
            n::get_special_station(n::special_station::kStart),
            kIntermediateDummyStation, j.start_time_,
            j.start_time_ + start_length,
            n::routing::offset{kIntermediateDummyStation, start_length,
                               parse_mode(start_mode_str)});
      }
      if (end_length != n::duration_t{0}) {
        j.legs_.emplace_back(
            n::direction::kForward, kIntermediateDummyStation,
            n::get_special_station(n::special_station::kEnd),
            j.dest_time_ - end_length, j.dest_time_,
            n::routing::offset{kIntermediateDummyStation, end_length,
                               parse_mode(end_mode_str)});
      }
    }

  } catch (std::exception const& e) {
    std::cout << e.what();
  }
  return j;
}

bool uses_odm(n::routing::journey const& j) {
  for (auto const& l : j.legs_) {
    if (std::holds_alternative<n::routing::offset>(l.uses_) &&
        std::get<n::routing::offset>(l.uses_).transport_mode_id_ == kODM) {
      return true;
    }
  }
  return false;
}

std::vector<requirement> read_requirements(std::string_view json) {
  auto reqs = std::vector<requirement>{};

  try {
    auto const o = boost::json::parse(json).as_object();

    for (auto const& cs : o.at("conSets").as_array()) {
      reqs.emplace_back();
      for (auto const& c : cs.as_array()) {
        auto j = read_journey(c.as_object());
        if (uses_odm(j)) {
          reqs.back().odm_.push_back(std::move(j));
          reqs.back().odm_to_dom_.set(reqs.back().odm_.size() - 1,
                                      c.as_object().at("toDom").as_bool());
        } else {
          reqs.back().pt_.add(std::move(j));
        }
      }
    }

  } catch (std::exception const& e) {
    std::cout << e.what();
  }

  return reqs;
}

}  // namespace motis::odm
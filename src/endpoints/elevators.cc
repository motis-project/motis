#include "motis/endpoints/elevators.h"

#include "net/too_many_exception.h"

#include "osr/geojson.h"

#include "boost/json.hpp"

#include "fmt/chrono.h"
#include "fmt/format.h"

#include "motis/data.h"
#include "motis/elevators/match_elevator.h"

namespace json = boost::json;
namespace n = nigiri;

namespace std {

n::unixtime_t tag_invoke(boost::json::value_to_tag<n::unixtime_t>,
                         boost::json::value const& jv) {
  auto x = n::unixtime_t{};
  auto ss = std::stringstream{std::string{jv.as_string()}};
  ss >> date::parse("%FT%T", x);
  return x;
}

void tag_invoke(boost::json::value_from_tag,
                boost::json::value& jv,
                n::unixtime_t const& v) {
  auto ss = std::stringstream{};
  ss << date::format("%FT%TZ", v);
  jv = json::string{ss.str()};
}

}  // namespace std

namespace nigiri {

template <typename T>
n::interval<T> tag_invoke(boost::json::value_to_tag<n::interval<T>>,
                          boost::json::value const& jv) {
  auto x = n::interval<T>{};
  x.from_ = json::value_to<T>(jv.as_array().at(0));
  x.to_ = json::value_to<T>(jv.as_array().at(1));
  return x;
}

template <typename T>
void tag_invoke(boost::json::value_from_tag,
                boost::json::value& jv,
                n::interval<T> const& v) {
  auto& a = (jv = boost::json::array{}).as_array();
  a.emplace_back(json::value_from(v.from_));
  a.emplace_back(json::value_from(v.to_));
}

}  // namespace nigiri

namespace motis::ep {

constexpr auto const kLimit = 4096U;

json::value elevators::operator()(json::value const& query) const {
  auto const rt = rt_;
  auto const e = rt->e_.get();

  auto matches = json::array{};
  if (e == nullptr) {
    return json::value{{"type", "FeatureCollection"}, {"features", matches}};
  }

  auto const& q = query.as_array();

  auto const min = geo::latlng{q[1].as_double(), q[0].as_double()};
  auto const max = geo::latlng{q[3].as_double(), q[2].as_double()};

  e->elevators_rtree_.find(geo::box{min, max}, [&](elevator_idx_t const i) {
    utl::verify<net::too_many_exception>(matches.size() < kLimit,
                                         "too many elevators");
    auto const& x = e->elevators_[i];
    matches.emplace_back(json::value{
        {"type", "Feature"},
        {"properties",
         {{"type", "api"},
          {"id", x.id_},
          {"desc", x.desc_},
          {"status", (x.status_ ? "ACTIVE" : "INACTIVE")},
          {"outOfService", json::value_from(x.out_of_service_)}}},
        {"geometry", osr::to_point(osr::point::from_latlng(x.pos_))}});
  });

  for (auto const n : l_.find_elevators({min, max})) {
    auto const match =
        match_elevator(e->elevators_rtree_, e->elevators_, w_, n);
    auto const pos = w_.get_node_pos(n);
    if (match != elevator_idx_t::invalid()) {
      auto const& x = e->elevators_[match];
      utl::verify<net::too_many_exception>(matches.size() < kLimit,
                                           "too many elevators");
      matches.emplace_back(json::value{
          {"type", "Feature"},
          {"properties",
           {{"type", "match"},
            {"osm_node_id", to_idx(w_.node_to_osm_[n])},
            {"id", x.id_},
            {"desc", x.desc_},
            {"status", x.status_ ? "ACTIVE" : "INACTIVE"},
            {"outOfService", json::value_from(x.out_of_service_)}}},
          {"geometry",
           osr::to_line_string({pos, osr::point::from_latlng(x.pos_)})}});
    }
  }

  return json::value{{"type", "FeatureCollection"}, {"features", matches}};
}

}  // namespace motis::ep
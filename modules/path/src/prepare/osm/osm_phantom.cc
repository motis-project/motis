#include "motis/path/prepare/osm/osm_phantom.h"

#include <type_traits>

#include "boost/geometry.hpp"

#include "geo/constants.h"
#include "geo/detail/register_latlng.h"
#include "geo/point_rtree.h"

#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/path/prepare/osm/segment_rtree.h"

namespace motis::path {

constexpr auto kMatchRadius = 50;
constexpr auto kMatchRadiusNoResult = 100;
// constexpr auto kMatchRadius = 150; // was 50, not enough for large Stations?
constexpr auto kStopMatchRadius = 20;

// following http://www.movable-type.co.uk/scripts/latlong.html
void located_osm_edge_phantom_with_dist::locate() {
  using std::acos;
  using std::asin;
  using std::atan2;
  using std::cos;
  using std::fmod;
  using std::sin;
  auto const to_rad = [](auto const& deg) { return deg * geo::kPI / 180.0; };
  auto const to_deg = [](auto const& rad) { return rad * 180.0 / geo::kPI; };

  // project p3 onto p1 -> p2 at p_x
  auto p1 = phantom_.segment_.first;
  auto p2 = phantom_.segment_.second;
  auto p3 = station_pos_;

  auto a_dist_12 = boost::geometry::distance(p1, p2);  // angular dist
  auto a_dist_13 = boost::geometry::distance(p1, p3);  // angular dist

  auto bearing_12 = to_rad(360. - geo::bearing(p1, p2));  // flip CCW to CW
  auto bearing_13 = to_rad(360. - geo::bearing(p1, p3));  // flip CCW to CW

  // angular cross-track distance : p_3 -> p_x
  auto a_dist_xt = asin(sin(a_dist_13) * sin(bearing_13 - bearing_12));

  // angular along-track distance : p_1 -> p_x
  auto a_dist_at = acos(cos(a_dist_13) / cos(a_dist_xt));

  if (a_dist_at <= 0) {  // clamp at from
    pos_ = phantom_.segment_.first;
    along_track_dist_ = 0;
    eq_from_ = true;
  } else if (a_dist_at >= a_dist_12) {  // clamp at to
    pos_ = phantom_.segment_.second;
    along_track_dist_ = a_dist_12 * geo::kEarthRadiusMeters;  // conv. to meters
    eq_to_ = true;
  } else {  // in between
    double lat_1_r = to_rad(p1.lat_);
    double lng_1_r = to_rad(p1.lng_);

    double lat_x_r = asin(sin(lat_1_r) * cos(a_dist_at) +
                          cos(lat_1_r) * sin(a_dist_at) * cos(bearing_12));
    double lng_x_r =
        lng_1_r + atan2(sin(bearing_12) * sin(a_dist_at) * cos(lat_1_r),
                        cos(a_dist_at) - sin(lat_1_r) * sin(lat_x_r));

    pos_ = {to_deg(lat_x_r), fmod(to_deg(lng_x_r) + 540., 360.) - 180.};
    along_track_dist_ = a_dist_at * geo::kEarthRadiusMeters;  // conv. to meters
  }
}

inline bool way_idx_match(std::vector<size_t> const& vec, size_t const val) {
  return std::find(begin(vec), end(vec), val) != end(vec);
}

inline bool way_idx_match(size_t const val, std::vector<size_t> const& vec) {
  return way_idx_match(vec, val);
}

inline bool dominated_with_tiebreaker(osm_node_phantom_with_dist const& self,
                                      osm_edge_phantom_with_dist const& other) {
  if (std::fabs(self.distance_ - other.distance_) < .1) {
    return false;  // self=node -> undominated by other=edge
  }
  return other.distance_ < self.distance_;
}

inline bool dominated_with_tiebreaker(osm_edge_phantom_with_dist const& self,
                                      osm_node_phantom_with_dist const& other) {
  if (std::fabs(self.distance_ - other.distance_) < .1) {
    return true;  // self=edge -> dominated by other=node
  }
  return other.distance_ < self.distance_;
}

struct osm_phantom_index {
  explicit osm_phantom_index(mcd::vector<osm_way> const& osm_ways) {
    mcd::hash_map<int64_t, osm_node_phantom> node_phantoms;

    for (auto i = 0UL; i < osm_ways.size(); ++i) {
      auto const& polyline = osm_ways[i].path_.polyline_;
      utl::verify(polyline.size() > 1, "rail: polyline too short");

      auto const make_node_phantom = [&](auto const id, auto const pos) {
        auto& phantom = utl::get_or_create(node_phantoms, id, [&] {
          return osm_node_phantom{id, pos};
        });
        phantom.way_idx_.push_back(i);
      };
      make_node_phantom(osm_ways[i].from(), polyline.front());
      make_node_phantom(osm_ways[i].to(), polyline.back());

      auto j = 0;
      for (auto const& [from, to] : utl::pairwise(polyline)) {
        edge_phantoms_.emplace_back(i, j++, from, to);
      }
    }

    node_phantoms_ = utl::to_vec(node_phantoms,
                                 [](auto const& pair) { return pair.second; });

    node_phantom_rtree_ = geo::make_point_rtree(
        node_phantoms_, [](auto const& p) { return p.pos_; });

    edge_phantom_rtree_ = make_segment_rtree(
        edge_phantoms_, [](auto const& p) { return p.segment_; });
  }

  std::pair<std::vector<osm_node_phantom_with_dist>,
            std::vector<osm_edge_phantom_with_dist>>
  get_osm_phantoms(geo::latlng const& pos, double const radius) const {
    auto const nodes = utl::to_vec(
        node_phantom_rtree_.in_radius(pos, radius), [&](auto const& idx) {
          return osm_node_phantom_with_dist(
              node_phantoms_[idx],
              geo::distance(pos, node_phantoms_[idx].pos_));
        });

    std::set<size_t> ways;
    std::vector<osm_edge_phantom_with_dist> edges;

    auto const intersecting_edge_phantoms =
        edge_phantom_rtree_.intersects_radius_with_distance(pos, radius);
    for (auto const& [dist, idx] : intersecting_edge_phantoms) {
      auto const& phantom = edge_phantoms_[idx];
      if (ways.find(phantom.way_idx_) != end(ways)) {
        continue;  // rtree: already sorted by distance
      }

      ways.insert(phantom.way_idx_);
      edges.emplace_back(osm_edge_phantom_with_dist(phantom, dist));
    }

    // retain only of no neighbor is better
    auto const filter = [](auto const& subjects, auto const& others) {
      typename std::decay<decltype(subjects)>::type result;
      std::copy_if(
          begin(subjects), end(subjects), std::back_inserter(result),
          [&others](auto const& s) {
            return std::none_of(begin(others), end(others), [&s](auto const o) {
              return way_idx_match(o.phantom_.way_idx_, s.phantom_.way_idx_) &&
                     dominated_with_tiebreaker(s, o);
            });
          });
      return result;
    };

    return {filter(nodes, edges), filter(edges, nodes)};
  }

  std::vector<osm_node_phantom> node_phantoms_;
  geo::point_rtree node_phantom_rtree_;

  std::vector<osm_edge_phantom> edge_phantoms_;
  segment_rtree edge_phantom_rtree_;
};

std::pair<std::vector<std::pair<osm_node_phantom_with_dist, station const*>>,
          std::vector<located_osm_edge_phantom_with_dist>>
make_phantoms(station_index const& station_idx,
              std::vector<size_t> const& matched_stations,
              mcd::vector<osm_way> const& osm_ways) {
  std::vector<std::pair<osm_node_phantom_with_dist, station const*>> n_phantoms;
  std::vector<located_osm_edge_phantom_with_dist> e_phantoms;

  auto const collect_phantoms = [&](auto& vec, auto& phantoms, auto&&... args) {
    for (auto&& p : phantoms) {
      vec.emplace_back(std::move(p), std::forward<decltype(args)>(args)...);
    }
  };

  osm_phantom_index phantom_idx{osm_ways};
  for (auto const& i : matched_stations) {
    auto const& station = station_idx.stations_[i];

    auto [np, ep] = phantom_idx.get_osm_phantoms(station.pos_, kMatchRadius);
    collect_phantoms(n_phantoms, np, &station);
    collect_phantoms(e_phantoms, ep, &station, station.pos_);

    bool have_phantoms = !np.empty() || !ep.empty();

    for (auto const& stop_position : station.stop_positions_) {
      auto [snp, sep] =
          phantom_idx.get_osm_phantoms(stop_position, kStopMatchRadius);
      collect_phantoms(n_phantoms, snp, &station);
      collect_phantoms(e_phantoms, sep, &station, stop_position);

      have_phantoms |= !snp.empty() || !sep.empty();
    }

    if (!have_phantoms) {
      auto [npnr, epnr] =
          phantom_idx.get_osm_phantoms(station.pos_, kMatchRadiusNoResult);
      collect_phantoms(n_phantoms, npnr, &station);
      collect_phantoms(e_phantoms, epnr, &station, station.pos_);
    }
  }

  utl::erase_duplicates(
      n_phantoms,
      [](auto const& a, auto const& b) {
        auto const& pa = a.first.phantom_;
        auto const& pb = b.first.phantom_;
        return std::tie(pa.id_, pa.pos_, a.second, a.first.distance_) <
               std::tie(pb.id_, pb.pos_, b.second, b.first.distance_);
      },
      [](auto const& a, auto const& b) {
        auto const& pa = a.first.phantom_;
        auto const& pb = b.first.phantom_;
        return std::tie(pa.id_, pa.pos_, a.second) ==
               std::tie(pb.id_, pb.pos_, b.second);
      });

  // keep "closes" phantom node between different station_pos
  utl::erase_duplicates(
      e_phantoms,
      [](auto const& a, auto const& b) {
        auto const& pa = a.phantom_;
        auto const& pb = b.phantom_;
        return std::tie(pa.way_idx_, pa.offset_,  //
                        a.station_, a.station_pos_, a.distance_) <
               std::tie(pb.way_idx_, pb.offset_,  //
                        b.station_, b.station_pos_, b.distance_);
      },
      [](auto const& a, auto const& b) {
        auto const& pa = a.phantom_;
        auto const& pb = b.phantom_;
        return std::tie(pa.way_idx_, pa.offset_, a.station_, a.station_pos_) ==
               std::tie(pb.way_idx_, pb.offset_, b.station_, b.station_pos_);
      });

  // sort for consumption during graph insertion
  std::for_each(begin(e_phantoms), end(e_phantoms),
                [](auto& p) { p.locate(); });
  std::sort(begin(e_phantoms), end(e_phantoms),
            [](auto const& a, auto const& b) {
              auto const& pa = a.phantom_;
              auto const& pb = b.phantom_;
              return std::tie(pa.way_idx_, pa.offset_, a.along_track_dist_) <
                     std::tie(pb.way_idx_, pb.offset_, b.along_track_dist_);
            });
  return std::make_pair(std::move(n_phantoms), std::move(e_phantoms));
}

}  // namespace motis::path
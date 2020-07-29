#include "motis/path/prepare/osm/osm_phantom.h"

#include <type_traits>

#include "boost/geometry.hpp"

#include "geo/constants.h"
#include "geo/detail/register_latlng.h"
#include "geo/point_rtree.h"

#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/path/prepare/osm/segment_rtree.h"
#include "motis/path/prepare/tuning_parameters.h"

namespace motis::path {

// following http://www.movable-type.co.uk/scripts/latlong.html
void osm_edge_phantom_match::locate() {
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

osm_phantom_builder::osm_phantom_builder(station_index const& station_idx,
                                         mcd::vector<osm_way> const& osm_ways)
    : station_idx_{station_idx}, osm_ways_{osm_ways} {
  {
    geo::box component_box;
    for (auto const& way : osm_ways) {
      for (auto const& pos : way.path_.polyline_) {
        component_box.extend(pos);
      }
    }
    component_box.extend(kMaxMatchDistance);

    matched_stations_ = utl::to_vec(
        station_idx_.index_.within(component_box),
        [&](auto const& idx) { return &station_idx_.stations_[idx]; });
  }

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

  node_phantoms_ =
      utl::to_vec(node_phantoms, [](auto const& pair) { return pair.second; });
  std::sort(begin(node_phantoms_), end(node_phantoms_),
            [](auto const& lhs, auto const& rhs) { return lhs.id_ < rhs.id_; });

  node_phantom_rtree_ = geo::make_point_rtree(
      node_phantoms_, [](auto const& p) { return p.pos_; });

  edge_phantom_rtree_ = make_segment_rtree(
      edge_phantoms_, [](auto const& p) { return p.segment_; });
}

void osm_phantom_builder::build_osm_phantoms(station const* station) {
  auto [np, ep] =
      match_osm_phantoms(station, station->pos_, kStationMatchDistance);
  append_phantoms(np, ep);

  bool have_phantoms = !np.empty() || !ep.empty();

  for (auto const& stop_position : station->stop_positions_) {
    auto [snp, sep] =
        match_osm_phantoms(station, stop_position.pos_, kStopMatchDistance);
    append_phantoms(snp, sep);
    have_phantoms |= !snp.empty() || !sep.empty();
  }

  // fallback if nothing has matched: search with large radius and take
  // everything close to the nearest match
  if (!have_phantoms) {
    auto [fbnp, fbep] =
        match_osm_phantoms(station, station->pos_, kMaxMatchDistance);

    std::vector<std::pair<double, geo::latlng>> nearest_matches;
    if (!fbnp.empty()) {
      nearest_matches.emplace_back(fbnp[0].distance_, fbnp[0].phantom_.pos_);
    }
    if (!fbep.empty()) {
      nearest_matches.emplace_back(fbep[0].distance_, fbep[0].pos_);
    }
    if (!nearest_matches.empty()) {
      std::sort(begin(nearest_matches), end(nearest_matches));
      auto const& nearest = nearest_matches[0].second;
      utl::erase_if(fbnp, [&](auto const& n) {
        return geo::distance(nearest, n.phantom_.pos_) > kStopMatchDistance;
      });
      utl::erase_if(fbep, [&](auto const& e) {
        return geo::distance(nearest, e.pos_) > kStopMatchDistance;
      });
    }

    append_phantoms(fbnp, fbep);
  }
}

bool way_idx_match(std::vector<size_t> const& vec, size_t const val) {
  return std::find(begin(vec), end(vec), val) != end(vec);
}

bool way_idx_match(size_t const val, std::vector<size_t> const& vec) {
  return way_idx_match(vec, val);
}

bool dominated_with_tiebreaker(mcd::vector<osm_way> const& osm_ways,
                               osm_node_phantom_match const& self,
                               osm_edge_phantom_match const& other) {
  auto const& osm_way = osm_ways.at(other.phantom_.way_idx_);
  if ((other.eq_from_ && other.phantom_.offset_ == 0 &&
       self.phantom_.id_ == osm_way.from()) ||
      (other.eq_to_ && other.phantom_.offset_ == osm_way.path_.size() - 2 &&
       self.phantom_.id_ == osm_way.to())) {
    return false;  // self=node -> undominated by corresponding 'end' edge
  }
  if (std::fabs(self.distance_ - other.distance_) < .1) {
    return false;  // self=node -> undominated by other=edge
  }
  return other.distance_ < self.distance_;
};

bool dominated_with_tiebreaker(mcd::vector<osm_way> const& osm_ways,
                               osm_edge_phantom_match const& self,
                               osm_node_phantom_match const& other) {
  auto const& osm_way = osm_ways.at(self.phantom_.way_idx_);
  if ((self.eq_from_ && self.phantom_.offset_ == 0 &&
       other.phantom_.id_ == osm_way.from()) ||
      (self.eq_to_ && self.phantom_.offset_ == osm_way.path_.size() - 2 &&
       other.phantom_.id_ == osm_way.to())) {
    return true;  // self='end' edge -> dominated by corresponding node
  }
  if (std::fabs(self.distance_ - other.distance_) < .1) {
    return true;  // self=edge -> dominated by other=node
  }
  return other.distance_ < self.distance_;
}

std::pair<std::vector<osm_node_phantom_match>,
          std::vector<osm_edge_phantom_match>>
osm_phantom_builder::match_osm_phantoms(station const* station,
                                        geo::latlng const& pos,
                                        double const radius) const {
  // find node phantoms
  auto n_matches = utl::to_vec(
      node_phantom_rtree_.in_radius(pos, radius), [&](auto const& idx) {
        return osm_node_phantom_match{
            node_phantoms_[idx], geo::distance(pos, node_phantoms_[idx].pos_),
            station};
      });

  auto const add_n_match = [&](auto const id) {
    if (std::find_if(begin(n_matches), end(n_matches), [&](auto const& m) {
          return m.phantom_.id_ == id;
        }) != end(n_matches)) {
      return;
    }

    auto const it = std::lower_bound(
        begin(node_phantoms_), end(node_phantoms_), id,
        [](auto const& lhs, auto const& rhs) { return lhs.id_ < rhs; });
    utl::verify(it != end(node_phantoms_) && it->id_ == id,
                "match_osm_phantoms: missing node phantom");
    n_matches.push_back(
        osm_node_phantom_match{*it, geo::distance(pos, it->pos_), station});
  };

  // find edge phantoms
  std::set<size_t> ways;
  std::vector<osm_edge_phantom_match> e_matches;
  auto const intersecting_edge_phantoms =
      edge_phantom_rtree_.intersects_radius_with_distance(pos, radius);
  for (auto const& [dist, idx] : intersecting_edge_phantoms) {
    auto const& phantom = edge_phantoms_[idx];
    if (ways.insert(phantom.way_idx_).second == false) {
      continue;  // rtree: already sorted by distance
    }

    auto em = osm_edge_phantom_match{phantom, dist, station, pos};
    em.locate();

    // ensure the node match exists for edge matches at/near both ends
    auto const& osm_way = osm_ways_.at(em.phantom_.way_idx_);
    if (em.eq_from_ && em.phantom_.offset_ == 0) {
      add_n_match(osm_way.from());
    } else if (em.eq_to_ && em.phantom_.offset_ == osm_way.path_.size() - 2) {
      add_n_match(osm_way.to());
    }

    e_matches.emplace_back(std::move(em));
  }

  // keep only the closest match of graph-adjacent phantom nodes
  auto const filter = [&](auto const& subjects, auto const& others) {
    typename std::decay<decltype(subjects)>::type result;
    std::copy_if(
        begin(subjects), end(subjects), std::back_inserter(result),
        [&](auto const& s) {
          return std::none_of(begin(others), end(others), [&](auto const& o) {
            return way_idx_match(o.phantom_.way_idx_, s.phantom_.way_idx_) &&
                   dominated_with_tiebreaker(osm_ways_, s, o);
          });
        });
    return result;
  };

  return {filter(n_matches, e_matches), filter(e_matches, n_matches)};
}

void osm_phantom_builder::append_phantoms(
    std::vector<osm_node_phantom_match> const& n_phantoms,
    std::vector<osm_edge_phantom_match> const& e_phantoms) {
  if (n_phantoms.empty() && e_phantoms.empty()) {
    return;
  }

  std::lock_guard lock{mutex_};
  utl::concat(n_phantoms_, n_phantoms);
  utl::concat(e_phantoms_, e_phantoms);
}

void osm_phantom_builder::finalize() {
  utl::erase_duplicates(
      n_phantoms_,
      [](auto const& a, auto const& b) {
        auto const& pa = a.phantom_;
        auto const& pb = b.phantom_;
        return std::tie(pa.id_, pa.pos_, a.station_, a.distance_) <
               std::tie(pb.id_, pb.pos_, b.station_, b.distance_);
      },
      [](auto const& a, auto const& b) {
        auto const& pa = a.phantom_;
        auto const& pb = b.phantom_;
        return std::tie(pa.id_, pa.pos_, a.station_) ==
               std::tie(pb.id_, pb.pos_, b.station_);
      });

  // keep "closes" phantom node between different station_pos
  utl::erase_duplicates(
      e_phantoms_,
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
  std::sort(begin(e_phantoms_), end(e_phantoms_),
            [](auto const& a, auto const& b) {
              auto const& pa = a.phantom_;
              auto const& pb = b.phantom_;
              return std::tie(pa.way_idx_, pa.offset_, a.along_track_dist_) <
                     std::tie(pb.way_idx_, pb.offset_, b.along_track_dist_);
            });
}

}  // namespace motis::path

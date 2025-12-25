#include "motis/endpoints/graph.h"

#include "net/too_many_exception.h"

#include "osr/geojson.h"
#include "osr/routing/profiles/car_sharing.h"
#include "osr/routing/route.h"

namespace json = boost::json;

namespace motis::ep {

constexpr auto const kMaxWays = 2048U;

json::value graph::operator()(json::value const& query) const {
  auto const& q = query.as_object();
  auto const& x = query.at("waypoints").as_array();
  auto const min = geo::latlng{x[1].as_double(), x[0].as_double()};
  auto const max = geo::latlng{x[3].as_double(), x[2].as_double()};
  auto const level = q.contains("level")
                         ? osr::level_t{q.at("level").to_number<float>()}
                         : osr::kNoLevel;

  auto gj = osr::geojson_writer{.w_ = w_};
  auto n_ways = 0U;
  l_.find({min, max}, [&](osr::way_idx_t const w) {
    if (++n_ways == kMaxWays) {
      throw utl::fail<net::too_many_exception>("too many ways");
    }

    if (level == osr::kNoLevel) {
      gj.write_way(w);
      return;
    }

    auto const way_prop = w_.r_->way_properties_[w];
    if (way_prop.is_elevator()) {
      auto const n = w_.r_->way_nodes_[w][0];
      auto const np = w_.r_->node_properties_[n];
      if (np.is_multi_level()) {
        auto has_level = false;
        utl::for_each_set_bit(
            osr::foot<true>::get_elevator_multi_levels(*w_.r_, n),
            [&](auto&& bit) {
              has_level |=
                  (level == osr::level_t{static_cast<std::uint8_t>(bit)});
            });
        if (has_level) {
          gj.write_way(w);
          return;
        }
      }
    }

    if ((level == osr::level_t{0.F} &&
         way_prop.from_level() == osr::kNoLevel) ||
        way_prop.from_level() == level || way_prop.to_level() == level) {
      gj.write_way(w);
      return;
    }
  });

  gj.finish(&osr::get_dijkstra<osr::car_sharing<osr::track_node_tracking>>());

  return gj.json();
}

}  // namespace motis::ep